// SPDX-License-Identifier: Apache-2.0
#include "ort_infer.h"

#include <onnxruntime_c_api.h>

#include <cmath>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#include "libra/encoding.h"

namespace libra_engine {

std::string exe_dir() {
#ifdef _WIN32
  char buf[MAX_PATH];
  DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
  std::string s(buf, n);
  auto k = s.find_last_of("\\/");
  return k == std::string::npos ? "." : s.substr(0, k);
#else
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return ".";
  std::string s(buf, n);
  auto k = s.find_last_of('/');
  return k == std::string::npos ? "." : s.substr(0, k);
#endif
}

struct OrtInfer::Impl {
  const OrtApi* api = nullptr;
  const OrtApiBase* base = nullptr;
  OrtEnv* env = nullptr;
  OrtSession* session = nullptr;
  OrtMemoryInfo* mi = nullptr;
  std::string provider = "none";
  ~Impl() {
    if (!api) return;
    if (session) api->ReleaseSession(session);
    if (mi) api->ReleaseMemoryInfo(mi);
    if (env) api->ReleaseEnv(env);
  }
  bool ok(OrtStatus* st, std::string* err) const {
    if (!st) return true;
    if (err) *err = api->GetErrorMessage(st);
    api->ReleaseStatus(st);
    return false;
  }
};

OrtInfer::OrtInfer() : p_(new Impl) {}
OrtInfer::~OrtInfer() = default;
bool OrtInfer::is_open() const { return p_->session != nullptr; }
std::string OrtInfer::provider() const { return p_->provider; }
std::string OrtInfer::version() const { return p_->base ? p_->base->GetVersionString() : ""; }

using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)(void);

static void* open_lib(const std::string& path) {
#ifdef _WIN32
  return (void*)LoadLibraryA(path.c_str());
#else
  return dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
#endif
}
static void* sym(void* h, const char* name) {
#ifdef _WIN32
  return (void*)GetProcAddress((HMODULE)h, name);
#else
  return dlsym(h, name);
#endif
}

bool OrtInfer::load(const std::string& lib_path, std::string* err) {
  std::vector<std::string> cands;
  if (!lib_path.empty()) cands.push_back(lib_path);
#ifdef _WIN32
  cands.push_back(exe_dir() + "\\onnxruntime.dll");
  cands.push_back("onnxruntime.dll");
#else
  cands.push_back(exe_dir() + "/libonnxruntime.so.1");
  cands.push_back(exe_dir() + "/libonnxruntime.so");
  cands.push_back("libonnxruntime.so.1");
  cands.push_back("libonnxruntime.so");
#endif
  void* h = nullptr;
  for (const std::string& c : cands) {
    h = open_lib(c);
    if (h) break;
  }
  if (!h) {
    if (err) *err = "ONNX Runtime library not found (tried " + cands.front() + " ...)";
    return false;
  }
  auto f = (GetApiBaseFn)sym(h, "OrtGetApiBase");
  if (!f) {
    if (err) *err = "OrtGetApiBase not found in the ONNX Runtime library";
    return false;
  }
  p_->base = f();
  p_->api = p_->base->GetApi(ORT_API_VERSION);
  if (!p_->api) {
    if (err) *err = std::string("ONNX Runtime ") + p_->base->GetVersionString() + " is older than the API this build needs (" + std::to_string(ORT_API_VERSION) + ")";
    return false;
  }
  return true;
}

static bool has_provider(const OrtApi* api, const char* name) {
  char** names = nullptr;
  int n = 0;
  OrtStatus* st = api->GetAvailableProviders(&names, &n);
  if (st) {
    api->ReleaseStatus(st);
    return false;
  }
  bool found = false;
  for (int i = 0; i < n; ++i)
    if (std::strcmp(names[i], name) == 0) found = true;
  st = api->ReleaseAvailableProviders(names, n);
  if (st) api->ReleaseStatus(st);
  return found;
}

bool OrtInfer::open(const std::string& model_path, int threads, const std::string& provider, std::string* err) {
  Impl& I = *p_;
  if (!I.api) {
    if (err) *err = "ONNX Runtime not loaded";
    return false;
  }
  if (I.session) {
    I.api->ReleaseSession(I.session);
    I.session = nullptr;
  }
  if (!I.env && !I.ok(I.api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "libra", &I.env), err)) return false;
  if (!I.mi && !I.ok(I.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &I.mi), err)) return false;

  std::vector<std::string> order;
  if (provider == "auto") order = {"cuda", "dml", "cpu"};
  else if (provider == "cpu") order = {"cpu"};
  else order = {provider, "cpu"};

#ifdef _WIN32
  int wn = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
  std::wstring wpath(wn, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, &wpath[0], wn);
  const ORTCHAR_T* path = wpath.c_str();
#else
  const ORTCHAR_T* path = model_path.c_str();
#endif
  std::string last_err;
  for (const std::string& ep : order) {
    OrtSessionOptions* so = nullptr;
    if (!I.ok(I.api->CreateSessionOptions(&so), err)) return false;
    if (!I.ok(I.api->SetIntraOpNumThreads(so, threads), err) || !I.ok(I.api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL), err)) {
      I.api->ReleaseSessionOptions(so);
      return false;
    }
    bool appended = true;
    std::string e;
    if (ep == "cuda") {
      if (!has_provider(I.api, "CUDAExecutionProvider")) appended = false;
      else {
        OrtCUDAProviderOptionsV2* co = nullptr;
        if (I.ok(I.api->CreateCUDAProviderOptions(&co), &e)) {
          appended = I.ok(I.api->SessionOptionsAppendExecutionProvider_CUDA_V2(so, co), &e);
          I.api->ReleaseCUDAProviderOptions(co);
        } else appended = false;
      }
    } else if (ep == "dml") {
      if (!has_provider(I.api, "DmlExecutionProvider")) appended = false;
      else appended = I.ok(I.api->SessionOptionsAppendExecutionProvider(so, "DML", nullptr, nullptr, 0), &e);
    } else if (ep != "cpu") {
      appended = false;
      e = "unknown provider " + ep;
    }
    if (!appended) {
      I.api->ReleaseSessionOptions(so);
      if (!e.empty()) last_err = ep + ": " + e;
      continue;
    }
    OrtSession* s = nullptr;
    bool ok = I.ok(I.api->CreateSession(I.env, path, so, &s), &e);
    I.api->ReleaseSessionOptions(so);
    if (!ok) {
      last_err = ep + ": " + e;
      continue;
    }
    I.session = s;
    I.provider = ep;
    // 暖機（CUDA は最初の Run が遅い）
    std::vector<float> sq(libra::SQ_NB * libra::SQ_FEATS, 0.f), gl(libra::GLOB_FEATS, 0.f), lg(libra::POLICY_SIZE), w(3);
    if (!run(1, sq.data(), gl.data(), lg.data(), w.data(), err)) return false;
    return true;
  }
  if (err) *err = "no usable execution provider (" + last_err + ")";
  return false;
}

bool OrtInfer::run(int n, const float* sq, const float* glob, float* logits, float* wdl, std::string* err) {
  Impl& I = *p_;
  if (!I.session) {
    if (err) *err = "model not open";
    return false;
  }
  const std::int64_t s1[3] = {n, libra::SQ_NB, libra::SQ_FEATS};
  const std::int64_t s2[2] = {n, libra::GLOB_FEATS};
  OrtValue* in[2] = {nullptr, nullptr};
  OrtValue* out[3] = {nullptr, nullptr, nullptr};
  bool ok = I.ok(I.api->CreateTensorWithDataAsOrtValue(I.mi, (void*)sq, sizeof(float) * n * libra::SQ_NB * libra::SQ_FEATS, s1, 3,
                                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[0]), err) &&
            I.ok(I.api->CreateTensorWithDataAsOrtValue(I.mi, (void*)glob, sizeof(float) * n * libra::GLOB_FEATS, s2, 2,
                                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[1]), err);
  static const char* in_names[2] = {"sq", "glob"};
  static const char* out_names[3] = {"policy", "wdl", "v41"};
  if (ok) ok = I.ok(I.api->Run(I.session, nullptr, in_names, in, 2, out_names, 3, out), err);
  if (ok) {
    float* p = nullptr;
    float* w = nullptr;
    ok = I.ok(I.api->GetTensorMutableData(out[0], (void**)&p), err) && I.ok(I.api->GetTensorMutableData(out[1], (void**)&w), err);
    if (ok) {
      std::memcpy(logits, p, sizeof(float) * n * libra::POLICY_SIZE);
      for (int i = 0; i < n; ++i) {
        float m = std::max(w[3 * i], std::max(w[3 * i + 1], w[3 * i + 2]));
        float e0 = std::exp(w[3 * i] - m), e1 = std::exp(w[3 * i + 1] - m), e2 = std::exp(w[3 * i + 2] - m);
        float z = e0 + e1 + e2;
        wdl[3 * i] = e0 / z;
        wdl[3 * i + 1] = e1 / z;
        wdl[3 * i + 2] = e2 / z;
      }
    }
  }
  for (OrtValue* v : in)
    if (v) I.api->ReleaseValue(v);
  for (OrtValue* v : out)
    if (v) I.api->ReleaseValue(v);
  return ok;
}

}  // namespace libra_engine
