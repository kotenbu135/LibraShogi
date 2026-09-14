// SPDX-License-Identifier: Apache-2.0
// ONNX Runtime（MIT）の C API を実行時にロードして使う推論部。
// ライブラリはリンクせず dlopen / LoadLibrary で読む（無ければ分かりやすいエラーにし、CUDA → DirectML → CPU の順で試す）。
#pragma once
#include <memory>
#include <string>

namespace libra_engine {

class OrtInfer {
 public:
  OrtInfer();
  ~OrtInfer();
  // 共有ライブラリを読む。lib_path が空なら実行ファイルの隣 → システムの順に探す
  bool load(const std::string& lib_path, std::string* err);
  // モデルを開く。provider: "auto" | "cuda" | "dml" | "cpu"
  bool open(const std::string& model_path, int threads, const std::string& provider, std::string* err);
  bool is_open() const;
  std::string provider() const;   // 実際に使っている EP の名前
  std::string fallback() const;   // 先に試して失敗した EP とその理由（無ければ空。1 行）
  std::string version() const;    // ONNX Runtime のバージョン文字列
  // n 局面を評価する。sq: n×81×SQ_FEATS、glob: n×GLOB_FEATS → logits: n×POLICY_SIZE、wdl: n×3（softmax 済み）
  bool run(int n, const float* sq, const float* glob, float* logits, float* wdl, std::string* err);

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

std::string exe_dir();  // 実行ファイルのあるディレクトリ（末尾に区切り無し）

}  // namespace libra_engine
