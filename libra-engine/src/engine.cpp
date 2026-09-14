// SPDX-License-Identifier: Apache-2.0
#include "engine.h"

#include <chrono>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <thread>

#include "libra/encoding.h"
#include "libra/position.h"

namespace libra_engine {

using namespace libra;
const char* VERSION = "0.0.2";

static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int winrate_to_cp(double p) {
  double q = std::min(std::max(p, 1e-6), 1 - 1e-6);
  int cp = int(std::floor(435.0 * std::log(q / (1 - q)) + 34.0 + 0.5));
  return cp == 0 ? 0 : cp;
}

static std::string env_or(const char* name, const std::string& def) {
  const char* v = std::getenv(name);
  return v && *v ? std::string(v) : def;
}

Engine::Engine() {
  opts_ = {
      {"Fuseki_Mode", "tenbin"},
      {"MultiPV", "1"},
      {"Threads", "4"},
      {"DNN_Model", env_or("LIBRA_MODEL", exe_dir() + "/libra.onnx")},
      {"DNN_Batch_Size", "64"},
      {"DNN_Provider", env_or("LIBRA_PROVIDER", "auto")},
      {"Sims_Fuseki", "400"},
      {"Sims_Normal", "800"},
      {"Scale_Table", ""},
      {"USI_Ponder", "false"},
      {"Declare_Win", "false"},
      {"Mate_Nodes", "2000"},
  };
  sq_.assign(SQ_NB * SQ_FEATS, 0.f);
  glob_.assign(GLOB_FEATS, 0.f);
  logits_.assign(POLICY_SIZE, 0.f);
  wdl_.assign(3, 0.f);
}
Engine::~Engine() = default;

int Engine::geti(const std::string& k) const { return std::atoi(opts_.at(k).c_str()); }
bool Engine::getb(const std::string& k) const { return opts_.at(k) == "true"; }

void Engine::out(const std::string& s) const {
  std::cout << s << '\n';
  std::cout.flush();
}

void Engine::declare_options() const {
  out("option name Fuseki_Mode type combo default tenbin var tenbin var fuseki");
  out("option name MultiPV type spin default 1 min 1 max 300");
  out("option name Threads type spin default 4 min 1 max 64");
  out("option name DNN_Model type string default " + opts_.at("DNN_Model"));
  out("option name DNN_Batch_Size type spin default 64 min 1 max 1024");
  out("option name DNN_Provider type combo default " + opts_.at("DNN_Provider") + " var auto var cuda var dml var cpu");
  out("option name Sims_Fuseki type spin default 400 min 1 max 1000000");
  out("option name Sims_Normal type spin default 800 min 1 max 1000000");
  out("option name Scale_Table type string default <empty>");
  out("option name USI_Ponder type check default false");
  out("option name Declare_Win type check default false");
  out("option name Mate_Nodes type spin default 2000 min 0 max 10000000");
}

void Engine::setoption(const std::vector<std::string>& t) {
  std::string name, value;
  for (size_t i = 1; i < t.size(); ++i) {
    if (t[i] == "name" && i + 1 < t.size()) name = t[++i];
    else if (t[i] == "value") {
      for (size_t j = i + 1; j < t.size(); ++j) value += (j == i + 1 ? "" : " ") + t[j];
      break;
    }
  }
  auto it = opts_.find(name);
  if (it == opts_.end()) return;
  if (value == "<empty>") value = "";
  it->second = value;
  if (name == "Threads" || name == "Mate_Nodes") eng_.reset();
  if (name == "DNN_Model" || name == "DNN_Provider") loaded_model_.clear();
  if (name == "Scale_Table") scale_loaded_ = "\x01";  // 次の go で読み直す
}

bool Engine::ready(std::string* err) {
  const std::string model = opts_.at("DNN_Model");
  if (loaded_model_ != model || !infer_.is_open()) {
    if (!infer_.version().size() && !infer_.load(env_or("LIBRA_ORT_LIB", ""), err)) return false;
    if (!infer_.open(model, std::max(1, geti("Threads")), opts_.at("DNN_Provider"), err)) return false;
    loaded_model_ = model;
    auto k = model.find_last_of("/\\");
    out("info string model " + (k == std::string::npos ? model : model.substr(k + 1)) + " onnxruntime " + infer_.version() + " provider " + infer_.provider());
    if (!infer_.fallback().empty()) out("info string provider fallback " + infer_.fallback());
  }
  int threads = std::max(1, geti("Threads")), mate = std::max(0, geti("Mate_Nodes"));
  if (!eng_ || eng_threads_ != threads || eng_mate_ != mate) {
    SearchConfig cfg;
    cfg.external = true;
    cfg.mate_nodes_root = mate;
    cfg.policy_topk = 300;
    eng_.reset(new SelfPlay(cfg, 1, std::uint64_t(now_s() * 1000) & 0xFFFF, threads));
    eng_threads_ = threads;
    eng_mate_ = mate;
  }
  const int batch = std::min(1024, std::max(1, geti("DNN_Batch_Size")));
  if (batch != batch_) {
    batch_ = batch;
    sq_.assign(size_t(batch) * SQ_NB * SQ_FEATS, 0.f);
    glob_.assign(size_t(batch) * GLOB_FEATS, 0.f);
    logits_.assign(size_t(batch) * POLICY_SIZE, 0.f);
    wdl_.assign(size_t(batch) * 3, 0.f);
  }
  return true;
}

// 葉を最大 DNN_Batch_Size 個まとめて 1 回の推論で評価する（評価待ちの枝には仮の負けを置いて散らす）
bool Engine::evaluate(std::string* err) {
  int k = eng_->collect_batch(0, batch_, sq_.data(), glob_.data());
  if (k == 0) return true;
  if (!infer_.run(k, sq_.data(), glob_.data(), logits_.data(), wdl_.data(), err)) return false;
  eng_->apply_batch(0, logits_.data(), wdl_.data(), k);
  return true;
}

void Engine::info_lines(const SearchResult& r, double elapsed_s, const char* phase, int ply, const char* method) const {
  int k = std::max(1, geti("MultiPV"));
  std::string pv0;
  for (std::uint32_t m : r.pv) pv0 += (pv0.empty() ? "" : " ") + move_to_usi(Move(m));
  for (size_t i = 0; i < r.cands.size() && int(i) < k; ++i) {
    const Candidate& c = r.cands[i];
    double p = (c.q + 1) / 2;
    std::ostringstream s;
    s << "info depth 1 seldepth " << r.pv.size() << " multipv " << (i + 1) << " score cp " << winrate_to_cp(p) << " winrate ";
    s.precision(4);
    s << std::fixed << p << " nodes " << r.sims << " time " << int(elapsed_s * 1000) << " pv " << (i == 0 ? pv0 : move_to_usi(Move(c.move)));
    out(s.str());
  }
  out(std::string("info string phase ") + phase + " ply " + std::to_string(ply) + " method " + method);
}

// scale.json（libra-scale）の "balanced": [["5i","5a"], ...] だけを読む最小の走査
bool Engine::load_scale(std::string* err) {
  const std::string path = opts_.at("Scale_Table");
  if (path == scale_loaded_) return true;
  scale_.clear();
  scale_loaded_ = path;
  if (path.empty()) return true;
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (err) *err = "cannot open Scale_Table " + path;
    return false;
  }
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  size_t k = s.find("\"balanced\"");
  if (k == std::string::npos) {
    if (err) *err = "no \"balanced\" in " + path;
    return false;
  }
  k = s.find('[', k);
  int depth = 0;
  std::vector<int> sqs;
  for (size_t i = k; i < s.size(); ++i) {
    char c = s[i];
    if (c == '[') ++depth;
    else if (c == ']') {
      if (--depth == 0) break;
    } else if (c == '"') {
      size_t e = s.find('"', i + 1);
      if (e == std::string::npos) break;
      int sq = sq_from_usi(s.substr(i + 1, e - i - 1));
      if (sq != SQ_NONE) sqs.push_back(sq);
      i = e;
    }
  }
  for (size_t i = 0; i + 1 < sqs.size(); i += 2) scale_.push_back({sqs[i], sqs[i + 1]});
  if (scale_.empty()) {
    if (err) *err = "empty balanced set in " + path;
    return false;
  }
  return true;
}

bool Engine::scale_move(const Position& pos, std::string* move) {
  if (scale_.empty() || pos.phase() != PHASE_FUSEKI || pos.ply() > 1) return false;
  std::vector<std::pair<int, int>> c;
  if (pos.ply() == 0) c = scale_;
  else {
    int kb = pos.king_sq(BLACK);
    for (auto& p : scale_)
      if (p.first == kb) c.push_back(p);
  }
  if (c.empty()) return false;
  std::uniform_int_distribution<int> d(0, int(c.size()) - 1);
  auto& p = c[d(rng_)];
  int sq = pos.ply() == 0 ? p.first : p.second;
  *move = "K*" + sq_to_usi(sq);
  return true;
}

static int arg_int(const std::vector<std::string>& a, const char* key, int def) {
  for (size_t i = 0; i + 1 < a.size(); ++i)
    if (a[i] == key) return std::atoi(a[i + 1].c_str());
  return def;
}
static bool has(const std::vector<std::string>& a, const char* key) {
  for (const std::string& s : a)
    if (s == key) return true;
  return false;
}

void Engine::go(const std::vector<std::string>& args) {
  std::string err;
  if (!ready(&err)) {
    out("info string not ready: " + err);
    out("bestmove resign");
    return;
  }
  const Mode mode = opts_.at("Fuseki_Mode") == "fuseki" ? MODE_FUSEKI : MODE_TENBIN;
  Position pos;
  if (!pos.set_position(position_line_, mode)) {
    out("info string bad position: " + position_line_);
    out("bestmove resign");
    return;
  }
  const char* phase = pos.phase() == PHASE_FUSEKI ? "fuseki" : "normal";
  const int ply = pos.ply();
  const Color turn = pos.turn();
  Outcome oc = pos.outcome();
  if (oc.result != ONGOING) {
    if (oc.reason == R_RULING41 && turn == BLACK) {
      out("info string phase normal ply 40 method ruling41");
      out("bestmove win");
    } else {
      out(std::string("info string game over: ") + result_name(oc.result) + " " + reason_name(oc.reason));
      out("bestmove resign");
    }
    return;
  }
  if (pos.phase() == PHASE_NORMAL && getb("Declare_Win") && pos.can_declare(turn)) {
    out("info string phase normal ply " + std::to_string(ply) + " method declaration");
    out("bestmove win");
    return;
  }
  MoveList ml;
  pos.legal_moves(ml);
  if (ml.n == 0) {
    out("bestmove resign");
    return;
  }
  // 両玉の配置: 玉配置表（Scale_Table）があれば釣り合い集合から一様に選ぶ
  if (!load_scale(&err)) out("info string " + err);
  std::string sm;
  if (scale_move(pos, &sm) && pos.is_legal(move_from_usi(sm))) {
    out("info depth 1 multipv 1 score cp 0 winrate 0.5000 nodes 0 time 0 pv " + sm);
    out("info string phase fuseki ply " + std::to_string(ply) + " method scale");
    out("bestmove " + sm);
    return;
  }
  // 思考量
  long sims = pos.phase() == PHASE_FUSEKI ? geti("Sims_Fuseki") : geti("Sims_Normal");
  const bool infinite = has(args, "infinite");
  double deadline = -1;
  if (has(args, "nodes")) sims = arg_int(args, "nodes", int(sims));
  else if (has(args, "movetime")) deadline = now_s() + arg_int(args, "movetime", 1000) / 1000.0;
  else if (has(args, "btime") || has(args, "wtime")) {
    const char* key = turn == BLACK ? "btime" : "wtime";
    const char* ik = turn == BLACK ? "binc" : "winc";
    double remain = arg_int(args, key, 0), byo = arg_int(args, "byoyomi", 0), inc = arg_int(args, ik, 0);
    double budget_ms = byo + inc + remain / 30.0;
    deadline = now_s() + std::max(0.05, budget_ms / 1000.0 * 0.9);
  }
  if (infinite || deadline >= 0) sims = 1000000000L;
  const double t0 = now_s();
  if (!eng_->set_position(0, position_line_, int(sims), true, mode)) {
    out("bestmove resign");
    return;
  }
  bool fail = false;
  while (!eng_->idle(0)) {
    if (stop_flag || (deadline >= 0 && now_s() >= deadline)) {
      eng_->finish_now(0);
      if (!eng_->idle(0)) {
        if (!evaluate(&err)) { fail = true; break; }
        eng_->finish_now(0);
      }
      break;
    }
    if (!evaluate(&err)) { fail = true; break; }
  }
  while (!fail && !eng_->idle(0))
    if (!evaluate(&err)) { fail = true; break; }
  if (fail) {
    out("info string inference failed: " + err);
    out("bestmove resign");
    return;
  }
  const SearchResult& r = eng_->result(0);
  if (infinite)
    while (!stop_flag) std::this_thread::sleep_for(std::chrono::milliseconds(20));
  if (!r.ready || r.best == MOVE_NONE) {
    // 探索が 1 回も終わらないうちに stop された場合など: 投了ではなく合法手の先頭を指す
    out("info string no search result; playing the first legal move");
    out("bestmove " + move_to_usi(ml.m[0]));
    return;
  }
  const char* method = pos.phase() == PHASE_FUSEKI ? "mcgs" : "mcts";
  if (r.cands.size() == 1 && r.sims == 0) method = "proof";
  info_lines(r, now_s() - t0, phase, ply, method);
  out("bestmove " + move_to_usi(Move(r.best)));
}

void Engine::handle(const std::string& line) {
  std::vector<std::string> t;
  {
    std::istringstream is(line);
    std::string w;
    while (is >> w) t.push_back(w);
  }
  if (t.empty()) return;
  const std::string& cmd = t[0];
  if (cmd == "usi") {
    out(std::string("id name LibraShogi ") + VERSION);
    out("id author kotenbu");
    declare_options();
    out("usiok");
  } else if (cmd == "isready") {
    std::string err;
    if (!ready(&err)) out("info string failed to load model: " + err);
    out("readyok");
  } else if (cmd == "setoption") {
    setoption(t);
  } else if (cmd == "position") {
    position_line_ = line;
  } else if (cmd == "go") {
    go(std::vector<std::string>(t.begin() + 1, t.end()));
  } else if (cmd == "quit") {
    quit = true;
  }
  // usinewgame / stop / gameover / ponderhit: 何もしない（stop は読み取りスレッドが stop_flag を立てる）
}

}  // namespace libra_engine
