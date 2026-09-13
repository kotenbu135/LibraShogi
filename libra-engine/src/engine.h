// SPDX-License-Identifier: Apache-2.0
// Libra の USI 拡張エンジン本体（docs/protocol.md）。libra-search の外部駆動 MCGS/MCTS ＋ ONNX Runtime 推論。
#pragma once
#include <atomic>
#include <random>
#include <utility>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "libra/selfplay.h"
#include "ort_infer.h"

namespace libra_engine {

extern const char* VERSION;

class Engine {
 public:
  Engine();
  ~Engine();
  void handle(const std::string& line);   // 1 行のコマンドを処理する（go は同期。stop は stop_flag で）
  std::atomic<bool> stop_flag{false};
  bool quit = false;

 private:
  std::map<std::string, std::string> opts_;
  std::string position_line_ = "position fuseki";
  OrtInfer infer_;
  std::string loaded_model_;
  std::unique_ptr<libra::SelfPlay> eng_;
  int eng_threads_ = 0, eng_mate_ = -1;
  int batch_ = 0;                          // 1 回の推論にまとめる葉の数（DNN_Batch_Size）。バッファはこの大きさ
  std::vector<float> sq_, glob_, logits_, wdl_;
  std::vector<std::pair<int, int>> scale_;  // Scale_Table の釣り合い集合（kb, kw）
  std::string scale_loaded_;
  std::mt19937 rng_{std::random_device{}()};
  bool load_scale(std::string* err);
  bool scale_move(const libra::Position& pos, std::string* move);  // 1・2 手目を表から決める

  int geti(const std::string& k) const;
  bool getb(const std::string& k) const;
  void out(const std::string& s) const;
  void declare_options() const;
  void setoption(const std::vector<std::string>& t);
  bool ready(std::string* err);
  bool evaluate(std::string* err);
  void go(const std::vector<std::string>& args);
  void info_lines(const libra::SearchResult& r, double elapsed_s, const char* phase, int ply, const char* method) const;
};

int winrate_to_cp(double p);  // docs/protocol.md の換算（S=435、offset +34、JS の Math.round）

}  // namespace libra_engine
