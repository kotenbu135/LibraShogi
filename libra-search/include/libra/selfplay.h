// SPDX-License-Identifier: Apache-2.0
// 自己対局エンジン: 同時進行する多数の対局で MCGS/MCTS（Gumbel ルート）を回し、
// 葉の評価だけをバッチにして外（PyTorch）へ渡す。docs/libra-design.md §3.3、docs/libra-local.md §3。
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "libra/encoding.h"
#include "libra/position.h"

namespace libra {

struct SearchConfig {
  int full_sims = 96;       // 全読み（方策ターゲットを取る）のシミュレーション数
  int fast_sims = 24;       // 速読み（価値ターゲットのみ）
  float full_prob = 0.25f;  // 全読みの割合（playout cap randomization）
  int gumbel_m_full = 16;   // Gumbel-Top-k の候補数
  int gumbel_m_fast = 8;
  float c_visit = 50.0f;    // Gumbel の σ(q) = (c_visit + max N) · c_scale · q
  float c_scale = 1.0f;
  float cpuct = 1.5f;       // ルート以外の PUCT
  float draw_value = 0.0f;  // 引き分けの値（手番側から）
  int max_ply = 320;
  bool count_from_41 = true;
  int policy_topk = 32;     // 方策ターゲットとして保存する上位数
  int max_moves_per_game = 400;  // 安全弁（規定上は max_ply で終わる）
  int mate_nodes_root = 200;     // 本将棋の各手の根で df-pn 詰み探索に使う節点数（0 で無効）
  int proof_nodes = 1000;        // 布石終盤の証明探索（41 手目の裁定・先手詰み）の節点数（0 で無効）
  int proof_min_ply = 36;        // 証明探索を始める手数
  bool external = false;         // 外部駆動（USI エンジン用）: 局面は set_position で与え、手は指さず結果を返す
  std::vector<int> king_pairs;   // 自己対局の玉配置を限定する（kb0, kw0, kb1, kw1, ...）。空なら 36×36 から一様
  std::vector<std::vector<std::uint32_t>> openings;  // 開始局面の手順（玉 2 手を含む）。搾取者が見つけた布石を本体の分布に混ぜる
  float openings_prob = 0.0f;    // 新規対局が openings から始まる確率
  bool prune_gote_rank4 = false; // 後手玉を一〜三段目に限る（四段目は桂打ちで先手の裁定勝ち。libra-scale の剪定と同じ）
};

struct Candidate {
  std::uint32_t move;
  int visits;
  float q;      // 手番側から見た値（−1..1）
  float prior;
};

struct SearchResult {
  bool ready = false;
  std::uint32_t best = MOVE_NONE;
  float root_q = 0;
  std::vector<Candidate> cands;   // 訪問数の多い順
  std::vector<std::uint32_t> pv;  // 最善手から辿った主変化
  std::uint64_t sims = 0;
};

struct MoveRecord {
  std::uint32_t move;
  bool full;
  float root_q;  // 探索後のルート値（手番側から）
  std::vector<std::pair<std::uint16_t, float>> policy;  // 改善方策の上位（添字, 確率）。full のときだけ
};

struct GameRecord {
  int slot = 0;                     // どの同時進行枠の対局か（評価対局で先後の割り当てに使う）
  int kb, kw;                       // 先手玉・後手玉のマス
  std::vector<MoveRecord> moves;    // 3 手目から
  int result;                       // 先手から見て +1 / 0 / −1
  std::string reason;
  float v41;                        // 41 手目局面の探索値（先手から見て）。40 手で終局ならその値
  int plies;                        // 通算手数
  std::string sfen41;               // 40 手完了時の SFEN（40 手で終局しても記録）
};

struct SelfPlayStats {
  std::uint64_t games = 0, moves = 0, sims = 0, evals = 0;
  std::uint64_t mate_found = 0, proof_found = 0, proof_nodes = 0, proof_calls = 0;  // 証明探索の統計
  std::uint64_t results[3] = {0, 0, 0};  // 先手勝ち・引き分け・後手勝ち
  std::uint64_t ruling41 = 0, no_legal = 0, sennichite = 0, perpetual = 0, max_ply = 0, timeout = 0;
  double plies_sum = 0;
  void add(const SelfPlayStats& o);
};

class SelfPlay {
 public:
  SelfPlay(const SearchConfig& cfg, int n_games, std::uint64_t seed, int threads);
  ~SelfPlay();
  int n_games() const { return int(games_.size()); }
  // 各対局の葉の特徴を書く。sq: n_games×81×SQ_FEATS、glob: n_games×GLOB_FEATS。戻り値は書いた数（常に n_games）
  int collect(float* sq, float* glob);
  // 葉の評価を受け取り、逆伝播して各対局を進める。logits: n_games×POLICY_SIZE、wdl: n_games×3（勝・分・負、手番側）
  void apply(const float* logits, const float* wdl);
  std::vector<GameRecord> take_finished();
  SelfPlayStats stats() const { return stats_; }
  void set_active(int n);  // 同時進行数を絞る（throttle）。n 以降の対局は止めたまま保持する
  void set_openings(std::vector<std::vector<std::uint32_t>> openings, float prob);  // 対局の合間（collect/apply の外）に呼ぶ
  int active() const { return active_; }
  // 各対局のルート（いま考えている手番）の色を書く（0 先手、1 後手）。評価対局で「どちらのネットで読むか」を決めるのに使う
  void root_turns(std::int8_t* out) const;
  // 外部駆動（cfg.external）: 枠 slot に局面を与えて sims 回読む。読み終わると idle になり result が取れる
  bool set_position(int slot, const std::string& usi_line, int sims, bool full, Mode mode = MODE_TENBIN);
  bool idle(int slot) const;
  void finish_now(int slot);  // 今の訪問数で打ち切って結果を出す（stop）
  const SearchResult& result(int slot) const;

  struct Game;  // 実装の詳細（selfplay.cpp）

 private:
  SearchConfig cfg_;
  std::vector<std::unique_ptr<Game>> games_;
  std::vector<GameRecord> finished_;
  SelfPlayStats stats_;
  int active_;
  int threads_;
  std::mt19937_64 rng_;
  // 対局ごとの処理はその対局のデータだけを触る（並列に呼べる）。統計と終局記録は対局側に貯め、あとで集める
  void step_game(Game& g);  // 次の葉まで進める（終局・着手・新規対局を含む）
  void apply_game(Game& g, const float* logits, const float* wdl);
  void finish_move(Game& g);
  void play_forced(Game& g, Move m, float value);
  void start_game(Game& g);
  void end_game(Game& g);
  void parallel_for(int n, const std::function<void(int)>& f);
  void gather();
  struct Pool;                  // parallel_for の常駐スレッド（最初の並列呼び出しで作る）
  std::unique_ptr<Pool> pool_;  // games_ より後に宣言する（先に止めて join する）
};

}  // namespace libra
