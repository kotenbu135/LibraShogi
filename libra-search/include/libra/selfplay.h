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
  // true なら σ に入れる q を著者の実装 mctx（qtransform_completed_by_mix_value）と同じ completed Q にする:
  // 未訪問の手は v_mix（根のネットの値と訪問した手の値の事前確率つき平均）で補い、根の手の間で [0, 1] に最小・最大で正規化する。
  // mctx の既定は c_scale（value_scale）0.1。false（既定）は 2026-09-17 までの形（未訪問は根の平均、正規化なし、q ∈ [−1, 1]）
  bool gumbel_rescale = false;
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
  bool defer_root_proof = false; // 根の証明探索を proof() の段に回す（根の評価を先に出す。棋譜は変わらない。external では無視）
  // 対局ごとに直近の手の探索で評価した局面のネットの出力を持ち、特徴量が同じ葉は評価に出さずに collect の中で展開する
  // （棋譜は変わらない。ネットの重みを替えたら clear_eval_cache を呼ぶ。2 つのネットで打つときは set_two_nets で両方の出力を持つ。
  // external では無視）
  bool eval_cache = false;
  // 根の Gumbel ノイズ。false なら候補の絞り込みと最終選択を log π + σ(q̂) だけで行う（手が乱数に依らない。評価・計測用。
  // 自己対局は true のまま。玉配置と布石は乱数で選ぶ）
  bool gumbel_noise = true;
  std::vector<int> king_pairs;  // 自己対局の玉配置を限定する（kb0, kw0, kb1, kw1, ...）。空なら 36×36 から一様
  std::vector<std::vector<std::uint32_t>> openings;  // 開始局面の手順（玉 2 手を含む）。搾取者が見つけた布石を本体の分布に混ぜる
  float openings_prob = 0.0f;    // 新規対局が openings から始まる確率
  bool prune_gote_rank4 = false; // 後手玉を一〜三段目に限る（四段目は桂打ちで先手の裁定勝ち。libra-scale の剪定と同じ）
  // 投了（AlphaGo Zero [Silver+ 2017] Methods「Resignation」）。resign_threshold <= 0 で無効（既定）。
  // 規則: 手番側の探索後の値 root_q が −resign_threshold 以下の状態がその側の連続 resign_runs 手続いたら、
  // その手を指した後にその側が投了する。原典は「根と最善の子の両方が下回ったら」だが、棋譜に子の値を残していないので根だけで見る
  // （そのぶん投了しやすい。docs/decisions.md 2026-09-19）。
  // resign_disable_prob の割合の対局は投了させず最後まで打つ（誤投了の割合を測り続けるため。原典と同じ 10%）。
  // 無効のときは乱数を一切引かないので、入れる前と棋譜は同じ
  float resign_threshold = 0.0f;
  int resign_runs = 1;
  float resign_disable_prob = 0.1f;
  int resign_min_ply = 40;       // これ未満の手数（布石）では投了しない
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
  std::uint64_t cache_hits = 0;  // 評価に出さずにキャッシュから展開した葉（eval_cache）
  std::uint64_t mate_found = 0, proof_found = 0, proof_nodes = 0, proof_calls = 0;  // 証明探索の統計
  std::uint64_t results[3] = {0, 0, 0};  // 先手勝ち・引き分け・後手勝ち
  std::uint64_t ruling41 = 0, no_legal = 0, sennichite = 0, perpetual = 0, max_ply = 0, timeout = 0;
  std::uint64_t resign = 0;  // 投了で終わった対局（resign_disable_prob の対局は数えない）
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
  // cfg.defer_root_proof のとき、collect で評価に出した根の証明探索を解く。collect の後・apply の前（GPU の評価中）に呼ぶ。
  // 呼ばなければ apply がその場で解く
  void proof();
  std::vector<GameRecord> take_finished();
  SelfPlayStats stats() const { return stats_; }
  void set_active(int n);  // 同時進行数を絞る（throttle）。n 以降の対局は止めたまま保持する
  void set_openings(std::vector<std::vector<std::uint32_t>> openings, float prob);  // 対局の合間（collect/apply の外）に呼ぶ
  // eval_cache: ネットの重みを替えたら呼ぶ（持っている評価を捨てる）。collect/apply の外で呼ぶ
  void clear_eval_cache() { ++eval_gen_; }
  void set_eval_cache(bool on) {
    cfg_.eval_cache = on;
    ++eval_gen_;
  }
  bool eval_cache() const { return cfg_.eval_cache; }
  // 2 つのネットで打つ（搾取者・リーグ）: 偶数枠は net 0 が先手、奇数枠は net 1 が先手。根の手番の側のネットで価値と方策を取り、
  // opponent_prior なら net 0 の根の木の中の net 1 の手番の葉だけ方策を net 1 から取る（価値は net 0）。
  // 評価は apply2 で両方のネットの全行を受け取り、行ごとに選ぶ。eval_cache は両方の出力を持つので、手番でネットが替わっても使える。
  // collect/apply の外で呼ぶ（持っている評価を捨てる）
  void set_two_nets(bool on, bool opponent_prior);
  bool two_nets() const { return two_nets_; }
  // 側ごとに違う探索設定で打つ（σ の形の比較など。docs/ls2-settings.md §2）。set_two_nets と同じ枠の約束で、
  // 偶数枠は cfg（A 側）が先手、奇数枠は cfg_b（B 側）が先手。いま考えている手番の側の設定で読む。
  // 側ごとに変えられるのは読む手の選び方だけ（σ: c_visit・c_scale・gumbel_rescale、読む回数、Gumbel の候補数とノイズ、PUCT）で、
  // ほかの項目（局面の作り方・詰み探索・記録の形）が cfg と違えば std::invalid_argument。collect/apply の外で呼ぶ
  void set_side_config(const SearchConfig& cfg_b);
  bool side_configs() const { return has_cfg_b_; }
  // set_two_nets のときの apply。logits0/wdl0 は net 0、logits1/wdl1 は net 1 の全行（形は apply と同じ）
  void apply2(const float* logits0, const float* wdl0, const float* logits1, const float* wdl1);
  int active() const { return active_; }
  // 各対局のルート（いま考えている手番）の色を書く（0 先手、1 後手）。評価対局で「どちらのネットで読むか」を決めるのに使う
  void root_turns(std::int8_t* out) const;
  // 直前の collect で各行に書いた葉の手番を書く（0 先手、1 後手。葉を出さなかった行は根の手番）。
  // 搾取者の探索木の中で、相手の手番の葉の方策を相手のネットから取るのに使う
  void leaf_turns(std::int8_t* out) const;
  // 外部駆動（cfg.external）: 枠 slot に局面を与えて sims 回読む。読み終わると idle になり result が取れる
  bool set_position(int slot, const std::string& usi_line, int sims, bool full, Mode mode = MODE_TENBIN);
  bool idle(int slot) const;
  // 外部駆動の複数葉の同時評価: 枠 slot の葉を最大 max_leaves 個選んで特徴を書く（sq: max_leaves×81×SQ_FEATS）。
  // 評価待ちの枝には仮の負け（virtual loss）を置いて、同じ葉を選ばないようにする。戻り値は書いた数（読み終わりなら 0）。
  // 根が未評価のうちは根だけを返す。max_leaves 1 なら collect / apply と同じ探索になる
  int collect_batch(int slot, int max_leaves, float* sq, float* glob);
  // collect_batch で出した k 個の評価を受け取って逆伝播する（finish_now で捨てた後なら無視する）
  void apply_batch(int slot, const float* logits, const float* wdl, int k);
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
  std::uint64_t eval_gen_ = 1;  // clear_eval_cache のたびに進める。対局のキャッシュの世代と違えば捨てる
  bool two_nets_ = false;       // set_two_nets
  bool opponent_prior_ = false;
  SearchConfig cfg_b_;          // set_side_config: B 側（奇数枠の先手）の探索設定
  bool has_cfg_b_ = false;
  const SearchConfig& cfg_for(const Game& g) const;  // いま考えている手番の側の設定
  // 対局ごとの処理はその対局のデータだけを触る（並列に呼べる）。統計と終局記録は対局側に貯め、あとで集める
  void step_game(Game& g);  // 次の葉まで進める（終局・着手・新規対局を含む）
  int descend(Game& g);     // ルートから 1 回選ぶ（selfplay.cpp の戻り値の説明）
  void apply_game(Game& g, const float* logits, const float* wdl);
  void apply_game2(Game& g, const float* logits0, const float* wdl0, const float* logits1, const float* wdl1);
  void apply_all(const std::function<void(Game&, int)>& eval_row);  // apply / apply2 の共通部分（eval_row が行 i の評価を入れる）
  // set_two_nets のとき、対局の今の葉の (価値を取るネット, 方策を取るネット)。0 か 1
  std::pair<int, int> nets_for(const Game& g) const;
  // 葉の鍵（h, aux）がキャッシュにあれば展開する。0 = 無い、1 = 展開した（次の葉へ進める）、2 = 本将棋の根を展開して証明探索の結果を待つ
  int use_cached(Game& g, std::uint64_t h, std::uint32_t aux);
  void finish_move(Game& g);
  void play_forced(Game& g, Move m, float value);
  // 投了の判定。指す側の値の連続を数え、投了するなら true（指した後に Position::resign を呼ぶ）
  bool resign_check(Game& g, Color mover, float root_q);
  void start_game(Game& g);
  void end_game(Game& g);
  Move root_proof(Game& g, float& value);  // 根の証明探索。手番側の勝ちが証明できればその手（無ければ MOVE_NONE）
  // serial_below: n がこれ未満なら呼んだスレッドだけで回す（0 ならスレッド数の 2 倍）
  void parallel_for(int n, const std::function<void(int)>& f, int serial_below = 0);
  void gather();
  struct Pool;                  // parallel_for の常駐スレッド（最初の並列呼び出しで作る）
  std::unique_ptr<Pool> pool_;  // games_ より後に宣言する（先に止めて join する）
};

}  // namespace libra
