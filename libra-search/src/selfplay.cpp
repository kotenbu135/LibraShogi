// SPDX-License-Identifier: Apache-2.0
#include "libra/selfplay.h"

#include "libra/dfpn.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace libra {

namespace {

constexpr std::uint32_t NONE = 0xffffffffu;

struct Edge {
  Move move;
  std::uint16_t index;     // 方策の添字（POLICY_SIZE 2268 未満）
  std::uint16_t vloss = 0; // 評価待ちの葉へ向かう数（複数葉の同時評価のときだけ 0 以外）
  float prior;
  int visits = 0;
  float wsum = 0;  // 親の手番側から見た値の和
  std::uint32_t child = NONE;
};

struct Node {
  std::uint64_t key;
  int visits = 0;
  float wsum = 0;     // この局面の手番側から見た値の和
  float net_value = 0;
  bool expanded = false;
  bool terminal = false;
  std::uint16_t vloss = 0;  // この節点を通る評価待ちの数（葉自身も数える）
  float terminal_value = 0;
  std::vector<Edge> edges;
};

float wdl_value(const float* w) { return w[0] - w[2]; }

// 仮の訪問を含めた辺の値。仮の訪問は親の手番側の負け（−1）として数える。仮の訪問が無ければ edge_q と同じ
float edge_q_vl(const Node& parent, const Edge& e) {
  if (e.vloss == 0) return e.visits > 0 ? e.wsum / e.visits : (parent.visits ? parent.wsum / parent.visits : parent.net_value);
  return (e.wsum - float(e.vloss)) / float(e.visits + e.vloss);
}

constexpr int kCacheMoves = 3;  // eval_cache: 直近何手の探索の評価を持つか（当たりはほぼ 2 手前まで。measurements.md 2026-09-14）
constexpr int kCacheHops = 64;  // eval_cache: 1 対局が 1 回の collect で当たりをたどる上限

// eval_cache の鍵。ネットの入力（write_features）は、局面キー（盤・持ち駒・手番）と、段階・布石の手数・本将棋の手数・
// 繰り返しの回数・モード（aux にまとめる）だけで決まる（手数の上限は対局を通して同じ）。特徴量を書かずに引けるよう、
// この組を鍵にする。h は表の番号で、当たりは局面キーと aux の一致で決める
std::uint64_t eval_key(const Position& p, std::uint32_t& aux) {
  const bool fuseki = p.phase() == PHASE_FUSEKI;
  const std::uint32_t t = std::uint32_t(fuseki ? p.ply() : p.normal_ply()) & 0xFFFF;
  const std::uint32_t rep = fuseki ? 0 : std::uint32_t(std::min(p.repetition_count(), 255));
  aux = t | (rep << 16) | (std::uint32_t(fuseki) << 24) | (std::uint32_t(p.mode()) << 25);
  std::uint64_t h = p.key() ^ (std::uint64_t(aux) * 0x9E3779B97F4A7C15ull);
  h ^= h >> 31;
  return h | 1;  // 0 は「数えていない」の印
}

Node make_node(std::uint64_t key) {
  Node n;
  n.key = key;
  return n;
}

// 証明探索の置換表はスレッドごとに 1 つ持つ。solve は前の solve の項を空として扱うので、対局ごとに持つのと結果は同じ。
// 対局ごと（512 局で約 190 MiB）だと毎回離れた冷たいメモリを触るので、同じスレッドが続けて解く問題で同じ表を使い回す
DfPn& thread_dfpn() {
  thread_local DfPn d{12};
  return d;
}

float gumbel_noise(std::mt19937_64& rng) {
  std::uniform_real_distribution<float> u(1e-7f, 1.0f - 1e-7f);
  return -std::log(-std::log(u(rng)));
}

}  // namespace

struct SelfPlay::Game {
  Position pos;      // 現在の対局局面（ルート）
  Position sp;       // 探索用の作業局面
  std::vector<Node> nodes;
  std::unordered_map<std::uint64_t, std::uint32_t> table;  // 布石の合流（この手の探索内）
  std::mt19937_64 rng;
  GameRecord rec;
  // この手の探索
  bool full = false;
  int budget = 0, sims = 0;
  std::vector<int> cand;       // ルートの候補（edge の添字）
  std::vector<float> gumbel;   // 候補ごとの Gumbel ノイズ
  int sh_phase = 0, sh_phases = 1, sh_target = 0;
  bool root_ready = false;
  // 待ち中の葉
  bool pending = false;
  std::uint32_t leaf = NONE;
  std::vector<std::pair<std::uint32_t, int>> path;  // (node, edge)
  std::vector<Move> path_moves;
  int moves_made = 0;
  int slot = 0;
  int resign_run[2] = {0, 0};  // 投了の判定: 値が −しきい値 以下だった連続の手数（0 先手、1 後手）
  bool resign_off = false;     // この対局は投了させず最後まで打つ（resign_disable_prob）
  bool idle = false;         // 外部駆動で局面待ち
  int forced_budget = -1;    // 外部駆動の読みの回数
  bool forced_full = true;
  SearchResult result;
  std::vector<GameRecord> done;  // 終局した記録（gather で集める）
  SelfPlayStats st;              // この対局の統計（gather で集める）
  MateProblem mate_prob;
  Ruling41Problem ruling_prob;
  Mate41Problem mate41_prob;
  std::unordered_map<std::uint64_t, float> proof_cache;  // 鍵 → 手番側の値（±1）。この手の探索内
  // 根の証明探索の先送り（cfg.defer_root_proof）: 1 = 根を評価に出して proof() を待つ、2 = 解いた（proof_move は証明手か MOVE_NONE）
  int proof_state = 0;
  Move proof_move = MOVE_NONE;
  float proof_value = 0.0f;
  // 外部駆動の複数葉の同時評価（collect_batch → apply_batch）で評価待ちの葉。g.sp はルートに戻してある
  struct Pending {
    std::uint32_t leaf;
    std::vector<std::pair<std::uint32_t, int>> path;
    std::vector<Move> moves;
  };
  std::vector<Pending> batch;
  // ネットの出力のキャッシュ（cfg.eval_cache）: 特徴量のハッシュ → 展開した辺と値。直近 kCacheMoves 手の探索で評価したもの
  struct CachedEval {
    std::uint64_t key = 0;  // 局面キー
    std::uint32_t aux = 0;  // 段階・手数・繰り返しの回数・モード（eval_key）
    int move_no = 0;        // 評価したときの moves_made
    float net_value = 0;
    std::vector<Edge> edges;  // 事前確率まで入れた辺（訪問数・子は空）
    // set_two_nets のとき: net_value と edges の事前確率は net 0、こちらは net 1 の出力（edges と同じ順）
    float net_value1 = 0;
    std::vector<float> prior1;
  };
  std::unordered_map<std::uint64_t, CachedEval> eval_cache;
  std::uint64_t eval_cache_gen = 0;
  std::uint64_t leaf_hash = 0;  // collect で評価に出した葉の鍵（0 = 数えていない）
  std::uint32_t leaf_aux = 0;
  std::int8_t leaf_turn = 0;    // collect で評価に出した葉の手番（0 先手、1 後手）。葉を出さない行は根の手番

  float root_q() const {
    const Node& r = nodes[0];
    return r.visits ? r.wsum / r.visits : r.net_value;
  }
};

SelfPlay::SelfPlay(const SearchConfig& cfg, int n_games, std::uint64_t seed, int threads)
    : cfg_(cfg), active_(n_games), threads_(std::max(1, threads)), rng_(seed) {
  for (int i = 0; i < n_games; ++i) {
    games_.push_back(std::make_unique<Game>());
    games_.back()->rng.seed(rng_());
    games_.back()->slot = i;
    start_game(*games_.back());
  }
}

SelfPlay::~SelfPlay() = default;

void SelfPlay::set_openings(std::vector<std::vector<std::uint32_t>> openings, float prob) {
  cfg_.openings = std::move(openings);
  cfg_.openings_prob = prob;
}

void SelfPlay::set_active(int n) { active_ = std::max(1, std::min(n, int(games_.size()))); }

void SelfPlay::start_game(Game& g) {
  g.resign_run[0] = g.resign_run[1] = 0;
  g.resign_off = false;
  if (cfg_.resign_threshold > 0.0f && cfg_.resign_disable_prob > 0.0f) {
    std::uniform_real_distribution<float> ur(0.0f, 1.0f);
    g.resign_off = ur(g.rng) < cfg_.resign_disable_prob;
  }
  g.eval_cache.clear();
  g.leaf_hash = 0;
  g.pos.reset(MODE_TENBIN);
  g.pos.set_max_ply(cfg_.max_ply, cfg_.count_from_41);
  if (cfg_.external) {
    g.idle = true;
    g.pending = false;
    g.nodes.clear();
    g.table.clear();
    g.proof_cache.clear();
    g.root_ready = false;
    g.sims = 0;
    return;
  }
  // 開始局面: 確率 openings_prob で openings（玉 2 手を含む手順）から始める。手順の手は探索しないので方策ターゲットは無し
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  if (!cfg_.openings.empty() && u01(g.rng) < cfg_.openings_prob) {
    std::uniform_int_distribution<int> d(0, int(cfg_.openings.size()) - 1);
    const std::vector<std::uint32_t>& op = cfg_.openings[d(g.rng)];
    g.rec = GameRecord();
    g.rec.slot = g.slot;
    g.rec.v41 = 0;
    g.moves_made = 0;
    bool ok = op.size() >= 2;
    for (size_t i = 0; ok && i < op.size(); ++i) {
      Move m = Move(op[i]);
      if (!g.pos.is_legal(m) || g.pos.outcome().result != ONGOING) { ok = false; break; }
      g.pos.do_move(m);
      if (i == 0) g.rec.kb = to_sq(m);
      else if (i == 1) g.rec.kw = to_sq(m);
      else {
        MoveRecord mr;
        mr.move = m;
        mr.full = false;
        mr.root_q = 0;
        g.rec.moves.push_back(std::move(mr));
        g.moves_made++;
      }
    }
    if (ok) {
      g.nodes.clear();
      g.table.clear();
      g.proof_cache.clear();
      g.root_ready = false;
      g.pending = false;
      g.sims = 0;
      return;
    }
    g.pos.reset(MODE_TENBIN);
    g.pos.set_max_ply(cfg_.max_ply, cfg_.count_from_41);
  }
  // 玉配置のペア: 既定は 36×36 から一様。cfg.king_pairs があればその中から一様（libra-scale の検証対局）
  int kb, kw;
  if (cfg_.king_pairs.size() >= 2) {
    std::uniform_int_distribution<int> d(0, int(cfg_.king_pairs.size() / 2) - 1);
    int i = d(g.rng);
    kb = cfg_.king_pairs[2 * i];
    kw = cfg_.king_pairs[2 * i + 1];
  } else {
    std::uniform_int_distribution<int> d(0, 35);
    kb = make_sq(d(g.rng) % 9, 5 + d(g.rng) / 9);
    if (cfg_.prune_gote_rank4) {
      std::uniform_int_distribution<int> d27(0, 26);
      int x = d27(g.rng);
      kw = make_sq(x % 9, x / 9);
    } else {
      kw = make_sq(d(g.rng) % 9, d(g.rng) / 9);
    }
  }
  g.pos.do_move(make_drop(KING, kb));
  g.pos.do_move(make_drop(KING, kw));
  g.rec = GameRecord();
  g.rec.slot = g.slot;
  g.rec.kb = kb;
  g.rec.kw = kw;
  g.rec.v41 = 0;
  g.moves_made = 0;
  g.nodes.clear();
  g.table.clear();
  g.proof_cache.clear();
  g.root_ready = false;
  g.pending = false;
  g.sims = 0;
}

void SelfPlay::end_game(Game& g) {
  Outcome o = g.pos.outcome();
  g.rec.result = o.result == BLACK_WIN ? 1 : o.result == WHITE_WIN ? -1 : 0;
  g.rec.reason = reason_name(o.reason);
  g.rec.plies = g.pos.ply();
  if (g.rec.sfen41.empty()) g.rec.sfen41 = g.pos.sfen();
  if (g.pos.ply() <= 40) g.rec.v41 = float(g.rec.result);
  g.st.games++;
  g.st.results[1 - g.rec.result]++;
  g.st.plies_sum += g.rec.plies;
  switch (o.reason) {
    case R_RULING41: g.st.ruling41++; break;
    case R_NO_LEGAL_MOVE: g.st.no_legal++; break;
    case R_SENNICHITE: g.st.sennichite++; break;
    case R_PERPETUAL_CHECK: g.st.perpetual++; break;
    case R_MAX_PLY: g.st.max_ply++; break;
    case R_TIMEOUT: g.st.timeout++; break;
    case R_RESIGN: g.st.resign++; break;
    default: break;
  }
  g.done.push_back(std::move(g.rec));
  start_game(g);
}

// ルートの探索を初期化する（ルートの評価が済んだ後に呼ぶ）
static void init_root_search(SelfPlay::Game& g, const SearchConfig& cfg) {
  Node& root = g.nodes[0];
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  g.full = u(g.rng) < cfg.full_prob;
  g.budget = g.full ? cfg.full_sims : cfg.fast_sims;
  if (g.forced_budget >= 0) {
    g.full = g.forced_full;
    g.budget = g.forced_budget;
  }
  int m = std::min(g.full ? cfg.gumbel_m_full : cfg.gumbel_m_fast, int(root.edges.size()));
  g.gumbel.assign(root.edges.size(), 0.0f);
  std::vector<int> order(root.edges.size());
  for (size_t i = 0; i < root.edges.size(); ++i) {
    if (cfg.gumbel_noise) g.gumbel[i] = gumbel_noise(g.rng);
    order[i] = int(i);
  }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return std::log(root.edges[a].prior) + g.gumbel[a] > std::log(root.edges[b].prior) + g.gumbel[b];
  });
  g.cand.assign(order.begin(), order.begin() + m);
  if (root.edges.size() == 1) g.budget = 0;  // 一手しか無ければ読まない
  g.sh_phase = 0;
  g.sh_phases = std::max(1, int(std::ceil(std::log2(std::max(2, m)))));
  g.sh_target = std::max(1, g.budget / (g.sh_phases * m));
  g.sims = 0;
  g.root_ready = true;
}

// Gumbel の σ(q)
static float sigma_q(const Node& root, float q, const SearchConfig& cfg) {
  int maxn = 0;
  for (const Edge& e : root.edges) maxn = std::max(maxn, e.visits);
  return (cfg.c_visit + maxn) * cfg.c_scale * q;
}

static float edge_q(const Node& parent, const Edge& e) {
  if (e.visits > 0) return e.wsum / e.visits;
  return parent.visits ? parent.wsum / parent.visits : parent.net_value;
}

// 根の全ての手の σ に入れる q（completed Q）。cfg.gumbel_rescale なら mctx の qtransform_completed_by_mix_value と同じ:
// 未訪問は v_mix = (v_net + ΣN · Σ_visited π q / Σ_visited π) / (ΣN + 1) で補い、[0, 1] に最小・最大で正規化する。
// そうでなければ今までどおり（未訪問は根の平均、正規化なし）
static void completed_q(const Node& root, const SearchConfig& cfg, std::vector<float>& out) {
  out.resize(root.edges.size());
  if (!cfg.gumbel_rescale) {
    for (size_t i = 0; i < root.edges.size(); ++i) out[i] = edge_q(root, root.edges[i]);
    return;
  }
  float sum_n = 0, sum_p = 0, wq = 0;
  for (const Edge& e : root.edges) {
    if (e.visits > 0) {
      sum_n += float(e.visits);
      sum_p += e.prior;
      wq += e.prior * (e.wsum / e.visits);
    }
  }
  const float mixed = sum_n > 0 ? (root.net_value + sum_n * (wq / std::max(sum_p, 1e-30f))) / (sum_n + 1) : root.net_value;
  float lo = 1e30f, hi = -1e30f;
  for (size_t i = 0; i < root.edges.size(); ++i) {
    const Edge& e = root.edges[i];
    out[i] = e.visits > 0 ? e.wsum / e.visits : mixed;
    lo = std::min(lo, out[i]);
    hi = std::max(hi, out[i]);
  }
  const float range = std::max(hi - lo, 1e-8f);
  for (float& q : out) q = (q - lo) / range;
}

// 逐次半減の次の候補を選ぶ。全候補が目標に達していれば半減。終わりなら -1
static int gumbel_pick(SelfPlay::Game& g, const SearchConfig& cfg) {
  Node& root = g.nodes[0];
  for (;;) {
    int best = -1, best_v = 1 << 30;
    for (int c : g.cand) {
      int v = root.edges[c].visits + root.edges[c].vloss;
      if (v < g.sh_target && v < best_v) {
        best_v = v;
        best = c;
      }
    }
    if (best >= 0) return best;
    if (g.cand.size() <= 1 || g.sh_phase + 1 >= g.sh_phases) return -1;
    // 半減: g + log π + σ(q̂) の上位半分を残す
    std::vector<float> cq;
    completed_q(root, cfg, cq);
    auto score = [&](int c) { return g.gumbel[c] + std::log(root.edges[c].prior) + sigma_q(root, cq[c], cfg); };
    std::sort(g.cand.begin(), g.cand.end(), [&](int a, int b) { return score(a) > score(b); });
    g.cand.resize(std::max<size_t>(1, g.cand.size() / 2));
    ++g.sh_phase;
    int remaining = g.budget - g.sims - int(g.batch.size());
    int phases_left = g.sh_phases - g.sh_phase;
    g.sh_target = root.edges[g.cand[0]].visits + root.edges[g.cand[0]].vloss +
                  std::max(1, remaining / std::max(1, phases_left * int(g.cand.size())));
  }
}

// 投了の判定（AlphaGo Zero [Silver+ 2017] Methods「Resignation」）。手番側の値が −しきい値 以下だった連続を側ごとに数え、
// resign_runs に達したら true を返す。布石（resign_min_ply 未満）では投了しない。resign_off の対局は数えるだけで投了しない
// （誤投了の割合を測り続けるための 10% の見本）。棋譜から見積もった効き目は docs/measurements.md 2026-09-19
bool SelfPlay::resign_check(Game& g, Color mover, float root_q) {
  if (cfg_.resign_threshold <= 0.0f) return false;
  int& run = g.resign_run[mover == BLACK ? 0 : 1];
  if (g.pos.ply() < cfg_.resign_min_ply || root_q > -cfg_.resign_threshold) {
    run = 0;
    return false;
  }
  ++run;
  return !g.resign_off && run >= cfg_.resign_runs;
}

void SelfPlay::finish_move(Game& g) {
  const SearchConfig& cfg = cfg_for(g);  // σ は指す側の設定（記録の形は cfg_ のまま）
  Node& root = g.nodes[0];
  // 最終選択: 残った候補から g + log π + σ(q̂) の最大
  int best = g.cand.empty() ? 0 : g.cand[0];
  float best_s = -1e30f;
  std::vector<float> cq;
  completed_q(root, cfg, cq);
  for (int c : g.cand) {
    const Edge& e = root.edges[c];
    float s = g.gumbel[c] + std::log(e.prior) + sigma_q(root, cq[c], cfg);
    if (s > best_s) {
      best_s = s;
      best = c;
    }
  }
  MoveRecord mr;
  mr.move = root.edges[best].move;
  mr.full = g.full;
  mr.root_q = g.root_q();
  if (g.full) {
    // 改善方策 π' = softmax(log π + σ(completed Q))。completed Q は completed_q（既定は未訪問を根の平均で補完）
    std::vector<std::pair<float, int>> sc;
    sc.reserve(root.edges.size());
    float mx = -1e30f;
    for (size_t i = 0; i < root.edges.size(); ++i) {
      const Edge& e = root.edges[i];
      float s = std::log(e.prior) + sigma_q(root, cq[i], cfg);
      sc.push_back({s, int(i)});
      mx = std::max(mx, s);
    }
    float z = 0;
    for (auto& p : sc) {
      p.first = std::exp(p.first - mx);
      z += p.first;
    }
    std::sort(sc.begin(), sc.end(), [](auto& a, auto& b) { return a.first > b.first; });
    float kept = 0;
    int k = std::min<int>(cfg_.policy_topk, int(sc.size()));
    for (int i = 0; i < k; ++i) kept += sc[i].first;
    for (int i = 0; i < k; ++i) mr.policy.push_back({std::uint16_t(root.edges[sc[i].second].index), sc[i].first / kept});
  }
  if (cfg_.external) {
    // 手は指さず、結果を残して局面待ちに戻る
    SearchResult& r = g.result;
    r = SearchResult();
    r.ready = true;
    r.best = root.edges[best].move;
    r.root_q = mr.root_q;
    r.sims = g.sims;
    std::vector<int> order(root.edges.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      if (root.edges[a].visits != root.edges[b].visits) return root.edges[a].visits > root.edges[b].visits;
      return root.edges[a].prior > root.edges[b].prior;
    });
    for (int i : order) {
      const Edge& e = root.edges[i];
      r.cands.push_back({e.move, e.visits, e.visits ? e.wsum / e.visits : mr.root_q, e.prior});
    }
    // 主変化: 最善手から訪問数最大の枝を辿る
    r.pv.push_back(r.best);
    std::uint32_t ni = root.edges[best].child;
    while (ni != NONE && ni < g.nodes.size()) {
      const Node& n = g.nodes[ni];
      int bi = -1;
      for (size_t i = 0; i < n.edges.size(); ++i)
        if (n.edges[i].visits > 0 && (bi < 0 || n.edges[i].visits > n.edges[bi].visits)) bi = int(i);
      if (bi < 0) break;
      r.pv.push_back(n.edges[bi].move);
      ni = n.edges[bi].child;
    }
    g.st.moves++;
    g.st.sims += g.sims;
    g.nodes.clear();
    g.table.clear();
    g.proof_cache.clear();
    g.root_ready = false;
    g.sims = 0;
    g.idle = true;
    return;
  }
  // 41 手目の局面（40 手完了、手番 先手）の探索値
  if (g.pos.phase() == PHASE_NORMAL && g.pos.ply() == 40) {
    g.rec.v41 = mr.root_q;
    g.rec.sfen41 = g.pos.sfen();
  }
  const float played_q = mr.root_q;
  g.rec.moves.push_back(std::move(mr));
  const Color mover = g.pos.turn();
  const bool resigning = resign_check(g, mover, played_q);
  g.pos.do_move(root.edges[best].move);
  g.moves_made++;
  g.st.moves++;
  if (resigning && g.pos.outcome().result == ONGOING) g.pos.resign(mover);
  for (auto it = g.eval_cache.begin(); it != g.eval_cache.end();) {
    if (it->second.move_no + kCacheMoves <= g.moves_made) it = g.eval_cache.erase(it);
    else ++it;
  }
  g.st.sims += g.sims;
  g.nodes.clear();
  g.table.clear();
  g.proof_cache.clear();
  g.root_ready = false;
  g.sims = 0;
  if (g.pos.outcome().result != ONGOING || g.moves_made >= cfg_.max_moves_per_game) {
    if (g.pos.outcome().result == ONGOING) g.pos.timeout(g.pos.turn());  // 安全弁
    end_game(g);
  }
}

// 布石終盤の証明探索。解けたら手番側から見た値（±1）を返し、best に証明手（手番側が勝つ側なら）を入れる
static bool fuseki_proof(SelfPlay::Game& g, Position& pos, const SearchConfig& cfg, float& value, Move* best) {
  if (cfg.proof_nodes <= 0 || pos.phase() != PHASE_FUSEKI || pos.ply() < cfg.proof_min_ply) return false;
  auto it = g.proof_cache.find(pos.key());
  if (it != g.proof_cache.end()) {
    value = it->second;
    return value != 0.0f;
  }
  Color mover = pos.turn();
  float v = 0.0f;
  // 先手の裁定（後手玉が当たったまま 40 手完了）
  {
    Move m = MOVE_NONE;
    ProofResult r = thread_dfpn().solve(pos, g.ruling_prob, mover == BLACK, cfg.proof_nodes, &m);
    g.st.proof_calls++;
    g.st.proof_nodes += thread_dfpn().nodes();
    if (r == PROOF_PROVEN) {
      v = mover == BLACK ? 1.0f : -1.0f;
      if (best && mover == BLACK) *best = m;
    }
  }
  // 41 手目に先手に合法手なし（後手の勝ち）
  if (v == 0.0f) {
    Move m = MOVE_NONE;
    ProofResult r = thread_dfpn().solve(pos, g.mate41_prob, mover == WHITE, cfg.proof_nodes, &m);
    g.st.proof_calls++;
    g.st.proof_nodes += thread_dfpn().nodes();
    if (r == PROOF_PROVEN) {
      v = mover == WHITE ? 1.0f : -1.0f;
      if (best && mover == WHITE) *best = m;
    }
  }
  g.proof_cache[pos.key()] = v;
  if (v != 0.0f) {
    g.st.proof_found++;
    value = v;
    return true;
  }
  return false;
}

// 証明済みの手をそのまま指す（探索しない）。方策ターゲットはその手の one-hot
void SelfPlay::play_forced(Game& g, Move m, float value) {
  if (cfg_.external) {
    g.result = SearchResult();
    g.result.ready = true;
    g.result.best = m;
    g.result.root_q = value;
    g.result.cands.push_back({m, 1, value, 1.0f});
    g.result.pv.push_back(m);
    g.nodes.clear();
    g.table.clear();
    g.proof_cache.clear();
    g.root_ready = false;
    g.sims = 0;
    g.idle = true;
    return;
  }
  MoveRecord mr;
  mr.move = m;
  mr.full = true;
  mr.root_q = value;
  mr.policy.push_back({std::uint16_t(move_index(g.pos, m)), 1.0f});
  if (g.pos.phase() == PHASE_NORMAL && g.pos.ply() == 40) {
    g.rec.v41 = value;
    g.rec.sfen41 = g.pos.sfen();
  }
  g.rec.moves.push_back(std::move(mr));
  const Color mover = g.pos.turn();
  const bool resigning = resign_check(g, mover, value);
  g.pos.do_move(m);
  g.moves_made++;
  g.st.moves++;
  if (resigning && g.pos.outcome().result == ONGOING) g.pos.resign(mover);
  g.nodes.clear();
  g.table.clear();
  g.proof_cache.clear();
  g.root_ready = false;
  g.sims = 0;
  if (g.pos.outcome().result != ONGOING || g.moves_made >= cfg_.max_moves_per_game) {
    if (g.pos.outcome().result == ONGOING) g.pos.timeout(g.pos.turn());
    end_game(g);
  }
}

// 根での証明探索: 本将棋の詰み、布石終盤の裁定・先手詰み。手番側の勝ちが証明できればその手を返す
Move SelfPlay::root_proof(Game& g, float& value) {
  value = 0.0f;
  if (g.pos.phase() == PHASE_NORMAL && cfg_.mate_nodes_root > 0) {
    Move m = MOVE_NONE;
    if (thread_dfpn().solve(g.pos, g.mate_prob, true, cfg_.mate_nodes_root, &m) == PROOF_PROVEN && m != MOVE_NONE) {
      g.st.mate_found++;
      value = 1.0f;
      return m;
    }
  } else if (g.pos.phase() == PHASE_FUSEKI) {
    Move m = MOVE_NONE;
    float v;
    if (fuseki_proof(g, g.pos, cfg_, v, &m) && v > 0 && m != MOVE_NONE) {
      value = v;
      return m;
    }
  }
  return MOVE_NONE;
}

// 根の証明探索が実際に df-pn を回すか（回さない局面は先送りしない）
static bool root_proof_runs(const Position& pos, const SearchConfig& cfg) {
  if (pos.phase() == PHASE_NORMAL) return cfg.mate_nodes_root > 0;
  return cfg.proof_nodes > 0 && pos.ply() >= cfg.proof_min_ply;
}

// 次の葉まで進める。終端は即座に逆伝播し、必要なら着手・終局・新規対局も行う
void SelfPlay::step_game(Game& g) {
  if (g.idle) return;
  for (int guard = 0; guard < 4096; ++guard) {
    if (g.idle) return;
    if (g.nodes.empty()) {
      // ルートを作る（未評価）
      Outcome o = g.pos.outcome();
      if (o.result != ONGOING) {  // 起こらないはず（着手後に終局を見る）
        if (cfg_.external) {
          g.result = SearchResult();
          g.result.ready = true;
          g.result.best = MOVE_NONE;
          g.result.root_q = 0;
          g.idle = true;
          return;
        }
        end_game(g);
        continue;
      }
      // 根での証明探索。手番側の勝ちが証明できればその手を指す
      if (cfg_.defer_root_proof && !cfg_.external && root_proof_runs(g.pos, cfg_)) {
        // 根の評価を先に出し、証明探索は proof()（GPU の評価中）で解く。証明できたら apply でこの評価を捨てて証明手を指す
        g.proof_state = 1;
      } else {
        float fv;
        Move forced = root_proof(g, fv);
        if (forced != MOVE_NONE) {
          play_forced(g, forced, fv);
          continue;
        }
      }
      g.nodes.push_back(make_node(g.pos.key()));
      g.sp = g.pos;
      g.path.clear();
      g.path_moves.clear();
      g.leaf = 0;
      g.pending = true;
      return;
    }
    if (!g.root_ready) init_root_search(g, cfg_for(g));
    if (g.sims >= g.budget) {
      if (g.proof_state == 1) {
        // eval_cache で根の証明探索の結果より先に読み終えた: 指す前にここで解く（証明できれば読みを捨てて証明手）
        g.proof_move = root_proof(g, g.proof_value);
        g.proof_state = 0;
        if (g.proof_move != MOVE_NONE) {
          g.nodes.clear();
          play_forced(g, g.proof_move, g.proof_value);
          continue;
        }
      }
      finish_move(g);
      continue;
    }
    // 選択
    int r = descend(g);
    if (r == 1) {
      g.pending = true;  // 評価待ち
      return;
    }
    if (r == 2) g.sims = g.budget;  // 逐次半減が終わった
    if (r == 3) return;             // 評価待ちの葉にぶつかった（同時評価の途中でだけ起こる。apply_batch を待つ）
  }
}

// 葉の値 v（葉の手番側から）を経路に沿って逆伝播する
static void backprop(SelfPlay::Game& g, Node& child, float v) {
  child.visits++;
  child.wsum += v;
  for (int k = int(g.path.size()) - 1; k >= 0; --k) {
    v = -v;
    Node& pn = g.nodes[g.path[k].first];
    Edge& pe = pn.edges[g.path[k].second];
    pe.visits++;
    pe.wsum += v;
    pn.visits++;
    pn.wsum += v;
  }
  g.sims++;
}

// ルートから 1 回選ぶ。戻り値: 0 = 終端か合流済みの節点の値を逆伝播した（1 シミュレーション）、
// 1 = 未評価の葉に着いた（g.leaf・g.path・g.path_moves を設定し、g.sp を葉まで進めた）、2 = 逐次半減が終わった、
// 3 = 評価待ちの葉にぶつかった（複数葉の同時評価のときだけ起こる。g.sp はルートのまま）
int SelfPlay::descend(Game& g) {
  const SearchConfig& cfg = cfg_for(g);  // set_side_config のときは、いま考えている手番の側の設定
  g.path.clear();
  g.path_moves.clear();
  std::uint32_t ni = 0;
  for (;;) {
    Node& n = g.nodes[ni];
    int ei;
    if (ni == 0) {
      ei = gumbel_pick(g, cfg);
      if (ei < 0) return 2;
    } else {
      // PUCT（評価待ちの枝は仮の負けを含める）
      float sq_n = std::sqrt(float(std::max(1, n.visits + n.vloss)));
      float best = -1e30f;
      ei = 0;
      for (size_t i = 0; i < n.edges.size(); ++i) {
        const Edge& e = n.edges[i];
        float u = edge_q_vl(n, e) + cfg.cpuct * e.prior * sq_n / (1 + e.visits + e.vloss);
        if (u > best) {
          best = u;
          ei = int(i);
        }
      }
    }
    Edge& e = n.edges[ei];
    g.path.push_back({ni, ei});
    g.path_moves.push_back(e.move);
    if (e.child == NONE) {
      // 新しい葉（布石中は合流を見る）
      Position& sp = g.sp;
      for (Move m : g.path_moves) sp.do_move(m);
      std::uint64_t key = sp.key();
      bool merged = false;
      if (sp.phase() == PHASE_FUSEKI) {
        auto it = g.table.find(key);
        if (it != g.table.end()) {
          e.child = it->second;
          merged = true;
        }
      }
      if (!merged) {
        g.nodes.push_back(make_node(key));
        e.child = std::uint32_t(g.nodes.size() - 1);
        if (sp.phase() == PHASE_FUSEKI) g.table[key] = e.child;
        Node& leaf = g.nodes.back();
        Outcome o = sp.outcome();
        if (o.result != ONGOING) {
          leaf.terminal = true;
          leaf.expanded = true;
          Color mover = sp.turn();
          leaf.terminal_value = o.result == DRAW ? cfg_.draw_value
                                : ((o.result == BLACK_WIN) == (mover == BLACK)) ? 1.0f : -1.0f;
        } else {
          float pv;
          if (fuseki_proof(g, sp, cfg_, pv, nullptr)) {
            leaf.terminal = true;
            leaf.expanded = true;
            leaf.terminal_value = pv;
          }
        }
      }
      ni = e.child;
      Node& child = g.nodes[ni];
      if (!child.expanded && child.vloss == 0) {
        g.leaf = ni;  // g.sp は葉まで進めたまま返す
        return 1;
      }
      for (size_t i = 0; i < g.path_moves.size(); ++i) sp.undo_move();
      if (!child.expanded) return 3;  // 合流した先が評価待ち
      // 合流済み or 終端 → 値を逆伝播
      backprop(g, child, child.terminal ? child.terminal_value : (child.visits ? child.wsum / child.visits : child.net_value));
      return 0;
    }
    ni = e.child;
    Node& child = g.nodes[ni];
    if (child.terminal) {
      backprop(g, child, child.terminal_value);
      return 0;
    }
    if (!child.expanded) {
      // 同時評価のときだけ起こる: 評価待ちならぶつかった。評価が来なかった葉（apply_batch に k が足りない）なら評価に出す
      if (child.vloss > 0) return 3;
      g.leaf = ni;
      for (Move m : g.path_moves) g.sp.do_move(m);
      return 1;
    }
  }
}

void SelfPlayStats::add(const SelfPlayStats& o) {
  games += o.games;
  moves += o.moves;
  sims += o.sims;
  evals += o.evals;
  cache_hits += o.cache_hits;
  mate_found += o.mate_found;
  proof_found += o.proof_found;
  proof_nodes += o.proof_nodes;
  proof_calls += o.proof_calls;
  for (int i = 0; i < 3; ++i) results[i] += o.results[i];
  ruling41 += o.ruling41;
  no_legal += o.no_legal;
  sennichite += o.sennichite;
  perpetual += o.perpetual;
  max_ply += o.max_ply;
  timeout += o.timeout;
  resign += o.resign;
  plies_sum += o.plies_sum;
}

// 常駐スレッドの組。呼ぶたびにスレッドを作らず、対局は空いたスレッドが次の番号を取る。
// 固定の等分だと重い対局（本将棋の根の詰み探索など）が 1 つの組に重なり、その組が 1 ラウンドの時間を決めていた
struct SelfPlay::Pool {
  std::vector<std::thread> workers;
  std::mutex mu;
  std::condition_variable wake, done;
  const std::function<void(int)>* f = nullptr;
  int n = 0;
  std::atomic<int> next{0};
  std::uint64_t epoch = 0;
  std::size_t acked = 0;  // この回を終えた常駐スレッドの数
  bool stop = false;

  explicit Pool(int threads) {
    for (int t = 0; t + 1 < threads; ++t) workers.emplace_back([this] { loop(); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
    }
    wake.notify_all();
    for (auto& w : workers) w.join();
  }
  void drain(const std::function<void(int)>* fn, int count) {
    for (int i; (i = next.fetch_add(1, std::memory_order_relaxed)) < count;) (*fn)(i);
  }
  void loop() {
    std::uint64_t seen = 0;
    for (;;) {
      const std::function<void(int)>* fn;
      int count;
      {
        std::unique_lock<std::mutex> lk(mu);
        wake.wait(lk, [&] { return stop || epoch != seen; });
        if (stop) return;
        seen = epoch;
        fn = f;
        count = n;
      }
      drain(fn, count);
      {
        std::lock_guard<std::mutex> lk(mu);
        ++acked;
      }
      done.notify_one();
    }
  }
  // 全スレッドがこの回を終えるまで待つので、次の回に前の回の f が残らない
  void run(int count, const std::function<void(int)>& fn) {
    {
      std::lock_guard<std::mutex> lk(mu);
      f = &fn;
      n = count;
      next.store(0);
      acked = 0;
      ++epoch;
    }
    wake.notify_all();
    drain(&fn, count);  // 呼んだスレッドも加わる
    std::unique_lock<std::mutex> lk(mu);
    done.wait(lk, [&] { return acked == workers.size(); });
  }
};

void SelfPlay::parallel_for(int n, const std::function<void(int)>& f, int serial_below) {
  if (threads_ <= 1 || n < (serial_below > 0 ? serial_below : 2 * threads_)) {
    for (int i = 0; i < n; ++i) f(i);
    return;
  }
  if (!pool_) pool_ = std::make_unique<Pool>(threads_);
  pool_->run(n, f);
}

void SelfPlay::gather() {
  for (auto& gp : games_) {
    Game& g = *gp;
    stats_.add(g.st);
    g.st = SelfPlayStats();
    for (auto& r : g.done) finished_.push_back(std::move(r));
    g.done.clear();
  }
}

// 展開した葉（edges と net_value が入った）の値を経路に沿って逆伝播し、作業局面をルートに戻す
static void finish_leaf(SelfPlay::Game& g, Node& leaf) {
  leaf.expanded = true;
  float v = leaf.net_value;
  leaf.visits++;
  leaf.wsum += v;
  for (int k = int(g.path.size()) - 1; k >= 0; --k) {
    v = -v;
    Node& pn = g.nodes[g.path[k].first];
    Edge& pe = pn.edges[g.path[k].second];
    pe.visits++;
    pe.wsum += v;
    pn.visits++;
    pn.wsum += v;
  }
  for (size_t i = 0; i < g.path_moves.size(); ++i) g.sp.undo_move();
  if (g.leaf != 0) g.sims++;  // ルート自身の評価は数えない
  g.pending = false;
}

int SelfPlay::collect(float* sq, float* glob) {
  int n = int(games_.size());
  const bool cache = cfg_.eval_cache && !cfg_.external;
  parallel_for(n, [&](int i) {
    Game& g = *games_[i];
    float* rs = sq + size_t(i) * SQ_NB * SQ_FEATS;
    float* rg = glob + size_t(i) * GLOB_FEATS;
    g.leaf_turn = g.pos.turn() == BLACK ? 0 : 1;
    if (i >= active_ || g.idle) {
      // 止めている対局: 特徴はゼロのまま（apply で無視する）
      std::fill(rs, rs + SQ_NB * SQ_FEATS, 0.0f);
      std::fill(rg, rg + GLOB_FEATS, 0.0f);
      return;
    }
    // eval_cache: 葉がキャッシュにあればその場で展開して次の葉へ進み、無い葉だけ特徴を書いて評価に出す
    for (int hop = 0;; ++hop) {
      if (!g.pending) step_game(g);
      g.leaf_hash = 0;
      if (cache && !g.idle && g.pending && hop < kCacheHops) {
        std::uint32_t aux;
        std::uint64_t h = eval_key(g.sp, aux);
        int c = use_cached(g, h, aux);
        if (c == 1) continue;
        if (c == 2) {  // 根を展開して証明探索の結果を待つ: この回は評価に出す葉が無い（行はゼロ。apply が続きを進める）
          std::fill(rs, rs + SQ_NB * SQ_FEATS, 0.0f);
          std::fill(rg, rg + GLOB_FEATS, 0.0f);
          break;
        }
        g.leaf_hash = h;
        g.leaf_aux = aux;
      }
      write_features(g.sp, rs, rg);
      g.leaf_turn = g.sp.turn() == BLACK ? 0 : 1;
      break;
    }
  });
  gather();
  return n;
}

int SelfPlay::use_cached(Game& g, std::uint64_t h, std::uint32_t aux) {
  if (g.eval_cache_gen != eval_gen_) {  // 重みが替わった
    g.eval_cache.clear();
    g.eval_cache_gen = eval_gen_;
    return 0;
  }
  auto it = g.eval_cache.find(h);
  if (it == g.eval_cache.end() || it->second.key != g.nodes[g.leaf].key || it->second.aux != aux) return 0;
  g.st.cache_hits++;
  if (g.proof_state == 1 && g.pos.phase() == PHASE_FUSEKI) {
    // 布石の根の証明探索は、評価に出さないのでここで解く（proof() を呼ばなかった apply と同じ順序）。先に木を進めると、
    // 証明できて読みを捨てたときに対局の乱数（Gumbel ノイズ・全読みの抽選）を余分に使って棋譜が変わる
    g.proof_move = root_proof(g, g.proof_value);
    g.proof_state = 0;
    if (g.proof_move != MOVE_NONE) {
      g.nodes.clear();
      g.pending = false;
      play_forced(g, g.proof_move, g.proof_value);
      return 1;
    }
  }
  Node& leaf = g.nodes[g.leaf];
  const Game::CachedEval& c = it->second;
  leaf.edges = c.edges;
  leaf.net_value = c.net_value;
  if (two_nets_) {
    const auto [value_net, prior_net] = nets_for(g);
    if (value_net == 1) leaf.net_value = c.net_value1;
    if (prior_net == 1)
      for (size_t k = 0; k < leaf.edges.size(); ++k) leaf.edges[k].prior = c.prior1[k];
  }
  finish_leaf(g, leaf);
  // 本将棋の根の詰み探索（proof_state 1）はそのまま proof()（GPU の評価中）に残し、根を展開したところで止まる。
  // 先に探索を進めると、証明できて読みを捨てたときに対局の乱数（Gumbel ノイズ・全読みの抽選）を余分に使って棋譜が変わるため。
  // apply が証明できていれば証明手を指し、できていなければ探索を続ける
  return g.proof_state == 1 ? 2 : 1;
}

// 葉の合法手の辺を作る（事前確率は set_priors で入れる）
static void make_edges(Position& sp, std::vector<Edge>& edges) {
  MoveList ml;
  sp.legal_moves(ml);
  edges.clear();
  edges.reserve(ml.n);
  for (Move m : ml) edges.push_back(Edge{m, std::uint16_t(move_index(sp, m)), 0, 0.0f});
}

// 辺の事前確率を、合法手の上でのロジットの softmax にする（下限 1e-8）
static void set_priors(std::vector<Edge>& edges, const float* logits) {
  float mx = -1e30f;
  for (Edge& e : edges) {
    e.prior = logits[e.index];
    mx = std::max(mx, e.prior);
  }
  float z = 0;
  for (Edge& e : edges) {
    e.prior = std::exp(e.prior - mx);
    z += e.prior;
  }
  for (Edge& e : edges) e.prior = std::max(e.prior / z, 1e-8f);
}

// eval_cache に書く項（重みが替わっていれば先に捨てる）。書かない葉なら nullptr
static SelfPlay::Game::CachedEval* cache_slot(SelfPlay::Game& g, bool enabled, std::uint64_t gen) {
  if (g.leaf_hash == 0 || !enabled) return nullptr;
  if (g.eval_cache_gen != gen) {
    g.eval_cache.clear();
    g.eval_cache_gen = gen;
  }
  SelfPlay::Game::CachedEval& c = g.eval_cache[g.leaf_hash];
  c.key = g.nodes[g.leaf].key;
  c.aux = g.leaf_aux;
  c.move_no = g.moves_made;
  return &c;
}

void SelfPlay::apply_game(Game& g, const float* logits, const float* wdl) {
  Node& leaf = g.nodes[g.leaf];
  make_edges(g.sp, leaf.edges);
  set_priors(leaf.edges, logits);
  leaf.net_value = wdl_value(wdl);
  if (Game::CachedEval* c = cache_slot(g, cfg_.eval_cache && !cfg_.external, eval_gen_)) {
    c->net_value = leaf.net_value;
    c->edges = leaf.edges;
  }
  g.leaf_hash = 0;
  finish_leaf(g, leaf);
  g.st.evals++;
}

std::pair<int, int> SelfPlay::nets_for(const Game& g) const {
  // 偶数枠は net 0 が先手。行の持ち主 = 手番（0 先手、1 後手）と枠の偶奇の排他的論理和
  const int swap = g.slot & 1;
  const int value_net = (g.pos.turn() == BLACK ? 0 : 1) ^ swap;
  const int leaf_owner = (g.sp.turn() == BLACK ? 0 : 1) ^ swap;
  return {value_net, opponent_prior_ ? (value_net | leaf_owner) : value_net};
}

void SelfPlay::apply_game2(Game& g, const float* logits0, const float* wdl0, const float* logits1, const float* wdl1) {
  Node& leaf = g.nodes[g.leaf];
  const auto [value_net, prior_net] = nets_for(g);
  make_edges(g.sp, leaf.edges);
  Game::CachedEval* c = cache_slot(g, cfg_.eval_cache && !cfg_.external, eval_gen_);
  if (c != nullptr) {
    set_priors(leaf.edges, logits1);
    c->prior1.resize(leaf.edges.size());
    for (size_t k = 0; k < leaf.edges.size(); ++k) c->prior1[k] = leaf.edges[k].prior;
    set_priors(leaf.edges, logits0);
    c->edges = leaf.edges;
    c->net_value = wdl_value(wdl0);
    c->net_value1 = wdl_value(wdl1);
    if (prior_net == 1)
      for (size_t k = 0; k < leaf.edges.size(); ++k) leaf.edges[k].prior = c->prior1[k];
  } else {
    set_priors(leaf.edges, prior_net == 1 ? logits1 : logits0);
  }
  leaf.net_value = wdl_value(value_net == 1 ? wdl1 : wdl0);
  g.leaf_hash = 0;
  finish_leaf(g, leaf);
  g.st.evals++;
}

void SelfPlay::set_two_nets(bool on, bool opponent_prior) {
  two_nets_ = on;
  opponent_prior_ = on && opponent_prior;
  ++eval_gen_;
}

void SelfPlay::set_side_config(const SearchConfig& cfg) {
  // 側ごとに変えられるのは読む手の選び方だけ。ほかが違えば、同じ対局の中で局面の作り方や記録の形が枠ごとに変わってしまう
  const bool same = cfg.max_ply == cfg_.max_ply && cfg.count_from_41 == cfg_.count_from_41 && cfg.policy_topk == cfg_.policy_topk &&
                    cfg.max_moves_per_game == cfg_.max_moves_per_game && cfg.draw_value == cfg_.draw_value &&
                    cfg.mate_nodes_root == cfg_.mate_nodes_root && cfg.proof_nodes == cfg_.proof_nodes &&
                    cfg.proof_min_ply == cfg_.proof_min_ply && cfg.external == cfg_.external &&
                    cfg.defer_root_proof == cfg_.defer_root_proof && cfg.eval_cache == cfg_.eval_cache &&
                    cfg.prune_gote_rank4 == cfg_.prune_gote_rank4 && cfg.king_pairs == cfg_.king_pairs &&
                    cfg.openings == cfg_.openings && cfg.openings_prob == cfg_.openings_prob &&
                    cfg.resign_threshold == cfg_.resign_threshold && cfg.resign_runs == cfg_.resign_runs &&
                    cfg.resign_disable_prob == cfg_.resign_disable_prob && cfg.resign_min_ply == cfg_.resign_min_ply;
  if (!same)
    throw std::invalid_argument("SelfPlay::set_side_config: 側ごとに変えられるのは full_sims・fast_sims・full_prob・gumbel_m_full・"
                                "gumbel_m_fast・c_visit・c_scale・gumbel_rescale・gumbel_noise・cpuct だけ");
  cfg_b_ = cfg_;  // 片側だけ違う項目を入れ替える（ほかは A 側に合わせたまま持つ）
  cfg_b_.full_sims = cfg.full_sims;
  cfg_b_.fast_sims = cfg.fast_sims;
  cfg_b_.full_prob = cfg.full_prob;
  cfg_b_.gumbel_m_full = cfg.gumbel_m_full;
  cfg_b_.gumbel_m_fast = cfg.gumbel_m_fast;
  cfg_b_.c_visit = cfg.c_visit;
  cfg_b_.c_scale = cfg.c_scale;
  cfg_b_.gumbel_rescale = cfg.gumbel_rescale;
  cfg_b_.gumbel_noise = cfg.gumbel_noise;
  cfg_b_.cpuct = cfg.cpuct;
  has_cfg_b_ = true;
}

// 偶数枠は A 側が先手、奇数枠は B 側が先手（set_two_nets と同じ約束）
const SearchConfig& SelfPlay::cfg_for(const Game& g) const {
  if (!has_cfg_b_) return cfg_;
  const int turn = g.pos.turn() == BLACK ? 0 : 1;
  return (turn ^ (g.slot & 1)) ? cfg_b_ : cfg_;
}

void SelfPlay::apply(const float* logits, const float* wdl) {
  if (two_nets_) throw std::logic_error("SelfPlay::apply: set_two_nets のときは apply2 を使う");
  apply_all([&](Game& g, int i) { apply_game(g, logits + size_t(i) * POLICY_SIZE, wdl + size_t(i) * 3); });
}

void SelfPlay::apply2(const float* logits0, const float* wdl0, const float* logits1, const float* wdl1) {
  if (!two_nets_) throw std::logic_error("SelfPlay::apply2: set_two_nets(true) の後で使う");
  apply_all([&](Game& g, int i) {
    apply_game2(g, logits0 + size_t(i) * POLICY_SIZE, wdl0 + size_t(i) * 3, logits1 + size_t(i) * POLICY_SIZE, wdl1 + size_t(i) * 3);
  });
}

void SelfPlay::apply_all(const std::function<void(Game&, int)>& eval_row) {
  int n = std::min(active_, int(games_.size()));
  parallel_for(n, [&](int i) {
    Game& g = *games_[i];
    if (g.idle || (!g.pending && g.proof_state == 0)) return;
    if (g.proof_state == 1) {  // proof() が呼ばれなかった: ここで解く
      g.proof_move = root_proof(g, g.proof_value);
      g.proof_state = 2;
    }
    if (g.proof_state == 2) {
      g.proof_state = 0;
      if (g.proof_move != MOVE_NONE) {
        // 証明できた: 根の評価は使わず（先送りしない場合は評価に出していない）証明手を指す
        g.nodes.clear();
        g.pending = false;
        play_forced(g, g.proof_move, g.proof_value);
        step_game(g);
        return;
      }
    }
    if (!g.pending) {  // eval_cache で根を展開して証明探索の結果を待っていた（この回の行は使わない）
      step_game(g);
      return;
    }
    eval_row(g, i);
    step_game(g);  // 次の葉まで進める（終局・着手を含む）
  });
  gather();
}

void SelfPlay::proof() {
  std::vector<Game*> todo;
  for (auto& gp : games_)
    if (gp->proof_state == 1) todo.push_back(gp.get());
  // 1 ラウンドに数局しかないので、2 局からスレッドに分ける
  parallel_for(int(todo.size()), [&](int i) {
    Game& g = *todo[i];
    g.proof_move = root_proof(g, g.proof_value);
    g.proof_state = 2;
  }, 2);
}

bool SelfPlay::set_position(int slot, const std::string& usi_line, int sims, bool full, Mode mode) {
  Game& g = *games_[slot];
  if (!g.pos.set_position(usi_line, mode)) return false;
  g.pos.set_max_ply(cfg_.max_ply, cfg_.count_from_41);
  g.forced_budget = sims;
  g.forced_full = full;
  g.result = SearchResult();
  g.nodes.clear();
  g.table.clear();
  g.proof_cache.clear();
  g.root_ready = false;
  g.pending = false;
  g.sims = 0;
  g.idle = false;
  g.batch.clear();
  step_game(g);
  return true;
}

bool SelfPlay::idle(int slot) const { return games_[slot]->idle; }

// 評価待ちの経路の仮の訪問を d だけ増減する
static void add_vloss(SelfPlay::Game& g, const SelfPlay::Game::Pending& p, int d) {
  for (const auto& pe : p.path) {
    Node& n = g.nodes[pe.first];
    n.vloss = std::uint16_t(n.vloss + d);
    n.edges[pe.second].vloss = std::uint16_t(n.edges[pe.second].vloss + d);
  }
  g.nodes[p.leaf].vloss = std::uint16_t(g.nodes[p.leaf].vloss + d);
}

int SelfPlay::collect_batch(int slot, int max_leaves, float* sq, float* glob) {
  Game& g = *games_[slot];
  if (g.idle || !g.batch.empty()) return 0;
  if (!g.pending) step_game(g);  // 読み終わりなら結果を出して idle、そうでなければ最初の葉で止まる
  if (g.idle) return 0;
  int k = 0;
  // g.sp にある葉の特徴を書き、ルートに戻して評価待ちに積む
  auto emit = [&] {
    write_features(g.sp, sq + size_t(k) * SQ_NB * SQ_FEATS, glob + size_t(k) * GLOB_FEATS);
    for (size_t i = 0; i < g.path_moves.size(); ++i) g.sp.undo_move();
    g.batch.push_back(Game::Pending{g.leaf, g.path, g.path_moves});
    add_vloss(g, g.batch.back(), +1);
    g.pending = false;
    ++k;
  };
  if (g.pending) emit();
  // 根が未評価なら根だけ。予算は評価待ちも数える
  for (int guard = 0; guard < 4096 && k < max_leaves && g.root_ready; ++guard) {
    if (g.sims + int(g.batch.size()) >= g.budget) break;
    int r = descend(g);
    if (r == 1) emit();
    else if (r != 0) break;  // 逐次半減の区切り（評価待ちを反映してから続ける）か、評価待ちの葉にぶつかった
  }
  return k;
}

void SelfPlay::apply_batch(int slot, const float* logits, const float* wdl, int k) {
  Game& g = *games_[slot];
  for (size_t i = 0; i < g.batch.size(); ++i) {
    Game::Pending& p = g.batch[i];
    add_vloss(g, p, -1);
    if (int(i) >= k) continue;  // 評価が来なかった葉は未評価のまま残す（次に選ばれたら評価に出す）
    g.leaf = p.leaf;
    g.path.swap(p.path);
    g.path_moves.swap(p.moves);
    for (Move m : g.path_moves) g.sp.do_move(m);
    apply_game(g, logits + size_t(i) * POLICY_SIZE, wdl + size_t(i) * 3);
  }
  g.batch.clear();
}

void SelfPlay::finish_now(int slot) {
  Game& g = *games_[slot];
  if (g.idle) return;
  if (g.nodes.empty()) {
    // ルート未作成: 1 回だけ評価させる（budget 0 なので次の apply 後に確定する）
    g.forced_budget = 0;
    g.budget = 0;
    step_game(g);
    return;
  }
  if (!g.root_ready) {
    // ルートの評価待ち: この評価は捨てられない（候補が無い）。apply 後に budget 0 で確定する
    g.forced_budget = 0;
    g.budget = 0;
    return;
  }
  if (!g.batch.empty()) {
    // 同時評価の評価待ちをまとめて捨てる
    for (const auto& p : g.batch) add_vloss(g, p, -1);
    g.batch.clear();
  }
  if (g.pending) {
    // 評価待ちの葉は捨てて、今の訪問数で決める
    for (size_t i = 0; i < g.path_moves.size(); ++i) g.sp.undo_move();
    g.pending = false;
  }
  g.budget = g.sims;
  step_game(g);
}

const SearchResult& SelfPlay::result(int slot) const { return games_[slot]->result; }

void SelfPlay::root_turns(std::int8_t* out) const {
  for (size_t i = 0; i < games_.size(); ++i) out[i] = games_[i]->pos.turn() == BLACK ? 0 : 1;
}

void SelfPlay::leaf_turns(std::int8_t* out) const {
  for (size_t i = 0; i < games_.size(); ++i) out[i] = games_[i]->leaf_turn;
}

std::vector<GameRecord> SelfPlay::take_finished() {
  std::vector<GameRecord> out;
  out.swap(finished_);
  return out;
}

}  // namespace libra
