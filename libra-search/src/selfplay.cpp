// SPDX-License-Identifier: Apache-2.0
#include "libra/selfplay.h"

#include "libra/dfpn.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace libra {

namespace {

constexpr std::uint32_t NONE = 0xffffffffu;

struct Edge {
  Move move;
  int index;      // 方策の添字
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
  float terminal_value = 0;
  std::vector<Edge> edges;
};

float wdl_value(const float* w) { return w[0] - w[2]; }

Node make_node(std::uint64_t key) {
  Node n;
  n.key = key;
  return n;
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
  bool idle = false;         // 外部駆動で局面待ち
  int forced_budget = -1;    // 外部駆動の読みの回数
  bool forced_full = true;
  SearchResult result;
  std::vector<GameRecord> done;  // 終局した記録（gather で集める）
  SelfPlayStats st;              // この対局の統計（gather で集める）
  DfPn dfpn{12};
  MateProblem mate_prob;
  Ruling41Problem ruling_prob;
  Mate41Problem mate41_prob;
  std::unordered_map<std::uint64_t, float> proof_cache;  // 鍵 → 手番側の値（±1）。この手の探索内

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
    g.gumbel[i] = gumbel_noise(g.rng);
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

// 逐次半減の次の候補を選ぶ。全候補が目標に達していれば半減。終わりなら -1
static int gumbel_pick(SelfPlay::Game& g, const SearchConfig& cfg) {
  Node& root = g.nodes[0];
  for (;;) {
    int best = -1, best_v = 1 << 30;
    for (int c : g.cand) {
      int v = root.edges[c].visits;
      if (v < g.sh_target && v < best_v) {
        best_v = v;
        best = c;
      }
    }
    if (best >= 0) return best;
    if (g.cand.size() <= 1 || g.sh_phase + 1 >= g.sh_phases) return -1;
    // 半減: g + log π + σ(q̂) の上位半分を残す
    auto score = [&](int c) {
      const Edge& e = root.edges[c];
      return g.gumbel[c] + std::log(e.prior) + sigma_q(root, edge_q(root, e), cfg);
    };
    std::sort(g.cand.begin(), g.cand.end(), [&](int a, int b) { return score(a) > score(b); });
    g.cand.resize(std::max<size_t>(1, g.cand.size() / 2));
    ++g.sh_phase;
    int remaining = g.budget - g.sims;
    int phases_left = g.sh_phases - g.sh_phase;
    g.sh_target = root.edges[g.cand[0]].visits + std::max(1, remaining / std::max(1, phases_left * int(g.cand.size())));
  }
}

void SelfPlay::finish_move(Game& g) {
  Node& root = g.nodes[0];
  // 最終選択: 残った候補から g + log π + σ(q̂) の最大
  int best = g.cand.empty() ? 0 : g.cand[0];
  float best_s = -1e30f;
  for (int c : g.cand) {
    const Edge& e = root.edges[c];
    float s = g.gumbel[c] + std::log(e.prior) + sigma_q(root, edge_q(root, e), cfg_);
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
    // 改善方策 π' = softmax(log π + σ(completed Q))。未訪問は根の値で補完
    std::vector<std::pair<float, int>> sc;
    sc.reserve(root.edges.size());
    float mx = -1e30f;
    for (size_t i = 0; i < root.edges.size(); ++i) {
      const Edge& e = root.edges[i];
      float q = e.visits > 0 ? e.wsum / e.visits : mr.root_q;
      float s = std::log(e.prior) + sigma_q(root, q, cfg_);
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
  g.rec.moves.push_back(std::move(mr));
  g.pos.do_move(root.edges[best].move);
  g.moves_made++;
  g.st.moves++;
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
    ProofResult r = g.dfpn.solve(pos, g.ruling_prob, mover == BLACK, cfg.proof_nodes, &m);
    g.st.proof_calls++;
    g.st.proof_nodes += g.dfpn.nodes();
    if (r == PROOF_PROVEN) {
      v = mover == BLACK ? 1.0f : -1.0f;
      if (best && mover == BLACK) *best = m;
    }
  }
  // 41 手目に先手に合法手なし（後手の勝ち）
  if (v == 0.0f) {
    Move m = MOVE_NONE;
    ProofResult r = g.dfpn.solve(pos, g.mate41_prob, mover == WHITE, cfg.proof_nodes, &m);
    g.st.proof_calls++;
    g.st.proof_nodes += g.dfpn.nodes();
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
  g.pos.do_move(m);
  g.moves_made++;
  g.st.moves++;
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
      // 根での証明探索: 本将棋の詰み、布石終盤の裁定・先手詰み。手番側の勝ちが証明できればその手を指す
      Move forced = MOVE_NONE;
      float fv = 0.0f;
      if (g.pos.phase() == PHASE_NORMAL && cfg_.mate_nodes_root > 0) {
        Move m = MOVE_NONE;
        if (g.dfpn.solve(g.pos, g.mate_prob, true, cfg_.mate_nodes_root, &m) == PROOF_PROVEN && m != MOVE_NONE) {
          forced = m;
          fv = 1.0f;
          g.st.mate_found++;
        }
      } else if (g.pos.phase() == PHASE_FUSEKI) {
        Move m = MOVE_NONE;
        float v;
        if (fuseki_proof(g, g.pos, cfg_, v, &m) && v > 0 && m != MOVE_NONE) {
          forced = m;
          fv = v;
        }
      }
      if (forced != MOVE_NONE) {
        play_forced(g, forced, fv);
        continue;
      }
      g.nodes.push_back(make_node(g.pos.key()));
      g.sp = g.pos;
      g.path.clear();
      g.path_moves.clear();
      g.leaf = 0;
      g.pending = true;
      return;
    }
    if (!g.root_ready) init_root_search(g, cfg_);
    if (g.sims >= g.budget) {
      finish_move(g);
      continue;
    }
    // 選択
    g.path.clear();
    g.path_moves.clear();
    std::uint32_t ni = 0;
    for (;;) {
      Node& n = g.nodes[ni];
      int ei;
      if (ni == 0) {
        ei = gumbel_pick(g, cfg_);
        if (ei < 0) {
          g.sims = g.budget;  // 逐次半減が終わった
          break;
        }
      } else {
        // PUCT
        float sq_n = std::sqrt(float(std::max(1, n.visits)));
        float best = -1e30f;
        ei = 0;
        for (size_t i = 0; i < n.edges.size(); ++i) {
          const Edge& e = n.edges[i];
          float u = edge_q(n, e) + cfg_.cpuct * e.prior * sq_n / (1 + e.visits);
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
        for (size_t i = 0; i < g.path_moves.size(); ++i) sp.undo_move();
        ni = e.child;
        Node& child = g.nodes[ni];
        if (!child.expanded) {
          // 評価待ち
          g.leaf = ni;
          g.pending = true;
          for (Move m : g.path_moves) g.sp.do_move(m);
          return;
        }
        // 合流済み or 終端 → 値を逆伝播
        float v = child.terminal ? child.terminal_value : (child.visits ? child.wsum / child.visits : child.net_value);
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
        break;
      }
      ni = e.child;
      Node& child = g.nodes[ni];
      if (child.terminal) {
        float v = child.terminal_value;
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
        break;
      }
    }
  }
}

void SelfPlayStats::add(const SelfPlayStats& o) {
  games += o.games;
  moves += o.moves;
  sims += o.sims;
  evals += o.evals;
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

void SelfPlay::parallel_for(int n, const std::function<void(int)>& f) {
  if (threads_ <= 1 || n < 2 * threads_) {
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

int SelfPlay::collect(float* sq, float* glob) {
  int n = int(games_.size());
  parallel_for(n, [&](int i) {
    Game& g = *games_[i];
    if (i >= active_ || g.idle) {
      // 止めている対局: 特徴はゼロのまま（apply で無視する）
      std::fill(sq + size_t(i) * SQ_NB * SQ_FEATS, sq + size_t(i + 1) * SQ_NB * SQ_FEATS, 0.0f);
      std::fill(glob + size_t(i) * GLOB_FEATS, glob + size_t(i + 1) * GLOB_FEATS, 0.0f);
      return;
    }
    if (!g.pending) step_game(g);
    write_features(g.sp, sq + size_t(i) * SQ_NB * SQ_FEATS, glob + size_t(i) * GLOB_FEATS);
  });
  gather();
  return n;
}

void SelfPlay::apply_game(Game& g, const float* logits, const float* wdl) {
  Node& leaf = g.nodes[g.leaf];
  Position& sp = g.sp;
  MoveList ml;
  sp.legal_moves(ml);
  leaf.edges.clear();
  leaf.edges.reserve(ml.n);
  float mx = -1e30f;
  for (Move m : ml) {
    int idx = move_index(sp, m);
    leaf.edges.push_back(Edge{m, idx, logits[idx]});
    mx = std::max(mx, logits[idx]);
  }
  float z = 0;
  for (Edge& e : leaf.edges) {
    e.prior = std::exp(e.prior - mx);
    z += e.prior;
  }
  for (Edge& e : leaf.edges) e.prior = std::max(e.prior / z, 1e-8f);
  leaf.net_value = wdl_value(wdl);
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
  for (size_t i = 0; i < g.path_moves.size(); ++i) sp.undo_move();
  if (g.leaf != 0) g.sims++;  // ルート自身の評価は数えない
  g.st.evals++;
  g.pending = false;
}

void SelfPlay::apply(const float* logits, const float* wdl) {
  int n = std::min(active_, int(games_.size()));
  parallel_for(n, [&](int i) {
    Game& g = *games_[i];
    if (g.idle || !g.pending) return;
    apply_game(g, logits + size_t(i) * POLICY_SIZE, wdl + size_t(i) * 3);
    step_game(g);  // 次の葉まで進める（終局・着手を含む）
  });
  gather();
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
  step_game(g);
  return true;
}

bool SelfPlay::idle(int slot) const { return games_[slot]->idle; }

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

std::vector<GameRecord> SelfPlay::take_finished() {
  std::vector<GameRecord> out;
  out.swap(finished_);
  return out;
}

}  // namespace libra
