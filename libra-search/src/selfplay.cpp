// SPDX-License-Identifier: Apache-2.0
#include "libra/selfplay.h"

#include <algorithm>
#include <cmath>
#include <functional>
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
  std::vector<GameRecord> done;  // 終局した記録（gather で集める）
  SelfPlayStats st;              // この対局の統計（gather で集める）

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

void SelfPlay::set_active(int n) { active_ = std::max(1, std::min(n, int(games_.size()))); }

void SelfPlay::start_game(Game& g) {
  g.pos.reset(MODE_TENBIN);
  g.pos.set_max_ply(cfg_.max_ply, cfg_.count_from_41);
  // 玉配置のペア: 一様（後で libra-scale の重点サンプルに置き換える）
  std::uniform_int_distribution<int> d(0, 35);
  int kb = make_sq(d(g.rng) % 9, 5 + d(g.rng) / 9);
  int kw = make_sq(d(g.rng) % 9, d(g.rng) / 9);
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
  g.root_ready = false;
  g.sims = 0;
  if (g.pos.outcome().result != ONGOING || g.moves_made >= cfg_.max_moves_per_game) {
    if (g.pos.outcome().result == ONGOING) g.pos.timeout(g.pos.turn());  // 安全弁
    end_game(g);
  }
}

// 次の葉まで進める。終端は即座に逆伝播し、必要なら着手・終局・新規対局も行う
void SelfPlay::step_game(Game& g) {
  for (int guard = 0; guard < 4096; ++guard) {
    if (g.nodes.empty()) {
      // ルートを作る（未評価）
      g.nodes.push_back(make_node(g.pos.key()));
      g.sp = g.pos;
      g.path.clear();
      g.path_moves.clear();
      g.leaf = 0;
      Outcome o = g.pos.outcome();
      if (o.result != ONGOING) {  // 起こらないはず（着手後に終局を見る）
        end_game(g);
        continue;
      }
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
  for (int i = 0; i < 3; ++i) results[i] += o.results[i];
  ruling41 += o.ruling41;
  no_legal += o.no_legal;
  sennichite += o.sennichite;
  perpetual += o.perpetual;
  max_ply += o.max_ply;
  timeout += o.timeout;
  plies_sum += o.plies_sum;
}

void SelfPlay::parallel_for(int n, const std::function<void(int)>& f) {
  if (threads_ <= 1 || n < 2 * threads_) {
    for (int i = 0; i < n; ++i) f(i);
    return;
  }
  std::vector<std::thread> ts;
  int per = (n + threads_ - 1) / threads_;
  for (int t = 0; t < threads_; ++t) {
    int lo = t * per, hi = std::min(n, lo + per);
    if (lo < hi)
      ts.emplace_back([&, lo, hi] {
        for (int i = lo; i < hi; ++i) f(i);
      });
  }
  for (auto& t : ts) t.join();
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
    if (i >= active_) {
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
    if (!g.pending) return;
    apply_game(g, logits + size_t(i) * POLICY_SIZE, wdl + size_t(i) * 3);
    step_game(g);  // 次の葉まで進める（終局・着手を含む）
  });
  gather();
}

void SelfPlay::root_turns(std::int8_t* out) const {
  for (size_t i = 0; i < games_.size(); ++i) out[i] = games_[i]->pos.turn() == BLACK ? 0 : 1;
}

std::vector<GameRecord> SelfPlay::take_finished() {
  std::vector<GameRecord> out;
  out.swap(finished_);
  return out;
}

}  // namespace libra
