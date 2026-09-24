// エンジン（libra-engine）の探索ループの CPU 側だけを測る。ネットの評価は即座に返す偽物。
// 目的: 「GPU の推論が無限に速くなったら、この探索は 1 秒あたり何ノード読めるか」の上限を知る。
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "libra/selfplay.h"
#include "libra/encoding.h"
#include "libra/position.h"

using namespace libra;
using Clock = std::chrono::steady_clock;

static double now_s() {
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// 布石 40 手をランダムな合法手で進めた局面（41 手目＝本将棋の最初の手番）の position 行を作る
static std::string make_line(int plies, std::uint64_t seed) {
  Position pos;
  std::mt19937_64 rng(seed);
  std::string line = "position fuseki moves";
  for (int i = 0; i < plies; ++i) {
    MoveList ml;
    pos.legal_moves(ml);
    if (ml.n == 0) break;
    Move m = ml.m[rng() % ml.n];
    line += " " + move_to_usi(m);
    pos.do_move(m);
    if (pos.outcome().result != ONGOING) break;
  }
  return line;
}

int main(int argc, char** argv) {
  int threads = argc > 1 ? std::atoi(argv[1]) : 4;
  int batch = argc > 2 ? std::atoi(argv[2]) : 64;
  int sims = argc > 3 ? std::atoi(argv[3]) : 1600;
  int plies = argc > 4 ? std::atoi(argv[4]) : 40;   // 40 なら本将棋の 41 手目、2 なら布石
  int mate = argc > 5 ? std::atoi(argv[5]) : 2000;  // Mate_Nodes（エンジンの既定 2000）
  int reps = argc > 6 ? std::atoi(argv[6]) : 5;

  SearchConfig cfg;
  cfg.external = true;
  cfg.mate_nodes_root = mate;
  cfg.policy_topk = 300;
  SelfPlay eng(cfg, 1, 12345, threads);

  std::vector<float> sq(size_t(batch) * SQ_NB * SQ_FEATS, 0.f), glob(size_t(batch) * GLOB_FEATS, 0.f);
  std::vector<float> logits(size_t(batch) * POLICY_SIZE, 0.f), wdl(size_t(batch) * 3, 0.f);
  // 偽のネット: 方策は一様（ロジット 0）、wdl は (0.34, 0.32, 0.34)
  for (int i = 0; i < batch; ++i) { wdl[i * 3] = 0.34f; wdl[i * 3 + 1] = 0.32f; wdl[i * 3 + 2] = 0.34f; }

  double total_t = 0; long long total_leaves = 0, total_calls = 0; long long total_sims = 0;
  for (int r = 0; r < reps; ++r) {
    std::string line = make_line(plies, 100 + r);
    if (!eng.set_position(0, line, sims, true, MODE_TENBIN)) { std::fprintf(stderr, "bad position\n"); return 1; }
    double t0 = now_s();
    long long leaves = 0, calls = 0;
    while (!eng.idle(0)) {
      int k = eng.collect_batch(0, batch, sq.data(), glob.data());
      if (k == 0) break;
      eng.apply_batch(0, logits.data(), wdl.data(), k);
      leaves += k; ++calls;
    }
    double dt = now_s() - t0;
    const SearchResult& res = eng.result(0);
    total_t += dt; total_leaves += leaves; total_calls += calls; total_sims += (long long)res.sims;
    std::printf("  rep %d: %.3f s  leaves %lld  calls %lld  sims %llu\n", r, dt, leaves, calls, (unsigned long long)res.sims);
  }
  std::printf("threads %d batch %d sims %d plies %d mate %d -> %.0f leaves/s  %.0f sims/s  avg batch %.1f  %.3f ms/batch\n",
              threads, batch, sims, plies, mate,
              total_leaves / total_t, total_sims / total_t,
              double(total_leaves) / std::max(1LL, total_calls), total_t / std::max(1LL, total_calls) * 1000.0);
  return 0;
}
