// SPDX-License-Identifier: Apache-2.0
// perft。--check で既知の値と照合して終了コードで返す。引数なしなら各深さの値と速度を表示する。
//
// 本将棋の平手初期局面の perft 値は公開されている周知の数値（perft(1)=30, (2)=900, (3)=25470,
// (4)=719731, (5)=19861490）。天秤将棋の布石局面の値はこのリポジトリで初めて計算したもので、
// 後続の実装者が同じ値を再現できるようここに固定する（docs/rules.md）。
#include <chrono>
#include <cstdio>
#include <cstring>
#include "libra/position.h"

using namespace libra;

struct Case {
  const char* name;
  const char* position;
  Mode mode;
  int depth;
  std::uint64_t expect;  // 0 なら照合しない
};

static const Case CASES[] = {
    {"startpos d1", "position startpos", MODE_TENBIN, 1, 30},
    {"startpos d2", "position startpos", MODE_TENBIN, 2, 900},
    {"startpos d3", "position startpos", MODE_TENBIN, 3, 25470},
    {"startpos d4", "position startpos", MODE_TENBIN, 4, 719731},
    {"tenbin empty d1", "position fuseki", MODE_TENBIN, 1, 36},
    {"tenbin empty d2", "position fuseki", MODE_TENBIN, 2, 1296},
    {"tenbin empty d3", "position fuseki", MODE_TENBIN, 3, 1296 * 7 * 35},
    {"tenbin K5i K5a d1", "position fuseki moves K*5i K*5a", MODE_TENBIN, 1, 7 * 35},
    {"tenbin K5i K5a d2", "position fuseki moves K*5i K*5a", MODE_TENBIN, 2, 7 * 35 * 7 * 35},
    {"fuseki empty d1", "position fuseki", MODE_FUSEKI, 1, 8 * 36},
    {"fuseki empty d2", "position fuseki", MODE_FUSEKI, 2, 8 * 36 * 8 * 36},
    // 二歩と筋埋め: 7筋に歩を置くと歩は 7 筋以外（8 筋 × 4 マス − ... は d2 で確かめる）
    {"tenbin P7g d1", "position fuseki moves K*5i K*5a P*7g", MODE_TENBIN, 1, 7 * 35},
    {"tenbin P7g P3c d1", "position fuseki moves K*5i K*5a P*7g P*3c", MODE_TENBIN, 1, 6 * 34 + (34 - 3)},
};

int main(int argc, char** argv) {
  bool check = argc > 1 && std::strcmp(argv[1], "--check") == 0;
  int fails = 0;
  for (const Case& c : CASES) {
    Position pos;
    if (!pos.set_position(c.position, c.mode)) {
      std::printf("FAIL %s: bad position\n", c.name);
      ++fails;
      continue;
    }
    auto t0 = std::chrono::steady_clock::now();
    std::uint64_t n = pos.perft(c.depth);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    bool ok = c.expect == 0 || n == c.expect;
    std::printf("%s %-22s depth %d = %llu (%.3fs, %.1f Mnps)\n", ok ? "ok  " : "FAIL", c.name, c.depth,
                (unsigned long long)n, sec, sec > 0 ? n / sec / 1e6 : 0.0);
    if (!ok) ++fails;
  }
  if (!check) {
    Position pos;
    pos.set_position("position startpos");
    auto t0 = std::chrono::steady_clock::now();
    std::uint64_t n = pos.perft(5);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("startpos depth 5 = %llu (%.2fs, %.1f Mnps) expect 19861490\n", (unsigned long long)n, sec, n / sec / 1e6);
  }
  return fails ? 1 : 0;
}
