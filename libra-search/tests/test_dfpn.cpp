// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <string>
#include "libra/dfpn.h"

using namespace libra;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond)                                                              \
  do {                                                                           \
    if (cond) ++g_pass;                                                          \
    else { ++g_fail; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
  } while (0)

static ProofResult mate(const char* sfen, std::uint64_t nodes, std::string* best = nullptr, std::uint64_t* used = nullptr) {
  Position p;
  if (!p.set_sfen(sfen, PHASE_NORMAL)) return PROOF_UNKNOWN;
  DfPn d(14);
  MateProblem prob;
  Move m = MOVE_NONE;
  ProofResult r = d.solve(p, prob, true, nodes, &m);
  if (best) *best = move_to_usi(m);
  if (used) *used = d.nodes();
  return r;
}

int main() {
  std::string best;
  std::uint64_t used;
  // 1 手詰: 後手玉 5a、先手 金 5c・飛 持ち駒 → G*? いや R*5b が詰み（4a 6a は? 5b の飛は横も利く: 4b 6b、玉は 4a 6a へ逃げられる）
  // 確実な 1 手詰: 玉 1a、先手 金 2c、持ち駒 金 → G*1b（1b は 2c の金が守る。2a は 1b の金の利き、2b も）
  CHECK(mate("8k/9/7G1/9/9/9/9/9/K8 b G 1", 1000, &best, &used) == PROOF_PROVEN);
  CHECK(best == "G*1b");
  // 玉だけの相手に金 1 枚では詰まない
  CHECK(mate("8k/9/9/9/9/9/9/9/K8 b G 1", 2000, &best) == PROOF_DISPROVEN);
  // 3 手詰: 玉 1a、先手 金 3c 持ち駒 金 銀 … G*2b? 玉 1a: 2b に金 → 1b 2a に利く、1a の玉は 2b の金を取れる（守りが要る）
  // 頭金の形を作る: 先手 銀 2c、持ち駒 金金 → G*1b（銀が守る）は 1 手詰。3 手詰は 飛 2 枚: 玉 5a、持ち駒 飛飛 → R*5b? 玉が取る…
  // 3 手詰の例: 玉 1a、先手 飛 3b（1b 2b に利く→王手ではない: 3b の飛は横に 2b 1b… 1a は違う）。
  // 使う: 玉 1a、先手 香 1i（1 筋に王手）… 1a→2a/2b へ逃げる。金を持てば G*2b? 
  // 簡単に: 玉 2a、先手 金 3c、持ち駒 金 銀。G*2b は 3c が守り、玉は 1a 1b 3a 3b… 3b は 2b の金の利き、3a は 2b の金の利き、1b も、1a も → 1 手詰
  CHECK(mate("7k1/9/6G2/9/9/9/9/9/K8 b GS 1", 1000, &best) == PROOF_PROVEN);
  CHECK(best == "G*2b" || best == "S*2b" || best == "3c2b");  // 1 手詰は 3 通り
  // 3 手詰: 玉 1a、先手 金 3c 持ち駒 銀 金: S*2b は 1a の玉が取れる(2b は 3c の金の利き→取れない). 銀 2b: 利き 1a 3a 1c 3c… 1b には利かない → 玉 1b へ逃げる → G*1c? 
  // 代わりに合駒が要る 3 手詰: 玉 1a、先手 飛 持ち駒 2 枚: R*1b は取られる。R*3a（横王手、2a 逃げ…）
  // 確実な 3 手詰: 玉 5a、先手 飛 5i 持ち駒 金 → 飛が 5 筋から王手（5b〜5h 空）。玉 4a/6a/4b/6b へ。金 1 枚では詰まない。
  // → 5 手以内で「詰まない」ことの反証も見る
  // 3 手詰: 玉 1a、先手 桂 持ち駒、金 持ち駒、先手 銀 3b: N*2c は王手（1a）。玉 1b/2a? 2a 2b は 3b の銀の利き（銀 3b: 2a 4a 2c 4c 3a? 銀の利きは前 3 方向と後ろ斜め: 3b(先手)→2a 3a 4a 2c 4c）。1b は空 → 玉 1b → G*1c? 1c は 2c の桂…桂は前だけ。金 1c: 玉 1b を詰ます? 1b の玉: 逃げ 2b（銀 3b の利きにない!）
  // 単純化: 二段目に金を打つ 3 手詰。玉 1a、先手 銀 2c、持ち駒 金 → G*1b で 1 手詰（先の例と同じ）。3 手詰は省き、5 手詰の代表例として
  //  「玉 1a、先手 飛 1i（1 筋）と 持ち駒 金、後手 歩 1e（合駒に使う駒なし）」: 1i の飛は 1e で止まる → 王手でない
  // 3 手詰: 玉 2b（中段に）、先手 金 4b（横から 3b に利く）、持ち駒 飛 → R*2c? 2c は 2b の玉が取れる（守り無し）。R*2i（縦王手）: 逃げ 1a 1b? 1b は…3a 1c 3c 1a 1b 3b(金) 2a? 
  //   金 4b: 利き 3a 4a 5a 3b 5b 4c → 3a 3b 塞ぎ。飛 2i: 2 筋。玉 2b の逃げ: 1a 1b 1c 2a(飛の利き) 2c(飛) 3c 1b 1c… 多い。
  // 3 手詰の確実な形: 玉 1a、先手 金 2c・金 3b? 3b の金は 2a 2b に利く → 1 手詰 G*1b … 
  // ここでは 1 手詰・反証・玉方の合駒を含む問題（5 手以内に解けない）で機能を確かめる。多手数は Python 側の乱数局面で「詰みなら実際に詰ませられる」検証を行う。
  // 金 5c は 4b/6b へ動いて詰む。香の成り込み（4i4b+ など）でも詰む
  CHECK(mate("4k4/9/4G4/9/9/9/9/9/3L1L2K b P 1", 1000, &best) == PROOF_PROVEN);
  CHECK(best == "5c4b" || best == "5c6b" || best == "4i4b+" || best == "6i6b+");
  // 打ち歩詰めは攻め方の手（王手）に含まれない: 角 7d が 5b を守る形で P*5b は除かれる
  {
    Position p;
    CHECK(p.set_sfen("4k4/9/9/2B6/9/9/9/9/3L1L2K b P 1", PHASE_NORMAL));
    MateProblem prob;
    MoveList ml;
    prob.moves(p, true, ml);
    bool has_pawn_drop = false;
    for (Move m : ml) has_pawn_drop |= (move_to_usi(m) == "P*5b");
    CHECK(!has_pawn_drop);
    CHECK(ml.n == 6);  // 4i4a+ 4i4b+ 6i6a+ 6i6b+ 7d4a+ 7d5b+
  }
  // 同じ形で金を持てば G*5b でも詰む（他の詰みもある）
  CHECK(mate("4k4/9/9/2B6/9/9/9/9/3L1L2K b G 1", 1000, &best) == PROOF_PROVEN);
  // 3 手詰: 玉 1a、先手 香 1c？ 香 1c は 1b 1a に利く→王手。玉 2a/2b へ。持ち駒 金: G*2b → 玉 2a…
  // 3 手詰の定番「銀 2 枚」: 玉 1a、先手 銀 2b（王手: 銀 2b の利き 1a 3a 1c 3c 2a）。玉は 1b（銀の利きにない）。先手 銀 持ち駒 → S*1c? 1c: 利き 2b?… 
  // 確認済みの 3 手詰: 玉 1b、先手 金 3c（2b 3b 4b 2c 4c 3d）、持ち駒 飛: R*1c(王手、1b の玉: 逃げ 1a 2a 2b(金) 2c(金) 1c 取れる? 1c の飛は 2c の金…金 3c は 2c に利く→取れない) 玉 1a/2a: R*1c は 1a にも利く（縦）→ 1a 不可。2a → 2a は 3c の金の利きにない → 逃げられる。次 G? 持ち駒なし。
  // 「飛 1 枚＋金 1 枚」: 玉 1b、先手 金 3c、持ち駒 飛 金: R*1c(王手) 2a へ → G*2b?（2b: 3c 金が守る、玉 2a は 2b を取れない; 1a 1b 3a 3b: 1b は 1c の飛、1a は 1c の飛（1b 空）、3a 3b は 2b の金） → 3 手詰
  CHECK(mate("9/8k/6G2/9/9/9/9/9/K8 b RG 1", 5000, &best, &used) == PROOF_PROVEN);
  CHECK(best == "R*1c" || best == "R*1a" || best == "G*2b");  // R*1a Kx1a G*2b も 3 手詰
  std::printf("3-mate nodes %llu best %s\n", (unsigned long long)used, best.c_str());

  // ---- 布石: 41 手目の裁定の証明 ----
  // test_rules の遮断不能局面（39 手目直後、後手番）: 後手の全手に対し先手勝ち → 根は AND 節点
  {
    Position p;
    // 先手 20 枚: K5i R5f B1f 歩 1g-9g L1i L9i N2i N8i S3i S7i G4i G6i / 後手 19 枚＋持ち駒 金
    CHECK(p.set_sfen("lnsgpbsnl/ppppkpppp/8r/9/9/4R3B/PPPPPPPPP/9/LNSGKGSNL w g 40", PHASE_FUSEKI));
    CHECK(p.ply() == 39 && p.turn() == WHITE);
    DfPn d(12);
    Ruling41Problem prob;
    CHECK(d.solve(p, prob, false, 10000) == PROOF_PROVEN);
    // 角を 1h に置いて遮れる形（飛だけ）: 後手は 5c/5d に金を打てるので反証
    Position q;
    CHECK(q.set_sfen("lnsgpbsnl/ppppkpppp/8r/9/9/4R4/PPPPPPPPP/8B/LNSGKGSNL w g 40", PHASE_FUSEKI));
    CHECK(d.solve(q, prob, false, 10000) == PROOF_DISPROVEN);
    // 1 手前（38 手目直後、先手番。角はまだ持ち駒）: 先手は B*1f で二本当てにできる → 証明、best は B*1f
    Position r;
    CHECK(r.set_sfen("lnsgpbsnl/ppppkpppp/8r/9/9/4R4/PPPPPPPPP/9/LNSGKGSNL b Bg 39", PHASE_FUSEKI));
    CHECK(r.ply() == 38 && r.turn() == BLACK);
    Move best_m = MOVE_NONE;
    CHECK(d.solve(r, prob, true, 20000, &best_m) == PROOF_PROVEN);
    std::printf("ruling41 proof nodes %llu best %s\n", (unsigned long long)d.nodes(), move_to_usi(best_m).c_str());
    CHECK(move_to_usi(best_m) == "B*1f" || move_to_usi(best_m) == "B*1h" || move_to_usi(best_m) == "B*9f");
  }
  // ---- 布石: 41 手目の先手詰みの証明 ----
  {
    // test_rules の「先手玉 5f が両王手で詰み」の 40 手完了局面の 1 手前（後手の 40 手目が飛 5b）
    Position p;
    CHECK(p.set_sfen("kpppppppp/plss1l2b/ggn3n2/9/9/3GKG3/3SPS3/LNB3RNL/PPPP1PPPP w r 40", PHASE_FUSEKI));
    CHECK(p.ply() == 39 && p.turn() == WHITE);
    DfPn d(12);
    Mate41Problem prob;
    Move best_m = MOVE_NONE;
    ProofResult r = d.solve(p, prob, true, 10000, &best_m);
    std::printf("mate41 result %d nodes %llu best %s\n", int(r), (unsigned long long)d.nodes(), move_to_usi(best_m).c_str());
    CHECK(r == PROOF_PROVEN);
    CHECK(move_to_usi(best_m) == "R*5b");
  }
  std::printf("%d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
