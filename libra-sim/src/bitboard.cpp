// SPDX-License-Identifier: Apache-2.0
#include "libra/bitboard.h"

namespace libra {
namespace bb {

Bitboard FileBB[9];
Bitboard RankBB[9];
Bitboard ZoneBB[COLOR_NB];
Bitboard PromoZoneBB[COLOR_NB];
Bitboard Ray[DIR_NB][SQ_NB];
Bitboard StepAttacks[COLOR_NB][PT_NB][SQ_NB];
Bitboard OrthoStep[SQ_NB];
Bitboard DiagStep[SQ_NB];

namespace {

bool initialized = false;

// (df, dr) だけ進んだマス。盤外なら SQ_NONE。
int step(int sq, int df, int dr) {
  int f = file_of(sq) + df, r = rank_of(sq) + dr;
  if (f < 0 || f > 8 || r < 0 || r > 8) return SQ_NONE;
  return make_sq(f, r);
}

// 方向 d の (df, dr)
void dir_delta(Dir d, int& df, int& dr) {
  switch (d) {
    case DIR_S:  df = 0;  dr = +1; break;
    case DIR_W:  df = +1; dr = 0;  break;
    case DIR_NW: df = +1; dr = -1; break;
    case DIR_SW: df = +1; dr = +1; break;
    case DIR_N:  df = 0;  dr = -1; break;
    case DIR_E:  df = -1; dr = 0;  break;
    case DIR_NE: df = -1; dr = -1; break;
    case DIR_SE: df = -1; dr = +1; break;
    default: df = dr = 0;
  }
}

// 先手視点の 1 歩の利き。後手は dr の符号を反転する。
struct Step { int df, dr; };
const Step PAWN_STEPS[] = {{0, -1}};
const Step KNIGHT_STEPS[] = {{-1, -2}, {+1, -2}};
const Step SILVER_STEPS[] = {{-1, -1}, {0, -1}, {+1, -1}, {-1, +1}, {+1, +1}};
const Step GOLD_STEPS[] = {{-1, -1}, {0, -1}, {+1, -1}, {-1, 0}, {+1, 0}, {0, +1}};
const Step KING_STEPS[] = {{-1, -1}, {0, -1}, {+1, -1}, {-1, 0}, {+1, 0}, {-1, +1}, {0, +1}, {+1, +1}};

template <size_t N>
Bitboard steps_bb(Color c, int sq, const Step (&steps)[N]) {
  Bitboard r;
  for (const Step& s : steps) {
    int t = step(sq, s.df, c == BLACK ? s.dr : -s.dr);
    if (t != SQ_NONE) r.set(t);
  }
  return r;
}

}  // namespace

void init() {
  if (initialized) return;
  initialized = true;
  for (int s = 0; s < SQ_NB; ++s) {
    FileBB[file_of(s)].set(s);
    RankBB[rank_of(s)].set(s);
  }
  for (int r = 5; r <= 8; ++r) ZoneBB[BLACK] |= RankBB[r];
  for (int r = 0; r <= 3; ++r) ZoneBB[WHITE] |= RankBB[r];
  for (int r = 0; r <= 2; ++r) PromoZoneBB[BLACK] |= RankBB[r];
  for (int r = 6; r <= 8; ++r) PromoZoneBB[WHITE] |= RankBB[r];
  for (int d = 0; d < DIR_NB; ++d) {
    int df, dr;
    dir_delta(Dir(d), df, dr);
    for (int s = 0; s < SQ_NB; ++s) {
      Bitboard r;
      for (int t = step(s, df, dr); t != SQ_NONE; t = step(t, df, dr)) r.set(t);
      Ray[d][s] = r;
    }
  }
  for (int c = 0; c < COLOR_NB; ++c)
    for (int s = 0; s < SQ_NB; ++s) {
      Color col = Color(c);
      StepAttacks[c][PAWN][s] = steps_bb(col, s, PAWN_STEPS);
      StepAttacks[c][KNIGHT][s] = steps_bb(col, s, KNIGHT_STEPS);
      StepAttacks[c][SILVER][s] = steps_bb(col, s, SILVER_STEPS);
      Bitboard gold = steps_bb(col, s, GOLD_STEPS);
      StepAttacks[c][GOLD][s] = gold;
      StepAttacks[c][PRO_PAWN][s] = gold;
      StepAttacks[c][PRO_LANCE][s] = gold;
      StepAttacks[c][PRO_KNIGHT][s] = gold;
      StepAttacks[c][PRO_SILVER][s] = gold;
      StepAttacks[c][KING][s] = steps_bb(col, s, KING_STEPS);
    }
  for (int s = 0; s < SQ_NB; ++s) {
    const Step ortho[] = {{0, -1}, {-1, 0}, {+1, 0}, {0, +1}};
    const Step diag[] = {{-1, -1}, {+1, -1}, {-1, +1}, {+1, +1}};
    OrthoStep[s] = steps_bb(BLACK, s, ortho);
    DiagStep[s] = steps_bb(BLACK, s, diag);
  }
}

Bitboard ray_attacks(Dir d, int sq, Bitboard occ) {
  Bitboard ray = Ray[d][sq];
  Bitboard blockers = ray & occ;
  if (blockers.none()) return ray;
  int first = d < 4 ? blockers.lsb() : blockers.msb();
  return ray ^ Ray[d][first];
}

Bitboard lance_attacks(Color c, int sq, Bitboard occ) { return ray_attacks(c == BLACK ? DIR_N : DIR_S, sq, occ); }

Bitboard bishop_attacks(int sq, Bitboard occ) {
  return ray_attacks(DIR_NW, sq, occ) | ray_attacks(DIR_SW, sq, occ) | ray_attacks(DIR_NE, sq, occ) | ray_attacks(DIR_SE, sq, occ);
}

Bitboard rook_attacks(int sq, Bitboard occ) {
  return ray_attacks(DIR_S, sq, occ) | ray_attacks(DIR_W, sq, occ) | ray_attacks(DIR_N, sq, occ) | ray_attacks(DIR_E, sq, occ);
}

Bitboard attacks(Color c, PieceType pt, int sq, Bitboard occ) {
  switch (pt) {
    case LANCE: return lance_attacks(c, sq, occ);
    case BISHOP: return bishop_attacks(sq, occ);
    case ROOK: return rook_attacks(sq, occ);
    case HORSE: return bishop_attacks(sq, occ) | OrthoStep[sq];
    case DRAGON: return rook_attacks(sq, occ) | DiagStep[sq];
    default: return StepAttacks[c][pt][sq];
  }
}

}  // namespace bb
}  // namespace libra
