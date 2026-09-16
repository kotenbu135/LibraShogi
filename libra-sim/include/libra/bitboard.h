// SPDX-License-Identifier: Apache-2.0
// 81 マスのビットボードと利きの表。
#pragma once
#include <cstddef>
#include <cstdint>
#include "libra/types.h"

namespace libra {

struct Bitboard {
  std::uint64_t b[2];  // b[0]: sq 0..62、b[1]: sq 63..80

  constexpr Bitboard() : b{0, 0} {}
  constexpr Bitboard(std::uint64_t lo, std::uint64_t hi) : b{lo, hi} {}
  static Bitboard sq(int s) { return s < 63 ? Bitboard(1ULL << s, 0) : Bitboard(0, 1ULL << (s - 63)); }

  constexpr bool any() const { return (b[0] | b[1]) != 0; }
  constexpr bool none() const { return !any(); }
  bool test(int s) const { return s < 63 ? (b[0] >> s) & 1 : (b[1] >> (s - 63)) & 1; }
  void set(int s) { if (s < 63) b[0] |= 1ULL << s; else b[1] |= 1ULL << (s - 63); }
  void clear(int s) { if (s < 63) b[0] &= ~(1ULL << s); else b[1] &= ~(1ULL << (s - 63)); }
  int count() const { return __builtin_popcountll(b[0]) + __builtin_popcountll(b[1]); }
  int lsb() const { return b[0] ? __builtin_ctzll(b[0]) : 63 + __builtin_ctzll(b[1]); }
  int msb() const { return b[1] ? 63 + 63 - __builtin_clzll(b[1]) : 63 - __builtin_clzll(b[0]); }
  int pop() { int s = lsb(); if (b[0]) b[0] &= b[0] - 1; else b[1] &= b[1] - 1; return s; }

  constexpr Bitboard operator&(Bitboard o) const { return {b[0] & o.b[0], b[1] & o.b[1]}; }
  constexpr Bitboard operator|(Bitboard o) const { return {b[0] | o.b[0], b[1] | o.b[1]}; }
  constexpr Bitboard operator^(Bitboard o) const { return {b[0] ^ o.b[0], b[1] ^ o.b[1]}; }
  constexpr Bitboard operator~() const { return {~b[0] & 0x7fffffffffffffffULL, ~b[1] & 0x3ffffULL}; }
  Bitboard& operator&=(Bitboard o) { b[0] &= o.b[0]; b[1] &= o.b[1]; return *this; }
  Bitboard& operator|=(Bitboard o) { b[0] |= o.b[0]; b[1] |= o.b[1]; return *this; }
  Bitboard& operator^=(Bitboard o) { b[0] ^= o.b[0]; b[1] ^= o.b[1]; return *this; }
  constexpr bool operator==(Bitboard o) const { return b[0] == o.b[0] && b[1] == o.b[1]; }
  constexpr bool operator!=(Bitboard o) const { return !(*this == o); }
};

constexpr Bitboard ALL_BB(0x7fffffffffffffffULL, 0x3ffffULL);

namespace bb {

// 方向。添字の順は Ray[] と一致させる。差分が正の 4 方向は lsb が最初の遮り駒、負の 4 方向は msb。
enum Dir : int { DIR_S = 0, DIR_W = 1, DIR_NW = 2, DIR_SW = 3, DIR_N = 4, DIR_E = 5, DIR_NE = 6, DIR_SE = 7, DIR_NB = 8 };
constexpr int DIR_DELTA[DIR_NB] = {+1, +9, +8, +10, -1, -9, -10, -8};

void init();  // 一度だけ呼ぶ（Position のコンストラクタで保証する）

extern Bitboard FileBB[9];
extern Bitboard RankBB[9];
extern Bitboard ZoneBB[COLOR_NB];      // 自陣四段（布石で打てる範囲）
extern Bitboard PromoZoneBB[COLOR_NB]; // 敵陣三段（成れる・宣言の範囲）
extern Bitboard Ray[DIR_NB][SQ_NB];
extern Bitboard StepAttacks[COLOR_NB][PT_NB][SQ_NB];  // 歩・桂・銀・金類・玉（飛角香と成りの滑り部分は含まない）
extern Bitboard OrthoStep[SQ_NB];  // 馬の追加利き（縦横 1 マス）
extern Bitboard DiagStep[SQ_NB];   // 竜の追加利き（斜め 1 マス）

// 利きの計算は合法手生成・詰み探索・特徴量の内側で呼ばれるので、ヘッダに置いて呼び出し側で展開できるようにする
inline Bitboard ray_attacks(Dir d, int sq, Bitboard occ) {
  Bitboard ray = Ray[d][sq];
  Bitboard blockers = ray & occ;
  if (blockers.none()) return ray;
  int first = d < 4 ? blockers.lsb() : blockers.msb();
  return ray ^ Ray[d][first];
}

inline Bitboard lance_attacks(Color c, int sq, Bitboard occ) { return ray_attacks(c == BLACK ? DIR_N : DIR_S, sq, occ); }

inline Bitboard bishop_attacks(int sq, Bitboard occ) {
  return ray_attacks(DIR_NW, sq, occ) | ray_attacks(DIR_SW, sq, occ) | ray_attacks(DIR_NE, sq, occ) | ray_attacks(DIR_SE, sq, occ);
}

inline Bitboard rook_attacks(int sq, Bitboard occ) {
  return ray_attacks(DIR_S, sq, occ) | ray_attacks(DIR_W, sq, occ) | ray_attacks(DIR_N, sq, occ) | ray_attacks(DIR_E, sq, occ);
}

// 駒種ごとの利き（駒の色 c、位置 sq、盤上の占有 occ）
inline Bitboard attacks(Color c, PieceType pt, int sq, Bitboard occ) {
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
