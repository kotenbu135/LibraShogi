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

Bitboard ray_attacks(Dir d, int sq, Bitboard occ);
Bitboard lance_attacks(Color c, int sq, Bitboard occ);
Bitboard bishop_attacks(int sq, Bitboard occ);
Bitboard rook_attacks(int sq, Bitboard occ);
// 駒種ごとの利き（駒の色 c、位置 sq、盤上の占有 occ）
Bitboard attacks(Color c, PieceType pt, int sq, Bitboard occ);

}  // namespace bb
}  // namespace libra
