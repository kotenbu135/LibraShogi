// SPDX-License-Identifier: Apache-2.0
#include "libra/encoding.h"

#include <cstring>

namespace libra {

namespace {

// 行き先から見た元マスの方向（手番側の座標系。df は筋の添字の差、dr は段の添字の差）。
// 添字 0..7。鏡映は df の符号反転。
const int DIR8[8][2] = {{0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}};
const int KNIGHT_FROM[2][2] = {{-1, 2}, {1, 2}};  // 桂は 2 段後ろ（手番側から見て）から跳ぶ

int dir_index(int df, int dr) {
  for (int i = 0; i < 8; ++i)
    if (DIR8[i][0] == df && DIR8[i][1] == dr) return i;
  return -1;
}

int sgn(int x) { return (x > 0) - (x < 0); }

}  // namespace

int move_index(const Position& pos, Move m) {
  Color us = pos.turn();
  int to = to_mover_frame(us, to_sq(m));
  int cls;
  if (is_drop(m)) {
    cls = 20 + (drop_type(m) - 1);
  } else {
    int from = to_mover_frame(us, from_sq(m));
    int df = file_of(from) - file_of(to), dr = rank_of(from) - rank_of(to);
    if (dr == 2 && (df == -1 || df == 1)) {
      cls = df == -1 ? 8 : 9;
    } else {
      cls = dir_index(sgn(df), sgn(dr));
    }
    if (is_promo(m)) cls += 10;
  }
  return to * POLICY_CLASSES + cls;
}

Move move_from_index(const Position& pos, int idx) {
  if (idx < 0 || idx >= POLICY_SIZE) return MOVE_NONE;
  Color us = pos.turn();
  int to_m = idx / POLICY_CLASSES, cls = idx % POLICY_CLASSES;
  int to = to_mover_frame(us, to_m);
  if (cls >= 20) return make_drop(PieceType(cls - 20 + 1), to);
  bool promo = cls >= 10;
  int d = cls % 10;
  int f = file_of(to_m), r = rank_of(to_m);
  if (d >= 8) {
    f += KNIGHT_FROM[d - 8][0];
    r += KNIGHT_FROM[d - 8][1];
    if (f < 0 || f > 8 || r < 0 || r > 8) return MOVE_NONE;
    return make_move(to_mover_frame(us, make_sq(f, r)), to, promo);
  }
  for (;;) {
    f += DIR8[d][0];
    r += DIR8[d][1];
    if (f < 0 || f > 8 || r < 0 || r > 8) return MOVE_NONE;
    int from = to_mover_frame(us, make_sq(f, r));
    Piece p = pos.piece_on(from);
    if (p != NO_PIECE) return color_of(p) == us ? make_move(from, to, promo) : MOVE_NONE;
  }
}

int mirror_index(int idx) {
  int to_m = idx / POLICY_CLASSES, cls = idx % POLICY_CLASSES;
  int to_mm = mirror_sq(to_m);
  if (cls < 20) {
    int promo = cls / 10, d = cls % 10;
    if (d >= 8) d = d == 8 ? 9 : 8;
    else d = dir_index(-DIR8[d][0], DIR8[d][1]);
    cls = promo * 10 + d;
  }
  return to_mm * POLICY_CLASSES + cls;
}

void write_features(const Position& pos, float* sq_out, float* glob_out, bool mirror) {
  std::memset(sq_out, 0, sizeof(float) * SQ_NB * SQ_FEATS);
  std::memset(glob_out, 0, sizeof(float) * GLOB_FEATS);
  Color us = pos.turn(), them = ~us;
  Bitboard occ = pos.pieces();
  // マスごとの利きの数。マスから attackers_to を 81×2 回引く代わりに、駒ごとに利きを 1 回引いて行き先に数える
  // （駒の利きは左右対称なので、sq に利く駒の数と同じ）
  std::uint8_t attacks[COLOR_NB][SQ_NB] = {};
  for (int c = 0; c < COLOR_NB; ++c) {
    Bitboard src = pos.pieces(Color(c));
    while (src.any()) {
      int from = src.pop();
      Bitboard a = bb::attacks(Color(c), type_of(pos.piece_on(from)), from, occ);
      while (a.any()) ++attacks[c][a.pop()];
    }
  }
  for (int sq = 0; sq < SQ_NB; ++sq) {
    int t = to_mover_frame(us, sq);
    float* f = sq_out + (mirror ? mirror_sq(t) : t) * SQ_FEATS;
    Piece p = pos.piece_on(sq);
    if (p != NO_PIECE) f[(color_of(p) == us ? 0 : 14) + type_of(p) - 1] = 1.0f;
    int a_us = attacks[us][sq];
    int a_them = attacks[them][sq];
    f[28] = a_us > 4 ? 1.0f : a_us / 4.0f;
    f[29] = a_them > 4 ? 1.0f : a_them / 4.0f;
    f[30] = bb::ZoneBB[us].test(sq) ? 1.0f : 0.0f;
    f[31] = bb::PromoZoneBB[us].test(sq) ? 1.0f : 0.0f;
  }
  const float norm[HAND_PT_NB] = {0, 9, 2, 2, 2, 1, 1, 2, 1};
  for (int pt = PAWN; pt <= KING; ++pt) {
    glob_out[pt - 1] = pos.hand(us, PieceType(pt)) / (norm[pt] * 2.0f);
    glob_out[8 + pt - 1] = pos.hand(them, PieceType(pt)) / (norm[pt] * 2.0f);
  }
  bool fuseki = pos.phase() == PHASE_FUSEKI;
  glob_out[16] = fuseki ? 1.0f : 0.0f;
  glob_out[17] = fuseki ? pos.ply() / 40.0f : 1.0f;
  float np = fuseki ? 0.0f : pos.normal_ply() / float(pos.max_ply() > 0 ? pos.max_ply() : 320);
  glob_out[18] = np > 1.0f ? 1.0f : np;
  glob_out[19] = pos.in_check() ? 1.0f : 0.0f;
  glob_out[20] = fuseki ? 0.0f : (pos.repetition_count() - 1) / 3.0f;
  glob_out[21] = pos.declaration_points(us) / 28.0f;
  glob_out[22] = pos.declaration_points(them) / 28.0f;
  glob_out[23] = pos.declaration_pieces(us) / 10.0f;
  glob_out[24] = pos.declaration_pieces(them) / 10.0f;
  glob_out[25] = us == BLACK ? 1.0f : 0.0f;
  glob_out[26] = pos.king_attacked(them) ? 1.0f : 0.0f;
  glob_out[27] = pos.king_attacked(us) ? 1.0f : 0.0f;
  glob_out[28] = pos.mode() == MODE_TENBIN ? 1.0f : 0.0f;
}

}  // namespace libra
