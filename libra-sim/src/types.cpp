// SPDX-License-Identifier: Apache-2.0
#include "libra/types.h"

namespace libra {

std::string sq_to_usi(int sq) {
  std::string s;
  s += char('1' + file_of(sq));
  s += char('a' + rank_of(sq));
  return s;
}

int sq_from_usi(const std::string& s) {
  if (s.size() < 2) return SQ_NONE;
  int f = s[0] - '1', r = s[1] - 'a';
  if (f < 0 || f > 8 || r < 0 || r > 8) return SQ_NONE;
  return make_sq(f, r);
}

char pt_to_char(PieceType pt) {
  static const char* tbl = " PLNSBRGK";
  return (pt >= PAWN && pt <= KING) ? tbl[pt] : '?';
}

PieceType pt_from_char(char c) {
  switch (c) {
    case 'P': case 'p': return PAWN;
    case 'L': case 'l': return LANCE;
    case 'N': case 'n': return KNIGHT;
    case 'S': case 's': return SILVER;
    case 'B': case 'b': return BISHOP;
    case 'R': case 'r': return ROOK;
    case 'G': case 'g': return GOLD;
    case 'K': case 'k': return KING;
    default: return NO_PT;
  }
}

std::string move_to_usi(Move m) {
  if (m == MOVE_NONE) return "none";
  if (is_drop(m)) return std::string(1, pt_to_char(drop_type(m))) + "*" + sq_to_usi(to_sq(m));
  return sq_to_usi(from_sq(m)) + sq_to_usi(to_sq(m)) + (is_promo(m) ? "+" : "");
}

Move move_from_usi(const std::string& s) {
  if (s.size() >= 4 && s[1] == '*') {
    PieceType pt = pt_from_char(s[0]);
    int to = sq_from_usi(s.substr(2, 2));
    if (pt == NO_PT || to == SQ_NONE || s.size() != 4) return MOVE_NONE;
    return make_drop(pt, to);
  }
  if (s.size() == 4 || (s.size() == 5 && s[4] == '+')) {
    int from = sq_from_usi(s.substr(0, 2)), to = sq_from_usi(s.substr(2, 2));
    if (from == SQ_NONE || to == SQ_NONE) return MOVE_NONE;
    return make_move(from, to, s.size() == 5);
  }
  return MOVE_NONE;
}

}  // namespace libra
