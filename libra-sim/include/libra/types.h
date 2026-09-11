// SPDX-License-Identifier: Apache-2.0
// 基本型。マス・駒・手の表現。ゼロから書いている（既存将棋 AI のコードは参照していない）。
#pragma once
#include <cstdint>
#include <string>

namespace libra {

enum Color : int { BLACK = 0, WHITE = 1, COLOR_NB = 2 };
constexpr Color operator~(Color c) { return Color(c ^ 1); }

// 駒種。9〜14 は成駒（基本種 + 8）。
enum PieceType : int {
  NO_PT = 0, PAWN = 1, LANCE = 2, KNIGHT = 3, SILVER = 4, BISHOP = 5, ROOK = 6, GOLD = 7, KING = 8,
  PRO_PAWN = 9, PRO_LANCE = 10, PRO_KNIGHT = 11, PRO_SILVER = 12, HORSE = 13, DRAGON = 14, PT_NB = 15
};
constexpr int HAND_PT_NB = 9;  // 持ち駒の添字は基本種 1..8（玉は布石中だけ）

// 駒 = 色*16 + 駒種。0 は空。
enum Piece : int { NO_PIECE = 0, PIECE_NB = 32 };
constexpr Piece make_piece(Color c, PieceType pt) { return Piece(c * 16 + pt); }
constexpr Color color_of(Piece p) { return Color(p >> 4); }
constexpr PieceType type_of(Piece p) { return PieceType(p & 15); }
constexpr PieceType base_type(PieceType pt) { return pt >= PRO_PAWN ? PieceType(pt - 8) : pt; }
constexpr PieceType promoted(PieceType pt) { return PieceType(pt + 8); }
constexpr bool is_promotable(PieceType pt) { return pt >= PAWN && pt <= ROOK; }
constexpr bool is_gold_like(PieceType pt) { return pt == GOLD || (pt >= PRO_PAWN && pt <= PRO_SILVER); }

// マス: sq = file*9 + rank。file 0 = 1筋、rank 0 = a段（一段目）。先手は下（i 段側）。
constexpr int SQ_NB = 81;
constexpr int SQ_NONE = 81;
constexpr int file_of(int sq) { return sq / 9; }
constexpr int rank_of(int sq) { return sq % 9; }
constexpr int make_sq(int file, int rank) { return file * 9 + rank; }
constexpr int mirror_sq(int sq) { return (8 - file_of(sq)) * 9 + rank_of(sq); }  // 1↔9 筋の鏡映

// 相対段: 先手は rank そのまま、後手は反転（0 が敵陣最奥）。
constexpr int rel_rank(Color c, int sq) { return c == BLACK ? rank_of(sq) : 8 - rank_of(sq); }

// 手: bit0-6 to、bit7-13 from（打つ手は 0x7f）、bit14 成、bit15-18 打つ駒種。
using Move = std::uint32_t;
constexpr Move MOVE_NONE = 0;
constexpr Move make_move(int from, int to, bool promo) { return Move(to | (from << 7) | (promo ? 1u << 14 : 0)); }
constexpr Move make_drop(PieceType pt, int to) { return Move(to | (0x7f << 7) | (unsigned(pt) << 15)); }
constexpr int to_sq(Move m) { return int(m & 0x7f); }
constexpr int from_sq(Move m) { return int((m >> 7) & 0x7f); }
constexpr bool is_drop(Move m) { return from_sq(m) == 0x7f; }
constexpr bool is_promo(Move m) { return (m >> 14) & 1; }
constexpr PieceType drop_type(Move m) { return PieceType((m >> 15) & 15); }

std::string sq_to_usi(int sq);
int sq_from_usi(const std::string& s);  // 失敗時 SQ_NONE
std::string move_to_usi(Move m);
Move move_from_usi(const std::string& s);  // 失敗時 MOVE_NONE（合法性は見ない）
char pt_to_char(PieceType pt);              // 基本種の大文字
PieceType pt_from_char(char c);             // 大文字小文字どちらも。失敗時 NO_PT

}  // namespace libra
