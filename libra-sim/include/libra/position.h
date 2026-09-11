// SPDX-License-Identifier: Apache-2.0
// 天秤将棋の局面。ルールの唯一の正は docs/rules.md。
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "libra/bitboard.h"
#include "libra/types.h"

namespace libra {

enum Mode : int { MODE_TENBIN = 0, MODE_FUSEKI = 1 };
enum Phase : int { PHASE_FUSEKI = 0, PHASE_NORMAL = 1 };
enum Result : int { ONGOING = 0, BLACK_WIN = 1, WHITE_WIN = 2, DRAW = 3 };
enum Reason : int {
  R_NONE = 0,
  R_NO_LEGAL_MOVE,        // 詰み・合法手なし（41 手目の先手を含む）
  R_RULING41,             // 40 手完了時に後手玉が当たっている → 先手勝ち
  R_SENNICHITE,           // 4 回目の同一局面
  R_PERPETUAL_CHECK,      // 連続王手の千日手（王手側の負け）
  R_DECLARATION,          // 正当な入玉宣言
  R_ILLEGAL_DECLARATION,  // 不当な宣言（宣言側の負け）
  R_ILLEGAL_MOVE,         // 非合法手（ハーネスが設定）
  R_MAX_PLY,              // 手数上限
  R_RESIGN,
  R_TIMEOUT,
};
struct Outcome {
  Result result = ONGOING;
  Reason reason = R_NONE;
};
const char* result_name(Result r);
const char* reason_name(Reason r);

struct MoveList {
  static constexpr int MAX = 640;
  Move m[MAX];
  int n = 0;
  void add(Move x) { m[n++] = x; }
  const Move* begin() const { return m; }
  const Move* end() const { return m + n; }
};

class Position {
 public:
  Position();
  void reset(Mode mode);
  // SFEN を読む。phase を明示する（布石の拡張 SFEN は持ち駒に玉を含む）。失敗時 false（局面は不定）。
  bool set_sfen(const std::string& sfen, Phase phase);
  // "position fuseki [moves ...]" / "position sfen <sfen> [moves ...]" / "position startpos [moves ...]"。
  // choose:* は読み飛ばす。非合法な手があれば false。
  bool set_position(const std::string& line, Mode mode = MODE_TENBIN);
  std::string sfen() const;
  std::string pretty() const;

  Mode mode() const { return mode_; }
  Phase phase() const { return phase_; }
  Color turn() const { return turn_; }
  int ply() const { return ply_; }                       // 通算で指された手の数（布石 0..40、以降も加算）。SFEN の手数 − 1
  int normal_ply() const { return ply_ - base_ply_; }    // 本将棋で指された手の数（天秤将棋なら ply − 40）
  Piece piece_on(int sq) const { return board_[sq]; }
  int hand(Color c, PieceType pt) const { return hand_[c][pt]; }
  Bitboard pieces() const { return occ_; }
  Bitboard pieces(Color c) const { return by_color_[c]; }
  Bitboard pieces(Color c, PieceType pt) const { return by_color_[c] & by_type_[pt]; }
  Bitboard pieces(PieceType pt) const { return by_type_[pt]; }
  int king_sq(Color c) const;  // 盤上に無ければ SQ_NONE

  void legal_moves(MoveList& out);
  bool is_legal(Move m);
  void do_move(Move m);
  void undo_move();
  Move last_move() const { return st_.empty() ? MOVE_NONE : st_.back().move; }

  bool in_check() const;                      // 手番の玉に王手（布石中は常に false）
  bool king_attacked(Color c) const;          // c の玉が相手の利きに当たっている（布石の判定にも使う）
  Bitboard attackers_to(int sq, Color by, Bitboard occ) const;
  bool ruling41_pending();                    // 39 手目直後で遮断不能（40 手目の制限が外れている）

  Outcome outcome();                          // 現在の裁定。合法手の有無も見る
  bool can_declare(Color c) const;
  int declaration_points(Color c) const;
  int declaration_pieces(Color c) const;
  void declare(Color c);                      // 宣言（正当なら c の勝ち、不当なら c の負け）
  void resign(Color c);
  void timeout(Color c);
  void illegal_move(Color c);
  int repetition_count() const;

  std::uint64_t key() const { return key_; }
  std::uint64_t mirror_key() const { return mkey_; }
  std::uint64_t norm_key() const { return key_ < mkey_ ? key_ : mkey_; }

  void set_max_ply(int n, bool count_from_41) { max_ply_ = n; count_from_41_ = count_from_41; }
  int max_ply() const { return max_ply_; }
  bool count_from_41() const { return count_from_41_; }

  std::uint64_t perft(int depth);

 private:
  struct StateInfo {
    Move move;
    Piece captured;
    std::uint64_t key, mkey;
    Phase phase;
    Outcome outcome;
    bool mate_checked;
    std::size_t hist_size;
  };
  struct HistEntry {
    std::uint64_t key;
    bool in_check;
    Color stm;
  };

  void clear();
  void put_piece(Piece p, int sq);
  void remove_piece(int sq);
  void add_hand(Color c, PieceType pt, int d);
  void set_turn(Color c);
  void fuseki_moves(MoveList& out);
  void normal_moves(MoveList& out, bool check_uchifuzume);
  bool has_legal_move(bool check_uchifuzume);
  bool attacked_after(int ksq, Color by, Bitboard occ2, Bitboard exclude) const;
  void after_move();
  Outcome repetition_outcome() const;
  bool fuseki_done() const;

  Piece board_[SQ_NB];
  Bitboard by_color_[COLOR_NB];
  Bitboard by_type_[PT_NB];
  Bitboard occ_;
  int hand_[COLOR_NB][HAND_PT_NB];
  Color turn_;
  int ply_;
  int base_ply_;  // 本将棋が始まった時点の通算手数（天秤将棋 40、平手などの標準 SFEN は 0）
  Mode mode_;
  Phase phase_;
  std::uint64_t key_, mkey_;
  std::vector<StateInfo> st_;
  std::vector<HistEntry> hist_;
  Outcome outcome_;
  bool mate_checked_;
  int max_ply_ = 256;
  bool count_from_41_ = true;
};

}  // namespace libra
