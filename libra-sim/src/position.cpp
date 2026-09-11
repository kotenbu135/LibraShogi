// SPDX-License-Identifier: Apache-2.0
#include "libra/position.h"

#include <cstdio>
#include <sstream>

namespace libra {

namespace {

std::uint64_t zob_psq[PIECE_NB][SQ_NB];
std::uint64_t zob_hand[COLOR_NB][HAND_PT_NB][19];
std::uint64_t zob_turn;
bool zob_ready = false;

std::uint64_t splitmix64(std::uint64_t& x) {
  std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

void zobrist_init() {
  if (zob_ready) return;
  zob_ready = true;
  std::uint64_t seed = 0x4c69627261u;  // "Libra"
  for (auto& row : zob_psq)
    for (auto& v : row) v = splitmix64(seed);
  for (auto& c : zob_hand)
    for (auto& pt : c) {
      pt[0] = 0;
      for (int n = 1; n < 19; ++n) pt[n] = splitmix64(seed);
    }
  zob_turn = splitmix64(seed);
}

const char* const STARTPOS_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

// 持ち駒の並び（出力用）。玉は布石中だけ現れる。
const PieceType HAND_ORDER[] = {KING, ROOK, BISHOP, GOLD, SILVER, KNIGHT, LANCE, PAWN};

}  // namespace

const char* result_name(Result r) {
  switch (r) {
    case ONGOING: return "ongoing";
    case BLACK_WIN: return "sente";
    case WHITE_WIN: return "gote";
    case DRAW: return "draw";
  }
  return "?";
}

const char* reason_name(Reason r) {
  switch (r) {
    case R_NONE: return "none";
    case R_NO_LEGAL_MOVE: return "no_legal_move";
    case R_RULING41: return "ruling41";
    case R_SENNICHITE: return "sennichite";
    case R_PERPETUAL_CHECK: return "perpetual_check";
    case R_DECLARATION: return "declaration";
    case R_ILLEGAL_DECLARATION: return "illegal_declaration";
    case R_ILLEGAL_MOVE: return "illegal_move";
    case R_MAX_PLY: return "max_ply";
    case R_RESIGN: return "resign";
    case R_TIMEOUT: return "timeout";
  }
  return "?";
}

Position::Position() {
  bb::init();
  zobrist_init();
  reset(MODE_TENBIN);
}

void Position::clear() {
  for (auto& p : board_) p = NO_PIECE;
  for (auto& b : by_color_) b = Bitboard();
  for (auto& b : by_type_) b = Bitboard();
  occ_ = Bitboard();
  for (auto& c : hand_)
    for (auto& n : c) n = 0;
  turn_ = BLACK;
  ply_ = 0;
  base_ply_ = 0;
  key_ = mkey_ = 0;
  st_.clear();
  hist_.clear();
  outcome_ = Outcome();
  mate_checked_ = false;
}

void Position::reset(Mode mode) {
  clear();
  mode_ = mode;
  phase_ = PHASE_FUSEKI;
  const int counts[HAND_PT_NB] = {0, 9, 2, 2, 2, 1, 1, 2, 1};
  for (int c = 0; c < COLOR_NB; ++c)
    for (int pt = PAWN; pt <= KING; ++pt) add_hand(Color(c), PieceType(pt), counts[pt]);
}

void Position::put_piece(Piece p, int sq) {
  board_[sq] = p;
  by_color_[color_of(p)].set(sq);
  by_type_[type_of(p)].set(sq);
  occ_.set(sq);
  key_ ^= zob_psq[p][sq];
  mkey_ ^= zob_psq[p][mirror_sq(sq)];
}

void Position::remove_piece(int sq) {
  Piece p = board_[sq];
  board_[sq] = NO_PIECE;
  by_color_[color_of(p)].clear(sq);
  by_type_[type_of(p)].clear(sq);
  occ_.clear(sq);
  key_ ^= zob_psq[p][sq];
  mkey_ ^= zob_psq[p][mirror_sq(sq)];
}

void Position::add_hand(Color c, PieceType pt, int d) {
  int old = hand_[c][pt];
  hand_[c][pt] = old + d;
  std::uint64_t k = zob_hand[c][pt][old] ^ zob_hand[c][pt][old + d];
  key_ ^= k;
  mkey_ ^= k;
}

void Position::set_turn(Color c) {
  if (turn_ != c) {
    key_ ^= zob_turn;
    mkey_ ^= zob_turn;
    turn_ = c;
  }
}

int Position::king_sq(Color c) const {
  Bitboard k = pieces(c, KING);
  return k.any() ? k.lsb() : SQ_NONE;
}

bool Position::fuseki_done() const {
  for (int c = 0; c < COLOR_NB; ++c)
    for (int pt = PAWN; pt <= KING; ++pt)
      if (hand_[c][pt]) return false;
  return true;
}

// ---- 利き ----

Bitboard Position::attackers_to(int sq, Color by, Bitboard occ) const {
  Color other = ~by;
  Bitboard golds = pieces(by, GOLD) | pieces(by, PRO_PAWN) | pieces(by, PRO_LANCE) | pieces(by, PRO_KNIGHT) | pieces(by, PRO_SILVER);
  Bitboard r;
  r |= bb::StepAttacks[other][PAWN][sq] & pieces(by, PAWN);
  r |= bb::StepAttacks[other][KNIGHT][sq] & pieces(by, KNIGHT);
  r |= bb::StepAttacks[other][SILVER][sq] & pieces(by, SILVER);
  r |= bb::StepAttacks[other][GOLD][sq] & golds;
  r |= bb::StepAttacks[other][KING][sq] & pieces(by, KING);
  r |= bb::OrthoStep[sq] & pieces(by, HORSE);
  r |= bb::DiagStep[sq] & pieces(by, DRAGON);
  r |= bb::lance_attacks(other, sq, occ) & pieces(by, LANCE);
  r |= bb::bishop_attacks(sq, occ) & (pieces(by, BISHOP) | pieces(by, HORSE));
  r |= bb::rook_attacks(sq, occ) & (pieces(by, ROOK) | pieces(by, DRAGON));
  return r;
}

bool Position::attacked_after(int ksq, Color by, Bitboard occ2, Bitboard exclude) const {
  return (attackers_to(ksq, by, occ2) & ~exclude).any();
}

bool Position::king_attacked(Color c) const {
  int k = king_sq(c);
  if (k == SQ_NONE) return false;
  return attackers_to(k, ~c, occ_).any();
}

bool Position::in_check() const { return phase_ == PHASE_NORMAL && king_attacked(turn_); }

// ---- 布石の合法手 ----

void Position::fuseki_moves(MoveList& out) {
  out.n = 0;
  Color us = turn_;
  Bitboard zone = bb::ZoneBB[us] & ~occ_;
  bool kings_only = mode_ == MODE_TENBIN && ply_ < 2;
  Bitboard own = pieces(us);
  Bitboard own_pawns = pieces(us, PAWN);
  Bitboard own_nonpawn = own & ~own_pawns;
  // 二歩の筋と、筋埋め禁止（自陣 4 マスのうち 3 マスが歩以外の自駒）の筋
  Bitboard pawn_files, fill_files;
  for (int f = 0; f < 9; ++f) {
    if ((own_pawns & bb::FileBB[f]).any()) pawn_files |= bb::FileBB[f];
    if ((own_nonpawn & bb::ZoneBB[us] & bb::FileBB[f]).count() == 3) fill_files |= bb::FileBB[f];
  }
  for (int pt = PAWN; pt <= KING; ++pt) {
    if (hand_[us][pt] == 0) continue;
    if (kings_only && pt != KING) continue;
    Bitboard targets = zone & ~(pt == PAWN ? pawn_files : fill_files);
    while (targets.any()) out.add(make_drop(PieceType(pt), targets.pop()));
  }
  // 40 手目の制限: 最後の 1 枚を打つ手番（自分の残り 1 枚、相手は 0 枚）で自玉が当たっているなら、
  // 打った後に自玉が当たらない手だけが合法。1 つも無ければ制限を外す（41 手目の裁定）。
  int remaining = 0;
  for (int pt = PAWN; pt <= KING; ++pt) remaining += hand_[us][pt] + hand_[~us][pt];
  if (remaining == 1) {
    MoveList safe;
    for (Move m : out) {
      int k = drop_type(m) == KING ? to_sq(m) : king_sq(us);
      if (k == SQ_NONE) continue;
      Bitboard occ2 = occ_ | Bitboard::sq(to_sq(m));
      if (!attacked_after(k, ~us, occ2, Bitboard())) safe.add(m);
    }
    if (safe.n > 0) out = safe;
  }
}

bool Position::ruling41_pending() {
  if (phase_ != PHASE_FUSEKI) return false;
  int remaining = 0;
  for (int pt = PAWN; pt <= KING; ++pt) remaining += hand_[turn_][pt] + hand_[~turn_][pt];
  if (remaining != 1) return false;
  MoveList ml;
  fuseki_moves(ml);
  for (Move m : ml) {
    int k = drop_type(m) == KING ? to_sq(m) : king_sq(turn_);
    Bitboard occ2 = occ_ | Bitboard::sq(to_sq(m));
    if (!attacked_after(k, ~turn_, occ2, Bitboard())) return false;
  }
  return ml.n > 0;
}

// ---- 本将棋の合法手 ----

void Position::normal_moves(MoveList& out, bool check_uchifuzume) {
  out.n = 0;
  Color us = turn_, them = ~us;
  Bitboard own = pieces(us);
  int ksq = king_sq(us);
  Bitboard promo_zone = bb::PromoZoneBB[us];

  Bitboard src = own;
  while (src.any()) {
    int from = src.pop();
    PieceType pt = type_of(board_[from]);
    Bitboard targets = bb::attacks(us, pt, from, occ_) & ~own;
    Bitboard from_bb = Bitboard::sq(from);
    while (targets.any()) {
      int to = targets.pop();
      Bitboard to_bb = Bitboard::sq(to);
      int k2 = pt == KING ? to : ksq;
      Bitboard occ2 = (occ_ ^ from_bb) | to_bb;
      if (k2 != SQ_NONE && attacked_after(k2, them, occ2, to_bb)) continue;
      bool can_promo = is_promotable(pt) && (promo_zone.test(from) || promo_zone.test(to));
      int rr = rel_rank(us, to);
      bool must_promo = (pt == PAWN || pt == LANCE) ? rr == 0 : (pt == KNIGHT ? rr <= 1 : false);
      if (can_promo) out.add(make_move(from, to, true));
      if (!must_promo) out.add(make_move(from, to, false));
    }
  }

  Bitboard empty = ~occ_;
  Bitboard last1 = us == BLACK ? bb::RankBB[0] : bb::RankBB[8];
  Bitboard last2 = us == BLACK ? bb::RankBB[1] : bb::RankBB[7];
  for (int pt = PAWN; pt <= GOLD; ++pt) {
    if (hand_[us][pt] == 0) continue;
    Bitboard targets = empty;
    if (pt == PAWN || pt == LANCE) targets &= ~last1;
    if (pt == KNIGHT) targets &= ~(last1 | last2);
    if (pt == PAWN) {
      Bitboard own_pawns = pieces(us, PAWN);
      for (int f = 0; f < 9; ++f)
        if ((own_pawns & bb::FileBB[f]).any()) targets &= ~bb::FileBB[f];
    }
    while (targets.any()) {
      int to = targets.pop();
      Bitboard occ2 = occ_ | Bitboard::sq(to);
      if (ksq != SQ_NONE && attacked_after(ksq, them, occ2, Bitboard())) continue;
      Move m = make_drop(PieceType(pt), to);
      if (pt == PAWN && check_uchifuzume) {
        // 打ち歩詰め: 打った歩が相手玉に王手で、相手に合法手が無ければ非合法
        int ek = king_sq(them);
        if (ek != SQ_NONE && bb::StepAttacks[us][PAWN][to].test(ek)) {
          do_move(m);
          bool escape = has_legal_move(false);
          undo_move();
          if (!escape) continue;
        }
      }
      out.add(m);
    }
  }
}

bool Position::has_legal_move(bool check_uchifuzume) {
  MoveList ml;
  if (phase_ == PHASE_FUSEKI) fuseki_moves(ml);
  else normal_moves(ml, check_uchifuzume);
  return ml.n > 0;
}

void Position::legal_moves(MoveList& out) {
  if (outcome_.result != ONGOING) {
    out.n = 0;
    return;
  }
  if (phase_ == PHASE_FUSEKI) fuseki_moves(out);
  else normal_moves(out, true);
}

bool Position::is_legal(Move m) {
  MoveList ml;
  legal_moves(ml);
  for (Move x : ml)
    if (x == m) return true;
  return false;
}

// ---- 着手 ----

void Position::do_move(Move m) {
  StateInfo si{m, NO_PIECE, key_, mkey_, phase_, outcome_, mate_checked_, hist_.size()};
  Color us = turn_;
  if (is_drop(m)) {
    PieceType pt = drop_type(m);
    add_hand(us, pt, -1);
    put_piece(make_piece(us, pt), to_sq(m));
  } else {
    int from = from_sq(m), to = to_sq(m);
    Piece p = board_[from];
    remove_piece(from);
    if (board_[to] != NO_PIECE) {
      si.captured = board_[to];
      remove_piece(to);
      add_hand(us, base_type(type_of(si.captured)), +1);
    }
    put_piece(is_promo(m) ? make_piece(us, promoted(type_of(p))) : p, to);
  }
  st_.push_back(si);
  set_turn(~us);
  ++ply_;
  after_move();
}

void Position::after_move() {
  mate_checked_ = false;
  outcome_ = Outcome();
  if (phase_ == PHASE_FUSEKI) {
    if (!fuseki_done()) return;
    // 40 手完了。41 手目の裁定 → 本将棋へ
    phase_ = PHASE_NORMAL;
    base_ply_ = ply_;
    hist_.clear();
    if (king_attacked(WHITE)) {
      outcome_ = {BLACK_WIN, R_RULING41};
      return;
    }
  }
  hist_.push_back({key_, in_check(), turn_});
  if (repetition_count() >= 4) {
    outcome_ = repetition_outcome();
    return;
  }
  int played = count_from_41_ ? normal_ply() : ply_;
  if (played >= max_ply_) outcome_ = {DRAW, R_MAX_PLY};
}

void Position::undo_move() {
  StateInfo si = st_.back();
  st_.pop_back();
  Move m = si.move;
  --ply_;
  Color us = ~turn_;
  set_turn(us);
  if (is_drop(m)) {
    remove_piece(to_sq(m));
    add_hand(us, drop_type(m), +1);
  } else {
    int from = from_sq(m), to = to_sq(m);
    Piece p = board_[to];
    remove_piece(to);
    put_piece(is_promo(m) ? make_piece(us, base_type(type_of(p))) : p, from);
    if (si.captured != NO_PIECE) {
      add_hand(us, base_type(type_of(si.captured)), -1);
      put_piece(si.captured, to);
    }
  }
  if (phase_ == PHASE_NORMAL && si.phase == PHASE_FUSEKI) base_ply_ = 0;
  phase_ = si.phase;
  outcome_ = si.outcome;
  mate_checked_ = si.mate_checked;
  hist_.resize(si.hist_size);
  key_ = si.key;
  mkey_ = si.mkey;
}

// ---- 裁定 ----

int Position::repetition_count() const {
  int n = 0;
  for (const HistEntry& h : hist_)
    if (h.key == key_) ++n;
  return n;
}

Outcome Position::repetition_outcome() const {
  std::size_t first = 0;
  while (first < hist_.size() && hist_[first].key != key_) ++first;
  std::size_t last = hist_.size() - 1;
  bool all_check[COLOR_NB] = {true, true};
  for (std::size_t j = first; j < last; ++j)
    if (!hist_[j + 1].in_check) all_check[hist_[j].stm] = false;
  if (all_check[BLACK] && all_check[WHITE]) return {DRAW, R_SENNICHITE};
  if (all_check[BLACK]) return {WHITE_WIN, R_PERPETUAL_CHECK};
  if (all_check[WHITE]) return {BLACK_WIN, R_PERPETUAL_CHECK};
  return {DRAW, R_SENNICHITE};
}

Outcome Position::outcome() {
  if (phase_ == PHASE_NORMAL && !mate_checked_) {
    mate_checked_ = true;
    if (outcome_.result == ONGOING || outcome_.reason == R_SENNICHITE || outcome_.reason == R_PERPETUAL_CHECK ||
        outcome_.reason == R_MAX_PLY) {
      if (!has_legal_move(true)) outcome_ = {turn_ == BLACK ? WHITE_WIN : BLACK_WIN, R_NO_LEGAL_MOVE};
    }
  }
  return outcome_;
}

int Position::declaration_points(Color c) const {
  int pts = 0;
  for (int pt = PAWN; pt <= GOLD; ++pt) pts += hand_[c][pt] * ((pt == ROOK || pt == BISHOP) ? 5 : 1);
  Bitboard zone = pieces(c) & bb::PromoZoneBB[c];
  while (zone.any()) {
    PieceType pt = type_of(board_[zone.pop()]);
    if (pt == KING) continue;
    PieceType b = base_type(pt);
    pts += (b == ROOK || b == BISHOP) ? 5 : 1;
  }
  return pts;
}

int Position::declaration_pieces(Color c) const { return (pieces(c) & bb::PromoZoneBB[c] & ~pieces(c, KING)).count(); }

bool Position::can_declare(Color c) const {
  if (phase_ != PHASE_NORMAL || turn_ != c) return false;
  int k = king_sq(c);
  if (k == SQ_NONE || !bb::PromoZoneBB[c].test(k)) return false;
  if (declaration_pieces(c) < 10) return false;
  if (declaration_points(c) < (c == BLACK ? 28 : 27)) return false;
  if (king_attacked(c)) return false;
  return true;
}

void Position::declare(Color c) {
  if (outcome_.result != ONGOING) return;
  bool ok = can_declare(c);
  Result win = c == BLACK ? BLACK_WIN : WHITE_WIN;
  Result lose = c == BLACK ? WHITE_WIN : BLACK_WIN;
  outcome_ = ok ? Outcome{win, R_DECLARATION} : Outcome{lose, R_ILLEGAL_DECLARATION};
  mate_checked_ = true;
}

void Position::resign(Color c) {
  if (outcome_.result != ONGOING) return;
  outcome_ = {c == BLACK ? WHITE_WIN : BLACK_WIN, R_RESIGN};
  mate_checked_ = true;
}

void Position::timeout(Color c) {
  if (outcome_.result != ONGOING) return;
  outcome_ = {c == BLACK ? WHITE_WIN : BLACK_WIN, R_TIMEOUT};
  mate_checked_ = true;
}

void Position::illegal_move(Color c) {
  if (outcome_.result != ONGOING) return;
  outcome_ = {c == BLACK ? WHITE_WIN : BLACK_WIN, R_ILLEGAL_MOVE};
  mate_checked_ = true;
}

// ---- SFEN ----

bool Position::set_sfen(const std::string& sfen, Phase phase) {
  std::istringstream in(sfen);
  std::string board, turn, hands, num;
  in >> board >> turn >> hands >> num;
  if (board.empty() || turn.empty() || hands.empty()) return false;
  Mode mode = mode_;
  clear();
  mode_ = mode;
  phase_ = phase;
  int file = 8, rank = 0;
  bool promo = false;
  for (char ch : board) {
    if (ch == '/') {
      if (file != -1) return false;
      file = 8;
      ++rank;
      continue;
    }
    if (ch >= '1' && ch <= '9') {
      file -= ch - '0';
      continue;
    }
    if (ch == '+') {
      promo = true;
      continue;
    }
    PieceType pt = pt_from_char(ch);
    if (pt == NO_PT || file < 0 || rank > 8) return false;
    if (promo && !is_promotable(pt)) return false;
    Color c = (ch >= 'a' && ch <= 'z') ? WHITE : BLACK;
    put_piece(make_piece(c, promo ? promoted(pt) : pt), make_sq(file, rank));
    promo = false;
    --file;
  }
  if (rank != 8 || file != -1) return false;
  if (hands != "-") {
    int count = 0;
    for (char ch : hands) {
      if (ch >= '0' && ch <= '9') {
        count = count * 10 + (ch - '0');
        continue;
      }
      PieceType pt = pt_from_char(ch);
      if (pt == NO_PT) return false;
      Color c = (ch >= 'a' && ch <= 'z') ? WHITE : BLACK;
      add_hand(c, pt, count == 0 ? 1 : count);
      count = 0;
    }
  }
  int n = num.empty() ? 1 : std::atoi(num.c_str());
  if (n < 1) n = 1;
  if (phase == PHASE_FUSEKI) {
    // 布石は着手順に依存しない: 手数と手番は盤上の駒数から決める
    ply_ = occ_.count();
    set_turn(ply_ % 2 == 0 ? BLACK : WHITE);
    if (fuseki_done()) {
      // 40 手完了の局面: 本将棋の開始局面として扱い、41 手目の裁定を見る
      phase_ = PHASE_NORMAL;
      base_ply_ = ply_;
      if (king_attacked(WHITE)) outcome_ = {BLACK_WIN, R_RULING41};
      else hist_.push_back({key_, in_check(), turn_});
    }
  } else {
    if (turn != "b" && turn != "w") return false;
    set_turn(turn == "b" ? BLACK : WHITE);
    // 手数は通算（40 手完了の局面は 41）。41 以上なら天秤将棋の本将棋、それ未満は平手などの標準局面
    ply_ = n - 1;
    base_ply_ = ply_ >= 40 ? 40 : 0;
    if (king_sq(BLACK) == SQ_NONE && king_sq(WHITE) == SQ_NONE) return false;
    if (king_attacked(~turn_)) {
      // 手番でない側の玉が当たっている: 41 手目の局面なら裁定（先手勝ち）、それ以外は不正な局面
      if (turn_ == BLACK && ply_ == 40) {
        outcome_ = {BLACK_WIN, R_RULING41};
        mate_checked_ = true;
        return true;
      }
      return false;
    }
    hist_.push_back({key_, in_check(), turn_});
  }
  return true;
}

std::string Position::sfen() const {
  std::string s;
  for (int r = 0; r < 9; ++r) {
    int empty = 0;
    for (int f = 8; f >= 0; --f) {
      Piece p = board_[make_sq(f, r)];
      if (p == NO_PIECE) {
        ++empty;
        continue;
      }
      if (empty) {
        s += char('0' + empty);
        empty = 0;
      }
      PieceType pt = type_of(p);
      if (pt >= PRO_PAWN) s += '+';
      char ch = pt_to_char(base_type(pt));
      s += color_of(p) == BLACK ? ch : char(ch - 'A' + 'a');
    }
    if (empty) s += char('0' + empty);
    if (r != 8) s += '/';
  }
  s += turn_ == BLACK ? " b " : " w ";
  std::string h;
  for (int c = 0; c < COLOR_NB; ++c)
    for (PieceType pt : HAND_ORDER) {
      int n = hand_[c][pt];
      if (n == 0) continue;
      if (n > 1) h += std::to_string(n);
      char ch = pt_to_char(pt);
      h += c == BLACK ? ch : char(ch - 'A' + 'a');
    }
  s += h.empty() ? "-" : h;
  s += " " + std::to_string(ply_ + 1);  // 手数は通算（desktop の wasm と同じ。40 手完了の局面は 41）
  return s;
}

bool Position::set_position(const std::string& line, Mode mode) {
  std::istringstream in(line);
  std::string tok;
  in >> tok;
  if (tok == "position") in >> tok;
  if (tok == "fuseki") {
    reset(mode);
  } else if (tok == "startpos") {
    reset(mode);
    if (!set_sfen(STARTPOS_SFEN, PHASE_NORMAL)) return false;
  } else if (tok == "sfen") {
    std::string b, t, h, n;
    in >> b >> t >> h;
    if (!(in >> n)) n = "1";
    if (n == "moves") {
      n = "1";
      tok = "moves";
    } else {
      tok.clear();
    }
    reset(mode);
    if (!set_sfen(b + " " + t + " " + h + " " + n, PHASE_NORMAL)) return false;
  } else {
    return false;
  }
  if (tok != "moves") {
    if (!(in >> tok)) return true;
    if (tok != "moves") return false;
  }
  while (in >> tok) {
    if (tok.rfind("choose:", 0) == 0) continue;
    Move m = move_from_usi(tok);
    if (m == MOVE_NONE || !is_legal(m)) return false;
    do_move(m);
  }
  return true;
}

std::string Position::pretty() const {
  std::string s;
  for (int r = 0; r < 9; ++r) {
    for (int f = 8; f >= 0; --f) {
      Piece p = board_[make_sq(f, r)];
      if (p == NO_PIECE) {
        s += " . ";
        continue;
      }
      PieceType pt = type_of(p);
      s += pt >= PRO_PAWN ? '+' : ' ';
      char ch = pt_to_char(base_type(pt));
      s += color_of(p) == BLACK ? ch : char(ch - 'A' + 'a');
      s += ' ';
    }
    s += '\n';
  }
  s += sfen();
  s += '\n';
  return s;
}

std::uint64_t Position::perft(int depth) {
  if (depth == 0) return 1;
  MoveList ml;
  legal_moves(ml);
  if (depth == 1) return ml.n;
  std::uint64_t n = 0;
  for (Move m : ml) {
    do_move(m);
    n += perft(depth - 1);
    undo_move();
  }
  return n;
}

}  // namespace libra
