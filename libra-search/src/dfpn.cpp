// SPDX-License-Identifier: Apache-2.0
#include "libra/dfpn.h"

#include <algorithm>

namespace libra {

// ---- 問題 ----

ProofResult MateProblem::terminal(Position& pos, bool or_node) {
  Outcome o = pos.outcome();
  if (o.result == ONGOING) return PROOF_UNKNOWN;
  // 攻め方 = OR 節点の手番。受け方に合法手が無い（AND 節点で終局）なら証明
  Color attacker = or_node ? pos.turn() : ~pos.turn();
  Result win = attacker == BLACK ? BLACK_WIN : WHITE_WIN;
  return (o.result == win && o.reason == R_NO_LEGAL_MOVE) ? PROOF_PROVEN : PROOF_DISPROVEN;
}

void MateProblem::moves(Position& pos, bool or_node, MoveList& out) {
  MoveList all;
  pos.legal_moves(all);
  if (!or_node) {
    out = all;
    return;
  }
  out.n = 0;
  for (Move m : all)
    if (pos.gives_check(m)) out.add(m);
}

void attacking_drops(Position& pos, MoveList& out, bool with_pass) {
  out.n = 0;
  MoveList all;
  pos.legal_moves(all);
  bool pass_done[HAND_PT_NB] = {};
  MoveList passes;
  for (Move m : all) {
    PieceType pt = drop_type(m);
    bool liner = pt == ROOK || pt == BISHOP || pt == LANCE || pt == KNIGHT;
    bool attacks = liner && pos.gives_check(m);
    if (attacks) {
      out.add(m);
    } else if (with_pass && !pass_done[pt]) {
      pass_done[pt] = true;
      passes.add(m);
    }
  }
  for (Move m : passes) out.add(m);
}

ProofResult Ruling41Problem::terminal(Position& pos, bool) {
  if (pos.phase() != PHASE_NORMAL) return PROOF_UNKNOWN;
  Outcome o = pos.outcome();
  return (o.result == BLACK_WIN && o.reason == R_RULING41) ? PROOF_PROVEN : PROOF_DISPROVEN;
}

void Ruling41Problem::moves(Position& pos, bool or_node, MoveList& out) {
  if (or_node) attacking_drops(pos, out, true);
  else pos.legal_moves(out);
}

ProofResult Mate41Problem::terminal(Position& pos, bool) {
  if (pos.phase() != PHASE_NORMAL) return PROOF_UNKNOWN;
  Outcome o = pos.outcome();
  return (o.result == WHITE_WIN && o.reason == R_NO_LEGAL_MOVE) ? PROOF_PROVEN : PROOF_DISPROVEN;
}

void Mate41Problem::moves(Position& pos, bool or_node, MoveList& out) {
  if (or_node) attacking_drops(pos, out, true);
  else pos.legal_moves(out);
}

// ---- df-pn ----

DfPn::DfPn(std::size_t tt_bits) : tt_((std::size_t(1) << tt_bits) * WAYS), mask_((std::uint64_t(1) << tt_bits) - 1) {}

DfPn::Entry& DfPn::look(std::uint64_t key) {
  Entry* b = &tt_[(key & mask_) * WAYS];
  ++clock_;
  Entry* victim = b;
  for (int i = 0; i < WAYS; ++i) {
    const bool live = b[i].gen == gen_;  // 前の solve の項は空として扱う（表を毎回消す代わり）
    if (live && b[i].key == key) {
      b[i].age = clock_;
      return b[i];
    }
    if (!live || b[i].key == 0) {
      victim = &b[i];
      break;
    }
    if (b[i].age < victim->age) victim = &b[i];
  }
  victim->key = key;
  victim->gen = gen_;
  victim->pn = victim->dn = 1;
  victim->age = clock_;
  return *victim;
}

// 節点の鍵。本将棋では千日手の回数を混ぜ、同一局面でも履歴（何回目か）が違えば別の節点にする
// （経路依存の終端を置換表で取り違える GHI 問題を減らす。完全ではない）
static std::uint64_t node_key(const Position& pos) {
  std::uint64_t k = pos.key();
  if (pos.phase() == PHASE_NORMAL) {
    // 千日手の回数と通算手数を混ぜる。手数上限の近くでは同じ局面でも残り手数で結論が変わる
    k ^= std::uint64_t(pos.repetition_count()) * 0x9e3779b97f4a7c15ULL;
    k ^= std::uint64_t(pos.ply()) * 0xbf58476d1ce4e5b9ULL;
  }
  return k;
}

ProofResult DfPn::solve(Position& pos, Problem& prob, bool or_node, std::uint64_t max_nodes, Move* best) {
  nodes_ = 0;
  max_nodes_ = max_nodes;
  if (++gen_ == 0) {  // 世代が一周したら本当に消す
    for (Entry& e : tt_) e = Entry();
    gen_ = 1;
  }
  // 年齢は同じ solve の項どうしでしか比べないので、solve ごとに数え直しても入れ替えは同じ。
  // 表をスレッドで共有すると clock_ の進みが速く、数え続けると solve の途中で一周して入れ替えを誤る
  clock_ = 0;
  mid(pos, prob, or_node, INF - 1, INF - 1, 0);
  Entry& r = look(node_key(pos));
  ProofResult res = r.pn == 0 ? PROOF_PROVEN : r.dn == 0 ? PROOF_DISPROVEN : PROOF_UNKNOWN;
  if (res == PROOF_PROVEN && or_node && best) {
    *best = MOVE_NONE;
    MoveList ml;
    prob.moves(pos, true, ml);
    for (Move m : ml) {
      pos.do_move(m);
      ProofResult t = prob.terminal(pos, false);
      bool proven = t == PROOF_PROVEN || (t == PROOF_UNKNOWN && look(node_key(pos)).pn == 0);
      pos.undo_move();
      if (proven) {
        *best = m;
        break;
      }
    }
  }
  return res;
}

void DfPn::mid(Position& pos, Problem& prob, bool or_node, std::uint32_t thpn, std::uint32_t thdn, int depth) {
  Entry& e = look(node_key(pos));
  if (e.pn >= thpn || e.dn >= thdn) return;
  ++nodes_;
  ProofResult t = prob.terminal(pos, or_node);
  if (t == PROOF_PROVEN) {
    e.pn = 0;
    e.dn = INF;
    return;
  }
  if (t == PROOF_DISPROVEN) {
    e.pn = INF;
    e.dn = 0;
    return;
  }
  if (depth >= max_depth_ || nodes_ >= max_nodes_) return;
  MoveList ml;
  prob.moves(pos, or_node, ml);
  if (ml.n == 0) {
    if (or_node) {
      e.pn = INF;
      e.dn = 0;
    } else {
      e.pn = 0;
      e.dn = INF;
    }
    return;
  }
  // 子の鍵は深さごとに keys_ に積む（節点ごとに確保しない。再帰の間に伸びるので添字で引く）
  struct KeyFrame {
    std::vector<std::uint64_t>& v;
    std::size_t base;
    ~KeyFrame() { v.resize(base); }
  } frame{keys_, keys_.size()};
  keys_.resize(frame.base + std::size_t(ml.n));
  for (int i = 0; i < ml.n; ++i) {
    pos.do_move(ml.m[i]);
    keys_[frame.base + i] = node_key(pos);
    pos.undo_move();
  }
  for (;;) {
    // 子の pn/dn を集計
    std::uint32_t pn, dn;
    int best = -1;
    std::uint32_t best_v = INF, second_v = INF, best_other = 0;
    if (or_node) {
      pn = INF;
      dn = 0;
      for (int i = 0; i < ml.n; ++i) {
        Entry& c = look(keys_[frame.base + i]);
        if (c.pn < best_v) {
          second_v = best_v;
          best_v = c.pn;
          best = i;
          best_other = c.dn;
        } else if (c.pn < second_v) {
          second_v = c.pn;
        }
        pn = std::min(pn, c.pn);
        dn = std::min(INF, dn + c.dn);
      }
    } else {
      pn = 0;
      dn = INF;
      for (int i = 0; i < ml.n; ++i) {
        Entry& c = look(keys_[frame.base + i]);
        if (c.dn < best_v) {
          second_v = best_v;
          best_v = c.dn;
          best = i;
          best_other = c.pn;
        } else if (c.dn < second_v) {
          second_v = c.dn;
        }
        pn = std::min(INF, pn + c.pn);
        dn = std::min(dn, c.dn);
      }
    }
    Entry& me = look(node_key(pos));
    me.pn = pn;
    me.dn = dn;
    if (pn >= thpn || dn >= thdn || pn == 0 || dn == 0 || nodes_ >= max_nodes_) return;
    std::uint32_t c_thpn, c_thdn;
    if (or_node) {
      c_thpn = std::min(thpn, second_v == INF ? INF : second_v + 1);
      c_thdn = std::min(INF, thdn - dn + best_other);
    } else {
      c_thdn = std::min(thdn, second_v == INF ? INF : second_v + 1);
      c_thpn = std::min(INF, thpn - pn + best_other);
    }
    pos.do_move(ml.m[best]);
    mid(pos, prob, !or_node, c_thpn, c_thdn, depth + 1);
    pos.undo_move();
  }
}

}  // namespace libra
