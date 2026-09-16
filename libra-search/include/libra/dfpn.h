// SPDX-License-Identifier: Apache-2.0
// df-pn（Nagai 2002 の手法を参考にゼロから実装）。AND/OR 木の証明数・反証数探索。
// 問題（Problem）を差し替えて、本将棋の詰み・布石の「41 手目の裁定」・「41 手目の先手詰み」を解く。
#pragma once
#include <cstdint>
#include <vector>
#include "libra/position.h"

namespace libra {

enum ProofResult : int { PROOF_UNKNOWN = 0, PROOF_PROVEN = 1, PROOF_DISPROVEN = 2 };

struct Problem {
  virtual ~Problem() = default;
  // 終端判定。or_node は「攻め方の手番」。
  virtual ProofResult terminal(Position& pos, bool or_node) = 0;
  // その節点で考える手。OR 節点で空なら反証、AND 節点で空なら証明
  virtual void moves(Position& pos, bool or_node, MoveList& out) = 0;
};

// 本将棋の詰み: 攻め方は王手だけ、受け方は全合法手。終端は libra-sim の裁定
struct MateProblem : Problem {
  ProofResult terminal(Position& pos, bool or_node) override;
  void moves(Position& pos, bool or_node, MoveList& out) override;
};

// 布石: 先手が 40 手完了時に「後手玉が当たっている」（41 手目の裁定）を強制できるか。
// 攻め方（先手）は後手玉に当たる打ち手＋各駒種のパス、受け方（後手）は全合法手。
struct Ruling41Problem : Problem {
  ProofResult terminal(Position& pos, bool or_node) override;
  void moves(Position& pos, bool or_node, MoveList& out) override;
};

// 布石: 後手が「41 手目に先手に合法手なし」を強制できるか。
// 攻め方（後手）は先手玉に当たる打ち手＋パス、受け方（先手）は全合法手。
struct Mate41Problem : Problem {
  ProofResult terminal(Position& pos, bool or_node) override;
  void moves(Position& pos, bool or_node, MoveList& out) override;
};

// 相手玉に当たる打ち手（飛角香桂）と、駒種ごとの「当たらない」打ち手を 1 つずつ（パス）
void attacking_drops(Position& pos, MoveList& out, bool with_pass);

class DfPn {
 public:
  explicit DfPn(std::size_t tt_bits = 16);
  // 根が or_node（攻め方の手番）かどうかを指定して解く。max_nodes を超えたら UNKNOWN。
  // 証明できたとき、根が OR 節点なら best に証明手（pn=0 の子）を入れる
  ProofResult solve(Position& pos, Problem& prob, bool or_node, std::uint64_t max_nodes, Move* best = nullptr);
  std::uint64_t nodes() const { return nodes_; }

 private:
  static constexpr std::uint32_t INF = 1u << 30;
  static constexpr int WAYS = 4;  // 4-way セットアソシアティブ。直接写像だと衝突で再展開を繰り返す
  struct Entry {
    std::uint64_t key = 0;
    std::uint32_t pn = 1, dn = 1;
    std::uint32_t age = 0;
    std::uint32_t gen = 0;  // 書いた solve の世代。gen_ と違えば空
  };
  std::vector<Entry> tt_;
  std::uint64_t mask_;
  std::uint64_t nodes_ = 0, max_nodes_ = 0;
  std::uint32_t clock_ = 0;
  std::uint32_t gen_ = 0;
  std::vector<std::uint64_t> keys_;  // mid の子の鍵（深さごとに積む）
  int max_depth_ = 64;
  Entry& look(std::uint64_t key);
  void mid(Position& pos, Problem& prob, bool or_node, std::uint32_t thpn, std::uint32_t thdn, int depth);
};

}  // namespace libra
