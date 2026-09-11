# SPDX-License-Identifier: Apache-2.0
"""玉配置ペアの列挙・剪定・鏡映（docs/libra-design.md §5 の 1）。

マスは libra-sim の添字（sq = file*9 + rank、file 0 = 1 筋、rank 0 = 一段目）。USI 表記との変換は librashogi。

剪定（ルール由来の即死）: 後手玉が四段目（rank 3）にあると、先手は 3 手目に六段目へ桂を打って後手玉に当てられる。
布石中は駒を取れず桂の利きは遮れないので、40 手完了時も当たったままで先手の裁定勝ち（docs/rules.md §3.4）。
先手玉が六段目でも後手の桂は四段目から当てられるが、41 手目は先手の手番なので取るか逃げればよく即死ではない。
よって置く側が選べるのは 36（先手玉）× 27（後手玉、一〜三段目）= 972 通り、1↔9 筋の鏡映で 492 通り。
"""
from __future__ import annotations

import librashogi as ls


def sq(file0: int, rank0: int) -> int:
    return file0 * 9 + rank0


SENTE_SQUARES = [sq(f, r) for r in range(5, 9) for f in range(9)]   # 六〜九段目
GOTE_SQUARES_ALL = [sq(f, r) for r in range(0, 4) for f in range(9)]  # 一〜四段目
GOTE_SQUARES = [sq(f, r) for r in range(0, 3) for f in range(9)]      # 剪定後（一〜三段目）


def mirror_sq(s: int) -> int:
    f, r = divmod(s, 9)
    return sq(8 - f, r)


def all_pairs() -> list[tuple[int, int]]:
    return [(kb, kw) for kb in SENTE_SQUARES for kw in GOTE_SQUARES_ALL]


def pruned_pairs() -> list[tuple[int, int]]:
    return [(kb, kw) for kb in SENTE_SQUARES for kw in GOTE_SQUARES]


def canonical(kb: int, kw: int) -> tuple[int, int]:
    """鏡映で同じになるペアの代表（辞書順で小さい方）。"""
    m = (mirror_sq(kb), mirror_sq(kw))
    return min((kb, kw), m)


def unique_pairs() -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out = []
    for p in pruned_pairs():
        c = canonical(*p)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def usi(s: int) -> str:
    return ls.sq_to_usi(s)


def from_usi(s: str) -> int:
    return ls.sq_from_usi(s)


def is_immediate_loss(kb: int, kw: int) -> bool:
    """剪定条件（後手玉が四段目）。libra-sim で裏付けるテストが tests/test_scale.py にある。"""
    return kw % 9 == 3
