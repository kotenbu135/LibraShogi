# SPDX-License-Identifier: Apache-2.0
"""USI 布石拡張（docs/protocol.md）の換算。GUI の src/usi/parse.ts の cpToWinrate / winrateToCp と厳密に同じ式。"""
import math

CP_SCALE = 435.0
CP_OFFSET = 34.0


def cp_to_winrate(cp: float, scale: float = CP_SCALE, offset: float = CP_OFFSET) -> float:
    """手番側の cp → 手番側の勝率 0..1。"""
    return 1.0 / (1.0 + math.exp(-(cp - offset) / scale))


def winrate_to_cp(p: float, scale: float = CP_SCALE, offset: float = CP_OFFSET) -> int:
    """手番側の勝率 → 擬似 cp（cp_to_winrate の逆）。-0 は 0 に揃える。"""
    q = min(max(p, 1e-6), 1 - 1e-6)
    # JS の Math.round（.5 は +∞ 側へ）に合わせる。Python の round は偶数丸めなので使わない
    cp = int(math.floor(scale * math.log(q / (1 - q)) + offset + 0.5))
    return cp if cp != 0 else 0
