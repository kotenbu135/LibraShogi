# SPDX-License-Identifier: Apache-2.0
"""本体と過去の搾取者の対局（docs/decisions.md 2026-09-14「本体の同時局の一部を過去の搾取者と打つ」）。

本体 ls の同時局の一部を、自己対局とは別のエンジン（SelfPlayLoop、別の固定バッチ）で、凍結した搾取者 lx のスナップショットと打つ。
本体はふだん通り自分のネットだけで読み、自分の手だけを方策の学習に使う（lx の手は full=0。価値は全局面で結果から）。
相手は lx が凍結相手を作り直すたびに保存するスナップショット（<pool>/lx-<step>.pt、workers.publish_weights の形式）の
新しい recent 体から、本体の勝率が低い相手ほど多く選ぶ（PFSP、重み (1 − x)²。AlphaStar の f_hard）。switch_games 局ごとに選び直す。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

POOL_NAME = re.compile(r"^lx-(\d{9})\.pt$")


def pool_name(step: int) -> str:
    return f"lx-{int(step):09d}.pt"


def list_pool(pool: Path, recent: int) -> list[tuple[int, Path]]:
    """スナップショットを step の新しい順に recent 個まで返す（書きかけの .tmp は数えない）。"""
    if not pool.is_dir():
        return []
    out = [(int(m.group(1)), p) for p in pool.iterdir() if (m := POOL_NAME.match(p.name))]
    out.sort(reverse=True)
    return out[: max(1, recent)]


def prune_pool(pool: Path, keep: int) -> list[Path]:
    """新しい keep 個を残して消し、消したパスを返す。"""
    removed = []
    for _, p in list_pool(pool, 10**9)[max(1, keep):]:
        p.unlink(missing_ok=True)
        removed.append(p)
    return removed


def main_winrate(st: dict | None) -> float:
    """本体の勝率（引き分けは半分）。対局の少ない相手は 0.5 に寄せる: (勝 + 分/2 + 1) / (局 + 2)。"""
    st = st or {}
    return (st.get("wins", 0) + 0.5 * st.get("draws", 0) + 1.0) / (st.get("games", 0) + 2.0)


def pfsp_pick(steps: list[int], stats: dict, rng: np.random.Generator) -> int:
    """PFSP: 本体の勝率 x の相手を (1 − x)² に比例して選ぶ。"""
    w = np.array([(1.0 - main_winrate(stats.get(str(s)))) ** 2 for s in steps], dtype=np.float64)
    p = w / w.sum() if w.sum() > 0 else np.full(len(steps), 1.0 / len(steps))
    return int(steps[int(rng.choice(len(steps), p=p))])


def add_result(stats: dict, opponent_step: int, main_result: int) -> None:
    st = stats.setdefault(str(int(opponent_step)), {"games": 0, "wins": 0, "draws": 0, "losses": 0})
    st["games"] += 1
    st["wins" if main_result > 0 else "draws" if main_result == 0 else "losses"] += 1


def tag_league_game(g: dict, opponent_step: int) -> dict:
    """SelfPlayLoop.round が付けた印（自分＝本体の手だけ full、exploiter_side・exploiter_result）を本体の対局の印に替える。
    v41（41 手目の探索値）は lx が読んだ値のことがあるので、対局の結果（先手から見て）に置き換える。"""
    g["league_main_side"] = g.pop("exploiter_side")
    g["league_result"] = int(g.pop("exploiter_result"))
    g["league_opponent"] = int(opponent_step)
    g["v41"] = float(g["result"])
    return g
