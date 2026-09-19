# SPDX-License-Identifier: Apache-2.0
"""自己対局に投了を入れたら、ネットの評価を何割減らせるかを、打ち終わった棋譜から見積もる。

**投了そのものは入れていない**。ここでやるのは「入れていたらどこで打ち切れたか」を、最後まで打った記録の上で
数え直すこと。AlphaGo Zero（Silver+ 2017, Methods「Resignation」）が誤投了の割合を測るのに使う形と同じで、
あちらは 10% の対局だけを最後まで打って調べる。Libra は全部の対局を最後まで打っているので、全部を使える。

規則: 手番側の探索値 `root_q` が −T 以下の状態が、その側の連続 K 手続いたら、その側が投了する。
布石（記録の添字 `FUSEKI_MOVES` 手目まで）では投了しない。

評価の回数は記録から数え直す: 全読みの手は `full_sims`、早読みの手は `fast_sims`、
布石の手順（`openings`）から指した手と証明済みの手（`play_forced`、|q| = 1）は探索していないので 0。
GPU が律速なので、評価の節約率 r はそのまま局/日の伸び（約 1/(1−r) 倍）と読める。
"""
from __future__ import annotations

import numpy as np

from .calibrate import CERTAIN_Q, FUSEKI_MOVES

DEFAULT_THRESHOLDS = (0.90, 0.95, 0.98, 0.99)
DEFAULT_RUNS = (1, 2, 3)


def _pad(games: list[dict]) -> dict:
    """棋譜を (局数 × 最大手数) の表にそろえる。盤外は valid=False。"""
    lens = np.array([len(g["root_q"]) for g in games], np.int64)
    n, w = len(games), int(lens.max())
    q = np.zeros((n, w), np.float64)
    full = np.zeros((n, w), bool)
    for i, g in enumerate(games):
        m = len(g["root_q"])
        q[i, :m] = np.asarray(g["root_q"], np.float64)
        full[i, :m] = np.asarray(g["full"]).astype(bool)
    valid = np.arange(w)[None, :] < lens[:, None]
    result = np.array([int(g["result"]) for g in games], np.float64)
    return {"q": q, "full": full, "valid": valid, "lens": lens, "result": result}


def _sims(t: dict, full_sims: int, fast_sims: int) -> tuple[np.ndarray, np.ndarray]:
    """1 手あたりのネットの評価回数と、布石の手順から指した手の印を返す。"""
    q, full, valid = t["q"], t["full"], t["valid"]
    # 布石の手順（openings）から指した手は full=False・root_q=0 で、対局の先頭に連なる。先頭の連なりだけを見る
    book = np.logical_and.accumulate(valid & ~full & (q == 0.0), axis=1)
    proven = valid & full & (np.abs(q) >= CERTAIN_Q)  # 証明済み（play_forced）。探索していない
    sims = np.where(valid & ~book & ~proven, np.where(full, full_sims, fast_sims), 0).astype(np.int64)
    return sims, book


def _first_resign(t: dict, thr: float, run: int, min_move: int) -> np.ndarray:
    """各局で最初に投了が出る手の添字（出なければ −1）。手番側は添字の偶奇（偶数が先手）。"""
    q, valid, w = t["q"], t["valid"], t["q"].shape[1]
    below = valid & (q <= -thr)
    out = np.full(len(q), -1, np.int64)
    for parity in (0, 1):
        cnt = np.zeros(len(q), np.int64)
        trig = np.full(len(q), -1, np.int64)
        for j in range(parity, w, 2):
            if j < min_move:
                continue
            b = below[:, j] & (trig < 0)
            cnt = np.where(b, cnt + 1, 0)
            hit = b & (cnt >= run)
            trig = np.where(hit, j, trig)
        both = (out >= 0) & (trig >= 0)
        out = np.where(both, np.minimum(out, trig), np.where(trig >= 0, trig, out))
    return out


def resign_scan(games: list[dict], full_sims: int, fast_sims: int, thresholds=DEFAULT_THRESHOLDS,
                runs=DEFAULT_RUNS, min_move: int = FUSEKI_MOVES) -> dict:
    """リーグの対局（本体と過去の搾取者）は自己対局と勝率の分布が違うので局ごと除く。"""
    n_all = len(games)
    games = [g for g in games if "league_opponent" not in g and len(g.get("root_q", ()))]
    if not games:
        return {"games": 0, "league_skipped": n_all, "rows": []}
    t = _pad(games)
    sims, book = _sims(t, full_sims, fast_sims)
    lens, result = t["lens"], t["result"]
    evals = sims.sum(axis=1)
    # cum[:, j] = 添字 j より前の手の評価回数の合計
    cum = np.concatenate([np.zeros((len(games), 1), np.int64), np.cumsum(sims, axis=1)], axis=1)
    rows = []
    for thr in thresholds:
        for run in runs:
            j = _first_resign(t, thr, run, min_move)
            hit = j >= 0
            jj = np.where(hit, j, lens - 1)
            saved_moves = np.where(hit, lens - 1 - jj, 0)
            saved_evals = np.where(hit, evals - cum[np.arange(len(games)), jj + 1], 0)
            sente = (jj % 2) == 0                       # 投了する側
            z = np.where(sente, result, -result)        # 投了する側から見た実際の結果
            wrong_win = int((hit & (z > 0)).sum())      # 実際には勝っていた
            wrong_draw = int((hit & (z == 0)).sum())    # 実際には引き分けだった
            rows.append({
                "thr": thr, "run": run,
                "resigned": int(hit.sum()), "resigned_frac": round(float(hit.mean()), 4),
                "moves_saved_frac": round(float(saved_moves.sum() / lens.sum()), 4),
                "evals_saved_frac": round(float(saved_evals.sum() / evals.sum()), 4),
                "speedup": round(float(1.0 / (1.0 - saved_evals.sum() / evals.sum())), 3),
                "mean_resign_move": round(float(jj[hit].mean()), 1) if hit.any() else None,
                "wrong_win": wrong_win, "wrong_draw": wrong_draw,
                "wrong_win_frac": round(wrong_win / max(1, int(hit.sum())), 4),
                "wrong_draw_frac": round(wrong_draw / max(1, int(hit.sum())), 4),
            })
    return {
        "games": len(games), "league_skipped": n_all - len(games),
        "full_sims": full_sims, "fast_sims": fast_sims, "min_move": min_move,
        "mean_moves": round(float(lens.mean()), 1),
        "mean_evals": round(float(evals.mean()), 1),
        "book_moves": int(book.sum()), "proven_moves": int((t["valid"] & (sims == 0)).sum() - book.sum()),
        "draw_frac": round(float((result == 0).mean()), 4),
        "rows": rows,
    }


def format_table(res: dict) -> str:
    if not res.get("games"):
        return "対局が無い"
    head = (f"games {res['games']}（リーグの対局 {res['league_skipped']} 局を除く）  "
            f"平均 {res['mean_moves']} 手・{res['mean_evals']} 評価/局  "
            f"読み {res['full_sims']}/{res['fast_sims']}  布石の手順 {res['book_moves']} 手・証明済み {res['proven_moves']} 手  "
            f"引き分け {res['draw_frac'] * 100:.1f}%")
    lines = [head, "投了の規則: 手番側の探索値が −しきい値 以下の状態がその側の連続 K 手続いたら投了（布石では投了しない）", "",
             f"{'しきい値':>8}{'K':>4}{'投了した局':>12}{'手数の節約':>12}{'評価の節約':>12}{'局/日':>9}"
             f"{'平均の投了手':>14}{'実は勝ち':>10}{'実は引分':>10}"]
    for r in res["rows"]:
        lines.append(f"{r['thr']:>8.2f}{r['run']:>4}{r['resigned_frac'] * 100:>11.1f}%{r['moves_saved_frac'] * 100:>11.1f}%"
                     f"{r['evals_saved_frac'] * 100:>11.1f}%{r['speedup']:>8.2f}x"
                     f"{(r['mean_resign_move'] if r['mean_resign_move'] is not None else float('nan')):>14.1f}"
                     f"{r['wrong_win_frac'] * 100:>9.2f}%{r['wrong_draw_frac'] * 100:>9.2f}%")
    lines.append("")
    lines.append("「実は勝ち」は投了した局のうち、その側が最後まで打てば勝っていた割合（誤投了）。AlphaGo Zero は 5% 以下に保つ。")
    lines.append("「局/日」は評価の節約から出した見込みの倍率（GPU が律速なので 1/(1−節約) 倍）。")
    return "\n".join(lines)
