# SPDX-License-Identifier: Apache-2.0
"""同じネットの自己対局での較正: 探索値（root_q）を得点の予測とみなし、実際の結果と区間ごとに比べる。

`libra eval` の較正は別の世代のネットとの対局を使うので実力差で崩れる（docs/method-evidence.md §4.4）。
ここでは自己対局の記録だけを使い、布石・本将棋と手番で分ける。定義は Guo et al. 2017 §2 の信頼度曲線と ECE・MCE を、
最上位の選択肢の確信度ではなく「手番側の得点の予測」（(q+1)/2、勝 1・分 0.5・負 0）に当てはめたもの。Brier は得点の二乗誤差。
docs/decisions.md 2026-09-17。
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from .state import StateDir

FUSEKI_MOVES = 38  # 記録の手（玉 2 手の後）の添字 0..37 が布石。添字 j の局面は j+2 手目の後で、偶数が先手の番
CERTAIN_Q = 1.0 - 1e-6  # |root_q| がこれ以上は証明済みの手（play_forced の ±1）などの確定値。ネットの予測ではないので別に数える
GROUPS = ("all", "start", "fuseki_sente", "fuseki_gote", "normal_sente", "normal_gote")


def reliability(pred: np.ndarray, actual: np.ndarray, bins: int = 10) -> dict:
    """pred・actual は [0, 1]。区間は幅 1/bins の等幅（1.0 は最後の区間）。"""
    pred = np.asarray(pred, np.float64)
    actual = np.asarray(actual, np.float64)
    n = len(pred)
    if n == 0:
        return {"n": 0}
    idx = np.clip((pred * bins).astype(np.int64), 0, bins - 1)  # float32 の丸めで範囲をわずかに外れる値も端の区間に入れる
    cnt = np.bincount(idx, minlength=bins)
    sp = np.bincount(idx, weights=pred, minlength=bins)
    sa = np.bincount(idx, weights=actual, minlength=bins)
    out_bins, ece, mce = [], 0.0, 0.0
    for i in np.flatnonzero(cnt):
        mp, ma = sp[i] / cnt[i], sa[i] / cnt[i]
        gap = abs(mp - ma)
        ece += cnt[i] / n * gap
        mce = max(mce, gap)
        out_bins.append({"lo": round(i / bins, 2), "hi": round((i + 1) / bins, 2), "n": int(cnt[i]), "pred": round(float(mp), 4), "actual": round(float(ma), 4)})
    mean_a = float(actual.mean())
    return {
        "n": int(n), "pred": round(float(pred.mean()), 4), "actual": round(mean_a, 4), "bias": round(float(pred.mean()) - mean_a, 4),
        "ece": round(float(ece), 4), "mce": round(float(mce), 4),
        "brier": round(float(((pred - actual) ** 2).mean()), 4),  # 小さいほど良い
        "brier_ref": round(float(((actual - mean_a) ** 2).mean()), 4),  # 実際の平均を言い続けたときの Brier（これより小さければ予測に情報がある）
        "bins": out_bins,
    }


def selfplay_calibration(games: list[dict], bins: int = 10) -> dict:
    """全読みの手（full）だけを使う。早読みの手・布石の手順（openings）・搾取者の相手の手は full=0 なので入らない。
    本体と過去の搾取者のリーグの対局（league_opponent）は、自己対局と勝率の分布が違うので局ごと除く（除いた局数を league_skipped に出す）。"""
    groups: dict = {}
    n_all = len(games)
    games = [g for g in games if "league_opponent" not in g]
    if not games:
        return {"games": 0, "league_skipped": n_all, "certain": {"n": 0, "agree": 0}, "groups": {k: {"n": 0} for k in GROUPS}}
    lens = np.array([len(g["root_q"]) for g in games], np.int64)
    q = np.concatenate([np.asarray(g["root_q"], np.float64) for g in games])
    full = np.concatenate([np.asarray(g["full"]).astype(bool) for g in games])
    res = np.repeat(np.array([int(g["result"]) for g in games], np.float64), lens)
    j = np.arange(len(q)) - np.repeat(np.cumsum(lens) - lens, lens)
    sente = j % 2 == 0
    z = np.where(sente, res, -res)  # 手番側の結果
    certain = full & (np.abs(q) >= CERTAIN_Q)
    use = full & ~certain
    pred, actual = (q + 1) / 2, (z + 1) / 2
    fuseki = j < FUSEKI_MOVES
    masks = {"all": use, "start": use & (j == 0), "fuseki_sente": use & fuseki & sente, "fuseki_gote": use & fuseki & ~sente,
             "normal_sente": use & ~fuseki & sente, "normal_gote": use & ~fuseki & ~sente}
    for k in GROUPS:
        groups[k] = reliability(pred[masks[k]], actual[masks[k]], bins)
    agree = int((np.sign(q[certain]) == np.sign(z[certain])).sum())
    return {"games": len(games), "league_skipped": n_all - len(games), "certain": {"n": int(certain.sum()), "agree": agree}, "groups": groups}


def newest_games(replay_dir: Path, n_games: int) -> list[dict]:
    """書き出し済みのチャンクを新しい側から読み、最新の n_games 局を古い順で返す。"""
    out: list[dict] = []
    for p in sorted(replay_dir.glob("chunk_*.pkl"), reverse=True):
        with open(p, "rb") as f:
            out = pickle.load(f) + out
        if len(out) >= n_games:
            break
    return out[-n_games:] if n_games > 0 else []


def calib_path(sd: StateDir) -> Path:
    return sd.root / "calib.jsonl"


def append_calib(sd: StateDir, row: dict) -> None:
    with open(calib_path(sd), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_calib(sd: StateDir, max_rows: int = 600) -> list[dict]:
    """新しい側から max_rows 行。区間（bins）は最後の行だけに残す（管理コンソールが 30 秒ごとに読むため）。"""
    p = calib_path(sd)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows = rows[-max_rows:]
    for r in rows[:-1]:
        for g in (r.get("groups") or {}).values():
            g.pop("bins", None)
    return rows


def make_row(games: list[dict], step: int | None, generation: int | None, bins: int = 10, now: float | None = None) -> dict:
    c = selfplay_calibration(games, bins)
    return {"t": round(time.time() if now is None else now, 1), "step": step, "generation": generation, **c}


def format_table(c: dict) -> str:
    names = {"all": "全体", "start": "3 手目（両玉の直後）", "fuseki_sente": "布石・先手の番", "fuseki_gote": "布石・後手の番",
             "normal_sente": "本将棋・先手の番", "normal_gote": "本将棋・後手の番"}
    lines = [f"games {c['games']}（リーグの対局 {c.get('league_skipped', 0)} 局を除く）  step {c.get('step')}  確定値（|q|=1）{c['certain']['n']} 手、結果と一致 {c['certain']['agree']}",
             f"{'':<22}{'n':>9}{'予測':>8}{'実際':>8}{'偏り':>8}{'ECE':>8}{'MCE':>8}{'Brier':>8}{'基準':>8}"]
    for k in GROUPS:
        g = c["groups"][k]
        if not g.get("n"):
            lines.append(f"{names[k]:<22}{0:>9}")
            continue
        lines.append(f"{names[k]:<22}{g['n']:>9}{g['pred']:>8.3f}{g['actual']:>8.3f}{g['bias']:>+8.3f}{g['ece']:>8.3f}{g['mce']:>8.3f}{g['brier']:>8.3f}{g['brier_ref']:>8.3f}")
    for k in GROUPS:
        g = c["groups"][k]
        if g.get("n"):
            lines.append(f"{names[k]}: " + " ".join(f"[{b['lo']:.1f},{b['hi']:.1f}) n={b['n']} pred={b['pred']:.3f} act={b['actual']:.3f}" for b in g["bins"]))
    return "\n".join(lines)
