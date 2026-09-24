# SPDX-License-Identifier: Apache-2.0
"""評価ハーネス: 2 つのネットを同じ探索設定で対局させ、先後別の得点と Elo 差、較正を出す（docs/libra-design.md §4.3）。

同時進行枠の偶奇で A の先後を入れ替える（偶数枠は A が先手）。各手の探索木はルートの手番のネットで丸ごと評価する。
`search_cfg_b` を渡すと B 側だけ別の探索設定で読む（同じ重みで σ の形を比べるときに使う。docs/ls2-settings.md §2）。
較正は、対局記録の root_q（手番側の探索値）を勝率に直したものと実際の結果を区間ごとに比べる。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import librasearch
import librashogi as ls
from libra_net.model import LibraNet, NetConfig

from .calibrate import reliability


def load_model(path: Path, device: torch.device, dtype: torch.dtype = torch.float16) -> LibraNet:
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = sd.get("config", {}).get("net", {})
    m = LibraNet(NetConfig.from_dict(cfg)).to(device)
    m.load_state_dict(sd["model"])
    m.eval()
    if dtype != torch.float32:
        m = m.to(dtype)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def elo_diff(score: float) -> float:
    s = min(max(score, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / s - 1.0)


def calibration(games: list[dict], side_of: "callable", bins: int = 10) -> list[dict]:
    """side_of(game, ply_index) が True の手だけを使う。root_q は手番側の値、結果も手番側に直す。"""
    qs, zs = [], []
    for g in games:
        res = int(g["result"])
        for j, q in enumerate(g["root_q"]):
            if not side_of(g, j):
                continue
            sente = j % 2 == 0  # 添字 0 は 3 手目（先手）
            z = res if sente else -res
            qs.append((float(q) + 1) / 2)
            zs.append((z + 1) / 2)
    return reliability(np.array(qs), np.array(zs), bins).get("bins", [])


@torch.no_grad()
def play_match(model_a: LibraNet, model_b: LibraNet, search_cfg: dict, n_games: int, concurrent: int, threads: int, seed: int,
               device: torch.device, dtype: torch.dtype = torch.float16, log=None, search_cfg_b: dict | None = None) -> dict:
    cfg = dict(search_cfg)
    cfg["full_prob"] = 1.0  # 評価は全読みで固定
    cfg["resign_threshold"] = 0.0  # 計測の対局では投了しない（誤投了が Elo に乗ると物差しが狂う。自己対局だけで使う）
    # 計測の対局は玉を 36×36 から一様に置く（後手玉四段目 25%）。四段目の局は 41 手目の裁定で先手が勝つので得点は両者 0.5 に寄り、
    # その割合が節目ごとに変わると同じ強さの差でも Elo の出方が変わる。自己対局の偏り（gote_rank4_prob）は持ち込まない
    cfg["gote_rank4_prob"] = -1.0
    eng = librasearch.SelfPlay(cfg, concurrent, seed, threads)
    if search_cfg_b is not None:
        cfg_b = {**cfg, **search_cfg_b, "full_prob": 1.0}
        eng.set_side_config(cfg_b)  # B 側の探索設定（変えられるのは読む手の選び方だけ。ほかが違えば例外）
    sq = np.zeros((concurrent, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((concurrent, ls.GLOB_FEATS), np.float32)
    slot_swap = (np.arange(concurrent) % 2).astype(np.int8)  # 奇数枠は B が先手
    games: list[dict] = []
    t0 = time.time()
    last_log = 0
    while len(games) < n_games:
        eng.collect(sq, glob)
        who = eng.root_turns() ^ slot_swap  # 0 なら A のネット、1 なら B
        logits = np.zeros((concurrent, ls.POLICY_SIZE), np.float32)
        wdl = np.zeros((concurrent, 3), np.float32)
        sq_t = torch.from_numpy(sq).to(device).to(dtype)
        gl_t = torch.from_numpy(glob).to(device).to(dtype)
        for k, model in ((0, model_a), (1, model_b)):
            idx = np.flatnonzero(who == k)
            if idx.size == 0:
                continue
            it = torch.from_numpy(idx).to(device)
            p, w, _ = model(sq_t[it], gl_t[it])
            logits[idx] = p.float().cpu().numpy()
            wdl[idx] = F.softmax(w.float(), dim=-1).cpu().numpy()
        eng.apply(logits, wdl)
        games.extend(eng.take_finished())
        if log and len(games) - last_log >= 20:
            last_log = len(games)
            log(f"eval: {len(games)}/{n_games} games ({time.time() - t0:.0f}s)")
    games = games[:n_games]
    # 集計: A が先手なのは偶数枠
    a_sente = {"w": 0, "d": 0, "l": 0}
    a_gote = {"w": 0, "d": 0, "l": 0}
    reasons: dict[str, int] = {}
    for g in games:
        res = int(g["result"])
        a_is_sente = g["slot"] % 2 == 0
        a_res = res if a_is_sente else -res
        key = "w" if a_res > 0 else "l" if a_res < 0 else "d"
        (a_sente if a_is_sente else a_gote)[key] += 1
        reasons[g["reason"]] = reasons.get(g["reason"], 0) + 1
    n = len(games)
    pts = a_sente["w"] + a_gote["w"] + 0.5 * (a_sente["d"] + a_gote["d"])
    score = pts / max(1, n)
    se = math.sqrt(max(score * (1 - score), 1e-6) / max(1, n))

    def a_moved(g: dict, j: int) -> bool:
        return (j % 2 == 0) == (g["slot"] % 2 == 0)

    return {
        "n": n,
        "score_a": round(score, 4),
        "score_se": round(se, 4),
        "elo_a_minus_b": round(elo_diff(score), 1),
        "elo_ci95": [round(elo_diff(max(1e-4, score - 1.96 * se)), 1), round(elo_diff(min(1 - 1e-4, score + 1.96 * se)), 1)],
        "a_as_sente": a_sente,
        "a_as_gote": a_gote,
        "reasons": reasons,
        "avg_plies": round(sum(int(g["plies"]) for g in games) / max(1, n), 1),
        "calibration_a": calibration(games, a_moved),
        "calibration_b": calibration(games, lambda g, j: not a_moved(g, j)),
        "sims": cfg.get("full_sims"),
        "gumbel_noise": bool(cfg.get("gumbel_noise", True)),
        "search_b": {k: v for k, v in (search_cfg_b or {}).items() if cfg.get(k) != v} or None,
        "seconds": round(time.time() - t0, 1),
    }


def main_eval(a: Path, b: Path, search_cfg: dict, n_games: int, concurrent: int, threads: int, seed: int, out: Path | None,
              search_cfg_b: dict | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ma = load_model(a, device)
    mb = load_model(b, device)
    res = play_match(ma, mb, search_cfg, n_games, concurrent, threads, seed, device, log=lambda s: print(s, flush=True),
                     search_cfg_b=search_cfg_b)
    res["a"] = str(a)
    res["b"] = str(b)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res
