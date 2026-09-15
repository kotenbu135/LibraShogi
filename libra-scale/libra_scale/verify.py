# SPDX-License-Identifier: Apache-2.0
"""検証対局（docs/libra-design.md §5 の 3）: 釣り合い候補のペアに限定した自己対局で実測勝率と信頼区間を出し、
釣り合い集合を「|w − 0.5| の信頼区間が最小値と重なるペア」に決め直す。v0 は各ペア同数の局を回す（逐次棄却は後で）。"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

import librasearch
import librashogi as ls
from libra_league.config import DEFAULTS

from .pairs import canonical, from_usi, mirror_sq, usi
from .table import expand_mirrors, wilson


def sampling_pairs(pairs: list[tuple[int, int]]) -> list[list[int]]:
    """自己対局の抽選の一覧（[kb, kw] の並び）。自己対局は一覧から一様に選ぶので、代表ごとに代表と鏡映を 1 回ずつ入れる。
    鏡映が自分と同じ組（両玉が 5 筋）は同じ組が 2 回入り、他の組と同じ確率で当たる。2026-09-15 までは鏡映を省いていたため
    5 筋の組が半分しか当たらず、全組が games_per_pair に届くまで回す verify の局数が延びた（v0.1 の上位 48 組・100 局で、
    模擬の中央値が約 10,200 局。直した後は約 5,950 局。measurements.md 同日）。"""
    return [[a, b] for a, b in pairs] + [[mirror_sq(a), mirror_sq(b)] for a, b in pairs]


@torch.no_grad()
def verify_pairs(model, pairs: list[tuple[int, int]], games_per_pair: int, sims: int, concurrent: int, threads: int,
                 seed: int, device: torch.device, dtype: torch.dtype = torch.float16, log=None,
                 search_overrides: dict | None = None) -> dict[tuple[int, int], list[int]]:
    """pairs（代表）に限定して自己対局し、ペアごとの [局数, 先手勝ち, 引き分け, 後手勝ち] を返す。鏡映も混ぜて代表に合算。"""
    cfg = dict(DEFAULTS["search"])
    cfg.update({"full_prob": 1.0, "full_sims": sims, "policy_topk": 8})
    if search_overrides:
        cfg.update(search_overrides)
    kp = sampling_pairs(pairs)
    cfg["king_pairs"] = kp
    n_slots = max(1, min(concurrent, len(kp) * 4))
    eng = librasearch.SelfPlay(cfg, n_slots, seed, threads)
    sq = np.zeros((n_slots, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((n_slots, ls.GLOB_FEATS), np.float32)
    counts: dict[tuple[int, int], list[int]] = {p: [0, 0, 0, 0] for p in pairs}
    need = len(pairs) * games_per_pair
    done = 0
    t0 = time.time()
    last = 0
    while min(c[0] for c in counts.values()) < games_per_pair:
        eng.collect(sq, glob)
        p, w, _ = model(torch.from_numpy(sq).to(device).to(dtype), torch.from_numpy(glob).to(device).to(dtype))
        eng.apply(np.ascontiguousarray(p.float().cpu().numpy()), np.ascontiguousarray(F.softmax(w.float(), dim=-1).cpu().numpy()))
        for g in eng.take_finished():
            key = canonical(int(g["kb"]), int(g["kw"]))
            if key not in counts:
                continue
            c = counts[key]
            c[0] += 1
            c[1 if g["result"] > 0 else 2 if g["result"] == 0 else 3] += 1
            done += 1
        if log and done - last >= 200:
            last = done
            log(f"verify: {done} games (target {need}) {time.time() - t0:.0f}s")
    return counts


def apply_verification(table: dict, counts: dict[tuple[int, int], list[int]], sims: int, games_per_pair: int) -> dict:
    """table の pairs に verify を書き込み、balanced を信頼区間の規則で決め直す。"""
    ent = {(from_usi(e["kb"]), from_usi(e["kw"])): e for e in table["pairs"]}
    scored = []
    diffs = []
    for key, c in counts.items():
        e = ent.get(key)
        if e is None or c[0] == 0:
            continue
        w = (c[1] + 0.5 * c[2]) / c[0]
        lo, hi = wilson(w, c[0])
        e["verify"] = {"games": c[0], "sente": c[1], "draw": c[2], "gote": c[3], "winrate": round(w, 4),
                       "ci95": [round(lo, 4), round(hi, 4)], "sims": sims}
        scored.append((key, abs(w - 0.5), (hi - lo) / 2))
        if "v_hat" in e:
            diffs.append(float(e["v_hat"]) - w)
    if scored:
        m = min(d for _, d, _ in scored)
        bal = [key for key, d, h in scored if d - h <= m]
        table["balanced"] = expand_mirrors(bal)
        table["balance_rule"] = f"検証対局 {games_per_pair} 局/ペア（sims {sims}）: |w - 0.5| - 半幅 <= min |w - 0.5|"
        table["verify"] = {"pairs": len(scored), "games_per_pair": games_per_pair, "sims": sims}
        if diffs:
            # V̂ の偏りの確認（docs/method-evidence.md §4.4 (b)）。標準誤差は不偏分散から
            mu = sum(diffs) / len(diffs)
            var = sum((d - mu) ** 2 for d in diffs) / (len(diffs) - 1) if len(diffs) > 1 else 0.0
            table["verify"]["v_hat_minus_w"] = round(mu, 4)
            table["verify"]["v_hat_minus_w_se"] = round((var / len(diffs)) ** 0.5, 4)
    return table
