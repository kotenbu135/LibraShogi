# SPDX-License-Identifier: Apache-2.0
"""scale.json の生成（docs/libra-design.md §5 の 2〜4）。

1. 剪定済み 972 ペア（鏡映で 492 通り）の 3 手目局面を外部駆動の MCGS で読み、先手の勝率 V̂ を得る
2. 自己対局の棋譜（games/*.jsonl、一様サンプル）からペアごとの実測勝率と信頼区間を足す
3. 釣り合い集合 balanced: |V̂ − 0.5| が最小値 + margin 以内（検証対局後は verify.py の信頼区間で決め直す）

置く側は balanced から一様に選び、選ぶ側は 2 手目後の局面の winrate（先手の勝率）が 0.5 以上なら先手を取る。
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
from libra_league.config import DEFAULTS

from .pairs import canonical, from_usi, mirror_sq, unique_pairs, usi

SCALE_VERSION = 0


def wilson(score: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """勝率（引き分け 0.5）の 95% 区間（正規近似）。n=0 なら [0,1]。"""
    if n <= 0:
        return (0.0, 1.0)
    p = score
    h = z * math.sqrt(max(p * (1 - p), 1e-9) / n)
    return (max(0.0, p - h), min(1.0, p + h))


@torch.no_grad()
def search_values(model, pairs: list[tuple[int, int]], sims: int, concurrent: int, threads: int, seed: int,
                  device: torch.device, dtype: torch.dtype = torch.float16, log=None) -> dict[tuple[int, int], float]:
    """各ペアの 3 手目局面（先手番）を sims 回読み、先手の勝率 V̂（ルート値を 0..1 に直したもの）を返す。"""
    cfg = dict(DEFAULTS["search"])
    cfg.update({"external": True, "full_prob": 1.0, "policy_topk": 8, "mate_nodes_root": 0})
    n_slots = max(1, min(concurrent, len(pairs)))
    eng = librasearch.SelfPlay(cfg, n_slots, seed, threads)
    sq = np.zeros((n_slots, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((n_slots, ls.GLOB_FEATS), np.float32)
    queue = list(pairs)
    slot_pair: dict[int, tuple[int, int]] = {}
    out: dict[tuple[int, int], float] = {}
    t0 = time.time()
    while queue or slot_pair:
        for s in range(n_slots):
            if s not in slot_pair and queue:
                kb, kw = queue.pop()
                assert eng.set_position(s, f"position fuseki moves K*{usi(kb)} K*{usi(kw)}", sims, True)
                slot_pair[s] = (kb, kw)
        eng.collect(sq, glob)
        p, w, _ = model(torch.from_numpy(sq).to(device).to(dtype), torch.from_numpy(glob).to(device).to(dtype))
        eng.apply(np.ascontiguousarray(p.float().cpu().numpy()), np.ascontiguousarray(F.softmax(w.float(), dim=-1).cpu().numpy()))
        for s in list(slot_pair):
            if eng.idle(s):
                r = eng.result(s)
                out[slot_pair.pop(s)] = (float(r["root_q"]) + 1) / 2
        if log and len(out) and len(out) % 100 == 0:
            log(f"scale: {len(out)}/{len(pairs)} pairs ({time.time() - t0:.0f}s)")
    return out


def measured_winrates(games_dir: Path) -> dict[tuple[int, int], list[int]]:
    """自己対局の棋譜からペアごとの [局数, 先手勝ち, 引き分け, 後手勝ち]（鏡映は代表に合算）。"""
    counts: dict[tuple[int, int], list[int]] = {}
    for f in sorted(Path(games_dir).glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    g = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = g.get("tokens", "").split()
                if len(t) < 2 or not t[0].startswith("K*") or not t[1].startswith("K*"):
                    continue
                key = canonical(from_usi(t[0][2:]), from_usi(t[1][2:]))
                c = counts.setdefault(key, [0, 0, 0, 0])
                c[0] += 1
                c[{"sente": 1, "draw": 2, "gote": 3}.get(g.get("result"), 2)] += 1
    return counts


def balanced_from_vhat(entries: list[dict], margin: float) -> list[dict]:
    d = [abs(e["v_hat"] - 0.5) for e in entries]
    m = min(d)
    return [e for e, x in zip(entries, d) if x <= m + margin]


def expand_mirrors(pairs: list[tuple[int, int]]) -> list[list[str]]:
    out: list[list[str]] = []
    seen: set[tuple[int, int]] = set()
    for kb, kw in pairs:
        for p in ((kb, kw), (mirror_sq(kb), mirror_sq(kw))):
            if p not in seen:
                seen.add(p)
                out.append([usi(p[0]), usi(p[1])])
    return out


def build_table(model, model_info: dict, sims: int, concurrent: int, threads: int, seed: int, device: torch.device,
                games_dir: Path | None, margin: float = 0.02, dtype: torch.dtype = torch.float16, log=None,
                pairs: list[tuple[int, int]] | None = None) -> dict:
    pairs = pairs or unique_pairs()
    vh = search_values(model, pairs, sims, concurrent, threads, seed, device, dtype, log)
    counts = measured_winrates(games_dir) if games_dir else {}
    entries = []
    for kb, kw in pairs:
        c = counts.get((kb, kw), [0, 0, 0, 0])
        score = (c[1] + 0.5 * c[2]) / c[0] if c[0] else None
        entries.append({
            "kb": usi(kb), "kw": usi(kw), "mirror": [usi(mirror_sq(kb)), usi(mirror_sq(kw))],
            "v_hat": round(vh[(kb, kw)], 4),
            "selfplay": {"games": c[0], "sente": c[1], "draw": c[2], "gote": c[3],
                         "winrate": None if score is None else round(score, 4),
                         "ci95": None if score is None else [round(x, 4) for x in wilson(score, c[0])]},
        })
    bal = balanced_from_vhat(entries, margin)
    return {
        "version": SCALE_VERSION,
        "license": "CC0-1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_info,
        "sims": sims,
        "n_pairs_pruned": 972,
        "n_pairs_unique": len(pairs),
        "pruned": "後手玉が四段目のペアは 3 手目の桂打ちで先手の裁定勝ち（libra_scale/pairs.py）",
        "balance_rule": f"|v_hat - 0.5| <= min + {margin}（検証対局後は verify の信頼区間で決め直す）",
        "pairs": entries,
        "balanced": expand_mirrors([(from_usi(e["kb"]), from_usi(e["kw"])) for e in bal]),
    }


def load_model(path: Path, device: torch.device, dtype: torch.dtype):
    from libra_net.model import LibraNet, NetConfig

    sd = torch.load(path, map_location=device, weights_only=False)
    m = LibraNet(NetConfig.from_dict(sd.get("config", {}).get("net", {}))).to(device)
    m.load_state_dict(sd["model"])
    m.eval()
    if device.type == "cuda":
        m = m.to(dtype)
    return m, {"path": str(path), "step": int(sd.get("step", 0))}
