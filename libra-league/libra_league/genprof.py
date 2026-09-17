# SPDX-License-Identifier: Apache-2.0
"""一般化の物差し（docs/restart-plan.md §3 M1、decisions.md 2026-09-17）。

ネットを「学習に使った窓の局面」と「学習に使っていない held-out の局面」で同じ物差しで測り、差を出す。
物差し: 価値 V = W − L と学習目標 t の相関・二乗誤差、方策の交差エントロピー（上位 32 の目標）・一致率。布石と本将棋で分ける。
窓の中だけ良ければ、ネットは窓の局の結果を記憶していて一般化していない（2026-09-17 の診断: 本将棋の価値の相関が窓の中 0.94・外 0.43）。
`libra genprof` は保存済みの重みを任意のチャンクで測る（run をまたいだ比較用）。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from libra_net.model import LibraNet

from .replay import sample_batch

PHASES = ("fuseki", "normal")


@torch.no_grad()
def measure(model: LibraNet, batch: dict, device: torch.device, chunk: int = 512) -> dict:
    """バッチ（replay.sample の形）をネットに通し、布石・本将棋ごとの物差しを返す。"""
    was_training = model.training
    model.eval()
    sq = torch.from_numpy(batch["sq"]).to(device)
    glob = torch.from_numpy(batch["glob"]).to(device)
    P, W = [], []
    for i in range(0, sq.shape[0], chunk):
        p, w, _ = model(sq[i:i + chunk], glob[i:i + chunk])
        P.append(p.float().cpu())
        W.append(w.float().cpu())
    if was_training:
        model.train()
    P = torch.cat(P)
    W = torch.cat(W)
    wp = F.softmax(W, -1)
    v = (wp[:, 0] - wp[:, 2]).numpy()
    t = batch["wdl"][:, 0] - batch["wdl"][:, 2]
    logp = F.log_softmax(P, -1)
    pidx = torch.from_numpy(batch["policy_idx"])
    pp = torch.from_numpy(batch["policy_p"])
    ce = -(torch.gather(logp, 1, pidx.clamp(min=0)) * pp).sum(-1).numpy()
    acc = (P.argmax(-1) == pidx[:, 0]).numpy()
    valid = batch["policy_valid"]
    fu = batch["fuseki"].astype(bool)
    out: dict = {}
    for name, mask in (("fuseki", fu), ("normal", ~fu)):
        n = int(mask.sum())
        mv = mask & valid
        row = {"n": n, "n_policy": int(mv.sum())}
        if n >= 2 and float(np.std(t[mask])) > 0 and float(np.std(v[mask])) > 0:
            row["corr_v"] = round(float(np.corrcoef(v[mask], t[mask])[0, 1]), 4)
        else:
            row["corr_v"] = None
        row["mse_v"] = round(float(np.mean((v[mask] - t[mask]) ** 2)), 4) if n else None
        row["policy_ce"] = round(float(ce[mv].mean()), 4) if mv.any() else None
        row["policy_acc"] = round(float(acc[mv].mean()), 4) if mv.any() else None
        out[name] = row
    return out


def generalization(model: LibraNet, rb, n: int, rng: np.random.Generator, device: torch.device, lambda_z: float, topk: int) -> dict | None:
    """窓の中と held-out で measure を取り、差（held-out − 窓）を付けて返す。held-out が無ければ None。"""
    if rb.n_heldout() <= 0 or rb.n_games() <= 0:
        return None
    win = measure(model, rb.sample(n, rng, 0.0, lambda_z, topk), device)
    held = measure(model, rb.sample_heldout(n, rng, 0.0, lambda_z, topk), device)
    gap = {}
    for ph in PHASES:
        gap[ph] = {k: (round(held[ph][k] - win[ph][k], 4) if isinstance(held[ph].get(k), float) and isinstance(win[ph].get(k), float) else None)
                   for k in ("corr_v", "mse_v", "policy_ce", "policy_acc")}
    return {"window": win, "heldout": held, "gap": gap, "window_games": rb.n_games(), "heldout_games": rb.n_heldout(), "positions": n}


def load_chunks(replay_dir: Path, start: int, n_chunks: int) -> list[dict]:
    games: list[dict] = []
    for i in range(start, start + n_chunks):
        p = replay_dir / f"chunk_{i:06d}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                games.extend(pickle.load(f))
    return [g for g in games if "league_opponent" not in g]


def profile(model: LibraNet, replay_dir: Path, starts: list[int], n_chunks: int, n: int, device: torch.device, lambda_z: float, topk: int,
            max_ply: int, count_from_41: bool, seed: int = 0) -> list[dict]:
    """チャンク starts のそれぞれ n_chunks 個から n 局面を取って measure する（`libra genprof`）。"""
    rows = []
    for s in starts:
        games = load_chunks(replay_dir, s, n_chunks)
        if not games:
            rows.append({"chunk": s, "games": 0})
            continue
        cum = np.cumsum(np.array([len(g["moves"]) for g in games], dtype=np.int64))
        b = sample_batch(games, 0, cum, n, np.random.default_rng(seed), 0.0, lambda_z, topk, max_ply, count_from_41)
        rows.append({"chunk": s, "games": len(games), **measure(model, b, device)})
    return rows


def format_rows(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        if not r.get("games"):
            lines.append(f"chunk {r['chunk']:6d}: no games")
            continue
        parts = [f"chunk {r['chunk']:6d} ({r['games']} games)"]
        for ph in PHASES:
            x = r[ph]
            parts.append(f"{ph}: corr_v {x['corr_v']} mse_v {x['mse_v']} policy_ce {x['policy_ce']} acc {x['policy_acc']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def format_gen(g: dict | None) -> str:
    if not g:
        return "gen: (no held-out yet)"
    parts = []
    for ph in PHASES:
        w, h = g["window"][ph], g["heldout"][ph]
        parts.append(f"{ph} corr_v {w['corr_v']}→{h['corr_v']} mse_v {w['mse_v']}→{h['mse_v']} policy_ce {w['policy_ce']}→{h['policy_ce']}")
    return f"gen (window→heldout, {g['window_games']}/{g['heldout_games']} games): " + " | ".join(parts)
