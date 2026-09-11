# SPDX-License-Identifier: Apache-2.0
"""実行設定。TOML（config.toml）で上書きする。既定は Libra-L の L-S 構成（docs/libra-local.md §3.2）。"""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "run_id": "ls",
    "seed": 1,
    "net": {"d_model": 320, "n_layers": 8, "n_heads": 8, "d_ff": 1280, "dropout": 0.0},  # 約 10M（L-S）
    "search": {
        "full_sims": 96,
        "fast_sims": 24,
        "full_prob": 0.25,
        "gumbel_m_full": 16,
        "gumbel_m_fast": 8,
        "c_visit": 50.0,
        "c_scale": 1.0,
        "cpuct": 1.5,
        "draw_value": 0.0,
        "max_ply": 256,
        "count_from_41": True,
        "policy_topk": 32,
        "max_moves_per_game": 400,
        "mate_nodes_root": 200,
        "proof_nodes": 1000,
        "proof_min_ply": 36,
    },
    "selfplay": {"n_games": 512, "threads": 12, "infer_dtype": "float16"},
    "train": {
        "batch_size": 1024,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 1000,
        "replay_ratio": 4.0,
        "window_games": 100000,
        "min_window_games": 2000,
        "train_every_games": 256,
        "policy_weight": 1.0,
        "value_weight": 1.0,
        "v41_weight": 0.5,
        "lambda_z": 0.5,
        "mirror_prob": 0.5,
        "grad_clip": 1.0,
    },
    "run": {"checkpoint_minutes": 10, "status_seconds": 30, "chunk_games": 100, "keep_checkpoints": 3, "archive_every_steps": 50000},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None and path.exists():
        with open(path, "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    return cfg


def dump_toml(cfg: dict[str, Any]) -> str:
    """設定を TOML 文字列にする（state ディレクトリへ写しを残す用。値は基本型だけ）。"""
    lines: list[str] = []

    def fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        return '"' + str(v).replace('"', '\\"') + '"'

    for k, v in cfg.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {fmt(v)}")
    for k, v in cfg.items():
        if isinstance(v, dict):
            lines.append(f"\n[{k}]")
            for k2, v2 in v.items():
                lines.append(f"{k2} = {fmt(v2)}")
    return "\n".join(lines) + "\n"
