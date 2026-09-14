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
        "max_ply": 320,
        "count_from_41": True,
        "policy_topk": 32,
        "max_moves_per_game": 400,
        "mate_nodes_root": 200,
        "proof_nodes": 1000,
        "proof_min_ply": 36,
        "defer_root_proof": True,  # 根の証明探索を GPU の評価中に解く（棋譜は変わらない。false で apply の中で解く）
    },
    "selfplay": {
        "n_games": 512,
        "threads": 12,
        "infer_dtype": "float16",
        "compile": "max-autotune",  # 推論の捕獲: none | default | max-autotune（CUDA のときだけ効く。CUDA Graphs は常に使う）
        "openings": "",            # 搾取者が見つけた布石（openings.json）。空なら使わない
        "openings_prob": 0.1,      # 新規対局が openings から始まる確率
        "openings_reload_seconds": 600,
    },
    # 搾取者（docs/libra-design.md §4.2）: main_ckpt を凍結した相手にして、自分の手だけを学習する。
    # 本体が強くなると凍結相手が古すぎて勝率が飽和するので、refresh_hours ごと、または main_source が
    # refresh_steps 以上先へ進んだら main_source で作り直し、成績と布石をそこで区切る
    # （収束判定「対本体勝率が頭打ち」は現行の本体に対してでないと意味がないため）。
    "exploiter": {"main_ckpt": "", "main_dtype": "float16",
                  "main_source": "",         # 作り直しの元（本体ランの checkpoints/latest.pt）。空なら作り直さない
                  "refresh_hours": 0.0,      # 作り直しの間隔（0 で無効。初回は起動直後に行う）
                  "refresh_steps": 0,        # main_source の step が凍結相手より何 step 進んだら作り直すか（0 で無効）
                  "refresh_check_minutes": 5.0,  # refresh_steps のために main_source の step を読む間隔
                  # 搾取者の手の探索木の中で、本体の手番の葉の方策を凍結した本体のネットから取る（価値は常に搾取者のネット）。
                  # 本体の応手を本体の方策で予測する（Wang+ 2023 の A-MCTS に倣う。docs/exploiter-literature.md）。false で木を丸ごと根の手番のネットで評価
                  "opponent_prior": True,
                  "openings_out": "",        # 見つけた布石の書き出し先（空なら書かない）。本体はこれを読む
                  "openings_minutes": 60.0,
                  "openings_chunks": 50,
                  "openings_moves": 12},
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
    # 自己対局ワーカー（libra worker、libra_league/workers.py）: 学習側は重みを <run>/weights/latest.pt に配り（学習のたび）、
    # <run>/inbox/ に届いた局を ingest_seconds ごとに手元の自己対局と同じようにリプレイへ足す。既定は無効（今の 1 プロセスのまま）。
    # max_lag_steps: 局を打った重みが学習側より何 step 遅れていたら捨てるか（0 で捨てない）。搾取者の run では無効。
    "workers": {"enabled": False, "ingest_seconds": 10.0, "max_lag_steps": 2000},
    "run": {"checkpoint_minutes": 10, "status_seconds": 30, "chunk_games": 100, "keep_checkpoints": 3,
            "export_onnx": True,   # チェックポイントごとに latest.onnx も書く（libra / libra.exe 用）
            "metrics_minutes": 5},  # metrics.jsonl（進捗の時系列）の追記間隔
    # 自動計測（docs/runbook.md §6）: every_hours ごとにチェックポイントを archive に残し、直前の archive と対局させて Elo を鎖にする。
    # match_games > 0 なら外部エンジン（fuseki_usi_server.py）とも少数局を指す。どちらも別プロセスで GPU を共有する。
    # anchor_*: 固定の基準ネットとの対局。連続世代どうしの Elo は伸びが測定幅（100 局で ±70 Elo）に埋もれ、
    # 鎖にすると誤差が回数の平方根で積み上がる。基準との差は大きいままなので信号が残り、誤差も積み上がらない。
    # 基準に対する勝率が anchor_rebaseline を超えたら基準を新しい世代に置き換え、それまでの差を offset に足す。
    "auto": {"enabled": False, "every_hours": 24.0, "eval_games": 100, "eval_sims": 96, "eval_concurrent": 64, "eval_threads": 4,
             "chain_eval": True, "anchor_games": 100, "anchor_rebaseline": 0.85,
             "match_games": 10, "match_go": "movetime 1000", "match_opponent_opt": "Threads=2"},
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
