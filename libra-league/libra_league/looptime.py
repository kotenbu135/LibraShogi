# SPDX-License-Identifier: Apache-2.0
"""学習器のループ（runner.py）の処理時間の内訳。

metrics の 1 行ぶん（既定 5 分）の実時間を段に分けて累計し、足りない分を other に入れて合計を窓の長さに揃える。
自己対局とリーグの collect / eval / proof / apply は SelfPlayLoop.timing（workers.py の perf と同じ区切り）から取る。
eval は GPU の評価の完了待ちなので、同じ GPU を使う別のプロセス（ls と lx）の待ちもここに入る。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator

# 表示の順。値は秒
PHASES = (
    "sp_collect",     # 自己対局: 葉を集めて特徴量を作る（CPU）
    "sp_eval",        # 自己対局: ネットの評価の待ち（GPU。proof の時間は除く）
    "sp_proof",       # 自己対局: 根の証明探索（CPU。GPU の評価と重なる）
    "sp_apply",       # 自己対局: 評価を木に戻して手を進める（CPU）
    "lg_collect",     # リーグ（対 lx）: 同上
    "lg_eval",
    "lg_proof",
    "lg_apply",
    "lg_other",       # リーグ: 相手の切り替え（プールの重みの読み込み）ほか
    "replay",         # 終局した局をリプレイと棋譜に書く
    "ingest",         # ワーカーの局の取り込み
    "train_sample",   # 学習: バッチ作り（CPU）の待ち
    "train_step",     # 学習: ステップ（GPU）
    "train_publish",  # 学習: 推論用ネットへの反映とワーカー用の重みの書き出し
    "housekeeping",   # 搾取者の本体の更新・布石の書き出しと読み込み
    "status",         # status.json・metrics・較正・自動計測の確認
    "checkpoint",     # チェックポイント（ONNX の書き出しと archive を含む）
    "other",          # 残り（上のどれにも入らない時間）
)
PARTS = ("collect", "eval", "proof", "apply")


class LoopTimer:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.start = clock()
        self.sec: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, key: str, sec: float) -> None:
        self.sec[key] = self.sec.get(key, 0.0) + sec

    def count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    @contextmanager
    def phase(self, key: str) -> Iterator[None]:
        t0 = self.clock()
        try:
            yield
        finally:
            self.add(key, self.clock() - t0)

    def take(self, sp: dict | None, lg: dict | None) -> dict:
        """窓を閉じて内訳を返し、0 から数え直す。sp・lg は SelfPlayLoop.timing（呼ぶ側で空にする）。
        lg を渡すときは add("league", リーグの 1 ラウンドの実時間) を足しておく（その差が lg_other）。"""
        now = self.clock()
        window = now - self.start
        sec = {k: 0.0 for k in PHASES}
        for k, v in self.sec.items():
            if k in sec:
                sec[k] += v
        sp, lg = sp or {}, lg or {}
        for p in PARTS:
            sec["sp_" + p] += float(sp.get(p, 0.0))
            sec["lg_" + p] += float(lg.get(p, 0.0))
        league_wall = self.sec.get("league", 0.0)
        sec["lg_other"] = max(0.0, league_wall - sum(float(lg.get(p, 0.0)) for p in PARTS))
        sec["other"] = max(0.0, window - sum(v for k, v in sec.items() if k != "other"))
        out = {
            "window_s": round(window, 1),
            "rounds": int(sp.get("rounds", 0)),
            "league_rounds": int(lg.get("rounds", 0)),
            "train_steps": int(self.counts.get("train_steps", 0)),
            "sec": {k: round(v, 3) for k, v in sec.items()},
        }
        self.start = now
        self.sec = {}
        self.counts = {}
        return out


def format_timing(tm: dict | None) -> str:
    """status の 1 行: 窓に占める割合（0.05% 未満は省く）と、自己対局 1 ラウンド・学習 1 ステップの ms。"""
    if not tm or not tm.get("window_s"):
        return ""
    w = float(tm["window_s"])
    sec = tm.get("sec") or {}
    shares = "  ".join(f"{k} {sec[k] / w * 100:.1f}%" for k in PHASES if sec.get(k, 0.0) / w >= 0.0005)
    s = f"window {w:.0f}s  {shares}"
    r = int(tm.get("rounds") or 0)
    if r:
        s += f"  |  round {r / w:.1f}/s (" + ", ".join(f"{p} {sec.get('sp_' + p, 0.0) / r * 1000:.1f} ms" for p in PARTS) + ")"
    n = int(tm.get("train_steps") or 0)
    if n:
        s += f"  train {sec.get('train_step', 0.0) / n * 1000:.1f} ms/step"
    return s
