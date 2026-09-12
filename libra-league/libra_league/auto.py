# SPDX-License-Identifier: Apache-2.0
"""進捗の可視化と自動計測（docs/runbook.md §6）。

- metrics.jsonl: status を数分ごとに 1 行ずつ追記した時系列（学習指標・終局内訳の累積カウンタ）。
- checkpoints/archive/: 一定時間ごとに残すチェックポイント。連続する 2 つを対局させて Elo を鎖でつなぐ。
- AutoJobs: ランナーが別プロセスで `libra eval`（自己評価）と `libra match`（外部エンジンとの計測）を起動する。
  GPU は共有で、既定は 1 日 1 回・少数局（自己対局の局/日をほぼ落とさない）。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .state import StateDir

_CKPT_RE = re.compile(r"ckpt_(\d+)\.pt$")

METRIC_ENGINE_KEYS = ("games", "moves", "sims", "sente_wins", "draws", "gote_wins", "ruling41", "no_legal_move",
                      "sennichite", "perpetual_check", "max_ply", "plies_sum", "mate_found", "proof_found")


def ckpt_step(p: Path | str) -> int | None:
    m = _CKPT_RE.search(str(p))
    return int(m.group(1)) if m else None


# ---- metrics.jsonl ----
def metrics_row(status: dict) -> dict:
    eng = status.get("engine") or {}
    row = {
        "t": round(time.time(), 1),
        "time": status.get("time"),
        "step": status.get("step"),
        "generation": status.get("generation"),
        "games_total": status.get("games_total"),
        "gpd": status.get("games_per_day_1h"),
        "active": status.get("active_games"),
        "window": status.get("window_games"),
        "train": {k: status["train"].get(k) for k in ("loss", "policy", "value", "v41", "policy_acc", "lr")} if status.get("train") else None,
        "engine": {k: eng.get(k) for k in METRIC_ENGINE_KEYS} if eng else None,
        "exploiter": {k: status["exploiter"].get(k) for k in ("games", "wins", "draws", "losses")} if status.get("exploiter") else None,
        "gpu_mb": (status.get("gpu") or {}).get("mem_reserved_mb"),
    }
    return row


def append_metrics(sd: StateDir, status: dict) -> None:
    with open(sd.root / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics_row(status), ensure_ascii=False) + "\n")


def load_metrics(sd: StateDir, max_points: int = 600) -> list[dict]:
    p = sd.root / "metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if len(rows) > max_points:
        stride = -(-len(rows) // max_points)
        rows = rows[::stride] + ([rows[-1]] if (len(rows) - 1) % stride else [])
    return rows


# ---- チェックポイントの保管（archive） ----
def archive_dir(sd: StateDir) -> Path:
    return sd.checkpoints / "archive"


def list_archives(sd: StateDir) -> list[Path]:
    d = archive_dir(sd)
    if not d.exists():
        return []
    return sorted((p for p in d.glob("ckpt_*.pt") if ckpt_step(p) is not None), key=ckpt_step)


def archive_checkpoint(sd: StateDir, ckpt: Path) -> Path | None:
    """ckpt を archive/ に写す（同じ step が既にあれば何もしない）。"""
    step = ckpt_step(ckpt)
    if step is None:
        return None
    d = archive_dir(sd)
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"ckpt_{step:09d}.pt"
    if dst.exists():
        return None
    tmp = dst.with_suffix(".tmp")
    shutil.copyfile(ckpt, tmp)
    os.replace(tmp, dst)
    return dst


# ---- 結果の集計（status --json --history 用） ----
def collect_evals(sd: StateDir) -> list[dict]:
    """eval/*.json を b の step 順に並べ、archive 同士の連続ペアなら Elo を累積する。"""
    out = []
    d = sd.root / "eval"
    if not d.exists():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sa, sb = ckpt_step(r.get("a", "")), ckpt_step(r.get("b", ""))
        out.append({
            "file": p.name, "time": p.stat().st_mtime, "step_a": sa, "step_b": sb, "n": r.get("n"),
            "elo": r.get("elo_a_minus_b"), "ci95": r.get("elo_ci95"), "score_a": r.get("score_a"),
            "seconds": r.get("seconds"), "auto": p.name.startswith("auto-"),
        })
    out.sort(key=lambda e: (e["step_b"] if e["step_b"] is not None else -1, e["time"]))
    # 鎖: a→b の Elo（b − a = −elo_a_minus_b）を、a が直前の鎖の末尾と一致する限り足す
    cum = 0.0
    tail: int | None = None
    for e in out:
        if e["elo"] is None or e["step_a"] is None or e["step_b"] is None or not e["auto"]:
            e["cumulative"] = None
            continue
        if tail is None or e["step_a"] == tail:
            cum += -float(e["elo"])
            tail = e["step_b"]
            e["cumulative"] = round(cum, 1)
        else:
            e["cumulative"] = None
    return out


def collect_matches(sd: StateDir) -> list[dict]:
    out = []
    d = sd.root / "matches"
    if not d.exists():
        return out
    for p in sorted(d.glob("*.summary.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        n = int(r.get("n") or 0)
        out.append({
            "file": p.name, "time": p.stat().st_mtime, "n": n, "a_points": r.get("a_points"),
            "winrate": (round(float(r.get("a_points", 0.0)) / n, 4) if n else None),
            "opponent": r.get("b"), "go": r.get("go"), "reasons": r.get("reasons"),
            "libra_step": ckpt_step(str((r.get("libra_options") or {}).get("DNN_Model", ""))),
            "auto": p.name.startswith("auto-"),
        })
    out.sort(key=lambda m: m["time"])
    return out


# ---- 自動ジョブ ----
class AutoJobs:
    """eval / match を順に 1 つずつ別プロセスで回す。状態は state["auto"] に持つ（再開しても続く）。"""

    def __init__(self, sd: StateDir, cfg: dict, state: dict, log, cmd_prefix: list[str] | None = None):
        self.sd = sd
        self.cfg = cfg
        self.acfg = cfg.get("auto", {})
        self.state = state
        self.log = log
        self.cmd_prefix = cmd_prefix if cmd_prefix is not None else [sys.executable, "-m", "libra_league.cli", "--root", str(sd.root.parent), "--run", sd.root.name]
        self.proc: subprocess.Popen | None = None
        self.current: dict | None = None
        self._st()

    def _st(self) -> dict:
        """state["auto"]（load() で state が置き換わっても欠けたキーを補う）。"""
        st = self.state.setdefault("auto", {})
        for k, v in (("last_archive", None), ("last_match", None), ("queue", []), ("history", []), ("running", None)):
            st.setdefault(k, v)
        return st

    @property
    def enabled(self) -> bool:
        return bool(self.acfg.get("enabled", False))

    # -- 起票 --
    def on_checkpoint(self, ckpt: Path, now: float | None = None) -> None:
        """チェックポイント直後に呼ぶ。archive の期限と EVAL_NOW / MATCH_NOW フラグを見てジョブを積む。"""
        if not self.enabled:
            return
        now = now or time.time()
        st = self._st()
        every = float(self.acfg.get("every_hours", 24.0)) * 3600
        eval_now = self.sd.flag("EVAL_NOW")
        if eval_now or st["last_archive"] is None or now - st["last_archive"] >= every:
            prev = list_archives(self.sd)
            new = archive_checkpoint(self.sd, ckpt)
            st["last_archive"] = now
            if new is not None:
                self.log(f"auto: archived {new.name}")
                if prev:
                    self.enqueue_eval(prev[-1], new)
            if eval_now:
                self.sd.clear_flag("EVAL_NOW")
        match_now = self.sd.flag("MATCH_NOW")
        if int(self.acfg.get("match_games", 0)) > 0 and (match_now or st["last_match"] is None or now - st["last_match"] >= every):
            self.enqueue_match()
            st["last_match"] = now
            if match_now:
                self.sd.clear_flag("MATCH_NOW")

    def enqueue_eval(self, a: Path, b: Path) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = self.sd.root / "eval" / f"auto-{ts}-{ckpt_step(a)}-{ckpt_step(b)}.json"
        args = ["eval", "--a", str(a), "--b", str(b), "--games", str(self.acfg.get("eval_games", 100)),
                "--sims", str(self.acfg.get("eval_sims", 96)), "--concurrent", str(self.acfg.get("eval_concurrent", 64)),
                "--threads", str(self.acfg.get("eval_threads", 4)), "--out", str(out)]
        self.state["auto"]["queue"].append({"kind": "eval", "args": args, "out": str(out)})
        self.log(f"auto: queued eval {a.name} vs {b.name}")

    def enqueue_match(self) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = self.sd.root / "matches" / f"auto-{ts}.jsonl"
        args = ["match", "--games", str(self.acfg.get("match_games", 10)), "--go", str(self.acfg.get("match_go", "movetime 1000")), "--out", str(out)]
        for kv in str(self.acfg.get("match_opponent_opt", "")).split(","):
            if kv.strip():
                args += ["--opponent-opt", kv.strip()]
        self.state["auto"]["queue"].append({"kind": "match", "args": args, "out": str(out)})
        self.log("auto: queued match")

    # -- 実行 --
    def poll(self) -> None:
        """定期的に呼ぶ。走っているジョブの終了を回収し、待ちがあれば次を起動する。"""
        st = self._st()
        if self.proc is not None:
            rc = self.proc.poll()
            if rc is None:
                return
            job = self.current or {}
            job["finished"] = time.time()
            job["rc"] = rc
            st["history"] = (st["history"] + [job])[-20:]
            st["running"] = None
            self.log(f"auto: {job.get('kind')} finished rc={rc} ({job['finished'] - job.get('started', job['finished']):.0f}s)")
            self.proc = None
            self.current = None
        if not st["queue"]:
            return
        job = st["queue"].pop(0)
        job["started"] = time.time()
        logf = open(self.sd.root / "auto.log", "a", encoding="utf-8")
        logf.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {job['kind']}: {' '.join(job['args'])}\n")
        logf.flush()
        try:
            self.proc = subprocess.Popen(self.cmd_prefix + job["args"], stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         cwd=str(Path(__file__).resolve().parents[2]), start_new_session=True)
        except OSError as e:
            self.log(f"auto: failed to start {job['kind']}: {e}")
            logf.close()
            return
        logf.close()
        self.current = job
        st["running"] = {"kind": job["kind"], "started": job["started"], "out": job["out"]}
        self.log(f"auto: started {job['kind']} (pid {self.proc.pid})")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.log("auto: job terminated (runner stopping)")
            if self.current:
                self.state["auto"]["queue"].insert(0, {k: self.current[k] for k in ("kind", "args", "out")})
            self.state["auto"]["running"] = None
            self.proc = None
            self.current = None
