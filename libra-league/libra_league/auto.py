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
import signal
import subprocess
import sys
import time
from pathlib import Path

from .state import StateDir, write_json_atomic

_CKPT_RE = re.compile(r"ckpt_(\d+)\.pt$")
JOB_FILE = "auto_job.json"  # 実行中の計測ジョブ（pid・引数）。ランナーが abort しても孤児を見つけられるように state とは別に置く


def _job_alive(pid: int, out: str) -> bool:
    """pid が生きていて、コマンド行に出力先 out を含む（使い回された pid の別プロセスには触らない）。"""
    if not out:
        return False
    try:
        return out.encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def _kill_job_group(pid: int, out: str) -> None:
    """start_new_session=True で起動したジョブ（pgid = pid）を、match の相手エンジンなど孫ごと止める。"""
    for sig, wait in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if not _job_alive(pid, out):
                return
            time.sleep(0.1)

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
        "train": {k: status["train"].get(k) for k in ("loss", "policy", "value", "v41", "policy_acc", "lr", "target")} if status.get("train") else None,
        "engine": {k: eng.get(k) for k in METRIC_ENGINE_KEYS} if eng else None,
        "exploiter": {k: status["exploiter"].get(k) for k in ("games", "wins", "draws", "losses")} if status.get("exploiter") else None,
        "gpu_mb": (status.get("gpu") or {}).get("mem_reserved_mb"),
        "timing": status.get("timing"),  # 処理時間の内訳（looptime.py。この行までの窓）
        "gen": status.get("gen"),        # 一般化の物差し（genprof.py。窓の中と held-out。gen_minutes ごとに更新）
        "heldout": status.get("heldout_games"),
    }
    return row


def append_metrics(sd: StateDir, status: dict) -> None:
    with open(sd.root / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics_row(status), ensure_ascii=False) + "\n")


GPD_5M_MAX_GAP_S = 900  # 行の間隔（metrics_minutes=5）の 3 倍。これより空いたら一時停止・再起動をまたぐので出さない


def add_gpd_5m(rows: list[dict]) -> None:
    """各行に直前の行との局数差から局/日（`gpd_5m`、約 5 分平均）を足す。出せない行は None。
    間引く前に計算するので、グラフの点を間引いても各点は 5 分の値のまま。"""
    prev = None
    for r in rows:
        r["gpd_5m"] = None
        try:
            if prev is not None:
                dt = float(r["t"]) - float(prev["t"])
                dg = float(r["games_total"]) - float(prev["games_total"])
                if 0 < dt <= GPD_5M_MAX_GAP_S and dg >= 0:
                    r["gpd_5m"] = round(dg / dt * 86400)
        except (KeyError, TypeError, ValueError):
            pass
        prev = r


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
    add_gpd_5m(rows)
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
            "seconds": r.get("seconds"), "auto": p.name.startswith(("auto-", "anchor-")),  # best-・reference- は鎖に入れない
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


def collect_anchor(sd: StateDir) -> list[dict]:
    """eval/anchor.jsonl（基準ネットとの差の推移）。"""
    p = sd.root / "eval" / "anchor.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda r: r.get("t", 0))
    return rows


def _collect_jsonl(sd: StateDir, name: str) -> list[dict]:
    p = sd.root / "eval" / name
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda r: r.get("t", 0))
    return rows


def collect_best(sd: StateDir) -> list[dict]:
    """eval/best.jsonl（最強比の推移）。"""
    return _collect_jsonl(sd, "best.jsonl")


def collect_reference(sd: StateDir) -> list[dict]:
    """eval/reference.jsonl（固定の参照との差の推移）。"""
    return _collect_jsonl(sd, "reference.jsonl")


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
        self.suspended = False  # 停止・一時停止中は新しいジョブを起動しない（stop / resume）
        self._st()

    def _st(self) -> dict:
        """state["auto"]（load() で state が置き換わっても欠けたキーを補う）。"""
        st = self.state.setdefault("auto", {})
        for k, v in (("last_archive", None), ("last_match", None), ("queue", []), ("history", []), ("running", None), ("anchor", None),
                     ("best", None), ("best_stall", 0), ("last_archive_games", None), ("last_match_games", None)):
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
        games = int(self.state.get("games_total") or 0)
        eval_now = self.sd.flag("EVAL_NOW")
        if eval_now or st["last_archive"] is None or self._due(st["last_archive"], st.get("last_archive_games"), now, games):
            prev = list_archives(self.sd)
            new = archive_checkpoint(self.sd, ckpt)
            st["last_archive"] = now
            st["last_archive_games"] = games
            if new is not None:
                self.log(f"auto: archived {new.name}")
                anc_file = str((st.get("anchor") or {}).get("file") or "")
                # 基準が直前の世代と同じなら比較が同一なので、基準の対局 1 回で鎖も兼ねる（collect_evals は anchor- も鎖に使う）
                if prev and self.acfg.get("chain_eval", True) and str(prev[-1]) != anc_file:
                    self.enqueue_eval(prev[-1], new)
                self.on_new_archive(new)
            if eval_now:
                self.sd.clear_flag("EVAL_NOW")
        match_now = self.sd.flag("MATCH_NOW")
        if int(self.acfg.get("match_games", 0)) > 0 and (match_now or st["last_match"] is None or self._due(st["last_match"], st.get("last_match_games"), now, games)):
            self.enqueue_match()
            st["last_match"] = now
            st["last_match_games"] = games
            if match_now:
                self.sd.clear_flag("MATCH_NOW")

    def _due(self, last_t: float | None, last_games: int | None, now: float, games: int) -> bool:
        """次の計測の時期か。every_games > 0 なら局数で（PC の利用状況で局/日が変わっても、判断に要る局数がたまったときに測る。
        2026-09-17 のユーザーの指示）、そうでなければ every_hours で。両方 0 なら最初の 1 回と eval-now / match-now だけ。"""
        every_games = int(self.acfg.get("every_games", 0))
        if every_games > 0:
            return last_games is None or games - int(last_games) >= every_games
        every = float(self.acfg.get("every_hours", 0.0)) * 3600
        return every > 0 and (last_t is None or now - float(last_t) >= every)

    def enqueue_eval(self, a: Path, b: Path) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = self.sd.root / "eval" / f"auto-{ts}-{ckpt_step(a)}-{ckpt_step(b)}.json"
        args = ["eval", "--a", str(a), "--b", str(b), "--games", str(self.acfg.get("eval_games", 100)),
                "--sims", str(self.acfg.get("eval_sims", 96)), "--concurrent", str(self.acfg.get("eval_concurrent", 64)),
                "--threads", str(self.acfg.get("eval_threads", 4)), "--out", str(out)]
        self.state["auto"]["queue"].append({"kind": "eval", "args": args, "out": str(out)})
        self.log(f"auto: queued eval {a.name} vs {b.name}")

    # -- 基準ネット（anchor） --
    def on_new_archive(self, new: Path) -> None:
        """基準が無ければこの世代を基準にし、あれば基準との対局を積む。最強比（best）と固定の参照（reference）も同じ節目で積む。"""
        self.enqueue_best(new)
        self.enqueue_references(new)
        if int(self.acfg.get("anchor_games", 0)) <= 0:
            return
        st = self._st()
        anc = st.get("anchor")
        if not anc or not Path(anc.get("file", "")).exists():
            st["anchor"] = {"file": str(new), "step": ckpt_step(new), "offset": 0.0, "since": time.time()}
            self.log(f"auto: anchor = {new.name} (offset 0)")
            return
        if int(anc.get("step") or -1) == ckpt_step(new):
            return
        self.enqueue_anchor(Path(anc["file"]), new)

    # -- 最強比（best、docs/restart-plan.md §3 M2）: これまでで最強の保存済みと打ち、有意に勝ったら最強を置き換える --
    def enqueue_best(self, new: Path) -> None:
        games = int(self.acfg.get("best_games", 0))
        if games <= 0:
            return
        st = self._st()
        best = st.get("best")
        if not best or not Path(best.get("file", "")).exists():
            st["best"] = {"file": str(new), "step": ckpt_step(new), "since": time.time()}
            st["best_stall"] = 0
            self.log(f"auto: best = {new.name}")
            return
        if int(best.get("step") or -1) == ckpt_step(new):
            return
        a = Path(best["file"])
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = self.sd.root / "eval" / f"best-{ts}-{ckpt_step(a)}-{ckpt_step(new)}.json"
        st["queue"].append({"kind": "best", "args": self._eval_args(a, new, games, out), "out": str(out)})
        self.log(f"auto: queued best {a.name} vs {new.name} ({games} games)")

    def _eval_args(self, a: Path, b: Path, games: int, out: Path) -> list[str]:
        return ["eval", "--a", str(a), "--b", str(b), "--games", str(games),
                "--sims", str(self.acfg.get("eval_sims", 96)), "--concurrent", str(self.acfg.get("eval_concurrent", 64)),
                "--threads", str(self.acfg.get("eval_threads", 4)), "--out", str(out)]

    def record_best(self, job: dict) -> None:
        """最強比の結果を eval/best.jsonl に足す。95% 区間の下限が 0 を超えたら最強を置き換え、そうでなければ足踏みを数える。"""
        st = self._st()
        best = st.get("best") or {}
        try:
            r = json.loads(Path(job["out"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.log(f"auto: best result unreadable: {e}")
            return
        if r.get("elo_a_minus_b") is None:
            return
        elo = -float(r["elo_a_minus_b"])
        ci = r.get("elo_ci95") or [None, None]
        lo = -float(ci[1]) if ci[1] is not None else None
        hi = -float(ci[0]) if ci[0] is not None else None
        step_b = ckpt_step(r.get("b", ""))
        improved = lo is not None and lo > 0
        if improved:
            st["best"] = {"file": str(r.get("b", "")), "step": step_b, "since": time.time()}
            st["best_stall"] = 0
        else:
            st["best_stall"] = int(st.get("best_stall", 0)) + 1
        row = {"t": time.time(), "step": step_b, "games": self.state.get("games_total"), "best_step": best.get("step"), "n": r.get("n"), "score_new": round(1 - float(r.get("score_a", 0.5)), 4),
               "elo_vs_best": round(elo, 1), "ci95": [round(lo, 1) if lo is not None else None, round(hi, 1) if hi is not None else None],
               "improved": improved, "stall": int(st["best_stall"])}
        with open(self.sd.root / "eval" / "best.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.log(f"auto: best step {best.get('step')} vs {step_b}: {elo:+.1f} Elo [{row['ci95'][0]}, {row['ci95'][1]}] "
                 + (f"-> best = step {step_b}" if improved else f"(best unchanged, stall {row['stall']})"))
        alert = int(self.acfg.get("best_stall_alert", 3))
        if not improved and alert > 0 and row["stall"] >= alert:
            self.log(f"auto: WARNING best not updated for {row['stall']} evals (best step {best.get('step')}); docs/restart-plan.md §3 M2 の見直し")

    # -- 固定の参照（reference、同 §3 M4）: run をまたいで同じ相手と打ち、絶対の物差しにする --
    def enqueue_references(self, new: Path) -> None:
        games = int(self.acfg.get("reference_games", 0))
        refs = [str(x) for x in (self.acfg.get("reference_ckpts") or []) if str(x).strip()]
        if games <= 0 or not refs:
            return
        st = self._st()
        for ref in refs:
            a = Path(ref).expanduser()
            if not a.exists():
                self.log(f"auto: reference not found: {a}")
                continue
            ts = time.strftime("%Y%m%d-%H%M%S")
            out = self.sd.root / "eval" / f"reference-{ts}-{a.stem}-{ckpt_step(new)}.json"
            st["queue"].append({"kind": "reference", "args": self._eval_args(a, new, games, out), "out": str(out), "ref": str(a)})
            self.log(f"auto: queued reference {a.name} vs {new.name} ({games} games)")

    def record_reference(self, job: dict) -> None:
        try:
            r = json.loads(Path(job["out"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.log(f"auto: reference result unreadable: {e}")
            return
        if r.get("elo_a_minus_b") is None:
            return
        ci = r.get("elo_ci95") or [None, None]
        ref = Path(str(job.get("ref") or r.get("a", "")))
        row = {"t": time.time(), "step": ckpt_step(r.get("b", "")), "games": self.state.get("games_total"), "ref": ref.name, "ref_step": ckpt_step(ref), "n": r.get("n"),
               "score_new": round(1 - float(r.get("score_a", 0.5)), 4), "elo": round(-float(r["elo_a_minus_b"]), 1),
               "ci95": [round(-float(ci[1]), 1) if ci[1] is not None else None, round(-float(ci[0]), 1) if ci[0] is not None else None]}
        with open(self.sd.root / "eval" / "reference.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.log(f"auto: reference {ref.name} vs step {row['step']}: {row['elo']:+.1f} Elo (new wins {row['score_new']:.0%})")

    def enqueue_anchor(self, a: Path, b: Path) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = self.sd.root / "eval" / f"anchor-{ts}-{ckpt_step(a)}-{ckpt_step(b)}.json"
        args = ["eval", "--a", str(a), "--b", str(b), "--games", str(self.acfg.get("anchor_games", 100)),
                "--sims", str(self.acfg.get("eval_sims", 96)), "--concurrent", str(self.acfg.get("eval_concurrent", 64)),
                "--threads", str(self.acfg.get("eval_threads", 4)), "--out", str(out)]
        self._st()["queue"].append({"kind": "anchor", "args": args, "out": str(out)})
        self.log(f"auto: queued anchor {a.name} vs {b.name}")

    def record_anchor(self, job: dict) -> None:
        """基準との対局の結果を anchor.jsonl に 1 行足し、勝ちすぎていれば基準を置き換える。"""
        st = self._st()
        anc = st.get("anchor") or {}
        try:
            r = json.loads(Path(job["out"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.log(f"auto: anchor result unreadable: {e}")
            return
        if r.get("elo_a_minus_b") is None:
            return
        elo = -float(r["elo_a_minus_b"])  # 新しい世代が基準より何 Elo 上か
        ci = r.get("elo_ci95") or [None, None]
        lo = -float(ci[1]) if ci[1] is not None else None
        hi = -float(ci[0]) if ci[0] is not None else None
        offset = float(anc.get("offset", 0.0))
        step_b = ckpt_step(r.get("b", "")) 
        row = {"t": time.time(), "step": step_b, "games": self.state.get("games_total"), "n": r.get("n"), "score_new": round(1 - float(r.get("score_a", 0.5)), 4),
               "anchor_step": anc.get("step"), "offset": round(offset, 1), "elo_vs_anchor": round(elo, 1),
               "elo": round(offset + elo, 1),
               "ci95": [round(offset + lo, 1) if lo is not None else None, round(offset + hi, 1) if hi is not None else None]}
        with open(self.sd.root / "eval" / "anchor.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.log(f"auto: anchor step {anc.get('step')} vs {step_b}: {elo:+.1f} Elo (total {row['elo']:+.1f}, new wins {row['score_new']:.0%})")
        if row["score_new"] >= float(self.acfg.get("anchor_rebaseline", 0.85)):
            new_file = str(r.get("b", ""))
            if Path(new_file).exists():
                st["anchor"] = {"file": new_file, "step": step_b, "offset": round(offset + elo, 1), "since": time.time()}
                self.log(f"auto: anchor -> step {step_b} (offset {st['anchor']['offset']:+.1f}; 基準に勝ちすぎたので置き換え)")

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
    def poll(self) -> bool:
        """定期的に呼ぶ。走っているジョブの終了を回収し、待ちがあれば次を起動する。state を変えたら True。"""
        st = self._st()
        changed = False
        if self.proc is not None:
            rc = self.proc.poll()
            if rc is None:
                return False
            changed = True
            job = self.current or {}
            job["finished"] = time.time()
            job["rc"] = rc
            if job.get("kind") == "anchor" and rc == 0:
                self.record_anchor(job)
            elif job.get("kind") == "best" and rc == 0:
                self.record_best(job)
            elif job.get("kind") == "reference" and rc == 0:
                self.record_reference(job)
            st["history"] = (st["history"] + [job])[-20:]
            st["running"] = None
            self.log(f"auto: {job.get('kind')} finished rc={rc} ({job['finished'] - job.get('started', job['finished']):.0f}s)")
            self.proc = None
            self.current = None
            (self.sd.root / JOB_FILE).unlink(missing_ok=True)
        if not st["queue"] or self.suspended:
            return changed
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
            return True
        logf.close()
        self.current = job
        st["running"] = {"kind": job["kind"], "started": job["started"], "out": job["out"], "args": job["args"], "pid": self.proc.pid}
        write_json_atomic(self.sd.root / JOB_FILE, st["running"])
        self.log(f"auto: started {job['kind']} (pid {self.proc.pid})")
        return True

    def stop(self) -> None:
        """実行中のジョブを止めて積み直し、以後 poll では起動しない（ランナーの停止・一時停止）。

        子は start_new_session=True なので、親の終了後に残ると GPU を使い続ける。積んだジョブは
        state に残るので、resume か次の run で走る（docs/runbook.md §6）。
        """
        self.suspended = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.log("auto: job terminated and requeued")
            if self.current:
                self._requeue(self.current)
            self.state["auto"]["running"] = None
            self.proc = None
            self.current = None
            (self.sd.root / JOB_FILE).unlink(missing_ok=True)

    def resume(self) -> None:
        """一時停止から戻ったとき（積んであるジョブを再び起動できるようにする）。"""
        self.suspended = False

    def _requeue(self, job: dict) -> None:
        """止めたジョブを待ち行列の先頭に戻す。途中まで書いた出力は .interrupted に改名する（match の棋譜は追記なので二重になる）。"""
        st = self._st()
        out = Path(job["out"])
        if out.exists():
            out.replace(out.with_name(out.name + ".interrupted"))
        if not any(q.get("out") == job["out"] for q in st["queue"]):
            st["queue"].insert(0, {k: job[k] for k in ("kind", "args", "out", "ref") if k in job})

    def recover(self) -> None:
        """前のランナーが stop() を通らずに終わった（abort など）ときに残った計測ジョブを止めて積み直す。load() の後に 1 回呼ぶ。

        子は start_new_session=True なので親が死んでも GPU を使い続け、state の running も回収されずに残る。
        監視役（supervise.py）が 60 秒後にランナーを起動し直すので、放置すると次の計測と二重に走る。
        """
        st = self._st()
        f = self.sd.root / JOB_FILE
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rec = None
        if rec and rec.get("pid") and _job_alive(int(rec["pid"]), str(rec.get("out", ""))):
            _kill_job_group(int(rec["pid"]), str(rec["out"]))
            self.log(f"auto: stopped orphaned {rec.get('kind')} job (pid {rec['pid']}) left by a crashed runner")
        f.unlink(missing_ok=True)
        lost = st.get("running") or rec
        st["running"] = None
        if not lost:
            return
        if lost.get("args"):
            self._requeue(lost)
            self.log(f"auto: requeued {lost.get('kind')} interrupted by a crashed runner")
        else:
            self.log(f"auto: dropped {lost.get('kind')} interrupted by a crashed runner (no args recorded)")
