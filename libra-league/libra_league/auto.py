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

_CKPT_RE = re.compile(r"ckpt_(\d+)\.(?:pt|onnx)$")  # 外部計測は archive の重みを書き出した .onnx で打つ
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
                      "sennichite", "perpetual_check", "max_ply", "resign", "plies_sum", "mate_found", "proof_found")


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
def crossed_games_multiple(last_games: int | None, games: int, every_games: int) -> bool:
    """局数の区切り: 前回から every_games の倍数（10 万・20 万…）を越えたか。前回が無い（None・負）なら True。
    前回の差で数えると区切りが半端（118,700 など）になるので、倍数に揃える（2026-09-17 のユーザーの希望）。"""
    if last_games is None or int(last_games) < 0:
        return True
    return games // every_games > int(last_games) // every_games


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
                     ("best", None), ("best_stall", 0), ("last_archive_games", None), ("last_match_games", None),
                     ("references", []), ("references_retired", []), ("references_seeded", False)):
            st.setdefault(k, v)
        return st

    @property
    def enabled(self) -> bool:
        return bool(self.acfg.get("enabled", False))

    def games_due(self, games: int) -> bool:
        """局数区切りの run で、前回の archive から every_games の倍数を越えたか（ランナーが 10 分を待たずにチェックポイントを取る合図）。
        前回が無い run（最初のチェックポイントで必ず積む）と時間区切りの run では False。"""
        every_games = int(self.acfg.get("every_games", 0))
        last = self._st().get("last_archive_games")
        return self.enabled and every_games > 0 and last is not None and crossed_games_multiple(last, games, every_games)

    # -- 起票 --
    def on_checkpoint(self, ckpt: Path, now: float | None = None) -> None:
        """チェックポイント直後に呼ぶ。archive の期限と EVAL_NOW / MATCH_NOW フラグを見てジョブを積む。"""
        if not self.enabled:
            return
        now = now or time.time()
        st = self._st()
        games = int(self.state.get("games_total") or 0)
        eval_now = self.sd.flag("EVAL_NOW")
        if eval_now or st["last_archive"] is None or self._due(st.get("last_archive_games"), games):
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
        if int(self.acfg.get("match_games", 0)) > 0 and (match_now or st["last_match"] is None or self._due(st.get("last_match_games"), games)):
            arch = archive_dir(self.sd) / ckpt.name  # 節目の写しがあればそれを使う（計測待ちの間も消えない）
            self.enqueue_match(arch if arch.exists() else ckpt)
            st["last_match"] = now
            st["last_match_games"] = games
            if match_now:
                self.sd.clear_flag("MATCH_NOW")

    def _due(self, last_games: int | None, games: int) -> bool:
        """次の計測の時期か。総局数が every_games の倍数を越えたら（PC の利用状況で局/日が変わっても、
        判断に要る局数がたまったときに測る。2026-09-17 のユーザーの指示）。0 なら最初の 1 回と eval-now / match-now だけ。
        時間区切り（every_hours）は 2026-09-19 に廃止した（docs/decisions.md）。"""
        every_games = int(self.acfg.get("every_games", 0))
        if every_games <= 0:
            return False
        return crossed_games_multiple(last_games, games, every_games)

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
        best_job = self.enqueue_best(new)
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
        # 最強と基準が同じ重みで局数も同じなら、同じ組の対局になるので最強比の結果を写す（2026-09-18 のユーザーの指示）
        reuse = None
        if best_job is not None and ckpt_step(best_job["a"]) == int(anc.get("step") or -1) \
                and int(self.acfg.get("best_games", 0)) == int(self.acfg.get("anchor_games", 0)):
            reuse = best_job["out"]
        self.enqueue_anchor(Path(anc["file"]), new, anc, reuse=reuse)

    # -- 最強比（best、docs/restart-plan.md §3 M2）: これまでで最強の保存済みと打ち、有意に勝ったら最強を置き換える --
    def enqueue_best(self, new: Path) -> dict | None:
        """積んだら {"a": 最強のファイル, "out": 結果の JSON} を返す（基準比が同じ組なら結果を写すため）。"""
        games = int(self.acfg.get("best_games", 0))
        if games <= 0:
            return None
        st = self._st()
        best = st.get("best")
        if not best or not Path(best.get("file", "")).exists():
            st["best"] = {"file": str(new), "step": ckpt_step(new), "since": time.time()}
            st["best_stall"] = 0
            self.log(f"auto: best = {new.name}")
            return None
        if int(best.get("step") or -1) == ckpt_step(new):
            return None
        a = Path(best["file"])
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = self.sd.root / "eval" / f"best-{ts}-{ckpt_step(a)}-{ckpt_step(new)}.json"
        st["queue"].append({"kind": "best", "args": self._eval_args(a, new, games, out), "out": str(out)})
        self.log(f"auto: queued best {a.name} vs {new.name} ({games} games)")
        return {"a": str(a), "out": str(out)}

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
    def active_references(self) -> list[str]:
        """今の参照。設定の `reference_ckpts` を種にして state に持ち、勝ちすぎた参照を入れ替えていく
        （docs/runbook.md §6。手で config を直さなくても物差しが天井に着かないようにする）。"""
        st = self._st()
        act = [str(x) for x in (st.get("references") or [])]
        retired = [str(x) for x in (st.get("references_retired") or [])]
        seeded = bool(st.get("references_seeded"))
        for ref in [str(x) for x in (self.acfg.get("reference_ckpts") or []) if str(x).strip()]:
            # 設定に新しく足された参照だけ取り込む（自動で外したものを設定が書き戻さないように）
            if ref not in act and (ref not in retired or not seeded):
                act.append(ref)
        st["references"] = act
        st["references_retired"] = retired
        st["references_seeded"] = True
        return act

    def rotate_references(self, ref_path: str, step: int | None, score_new: float | None) -> None:
        """参照に勝ちすぎたら外し、代わりに**今の重みの archive** を参照にする。

        新しい参照をこの run 自身の archive にするのは、足した時点では自分自身（＝互角）なので、
        古い参照が既に天井に着いていても目盛りが必ずつながるため（docs/scaling-2026-09-18.md §6.5）。"""
        thr = float(self.acfg.get("reference_rotate", 0.0) or 0.0)
        if thr <= 0 or score_new is None or float(score_new) <= thr or step is None:
            return
        st = self._st()
        act = self.active_references()
        if ref_path not in act:
            return
        new_ref = self.sd.root / "checkpoints" / "archive" / f"ckpt_{int(step):09d}.pt"
        if not new_ref.exists():
            self.log(f"auto: reference {Path(ref_path).name} は得点 {score_new} だが、置き換える archive が無い（{new_ref.name}）")
            return
        if str(new_ref) in act:
            keep_min = max(int(self.acfg.get("reference_min", 1)), 1)
            if len(act) <= keep_min:
                return
            act.remove(ref_path)
            st["references_retired"] = [*st.get("references_retired", []), ref_path]
            self.log(f"auto: reference {Path(ref_path).name} を外した（得点 {score_new} > {thr}。代わりは既にある）")
            return
        act[act.index(ref_path)] = str(new_ref)
        st["references_retired"] = [*st.get("references_retired", []), ref_path]
        st["references"] = act
        self.log(f"auto: reference {Path(ref_path).name}（得点 {score_new} > {thr}）を {new_ref.name} に入れ替えた")
        with open(self.sd.root / "eval" / "references.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "step": int(step), "out": Path(ref_path).name,
                                "in": new_ref.name, "score_new": float(score_new), "threshold": thr},
                               ensure_ascii=False) + "\n")

    def enqueue_references(self, new: Path) -> None:
        games = int(self.acfg.get("reference_games", 0))
        refs = self.active_references()
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
        self.rotate_references(str(job.get("ref") or ""), row["step"], row["score_new"])

    def enqueue_anchor(self, a: Path, b: Path, anc: dict | None = None, reuse: str | None = None) -> None:
        """基準との対局を積む。積んだときの基準の step と offset をジョブに残す（結果が出る前に先の結果で基準が替わっても、
        実際に打った相手の offset で数えるため）。reuse があれば、打たずにその結果（同じ組の最強比）を写す。"""
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = self.sd.root / "eval" / f"anchor-{ts}-{ckpt_step(a)}-{ckpt_step(b)}.json"
        args = ["eval", "--a", str(a), "--b", str(b), "--games", str(self.acfg.get("anchor_games", 100)),
                "--sims", str(self.acfg.get("eval_sims", 96)), "--concurrent", str(self.acfg.get("eval_concurrent", 64)),
                "--threads", str(self.acfg.get("eval_threads", 4)), "--out", str(out)]
        job = {"kind": "anchor", "args": args, "out": str(out)}
        if anc is not None:
            job["anchor_step"] = int(anc.get("step") or ckpt_step(a))
            job["anchor_offset"] = float(anc.get("offset", 0.0))
        if reuse:
            job["reuse"] = reuse
        self._st()["queue"].append(job)
        self.log(f"auto: queued anchor {a.name} vs {b.name}" + (" (reuses the best result)" if reuse else ""))

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
        step_a = ckpt_step(r.get("a", ""))
        offset = self._anchor_offset(step_a, job)
        step_b = ckpt_step(r.get("b", ""))
        row = {"t": time.time(), "step": step_b, "games": self.state.get("games_total"), "n": r.get("n"), "score_new": round(1 - float(r.get("score_a", 0.5)), 4),
               "anchor_step": step_a, "offset": round(offset, 1), "elo_vs_anchor": round(elo, 1),
               "elo": round(offset + elo, 1),
               "ci95": [round(offset + lo, 1) if lo is not None else None, round(offset + hi, 1) if hi is not None else None]}
        with open(self.sd.root / "eval" / "anchor.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.log(f"auto: anchor step {step_a} vs {step_b}: {elo:+.1f} Elo (total {row['elo']:+.1f}, new wins {row['score_new']:.0%})")
        # 置き換えるのは今の基準と打った結果のときだけ（積み上がった古い基準との結果では替えない）
        if row["score_new"] >= float(self.acfg.get("anchor_rebaseline", 0.85)) and step_a == int(anc.get("step") or -1):
            new_file = str(r.get("b", ""))
            if Path(new_file).exists():
                st["anchor"] = {"file": new_file, "step": step_b, "offset": round(offset + elo, 1), "since": time.time()}
                self.log(f"auto: anchor -> step {step_b} (offset {st['anchor']['offset']:+.1f}; 基準に勝ちすぎたので置き換え)")

    def _anchor_offset(self, step_a: int, job: dict | None = None) -> float:
        """基準 step_a の offset。ジョブに積んだときの値があればそれ、今の基準ならその値、過去の基準なら anchor.jsonl のその step の行の累積、
        どれも無ければ最初の基準として 0。"""
        if job and "anchor_offset" in job and int(job.get("anchor_step", -1)) == step_a:
            return float(job["anchor_offset"])
        anc = self._st().get("anchor") or {}
        if int(anc.get("step") or -1) == step_a:
            return float(anc.get("offset", 0.0))
        for row in reversed(collect_anchor(self.sd)):
            if int(row.get("step") or -1) == step_a and row.get("elo") is not None:
                return float(row["elo"])
        return 0.0

    def repair_anchor_chain(self) -> bool:
        """eval/anchor-*.json（積んだ順）から基準比の行と今の基準を数え直し、anchor.jsonl・state と違えば、古い anchor.jsonl を
        anchor.jsonl.bak-<時刻> に残して書き直す。直したら True。起動時に 1 回呼ぶ（2026-09-18 に、積み上がった計測の結果を
        替わった後の基準の offset で数えた誤りを直すため）。"""
        ev = self.sd.root / "eval"
        files = sorted(p for p in ev.glob("anchor-*.json")) if ev.is_dir() else []
        results = []
        for p in files:
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if r.get("elo_a_minus_b") is None or not r.get("a") or not r.get("b"):
                continue
            results.append((p, r))
        if not results:
            return False
        thr = float(self.acfg.get("anchor_rebaseline", 0.85))
        old = collect_anchor(self.sd)
        by_step = {int(o.get("step") or -1): o for o in old}
        first = results[0][1]
        cur = {"file": str(first["a"]), "step": ckpt_step(first["a"]), "offset": 0.0}
        offsets = {cur["step"]: 0.0}
        rows = []
        for p, r in results:
            step_a, step_b = ckpt_step(r["a"]), ckpt_step(r["b"])
            if step_a not in offsets:
                self.log(f"auto: anchor chain not repaired: {p.name} was played against step {step_a}, which never was the anchor")
                return False
            off = offsets[step_a]
            elo = -float(r["elo_a_minus_b"])
            ci = r.get("elo_ci95") or [None, None]
            score_new = round(1 - float(r.get("score_a", 0.5)), 4)
            prev = by_step.get(step_b, {})
            rows.append({"t": prev.get("t", p.stat().st_mtime), "step": step_b, "games": prev.get("games"), "n": r.get("n"), "score_new": score_new,
                         "anchor_step": step_a, "offset": round(off, 1), "elo_vs_anchor": round(elo, 1), "elo": round(off + elo, 1),
                         "ci95": [round(off - float(ci[1]), 1) if ci[1] is not None else None, round(off - float(ci[0]), 1) if ci[0] is not None else None]})
            if score_new >= thr and step_a == cur["step"] and step_b > cur["step"]:
                cur = {"file": str(r["b"]), "step": step_b, "offset": round(off + elo, 1)}
                offsets[step_b] = cur["offset"]
        st = self._st()
        anc = st.get("anchor") or {}
        key = lambda rs: [(x.get("step"), x.get("anchor_step"), x.get("offset"), x.get("elo"), x.get("ci95")) for x in rs]  # noqa: E731
        same_rows = key(rows) == key(old)
        same_anchor = int(anc.get("step") or -1) == cur["step"] and float(anc.get("offset", 0.0)) == cur["offset"]
        if same_rows and same_anchor:
            return False
        path = ev / "anchor.jsonl"
        if path.exists():
            shutil.copy2(path, ev / f"anchor.jsonl.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
        os.replace(tmp, path)
        if not same_anchor:
            st["anchor"] = {**cur, "since": anc.get("since", time.time()) if int(anc.get("step") or -1) == cur["step"] else time.time()}
        self.log(f"auto: anchor chain repaired from {len(rows)} result files (anchor step {cur['step']}, offset {cur['offset']:+.1f}; "
                 f"old rows kept as a .bak)")
        return True

    def enqueue_match(self, ckpt: Path | None = None) -> None:
        """外部エンジンとの計測を積む。ckpt を渡すとその重みで打つ（渡さないと latest.onnx = 中身が動く別名になり、
        計測待ちの間に世代が変わって時系列の比較にならない。docs/restart-plan.md §7 P2）。"""
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = self.sd.root / "matches" / f"auto-{ts}.jsonl"
        args = ["match", "--games", str(self.acfg.get("match_games", 10)), "--go", str(self.acfg.get("match_go", "movetime 1000")), "--out", str(out)]
        for kv in str(self.acfg.get("match_opponent_opt", "")).split(","):
            if kv.strip():
                args += ["--opponent-opt", kv.strip()]
        if ckpt is not None:
            args += ["--ckpt", str(ckpt)]
            self.snapshot_onnx(ckpt)
        self.state["auto"]["queue"].append({"kind": "match", "args": args, "out": str(out)})
        self.log("auto: queued match" + (f" ({ckpt.name})" if ckpt is not None else ""))

    def snapshot_onnx(self, ckpt: Path) -> Path | None:
        """この世代の推論用の重み（latest.onnx）を ckpt の隣に写す。ランナーはチェックポイントの直後に
        latest.pt から書き出すので、この時点の latest.onnx は ckpt と同じ重み。写せたら match 側の書き出しは要らない。"""
        out = ckpt.with_suffix(".onnx")
        if out.exists():
            return out
        latest = self.sd.checkpoints / "latest.onnx"
        if not latest.exists():
            return None
        if latest.stat().st_mtime < ckpt.stat().st_mtime - 5.0:  # 書き出しに失敗して古いまま（match が .pt から書き出す）
            self.log(f"auto: latest.onnx is older than {ckpt.name}; match will export from the checkpoint")
            return None
        tmp = out.with_suffix(".onnx.tmp")
        shutil.copyfile(latest, tmp)
        os.replace(tmp, out)
        return out

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
        if job.get("reuse"):
            src = Path(job["reuse"])
            if src.exists():  # 同じ組の最強比の結果を写す（打たない）
                shutil.copy2(src, job["out"])
                job.update({"finished": time.time(), "rc": 0, "reused": True})
                self.record_anchor(job)
                st["history"] = (st["history"] + [job])[-20:]
                self.log(f"auto: {job['kind']} reused {src.name} (same pair as the best eval)")
                return True
            self.log(f"auto: {job['kind']}: {src.name} is missing, playing the games instead")
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
            st["queue"].insert(0, {k: job[k] for k in ("kind", "args", "out", "ref", "anchor_step", "anchor_offset", "reuse") if k in job})

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
