# SPDX-License-Identifier: Apache-2.0
"""学習の進み具合をリポジトリの機械可読なファイルに書き出し、節目ごとに push する（docs/runbook.md §6）。

`~/libra-run` の状態（state.json・status.json・eval/*.jsonl・matches/*.summary.json・metrics.jsonl）は
手元の PC にしか無く、クラウドのセッションからは読めない。節目ごとにその要約を GitHub の別ブランチへ
push しておけば、どこからでも同じ数値を読める（2026-09-18 のユーザーの困りごと）。

書き出すのは数値と step・局数と重みのファイル名だけで、絶対パスはファイル名に直し、秘密情報・ホストの
情報・棋譜は入れない（`scrub`）。作業ツリーと HEAD には触れない: git の下位コマンドで blob → tree →
commit を作り、`<commit>:refs/heads/<branch>` を push するだけなので、ランの最中に別の作業をしていても
邪魔せず、main の履歴も CI も動かさない。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .state import StateDir, read_json

SCHEMA = 1

# 絶対パスはファイル名だけにする（重みの置き場も run の置き場も外に出さない）
_PATHY = re.compile(r"(?:[A-Za-z]:)?(?:[/\\][^\s:*?\"<>|]+)*[/\\]([^\s/\\:*?\"<>|]+\.(?:pt|onnx|json|jsonl|log|toml|py|exe|sh))")


def scrub(obj, home: str | None = None):
    """入れ子の dict / list の文字列から、絶対パスと自分のホームを消す。"""
    home = home if home is not None else str(Path.home())
    if isinstance(obj, dict):
        return {k: scrub(v, home) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v, home) for v in obj]
    if isinstance(obj, str):
        s = _PATHY.sub(r"\1", obj)
        return s.replace(home, "~") if home else s
    return obj


# ---- 要約を作る ----
_METRIC_KEYS = ("t", "step", "games_total", "gpd", "gpd_5m", "window", "rss_mb", "swap_mb", "gpu_mb")


def _metric_row(r: dict) -> dict:
    out = {k: r.get(k) for k in _METRIC_KEYS}
    tr = r.get("train") or {}
    out["loss"] = tr.get("loss")
    out["policy_acc"] = tr.get("policy_acc")
    gen = r.get("gen") or {}
    # 価値の相関だけだと「横ばい」の読み分けができないので、方策の側と布石の側も出す
    # （2026-09-19: 本将棋の corr_v は結果 z が上限を決めるので上がり続けない。docs/gen-metric-2026-09-19.md）
    for side in ("window", "heldout"):
        g = gen.get(side) or {}
        out[f"gen_{side}_corr_v"] = (g.get("normal") or {}).get("corr_v")
        out[f"gen_{side}_policy_acc"] = (g.get("normal") or {}).get("policy_acc")
        out[f"gen_{side}_policy_ce"] = (g.get("normal") or {}).get("policy_ce")
        out[f"gen_{side}_fuseki_corr_v"] = (g.get("fuseki") or {}).get("corr_v")
    return out


def snapshot(sd: StateDir, cfg: dict | None = None, points: int = 120, now: float | None = None) -> dict:
    """リポジトリに置く要約。`libra status --json --history` と `libra review --json` と同じ値を、
    クラウドから読むぶんだけ絞って 1 つの dict にする。"""
    from .auto import collect_anchor, collect_best, collect_matches, collect_reference, list_archives, load_metrics
    from .review import review
    from .supervise import running_pid

    now = now or time.time()
    st = read_json(sd.status_json, {}) or {}
    state = sd.read_state() or {}
    acfg = (cfg or {}).get("auto", {}) or {}
    auto = state.get("auto") or {}
    anchor, best = auto.get("anchor") or {}, auto.get("best") or {}
    running = auto.get("running") or {}
    out = {
        "schema": SCHEMA,
        "run": sd.root.name,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "t": round(now, 1),
        "process": "running" if running_pid(sd.root / "run.lock") is not None else "not running",
        "flags": [f for f in StateDir.FLAGS if sd.flag(f)],
        "now": {
            "step": state.get("step"), "generation": state.get("generation"), "games_total": state.get("games_total"),
            "games_per_day_1h": st.get("games_per_day_1h"), "window_games": st.get("window_games"),
            "active_games": st.get("active_games"), "elapsed_h": st.get("elapsed_h"),
            "heldout_games": st.get("heldout_games"), "status_time": st.get("time"),
            "train": {k: (st.get("train") or {}).get(k) for k in ("loss", "policy", "value", "v41", "policy_acc", "lr")} if st.get("train") else None,
            "gen": st.get("gen"),
        },
        "auto": {
            "anchor_step": anchor.get("step"), "anchor_offset": anchor.get("offset"),
            "best_step": best.get("step"), "best_stall": auto.get("best_stall"),
            "queue": len(auto.get("queue") or []), "running": running.get("kind"),
            "last_archive_games": auto.get("last_archive_games"),
            # 今の固定の参照と、勝ちすぎて自動で外したもの（reference_rotate。docs/runbook.md §6）
            "references": [Path(x).name for x in (auto.get("references") or [])],
            "references_retired": [Path(x).name for x in (auto.get("references_retired") or [])],
        },
        "auto_cfg": {k: acfg.get(k) for k in ("enabled", "every_games", "eval_games", "eval_sims", "anchor_games",
                                              "best_games", "reference_games", "reference_ckpts", "reference_rotate",
                                              "reference_min", "match_games",
                                              "match_go", "match_go_opp", "match_opponent_opt", "match_libra_opt",
                                              "match_fuseki", "match_opponent", "match_libra_standard", "match_use_best")},
        "best": collect_best(sd),
        "anchor": collect_anchor(sd),
        "reference": collect_reference(sd),
        "matches": collect_matches(sd),
        "archives": [{"step": s} for p in list_archives(sd) if (s := int(p.stem.split("_")[1])) is not None],
        "metrics": [_metric_row(r) for r in load_metrics(sd, max(1, points))],
    }
    try:
        out["review"] = review(sd, now=now)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as e:  # 物差しが揃う前でも要約は出す
        out["review"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        from .scaling import scaling

        out["scaling"] = scaling(sd)
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as e:
        out["scaling"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        from .rating import curve as rating_curve
        from .rating import rating

        r = rating(sd)
        r["curve"] = rating_curve(sd, r)
        r["pairs"] = (r.get("pairs") or [])[:5]      # ずれの大きい組だけ残す（全部だと長い）
        out["rating"] = r
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError, StopIteration) as e:
        out["rating"] = {"error": f"{type(e).__name__}: {e}"}
    return scrub(out)


def _fmt(v, digits: int = 1, plus: bool = False) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:{'+' if plus else ''}.{digits}f}"
    if isinstance(v, int):
        return f"{v:{'+' if plus else ''},}"
    return str(v)


def format_md(s: dict) -> str:
    """GitHub でそのまま読める短い要約（数値の正は同じ場所の .json）。"""
    n, a = s.get("now") or {}, s.get("auto") or {}
    L = [f"# {s['run']} の進み具合", "",
         f"{s['generated']} 時点（{s['process']}）。数値の正は同じ場所の `{s['run']}.json`。自動で書き出しているので手で直さない。", "",
         "| 項目 | 値 |", "|---|---|",
         f"| step | {_fmt(n.get('step'))} |",
         f"| 総局数 | {_fmt(n.get('games_total'))} |",
         f"| 局/日（1 時間平均） | {_fmt(n.get('games_per_day_1h'))} |",
         f"| リプレイの窓 | {_fmt(n.get('window_games'))} |",
         f"| 最強の step | {_fmt(a.get('best_step'))}（足踏み {_fmt(a.get('best_stall'))}） |",
         f"| 基準の step | {_fmt(a.get('anchor_step'))}（offset {_fmt(a.get('anchor_offset'), plus=True)}） |"]
    b = (s.get("best") or [])[-1:]
    if b:
        r = b[0]
        L.append(f"| 最強比（直近） | step {_fmt(r.get('step'))} vs {_fmt(r.get('best_step'))}: {_fmt(r.get('elo_vs_best'), plus=True)} Elo 区間 {r.get('ci95')}（{_fmt(r.get('n'))} 局） |")
    an = (s.get("anchor") or [])[-1:]
    if an:
        r = an[0]
        L.append(f"| 基準比（直近） | step {_fmt(r.get('step'))}: 累積 {_fmt(r.get('elo'), plus=True)} Elo 区間 {r.get('ci95')}（{_fmt(r.get('n'))} 局） |")
    for r in (s.get("reference") or [])[-4:]:
        L.append(f"| 参照 {r.get('ref')} | step {_fmt(r.get('step'))}: {_fmt(r.get('elo'), plus=True)} Elo 区間 {r.get('ci95')}（勝ち {_fmt((r.get('score_new') or 0) * 100, 0)}%、{_fmt(r.get('n'))} 局） |")
    for m in (s.get("matches") or [])[-6:]:   # 段に分けた節目は 1 回で何行にもなる
        wr = m.get("winrate")
        go = str(m.get("go"))
        if m.get("go_opp") and m.get("go_opp") != m.get("go"):
            go += f" / 相手 {m.get('go_opp')}"   # 相手と読む量が違う（段・ハンデ）
        if (m.get("fuseki") or "engine") != "engine":
            go += "、布石は Libra 同士"
        L.append(f"| 外部計測 {m.get('opponent')} | step {_fmt(m.get('libra_step'))}: 得点 {_fmt(wr, 3)}（{_fmt(m.get('n'))} 局、{go}） |")
    sc = s.get("scaling") or {}
    curve = (sc.get("curve") or {}).get("fit")
    outlook = sc.get("outlook")
    if curve and outlook:
        L.append(f"| 局を 2 倍にしたときの伸び | {_fmt(curve.get('elo_per_doubling'), plus=True)} Elo / 2 倍"
                 f"（帯の中の {_fmt(curve.get('n'))} 点、残差 {_fmt(curve.get('rms_resid'))}） |")
        L.append(f"| 100 万局の買い足しの見込み | {_fmt(outlook.get('next_1m_elo'), plus=True)} Elo"
                 f"（2 倍 ＝ +{_fmt(outlook.get('double_games'))} 局で ${_fmt(outlook.get('double_cost_usd'), 2)}、1 Elo あたり ${_fmt(outlook.get('usd_per_elo'), 3)}） |")
    rt = s.get("rating") or {}
    rc = (rt.get("curve") or {}).get("fit")
    if rc:
        L.append(f"| 全部の対局から出した伸び | {_fmt(rc.get('elo_per_doubling'), plus=True)} Elo / 2 倍"
                 f"（{_fmt(rc.get('n'))} 点、残差 {_fmt(rc.get('rms_resid'))}。Bradley-Terry） |")
    # 学習の初めは伸び方が違うので、頭打ちかどうかは「最近の伸び」で見る（全部の点の当てはめは残差が大きい）
    rr = (rt.get("curve") or {}).get("fit_recent")
    if rr and rr is not rc and rr != rc:
        d = (rt.get("curve") or {}).get("recent_doublings")
        L.append(f"| 最近の伸び（直近 {_fmt(d)} 回の倍化） | {_fmt(rr.get('elo_per_doubling'), plus=True)} Elo / 2 倍"
                 f"（{_fmt(rr.get('n'))} 点、残差 {_fmt(rr.get('rms_resid'))}。頭打ちかはこちらで見る） |")
    rf = rt.get("fit") or {}
    if rf.get("chi2_per_df") is not None:
        L.append(f"| じゃんけん度（当てはまり） | χ²/自由度 {_fmt(rf.get('chi2_per_df'), 2)}"
                 f"（1 なら Elo の 1 本の目盛りで説明できる。{_fmt(rf.get('n_pairs'))} 組・{_fmt(rf.get('n_nodes'))} 点） |")
    rv = s.get("review") or {}
    if rv.get("items"):
        L += ["", f"## 物差しの判定: {rv.get('verdict')}", ""]
        L += [f"- [{i['verdict']}] {i['name']}: {i['why']}" for i in rv["items"]]
    if sc.get("notes"):
        L += ["", "## 伸びの曲線の注意", ""] + [f"- {n}" for n in sc["notes"]]
    return "\n".join(L) + "\n"


# ---- git（作業ツリーと HEAD には触れない） ----
def _git(repo: Path, *args: str, input: str | None = None, env: dict | None = None, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], input=input, capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r.stdout.strip()


def _signoff(repo: Path) -> str:
    name = _git(repo, "config", "user.name", check=False)
    mail = _git(repo, "config", "user.email", check=False)
    return f"\n\nSigned-off-by: {name} <{mail}>" if name and mail else ""


def publish(repo: Path, branch: str, files: dict[str, str], message: str, remote: str = "origin",
            push: bool = True, attempts: int = 3, log=print) -> str | None:
    """files（リポジトリ内の相対パス → 中身）を branch に 1 コミット足して push する。中身が同じなら何もしない。

    作業ツリー・index・HEAD・ローカルのブランチには触れない（blob → tree → commit を作り、コミットの
    ハッシュを直接 push する）。push が弾かれたら remote を読み直して作り直す。積んだコミットのハッシュを返す。
    """
    repo = Path(repo)
    for attempt in range(1, attempts + 1):
        parent = None
        if push:
            _git(repo, "fetch", "--quiet", remote, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}", check=False)
            parent = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/remotes/{remote}/{branch}", check=False) or None
        with tempfile.TemporaryDirectory() as td:
            env = {**os.environ, "GIT_INDEX_FILE": str(Path(td) / "index")}
            if parent:
                _git(repo, "read-tree", parent, env=env)
            for path, content in sorted(files.items()):
                blob = _git(repo, "hash-object", "-w", "--stdin", input=content)
                _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
            tree = _git(repo, "write-tree", env=env)
        if parent and tree == _git(repo, "rev-parse", f"{parent}^{{tree}}"):
            log("progress: no change")
            return None
        args = ["commit-tree", tree] + (["-p", parent] if parent else []) + ["-m", message + _signoff(repo)]
        commit = _git(repo, *args)
        if not push:
            return commit
        r = subprocess.run(["git", "-C", str(repo), "push", "--quiet", remote, f"{commit}:refs/heads/{branch}"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            log(f"progress: pushed {commit[:9]} to {remote}/{branch}")
            return commit
        log(f"progress: push failed ({attempt}/{attempts}): {(r.stderr or r.stdout).strip()}")
        if attempt < attempts:
            time.sleep(2 ** attempt)
    return None


def repo_root() -> Path:
    """既定のリポジトリ（このファイルの入っているチェックアウト）。"""
    return Path(__file__).resolve().parents[2]


def files_for(sd: StateDir, cfg: dict | None, points: int, subdir: str) -> tuple[dict[str, str], dict]:
    snap = snapshot(sd, cfg, points)
    d = subdir.strip("/")
    pre = f"{d}/" if d else ""
    files = {f"{pre}{sd.root.name}.json": json.dumps(snap, ensure_ascii=False, indent=1) + "\n",
             f"{pre}{sd.root.name}.md": format_md(snap)}
    # 効いている設定そのもの（ホームは ~ に直す）。外から「今どの設定で動いているか」を読めるようにする
    if sd.config_toml.exists():
        from .runconfig import home_to_tilde

        files[f"{pre}{sd.root.name}-config.toml"] = home_to_tilde(sd.config_toml.read_text(encoding="utf-8"))
    return files, snap


def message_for(snap: dict) -> str:
    n = snap.get("now") or {}
    return f"進捗 {snap['run']}: step {_fmt(n.get('step'))}、総局数 {_fmt(n.get('games_total'))}（{snap['generated']}）"


class Publisher:
    """ランナーから節目ごとに `libra progress --publish` を別プロセスで起動する（学習のループを止めない）。"""

    def __init__(self, sd: StateDir, cfg: dict, log, cmd_prefix: list[str] | None = None):
        self.sd = sd
        self.pcfg = (cfg or {}).get("progress", {}) or {}
        self.log = log
        self.cmd_prefix = cmd_prefix if cmd_prefix is not None else [
            sys.executable, "-m", "libra_league.cli", "--root", str(sd.root.parent), "--run", sd.root.name]
        self.proc: subprocess.Popen | None = None
        self.last = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.pcfg.get("enabled", False))

    def poll(self) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            self.proc = None

    def maybe_publish(self, now: float, milestone: bool = False) -> bool:
        """milestone（自動計測が動いた）か、heartbeat_minutes ごとに書き出す。前の書き出しが走っていれば見送る。"""
        if not self.enabled:
            return False
        self.poll()
        if self.proc is not None or now - self.last < float(self.pcfg.get("min_seconds", 120)):
            return False
        hb = float(self.pcfg.get("heartbeat_minutes", 180)) * 60
        if not milestone and not (hb > 0 and now - self.last >= hb):
            return False
        args = ["progress", "--publish", "--points", str(int(self.pcfg.get("metrics_points", 120))),
                "--branch", str(self.pcfg.get("branch", "progress")), "--dir", str(self.pcfg.get("dir", "progress"))]
        if self.pcfg.get("repo"):
            args += ["--repo", str(self.pcfg["repo"])]
        if not self.pcfg.get("push", True):
            args += ["--no-push"]
        try:
            self.proc = subprocess.Popen(self.cmd_prefix + args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, cwd=str(repo_root()), start_new_session=True)
        except OSError as e:
            self.log(f"progress: failed to start: {e}")
            return False
        self.last = now
        return True
