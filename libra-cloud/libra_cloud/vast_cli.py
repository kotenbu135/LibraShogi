# SPDX-License-Identifier: Apache-2.0
"""vast.ai の自己対局ワーカーを起動・停止・確認するコマンド（bin/libra-vast。管理コンソールの「クラウド」タブもこれを呼ぶ）。

    bin/libra-vast start --run ls --gpu "RTX 5070 Ti" --max-dph 0.28 --hours 3
    bin/libra-vast status [--json] [--account]
    bin/libra-vast history [--json]
    bin/libra-vast stop
    bin/libra-vast offers --gpu "RTX 5070 Ti" --max-dph 0.28 [--min-cores 16 --max-inet-cost 0.02]   # 全件に落ちた理由を付ける
    bin/libra-vast cleanup --yes

1 回の起動を「セッション」と呼び、~/libra-run/cloud/<run>-<時刻>/ に置く（session.json、launcher.log、束 worker/、
vast_worker.py の出力: instance.json・setup.log・bridge/・result.json）。同時に動かせるセッションは 1 つ（cloud/current が指す）。
start は束の作成と vast_worker.py を setsid で切り離して起動し、すぐ返る（wsl.exe が終わっても動き続ける）。
stop はブリッジが動いていれば bridge/STOP を置き（ワーカーを止めて残りの局を取ってからインスタンスを消す）、
まだ借りている途中ならプロセスグループに SIGTERM を送る（インスタンスを消して抜ける）。
vastai の SDK を使うのは offers・cleanup・status --account だけ（~/.venvs/vastai の Python で動かす）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from libra_cloud import hosts, interrupts
from libra_cloud.bench import MAX_INET_COST

REPO = Path(__file__).resolve().parents[2]
PROJECT_PY = REPO / ".venv" / "bin" / "python"
VAST_PY = Path.home() / ".venvs" / "vastai" / "bin" / "python"
PROJECT_PATH = ":".join(str(REPO / p) for p in ("libra-sim/python", "libra-search/python", "libra-net", "libra-league", "libra-cloud"))
LABEL_PREFIX = "libra-"  # vast_worker.py（libra-worker）と vast_bench.py（libra-bench）のラベル

# launcher.log の行の目印と段階（最後に現れた目印の段階にする）
PHASES = (("credit $", "準備"), ("create #", "インスタンス作成"), ("ssh ready", "セットアップ"), ("bridge pid", "稼働"),
          ("bridge: stopping", "停止処理"), (" 0 usable", "終了（条件に合うオファーなし）"), ("not renting", "終了（残高不足）"),
          ("no instance started", "終了（借りられなかった）"), ("destroyed instance", "終了"),
          (" lost: ", "打ち切り"), ("relaunched: ", "終了（打ち切り・借り直し）"))  # 入札で止められた・ホストが落ちた → 残りの時間で次のセッション


def pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # 回収されていない子（ゾンビ）は終わったとみなす
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def phase_of(text: str) -> str:
    """launcher.log の中身から段階を決める（最後に現れた目印の段階）。一致する行が無ければ空文字。"""
    best, pos = "", -1
    for key, name in PHASES:
        i = text.rfind(key)
        if i > pos:
            best, pos = name, i
    return best


class Sessions:
    def __init__(self, root: Path):
        self.root = root

    def current(self) -> Path | None:
        try:
            d = self.root / (self.root / "current").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return d if d.is_dir() else None

    def create(self, run: str) -> Path:
        base = f"{run}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.root.mkdir(parents=True, exist_ok=True)
        for i in range(1, 100):  # 同じ秒に作ったら -2, -3 … を付ける（借り直しの鎖で続けて作るとき）
            d = self.root / (base if i == 1 else f"{base}-{i}")
            try:
                d.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(f"cannot create a session directory for {base}")
        (self.root / "current").write_text(d.name, encoding="utf-8")
        return d


def session_status(d: Path, tail: int = 10, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    s = read_json(d / "session.json") or {}
    log = d / "launcher.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    alive = pid_alive(s.get("pid"))
    phase = phase_of(text)
    ended = phase.startswith("終了")
    if not alive and not ended:
        phase = "異常終了（インスタンスが残っていないか確認）" if s else ""
    elif alive and not phase:
        phase = "準備"
    if alive and s.get("stop_requested") and not ended and phase != "停止処理":
        phase += "（停止を要求済み）"
    inst = read_json(d / "instance.json")
    result = read_json(d / "result.json")
    out = {"session": {**s, "dir": str(d), "alive": alive, "phase": phase}, "instance": inst, "bridge": read_json(d / "bridge" / "bridge.json"),
           "result": result, "rented_h": None, "est_cost_usd": None, "remaining_h": None,
           "log_tail": text.splitlines()[-tail:] if tail > 0 else []}
    if result and result.get("rented_h") is not None:
        out["rented_h"], out["est_cost_usd"] = result["rented_h"], result.get("est_cost_usd")
    elif inst and inst.get("t_rent") and not inst.get("destroyed"):
        h = (now - float(inst["t_rent"])) / 3600
        out["rented_h"] = round(h, 3)
        off = inst.get("offer") or {}
        out["est_cost_usd"] = round(h * float(off.get("dph_eff") or off.get("dph_total") or 0), 3)  # 入札は実効単価
    if inst and inst.get("t_bridge") and not ended and s.get("hours") is not None:
        out["remaining_h"] = round(max(0.0, float(s["hours"]) - (now - float(inst["t_bridge"])) / 3600), 2)
    return out


STALE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) workers: dropped (\d+) stale games from (\S+)", re.M)


def stale_drops(log_text: str, t0: float, t1: float, worker: str | None = None) -> int:
    """学習側の log.txt から、[t0, t1) の間に古すぎて捨てた局数を数える（workers.py の「workers: dropped N stale games from W」の行）。"""
    n = 0
    for m in STALE_RE.finditer(log_text):
        if worker and m.group(3) != worker:
            continue
        t = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        if t0 <= t < t1:
            n += int(m.group(2))
    return n


def per_million(cost, games):
    return round(float(cost) / games * 1e6, 2) if cost is not None and games and games > 0 else None


PULL_BYTES_PER_GAME = 5_300  # 対局ファイルは 100 局で約 530 kB（2026-09-15 の ls の inbox）


def transfer_usd(bridge: dict, offer: dict, weights_bytes: int | None) -> tuple[float | None, bool]:
    """転送料（$）と見積もりかどうか。送った重み・布石はホストの下り（inet_down_cost $/GB）、取ってきた局は上り（inet_up_cost）。
    バイト数の記録が無い古いセッションは、重みの送信回数 × 今の重みの大きさと、局数 × PULL_BYTES_PER_GAME で見積もる。単価が無ければ None。"""
    down, up = offer.get("inet_down_cost"), offer.get("inet_up_cost")
    if down is None or up is None:
        return None, False
    estimated = False
    push = bridge.get("push_bytes")
    if push is None:
        n = int((bridge.get("pushes") or {}).get("weights/latest.pt") or 0)
        if n and weights_bytes is None:
            return None, False
        push, estimated = n * (weights_bytes or 0), True
    pull = bridge.get("pull_bytes")
    if pull is None:
        pull, estimated = int(bridge.get("games") or 0) * PULL_BYTES_PER_GAME, True
    return round(push / 1e9 * float(down) + pull / 1e9 * float(up), 4), estimated


def history(root: Path, run_root: Path, now: float | None = None) -> dict:
    """過去のセッションを古い順に並べ、費用・回収局数・学習側が捨てた局・100 万局あたりの費用を出す（管理コンソールの「クラウド履歴」）。
    捨てた局は学習側の log.txt の行を、そのセッションのブリッジ起動から次のセッションの開始（無ければ終わりの 10 分後）までで数える。"""
    now = time.time() if now is None else now
    dirs = sorted((d for d in root.glob("*-*") if d.is_dir() and (d / "session.json").exists()),
                  key=lambda d: (read_json(d / "session.json") or {}).get("started") or 0)
    logs: dict[str, str] = {}
    wbytes: dict[str, int | None] = {}
    rows = []
    for i, d in enumerate(dirs):
        st = session_status(d, tail=0, now=now)
        s, inst, res = st["session"], st["instance"] or {}, st["result"] or {}
        b = res.get("bridge") or st["bridge"] or {}
        offer = inst.get("offer") or res.get("offer") or {}
        run = s.get("run") or "ls"
        games = int(b.get("games") or 0)
        t_bridge = inst.get("t_bridge")
        end = now if s["alive"] else b.get("time")
        bridge_h = round((float(end) - float(t_bridge)) / 3600, 3) if t_bridge and end else None
        stale = 0
        if t_bridge:
            if run not in logs:
                try:
                    logs[run] = (run_root / run / "log.txt").read_text(encoding="utf-8", errors="replace")
                except OSError:
                    logs[run] = ""
            nxt = (read_json(dirs[i + 1] / "session.json") or {}).get("started") if i + 1 < len(dirs) else None
            stale = stale_drops(logs[run], float(t_bridge), float(nxt or (end or now) + 600), res.get("worker"))
        net = games - stale
        if run not in wbytes:
            try:
                wbytes[run] = (run_root / run / "weights" / "latest.pt").stat().st_size
            except OSError:
                wbytes[run] = None
        tr, tr_est = transfer_usd(b, offer, wbytes[run])
        total = None if st["est_cost_usd"] is None else round(float(st["est_cost_usd"]) + (tr or 0.0), 4)
        rows.append({"name": d.name, "run": run, "started": s.get("started"), "alive": s["alive"], "phase": s["phase"],
                     "stopped_by_user": bool(s.get("stop_requested")), "hours": s.get("hours"), "max_dph": s.get("max_dph"),
                     "lost": bool(res.get("lost")), "continues": s.get("continues"), "t_rent": inst.get("t_rent"), "t_bridge": t_bridge,
                     "t_end": end, "last_pull": b.get("last_pull"),
                     "gpu": offer.get("gpu_name") or s.get("gpu"), "cpu": str(offer.get("cpu_name") or "").strip() or None,
                     "where": offer.get("geolocation"), "reliability": offer.get("reliability2"),
                     "dph": offer.get("dph_eff") or offer.get("dph_total"), "rent": s.get("rent") or "on-demand", "bid": offer.get("bid"),
                     "instance": inst.get("instance"), "t_ready_s": res.get("t_ready_s"),
                     "rented_h": st["rented_h"], "est_cost_usd": st["est_cost_usd"], "bridge_h": bridge_h,
                     "games": games, "stale_games": stale, "net_games": net, "files": b.get("files"),
                     "rejected_files": b.get("rejected_files"), "errors": b.get("errors"), "verify_ms_per_game": b.get("verify_ms_per_game"),
                     "games_per_day": round(games / bridge_h * 24) if bridge_h and bridge_h > 0 else None,
                     "transfer_usd": tr, "transfer_estimated": tr_est, "total_usd": total,
                     "usd_per_1m": per_million(total, net), "usd_per_1m_gross": per_million(total, games)})
    rows = interrupts.annotate(rows)
    rented = [r for r in rows if r["est_cost_usd"] is not None]
    cost = round(sum(float(r["est_cost_usd"]) for r in rented), 3)
    transfer = round(sum(float(r["transfer_usd"] or 0) for r in rented), 4)
    total = round(cost + transfer, 4)
    net = sum(r["net_games"] for r in rows)
    totals = {"sessions": len(rows), "rented": len(rented), "rented_h": round(sum(float(r["rented_h"] or 0) for r in rented), 3),
              "est_cost_usd": cost, "transfer_usd": transfer, "total_usd": total,
              "games": sum(r["games"] for r in rows), "stale_games": sum(r["stale_games"] for r in rows),
              "net_games": net, "usd_per_1m": per_million(total, net)}
    months: dict[str, dict] = {}
    for r in rows:
        key = time.strftime("%Y-%m", time.localtime(r["started"])) if r["started"] else "?"
        m = months.setdefault(key, {"month": key, "sessions": 0, "est_cost_usd": 0.0, "transfer_usd": 0.0, "total_usd": 0.0, "games": 0, "net_games": 0})
        m["sessions"] += 1
        m["est_cost_usd"] = round(m["est_cost_usd"] + float(r["est_cost_usd"] or 0), 3)
        m["transfer_usd"] = round(m["transfer_usd"] + float(r["transfer_usd"] or 0), 4)
        m["total_usd"] = round(m["total_usd"] + float(r["total_usd"] or 0), 4)
        m["games"] += r["games"]
        m["net_games"] += r["net_games"]
    return {"root": str(root), "sessions": rows, "totals": totals, "months": list(months.values()),
            "interrupts": interrupts.summary(rows)}


def cmd_history(a: argparse.Namespace) -> int:
    h = history(Path(a.root).expanduser(), Path(a.run_root).expanduser())
    if a.json:
        print(json.dumps(h, ensure_ascii=False))
        return 0
    if not h["sessions"]:
        print("セッションはまだありません")
        return 0

    def money(v, fmt="{:.2f}"):
        return "-" if v is None else "$" + fmt.format(v)

    print(f"{'開始':<11} {'GPU':<12} {'$/h':>6} {'借りた h':>8} {'費用':>7} {'回収局':>8} {'捨てた':>7} {'有効局':>8} {'局/日':>9} {'$/100万局':>9}  結果")
    for r in h["sessions"]:
        started = time.strftime("%m/%d %H:%M", time.localtime(r["started"])) if r["started"] else "-"
        print(f"{started:<11} {str(r['gpu'] or '-'):<12} {money(r['dph'], '{:.3f}'):>6} {r['rented_h'] if r['rented_h'] is not None else '-':>8} "
              f"{money(r['total_usd']):>7} {r['games']:>8,} {r['stale_games']:>7,} {r['net_games']:>8,} "
              f"{format(r['games_per_day'], ',') if r['games_per_day'] is not None else '-':>9}{money(r['usd_per_1m']):>9}  {r['phase']}")
    t = h["totals"]
    print(f"合計 {t['sessions']} 回（借りた {t['rented']} 回、{t['rented_h']:.2f} h）  費用 {money(t['total_usd'])}"
          f"（うち転送料 {money(t['transfer_usd'], '{:.3f}')}）  "
          f"有効局 {t['net_games']:,}（捨てた {t['stale_games']:,}）  100 万局あたり {money(t['usd_per_1m'])}")
    for m in h["months"]:
        print(f"  {m['month']}: {m['sessions']} 回  費用 {money(m['total_usd'])}（うち転送料 {money(m['transfer_usd'], '{:.3f}')}）  有効局 {m['net_games']:,}")
    print(waste_lines(h["interrupts"]))
    return 0


def waste_lines(w: dict) -> str:
    """打ち切りの損を人が読む形にする（コンソールの「クラウド履歴」の要約と同じ文言）。"""
    if not w["interruptions"]:
        return f"打ち切り 0 回（打った {w['bridge_h']:.1f} 時間）。"
    out = [f"打ち切り {w['interruptions']} 回（打った {w['bridge_h']:.1f} 時間、平均 {w['h_per_loss']:.1f} 時間に 1 回。"
           f"借り直し {w['relaunches']} 回、借り直せず {w['not_relaunched']} 回）",
           f"  損: 余分な準備代 ${w['extra_setup_usd']:.2f} ＋ 打てなかった {w['lost_h']:.1f} 時間 = 約 {w['lost_games']:,} 局"
           + (f"（うち借り直せずに捨てた予定 {w['unused_h']:.1f} 時間）" if w["unused_h"] else "")]
    if w["usd_per_1m"] and w["usd_per_1m_ideal"]:
        out.append(f"  お金の損 100 万局あたり ${w['usd_per_1m']:.2f}（打ち切りが無ければ ${w['usd_per_1m_ideal']:.2f}、+{w['waste_pct']:.1f}%）"
                   + (f"　時間の損 {w['lost_time_pct']:.1f}%（止まっている間は課金されないので費用には出ないが、局/日には効く）"
                      if w["lost_time_pct"] is not None else ""))
    for g in w["by_rent"]:
        out.append(f"  {g['name']}: {g['sessions']} 回・{g['bridge_h']:.1f} h、打ち切り {g['lost']} 回"
                   + (f"（{g['h_per_loss']:.1f} h に 1 回）" if g["h_per_loss"] else "")
                   + (f"、${g['dph']:.3f}/h" if g["dph"] else "")
                   + (f"、100 万局あたり ${g['usd_per_1m']:.2f}" if g["usd_per_1m"] else "")
                   + (f"（打ち切りが無ければ ${g['usd_per_1m_ideal']:.2f}）" if g["lost"] and g["usd_per_1m_ideal"] else "")
                   + (f"、時間の損 {g['lost_time_pct']:.1f}%" if g["lost_time_pct"] is not None else ""))
    return "\n".join(out)


def launch_argv(a: argparse.Namespace, run_dir: Path, d: Path) -> list[str]:
    """束を作ってから vast_worker.py を起動するコマンド（bash の exec で、プロセスの pid は起動のまま vast_worker.py になる）。"""
    prep = [str(PROJECT_PY), "-m", "libra_cloud.prepare", "--worker", "--ckpt", str(run_dir / "checkpoints" / "latest.pt"),
            "--config", str(run_dir / "config.toml"), "--out", str(d / "worker"), "--n-games", str(a.n_games)]
    work = [str(VAST_PY), "-u", str(REPO / "libra-cloud" / "vast_worker.py"), "--gpu", a.gpu, "--max-dph", str(a.max_dph),
            "--hours", str(a.hours), "--run-dir", str(run_dir), "--bundle", str(d / "worker" / "bundle.tar.gz"), "--out", str(d),
            "--min-rel", str(a.min_rel), "--min-cpu-ghz", str(a.min_cpu_ghz), "--min-cores", str(a.min_cores),
            "--max-inet-cost", str(a.max_inet_cost), "--n-games", str(a.n_games), "--rent", a.rent, "--bid-margin", str(a.bid_margin),
            "--id", getattr(a, "worker_id", "vast1")]
    return ["bash", "-c", f"{shlex.join(prep)} && exec {shlex.join(work)}"]


CONTINUE_KEYS = ("run", "gpu", "max_dph", "min_rel", "min_cpu_ghz", "min_cores", "max_inet_cost", "n_games", "rent", "bid_margin", "worker_id")
MIN_CONTINUE_H = 0.25     # 残りがこれより短ければ借り直さない（借りてから打ち始めるまで 2〜6 分かかる）
DEADLINE_SLACK_S = 1800   # セッションの鎖の締め切り = 最初の開始 + 時間 + これ（借り直しの待ちで際限なく延びないように）


def continue_settings(prev: Path, now: float) -> tuple[dict | None, str]:
    """ホストを失った前のセッションから、次のセッションの設定・残りの時間（ブリッジが動いた時間を引く）・締め切りを決める。
    借り直さないときは (None, 理由)。"""
    s = read_json(prev / "session.json") or {}
    if not s:
        return None, f"{prev.name} の session.json がありません"
    if s.get("stop_requested"):
        return None, "停止を要求されていた"
    inst = read_json(prev / "instance.json") or {}
    used = max(0.0, (now - float(inst["t_bridge"])) / 3600) if inst.get("t_bridge") else 0.0
    deadline = float(s.get("deadline") or (float(s.get("started") or now) + float(s["hours"]) * 3600 + DEADLINE_SLACK_S))
    hours = round(min(float(s["hours"]) - used, (deadline - now) / 3600), 2)
    if hours < MIN_CONTINUE_H:
        return None, f"残り {max(hours, 0.0):.2f} 時間（{MIN_CONTINUE_H} 時間未満）"
    return {**{k: s[k] for k in CONTINUE_KEYS if k in s}, "hours": hours, "deadline": deadline, "continues": prev.name}, ""


def cmd_start(a: argparse.Namespace) -> int:
    ss = Sessions(Path(a.root).expanduser())
    cur = ss.current()
    cont = None
    if getattr(a, "continue_from", None):
        prev = ss.root / a.continue_from
        if cur is not None and cur != prev and pid_alive((read_json(cur / "session.json") or {}).get("pid")):
            print(f"借り直しません: 別のセッションが動いています（{cur.name}）")
            return 3
        cont, why = continue_settings(prev, time.time())
        if cont is None:
            print(f"借り直しません: {why}")
            return 3
        for k in (*CONTINUE_KEYS, "hours"):
            if k in cont:
                setattr(a, k, cont[k])
    elif cur is not None and pid_alive((read_json(cur / "session.json") or {}).get("pid")):
        print(f"既に動いています: {cur}（止めるときは stop）")
        return 1
    run_dir = Path(a.run_root).expanduser() / a.run
    cfg_path = run_dir / "config.toml"
    if not cfg_path.exists():
        print(f"run がありません: {cfg_path}")
        return 2
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    if (cfg.get("exploiter") or {}).get("main_ckpt"):
        print(f"{a.run} は搾取者の run なので、ワーカーの局を足せません")
        return 2
    if not (cfg.get("workers") or {}).get("enabled") and not a.force:
        print(f"{cfg_path} に [workers] enabled = true がありません。書いて {a.run} を停止 → 起動してから始めてください（局を取り込めないまま課金されるため）")
        return 2
    if not (run_dir / "checkpoints" / "latest.pt").exists():
        print(f"チェックポイントがありません: {run_dir / 'checkpoints' / 'latest.pt'}")
        return 2
    d = ss.create(a.run)
    with open(d / "launcher.log", "ab") as logf:
        p = subprocess.Popen(launch_argv(a, run_dir, d), stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, cwd=str(REPO), env=dict(os.environ, PYTHONPATH=PROJECT_PATH))
    write_json(d / "session.json", {"run": a.run, "gpu": a.gpu, "max_dph": a.max_dph, "hours": a.hours, "min_rel": a.min_rel,
                                    "min_cpu_ghz": a.min_cpu_ghz, "min_cores": a.min_cores, "max_inet_cost": a.max_inet_cost, "n_games": a.n_games,
                                    "rent": a.rent, "bid_margin": a.bid_margin, "worker_id": a.worker_id, "started": time.time(), "pid": p.pid,
                                    "deadline": cont["deadline"] if cont else time.time() + a.hours * 3600 + DEADLINE_SLACK_S,
                                    "continues": cont["continues"] if cont else None})
    how = "入札" if a.rent == "bid" else "on-demand"
    print(f"起動しました: {d.name}（pid {p.pid}。{a.gpu} を{how}で最大 ${a.max_dph:.2f}/h、{a.hours:g} 時間。準備に 5〜15 分）")
    return 0


def cmd_stop(a: argparse.Namespace) -> int:
    ss = Sessions(Path(a.root).expanduser())
    cur = ss.current()
    s = read_json(cur / "session.json") if cur is not None else None
    if cur is None or s is None or not pid_alive(s.get("pid")):
        print("動いているセッションはありません")
        return 1
    if (cur / "bridge").is_dir():
        (cur / "bridge" / "STOP").write_text(str(time.time()), encoding="utf-8")
        how = "ワーカーを止めて残りの局を取ってから、インスタンスを消します（数分）"
    else:
        os.killpg(int(s["pid"]), signal.SIGTERM)
        how = "借りる途中なので中断し、インスタンスがあれば消します"
    s["stop_requested"] = time.time()
    write_json(cur / "session.json", s)
    print(f"停止を送りました: {cur.name}。{how}")
    return 0


def account() -> dict:
    try:
        from vastai.sdk import VastAI

        v = VastAI(raw=True, quiet=True)
        u = v.show_user() or {}
        inst = v.show_instances() or []
    except Exception as e:  # noqa: BLE001  ネットワーク・API キーの不備
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"credit": u.get("credit"), "time": time.time(),
            "instances": [{"id": i.get("id"), "label": i.get("label"), "status": i.get("actual_status"), "gpu": i.get("gpu_name"),
                           "dph": i.get("dph_total"), "start": i.get("start_date")} for i in inst]}


def cmd_status(a: argparse.Namespace) -> int:
    root = Path(a.root).expanduser()
    cur = Sessions(root).current()
    out: dict = {"root": str(root), "session": None}
    if cur is not None:
        out.update(session_status(cur, a.tail))
    if a.account:
        out["account"] = account()
    if a.json:
        print(json.dumps(out, ensure_ascii=False))
        return 0
    s = out.get("session")
    if not s:
        print("セッションはまだありません")
    else:
        print(f"{Path(s['dir']).name}: {s['phase']}  {s.get('gpu')}  最大 ${s.get('max_dph')}/h  {s.get('hours')} 時間")
        if out["rented_h"] is not None:
            print(f"借りた時間 {out['rented_h']:.2f} h  費用（見積もり）${out['est_cost_usd']:.2f}"
                  + (f"  残り {out['remaining_h']:.2f} h" if out["remaining_h"] is not None else ""))
        b = out.get("bridge")
        if b:
            print(f"回収 {b['games']} 局（{b['files']} ファイル）  弾いた {b['rejected_files']}  エラー {b['errors']}  検査 {b['verify_ms_per_game']} ms/局")
        for line in out["log_tail"]:
            print("  " + line)
    if a.account:
        ac = out["account"]
        if "error" in ac:
            print(f"vast.ai: {ac['error']}")
        else:
            print(f"残高 ${float(ac['credit'] or 0):.2f}  インスタンス {len(ac['instances'])} 台"
                  + "".join(f"\n  #{i['id']} {i['label']} {i['status']} {i['gpu']} ${float(i['dph'] or 0):.3f}/h" for i in ac["instances"]))
    return 0


# offer_rejects の key の表示名（管理コンソールの入力欄の名前に合わせる）と、緩めるときの注意
COND_LABELS = {"max_dph": "上限 $/h", "min_cores": "コア数の下限", "min_cpu_ghz": "CPU GHz の下限", "min_rel": "信頼度の下限",
               "max_inet_cost": "転送料の上限", "min_down": "下り回線", "min_cuda": "CUDA", "num_gpus": "GPU 枚数", "price": "価格"}
COND_FORMATS = {"max_dph": "{:.2f}", "min_cores": "{:d}", "min_cpu_ghz": "{:.1f}", "min_rel": "{:.2f}", "max_inet_cost": "{:.3f}"}
COND_NOTES = {"min_cores": "コアが少ないと自己対局のスレッドが減って局/日が落ちる",
              "min_cpu_ghz": "遅い CPU では探索が律速して GPU が遊ぶ",
              "min_rel": "途中で落ちやすい（落ちてもインスタンスは消す）",
              "max_inet_cost": "重みと局の転送で 1 日に数 GB〜数十 GB 流れる"}


def _width(s: str) -> int:
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, w: int) -> str:
    return s + " " * max(0, w - _width(s))


def _rjust(s: str, w: int) -> str:
    return " " * max(0, w - _width(s)) + s


# 表の幅に収めるため CPU 名から型番に要らない語を落とす（"AMD EPYC 7B13 64-Core Processor" → "AMD EPYC 7B13"）。
# 実績を同じ CPU でまとめる鍵と同じ関数を使う（hosts.keys_of）
_short_cpu = hosts.short_cpu


EST_MARKS = {"machine": "◎", "cpu": "○", "gpu": "△"}


def _est_games(r: dict) -> str:
    """見込みの局/日（万単位）と、どの実測から当てたかの印。実測が無ければ "-"。"""
    g = r.get("est_games_per_day")
    return "-" if not g else EST_MARKS.get(r.get("est_from"), "") + f"{g / 1e4:.1f}万"


def _est_cost(r: dict) -> str:
    c = r.get("est_usd_per_1m")
    return "-" if not c else f"${c:.2f}"


def offers_report(gpu: str, offers: list[dict], cond: dict, disk: float) -> dict:
    """検索したオファー全件に落ちた理由を付け、条件を 1 つ緩めれば通るものを添える（offers コマンドと管理コンソールの「候補を見る」）。
    cond: max_dph・min_cores・min_cpu_ghz・min_rel・max_inet_cost・min_cuda（vast_worker.py の pick_offers と同じ値）。"""
    from libra_cloud.bench import near_misses, offer_rejects, pick_offers

    cands = pick_offers(offers, **cond)
    rank = {id(o): i + 1 for i, o in enumerate(cands)}
    price_key = cond.get("price_key", "dph_total")

    def row(o: dict) -> dict:
        return {"id": o.get("id"), "dph": round(float(o.get(price_key) or 0), 3), "bid": o.get("bid"), "cpu": _short_cpu(o.get("cpu_name")),
                "cores": o.get("cpu_cores_effective"), "ghz": round(float(o.get("cpu_ghz") or 0), 2),
                "reliability": round(float(o.get("reliability2") or o.get("reliability") or 0), 3),
                "inet_cost": round(max(o.get("inet_up_cost") or 0, o.get("inet_down_cost") or 0), 4), "where": o.get("geolocation"),
                "est_games_per_day": o.get("est_games_per_day"), "est_usd_per_1m": o.get("est_usd_per_1m"), "est_from": o.get("est_from")}

    rows = []
    by_offer: dict[int, dict] = {}
    counts: dict[str, int] = {}
    for o in sorted(offers, key=lambda o: (float(o.get(price_key) or 0), -(o.get("cpu_cores_effective") or 0))):
        reasons = offer_rejects(o, **cond)
        for r in reasons:
            counts[r["key"]] = counts.get(r["key"], 0) + 1
        rows.append({**row(o), "ok": not reasons, "rank": rank.get(id(o)), "reasons": reasons})
        by_offer[id(o)] = rows[-1]
    hints = []
    for h in near_misses(offers, **cond):
        need_text = COND_FORMATS[h["key"]].format(h["need"])
        hints.append({"key": h["key"], "need": h["need"], "label": COND_LABELS[h["key"]], "need_text": need_text,
                      "note": COND_NOTES.get(h["key"], ""), "offer": row(h["offer"])})

    lines = [f"{gpu}: 検索 {len(offers)} 件、条件に合う {len(cands)} 件"
             + ("（起動すると ○1 から順に借りる。見込みの 100 万局あたりの費用の安い順）" if cands else ""),
             ("借り方: 入札（割り込みあり。$/h は最低入札に上乗せした入札額での実効単価）" if price_key == "dph_eff" and any(o.get("bid") for o in offers)
              else "借り方: on-demand"),
             f"条件: 上限 ${cond['max_dph']:.2f}/h、コア {cond['min_cores']} 以上、CPU {cond['min_cpu_ghz']:.1f} GHz 以上、"
             f"信頼度 {cond['min_rel']:.2f} 以上、転送料 ${cond['max_inet_cost']:.3f}/GB 以下",
             f"（検索の時点で 1 GPU・verified・下り 200 Mbps 以上・CUDA {cond['min_cuda']:g} 以上・ディスク {disk:g} GB 以上に絞っている）"]
    if counts:
        lines.append("落ちた理由: " + "、".join(f"{COND_LABELS.get(k, k)} {n} 件" for k, n in sorted(counts.items(), key=lambda kv: -kv[1])))
    if any(r["est_games_per_day"] is not None for r in rows):
        lines.append("見込みは過去に借りたホストの実測（定常状態の局/日）から: ◎同じ機械、○同じ GPU と CPU、△同じ GPU。"
                     "印の無いものは実測が無く、実測のあるホストの中央値を当てている")
    if rows:
        lines += ["", f"{_pad('判定', 5)}{'$/h':>6} {_rjust('局/日', 8)} {_rjust('$/100万', 8)} {_rjust('コア', 4)} {'GHz':>5} "
                      f"{_rjust('信頼度', 6)} {_rjust('転送料', 6)}  {_pad('CPU', 30)} {_pad('場所', 18)} 理由"]
        for r in rows:
            mark = f"○{r['rank']}" if r["ok"] else "×"
            cores = "-" if r["cores"] is None else f"{r['cores']:g}"
            lines.append(f"{_pad(mark, 5)}{r['dph']:>6.3f} {_rjust(_est_games(r), 8)} {_rjust(_est_cost(r), 8)} {cores:>4} {r['ghz']:>5.2f} "
                         f"{r['reliability']:>6.3f} {r['inet_cost']:>6.3f}  {_pad(r['cpu'][:30], 30)} {_pad(str(r['where'] or '-')[:18], 18)} "
                         + "、".join(x["text"] for x in r["reasons"]))
    if hints:
        lines += ["", "1 つ緩めれば借りられる:"]
        for h in hints:
            f = h["offer"]
            lines.append(f"  {h['label']} を {h['need_text']} にすると ${f['dph']:.3f}/h {f['cpu'][:30]}（{f['cores']:g} コア、{f['ghz']:.2f} GHz、"
                         f"信頼度 {f['reliability']:.3f}、{f['where']}）" + (f"  ※{h['note']}" if h["note"] else ""))
    elif not cands and offers:
        lines += ["", "1 つ緩めるだけで借りられるオファーはありません。GPU を変えるか、時間をおいて見直してください。"]
    return {"gpu": gpu, "cond": cond, "offers": len(offers), "usable": len(cands), "top": [by_offer[id(o)] for o in cands[:8]],  # 借りる順
            "all": rows, "reject_counts": counts, "hints": hints, "text": "\n".join(lines)}


def cmd_offers(a: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO / "libra-cloud"))
    from vastai.sdk import VastAI

    from libra_cloud.bench import annotate_price, offer_query
    from vast_bench import IMAGE, image_cuda

    min_cuda = image_cuda(IMAGE)
    v = VastAI(raw=True, quiet=True)
    offers = v.search_offers(query=offer_query(a.gpu, a.min_rel, min_cuda, 30), type=a.rent, order="dph_total", limit=100, storage=30) or []
    offers = annotate_price(offers, a.rent, a.bid_margin)
    offers = hosts.annotate(offers, hosts.speed_table(hosts.scan_sessions(Path(a.root).expanduser())), "dph_eff")
    cond = {"max_dph": a.max_dph, "min_cores": a.min_cores, "min_cpu_ghz": a.min_cpu_ghz, "min_rel": a.min_rel,
            "max_inet_cost": a.max_inet_cost, "min_cuda": min_cuda, "price_key": "dph_eff"}
    rep = offers_report(a.gpu, offers, cond, disk=30)
    print(json.dumps(rep, ensure_ascii=False) if a.json else rep["text"])
    return 0


def cmd_cleanup(a: argparse.Namespace) -> int:
    """libra- で始まるラベルのインスタンスをすべて消す（起動の途中で落ちて残ったものの後始末）。"""
    cur = Sessions(Path(a.root).expanduser()).current()
    if cur is not None and pid_alive((read_json(cur / "session.json") or {}).get("pid")) and not a.force:
        print(f"セッションが動いています（{cur.name}）。先に stop してください")
        return 1
    from vastai.sdk import VastAI

    v = VastAI(raw=True, quiet=True)
    mine = [i for i in (v.show_instances() or []) if str(i.get("label") or "").startswith(LABEL_PREFIX)]
    if not mine:
        print("消すインスタンスはありません")
        return 0
    for i in mine:
        print(f"#{i.get('id')} {i.get('label')} {i.get('actual_status')} {i.get('gpu_name')} ${float(i.get('dph_total') or 0):.3f}/h")
    if not a.yes:
        print("消すときは --yes を付けて実行してください")
        return 1
    for i in mine:
        v.destroy_instance(i["id"])
    time.sleep(3)
    left = [i.get("id") for i in (v.show_instances() or []) if i.get("id") in {m["id"] for m in mine}]
    print(f"消しました: {len(mine)} 台" + (f"（まだ一覧に残っている: {left}。数十秒後に status --account で確認）" if left else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="libra-vast")
    ap.add_argument("--root", default="~/libra-run/cloud", help="セッションの置き場所")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_s = sub.add_parser("start", help="GPU を借りてワーカーを起動し、学習側の run に局を足す（すぐ返る）")
    p_s.add_argument("--run", default="ls")
    p_s.add_argument("--run-root", default="~/libra-run")
    p_s.add_argument("--force", action="store_true", help="[workers] enabled でなくても起動する")
    p_s.add_argument("--worker-id", default="vast1",
                     help="ワーカー名（対局ファイル名・seed・ls の workers の内訳）。2 台目を別の --root で動かすときは vast2 など別の名前にする")
    p_s.add_argument("--continue-from", default=None,
                     help="ホストを失ったセッション名。その設定と残りの時間・締め切りで次のセッションを起動する（vast_worker.py が呼ぶ）")
    for p in (p_s, sub.add_parser("offers", help="検索したオファーを安い順に、落ちた理由と 1 つ緩めれば通る条件を付けて出す（借りない）")):
        p.add_argument("--gpu", default="RTX 5070 Ti")
        p.add_argument("--max-dph", type=float, default=0.28, help="1 時間あたりの上限（$、ストレージ込み）")
        p.add_argument("--min-rel", type=float, default=0.94)
        p.add_argument("--min-cpu-ghz", type=float, default=4.4,
                       help="CPU の最大周波数の下限。遅い CPU のホストでは探索が律速して GPU が遊ぶ（measurements.md 2026-09-14）")
        p.add_argument("--min-cores", type=int, default=16, help="実効コア数の下限（自己対局のスレッドはホストの CPU 数、12 まで）")
        p.add_argument("--max-inet-cost", type=float, default=MAX_INET_COST, help="転送料（$/GB、上り・下りの高い方）の上限")
        p.add_argument("--rent", choices=("bid", "on-demand"), default="bid",
                       help="借り方。bid は入札（割り込みあり、同じホストで on-demand より 16〜32%% 安い。2026-09-15 のユーザーの決定で既定）")
        p.add_argument("--bid-margin", type=float, default=0.1, help="入札額 = 最低入札 × (1 + この値)。上限 $/h は実効単価で判定する")
        p.add_argument("--json", action="store_true")
    p_s.add_argument("--hours", type=float, default=3.0)
    p_s.add_argument("--n-games", type=int, default=512)
    sub.add_parser("stop", help="ワーカーを止めて残りの局を取り、インスタンスを消す")
    p_st = sub.add_parser("status", help="今のセッションの段階・回収局数・費用")
    p_st.add_argument("--json", action="store_true")
    p_st.add_argument("--tail", type=int, default=10)
    p_st.add_argument("--account", action="store_true", help="vast.ai の残高と借りているインスタンスも出す")
    p_h = sub.add_parser("history", help="過去のセッションごとの費用・回収局数・捨てた局・100 万局あたりの費用")
    p_h.add_argument("--run-root", default="~/libra-run", help="学習側の run の置き場所（log.txt から捨てた局を数える）")
    p_h.add_argument("--json", action="store_true")
    p_c = sub.add_parser("cleanup", help="libra- のラベルのインスタンスをすべて消す（残ったときの後始末）")
    p_c.add_argument("--yes", action="store_true")
    p_c.add_argument("--force", action="store_true", help="セッションが動いていても消す")
    a = ap.parse_args(argv)
    if getattr(a, "gpu", None):
        a.gpu = a.gpu.replace("_", " ")  # 管理コンソールは wsl.exe に渡すので空白の代わりに _ を使う
    return {"start": cmd_start, "stop": cmd_stop, "status": cmd_status, "history": cmd_history, "offers": cmd_offers, "cleanup": cmd_cleanup}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
