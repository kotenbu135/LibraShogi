# SPDX-License-Identifier: Apache-2.0
"""vast.ai の自己対局ワーカーを起動・停止・確認するコマンド（bin/libra-vast。管理コンソールの「クラウド」タブもこれを呼ぶ）。

    bin/libra-vast start --run ls --gpu "RTX 5070 Ti" --max-dph 0.28 --hours 3
    bin/libra-vast status [--json] [--account]
    bin/libra-vast history [--json]
    bin/libra-vast stop
    bin/libra-vast offers --gpu "RTX 5070 Ti" --max-dph 0.28
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

REPO = Path(__file__).resolve().parents[2]
PROJECT_PY = REPO / ".venv" / "bin" / "python"
VAST_PY = Path.home() / ".venvs" / "vastai" / "bin" / "python"
PROJECT_PATH = ":".join(str(REPO / p) for p in ("libra-sim/python", "libra-search/python", "libra-net", "libra-league", "libra-cloud"))
LABEL_PREFIX = "libra-"  # vast_worker.py（libra-worker）と vast_bench.py（libra-bench）のラベル

# launcher.log の行の目印と段階（最後に現れた目印の段階にする）
PHASES = (("credit $", "準備"), ("create #", "インスタンス作成"), ("ssh ready", "セットアップ"), ("bridge pid", "稼働"),
          ("bridge: stopping", "停止処理"), (" 0 usable", "終了（条件に合うオファーなし）"), ("not renting", "終了（残高不足）"),
          ("no instance started", "終了（借りられなかった）"), ("destroyed instance", "終了"))


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
        d = self.root / f"{run}-{time.strftime('%Y%m%d-%H%M%S')}"
        d.mkdir(parents=True)
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
        out["est_cost_usd"] = round(h * float((inst.get("offer") or {}).get("dph_total") or 0), 3)
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


def history(root: Path, run_root: Path, now: float | None = None) -> dict:
    """過去のセッションを古い順に並べ、費用・回収局数・学習側が捨てた局・100 万局あたりの費用を出す（管理コンソールの「クラウド履歴」）。
    捨てた局は学習側の log.txt の行を、そのセッションのブリッジ起動から次のセッションの開始（無ければ終わりの 10 分後）までで数える。"""
    now = time.time() if now is None else now
    dirs = sorted((d for d in root.glob("*-*") if d.is_dir() and (d / "session.json").exists()),
                  key=lambda d: (read_json(d / "session.json") or {}).get("started") or 0)
    logs: dict[str, str] = {}
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
        rows.append({"name": d.name, "run": run, "started": s.get("started"), "alive": s["alive"], "phase": s["phase"],
                     "stopped_by_user": bool(s.get("stop_requested")), "hours": s.get("hours"), "max_dph": s.get("max_dph"),
                     "gpu": offer.get("gpu_name") or s.get("gpu"), "cpu": str(offer.get("cpu_name") or "").strip() or None,
                     "where": offer.get("geolocation"), "reliability": offer.get("reliability2"), "dph": offer.get("dph_total"),
                     "instance": inst.get("instance"), "t_ready_s": res.get("t_ready_s"),
                     "rented_h": st["rented_h"], "est_cost_usd": st["est_cost_usd"], "bridge_h": bridge_h,
                     "games": games, "stale_games": stale, "net_games": net, "files": b.get("files"),
                     "rejected_files": b.get("rejected_files"), "errors": b.get("errors"), "verify_ms_per_game": b.get("verify_ms_per_game"),
                     "games_per_day": round(games / bridge_h * 24) if bridge_h and bridge_h > 0 else None,
                     "usd_per_1m": per_million(st["est_cost_usd"], net), "usd_per_1m_gross": per_million(st["est_cost_usd"], games)})
    rented = [r for r in rows if r["est_cost_usd"] is not None]
    cost = round(sum(float(r["est_cost_usd"]) for r in rented), 3)
    net = sum(r["net_games"] for r in rows)
    totals = {"sessions": len(rows), "rented": len(rented), "rented_h": round(sum(float(r["rented_h"] or 0) for r in rented), 3),
              "est_cost_usd": cost, "games": sum(r["games"] for r in rows), "stale_games": sum(r["stale_games"] for r in rows),
              "net_games": net, "usd_per_1m": per_million(cost, net)}
    months: dict[str, dict] = {}
    for r in rows:
        key = time.strftime("%Y-%m", time.localtime(r["started"])) if r["started"] else "?"
        m = months.setdefault(key, {"month": key, "sessions": 0, "est_cost_usd": 0.0, "games": 0, "net_games": 0})
        m["sessions"] += 1
        m["est_cost_usd"] = round(m["est_cost_usd"] + float(r["est_cost_usd"] or 0), 3)
        m["games"] += r["games"]
        m["net_games"] += r["net_games"]
    return {"root": str(root), "sessions": rows, "totals": totals, "months": list(months.values())}


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
              f"{money(r['est_cost_usd']):>7} {r['games']:>8,} {r['stale_games']:>7,} {r['net_games']:>8,} "
              f"{format(r['games_per_day'], ',') if r['games_per_day'] is not None else '-':>9}{money(r['usd_per_1m']):>9}  {r['phase']}")
    t = h["totals"]
    print(f"合計 {t['sessions']} 回（借りた {t['rented']} 回、{t['rented_h']:.2f} h）  費用 {money(t['est_cost_usd'])}  "
          f"有効局 {t['net_games']:,}（捨てた {t['stale_games']:,}）  100 万局あたり {money(t['usd_per_1m'])}")
    for m in h["months"]:
        print(f"  {m['month']}: {m['sessions']} 回  費用 {money(m['est_cost_usd'])}  有効局 {m['net_games']:,}")
    return 0


def launch_argv(a: argparse.Namespace, run_dir: Path, d: Path) -> list[str]:
    """束を作ってから vast_worker.py を起動するコマンド（bash の exec で、プロセスの pid は起動のまま vast_worker.py になる）。"""
    prep = [str(PROJECT_PY), "-m", "libra_cloud.prepare", "--worker", "--ckpt", str(run_dir / "checkpoints" / "latest.pt"),
            "--config", str(run_dir / "config.toml"), "--out", str(d / "worker"), "--n-games", str(a.n_games)]
    work = [str(VAST_PY), "-u", str(REPO / "libra-cloud" / "vast_worker.py"), "--gpu", a.gpu, "--max-dph", str(a.max_dph),
            "--hours", str(a.hours), "--run-dir", str(run_dir), "--bundle", str(d / "worker" / "bundle.tar.gz"), "--out", str(d),
            "--min-rel", str(a.min_rel), "--min-cpu-ghz", str(a.min_cpu_ghz), "--min-cores", str(a.min_cores), "--n-games", str(a.n_games)]
    return ["bash", "-c", f"{shlex.join(prep)} && exec {shlex.join(work)}"]


def cmd_start(a: argparse.Namespace) -> int:
    ss = Sessions(Path(a.root).expanduser())
    cur = ss.current()
    if cur is not None and pid_alive((read_json(cur / "session.json") or {}).get("pid")):
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
                                    "min_cpu_ghz": a.min_cpu_ghz, "min_cores": a.min_cores, "n_games": a.n_games,
                                    "started": time.time(), "pid": p.pid})
    print(f"起動しました: {d.name}（pid {p.pid}。{a.gpu} を最大 ${a.max_dph:.2f}/h で {a.hours:g} 時間。準備に 5〜15 分）")
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


def cmd_offers(a: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO / "libra-cloud"))
    from vastai.sdk import VastAI

    from libra_cloud.bench import offer_query, pick_offers
    from vast_bench import IMAGE, image_cuda

    min_cuda = image_cuda(IMAGE)
    v = VastAI(raw=True, quiet=True)
    offers = v.search_offers(query=offer_query(a.gpu, a.min_rel, min_cuda, 30), type="on-demand", order="dph_total", limit=100, storage=30) or []
    cands = pick_offers(offers, max_dph=a.max_dph, min_cores=a.min_cores, min_cpu_ghz=a.min_cpu_ghz, min_cuda=min_cuda, min_rel=a.min_rel)
    rows = [{"id": o["id"], "dph": round(float(o["dph_total"]), 3), "cpu": str(o.get("cpu_name") or "").strip(), "cores": o.get("cpu_cores_effective"),
             "reliability": round(float(o.get("reliability2") or 0), 3), "where": o.get("geolocation")} for o in cands[:8]]
    if a.json:
        print(json.dumps({"gpu": a.gpu, "offers": len(offers), "usable": len(cands), "top": rows}, ensure_ascii=False))
    else:
        print(f"{a.gpu}: 検索 {len(offers)} 件、条件に合う {len(cands)} 件")
        for r in rows:
            print(f"  ${r['dph']:.3f}/h  {r['cpu'][:34]}  {r['cores']} コア  信頼度 {r['reliability']}  {r['where']}")
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
    for p in (p_s, sub.add_parser("offers", help="条件に合うオファーを安い順に出す（借りない）")):
        p.add_argument("--gpu", default="RTX 5070 Ti")
        p.add_argument("--max-dph", type=float, default=0.28, help="1 時間あたりの上限（$、ストレージ込み）")
        p.add_argument("--min-rel", type=float, default=0.94)
        p.add_argument("--min-cpu-ghz", type=float, default=4.4,
                       help="CPU の最大周波数の下限。遅い CPU のホストでは探索が律速して GPU が遊ぶ（measurements.md 2026-09-14）")
        p.add_argument("--min-cores", type=int, default=16)
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
