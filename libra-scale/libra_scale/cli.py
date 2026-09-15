# SPDX-License-Identifier: Apache-2.0
"""`libra-scale build|verify|show`。出力は既定で ~/libra-run/<run>/scale/scale.json。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from libra_league.state import DEFAULT_ROOT

from .pairs import from_usi
from .table import build_table, load_model
from .verify import apply_verification, verify_pairs


def log(s: str) -> None:
    print(s, file=sys.stderr, flush=True)


def write_atomic(path: Path, table: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(table, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="libra-scale")
    ap.add_argument("--run", default="ls")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="探索値 V̂ と自己対局の実測から scale.json を作る")
    b.add_argument("--model", default=None, help="既定: <run>/checkpoints/latest.pt")
    b.add_argument("--sims", type=int, default=1600)
    b.add_argument("--concurrent", type=int, default=512)
    b.add_argument("--threads", type=int, default=8)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--margin", type=float, default=0.02)
    b.add_argument("--no-games", action="store_true", help="自己対局棋譜の集計を省く")
    b.add_argument("--out", default=None)
    v = sub.add_parser("verify", help="釣り合い候補の上位ペアで検証対局を回し、balanced を決め直す")
    v.add_argument("--model", default=None)
    v.add_argument("--table", default=None, help="既定: <run>/scale/scale.json")
    v.add_argument("--top", type=int, default=48, help="|V̂ − 0.5| の小さい順に何ペア検証するか")
    v.add_argument("--games", type=int, default=100, help="ペアあたりの局数")
    v.add_argument("--sims", type=int, default=96)
    v.add_argument("--concurrent", type=int, default=256)
    v.add_argument("--threads", type=int, default=8)
    v.add_argument("--seed", type=int, default=1)
    v.add_argument("--out", default=None)
    s = sub.add_parser("show", help="balanced と上位ペアを表示")
    s.add_argument("--table", default=None)
    s.add_argument("--top", type=int, default=20)
    q = sub.add_parser("seq", help="全組の逐次の検証対局（組ごとに打ち切り、対称な組を先に打って後手に傾いたらアラート）")
    qs = q.add_subparsers(dest="seq_cmd", required=True)
    qr = qs.add_parser("run", help="手元の GPU で打つ（初回は --table で状態のディレクトリを作る。再開は --dir だけ）")
    qr.add_argument("--dir", required=True)
    qr.add_argument("--table", default=None, help="初回だけ: V̂ を持つ scale.json（build の出力。モデルもここから）")
    qr.add_argument("--pairs", default="", help="初回だけ: 組を限る（例 5i5a,5h5b。既定は剪定後の全 492 組）")
    qr.add_argument("--sims", type=int, default=96)
    qr.add_argument("--eps", type=float, default=0.02)
    qr.add_argument("--min-games", type=int, default=100)
    qr.add_argument("--look-every", type=int, default=50)
    qr.add_argument("--eps-place", type=float, default=None, help="置く側の精度（0 で 2 段目なし）。再開時にも変えられる")
    qr.add_argument("--n-games", type=int, default=512)
    qr.add_argument("--threads", type=int, default=12)
    qr.add_argument("--compile", default="max-autotune", choices=("none", "default", "max-autotune"))
    qr.add_argument("--worker", default="local")
    qr.add_argument("--notify", choices=("none", "windows"), default="none", help="windows: アラートを Windows の通知領域にも出す（WSL の powershell.exe）")
    qst = qs.add_parser("status", help="進み具合・打ち切りの内訳・アラート・対称な組の勝率")
    qst.add_argument("--dir", required=True)
    qw = qs.add_parser("worker", help="別マシンで打つだけ（--dir に config.json・weights.pt・active.json。libra-cloud の host_scale.sh が起動）")
    qw.add_argument("--dir", required=True)
    qw.add_argument("--id", required=True)
    qw.add_argument("--n-games", type=int, default=512)
    qw.add_argument("--threads", type=int, default=12)
    qw.add_argument("--compile", default="max-autotune", choices=("none", "default", "max-autotune"))
    qt = qs.add_parser("table", help="scale.json（version 1）を書く")
    qt.add_argument("--dir", required=True)
    qt.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "seq":
        return seq_main(a)
    run = Path(a.root) / a.run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.cmd == "build":
        model, info = load_model(Path(a.model) if a.model else run / "checkpoints" / "latest.pt", device, torch.float16)
        table = build_table(model, info, a.sims, a.concurrent, a.threads, a.seed, device,
                            None if a.no_games else run / "games", a.margin, log=log)
        out = Path(a.out) if a.out else run / "scale" / "scale.json"
        write_atomic(out, table)
        log(f"wrote {out}: {table['n_pairs_unique']} pairs, balanced {len(table['balanced'])}")
        return 0
    if a.cmd == "verify":
        tpath = Path(a.table) if a.table else run / "scale" / "scale.json"
        table = json.loads(tpath.read_text())
        model, info = load_model(Path(a.model) if a.model else Path(table["model"]["path"]), device, torch.float16)
        ent = sorted(table["pairs"], key=lambda e: abs(e["v_hat"] - 0.5))[: a.top]
        pairs = [(from_usi(e["kb"]), from_usi(e["kw"])) for e in ent]
        counts = verify_pairs(model, pairs, a.games, a.sims, a.concurrent, a.threads, a.seed, device, log=log)
        table = apply_verification(table, counts, a.sims, a.games)
        out = Path(a.out) if a.out else tpath
        write_atomic(out, table)
        log(f"wrote {out}: verified {len(pairs)} pairs, balanced {len(table['balanced'])}")
        vb = table.get("verify") or {}
        if "v_hat_minus_w" in vb:
            log(f"v_hat - verify winrate: mean {vb['v_hat_minus_w']:+.4f} (se {vb['v_hat_minus_w_se']:.4f}) -> docs/measurements.md に書く")
        return 0
    if a.cmd == "show":
        tpath = Path(a.table) if a.table else run / "scale" / "scale.json"
        table = json.loads(tpath.read_text())
        print(f"model step {table['model'].get('step')} sims {table['sims']} balanced {len(table['balanced'])}: {table['balanced'][:12]}")
        for e in sorted(table["pairs"], key=lambda e: abs(e["v_hat"] - 0.5))[: a.top]:
            v = e.get("verify")
            print(f"  {e['kb']} {e['kw']}  v_hat {e['v_hat']:.3f}  selfplay {e['selfplay']['games']:4d} games wr {e['selfplay']['winrate']}"
                  + (f"  verify {v['games']} games wr {v['winrate']} ci {v['ci95']}" if v else ""))
        return 0
    return 1


def seq_main(a) -> int:
    import signal

    from . import seqrule as R
    from . import seqrun as S

    d = Path(a.dir).expanduser()
    if a.seq_cmd == "status":
        print("\n".join(S.status_lines(d)))
        return 0
    if a.seq_cmd == "table":
        co = S.Coordinator(d, log=lambda s: None)
        table = R.build_table(S.read_json(Path(co.config["base_table"])), co.config, co.state)
        write_atomic(Path(a.out).expanduser(), table)
        v = table["verify"]
        log(f"wrote {a.out}: {v['pairs']} pairs verified ({v['games']:,} games, complete {v['complete']}), balanced {len(table['balanced'])}")
        return 0
    if a.seq_cmd == "worker":
        stop_w: list[int] = []
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda signum, frame: stop_w.append(signum))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return S.run_worker(d, device, a.id, a.n_games, a.threads, a.compile, log=lambda s: log(f"{time.strftime('%H:%M:%S')} {s}"),
                            should_stop=lambda: bool(stop_w))
    if not (d / "config.json").exists():
        if not a.table:
            log(f"{d} has no config.json; pass --table to start")
            return 2
        rule = R.Rule(eps=a.eps, min_games=a.min_games, look_every=a.look_every, eps_place=a.eps_place or 0.0)
        S.init_dir(d, Path(a.table).expanduser(), a.sims, rule, S.parse_pairs(a.pairs) if a.pairs else None)
        log(f"seq: created {d}")
    elif a.eps_place is not None:
        S.Coordinator(d, log=log).set_eps_place(a.eps_place)
    stop: list[int] = []
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: stop.append(signum))

    def log_file(s: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
        log(line)
        with open(d / "log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return S.run_local(d, device, a.n_games, a.threads, a.compile, a.worker, log=log_file, should_stop=lambda: bool(stop),
                       notify=S.windows_notify if a.notify == "windows" else None)


if __name__ == "__main__":
    sys.exit(main())
