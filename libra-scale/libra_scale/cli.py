# SPDX-License-Identifier: Apache-2.0
"""`libra-scale build|verify|show`。出力は既定で ~/libra-run/<run>/scale/scale.json。"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
    a = ap.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
