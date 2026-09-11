# SPDX-License-Identifier: Apache-2.0
"""`libra` コマンド: run / pause / resume / stop / throttle / status（docs/libra-local.md §7.2）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .state import DEFAULT_ROOT, StateDir, read_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="libra", description="Libra の自己対局・学習ランナー")
    ap.add_argument("--run", default="ls", help="run-id（状態ディレクトリ ~/libra-run/<run-id>）")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="状態ディレクトリの親")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="前回状態から再開（無ければ新規）")
    p_run.add_argument("--resume", action="store_true", help="（既定と同じ。互換のため）")
    p_run.add_argument("--config", default=None, help="config.toml（初回だけ有効。以後は状態ディレクトリの写しを使う）")
    sub.add_parser("pause", help="PAUSE フラグを置く。ワーカーは現在のバッチを終えて待機")
    sub.add_parser("resume", help="PAUSE フラグを消す")
    sub.add_parser("stop", help="STOP フラグを置く。チェックポイントを書いて終了")
    p_th = sub.add_parser("throttle", help="同時進行局数を絞る")
    p_th.add_argument("--games", type=int, required=True, help="同時進行局数（0 で解除）")
    sub.add_parser("status", help="状態を表示")
    a = ap.parse_args(argv)
    sd = StateDir(Path(a.root) / a.run)
    if a.cmd == "run":
        from .runner import main_run

        main_run(sd.root, Path(a.config) if a.config else None)
        return 0
    if a.cmd == "pause":
        sd.set_flag("PAUSE")
        print("PAUSE set")
        return 0
    if a.cmd == "resume":
        sd.clear_flag("PAUSE")
        print("PAUSE cleared")
        return 0
    if a.cmd == "stop":
        sd.set_flag("STOP")
        print("STOP set")
        return 0
    if a.cmd == "throttle":
        if a.games <= 0:
            sd.clear_flag("THROTTLE")
            print("THROTTLE cleared")
        else:
            sd.set_flag("THROTTLE", str(a.games))
            print(f"THROTTLE {a.games}")
        return 0
    if a.cmd == "status":
        st = read_json(sd.status_json)
        state = sd.read_state()
        if not st and not state:
            print(f"no run at {sd.root}")
            return 1
        lock = sd.root / "run.lock"
        running = False
        if lock.exists():
            pid = lock.read_text().strip()
            running = Path(f"/proc/{pid}").exists()
        flags = [f for f in StateDir.FLAGS if sd.flag(f)]
        print(f"run: {sd.root}  process: {'running' if running else 'not running'}  flags: {flags or '-'}")
        if st:
            eng = st.get("engine", {})
            g = max(1, eng.get("games", 0) or 1)
            print(f"time {st['time']}  step {st['step']}  generation {st['generation']}  games_total {st['games_total']}  "
                  f"games/day(1h) {st['games_per_day_1h']}  window {st['window_games']}  elapsed {st['elapsed_h']} h  active {st['active_games']}")
            if eng:
                print(f"session: games {eng.get('games')}  avg plies {eng.get('plies_sum', 0) / g:.1f}  "
                      f"sente/draw/gote {eng.get('sente_wins')}/{eng.get('draws')}/{eng.get('gote_wins')}  "
                      f"ruling41 {eng.get('ruling41')}  mate {eng.get('no_legal_move')}  sennichite {eng.get('sennichite')}  "
                      f"perpetual {eng.get('perpetual_check')}  max_ply {eng.get('max_ply')}  sims/move {eng.get('sims', 0) / max(1, eng.get('moves', 1)):.1f}")
            if st.get("train"):
                print("train:", json.dumps(st["train"]))
            if st.get("restarts"):
                print("restarts:", ", ".join(st["restarts"]))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
