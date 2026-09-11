# SPDX-License-Identifier: Apache-2.0
"""テスト用のランダム USI エンジン（布石拡張対応）。合法手から無作為に 1 手返し、winrate は乱数。"""
import random
import sys

import librashogi as ls

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
position = "position fuseki"
for line in sys.stdin:
    t = line.split()
    if not t:
        continue
    if t[0] == "usi":
        print("id name RandomUSI\noption name Fuseki_Mode type combo default tenbin var tenbin var fuseki\nusiok", flush=True)
    elif t[0] == "isready":
        print("readyok", flush=True)
    elif t[0] == "position":
        position = line.strip()
    elif t[0] == "go":
        p = ls.Position()
        p.set_position(position)
        res, reason = p.outcome()
        if res != "ongoing":
            print("bestmove win" if reason == "ruling41" else "bestmove resign", flush=True)
            continue
        moves = p.legal_moves()
        m = rng.choice(moves)
        w = rng.random()
        print(f"info depth 1 multipv 1 score cp 0 winrate {w:.3f} pv {m}", flush=True)
        print(f"bestmove {m}", flush=True)
    elif t[0] == "quit":
        break
