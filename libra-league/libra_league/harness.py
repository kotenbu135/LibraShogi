# SPDX-License-Identifier: Apache-2.0
"""計測ハーネス: 2 つの USI エンジンを無人で対局させ、docs/rules.md で裁定し、棋譜を JSONL に残す。

docs/protocol.md §2 のとおり、両玉は置く側のエンジンに打たせ、選ぶ側はその局面に `go` して返る
`winrate`（手番＝先手の勝率）が 0.5 以上なら先手を取る。40 手完了時の `bestmove win` と本将棋の
宣言 `win` は libra-sim で条件を検証して受ける。非合法手は負け。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import librashogi as ls

from .usi_client import UsiEngine


def cp_to_winrate(cp: int) -> float:
    from librashogi.usi import cp_to_winrate as f

    return f(cp)


class Match:
    def __init__(self, a: UsiEngine, b: UsiEngine, go_args: str, max_ply: int, count_from_41: bool, log=None):
        self.a, self.b = a, b
        self.go_args = go_args
        self.max_ply = max_ply
        self.count_from_41 = count_from_41
        self.log = log or (lambda s: None)

    def play(self, placer: str, game_no: int) -> dict:
        """placer: 'a' | 'b'。1 局を指して記録（dict）を返す。"""
        eng = {"a": self.a, "b": self.b}
        P = eng[placer]
        C = eng["b" if placer == "a" else "a"]
        for e in (self.a, self.b):
            e.new_game()
        pos = ls.Position()
        pos.set_max_ply(self.max_ply, self.count_from_41)
        tokens: list[str] = []
        moves_info: list[dict] = []
        result = reason = None
        illegal_by = None
        # 両玉
        for ply in range(2):
            line = "position fuseki" + (" moves " + " ".join(tokens) if tokens else "")
            bm, info = P.go(line, self.go_args)
            if not (bm.startswith("K*") and pos.is_legal(bm)):
                illegal_by = placer
                break
            pos.do_move(bm)
            tokens.append(bm)
            moves_info.append({"by": placer, "move": bm, **info})
        chosen = None
        if illegal_by is None:
            bm, info = C.go("position fuseki moves " + " ".join(tokens), self.go_args)
            w = info.get("winrate")
            if w is None and "cp" in info:
                w = cp_to_winrate(info["cp"])
            if w is None:
                w = 0.5
            chosen = "sente" if w >= 0.5 else "gote"
            tokens.append(f"choose:{chosen}")
            moves_info.append({"by": "b" if placer == "a" else "a", "choose": chosen, "winrate": w})
        # 席
        chooser = "b" if placer == "a" else "a"
        side = {chosen: chooser, ("gote" if chosen == "sente" else "sente"): placer} if chosen else {}
        while illegal_by is None and not pos.is_over():
            turn = pos.turn
            who = side[turn]
            E = eng[who]
            if pos.phase == "fuseki":
                line = "position fuseki moves " + " ".join(t for t in tokens if not t.startswith("choose:"))
            else:
                normal = [t for t in tokens if t.startswith("n:")]
                line = f"position sfen {self.sfen41} moves {' '.join(m[2:] for m in normal)}".rstrip()
            bm, info = E.go(line, self.go_args)
            rec = {"by": who, "move": bm, "ply": pos.ply + 1, **info}
            if bm == "resign":
                pos.resign(turn)
                moves_info.append(rec)
                break
            if bm == "win":
                if pos.phase == "normal":
                    pos.declare(turn)  # 正当なら勝ち、不当なら負け
                else:
                    illegal_by = who
                moves_info.append(rec)
                break
            if not pos.is_legal(bm):
                illegal_by = who
                moves_info.append(rec)
                break
            pos.do_move(bm)
            if pos.phase == "normal":
                if pos.ply == 40:
                    self.sfen41 = pos.sfen()
                    tokens.append(bm)
                    if pos.is_over() and pos.outcome()[1] == "ruling41":
                        # 40 手完了で裁定に当たる: 手番（先手）に go を送って bestmove win を確かめる
                        E2 = eng[side["sente"]]
                        bm2, _ = E2.go(f"position sfen {self.sfen41}", self.go_args)
                        rec["ruling41_reply"] = bm2
                else:
                    tokens.append("n:" + bm)
            else:
                tokens.append(bm)
            moves_info.append(rec)
        if illegal_by is not None:
            pos.illegal_move("sente" if side.get("sente") == illegal_by else "gote") if side else pos.illegal_move("sente")
        result, reason = pos.outcome()
        winner_engine = None
        if result in ("sente", "gote") and side:
            winner_engine = side[result]
        return {
            "game": game_no,
            "placer": placer,
            "chosen": chosen,
            "sente": side.get("sente"),
            "gote": side.get("gote"),
            "tokens": " ".join(t[2:] if t.startswith("n:") else t for t in tokens),
            "result": result,
            "reason": reason,
            "winner": winner_engine,
            "plies": pos.ply,
            "sfen41": getattr(self, "sfen41", None) if pos.phase == "normal" else None,
            "moves": moves_info,
        }


def run_match(a: UsiEngine, b: UsiEngine, n_games: int, go_args: str, out_jsonl: Path, max_ply: int = 256,
              count_from_41: bool = True, log=None, first_placer: str = "a") -> dict:
    m = Match(a, b, go_args, max_ply, count_from_41, log)
    summary = {"a": a.id_name, "b": b.id_name, "n": 0, "a_points": 0.0, "by_engine_side": {}, "reasons": {}, "games": []}
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for i in range(n_games):
            placer = first_placer if i % 2 == 0 else ("b" if first_placer == "a" else "a")
            t0 = time.time()
            g = m.play(placer, i)
            g["seconds"] = round(time.time() - t0, 1)
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
            f.flush()
            summary["n"] += 1
            pts = 1.0 if g["winner"] == "a" else 0.5 if g["result"] == "draw" else 0.0
            summary["a_points"] += pts
            key = f"a_as_{'sente' if g['sente'] == 'a' else 'gote'}"
            d = summary["by_engine_side"].setdefault(key, {"w": 0, "d": 0, "l": 0})
            d["w" if pts == 1.0 else "d" if pts == 0.5 else "l"] += 1
            summary["reasons"][g["reason"]] = summary["reasons"].get(g["reason"], 0) + 1
            summary["games"].append({k: g[k] for k in ("game", "placer", "chosen", "sente", "result", "reason", "plies", "seconds")})
            if log:
                log(f"game {i + 1}/{n_games}: placer={placer} chosen={g['chosen']} result={g['result']} ({g['reason']}) plies={g['plies']} {g['seconds']}s  A {summary['a_points']}/{summary['n']}")
    return summary
