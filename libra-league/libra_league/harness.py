# SPDX-License-Identifier: Apache-2.0
"""計測ハーネス: 2 つの USI エンジンを無人で対局させ、docs/rules.md で裁定し、棋譜を JSONL に残す。

docs/protocol.md §2 のとおり、両玉は置く側のエンジンに打たせ、選ぶ側はその局面に `go` して返る
`winrate`（手番＝先手の勝率）が 0.5 以上なら先手を取る。40 手完了時の `bestmove win` と本将棋の
宣言 `win` は libra-sim で条件を検証して受ける。非合法手は負け。

**布石を a 側（Libra）だけで打つ形**もある（`self_fuseki`）。両陣とも同じ Libra が作り、41 手目の局面から
本将棋だけを a 対 b で指す。同じ局面を先後入れ替えて 2 局ずつ打つので、布石の有利不利は打ち消し合う。
相手が布石を指せないふつうの将棋エンジンでも計測できる（docs/runbook.md §7.0、2026-09-20 のユーザーの
決定「40 手目までは Elo 測定で測った最強 Libra 同士で行い、41 手目から 最強 Libra vs やねうら王/水匠5」）。
`openings`（41 手目の局面の一覧）を渡せば、作る代わりにそれを使う。
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
    def __init__(self, a: UsiEngine, b: UsiEngine, go_args: str, max_ply: int, count_from_41: bool, log=None,
                 go_args_b: str | None = None):
        self.a, self.b = a, b
        self.go_args = go_args
        # b 側だけ別の `go`。読む量に差を付けて測る（ハンデ）ために使う。既定は a と同じ
        self.go_args_b = go_args_b or go_args
        self.max_ply = max_ply
        self.count_from_41 = count_from_41
        self.log = log or (lambda s: None)

    def go(self, E: UsiEngine, line: str):
        """その側の `go` の引数で読ませる。"""
        return E.go(line, self.go_args_b if E is self.b else self.go_args)

    def play_moves(self, pos, side: dict, tokens: list[str], moves_info: list[dict]) -> str | None:
        """終局まで指させる。非合法手を指した側（'a' | 'b'）を返す（無ければ None）。"""
        eng = {"a": self.a, "b": self.b}
        while not pos.is_over():
            turn = pos.turn
            who = side[turn]
            E = eng[who]
            if pos.phase == "fuseki":
                line = "position fuseki moves " + " ".join(t for t in tokens if not t.startswith("choose:"))
            else:
                normal = [t for t in tokens if t.startswith("n:")]
                line = f"position sfen {self.sfen41} moves {' '.join(m[2:] for m in normal)}".rstrip()
            bm, info = self.go(E, line)
            rec = {"by": who, "move": bm, "ply": pos.ply + 1, **info}
            if bm == "resign":
                pos.resign(turn)
                moves_info.append(rec)
                return None
            if bm == "win":
                if pos.phase == "normal":
                    pos.declare(turn)  # 正当なら勝ち、不当なら負け
                else:
                    moves_info.append(rec)
                    return who
                moves_info.append(rec)
                return None
            if not pos.is_legal(bm):
                moves_info.append(rec)
                return who
            pos.do_move(bm)
            if pos.phase == "normal":
                if pos.ply == 40:
                    self.sfen41 = pos.sfen()
                    tokens.append(bm)
                    if pos.is_over() and pos.outcome()[1] == "ruling41":
                        # 40 手完了で裁定に当たる: 手番（先手）に go を送って bestmove win を確かめる
                        E2 = eng[side["sente"]]
                        bm2, _ = self.go(E2, f"position sfen {self.sfen41}")
                        rec["ruling41_reply"] = bm2
                else:
                    tokens.append("n:" + bm)
            else:
                tokens.append(bm)
            moves_info.append(rec)
        return None

    def make_opening(self, tries: int = 5) -> tuple[str, list[str]] | None:
        """a 側のエンジンだけで布石 40 手を打ち、(41 手目の局面, 打った手順) を返す。

        置く側も選ぶ側も無く、**両陣とも同じ Libra が作る**。根の Gumbel ノイズで手が散るので局ごとに
        違う布石になる。41 手目の裁定（docs/rules.md §3.4）で終わった布石は捨てて打ち直す。
        """
        for _ in range(tries):
            self.a.new_game()
            pos = ls.Position()
            pos.set_max_ply(self.max_ply, self.count_from_41)
            tokens: list[str] = []
            while pos.phase == "fuseki":
                line = "position fuseki" + (" moves " + " ".join(tokens) if tokens else "")
                bm, _ = self.go(self.a, line)
                if not pos.is_legal(bm):
                    self.log(f"布石で非合法手（{bm}）。打ち直す")
                    break
                pos.do_move(bm)
                tokens.append(bm)
            else:
                if not pos.is_over() and pos.legal_moves():
                    return pos.sfen(), tokens
                self.log(f"布石が 41 手目で終わった（{pos.outcome()[1]}）。打ち直す")
        return None

    def play_from_sfen(self, sfen41: str, sente: str, game_no: int, fuseki: list[str] | None = None) -> dict:
        """41 手目の局面から本将棋だけを対局する（布石は持ち込み）。sente: 'a' | 'b'。"""
        for e in (self.a, self.b):
            e.new_game()
        pos = ls.Position()
        pos.set_sfen(sfen41, "normal")
        pos.set_max_ply(self.max_ply, self.count_from_41)
        self.sfen41 = sfen41
        side = {"sente": sente, "gote": "b" if sente == "a" else "a"}
        tokens: list[str] = []
        moves_info: list[dict] = []
        illegal_by = self.play_moves(pos, side, tokens, moves_info)
        if illegal_by is not None:
            pos.illegal_move("sente" if side["sente"] == illegal_by else "gote")
        result, reason = pos.outcome()
        return {
            "game": game_no,
            "placer": None,
            "chosen": None,
            "sente": side["sente"],
            "gote": side["gote"],
            "tokens": " ".join(t[2:] if t.startswith("n:") else t for t in tokens),
            "result": result,
            "reason": reason,
            "winner": side[result] if result in ("sente", "gote") else None,
            "plies": pos.ply,
            "sfen41": sfen41,
            "fuseki": " ".join(fuseki) if fuseki else None,
            "moves": moves_info,
        }

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
            bm, info = self.go(P, line)
            if not (bm.startswith("K*") and pos.is_legal(bm)):
                illegal_by = placer
                break
            pos.do_move(bm)
            tokens.append(bm)
            moves_info.append({"by": placer, "move": bm, **info})
        chosen = None
        if illegal_by is None:
            bm, info = self.go(C, "position fuseki moves " + " ".join(tokens))
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
        if illegal_by is None:
            illegal_by = self.play_moves(pos, side, tokens, moves_info)
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


def run_match(a: UsiEngine, b: UsiEngine, n_games: int, go_args: str, out_jsonl: Path, max_ply: int = 320,
              count_from_41: bool = True, log=None, first_placer: str = "a", go_args_b: str | None = None,
              openings: list[str] | None = None, self_fuseki: bool = False) -> dict:
    """self_fuseki なら布石を a 側だけで打ち（両陣とも同じ Libra）、41 手目から本将棋を a 対 b で指す。
    openings（41 手目の局面の一覧）を渡せば布石を作らずそれを使う。どちらも**同じ局面を先後入れ替えて
    2 局ずつ**打つので、布石の有利不利が打ち消し合う。"""
    m = Match(a, b, go_args, max_ply, count_from_41, log, go_args_b=go_args_b)
    summary = {"a": a.id_name, "b": b.id_name, "n": 0, "a_points": 0.0, "by_engine_side": {}, "reasons": {}, "games": [],
               "go_a": go_args, "go_b": m.go_args_b, "openings": len(openings) if openings else 0,
               "self_fuseki": bool(self_fuseki)}
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    opening: tuple[str, list[str]] | None = None
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for i in range(n_games):
            placer = first_placer if i % 2 == 0 else ("b" if first_placer == "a" else "a")
            t0 = time.time()
            if openings or self_fuseki:
                # 同じ局面で先後を入れ替えて 2 局ずつ（偶数局は a が先手）
                if openings:
                    opening = (openings[(i // 2) % len(openings)], [])
                elif i % 2 == 0 or opening is None:
                    opening = m.make_opening()
                    if opening is None:
                        raise RuntimeError("布石を作れなかった（41 手目まで進む布石が取れない）")
                g = m.play_from_sfen(opening[0], "a" if i % 2 == 0 else "b", i, fuseki=opening[1] or None)
            else:
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
                head = f"sfen41={g['sfen41']}" if (openings or self_fuseki) else f"placer={placer} chosen={g['chosen']}"
                log(f"game {i + 1}/{n_games}: {head} a={'sente' if g['sente'] == 'a' else 'gote'} result={g['result']} "
                    f"({g['reason']}) plies={g['plies']} {g['seconds']}s  A {summary['a_points']}/{summary['n']}")
    return summary


def sfen41_from_selfplay(games_dir: Path, n: int, seed: int = 0, scan: int = 5000) -> list[str]:
    """run の自己対局の棋譜（<run>/games/games_*.jsonl）から 41 手目の局面を n 個選ぶ。

    自己対局は最新の重みが両側を持つので、これが「最強 Libra 同士の布石」になる（2026-09-20 のユーザーの決定）。
    エンジンは同じ局面に同じ手を返す（乱数を使わない）ので、布石をその場で指させると 40 局とも同じ将棋になる。
    新しい側から scan 局を見て、そこから重複しない局面を一様に選ぶ。搾取者・リーグの局と、41 手目で
    終わっている局（§3.4 の裁定・合法手なし）は外す。
    """
    import random

    rows: list[str] = []
    seen: set[str] = set()
    for p in sorted(games_dir.glob("games_*.jsonl"), reverse=True):
        if len(rows) >= scan:
            break
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if len(rows) >= scan:
                break
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = g.get("sfen41")
            if not s or s in seen or "exploiter" in g or "league" in g:
                continue
            if int(g.get("plies") or 0) <= 40:  # 41 手目の裁定で終わった局
                continue
            seen.add(s)
            rows.append(s)
    out: list[str] = []
    for s in random.Random(seed).sample(rows, len(rows)):
        if len(out) >= n:
            break
        pos = ls.Position()
        try:
            pos.set_sfen(s, "normal")
        except ValueError:
            continue
        if pos.is_over() or not pos.legal_moves():
            continue
        out.append(s)
    return out


def load_openings(spec: str, games_dir: Path, n: int, seed: int = 0) -> list[str]:
    """--fuseki の値から 41 手目の局面を作る。'selfplay' は自己対局から、ほかはファイル
    （1 行 1 つの SFEN か、`sfen41` を持つ JSONL）。"""
    if spec == "selfplay":
        return sfen41_from_selfplay(games_dir, n, seed)
    out: list[str] = []
    for line in Path(spec).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            s = json.loads(line).get("sfen41")
            if s:
                out.append(s)
        else:
            out.append(line)
    return out[:n] if n > 0 else out
