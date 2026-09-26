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

**両玉を読みで置く形**もある（`place="search"`）。置く側は先手玉を乱数で 1 マスに置き、後手玉の候補をそれぞれ置く側の
エンジンに `go` して、先手の勝率がいちばん 0.5 に近いマスに後手玉を置く。選ぶ側も同じ読みで先後を決めるので、
玉配置表（読み 96 回で作る）と対局の読みの量が違うときに、選ぶ側だけが得をする片寄りが出ない（2026-09-25 のユーザーの決定
「matchにいれる」。動画用の棋譜を読み 1600 回などで作るため）。

**両玉と先後を決め打ちする形**もある（`kings`・`choose`）。両玉は渡したマスに置き、選ぶ側はいつも渡した側を取る
（選ぶ側の読みの勝率は形勢の表示のために残す）。同じ始まりから何局も打てる（2026-09-26 のユーザーの依頼
「任意の玉配置・先後選択から対局を始められるように」）。

**41 手目から別のエンジンに替える形**もある（`e41`）。布石（両玉・選択・40 手）は a・b が打ち、41 手目からは渡した
側の席を別のエンジン（手元のやねうら王＋水匠5 など、本将棋だけのエンジン）が指す。両陣とも替えればそのエンジン同士、
片方だけなら Libra 対そのエンジンになる（2026-09-26 のユーザーの依頼「41手目はやねうら王・水匠５で対局できる
オプション（Libra vs 水匠５も）」）。エンジンはリポジトリに入れず、起動のたびに手元のパスを渡す。勝ち負けは席（a・b）で数える。
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import librashogi as ls

from .usi_client import UsiEngine


def cp_to_winrate(cp: int) -> float:
    from librashogi.usi import cp_to_winrate as f

    return f(cp)


class Match:
    def __init__(self, a: UsiEngine, b: UsiEngine, go_args: str, max_ply: int, count_from_41: bool, log=None,
                 go_args_b: str | None = None, place: str = "engine", place_seed: int = 0,
                 kings: tuple[str, str] | None = None, choose: str | None = None,
                 e41: dict | None = None, go_args_41: str | None = None):
        self.a, self.b = a, b
        self.go_args = go_args
        # b 側だけ別の `go`。読む量に差を付けて測る（ハンデ）ために使う。既定は a と同じ
        self.go_args_b = go_args_b or go_args
        self.max_ply = max_ply
        self.count_from_41 = count_from_41
        self.log = log or (lambda s: None)
        # 両玉の置き方: engine＝置く側のエンジンに任せる（玉配置表か探索）／search＝後手玉を読みで釣り合わせる
        if place not in ("engine", "search"):
            raise ValueError(f"place は engine か search: {place}")
        self.place = place
        self.place_rng = random.Random(place_seed)
        # 決め打ちの両玉（先手玉, 後手玉。K*5i の形）と、選ぶ側がいつも取る側（sente | gote）
        if kings is not None and (len(kings) != 2 or place != "engine"):
            raise ValueError("kings は (先手玉, 後手玉) の 2 つで、place は engine のときだけ")
        if choose not in (None, "sente", "gote"):
            raise ValueError(f"choose は sente か gote: {choose}")
        self.kings = tuple(kings) if kings else None
        self.choose = choose
        # 41 手目から席を替えるエンジン（{'a': E, 'b': E} の一部）と、その `go`（既定は b 側と同じ）
        self.e41 = {k: v for k, v in (e41 or {}).items() if v is not None}
        if not set(self.e41) <= {"a", "b"}:
            raise ValueError(f"e41 の席は a か b: {sorted(self.e41)}")
        self.go_args_41 = go_args_41 or self.go_args_b

    def engines(self) -> list:
        """この対局で使うエンジンすべて（41 手目からのエンジンを含む）。"""
        return [self.a, self.b, *self.e41.values()]

    def go(self, E: UsiEngine, line: str):
        """その側の `go` の引数で読ませる。"""
        if any(E is x for x in self.e41.values()):
            return E.go(line, self.go_args_41)
        return E.go(line, self.go_args_b if E is self.b else self.go_args)

    def play_moves(self, pos, side: dict, tokens: list[str], moves_info: list[dict]) -> str | None:
        """終局まで指させる。非合法手を指した側（'a' | 'b'）を返す（無ければ None）。"""
        eng = {"a": self.a, "b": self.b}
        while not pos.is_over():
            turn = pos.turn
            who = side[turn]
            E = eng[who]
            if pos.phase == "normal" and who in self.e41:
                E = self.e41[who]
            if pos.phase == "fuseki":
                line = "position fuseki moves " + " ".join(t for t in tokens if not t.startswith("choose:"))
            else:
                normal = [t for t in tokens if t.startswith("n:")]
                line = f"position sfen {self.sfen41} moves {' '.join(m[2:] for m in normal)}".rstrip()
            bm, info = self.go(E, line)
            rec = {"by": who, "move": bm, "ply": pos.ply + 1, **info}
            if E is not eng[who]:
                rec["engine41"] = True
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

    def place_by_search(self, P, pos, tokens: list[str], moves_info: list[dict], placer: str) -> None:
        """両玉を置く（`place="search"`）。先手玉は乱数、後手玉は候補を置く側のエンジンで読み、先手の勝率が 0.5 に
        いちばん近いマス。後手玉の四段目は除く（3 手目の桂打ちで遮断不能になり先手の勝ちがほぼ決まる。rules.md §3.4、
        measurements.md 2026-09-21）。"""
        kb = self.place_rng.choice(sorted(pos.legal_moves()))
        pos.do_move(kb)
        tokens.append(kb)
        moves_info.append({"by": placer, "move": kb, "place": "random"})
        cands: dict[str, float] = {}
        for kw in sorted(m for m in pos.legal_moves() if not m.endswith("d")):
            _, info = self.go(P, f"position fuseki moves {kb} {kw}")
            w = info.get("winrate")
            if w is None and "cp" in info:
                w = cp_to_winrate(info["cp"])
            if w is None:
                continue
            p2 = ls.Position()
            p2.set_position(f"position fuseki moves {kb} {kw}")
            cands[kw] = round(w if p2.turn == "sente" else 1.0 - w, 4)  # 先手の勝率に直す
        if not cands:
            raise RuntimeError(f"両玉を読みで置けない: 置く側のエンジンが後手玉の候補に winrate を返さない（先手玉 {kb}）")
        kw = min(cands, key=lambda m: (abs(cands[m] - 0.5), m))
        pos.do_move(kw)
        tokens.append(kw)
        # winrate は他の手と同じく指した側（後手）から見た値。candidates は先手の勝率
        moves_info.append({"by": placer, "move": kw, "winrate": round(1.0 - cands[kw], 4), "place": "search",
                           "candidates": cands})
        self.log(f"両玉を読みで置いた: {kb} {kw}（先手の勝率 {cands[kw]:.3f}、候補 {len(cands)} マス）")

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
        for e in self.engines():
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
        for e in self.engines():
            e.new_game()
        pos = ls.Position()
        pos.set_max_ply(self.max_ply, self.count_from_41)
        tokens: list[str] = []
        moves_info: list[dict] = []
        result = reason = None
        illegal_by = None
        # 両玉
        if self.place == "search":
            self.place_by_search(P, pos, tokens, moves_info, placer)
        elif self.kings:
            for k in self.kings:
                if not pos.is_legal(k):
                    raise ValueError(f"決め打ちの玉 {k} は置けない（{' '.join(tokens) or '1 手目'}）")
                pos.do_move(k)
                tokens.append(k)
                moves_info.append({"by": placer, "move": k, "place": "fixed"})
        for ply in range(2 if self.place == "engine" and not self.kings else 0):
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
            chosen = self.choose or ("sente" if w >= 0.5 else "gote")
            tokens.append(f"choose:{chosen}")
            moves_info.append({"by": "b" if placer == "a" else "a", "choose": chosen, "winrate": w,
                               **({"fixed": True} if self.choose else {})})
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
              openings: list[str] | None = None, self_fuseki: bool = False, place: str = "engine",
              place_seed: int = 0, kings: tuple[str, str] | None = None, choose: str | None = None,
              e41: dict | None = None, go_args_41: str | None = None, opening_file: Path | None = None) -> dict:
    """self_fuseki なら布石を a 側だけで打ち（両陣とも同じ Libra）、41 手目から本将棋を a 対 b で指す。
    openings（41 手目の局面の一覧）を渡せば布石を作らずそれを使う。どちらも**同じ局面を先後入れ替えて
    2 局ずつ**打つので、布石の有利不利が打ち消し合う。kings・choose は両玉と先後の決め打ち（布石を作る形のときだけ）、
    e41 は 41 手目から席（'a' | 'b'）を替えるエンジン。

    opening_file は self_fuseki の布石の控え（1 行 1 布石の JSONL、`sfen41` と `fuseki`）。j 番目の布石が控えにあれば
    作らずにそれを使い、無ければ作って書き足す。相手の読む量の段ごとに同じ控えを渡すと、段をまたいで同じ布石になる
    （エンジンは根の乱数の種を時刻から取り、読みも複数スレッドなので、同じ種を渡しても布石はそろわない。2026-09-26）。"""
    if (kings or choose) and (openings or self_fuseki):
        raise ValueError("kings・choose は両玉から打つ形（openings も self_fuseki も無し）のときだけ")
    m = Match(a, b, go_args, max_ply, count_from_41, log, go_args_b=go_args_b, place=place, place_seed=place_seed,
              kings=kings, choose=choose, e41=e41, go_args_41=go_args_41)
    summary = {"a": a.id_name, "b": b.id_name, "n": 0, "a_points": 0.0, "by_engine_side": {}, "reasons": {}, "games": [],
               "go_a": go_args, "go_b": m.go_args_b, "openings": len(openings) if openings else 0,
               "self_fuseki": bool(self_fuseki), "place": place,
               "kings": list(kings) if kings else None, "choose": choose,
               "engine41": {k: getattr(v, "id_name", None) for k, v in m.e41.items()} or None,
               "go_41": m.go_args_41 if m.e41 else None}
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    opening: tuple[str, list[str]] | None = None
    saved: list[tuple[str, list[str]]] = []
    if self_fuseki and opening_file is not None and Path(opening_file).exists():
        for line in Path(opening_file).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # 書きかけで落ちた最後の行
            if r.get("sfen41"):
                saved.append((r["sfen41"], str(r.get("fuseki") or "").split()))
    if self_fuseki and opening_file is not None:
        summary["openings_reused"] = 0
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for i in range(n_games):
            placer = first_placer if i % 2 == 0 else ("b" if first_placer == "a" else "a")
            t0 = time.time()
            if openings or self_fuseki:
                # 同じ局面で先後を入れ替えて 2 局ずつ（偶数局は a が先手）
                if openings:
                    opening = (openings[(i // 2) % len(openings)], [])
                elif i % 2 == 0 or opening is None:
                    j = i // 2
                    if j < len(saved):
                        opening = saved[j]
                        summary["openings_reused"] += 1
                    else:
                        opening = m.make_opening()
                        if opening is None:
                            raise RuntimeError("布石を作れなかった（41 手目まで進む布石が取れない）")
                        if opening_file is not None:
                            Path(opening_file).parent.mkdir(parents=True, exist_ok=True)
                            with open(opening_file, "a", encoding="utf-8") as fo:
                                fo.write(json.dumps({"sfen41": opening[0], "fuseki": " ".join(opening[1])},
                                                    ensure_ascii=False) + "\n")
                            saved.append(opening)
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
