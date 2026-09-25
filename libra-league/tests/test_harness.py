# SPDX-License-Identifier: Apache-2.0
import json
import os
import sys
from pathlib import Path

import librashogi as ls
from libra_league.harness import Match, run_match
from libra_league.usi_client import UsiEngine

HERE = Path(__file__).resolve().parent


def test_random_vs_random_match(tmp_path: Path):
    env_path = os.environ.get("PYTHONPATH", "")
    cmd = lambda seed: [sys.executable, str(HERE / "random_usi.py"), str(seed)]  # noqa: E731
    a = UsiEngine("A", cmd(1))
    b = UsiEngine("B", cmd(2))
    a.start(ready_timeout=30)
    b.start(ready_timeout=30)
    try:
        out = tmp_path / "m.jsonl"
        s = run_match(a, b, 4, "nodes 1", out, max_ply=320)
    finally:
        a.quit()
        b.quit()
    assert s["n"] == 4
    assert s["go_a"] == s["go_b"] == "nodes 1"  # --go-opp を渡さなければ両者同じ
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    for line in lines:
        g = json.loads(line)
        assert g["placer"] in ("a", "b") and g["chosen"] in ("sente", "gote")
        assert g["result"] in ("sente", "gote", "draw")
        # 棋譜を再生して同じ裁定になる（投了・宣言・非合法手以外）
        if g["reason"] in ("resign", "declaration", "illegal_declaration", "illegal_move", "timeout"):
            continue
        p = ls.Position()
        p.set_max_ply(320, True)
        p.set_position("position fuseki moves " + g["tokens"])  # choose: は読み飛ばされる
        assert p.outcome() == (g["result"], g["reason"]), g["tokens"]
    # 置く側が交互
    assert [x["placer"] for x in s["games"]] == ["a", "b", "a", "b"]


def test_each_side_can_have_its_own_go_args():
    """読む量に差を付けて測るため、a（Libra）と b（相手）で別の `go` を送れる。既定は両者同じ。"""

    class Fake:
        def __init__(self):
            self.seen: list[str] = []

        def go(self, line, go_args):
            self.seen.append(go_args)
            return "resign", {}

    a, b = Fake(), Fake()
    m = Match(a, b, "nodes 400", 320, True, go_args_b="movetime 1000")
    m.go(a, "position fuseki")
    m.go(b, "position fuseki")
    assert (a.seen, b.seen) == (["nodes 400"], ["movetime 1000"])

    c, d = Fake(), Fake()
    m2 = Match(c, d, "movetime 1000", 320, True)
    m2.go(c, "position fuseki")
    m2.go(d, "position fuseki")
    assert (c.seen, d.seen) == (["movetime 1000"], ["movetime 1000"])


def _sfen41(seed: int) -> str:
    """乱数で布石を 40 手打ち、41 手目の局面（標準 SFEN）を返す。裁定で終わった局は捨てる。"""
    import random

    rng = random.Random(seed)
    while True:
        p = ls.Position()
        p.set_max_ply(320, True)
        while p.phase == "fuseki":
            p.do_move(rng.choice(p.legal_moves()))
        if not p.is_over() and p.legal_moves():
            return p.sfen()


def test_match_can_start_from_a_given_sfen41(tmp_path: Path):
    """布石を持ち込む形: 41 手目の局面から本将棋だけを打ち、同じ局面を先後入れ替えて 2 局ずつ指す。
    相手が布石を指せないふつうの将棋エンジンでも測れるようにするため（2026-09-20 のユーザーの決定）。"""
    cmd = lambda seed: [sys.executable, str(HERE / "random_usi.py"), str(seed)]  # noqa: E731
    a, b = UsiEngine("A", cmd(1)), UsiEngine("B", cmd(2))
    a.start(ready_timeout=30)
    b.start(ready_timeout=30)
    openings = [_sfen41(11), _sfen41(12)]
    try:
        out = tmp_path / "m.jsonl"
        s = run_match(a, b, 4, "nodes 1", out, openings=openings)
    finally:
        a.quit()
        b.quit()
    assert s["n"] == 4 and s["openings"] == 2
    games = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [g["sfen41"] for g in games] == [openings[0], openings[0], openings[1], openings[1]]
    assert [g["sente"] for g in games] == ["a", "b", "a", "b"]   # 同じ局面を先後入れ替えて 2 局
    for g in games:
        assert g["placer"] is None and g["chosen"] is None      # 布石は指していない
        assert g["result"] in ("sente", "gote", "draw")
        if g["reason"] in ("resign", "declaration", "illegal_declaration", "illegal_move", "timeout"):
            continue
        p = ls.Position()
        p.set_max_ply(320, True)
        p.set_sfen(g["sfen41"], "normal")
        for m in g["tokens"].split():
            p.do_move(m)
        assert p.outcome() == (g["result"], g["reason"])


def test_openings_come_from_the_runs_own_selfplay(tmp_path: Path):
    """布石は run の自己対局（＝最強 Libra 同士）から取る。搾取者・リーグの局と 41 手目で終わった局は外す。"""
    from libra_league.harness import sfen41_from_selfplay

    good = [_sfen41(21), _sfen41(22), _sfen41(23)]
    rows = [
        {"sfen41": good[0], "plies": 90},
        {"sfen41": good[0], "plies": 88},               # 同じ局面は 1 つだけ
        {"sfen41": good[1], "plies": 40},               # 41 手目の裁定で終わった局
        {"sfen41": good[1], "plies": 120, "exploiter": "sente"},  # 搾取者の局
        {"sfen41": good[1], "plies": 120, "league": {"opponent_step": 1, "main": "sente"}},
        {"sfen41": good[2], "plies": 77},
        {"plies": 99},                                   # 布石の途中で終わった局（sfen41 なし）
    ]
    d = tmp_path / "games"
    d.mkdir()
    (d / "games_000001.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    got = sfen41_from_selfplay(d, 10, seed=0)
    assert sorted(got) == sorted([good[0], good[2]])
    assert len(sfen41_from_selfplay(d, 1, seed=0)) == 1   # 要る数だけ返す


def test_libra_builds_both_camps_and_then_plays_from_move_41(tmp_path: Path):
    """--fuseki self: a 側のエンジンだけが布石 40 手を打ち（両陣とも同じ Libra）、41 手目から a 対 b で指す。
    同じ布石を先後入れ替えて 2 局ずつ。相手（b）は布石を 1 手も指さない＝ふつうの将棋エンジンでよい。"""
    cmd = lambda seed: [sys.executable, str(HERE / "random_usi.py"), str(seed)]  # noqa: E731
    a, b = UsiEngine("A", cmd(3)), UsiEngine("B", cmd(4))
    a.start(ready_timeout=30)
    b.start(ready_timeout=30)
    try:
        out = tmp_path / "m.jsonl"
        s = run_match(a, b, 4, "nodes 1", out, self_fuseki=True)
    finally:
        a.quit()
        b.quit()
    assert s["n"] == 4 and s["self_fuseki"] and s["openings"] == 0
    games = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [g["sente"] for g in games] == ["a", "b", "a", "b"]
    assert games[0]["sfen41"] == games[1]["sfen41"] and games[2]["sfen41"] == games[3]["sfen41"]
    assert games[0]["sfen41"] != games[2]["sfen41"]        # 布石は局ごとに作り直す
    for g in games:
        assert g["placer"] is None and g["chosen"] is None
        assert len(g["fuseki"].split()) == 40              # 布石の手順も残す
        assert g["fuseki"].split()[0].startswith("K*")
        p = ls.Position()
        p.set_max_ply(320, True)
        p.set_position("position fuseki moves " + g["fuseki"])
        assert p.sfen() == g["sfen41"] and p.phase == "normal"


class _PlaceFake:
    """後手玉の段で先手の勝率を返す偽のエンジン（a 段 0.30・b 段 0.52・c 段 0.65・d 段 0.99）。本将棋に入ったら投了する。"""

    RATE = {"a": 0.30, "b": 0.52, "c": 0.65, "d": 0.99}
    id_name = "PlaceFake"

    def __init__(self):
        self.lines: list[str] = []

    def new_game(self):
        pass

    def go(self, line, go_args):
        self.lines.append(line)
        t = line.split()
        moves = t[t.index("moves") + 1:] if "moves" in t else []
        if len(moves) == 2:
            return "5g5f", {"winrate": self.RATE[moves[1][-1]]}  # 手番は先手
        if len(moves) < 2:
            return "K*5i" if not moves else "K*5a", {"winrate": 0.5}
        return "resign", {}


def test_place_search_puts_gote_king_where_sente_is_closest_to_half(tmp_path: Path):
    """--place search: 後手玉は置く側の読みで先手の勝率が 0.5 にいちばん近いマス、四段目は読まない。選ぶ側は従来どおり。"""
    a, b = _PlaceFake(), _PlaceFake()
    out = tmp_path / "m.jsonl"
    s = run_match(a, b, 2, "nodes 1", out, place="search", place_seed=3)
    assert s["place"] == "search"
    games = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    for g, placer in zip(games, ("a", "b")):
        kb, kw = g["tokens"].split()[:2]
        assert kw.endswith("b")  # 0.52 が 0.5 にいちばん近い
        k0, k1 = g["moves"][0], g["moves"][1]
        assert k0["move"] == kb and k0["place"] == "random" and "winrate" not in k0
        assert k1["move"] == kw and k1["place"] == "search" and k1["by"] == placer
        assert k1["winrate"] == 0.48  # 指した側（後手）から見た値
        assert len(k1["candidates"]) == 27 and not any(m.endswith("d") for m in k1["candidates"])
        assert g["moves"][2]["choose"] == "sente" and g["chosen"] == "sente"
    # 候補を読むのは置く側だけ（1 局目は a、2 局目は b）。両玉の後の局面は、ほかに選ぶ側の選択と先手の 3 手目で 1 回ずつ読む
    assert sum(1 for x in a.lines if len(x.split()) == 5) == 27 + 2
    assert sum(1 for x in b.lines if len(x.split()) == 5) == 2 + 27


def test_place_search_first_king_follows_seed(tmp_path: Path):
    """先手玉は --place-seed で決まる（同じ種なら同じ玉）。"""
    firsts = []
    for seed in (5, 5, 6):
        out = tmp_path / f"m{len(firsts)}.jsonl"
        run_match(_PlaceFake(), _PlaceFake(), 1, "nodes 1", out, place="search", place_seed=seed)
        firsts.append(json.loads(out.read_text(encoding="utf-8"))["tokens"].split()[0])
    assert firsts[0] == firsts[1]


def test_place_engine_is_default(tmp_path: Path):
    """既定は置く側のエンジンに任せる（自動計測・外部計測の形は変わらない）。"""
    a, b = _PlaceFake(), _PlaceFake()
    out = tmp_path / "m.jsonl"
    s = run_match(a, b, 1, "nodes 1", out)
    g = json.loads(out.read_text(encoding="utf-8"))
    assert s["place"] == "engine"
    assert g["tokens"].split()[:2] == ["K*5i", "K*5a"]
    assert "place" not in g["moves"][0] and "place" not in g["moves"][1]
