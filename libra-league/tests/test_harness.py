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


class _RandFake:
    """合法手から乱数で指す偽のエンジン。受けた局面の段（布石・本将棋）と go の引数を残す。"""

    def __init__(self, seed: int, name: str, winrate: float = 0.5):
        import random

        self.rng = random.Random(seed)
        self.id_name = name
        self.winrate = winrate
        self.calls: list[tuple[str, str, str]] = []  # (段, 局面の行, go の引数)
        self.new_games = 0

    def new_game(self):
        self.new_games += 1

    def go(self, line, go_args):
        p = ls.Position()
        p.set_max_ply(320, True)
        p.set_position(line)
        self.calls.append((p.phase, line, go_args))
        res, reason = p.outcome()
        if res != "ongoing":
            return ("win" if reason == "ruling41" else "resign"), {}
        return self.rng.choice(sorted(p.legal_moves())), {"winrate": self.winrate}


def test_kings_and_choose_can_be_fixed(tmp_path: Path):
    """--kings・--choose: 両玉は渡したマスに置き（置く側は読まない）、選ぶ側は読みの勝率によらず渡した側を取る。
    置く側は今までどおり交互なので、先後は局ごとに入れ替わる。"""
    a, b = _RandFake(1, "A", winrate=0.9), _RandFake(2, "B", winrate=0.9)  # 読みでは先手を取りたがる
    out = tmp_path / "m.jsonl"
    s = run_match(a, b, 2, "nodes 1", out, kings=("K*3h", "K*7b"), choose="gote")
    assert s["kings"] == ["K*3h", "K*7b"] and s["choose"] == "gote"
    games = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    for g, placer in zip(games, ("a", "b")):
        assert g["tokens"].split()[:3] == ["K*3h", "K*7b", "choose:gote"]
        assert g["placer"] == placer and g["chosen"] == "gote"
        assert g["gote"] == ("b" if placer == "a" else "a") and g["sente"] == placer
        k0, k1, c = g["moves"][:3]
        assert (k0["place"], k1["place"]) == ("fixed", "fixed") and k0["by"] == k1["by"] == placer
        assert c["choose"] == "gote" and c["fixed"] is True and c["winrate"] == 0.9  # 表示用に読みの値は残す
    # 置く側は両玉を読まない（1 局目の a、2 局目の b が受けた最初の go は 3 手目以降の局面）
    assert not any(line.endswith("position fuseki") for _, line, _ in a.calls + b.calls)


def test_kings_must_be_legal_and_only_with_engine_fuseki(tmp_path: Path):
    import pytest

    with pytest.raises(ValueError):
        run_match(_RandFake(1, "A"), _RandFake(2, "B"), 1, "nodes 1", tmp_path / "x.jsonl", kings=("K*5a", "K*5i"))  # 先手玉は敵陣に置けない
    with pytest.raises(ValueError):
        run_match(_RandFake(1, "A"), _RandFake(2, "B"), 1, "nodes 1", tmp_path / "y.jsonl", choose="sente",
                  self_fuseki=True)


def test_engine41_plays_both_seats_from_move_41(tmp_path: Path):
    """--engine41（両陣）: 両玉・選択・布石 40 手は a・b が打ち、41 手目からは両席とも別のエンジンが指す。
    そのエンジンは本将棋の局面（position sfen）しか受けず、go は --go41。勝ち負けは席で数える。"""
    a, b = _RandFake(1, "A"), _RandFake(2, "B")
    ya, yb = _RandFake(3, "Y"), _RandFake(4, "Y")
    out = tmp_path / "m.jsonl"
    s = run_match(a, b, 2, "nodes 1", out, go_args_b="nodes 2", e41={"a": ya, "b": yb}, go_args_41="nodes 1000")
    assert s["engine41"] == {"a": "Y", "b": "Y"} and s["go_41"] == "nodes 1000"
    games = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    for g in games:
        assert "choose:" in g["tokens"]
        normal = [m for m in g["moves"] if m.get("ply", 0) > 40]
        early = [m for m in g["moves"] if 0 < m.get("ply", 0) <= 40]
        if g["plies"] > 40:
            assert normal and all(m.get("engine41") for m in normal)
        assert not any(m.get("engine41") for m in early)
        # 棋譜を再生して同じ裁定になる
        if g["reason"] not in ("resign", "declaration", "illegal_declaration", "illegal_move"):
            p = ls.Position()
            p.set_max_ply(320, True)
            p.set_position("position fuseki moves " + g["tokens"])
            assert p.outcome() == (g["result"], g["reason"])
    assert all(ph == "normal" and line.startswith("position sfen ") and ga == "nodes 1000"
               for e in (ya, yb) for ph, line, ga in e.calls)
    assert ya.calls and yb.calls
    # a・b は本将棋を指さない（41 手目の裁定の確かめだけは布石の局面を知るエンジンに聞く）
    assert all(ph == "fuseki" or " moves" not in line for e in (a, b) for ph, line, _ in e.calls)
    assert all(e.new_games == 2 for e in (a, b, ya, yb))


def test_engine41_can_take_only_the_opponents_seat(tmp_path: Path):
    """--engine41-side b: 41 手目から相手の席だけが別のエンジン（Libra 対 そのエンジン）。"""
    a, b, yb = _RandFake(1, "A"), _RandFake(2, "B"), _RandFake(4, "Y")
    out = tmp_path / "m.jsonl"
    s = run_match(a, b, 2, "nodes 1", out, e41={"b": yb}, go_args_41="nodes 1000")
    assert s["engine41"] == {"b": "Y"}
    for line in out.read_text(encoding="utf-8").splitlines():
        g = json.loads(line)
        for m in g["moves"]:
            if m.get("ply", 0) > 40:
                assert bool(m.get("engine41")) == (m["by"] == "b")
    assert any(ph == "normal" for ph, _, _ in a.calls)
    assert all(ph == "normal" for ph, _, _ in yb.calls)


def test_self_fuseki_reuses_the_saved_openings_across_levels(tmp_path: Path):
    """--fuseki self と --opening-file: 最初の段が作った布石を控えに残し、次の段は作らずに同じ布石で打つ。
    エンジンの乱数は種を渡してもそろわないので、段どうしの比較を対にするにはこれが要る（2026-09-26）。"""
    ofile = tmp_path / "auto.openings.jsonl"
    first = run_match(_RandFake(1, "A"), _RandFake(2, "B"), 4, "nodes 1", tmp_path / "1.jsonl",
                      self_fuseki=True, opening_file=ofile)
    saved = [json.loads(x) for x in ofile.read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 2 and first["openings_reused"] == 0
    # 乱数の違う a で打っても、布石は控えのまま（a に布石を作らせない）
    a2 = _RandFake(99, "A")
    second = run_match(a2, _RandFake(98, "B"), 4, "nodes 1", tmp_path / "2.jsonl", self_fuseki=True, opening_file=ofile)
    assert second["openings_reused"] == 2
    assert not any(ph == "fuseki" for ph, _, _ in a2.calls)
    g1 = [json.loads(x) for x in (tmp_path / "1.jsonl").read_text(encoding="utf-8").splitlines()]
    g2 = [json.loads(x) for x in (tmp_path / "2.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [g["sfen41"] for g in g1] == [g["sfen41"] for g in g2]
    assert [g["fuseki"] for g in g1] == [g["fuseki"] for g in g2]
    # 局数が多い段は、足りない分だけ作って書き足す
    third = run_match(_RandFake(5, "A"), _RandFake(6, "B"), 6, "nodes 1", tmp_path / "3.jsonl", self_fuseki=True,
                      opening_file=ofile)
    assert third["openings_reused"] == 2 and len(ofile.read_text(encoding="utf-8").splitlines()) == 3
