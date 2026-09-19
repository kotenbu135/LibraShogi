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
