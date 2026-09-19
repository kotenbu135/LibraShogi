# SPDX-License-Identifier: Apache-2.0
"""自己対局に投了を入れたときの評価の節約と誤投了の見積もり（libra_league.resign）。docs/decisions.md 2026-09-19。"""
from __future__ import annotations

import json
import pickle

import pytest

from libra_league.cli import main
from libra_league.resign import format_table, resign_scan
from libra_league.state import StateDir

FULL, FAST = 96, 24


def _game(n: int, result: int, q_at: dict[int, float] | None = None, full: list[bool] | None = None,
          q0: float = 0.0) -> dict:
    q = [q0] * n
    for j, v in (q_at or {}).items():
        q[j] = v
    return {"root_q": q, "full": [True] * n if full is None else full, "result": result}


def _row(res: dict, thr: float, run: int) -> dict:
    return next(r for r in res["rows"] if r["thr"] == thr and r["run"] == run)


def test_resign_triggers_on_the_losing_side_and_counts_saved_evals():
    # 添字 40・42 は先手の番（偶数）。後手の勝ち（result = -1）なので、先手の投了は正しい
    g = _game(60, -1, {40: -0.96, 42: -0.96})
    res = resign_scan([g], FULL, FAST, thresholds=(0.95,), runs=(1, 2))
    r1, r2 = _row(res, 0.95, 1), _row(res, 0.95, 2)
    assert r1["resigned"] == 1 and r1["mean_resign_move"] == 40
    assert r2["resigned"] == 1 and r2["mean_resign_move"] == 42  # 自分の番で 2 手続く必要がある
    # 投了した手より後の手が浮く（全部が全読み）
    assert r1["moves_saved_frac"] == pytest.approx((59 - 40) / 60, abs=5e-5)
    assert r1["evals_saved_frac"] == pytest.approx((59 - 40) / 60, abs=5e-5)
    assert r1["speedup"] == pytest.approx(1 / (1 - (59 - 40) / 60), abs=5e-4)
    assert r1["wrong_win"] == 0 and r1["wrong_draw"] == 0


def test_no_resignation_during_the_opening_phase():
    g = _game(60, -1, {10: -0.99, 12: -0.99})  # 布石（添字 38 手目まで）
    res = resign_scan([g], FULL, FAST, thresholds=(0.95,), runs=(1,))
    assert _row(res, 0.95, 1)["resigned"] == 0
    res2 = resign_scan([g], FULL, FAST, thresholds=(0.95,), runs=(1,), min_move=0)
    assert _row(res2, 0.95, 1)["mean_resign_move"] == 10


def test_false_resignation_is_counted_for_wins_and_draws():
    win = _game(60, 1, {40: -0.96})    # 先手が投了するが、実際は先手の勝ち
    draw = _game(60, 0, {40: -0.96})   # 引き分けだった
    lose = _game(60, -1, {40: -0.96})  # 正しい投了
    res = resign_scan([win, draw, lose], FULL, FAST, thresholds=(0.95,), runs=(1,))
    r = _row(res, 0.95, 1)
    assert r["resigned"] == 3 and r["wrong_win"] == 1 and r["wrong_draw"] == 1
    assert r["wrong_win_frac"] == pytest.approx(1 / 3, abs=5e-5)


def test_the_first_side_to_reach_the_threshold_ends_the_game():
    # 後手（奇数の添字）が 41 手目、先手（偶数）が 44 手目で届く。早い方で対局が終わる
    g = _game(60, 1, {41: -0.99, 44: -0.99})  # result = +1 は先手の勝ち
    r = _row(resign_scan([g], FULL, FAST, thresholds=(0.95,), runs=(1,)), 0.95, 1)
    assert r["mean_resign_move"] == 41
    assert r["wrong_win"] == 0 and r["wrong_draw"] == 0  # 投了したのは後手で、実際に負けている


def test_opening_book_and_proven_moves_cost_no_evaluations():
    # 先頭 4 手が布石の手順（full=False・q=0）、最後の手が証明済み（|q|=1）
    n = 50
    full = [False] * 4 + [True] * (n - 4)
    g = _game(n, -1, {n - 1: -1.0}, full=full)
    res = resign_scan([g], FULL, FAST, thresholds=(0.95,), runs=(1,), min_move=0)
    assert res["book_moves"] == 4 and res["proven_moves"] == 1
    assert res["mean_evals"] == (n - 5) * FULL  # 布石 4 手と証明済み 1 手は 0


def test_a_fast_move_with_a_nonzero_value_is_not_treated_as_the_opening_book():
    g = _game(20, -1, {0: -0.5}, full=[False] * 20)
    res = resign_scan([g], FULL, FAST, thresholds=(0.95,), runs=(1,))
    assert res["book_moves"] == 0 and res["mean_evals"] == 20 * FAST


def test_league_games_are_skipped():
    g = _game(50, -1, {40: -0.99})
    lg = {**_game(50, -1, {40: -0.99}), "league_opponent": "lx-000000001.pt"}
    res = resign_scan([g, lg], FULL, FAST, thresholds=(0.95,), runs=(1,))
    assert res["games"] == 1 and res["league_skipped"] == 1


def test_empty_input():
    res = resign_scan([], FULL, FAST)
    assert res["games"] == 0 and res["rows"] == []
    assert format_table(res) == "対局が無い"


def test_cli_reads_the_chunks_and_prints_a_table(tmp_path, capsys):
    sd = StateDir(tmp_path / "x")
    sd.create()
    games = [_game(60, -1, {40: -0.96, 42: -0.96}) for _ in range(5)]
    with open(sd.replay / "chunk_000001.pkl", "wb") as f:
        pickle.dump(games, f)
    assert main(["--root", str(tmp_path), "--run", "x", "resign", "--games", "5", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["games"] == 5 and out["full_sims"] == 96 and out["fast_sims"] == 24
    assert _row(out, 0.95, 1)["resigned"] == 5
    assert main(["--root", str(tmp_path), "--run", "x", "resign", "--games", "5"]) == 0
    assert "投了の規則" in capsys.readouterr().out
