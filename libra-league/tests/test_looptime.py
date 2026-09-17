# SPDX-License-Identifier: Apache-2.0
"""学習器のループの処理時間の内訳（looptime.py）。"""

from libra_league.looptime import PHASES, LoopTimer, format_timing


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_take_merges_selfplay_parts_and_puts_the_rest_in_other():
    clock = FakeClock()
    lt = LoopTimer(clock)
    sp = {"rounds": 10, "collect": 1.0, "eval": 3.0, "proof": 0.5, "apply": 1.5}
    lg = {"rounds": 10, "collect": 0.2, "eval": 0.6, "proof": 0.0, "apply": 0.2}
    lt.add("league", 1.5)  # 相手の切り替えの 0.5 秒を含む
    lt.add("train_step", 2.0)
    lt.add("train_step", 1.0)
    lt.count("train_steps", 30)
    clock.t = 120.0
    out = lt.take(sp, lg)
    s = out["sec"]
    assert out["window_s"] == 20.0 and out["rounds"] == 10 and out["league_rounds"] == 10 and out["train_steps"] == 30
    assert s["sp_collect"] == 1.0 and s["sp_eval"] == 3.0 and s["sp_proof"] == 0.5 and s["sp_apply"] == 1.5
    assert s["lg_eval"] == 0.6 and s["lg_other"] == 0.5 and "league" not in s
    assert s["train_step"] == 3.0
    assert abs(sum(s.values()) - 20.0) < 1e-6 and abs(s["other"] - 9.5) < 1e-6
    assert set(s) == set(PHASES)
    # 取ったら 0 から数え直す
    clock.t = 130.0
    out2 = lt.take({}, None)
    assert out2["window_s"] == 10.0 and out2["rounds"] == 0 and out2["sec"]["other"] == 10.0 and out2["sec"]["train_step"] == 0.0


def test_other_is_not_negative_when_parts_overlap_the_window():
    clock = FakeClock()
    lt = LoopTimer(clock)
    lt.add("checkpoint", 5.0)
    clock.t = 104.0
    out = lt.take(None, None)
    assert out["sec"]["other"] == 0.0 and out["sec"]["checkpoint"] == 5.0


def test_phase_context_manager_adds_elapsed():
    clock = FakeClock()
    lt = LoopTimer(clock)
    with lt.phase("replay"):
        clock.t += 0.25
    assert lt.take(None, None)["sec"]["replay"] == 0.25


def test_format_timing_shows_shares_and_per_round_ms():
    tm = {"window_s": 100.0, "rounds": 1000, "league_rounds": 1000, "train_steps": 50,
          "sec": {k: 0.0 for k in PHASES} | {"sp_eval": 60.0, "train_step": 30.0, "other": 10.0}}
    line = format_timing(tm)
    assert "sp_eval 60.0%" in line and "train_step 30.0%" in line and "sp_collect" not in line
    assert "round 10.0/s (collect 0.0 ms, eval 60.0 ms" in line and "train 600.0 ms/step" in line
    assert format_timing(None) == ""
