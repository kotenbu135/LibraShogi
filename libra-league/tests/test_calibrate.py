# SPDX-License-Identifier: Apache-2.0
"""同じネットの自己対局での較正（信頼度曲線・ECE・Brier）。docs/decisions.md 2026-09-17。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import librasearch
import librashogi as ls
from libra_league.calibrate import FUSEKI_MOVES, append_calib, load_calib, newest_games, reliability, selfplay_calibration
from libra_league.cli import main
from libra_league.replay import ReplayBuffer
from libra_league.state import StateDir


def _games(n: int, seed: int) -> list[dict]:
    cfg = {"full_sims": 8, "fast_sims": 4, "full_prob": 0.5, "max_ply": 320}
    sp = librasearch.SelfPlay(cfg, 8, seed=seed, threads=2)
    sq = np.zeros((8, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((8, ls.GLOB_FEATS), np.float32)
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    while len(out) < n:
        sp.collect(sq, glob)
        sp.apply(rng.standard_normal((8, ls.POLICY_SIZE), dtype=np.float32), np.tile(np.array([0.4, 0.2, 0.4], np.float32), (8, 1)))
        out += sp.take_finished()
    return out[:n]


def test_reliability_ece_brier():
    # 予測 0.15 で実際 15/100、予測 0.85 で実際 85/100 → 完全に較正されている
    p = np.array([0.15] * 100 + [0.85] * 100)
    a = np.array([1.0] * 15 + [0.0] * 85 + [1.0] * 85 + [0.0] * 15)
    r = reliability(p, a, bins=10)
    assert r["n"] == 200 and abs(r["ece"]) < 1e-9 and abs(r["mce"]) < 1e-9 and abs(r["bias"]) < 1e-9
    assert [(b["lo"], b["n"]) for b in r["bins"]] == [(0.1, 100), (0.8, 100)]
    assert abs(r["brier"] - (0.15 * 0.85)) < 1e-9  # 各区間 p(1−p)
    assert abs(r["brier_ref"] - 0.25) < 1e-9       # 実際の平均 0.5 を言い続けたとき
    # 予測を 0.1 ずらすと ECE・MCE・偏りは 0.1
    r = reliability(p + 0.1, a, bins=10)
    assert abs(r["ece"] - 0.1) < 1e-9 and abs(r["mce"] - 0.1) < 1e-9 and abs(r["bias"] - 0.1) < 1e-9
    # 1.0 は最後の区間、空の入力は n 0
    assert reliability(np.array([1.0]), np.array([1.0]), bins=10)["bins"][0]["hi"] == 1.0
    assert reliability(np.array([]), np.array([]), bins=10) == {"n": 0}
    assert reliability(np.array([-1e-7]), np.array([0.0]), bins=10)["bins"][0]["lo"] == 0.0


def test_eval_calibration_uses_same_bins():
    from libra_league.evaluate import calibration

    g = _game(4, 1, 0.5)  # 得点の予測 0.75。先手の番は当たり（1）、後手の番は外れ（0）
    out = calibration([g], lambda g, j: True)
    assert out == [{"lo": 0.7, "hi": 0.8, "n": 4, "pred": 0.75, "actual": 0.5}]


def _game(n_moves: int, result: int, q: float) -> dict:
    return {"kb": 0, "kw": 80, "result": result, "moves": np.zeros(n_moves, np.uint32),
            "full": np.ones(n_moves, np.uint8), "root_q": np.full(n_moves, q, np.float32)}


def test_selfplay_calibration_groups():
    g = _game(FUSEKI_MOVES + 4, 1, 0.2)  # 先手勝ち。探索値は手番側で 0.2 → 得点 0.6
    g["full"][1] = 0                      # 早読みの手は使わない
    g["root_q"][2] = 1.0                  # 証明済み（先手の番で先手勝ち＝当たり）
    g["root_q"][3] = -1.0                 # 後手の番で手番側の負け＝当たり
    g["root_q"][5] = 1.0                  # 後手の番で手番側の勝ち＝外れ
    c = selfplay_calibration([g], bins=10)
    assert c["games"] == 1 and c["certain"] == {"n": 3, "agree": 2}
    gr = c["groups"]
    # 布石: 添字 0..37。使うのは添字 1（早読み）と 2・3・5（確定）を除いた 34 手
    assert gr["fuseki_sente"]["n"] == 18 and gr["fuseki_gote"]["n"] == 16
    assert abs(gr["fuseki_sente"]["actual"] - 1.0) < 1e-9 and abs(gr["fuseki_gote"]["actual"]) < 1e-9
    assert abs(gr["fuseki_sente"]["pred"] - 0.6) < 1e-6 and abs(gr["fuseki_gote"]["bias"] - 0.6) < 1e-6
    assert gr["normal_sente"]["n"] == 2 and gr["normal_gote"]["n"] == 2
    assert gr["start"]["n"] == 1 and gr["all"]["n"] == 38
    # 引き分けは得点 0.5
    d = selfplay_calibration([_game(2, 0, 0.0)], bins=10)["groups"]
    assert abs(d["start"]["actual"] - 0.5) < 1e-9 and abs(d["all"]["ece"]) < 1e-9
    assert selfplay_calibration([], bins=10)["groups"]["all"] == {"n": 0}
    # リーグの対局は局ごと除く
    lg = _game(4, 1, 0.2)
    lg["league_opponent"] = 100
    c = selfplay_calibration([lg, _game(4, 1, 0.2)], bins=10)
    assert c["games"] == 1 and c["league_skipped"] == 1 and c["groups"]["all"]["n"] == 4


def test_fuseki_and_side_match_replay_features():
    """布石・手番の区別が学習のバッチ（replay_features）と同じ。"""
    games = [g for g in _games(8, 3) if len(g["moves"]) > FUSEKI_MOVES + 2]
    assert games
    for g in games:
        n = len(g["moves"])
        j = np.arange(n)
        sq = np.empty((n, 81, ls.SQ_FEATS), np.float32)
        glob = np.empty((n, ls.GLOB_FEATS), np.float32)
        side = np.empty(n, np.uint8)
        fuseki = np.empty(n, np.uint8)
        ls.replay_features(np.full(n, g["kb"], np.int32), np.full(n, g["kw"], np.int32), [g["moves"]] * n, (j + 2).astype(np.int32),
                           np.zeros(n, np.uint8), sq, glob, side, fuseki, 320, True)
        assert (fuseki.astype(bool) == (j < FUSEKI_MOVES)).all()
        assert ((side == 1) == (j % 2 == 0)).all()


def test_newest_games_and_cli(tmp_path: Path, capsys):
    sd = StateDir(tmp_path / "x")
    sd.create()
    rb = ReplayBuffer(sd.replay, sd.games, window_games=100, chunk_games=5, max_ply=320, count_from_41=True)
    games = _games(12, 4)
    rb.add_games(games)
    got = newest_games(sd.replay, 7)
    assert len(got) == 7 and all(np.array_equal(a["moves"], b["moves"]) for a, b in zip(got, games[3:10]))  # 書かれたチャンクは 10 局
    assert len(newest_games(sd.replay, 100)) == 10
    assert main(["--root", str(tmp_path), "--run", "x", "calib", "--games", "7", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["games"] == 7 and "all" in out["groups"]
    assert main(["--root", str(tmp_path), "--run", "x", "calib", "--games", "7"]) == 0
    assert "ECE" in capsys.readouterr().out


def test_append_and_load_calib_keep_bins_only_last(tmp_path: Path):
    sd = StateDir(tmp_path / "x")
    sd.create()
    for i in range(3):
        append_calib(sd, {"t": i, "groups": {"all": {"n": 1, "ece": 0.1, "bins": [{"lo": 0.0}]}}, "certain": {"n": 0, "agree": 0}})
    rows = load_calib(sd, 10)
    assert [r["t"] for r in rows] == [0, 1, 2]
    assert "bins" not in rows[0]["groups"]["all"] and rows[-1]["groups"]["all"]["bins"] == [{"lo": 0.0}]
    assert [r["t"] for r in load_calib(sd, 2)] == [1, 2]


def test_status_history_includes_calib(tmp_path: Path, capsys):
    sd = StateDir(tmp_path / "x")
    sd.create()
    sd.write_state({"step": 5, "generation": 1, "games_total": 0})
    append_calib(sd, {"t": 1, "groups": {"all": {"n": 1, "ece": 0.1}}, "certain": {"n": 0, "agree": 0}})
    assert main(["--root", str(tmp_path), "--run", "x", "status", "--json", "--history", "5"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["calib"][0]["groups"]["all"]["ece"] == 0.1


def test_runner_writes_calib_on_interval(tmp_path: Path):
    from libra_league.runner import Runner

    sd = StateDir(tmp_path / "x")
    sd.create()
    games = [_game(FUSEKI_MOVES + 2, 1, 0.0) for _ in range(5)]
    fake = SimpleNamespace(cfg={"run": {"calib_minutes": 60, "calib_games": 3}}, sd=sd, last_calib=0.0,
                           replay=SimpleNamespace(games=games, window_games_list=lambda: games), state={"step": 9, "generation": 2}, log=lambda s: None)
    Runner.maybe_write_calib(fake, 4000.0)
    Runner.maybe_write_calib(fake, 4000.0 + 60)  # 間隔の前は書かない
    rows = load_calib(sd, 10)
    assert len(rows) == 1 and rows[0]["games"] == 3 and rows[0]["step"] == 9 and rows[0]["groups"]["all"]["n"] > 0
    fake.cfg["run"]["calib_minutes"] = 0  # 0 で無効
    Runner.maybe_write_calib(fake, 99999.0)
    assert len(load_calib(sd, 10)) == 1
    fake.cfg["run"]["calib_minutes"] = 60  # 搾取者の run では書かない
    fake.cfg["exploiter"] = {"main_ckpt": "main.pt"}
    Runner.maybe_write_calib(fake, 99999.0)
    assert len(load_calib(sd, 10)) == 1
