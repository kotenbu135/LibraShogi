# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import numpy as np
import pytest

import librasearch
import librashogi as ls
from libra_league.config import DEFAULTS, dump_toml, load_config
from libra_league.replay import MIRROR_TABLE, ReplayBuffer, add_target_stats, game_to_jsonl, soft_wdl, summarize_target_stats
from libra_league.state import StateDir


def test_config_merge(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text('run_id = "x"\n[train]\nbatch_size = 64\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg["run_id"] == "x" and cfg["train"]["batch_size"] == 64
    assert cfg["train"]["lr"] == DEFAULTS["train"]["lr"] and cfg["search"]["max_ply"] == 320
    # dump → load で往復
    q = tmp_path / "d.toml"
    q.write_text(dump_toml(cfg), encoding="utf-8")
    assert load_config(q) == cfg


def test_state_flags(tmp_path: Path):
    sd = StateDir(tmp_path / "r")
    sd.create()
    assert not sd.flag("PAUSE")
    sd.set_flag("PAUSE")
    assert sd.flag("PAUSE")
    sd.clear_flag("PAUSE")
    assert not sd.flag("PAUSE")
    assert not hasattr(sd, "throttle_value")  # 絞る機能は廃止（decisions.md 2026-09-12）
    sd.write_state({"a": 1})
    assert sd.read_state() == {"a": 1}


def test_soft_wdl():
    t = np.array([1.0, 0.0, -1.0, 0.5])
    w = soft_wdl(t)
    assert np.allclose(w.sum(1), 1) and np.allclose(w[:, 0] - w[:, 2], t)
    assert w[1].tolist() == [0, 1, 0]


def _dummy_games(n: int, seed: int) -> list[dict]:
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


def test_replay_chunks_and_sampling(tmp_path: Path):
    games = _dummy_games(25, 5)
    rb = ReplayBuffer(tmp_path / "replay", tmp_path / "games", window_games=20, chunk_games=10, max_ply=320, count_from_41=True)
    (tmp_path / "replay").mkdir()
    (tmp_path / "games").mkdir()
    assert rb.add_games(games) == 2
    assert rb.chunk_index == 2 and rb.n_games() == 20 and rb.total_games == 25
    assert sorted(p.name for p in (tmp_path / "replay").iterdir()) == ["chunk_000000.pkl", "chunk_000001.pkl"]
    lines = (tmp_path / "games" / "games_000000.jsonl").read_text().splitlines()
    assert len(lines) == 10
    rec = json.loads(lines[0])
    assert rec["tokens"].startswith("K*") and rec["result"] in ("sente", "gote", "draw")
    # 棋譜行を再生して同じ結果
    p = ls.Position()
    for tok in rec["tokens"].split():
        p.do_move(tok)
    assert p.outcome() == (rec["result"], rec["reason"])
    # 再読込は窓の分だけ
    rb2 = ReplayBuffer(tmp_path / "replay", tmp_path / "games", window_games=15, chunk_games=10, max_ply=320, count_from_41=True)
    rb2.load(2, 25)
    assert rb2.n_games() == 15 and rb2.chunk_index == 2
    rng = np.random.default_rng(0)
    b = rb.sample(64, rng, mirror_prob=0.5, lambda_z=0.5)
    assert b["sq"].shape == (64, 81, ls.SQ_FEATS) and b["wdl"].shape == (64, 3) and b["policy_idx"].shape == (64, 32)
    assert np.allclose(b["wdl"].sum(1), 1)
    assert b["policy_valid"].any()
    for i in np.flatnonzero(b["policy_valid"]):
        k = (b["policy_idx"][i] >= 0).sum()
        assert abs(b["policy_p"][i, :k].sum() - 1) < 1e-3
    # 鏡映の表は対合
    assert (MIRROR_TABLE[MIRROR_TABLE] == np.arange(ls.POLICY_SIZE)).all()


def test_target_stats_compare_targets_with_results(tmp_path: Path):
    """学習目標・v41・探索値と実際の結果の差（docs/method-evidence.md §4.4 (a)）。値は得点の尺度（差 / 2）。"""
    games = _dummy_games(20, 7)
    rb = ReplayBuffer(tmp_path / "replay", tmp_path / "games", window_games=100, chunk_games=1000, max_ply=320, count_from_41=True)
    # 全局 先手勝ち、v41 = 0.2（先手から見て）、探索値 0（手番側）
    for g in games:
        g["result"] = 1
        g["v41"] = 0.2
        g["root_q"] = np.zeros(len(g["moves"]), np.float32)
    rb.add_games(games)
    acc: dict = {}
    for _ in range(4):
        add_target_stats(acc, rb.sample(256, np.random.default_rng(1), mirror_prob=0.0, lambda_z=0.5)["target_stats"])
    s = summarize_target_stats(acc)
    assert s["fuseki_n"] > 0 and s["normal_n"] > 0
    # 目標 t = 0.5·1 ＋ 0.5·0.2 = 0.6。t − z = −0.4 → 得点で −0.2。v41 − z = −0.8 → −0.4。目標の引き分け 1 − 0.6
    assert abs(s["target_minus_z"] + 0.2) < 1e-4 and abs(s["v41_minus_z"] + 0.4) < 1e-4
    assert abs(s["draw_target"] - 0.4) < 1e-4 and s["draw_actual"] == 0.0
    # 全局 引き分け、v41 = 0、探索値 0.4 → 目標は z と一致、引き分けの目標 1、探索値は手番側から +0.2
    for g in games:
        g["result"] = 0
        g["v41"] = 0.0
        g["root_q"] = np.full(len(g["moves"]), 0.4, np.float32)
    s = summarize_target_stats(rb.sample(256, np.random.default_rng(2), mirror_prob=0.0, lambda_z=0.5)["target_stats"])
    assert abs(s["target_minus_z"]) < 1e-4 and abs(s["draw_target"] - 1.0) < 1e-4 and s["draw_actual"] == 1.0
    assert abs(s["rootq_minus_z_fuseki"] - 0.2) < 1e-4 and abs(s["rootq_minus_z_normal"] - 0.2) < 1e-4
    assert summarize_target_stats({}) is None
