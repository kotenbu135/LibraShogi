# SPDX-License-Identifier: Apache-2.0
"""リプレイの窓の拡大・held-out・一般化の物差し（docs/restart-plan.md §3 M1・§4、decisions.md 2026-09-17）。"""
import json
from pathlib import Path

import numpy as np
import torch

import librasearch
import librashogi as ls
from libra_league.cli import main
from libra_league.genprof import generalization, measure, profile
from libra_league.replay import ReplayBuffer
from libra_league.state import StateDir
from libra_net.model import LibraNet, NetConfig

SMALL = NetConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64)


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
    out = out[:n]
    for i, g in enumerate(out):
        g["tag"] = i  # どの局が窓・held-out に入ったかを追う印
    return out


def _mk(tmp_path: Path, **kw) -> ReplayBuffer:
    (tmp_path / "replay").mkdir(parents=True, exist_ok=True)
    (tmp_path / "games").mkdir(parents=True, exist_ok=True)
    return ReplayBuffer(tmp_path / "replay", tmp_path / "games", max_ply=320, count_from_41=True, **kw)


def test_window_grows_with_total_games(tmp_path: Path):
    rb = _mk(tmp_path, window_games=10, chunk_games=5, window_frac=0.5, window_games_max=30)
    games = _games(100, 1)
    assert rb.window() == 10
    rb.add_games(games[:20])
    assert rb.total_games == 20 and rb.window() == 10 and rb.n_games() == 10
    rb.add_games(games[20:60])
    assert rb.window() == 30 and rb.n_games() == 30
    rb.add_games(games[60:])
    assert rb.window() == 30 and rb.n_games() == 30  # 上限
    assert [g["tag"] for g in rb.window_games_list()] == list(range(70, 100))
    assert rb.n_positions() == sum(len(g["moves"]) for g in games[70:])
    # 固定の窓（window_frac 0）は今までどおり
    rb0 = _mk(tmp_path / "b", window_games=12, chunk_games=5)
    rb0.add_games(games[:40])
    assert rb0.window() == 12 and rb0.n_games() == 12 and [g["tag"] for g in rb0.window_games_list()] == list(range(28, 40))


def test_heldout_chunks_are_kept_out_of_training_and_reloaded(tmp_path: Path):
    games = _games(60, 2)
    rb = _mk(tmp_path, window_games=100, chunk_games=5, heldout_every_chunks=3, heldout_games=1000)
    rb.add_games(games[:23])  # チャンク 0..3 を書き、4 は書きかけ（3 局）
    held = {g["tag"] for g in rb.heldout}
    win = {g["tag"] for g in rb.window_games_list()}
    assert held == set(range(0, 5)) | set(range(15, 20))  # チャンク 0 と 3
    assert win == set(range(5, 15)) | set(range(20, 23)) and not (held & win)
    rb.add_games(games[23:])
    assert rb.chunk_index == 12
    held = {g["tag"] for g in rb.heldout}
    assert held == {t for t in range(60) if (t // 5) % 3 == 0} and rb.n_heldout() == 20 and rb.n_games() == 40
    # サンプルは窓の中の局だけから
    rng = np.random.default_rng(0)
    for _ in range(5):
        b = rb.sample(64, rng, 0.0, 0.5)
        assert b["sq"].shape == (64, 81, ls.SQ_FEATS)
    b = rb.sample_heldout(32, rng, 0.0, 0.5)
    assert b["wdl"].shape == (32, 3)
    # 再読込は同じ振り分け
    rb2 = _mk(tmp_path, window_games=100, chunk_games=5, heldout_every_chunks=3, heldout_games=1000)
    rb2.load(12, 60)
    assert {g["tag"] for g in rb2.heldout} == held and {g["tag"] for g in rb2.window_games_list()} == {g["tag"] for g in rb.window_games_list()}
    # held-out の上限と、窓の分だけ読む
    rb3 = _mk(tmp_path, window_games=15, chunk_games=5, heldout_every_chunks=3, heldout_games=7)
    rb3.load(12, 60)
    assert rb3.n_games() == 15 and rb3.n_heldout() == 7 and [g["tag"] for g in rb3.heldout_games_list()] == [33, 34, 45, 46, 47, 48, 49]  # held-out はチャンク 0・3・6・9
    assert [g["tag"] for g in rb3.window_games_list()] == [40, 41, 42, 43, 44, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]  # チャンク 8・10・11


def test_trim_keeps_last_window(tmp_path: Path):
    rb = _mk(tmp_path, window_games=20, chunk_games=5)
    games = _games(80, 3)
    for i in range(0, 80, 7):
        rb.add_games(games[i:i + 7])
    assert rb.n_games() == 20 and [g["tag"] for g in rb.window_games_list()] == list(range(60, 80))
    assert len(rb.games) <= 20 + max(5, 1) + 7 and rb.n_positions() == sum(len(g["moves"]) for g in games[60:])
    b = rb.sample(16, np.random.default_rng(1), 0.5, 0.5)
    assert b["z"].shape == (16,)


def test_measure_and_generalization(tmp_path: Path):
    torch.manual_seed(0)
    rb = _mk(tmp_path, window_games=100, chunk_games=5, heldout_every_chunks=2, heldout_games=100)
    rb.add_games(_games(40, 4))
    m = LibraNet(SMALL)
    g = generalization(m, rb, 64, np.random.default_rng(0), torch.device("cpu"), 0.5, 32)
    assert g is not None and g["window_games"] == 20 and g["heldout_games"] == 20
    for side in ("window", "heldout"):
        for ph in ("fuseki", "normal"):
            r = g[side][ph]
            assert r["n"] > 0 and r["mse_v"] is not None and 0 <= r["mse_v"] <= 4
            if r["n_policy"]:
                assert r["policy_ce"] > 0 and 0 <= r["policy_acc"] <= 1
    assert set(g["gap"]["fuseki"]) == {"corr_v", "mse_v", "policy_ce", "policy_acc"}
    # 目標を丸ごと当てるネットなら二乗誤差 0・相関 1 に近い: measure は同じバッチで決定的
    b = rb.sample(64, np.random.default_rng(2), 0.0, 0.5)
    assert measure(m, b, torch.device("cpu")) == measure(m, b, torch.device("cpu"))
    # held-out の無い run では None
    rb0 = _mk(tmp_path / "b", window_games=100, chunk_games=5)
    rb0.add_games(_games(10, 5))
    assert generalization(m, rb0, 16, np.random.default_rng(0), torch.device("cpu"), 0.5, 32) is None


def test_profile_and_cli(tmp_path: Path, capsys):
    sd = StateDir(tmp_path / "x")
    sd.create()
    rb = ReplayBuffer(sd.replay, sd.games, window_games=100, chunk_games=5, max_ply=320, count_from_41=True)
    rb.add_games(_games(30, 6))
    m = LibraNet(SMALL)
    ck = sd.checkpoints / "latest.pt"
    torch.save({"model": m.state_dict(), "step": 7, "config": {"net": {"d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64}}}, ck)
    rows = profile(m, sd.replay, [0, 3, 99], 3, 32, torch.device("cpu"), 0.5, 32, 320, True)
    assert rows[0]["games"] == 15 and rows[1]["games"] == 15 and rows[2]["games"] == 0
    assert main(["--root", str(tmp_path), "--run", "x", "genprof", "--chunks", "0,3", "--n-chunks", "3", "--positions", "32", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [r["chunk"] for r in out["rows"]] == [0, 3] and out["rows"][0]["fuseki"]["n"] > 0
    assert main(["--root", str(tmp_path), "--run", "x", "genprof", "--n-chunks", "2", "--positions", "16"]) == 0
    assert "chunk" in capsys.readouterr().out


def test_targets_are_from_the_side_to_move(tmp_path: Path):
    """学習目標の符号: z と v41 は手番側から見た値（3 手目は先手、4 手目は後手）。docs/restart-plan.md §2 C4。"""
    rb = _mk(tmp_path, window_games=100, chunk_games=1000)
    games = _games(6, 9)
    for g in games:
        g["result"] = 1       # 先手勝ち
        g["v41"] = 0.5        # 先手から見て +0.5
    rb.add_games(games)
    cum = np.cumsum(np.array([len(g["moves"]) for g in games], dtype=np.int64))
    from libra_league.replay import sample_batch

    rng = np.random.default_rng(0)
    b = sample_batch(games, 0, cum, 256, rng, 0.0, 0.5, 32, 320, True)
    # sample_batch は局面を一様に取るので、手番は glob の特徴（添字 25: 先手の番なら 1）で読む
    sente = b["glob"][:, 25] > 0.5
    z = b["z"]
    assert (z[sente] == 1).all() and (z[~sente] == -1).all()
    # 布石の目標 t = 0.5·z + 0.5·v41 は先手の番で 0.75、後手の番で −0.75。本将棋は z
    t = b["wdl"][:, 0] - b["wdl"][:, 2]
    fu = b["fuseki"]
    assert np.allclose(t[fu & sente], 0.75) and np.allclose(t[fu & ~sente], -0.75)
    assert np.allclose(t[~fu], z[~fu])
    # V̂41 の目標は soft_wdl(v41): 先手の番で (0.5, 0.5, 0)、後手の番で (0, 0.5, 0.5)
    assert np.allclose(b["v41"][sente], [0.5, 0.5, 0.0]) and np.allclose(b["v41"][~sente], [0.0, 0.5, 0.5])


def test_runner_measures_gen_by_games(tmp_path: Path):
    """gen_games > 0 なら、総局数がその分たまるごとに gen を測る（時間は見ない）。"""
    from types import SimpleNamespace

    from libra_league.runner import Runner

    calls = []

    class RB:
        total_games = 0

        def n_heldout(self):
            return 5

        def n_games(self):
            return 100

    rb = RB()
    fake = SimpleNamespace(cfg={"run": {"gen_minutes": 60, "gen_games": 1000, "gen_positions": 8}, "train": {"min_window_games": 10, "lambda_z": 0.5},
                                "search": {"policy_topk": 32}},
                           replay=rb, last_gen=0.0, last_gen_games=-1, gen=None, model=None, rng=None, device=None,
                           trainer=SimpleNamespace(step_count=3), log=lambda s: calls.append(s))
    import libra_league.runner as runner_mod
    import libra_league.genprof as gp

    orig = gp.generalization
    row = {"corr_v": 0.5, "mse_v": 0.3, "policy_ce": 2.0, "policy_acc": 0.3, "n": 4, "n_policy": 2}
    gp.generalization = lambda *a, **k: {"window": {"fuseki": dict(row), "normal": dict(row)}, "heldout": {"fuseki": dict(row), "normal": dict(row)},
                                        "gap": {}, "window_games": 1, "heldout_games": 1}
    try:
        Runner.maybe_measure_gen(fake, 1.0)            # 最初は測る
        assert fake.gen["games"] == 0 and len(calls) == 1
        rb.total_games = 900
        Runner.maybe_measure_gen(fake, 1.0 + 7200)     # 2 時間たっても 900 局では測らない
        assert len(calls) == 1
        rb.total_games = 1000
        Runner.maybe_measure_gen(fake, 1.0 + 7201)     # 1,000 局で測る
        assert len(calls) == 2 and fake.gen["games"] == 1000
    finally:
        gp.generalization = orig
