# SPDX-License-Identifier: Apache-2.0
"""「仮」の設定値のオフライン比較（libra abtest、docs/restart-plan.md §0、docs/ls2-settings.md §5.2）。"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import librasearch
import librashogi as ls
from libra_league.abtest import apply_sets, diff_of, parse_arm, parse_set, run_abtest, window_key
from libra_league.cli import main
from libra_league.config import load_config
from libra_league.replay import ReplayBuffer
from libra_league.state import StateDir
from libra_league.trainer import Trainer
from libra_net.model import LibraNet, NetConfig

SMALL = {"d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64, "dropout": 0.0}


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


def _run(tmp_path: Path, games: int = 60) -> tuple[StateDir, dict]:
    """小さな run（リプレイのチャンクと、重み＋AdamW の状態を持つチェックポイント）を作る。"""
    sd = StateDir(tmp_path / "ls")
    sd.create()
    cfg = load_config(None)
    cfg["net"] = dict(SMALL)
    cfg["train"].update({"batch_size": 8, "compile": "none", "warmup_steps": 1, "window_games": 20, "min_window_games": 1})
    cfg["run"].update({"chunk_games": 10, "heldout_every_chunks": 3, "heldout_games": 100})
    rb = ReplayBuffer(sd.replay, sd.games, cfg["train"]["window_games"], cfg["run"]["chunk_games"], cfg["search"]["max_ply"],
                      cfg["search"]["count_from_41"], heldout_every_chunks=cfg["run"]["heldout_every_chunks"],
                      heldout_games=cfg["run"]["heldout_games"])
    rb.add_games(_games(games, 3))
    model = LibraNet(NetConfig.from_dict(cfg["net"]))
    trainer = Trainer(model, cfg["train"], torch.device("cpu"))  # AdamW の状態も本番と同じ形で持たせる
    torch.save({**trainer.state_dict(), "step": 7, "config": cfg,
                "state": {"step": 7, "chunk_index": rb.chunk_index, "games_total": rb.total_games}}, sd.checkpoints / "latest.pt")
    return sd, cfg


def test_parse_set_types_and_unknown_keys():
    assert parse_set("train.lambda_z=1.0") == ("train", "lambda_z", 1.0)
    assert parse_set("train.batch_size=256") == ("train", "batch_size", 256)
    assert parse_set("search.gumbel_rescale=true") == ("search", "gumbel_rescale", True)
    assert parse_set("train.compile=none") == ("train", "compile", "none")
    for bad in ("train.lambda_zz=1.0", "nosuch.key=1", "train.lambda_z", "lambda_z=1.0", "search.gumbel_rescale=yes"):
        with pytest.raises(ValueError):
            parse_set(bad)
    assert parse_arm("lam10:train.lambda_z=1.0, train.mirror_prob=0.0") == ("lam10", ["train.lambda_z=1.0", "train.mirror_prob=0.0"])
    assert parse_arm("base") == ("base", [])


def test_apply_sets_does_not_touch_base_and_rejects_net():
    cfg = load_config(None)
    out = apply_sets(cfg, ["train.lambda_z=1.0"])
    assert out["train"]["lambda_z"] == 1.0 and cfg["train"]["lambda_z"] == 0.5
    assert diff_of(cfg, out) == {"train.lambda_z": 1.0}
    with pytest.raises(ValueError):
        apply_sets(cfg, ["net.d_model=64"])
    # 窓の作り方が同じ腕は同じ鍵になり、窓の大きさを変えた腕は別の鍵になる（窓を 1 つだけ持つためのグループ分け）
    assert window_key(cfg) == window_key(apply_sets(cfg, ["train.lambda_z=1.0"]))
    assert window_key(cfg) != window_key(apply_sets(cfg, ["train.window_games=1000"]))


def test_arms_see_the_same_positions_and_only_the_setting_differs(tmp_path: Path):
    """同じ設定の 2 つの腕は同じ重みになり（同じ seed・同じバッチ）、λ を変えた腕だけが違う重みになる。"""
    sd, _ = _run(tmp_path)
    logs: list[str] = []
    res = run_abtest(sd, load_config(None), sd.checkpoints / "latest.pt",
                     ["a", "b", "lam10:train.lambda_z=1.0"], [], steps=6, games=0, sims=8, concurrent=4, threads=2, seed=5,
                     positions=64, every=3, vs_base=False, out_dir=tmp_path / "out", device=torch.device("cpu"),
                     chunk_index=None, games_total=None, log=logs.append)
    a, b, lam = (torch.load(tmp_path / "out" / f"{n}.pt", map_location="cpu", weights_only=False) for n in ("a", "b", "lam10"))
    assert a["step"] == 13 and all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert any(not torch.equal(a["model"][k], lam["model"][k]) for k in a["model"])
    assert res["arms"]["lam10"]["diff"] == {"train.lambda_z": 1.0} and res["arms"]["a"]["diff"] == {}
    # 物差しは 2 通り（腕の間で同じ λ=1.0 の物差しと、腕自身の λ）。布石と本将棋で分かれている
    for which in ("gen_z", "gen_own"):
        assert res["arms"]["a"][which]["heldout"]["fuseki"]["n"] > 0
    assert res["arms"]["a"]["gen_z"]["window"]["normal"]["corr_v"] == res["arms"]["b"]["gen_z"]["window"]["normal"]["corr_v"]
    assert len(res["arms"]["a"]["curve"]) == 2 and res["arms"]["a"]["window_games"] > 0
    # 稼働中の run のディレクトリには書かない（読むだけ）
    assert sorted(p.name for p in sd.checkpoints.iterdir()) == ["latest.pt"]
    assert not (sd.root / "eval").exists() and not sd.state_json.exists()
    assert json.loads((tmp_path / "out" / "abtest.json").read_text())["ckpt_step"] == 7


def test_window_from_checkpoint_state_and_cli(tmp_path: Path):
    sd, _ = _run(tmp_path)
    rc = main(["--root", str(tmp_path), "--run", "ls", "abtest", "--ckpt", str(sd.checkpoints / "latest.pt"),
               "--arm", "lam05:train.lambda_z=0.5", "--arm", "lam10:train.lambda_z=1.0", "--steps", "4", "--games", "4",
               "--sims", "8", "--concurrent", "4", "--threads", "2", "--positions", "64", "--log-every", "2",
               "--device", "cpu", "--out", str(tmp_path / "ab")])
    assert rc == 0
    res = json.loads((tmp_path / "ab" / "abtest.json").read_text())
    assert res["chunk_index"] == 6 and res["games_total"] == 60  # チェックポイントに保存された窓の位置
    got = {(m["a"], m["b"]) for m in res["matches"]}
    assert got == {("lam05", "lam10"), ("lam05", "base"), ("lam10", "base")}
    assert all(m["n"] == 4 for m in res["matches"])
