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


def test_eval_with_a_different_search_config_on_one_side(tmp_path: Path):
    """σ の形の比較（docs/ls2-settings.md §2）: 同じ重みで、B 側だけ mctx 形の σ で読ませる。"""
    sd, cfg = _run(tmp_path, games=10)
    scfg = {**cfg["search"], "full_sims": 8, "mate_nodes_root": 0, "proof_nodes": 0}
    rc = main(["--root", str(tmp_path), "--run", "ls", "eval", "--a", str(sd.checkpoints / "latest.pt"),
               "--b", str(sd.checkpoints / "latest.pt"), "--games", "4", "--sims", "8", "--concurrent", "4", "--threads", "2",
               "--b-set", "gumbel_rescale=true", "--b-set", "c_scale=0.1"])
    assert rc == 0
    # --out を渡さないときは run の eval/ ではなく experiments/ に置く（コンソールの Elo の一覧に混ざらないように）
    assert not (sd.root / "eval").exists()
    written = sorted((tmp_path / "experiments").glob("*-sigma.json"))
    assert len(written) == 1
    res = json.loads(written[0].read_text())
    assert res["n"] == 4 and res["search_b"] == {"gumbel_rescale": True, "c_scale": 0.1}
    # 側ごとに変えられない鍵は例外（棋譜と記録の形が枠ごとに変わってしまうもの）
    from libra_league.evaluate import load_model, play_match

    m = load_model(sd.checkpoints / "latest.pt", torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError):
        play_match(m, m, scfg, 1, 2, 1, 0, torch.device("cpu"), torch.float32, search_cfg_b={"policy_topk": 4})


def test_publish_result_writes_to_a_branch_without_touching_main(tmp_path: Path, monkeypatch):
    """--publish は progress ブランチへ 1 コミット足すだけ（作業ツリー・HEAD・main に触れない。絶対パスは消す）。"""
    import subprocess

    from libra_league.abtest import publish_result

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.name", "t"], ["config", "user.email", "t@example.com"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    (repo / "README.md").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    monkeypatch.setenv("HOME", str(tmp_path))
    res = {"ckpt": f"{tmp_path}/libra-run/ls/checkpoints/archive/ckpt_000062426.pt", "ckpt_step": 62426, "steps": 4,
           "seed": 5, "chunk_index": 6, "games_total": 60, "arms": {"lam05": {"diff": {}}, "lam10": {"diff": {"train.lambda_z": 1.0}}},
           "matches": [{"a": "lam05", "b": "lam10", "n": 4, "score_a": 0.5, "elo_a_minus_b": 0.0, "elo_ci95": [-1, 1], "avg_plies": 80}]}
    commit = publish_result(res, "x-abtest", repo, "progress", "experiments", log=lambda s: None, push=False)
    assert commit
    shown = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:experiments/x-abtest.json"], capture_output=True, text=True).stdout
    assert str(tmp_path) not in shown and "ckpt_000062426.pt" in shown
    md = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:experiments/x-abtest.md"], capture_output=True, text=True).stdout
    assert "lam05 vs lam10" in md and str(tmp_path) not in md
    # main と HEAD は動かない（push はしない: remote が無いので commit を返すだけ）
    assert subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout == head
    assert subprocess.run(["git", "-C", str(repo), "status", "--short"], capture_output=True, text=True).stdout == ""


def test_play_match_places_kings_uniformly_whatever_the_selfplay_bias(monkeypatch):
    """計測の対局は自己対局の後手玉四段目の偏り（search.gote_rank4_prob）を持ち込まず、36×36 から一様に置く
    （四段目の局は得点が両者 0.5 に寄るので、割合が変わると同じ強さの差でも Elo の出方が変わる）。"""
    from libra_league import evaluate

    seen = {}

    class Stop(Exception):
        pass

    def fake_selfplay(cfg, *args):
        seen.update(cfg)
        raise Stop

    monkeypatch.setattr(evaluate.librasearch, "SelfPlay", fake_selfplay)
    with pytest.raises(Stop):
        evaluate.play_match(None, None, {"full_sims": 8, "gote_rank4_prob": 0.05}, 1, 2, 1, 0, torch.device("cpu"), torch.float32)
    assert seen["gote_rank4_prob"] < 0
