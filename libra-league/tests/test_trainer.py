# SPDX-License-Identifier: Apache-2.0
"""学習の 1 ステップ（Trainer）: compile の設定、CPU では eager、CUDA では compile しても損失が eager と浮動小数点の誤差の範囲で同じ。"""
import copy

import numpy as np
import pytest
import torch

import librashogi as ls
from libra_league.config import DEFAULTS
from libra_league.trainer import Trainer
from libra_net.model import LibraNet, NetConfig

SMALL = NetConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64)


def _batch(n: int, seed: int) -> dict:
    r = np.random.default_rng(seed)
    t = r.uniform(-1, 1, n).astype(np.float32)
    wdl = np.stack([np.clip(t, 0, 1), 1 - abs(t), np.clip(-t, 0, 1)], 1).astype(np.float32)
    return {"sq": (r.random((n, 81, ls.SQ_FEATS)) < 0.1).astype(np.float32), "glob": r.random((n, ls.GLOB_FEATS)).astype(np.float32),
            "wdl": wdl, "v41": wdl.copy(), "fuseki": r.random(n) < 0.4, "policy_idx": r.integers(0, ls.POLICY_SIZE, (n, 8)).astype(np.int64),
            "policy_p": r.dirichlet(np.ones(8), n).astype(np.float32), "policy_valid": r.random(n) < 0.9}


def _cfg(mode: str) -> dict:
    return {**DEFAULTS["train"], "compile": mode}


def test_bad_compile_mode():
    with pytest.raises(ValueError):
        Trainer(LibraNet(SMALL), _cfg("fast"), torch.device("cpu"))


def test_cpu_stays_eager_and_state_dict_keys_unchanged():
    m = LibraNet(SMALL)
    tr = Trainer(m, _cfg("max-autotune"), torch.device("cpu"))
    assert tr.mode_used == "eager" and tr.forward is m
    tr.step(_batch(8, 0))
    assert set(tr.state_dict()["model"]) == set(LibraNet(SMALL).state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("mode", ["default", "max-autotune"])
def test_cuda_compile_matches_eager(mode):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    base = LibraNet(SMALL)
    runs = {}
    for m in ("none", mode):
        model = copy.deepcopy(base).to(dev)
        tr = Trainer(model, _cfg(m), dev)
        losses = [tr.step(_batch(64, i))["loss"] for i in range(5)]
        runs[m] = (losses, tr.mode_used, [p.detach().float().cpu() for p in model.parameters()])
    assert runs[mode][1] == f"compile({mode})"
    np.testing.assert_allclose(runs[mode][0], runs["none"][0], rtol=0, atol=1e-4)
    for a, b in zip(runs[mode][2], runs["none"][2]):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-4)


def _opp(batch: dict, seed: int) -> dict:
    r = np.random.default_rng(seed)
    n = len(batch["sq"])
    return {**batch, "opp_idx": r.integers(0, ls.POLICY_SIZE, (n, 8)).astype(np.int64), "opp_p": r.dirichlet(np.ones(8), n).astype(np.float32),
            "opp_valid": r.random(n) < 0.3}


def test_opp_head_starts_from_weights_without_it():
    """補助方策「相手の次の手」（KataGo [Wu19] §3.4）: 頭の無い重みと AdamW の状態から引き継ぎ、頭だけ初期値から学ぶ。
    forward（対局と ONNX）は 3 つのまま、forward_aux だけが 4 つ目を出す。"""
    torch.manual_seed(0)
    old = Trainer(LibraNet(SMALL), _cfg("none"), torch.device("cpu"))
    for i in range(2):
        old.step(_batch(8, i))
    saved = copy.deepcopy(old.state_dict())
    with pytest.raises(ValueError):
        Trainer(LibraNet(SMALL), {**_cfg("none"), "opp_weight": 0.15}, torch.device("cpu"))
    net = NetConfig(**{**SMALL.__dict__, "opp_head": True})
    new = Trainer(LibraNet(net), {**_cfg("none"), "opp_weight": 0.15}, torch.device("cpu"))
    new.load_state_dict(saved)
    assert new.step_count == 2 and new.aux
    for k, v in saved["model"].items():
        assert torch.equal(new.model.state_dict()[k], v)
    # 引き継いだパラメータは AdamW の状態（exp_avg）も同じ。頭の分は状態なし
    by_name = dict(new.model.named_parameters())
    old_by_name = dict(old.model.named_parameters())
    for n, p in by_name.items():
        st = new.opt.state.get(p, {})
        if n.startswith("opp_head."):
            assert not st
        else:
            assert torch.equal(st["exp_avg"], old.opt.state[old_by_name[n]]["exp_avg"])
    out = new.step(_opp(_batch(8, 5), 5))
    assert "opp" in out and out["opp"] > 0 and all(new.opt.state.get(p) for p in by_name.values())
    sq = torch.zeros(2, 81, ls.SQ_FEATS)
    glob = torch.zeros(2, ls.GLOB_FEATS)
    three, four = new.model(sq, glob), new.model.forward_aux(sq, glob)
    assert len(three) == 3 and len(four) == 4 and four[3].shape == (2, ls.POLICY_SIZE)
    for a, b in zip(three, four):
        assert torch.equal(a, b)
    # 頭の無い重みに頭のある重みを戻そうとしたら誤り（黙って捨てない）
    with pytest.raises(RuntimeError):
        Trainer(LibraNet(SMALL), _cfg("none"), torch.device("cpu")).load_state_dict(new.state_dict())


def test_accum_steps_splits_the_batch():
    """accum_steps = 2 はバッチを半分ずつ流して 1 回だけ更新する（大きいネットがメモリに載らないとき用）。"""
    torch.manual_seed(0)
    base = LibraNet(SMALL)
    a = Trainer(copy.deepcopy(base), _cfg("none"), torch.device("cpu"))
    b = Trainer(copy.deepcopy(base), {**_cfg("none"), "accum_steps": 2}, torch.device("cpu"))
    batch = _batch(8, 1)
    ra, rb = a.step(batch), b.step(batch)
    assert a.step_count == b.step_count == 1
    assert abs(ra["value"] - rb["value"]) < 1e-5  # 価値の損失は行の平均なので、半分ずつの平均の平均と同じ
    for p, q in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=5e-4)
    with pytest.raises(ValueError):
        Trainer(copy.deepcopy(base), {**_cfg("none"), "accum_steps": 3}, torch.device("cpu")).step(batch)
