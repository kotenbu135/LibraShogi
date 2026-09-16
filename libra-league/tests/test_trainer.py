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
