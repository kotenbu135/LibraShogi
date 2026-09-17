# SPDX-License-Identifier: Apache-2.0
"""自己対局の推論の写し（InferenceNet）: 重みのその場更新、形が変わったときの作り直し、搾取者の行の選び分け、CUDA の捕獲。"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import librashogi as ls
from libra_league.config import DEFAULTS
from libra_league.selfplay import InferenceNet, SelfPlayLoop
from libra_net.model import LibraNet, NetConfig

SMALL = NetConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64)


def _inputs(n: int, seed: int, pin: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sq = (torch.rand(n, 81, ls.SQ_FEATS, generator=g) < 0.1).float()
    glob = torch.rand(n, ls.GLOB_FEATS, generator=g)
    return (sq.pin_memory(), glob.pin_memory()) if pin else (sq, glob)


def _eager(model: LibraNet, sq: torch.Tensor, glob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        p, w, _ = model.float().cpu().eval()(sq.cpu(), glob.cpu())
    return p, F.softmax(w, dim=-1)


def test_load_updates_weights_in_place():
    torch.manual_seed(0)
    a, b = LibraNet(SMALL), LibraNet(SMALL)
    net = InferenceNet(a, 4, torch.device("cpu"), torch.float32, "max-autotune")
    sq, glob = _inputs(4, 1)
    params = [p.data_ptr() for p in net.net.parameters()]
    assert net.load(b)
    assert [p.data_ptr() for p in net.net.parameters()] == params  # 捕獲したグラフが指す番地を変えない
    p, w = net(sq, glob)
    ep, ew = _eager(b, sq, glob)
    torch.testing.assert_close(p, ep)
    torch.testing.assert_close(w, ew)


def test_load_rejects_other_shape():
    net = InferenceNet(LibraNet(SMALL), 4, torch.device("cpu"), torch.float32, "none")
    assert not net.load(LibraNet(NetConfig(d_model=48, n_layers=2, n_heads=4, d_ff=64)))


def test_set_model_keeps_one_copy():
    loop = SelfPlayLoop(DEFAULTS["search"], 4, 1, 0, torch.device("cpu"), "float32")
    loop.set_model(LibraNet(SMALL))
    first = loop.model
    loop.set_model(LibraNet(SMALL))
    assert loop.model is first
    loop.set_model(LibraNet(NetConfig(d_model=48, n_layers=2, n_heads=4, d_ff=64)))
    assert loop.model is not first


@pytest.mark.parametrize("opponent_prior", [True, False])
def test_exploiter_round_passes_both_nets_to_the_engine(opponent_prior):
    """搾取者モードの round: 自分のネットと凍結した本体の全行の出力をそのまま apply2 に渡す（行の選び分けはエンジン。
    libra-search の test_two_nets_select_rows_like_the_exploiter_loop_and_cache_keeps_records）。"""
    torch.manual_seed(0)
    n = 6
    me, opp = LibraNet(SMALL), LibraNet(NetConfig(d_model=48, n_layers=2, n_heads=4, d_ff=96))
    loop = SelfPlayLoop(DEFAULTS["search"], n, 1, 0, torch.device("cpu"), "float32")
    loop.set_model(me)
    loop.set_opponent(opp, opponent_prior=opponent_prior)
    assert loop.engine.two_nets() and loop.engine.eval_cache_enabled()
    sq, glob = _inputs(n, 2)
    seen = {}

    class Engine:
        def collect(self, s, g):
            s[...] = sq.numpy()
            g[...] = glob.numpy()

        def proof(self):
            pass

        def apply2(self, l0, w0, l1, w1):
            seen.update(l0=l0.copy(), w0=w0.copy(), l1=l1.copy(), w1=w1.copy())

        def take_finished(self):
            return []

    loop.engine = Engine()
    loop.round()
    mp, mw = _eager(me, sq, glob)
    op, ow = _eager(opp, sq, glob)
    for got, want in (("l0", mp), ("w0", mw), ("l1", op), ("w1", ow)):
        torch.testing.assert_close(torch.from_numpy(seen[got]), want)


def test_set_opponent_none_returns_to_one_net():
    loop = SelfPlayLoop(DEFAULTS["search"], 4, 1, 0, torch.device("cpu"), "float32")
    loop.set_model(LibraNet(SMALL))
    loop.set_opponent(LibraNet(SMALL))
    assert loop.engine.two_nets()
    loop.set_opponent(None)
    assert not loop.engine.two_nets()
    loop.round()


def test_bad_compile_mode():
    with pytest.raises(ValueError):
        InferenceNet(LibraNet(SMALL), 4, torch.device("cpu"), torch.float32, "fast")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA が無い")
@pytest.mark.parametrize("mode", ["none", "default", "max-autotune"])
def test_cuda_graph_matches_eager_after_weight_update(mode):
    torch.manual_seed(0)
    n = 16
    a, b = LibraNet(SMALL), LibraNet(SMALL)
    dev = torch.device("cuda")
    net = InferenceNet(a, n, dev, torch.float16, mode)
    sq, glob = _inputs(n, 3, pin=True)
    p, w = net(sq, glob)
    assert net.graph is not None and net.mode_used.endswith("cudagraph")
    ep, ew = _eager(a, sq, glob)
    torch.testing.assert_close(p.cpu(), ep, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(w.cpu(), ew, atol=2e-3, rtol=2e-2)
    assert net.load(b)  # 学習後の重み: 捕獲し直さずに反映される
    sq2, glob2 = _inputs(n, 4, pin=True)
    p, w = net(sq2, glob2)
    ep, ew = _eager(b, sq2, glob2)
    torch.testing.assert_close(p.cpu(), ep, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(w.cpu(), ew, atol=2e-3, rtol=2e-2)
    net.release()  # グラフを捨てると、次の呼び出しで捕獲し直す
    assert net.graph is None
    p, _ = net(sq2, glob2)
    assert net.graph is not None
    torch.testing.assert_close(p.cpu(), ep, atol=2e-2, rtol=2e-2)
