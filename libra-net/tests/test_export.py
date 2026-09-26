# SPDX-License-Identifier: Apache-2.0
import torch

from libra_net.export_onnx import check, export_checkpoint, load_checkpoint
from libra_net.model import LibraNet, NetConfig


def test_onnx_matches_torch(tmp_path):
    cfg = NetConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64)
    m = LibraNet(cfg)
    ckpt = tmp_path / "small.pt"
    torch.save({"model": m.state_dict(), "config": {"net": cfg.__dict__}, "step": 7}, ckpt)
    out = tmp_path / "small.onnx"
    meta = export_checkpoint(ckpt, out)
    assert meta["libra_step"] == "7"
    model, _ = load_checkpoint(ckpt)
    assert check(model, out, n=5) < 1e-4
    import onnx

    om = onnx.load(str(out))
    names = [i.name for i in om.graph.input]
    assert names == ["sq", "glob"] and [o.name for o in om.graph.output] == ["policy", "wdl", "v41"]
    assert any(p.key == "libra_step" for p in om.metadata_props)


def test_onnx_drops_the_opp_head(tmp_path):
    """補助の頭（opp_head・own_head）は学習だけで使う。ONNX は頭の無いネットと同じ 3 出力・同じ大きさの重みになる。"""
    import onnx

    sizes = {}
    for opp in (False, True):
        cfg = NetConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64, opp_head=opp, own_head=opp)
        m = LibraNet(cfg)
        ckpt = tmp_path / f"m{opp}.pt"
        torch.save({"model": m.state_dict(), "config": {"net": cfg.__dict__}, "step": 7}, ckpt)
        out = tmp_path / f"m{opp}.onnx"
        export_checkpoint(ckpt, out)
        model, _ = load_checkpoint(ckpt)
        assert check(model, out, n=3) < 1e-4
        om = onnx.load(str(out))
        assert [o.name for o in om.graph.output] == ["policy", "wdl", "v41"]
        sizes[opp] = sum(int(torch.tensor(i.dims).prod()) for i in om.graph.initializer)
    assert sizes[True] == sizes[False]
