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
