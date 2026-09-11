# SPDX-License-Identifier: Apache-2.0
"""学習チェックポイント（.pt）→ 推論用 ONNX（opset 固定、fp32、バッチ可変）。

  python -m libra_net.export_onnx <ckpt.pt> <out.onnx> [--check]

入力 sq [B, 81, 32]・glob [B, 32]、出力 policy [B, 2268]・wdl [B, 3]・v41 [B, 3]（いずれもロジット）。
メタデータに step とネット設定を入れる。libra.exe（libra-engine）はこの形だけを読む。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import GLOB_FEATS, SQ_FEATS, SQ_NB, LibraNet, NetConfig

OPSET = 17
INPUT_NAMES = ("sq", "glob")
OUTPUT_NAMES = ("policy", "wdl", "v41")


def load_checkpoint(path: Path) -> tuple[LibraNet, dict]:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    cfg = sd.get("config", {}).get("net", {})
    m = LibraNet(NetConfig.from_dict(cfg))
    m.load_state_dict(sd["model"])
    m.eval()
    return m, sd


def export_model(model: LibraNet, out: Path, meta: dict | None = None) -> None:
    model = model.float().eval()
    sq = torch.zeros(2, SQ_NB, SQ_FEATS)
    glob = torch.zeros(2, GLOB_FEATS)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model, (sq, glob), str(out), input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
            dynamic_axes={n: {0: "batch"} for n in INPUT_NAMES + OUTPUT_NAMES}, opset_version=OPSET, dynamo=False,
        )
    if meta:
        import onnx

        m = onnx.load(str(out))
        for k, v in meta.items():
            e = m.metadata_props.add()
            e.key, e.value = k, v if isinstance(v, str) else json.dumps(v)
        onnx.save(m, str(out))


def export_checkpoint(ckpt: Path, out: Path) -> dict:
    model, sd = load_checkpoint(ckpt)
    meta = {"libra_step": str(sd.get("step", 0)), "libra_net": sd.get("config", {}).get("net", {}),
            "libra_source": ckpt.name, "license": "Apache-2.0"}
    export_model(model, out, meta)
    return meta


def check(model: LibraNet, onnx_path: Path, n: int = 4, seed: int = 0) -> float:
    """ONNX Runtime（CPU）と PyTorch の出力の最大差を返す。"""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    sq = rng.standard_normal((n, SQ_NB, SQ_FEATS), dtype=np.float32)
    glob = rng.standard_normal((n, GLOB_FEATS), dtype=np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(list(OUTPUT_NAMES), {"sq": sq, "glob": glob})
    with torch.no_grad():
        ref = model.float()(torch.from_numpy(sq), torch.from_numpy(glob))
    return max(float(np.abs(o - r.numpy()).max()) for o, r in zip(outs, ref))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    meta = export_checkpoint(a.ckpt, a.out)
    print(f"exported {a.out} step {meta['libra_step']} net {meta['libra_net']}")
    if a.check:
        model, _ = load_checkpoint(a.ckpt)
        print(f"max |onnxruntime - torch| = {check(model, a.out):.2e}")


if __name__ == "__main__":
    main()
