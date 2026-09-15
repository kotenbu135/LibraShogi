# SPDX-License-Identifier: Apache-2.0
"""手元でベンチの束を作る: チェックポイントからワーカー用の重み（fp16）とベンチ用の設定、ワーカーに要るソースの tar。

    PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-cloud \\
      .venv/bin/python -m libra_cloud.prepare --ckpt ~/libra-run/ls/checkpoints/latest.pt --config ~/libra-run/ls/config.toml --out <dir>

チェックポイントは読むだけ（稼働中の run は latest.pt を原子的に置き換えるので、読み途中で壊れない）。
--worker を付けると常駐ワーカー用（vast_worker.py）: 布石を使う run では、ブリッジが送る先を布石のパスにする。
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
from pathlib import Path

# ワーカーに要るもの（C++ のシミュレータと探索、ネット、ランナー、玉配置表の検証、ベンチの道具）。エンジンと ONNX Runtime は要らない
BUNDLE_PATHS = ("CMakeLists.txt", "libra-sim", "libra-search", "libra-net", "libra-league", "libra-scale", "libra-cloud")
HOST_RUN_ROOT = "/root/libra/run"  # 束を展開するホストの場所（host_setup.sh / host_worker.sh と揃える）


def write_scale_dir(scale_dir: Path, out: Path) -> Path:
    """libra-scale seq の run からホスト用のディレクトリ（束の scale/<名前>。host_scale.sh と揃える）を作る:
    config.json・active.json の写しと、config のモデルから作った fp16 の重み weights.pt。"""
    import json
    import shutil

    import torch

    from libra_league.workers import publish_weights
    from libra_net.model import LibraNet, NetConfig

    cfg = json.loads((scale_dir / "config.json").read_text(encoding="utf-8"))
    sd = torch.load(cfg["model"], map_location="cpu", weights_only=False)
    net = sd.get("config", {}).get("net", {})
    m = LibraNet(NetConfig.from_dict(net))
    m.load_state_dict(sd["model"])
    dst = out / "scale" / scale_dir.name
    (dst / "inbox").mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "active.json"):
        shutil.copyfile(scale_dir / name, dst / name)
    publish_weights(dst / "weights.pt", m, int(sd.get("step", 0)), net, "scale")
    return dst


def write_run_dir(ckpt: Path, config: Path, out: Path, n_games: int = 512, threads: int = 12, worker: bool = False) -> Path:
    """worker が真なら常駐ワーカー用: 布石を使う run では、ブリッジが送る先（ホストの /root/libra/run/<run_id>/openings.json）を指す。"""
    import torch

    from libra_league.config import dump_toml, load_config
    from libra_league.workers import publish_weights
    from libra_net.model import LibraNet, NetConfig

    from .bench import bench_config

    cfg = load_config(config)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = sd.get("config", {}).get("net") or cfg["net"]
    if net != cfg["net"]:
        raise SystemExit(f"net in {ckpt} {net} != config {cfg['net']}")
    m = LibraNet(NetConfig.from_dict(net))
    m.load_state_dict(sd["model"])
    run = out / "run" / cfg["run_id"]
    (run / "weights").mkdir(parents=True, exist_ok=True)
    (run / "inbox").mkdir(exist_ok=True)
    publish_weights(run / "weights" / "latest.pt", m, int(sd.get("step", 0)), net, cfg["run_id"])
    host_cfg = bench_config(cfg, n_games, threads)
    if worker and cfg["selfplay"].get("openings"):
        host_cfg["selfplay"]["openings"] = f"{HOST_RUN_ROOT}/{cfg['run_id']}/openings.json"
    (run / "config.toml").write_text(dump_toml(host_cfg), encoding="utf-8")
    return run


def make_bundle(repo: Path, out: Path) -> Path:
    """git の HEAD にあるワーカー用のソースと out/run・out/scale（あるもの）を 1 つの tar.gz にする（作業ツリーの未コミットの変更は入らない）。"""
    src = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "HEAD", *BUNDLE_PATHS], check=True, capture_output=True).stdout
    tar = out / "bundle.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        with tarfile.open(fileobj=io.BytesIO(src)) as s:
            for m in s.getmembers():
                t.addfile(m, s.extractfile(m) if m.isfile() else None)
        for sub in ("run", "scale"):
            if (out / sub).exists():
                t.add(out / sub, arcname=sub)
    return tar


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="libra_cloud.prepare")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scale-dir", default=None, help="libra-scale seq の run（全組の検証対局のワーカー用。--ckpt・--config の代わり）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-games", type=int, default=512)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--worker", action="store_true", help="常駐ワーカー用（布石をブリッジが送る先から読む）")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.scale_dir:
        run = write_scale_dir(Path(a.scale_dir).expanduser(), out)
    elif a.ckpt and a.config:
        run = write_run_dir(Path(a.ckpt).expanduser(), Path(a.config).expanduser(), out, a.n_games, a.threads, a.worker)
    else:
        ap.error("--scale-dir か、--ckpt と --config を指定する")
    tar = make_bundle(Path(__file__).resolve().parents[2], out)
    print(f"run dir {run}\nbundle {tar} ({tar.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
