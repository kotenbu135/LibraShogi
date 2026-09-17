# SPDX-License-Identifier: Apache-2.0
"""外部計測（libra match）で使う重みの決め方。torch は使わない（書き出し済みの ONNX がある場合だけを見る）。"""
from libra_league.cli import resolve_match_model
from libra_league.state import StateDir


def test_model_option_wins(tmp_path):
    sd = StateDir(tmp_path / "x")
    sd.create()
    (sd.checkpoints / "latest.onnx").write_bytes(b"latest")
    assert resolve_match_model(sd, "/tmp/given.onnx", None) == "/tmp/given.onnx"


def test_ckpt_uses_onnx_beside_it(tmp_path):
    """--ckpt には .pt を渡し、隣の同じ名前の .onnx（節目の写し）で打つ。"""
    sd = StateDir(tmp_path / "x")
    sd.create()
    (sd.checkpoints / "latest.onnx").write_bytes(b"latest")
    arch = sd.checkpoints / "archive"
    arch.mkdir()
    ck = arch / "ckpt_000028908.pt"
    ck.write_bytes(b"pt")
    onnx = arch / "ckpt_000028908.onnx"
    onnx.write_bytes(b"snapshot")
    assert resolve_match_model(sd, None, str(ck)) == str(onnx)


def test_default_is_latest(tmp_path):
    sd = StateDir(tmp_path / "x")
    sd.create()
    (sd.checkpoints / "latest.onnx").write_bytes(b"latest")
    assert resolve_match_model(sd, None, None) == str(sd.checkpoints / "latest.onnx")


def test_missing_ckpt_falls_back_to_latest(tmp_path):
    """節目の写しも .pt も無いとき（回転で消えた）は止めずに最新の重みで打つ。"""
    sd = StateDir(tmp_path / "x")
    sd.create()
    (sd.checkpoints / "latest.onnx").write_bytes(b"latest")
    logs = []
    got = resolve_match_model(sd, None, str(sd.checkpoints / "ckpt_000000100.pt"), logs.append)
    assert got == str(sd.checkpoints / "latest.onnx")
    assert logs and "falling back" in logs[0]
