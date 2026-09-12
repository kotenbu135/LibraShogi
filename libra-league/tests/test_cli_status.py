# SPDX-License-Identifier: Apache-2.0
"""`libra status --json`（Windows の管理コンソールが読む形）。"""
import json

from libra_league.cli import main
from libra_league.state import StateDir


def test_status_json_no_run(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "--run", "x", "status", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["exists"] is False and out["process"] == "not running" and out["status"] is None


def test_status_json_with_run(tmp_path, capsys):
    sd = StateDir(tmp_path / "x")
    sd.create()
    sd.write_state({"step": 5, "generation": 1, "games_total": 42, "last_checkpoint": 1.0, "elapsed": 2.0})
    sd.status_json.write_text(json.dumps({"time": "t", "step": 5, "generation": 1, "games_total": 42, "games_per_day_1h": 100, "window_games": 10, "elapsed_h": 0.1, "active_games": 8}), encoding="utf-8")
    sd.set_flag("EVAL_NOW")
    sd.append_log("hello")
    sd.append_log("world")
    (sd.root / "run.lock").write_text("999999999")  # 存在しない pid → not running
    assert main(["--root", str(tmp_path), "--run", "x", "status", "--json", "--tail", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["exists"] and out["process"] == "not running"
    assert out["flags"] == ["EVAL_NOW"] and "throttle" not in out
    assert out["state"]["games_total"] == 42 and out["status"]["games_per_day_1h"] == 100
    assert len(out["log_tail"]) == 1 and out["log_tail"][0].endswith("world")
    # 従来の表示も壊れていない
    assert main(["--root", str(tmp_path), "--run", "x", "status"]) == 0
    assert "not running" in capsys.readouterr().out


def test_default_match_model_prefers_onnx(tmp_path):
    from libra_league.cli import default_match_model

    sd = StateDir(tmp_path / "x")
    sd.create()
    assert default_match_model(sd).name == "latest.pt"
    (sd.checkpoints / "latest.onnx").write_bytes(b"x")
    assert default_match_model(sd).name == "latest.onnx"
