# SPDX-License-Identifier: Apache-2.0
"""bin/libra-vast（libra_cloud.vast_cli）: 起動の前提の検査、二重起動の拒否、切り離した起動、停止（STOP か SIGTERM）、段階と費用の表示。
vast.ai は使わない（launch_argv を手元のダミーのコマンドに差し替える）。"""
import json
import os
import time
from pathlib import Path

from libra_cloud import vast_cli
from libra_cloud.vast_cli import Sessions, phase_of, pid_alive, session_status


def _run_dir(tmp_path: Path, workers: bool = True) -> Path:
    run = tmp_path / "runs" / "ls"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "latest.pt").write_bytes(b"x")
    (run / "config.toml").write_text('run_id = "ls"\n' + ("[workers]\nenabled = true\n" if workers else ""), encoding="utf-8")
    return run


def _argv(tmp_path: Path, *rest: str) -> list[str]:
    return ["--root", str(tmp_path / "cloud"), *rest]


def _wait(cond, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, what
        time.sleep(0.05)


def test_phase_of_follows_launcher_log():
    assert phase_of("") == ""
    assert phase_of("11:00 credit $9.55; run x\n") == "準備"
    assert phase_of("credit $9\ncreate #1 $0.2/h -> instance 5\n") == "インスタンス作成"
    assert phase_of("credit $9\ncreate #1\nssh ready after 100 s\nbridge pid 7 for 3 h\n") == "稼働"
    assert phase_of("bridge pid 7\nbridge: placed 100 games\nbridge: stopping the worker\n") == "停止処理"
    assert phase_of("create #1\nbridge pid 7\nbridge: stopping\ndestroyed instance 5 (show_instance after: none)\nresult x\n") == "終了"
    assert phase_of("credit $9\n10 offers, 0 usable; cheapest: \n") == "終了（条件に合うオファーなし）"
    assert phase_of("credit $1.00\ncredit below $2.00; not renting\n") == "終了（残高不足）"


def test_start_checks_the_learner_and_refuses_a_second_session(tmp_path: Path, monkeypatch, capsys):
    _run_dir(tmp_path, workers=False)
    base = ["start", "--run-root", str(tmp_path / "runs"), "--hours", "2"]
    assert vast_cli.main(_argv(tmp_path, *base)) == 2
    assert "[workers] enabled" in capsys.readouterr().out and Sessions(tmp_path / "cloud").current() is None
    _run_dir(tmp_path / "w")
    base = ["start", "--run-root", str(tmp_path / "w" / "runs"), "--hours", "2", "--gpu", "RTX 3090", "--max-dph", "0.2"]
    seen = {}

    def fake_argv(a, run_dir, d):
        seen.update(run_dir=run_dir, gpu=a.gpu, hours=a.hours)
        return ["bash", "-c", "echo 'credit $9.00; run x'; echo 'create #1 $0.200/h -> instance 5 '; sleep 60"]

    monkeypatch.setattr(vast_cli, "launch_argv", fake_argv)
    assert vast_cli.main(_argv(tmp_path, *base)) == 0
    d = Sessions(tmp_path / "cloud").current()
    s = json.loads((d / "session.json").read_text(encoding="utf-8"))
    try:
        assert seen == {"run_dir": tmp_path / "w" / "runs" / "ls", "gpu": "RTX 3090", "hours": 2.0} and s["max_dph"] == 0.2
        assert pid_alive(s["pid"])
        assert vast_cli.main(_argv(tmp_path, *base)) == 1 and "既に動いています" in capsys.readouterr().out
        _wait(lambda: "create #1" in (d / "launcher.log").read_text(), 10, "launcher log")
        st = session_status(d)
        assert st["session"]["alive"] and st["session"]["phase"] == "インスタンス作成" and st["session"]["gpu"] == "RTX 3090"
        # ブリッジがまだ無い（借りる途中）ので、停止はプロセスグループへの SIGTERM
        assert vast_cli.main(_argv(tmp_path, "stop")) == 0
        _wait(lambda: not pid_alive(s["pid"]), 10, "launcher stopped")
        assert json.loads((d / "session.json").read_text(encoding="utf-8"))["stop_requested"] > 0
        assert "異常終了" in session_status(d)["session"]["phase"]  # 消した記録（destroyed instance）が無いまま止まった
        assert vast_cli.main(_argv(tmp_path, "stop")) == 1
    finally:
        if pid_alive(s["pid"]):
            import os
            import signal

            os.killpg(s["pid"], signal.SIGKILL)


def test_stop_with_a_running_bridge_asks_it_to_drain(tmp_path: Path, monkeypatch, capsys):
    _run_dir(tmp_path)
    monkeypatch.setattr(vast_cli, "launch_argv", lambda a, run_dir, d: ["bash", "-c", "echo 'bridge pid 1 for 3 h'; sleep 60"])
    assert vast_cli.main(_argv(tmp_path, "start", "--run-root", str(tmp_path / "runs"))) == 0
    d = Sessions(tmp_path / "cloud").current()
    pid = json.loads((d / "session.json").read_text(encoding="utf-8"))["pid"]
    try:
        (d / "bridge").mkdir()
        assert vast_cli.main(_argv(tmp_path, "stop")) == 0
        assert (d / "bridge" / "STOP").exists() and pid_alive(pid)  # 殺さない（ブリッジが残りを取ってから抜ける）
        _wait(lambda: "bridge pid" in (d / "launcher.log").read_text(), 10, "launcher log")
        assert session_status(d)["session"]["phase"] == "稼働（停止を要求済み）"
    finally:
        import os
        import signal

        os.killpg(pid, signal.SIGKILL)


def test_status_reports_cost_remaining_and_result(tmp_path: Path, capsys):
    ss = Sessions(tmp_path / "cloud")
    d = ss.create("ls")
    now = 1_000_000.0
    (d / "session.json").write_text(json.dumps({"run": "ls", "gpu": "RTX 5070 Ti", "max_dph": 0.28, "hours": 3.0, "pid": None}), encoding="utf-8")
    (d / "launcher.log").write_text("credit $9\ncreate #1 -> instance 5\nssh ready after 90 s\nbridge pid 9 for 3.0 h\n", encoding="utf-8")
    (d / "instance.json").write_text(json.dumps({"instance": 5, "offer": {"dph_total": 0.25}, "t_rent": now - 7200, "t_bridge": now - 3600}),
                                     encoding="utf-8")
    (d / "bridge").mkdir()
    (d / "bridge" / "bridge.json").write_text(json.dumps({"games": 1200, "files": 12, "rejected_files": 0, "errors": 0,
                                                          "verify_ms_per_game": 0.7}), encoding="utf-8")
    st = session_status(d, tail=2, now=now)
    # launcher が落ちて消した記録が無い: 課金が続いているかもしれないので異常として出す
    assert st["session"]["phase"].startswith("異常終了") and st["rented_h"] == 2.0 and st["est_cost_usd"] == 0.5
    assert st["remaining_h"] == 2.0 and st["bridge"]["games"] == 1200 and len(st["log_tail"]) == 2
    with open(d / "launcher.log", "a", encoding="utf-8") as f:
        f.write("destroyed instance 5 (show_instance after: none)\n")
    (d / "result.json").write_text(json.dumps({"rented_h": 3.2, "est_cost_usd": 0.8}), encoding="utf-8")
    st = session_status(d, now=now)
    assert st["session"]["phase"] == "終了" and st["rented_h"] == 3.2 and st["est_cost_usd"] == 0.8 and st["remaining_h"] is None
    assert vast_cli.main(["--root", str(tmp_path / "cloud"), "status", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["session"]["phase"] == "終了" and out["bridge"]["games"] == 1200
    assert vast_cli.main(["--root", str(tmp_path / "empty"), "status"]) == 0 and "まだありません" in capsys.readouterr().out


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")


def _stamp(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def test_history_lists_every_session_with_cost_per_million_games(tmp_path: Path, capsys):
    """管理コンソールの「クラウド履歴」: 過去のセッションごとの費用・回収局数・学習側が古すぎて捨てた局・100 万局あたりの費用。"""
    root = tmp_path / "cloud"
    t0 = time.mktime(time.strptime("2026-09-14 12:00:00", "%Y-%m-%d %H:%M:%S"))
    # 1 回目: 条件に合うオファーが無く借りなかった（session.json と launcher.log だけ）
    a = root / "ls-20260914-120000"
    _write(a / "session.json", {"run": "ls", "gpu": "RTX 5070 Ti", "max_dph": 0.28, "hours": 3.0, "started": t0, "pid": None})
    _write(a / "launcher.log", "credit $9.55; run x\n8 offers, 0 usable; cheapest: \n")
    # 2 回目: 2 時間借りて 40,000 局、そのうち学習側が 5,000 局を捨てた（ユーザーが停止）
    b = root / "ls-20260914-130000"
    t_rent, t_bridge = t0 + 3600, t0 + 3600 + 300
    offer = {"dph_total": 0.25, "gpu_name": "RTX 5080", "cpu_name": "AMD Ryzen 9 7900 ", "geolocation": "Australia, AU", "reliability2": 0.998}
    _write(b / "session.json", {"run": "ls", "gpu": "RTX 5080", "max_dph": 0.3, "hours": 3.0, "started": t0 + 3590, "pid": None,
                                "stop_requested": t_bridge + 7000})
    _write(b / "launcher.log", "credit $9\ncreate #1\nssh ready\nbridge pid 7\nbridge: stopping\ndestroyed instance 5 (show_instance after: none)\n")
    _write(b / "instance.json", {"instance": 5, "offer": offer, "t_rent": t_rent, "t_bridge": t_bridge})
    _write(b / "result.json", {"worker": "vast1", "offer": offer, "t_ready_s": 140, "rented_h": 2.0, "est_cost_usd": 0.5,
                               "bridge": {"time": t_bridge + 7200, "games": 40000, "files": 400, "rejected_files": 1, "errors": 2}})
    # 3 回目: 動いている途中（result.json はまだ無い。launcher のプロセスは自分自身）
    c = root / "ls-20260914-160000"
    t3 = t0 + 4 * 3600
    _write(c / "session.json", {"run": "ls", "gpu": "RTX 5070 Ti", "max_dph": 0.28, "hours": 3.0, "started": t3, "pid": os.getpid()})
    _write(c / "launcher.log", "credit $9\ncreate #1\nssh ready\nbridge pid 7\n")
    _write(c / "instance.json", {"instance": 6, "offer": {**offer, "dph_total": 0.2, "gpu_name": "RTX 5070 Ti"}, "t_rent": t3, "t_bridge": t3})
    _write(c / "bridge" / "bridge.json", {"time": t3 + 3600, "games": 30000, "files": 300, "rejected_files": 0, "errors": 0})
    (root / "current").write_text(c.name, encoding="utf-8")
    # 学習側のログ: 2 回目の間に 5,000 局、3 回目の間に 100 局を捨てた。2 回目より前の行と別のワーカーの行は数えない
    lines = [f"{_stamp(t0 + 60)} workers: dropped 100 stale games from vast1 (weights step 1, 3000 steps behind)"]
    lines += [f"{_stamp(t_bridge + 3000)} workers: dropped 100 stale games from vast1 (weights step 2, 2441 steps behind)"] * 50
    lines += [f"{_stamp(t_bridge + 3100)} workers: dropped 100 stale games from w1 (weights step 2, 2441 steps behind)",
              f"{_stamp(t_bridge + 3200)} workers: ingested 100 games (total 100)",
              f"{_stamp(t3 + 1800)} workers: dropped 100 stale games from vast1 (weights step 3, 2500 steps behind)"]
    _write(tmp_path / "runs" / "ls" / "log.txt", "\n".join(lines) + "\n")

    h = vast_cli.history(root, tmp_path / "runs", now=t3 + 3600)
    s1, s2, s3 = h["sessions"]
    assert s1["name"] == a.name and s1["phase"] == "終了（条件に合うオファーなし）" and s1["est_cost_usd"] is None and s1["games"] == 0
    assert s1["usd_per_1m"] is None
    assert s2["gpu"] == "RTX 5080" and s2["cpu"] == "AMD Ryzen 9 7900" and s2["where"] == "Australia, AU" and s2["dph"] == 0.25
    assert s2["phase"] == "終了" and s2["stopped_by_user"] and s2["rented_h"] == 2.0 and s2["est_cost_usd"] == 0.5 and s2["t_ready_s"] == 140
    assert s2["games"] == 40000 and s2["stale_games"] == 5000 and s2["net_games"] == 35000 and s2["rejected_files"] == 1 and s2["errors"] == 2
    assert s2["bridge_h"] == 2.0 and s2["games_per_day"] == 480000
    assert s2["usd_per_1m"] == round(0.5 / 35000 * 1e6, 2) and s2["usd_per_1m_gross"] == round(0.5 / 40000 * 1e6, 2)
    assert s3["alive"] and s3["phase"] == "稼働" and s3["rented_h"] == 1.0 and s3["est_cost_usd"] == 0.2 and s3["stale_games"] == 100
    assert s3["games"] == 30000 and s3["net_games"] == 29900
    tot = h["totals"]
    assert tot["sessions"] == 3 and tot["rented"] == 2 and tot["est_cost_usd"] == 0.7 and tot["games"] == 70000 and tot["net_games"] == 64900
    assert tot["usd_per_1m"] == round(0.7 / 64900 * 1e6, 2)
    assert h["months"] == [{"month": "2026-09", "sessions": 3, "est_cost_usd": 0.7, "games": 70000, "net_games": 64900}]
    assert vast_cli.main(["--root", str(root), "history", "--json", "--run-root", str(tmp_path / "runs")]) == 0
    assert [s["name"] for s in json.loads(capsys.readouterr().out)["sessions"]] == [a.name, b.name, c.name]
    assert vast_cli.main(["--root", str(root), "history", "--run-root", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "RTX 5080" in out and "35,000" in out and "合計" in out
    assert vast_cli.main(["--root", str(tmp_path / "empty"), "history", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["sessions"] == []
