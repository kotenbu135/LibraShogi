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


def test_offers_report_explains_why_each_offer_was_rejected():
    """「検索 6 件、条件に合う 0 件」だけでは直しようがないので、全件に理由を付け、1 つ緩めれば通る条件を出す（2026-09-15 の実際の検索結果）。"""
    def o(i, dph, cores, ghz, cpu, inet, rel=0.99, where="Texas, US"):
        return {"id": i, "dph_total": dph, "num_gpus": 1, "cpu_cores_effective": cores, "cpu_ghz": ghz, "cpu_name": cpu + " ",
                "inet_down": 600.0, "reliability2": rel, "cuda_max_good": 13.2, "inet_up_cost": inet, "inet_down_cost": inet, "geolocation": where}

    offers = [o(49263616, 0.12166, 6.0, 3.5, "Xeon® E5-2687W v4", 0.0039, where="South Korea, KR"),
              o(38692155, 0.12166, 4.0, 3.2, "Xeon® E5-2630 v3", 0.0039, where="South Korea, KR"),
              o(42372176, 0.21111, 8.0, 5.2716, "AMD Ryzen 7 9800X3D 8-Core Processor", 0.0390625, rel=0.9778, where="Kansas, US"),
              o(49877056, 0.54444, 64.0, 3.5413679, "AMD EPYC 7B13 64-Core Processor", 0.0260416),
              o(49702284, 0.54444, 64.0, 3.5413679, "AMD EPYC 7B13 64-Core Processor", 0.0260416, rel=0.9845),
              o(50306639, 1.35555, 32.0, 5.4627109, "AMD Ryzen 9 7945HX with Radeon Graphics", 0.0065, where="Wisconsin, US")]
    cond = {"max_dph": 1.30, "min_cores": 16, "min_cpu_ghz": 0.0, "min_rel": 0.90, "max_inet_cost": 0.02, "min_cuda": 12.9}
    rep = vast_cli.offers_report("RTX 5070 Ti", offers, cond, disk=30)
    assert rep["offers"] == 6 and rep["usable"] == 0 and rep["top"] == []
    assert [len(r["reasons"]) for r in rep["all"]] == [1, 1, 2, 1, 1, 1]
    assert rep["reject_counts"] == {"min_cores": 3, "max_inet_cost": 3, "max_dph": 1}
    assert [(h["key"], h["need"], h["offer"]["id"]) for h in rep["hints"]] == [("min_cores", 6, 49263616), ("max_inet_cost", 0.027, 49877056),
                                                                              ("max_dph", 1.36, 50306639)]
    text = rep["text"]
    assert "検索 6 件、条件に合う 0 件" in text and "コア数の下限 3 件" in text and "転送料の上限 3 件" in text
    assert "上限 $/h を 1.36 にすると" in text and "Ryzen 9 7945HX" in text and "コア 6 < 16" in text
    # 緩めると通って、借りる順（安い順）の先頭に来る
    rep = vast_cli.offers_report("RTX 5070 Ti", offers, {**cond, "max_dph": 1.36}, disk=30)
    assert rep["usable"] == 1 and rep["top"][0]["id"] == 50306639 and rep["all"][-1]["ok"] and "条件に合う 1 件" in rep["text"]


def test_start_continue_from_a_lost_host_uses_the_remaining_time(tmp_path: Path, monkeypatch, capsys):
    """入札で止められた・ホストが落ちたセッションの launcher が、残りの時間で次のセッションを起動する（ユーザーの決定「残り時間で借り直す」）。
    前のセッションの launcher はまだ動いている（自分が呼ぶ）ので、二重起動の拒否はそのセッションからの続きだけ通す。"""
    run_root = _run_dir(tmp_path).parent
    ss = Sessions(tmp_path / "cloud")
    now = time.time()
    settings = {"run": "ls", "gpu": "RTX 5070 Ti", "max_dph": 0.26, "hours": 3.0, "min_rel": 0.95, "min_cpu_ghz": 4.4, "min_cores": 16,
                "max_inet_cost": 0.02, "n_games": 512, "rent": "bid", "bid_margin": 0.1}
    prev = ss.create("ls")
    _write(prev / "session.json", {**settings, "started": now - 3600, "pid": os.getpid()})
    _write(prev / "instance.json", {"instance": 5, "offer": {"dph_eff": 0.2}, "t_rent": now - 3000, "t_bridge": now - 1800})
    seen = {}

    def fake_argv(a, run_dir, d):
        seen.update(hours=a.hours, gpu=a.gpu, rent=a.rent, bid_margin=a.bid_margin, max_dph=a.max_dph, min_rel=a.min_rel, n_games=a.n_games)
        return ["bash", "-c", "sleep 30"]

    monkeypatch.setattr(vast_cli, "launch_argv", fake_argv)
    base = ["--root", str(tmp_path / "cloud"), "start", "--run-root", str(run_root)]
    assert vast_cli.main(base) == 1 and "既に動いています" in capsys.readouterr().out  # 続きでなければ断る
    assert vast_cli.main([*base, "--continue-from", prev.name]) == 0
    nxt = ss.current()
    s = json.loads((nxt / "session.json").read_text(encoding="utf-8"))
    try:
        assert nxt != prev and s["continues"] == prev.name
        # 3 時間のうちブリッジが 0.5 時間動いたので残り 2.5 時間。締め切りは最初の開始 + 3 時間 + 30 分のまま引き継ぐ
        assert seen == {"hours": 2.5, "gpu": "RTX 5070 Ti", "rent": "bid", "bid_margin": 0.1, "max_dph": 0.26, "min_rel": 0.95, "n_games": 512}
        assert abs(s["deadline"] - (now - 3600 + 3 * 3600 + 1800)) < 1 and s["hours"] == 2.5 and s["rent"] == "bid"
    finally:
        import signal

        os.killpg(s["pid"], signal.SIGKILL)
    # 停止を要求されていた、残りが 15 分未満、締め切りが近いときは借り直さない
    for extra, t_bridge in (({"stop_requested": now}, now - 1800), ({}, now - 2.9 * 3600), ({"deadline": now + 600}, now - 60)):
        prev = ss.create("ls")
        _write(prev / "session.json", {**settings, "started": now - 3600, "pid": os.getpid(), **extra})
        _write(prev / "instance.json", {"instance": 6, "offer": {}, "t_rent": t_bridge, "t_bridge": t_bridge})
        assert vast_cli.main([*base, "--continue-from", prev.name]) == 3, extra
        assert ss.current() == prev and "借り直しません" in capsys.readouterr().out


def test_phase_shows_lost_host_and_relaunch():
    assert phase_of("bridge pid 7\ninstance 5 lost: exited / running\n") == "打ち切り"
    assert phase_of("bridge pid 7\ninstance 5 lost: exited\ndestroyed instance 5 (show_instance after: none)\nrelaunched: ls-20260915-130000\n") \
        == "終了（打ち切り・借り直し）"
    assert phase_of("bridge pid 7\ninstance 5 lost: exited\ndestroyed instance 5 (show_instance after: none)\n") == "終了"


def test_offers_report_for_bids_uses_effective_price():
    from libra_cloud.bench import annotate_price

    def o(i, dph, min_bid, cpu):
        return {"id": i, "dph_total": dph, "min_bid": min_bid, "is_bid": True, "num_gpus": 1, "cpu_cores_effective": 24.0, "cpu_ghz": 5.4,
                "cpu_name": cpu, "inet_down": 600.0, "reliability2": 0.99, "cuda_max_good": 13.2, "inet_up_cost": 0.004, "inet_down_cost": 0.004,
                "geolocation": "Japan, JP"}

    offers = annotate_price([o(1, 0.2944, 0.2667, "Intel Core i7-14700F"), o(2, 0.1817, 0.1733, "AMD Ryzen 9 7900X 12-Core Processor")], "bid", 0.1)
    cond = {"max_dph": 0.30, "min_cores": 16, "min_cpu_ghz": 0.0, "min_rel": 0.90, "max_inet_cost": 0.02, "min_cuda": 12.9, "price_key": "dph_eff"}
    rep = vast_cli.offers_report("RTX 5070 Ti", offers, cond, disk=30)
    assert rep["usable"] == 1 and rep["top"][0]["id"] == 2 and rep["top"][0]["dph"] == round(0.1817 - 0.1733 + 0.1733 * 1.1, 3)
    assert rep["top"][0]["bid"] == round(0.1733 * 1.1, 4) and rep["hints"][0]["key"] == "max_dph" and rep["hints"][0]["need"] == 0.33
    assert "入札" in rep["text"]


def test_launch_argv_passes_rent_type_and_bid_margin(tmp_path: Path):
    import argparse

    a = argparse.Namespace(gpu="RTX 5070 Ti", max_dph=0.28, hours=3.0, min_rel=0.94, min_cpu_ghz=4.4, min_cores=16, max_inet_cost=0.02,
                           n_games=512, rent="bid", bid_margin=0.1)
    cmd = vast_cli.launch_argv(a, tmp_path / "ls", tmp_path / "s")[2]
    assert "--rent bid" in cmd and "--bid-margin 0.1" in cmd


def test_status_cost_uses_effective_bid_price(tmp_path: Path):
    d = Sessions(tmp_path / "cloud").create("ls")
    now = 1_000_000.0
    _write(d / "session.json", {"run": "ls", "gpu": "RTX 5070 Ti", "max_dph": 0.28, "hours": 3.0, "pid": None, "rent": "bid"})
    _write(d / "launcher.log", "credit $9\ncreate #1\n")
    _write(d / "instance.json", {"instance": 5, "offer": {"dph_total": 0.25, "dph_eff": 0.2, "bid": 0.19}, "t_rent": now - 7200})
    assert session_status(d, now=now)["est_cost_usd"] == 0.4


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")


def _stamp(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def test_transfer_cost_is_estimated_for_sessions_without_byte_counts():
    offer = {"inet_down_cost": 0.005, "inet_up_cost": 0.004}
    # 記録あり: 送った 1 GB × 0.005 + 取ってきた 0.25 GB × 0.004
    assert vast_cli.transfer_usd({"push_bytes": 1_000_000_000, "pull_bytes": 250_000_000}, offer, None) == (0.006, False)
    # 記録なし: 重み 20 MB × 100 回 = 2 GB、局 100,000 × 5,300 バイト = 0.53 GB
    old = {"pushes": {"weights/latest.pt": 100, "openings.json": 3}, "games": 100_000}
    assert vast_cli.transfer_usd(old, offer, 20_000_000) == (round(2.0 * 0.005 + 0.53 * 0.004, 4), True)
    assert vast_cli.transfer_usd(old, offer, None) == (None, False)  # 重みの大きさが分からない
    assert vast_cli.transfer_usd(old, {}, 20_000_000) == (None, False)  # 単価が無い


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
    offer = {"dph_total": 0.25, "gpu_name": "RTX 5080", "cpu_name": "AMD Ryzen 9 7900 ", "geolocation": "Australia, AU", "reliability2": 0.998,
             "inet_down_cost": 0.005, "inet_up_cost": 0.004}
    _write(b / "session.json", {"run": "ls", "gpu": "RTX 5080", "max_dph": 0.3, "hours": 3.0, "started": t0 + 3590, "pid": None,
                                "stop_requested": t_bridge + 7000})
    _write(b / "launcher.log", "credit $9\ncreate #1\nssh ready\nbridge pid 7\nbridge: stopping\ndestroyed instance 5 (show_instance after: none)\n")
    _write(b / "instance.json", {"instance": 5, "offer": offer, "t_rent": t_rent, "t_bridge": t_bridge})
    _write(b / "result.json", {"worker": "vast1", "offer": offer, "t_ready_s": 140, "rented_h": 2.0, "est_cost_usd": 0.5,
                               "bridge": {"time": t_bridge + 7200, "games": 40000, "files": 400, "rejected_files": 1, "errors": 2,
                                          "push_bytes": 2_000_000_000, "pull_bytes": 500_000_000}})
    # 3 回目: 動いている途中（result.json はまだ無い。launcher のプロセスは自分自身）。転送料の単価が無い（古い記録）
    c = root / "ls-20260914-160000"
    t3 = t0 + 4 * 3600
    _write(c / "session.json", {"run": "ls", "gpu": "RTX 5070 Ti", "max_dph": 0.28, "hours": 3.0, "started": t3, "pid": os.getpid()})
    _write(c / "launcher.log", "credit $9\ncreate #1\nssh ready\nbridge pid 7\n")
    old_offer = {k: v for k, v in offer.items() if not k.startswith("inet_")}
    _write(c / "instance.json", {"instance": 6, "offer": {**old_offer, "dph_total": 0.2, "gpu_name": "RTX 5070 Ti"}, "t_rent": t3, "t_bridge": t3})
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
    # 転送料: 送った 2 GB × $0.005 + 取ってきた 0.5 GB × $0.004。100 万局あたりは借りた費用と転送料の合計で割る
    assert s2["transfer_usd"] == 0.012 and not s2["transfer_estimated"] and s2["total_usd"] == 0.512
    assert s2["usd_per_1m"] == round(0.512 / 35000 * 1e6, 2) and s2["usd_per_1m_gross"] == round(0.512 / 40000 * 1e6, 2)
    assert s3["alive"] and s3["phase"] == "稼働" and s3["rented_h"] == 1.0 and s3["est_cost_usd"] == 0.2 and s3["stale_games"] == 100
    assert s3["games"] == 30000 and s3["net_games"] == 29900
    assert s3["transfer_usd"] is None and s3["total_usd"] == 0.2
    tot = h["totals"]
    assert tot["sessions"] == 3 and tot["rented"] == 2 and tot["est_cost_usd"] == 0.7 and tot["games"] == 70000 and tot["net_games"] == 64900
    assert tot["transfer_usd"] == 0.012 and tot["total_usd"] == 0.712 and tot["usd_per_1m"] == round(0.712 / 64900 * 1e6, 2)
    assert h["months"] == [{"month": "2026-09", "sessions": 3, "est_cost_usd": 0.7, "transfer_usd": 0.012, "total_usd": 0.712,
                            "games": 70000, "net_games": 64900}]
    assert vast_cli.main(["--root", str(root), "history", "--json", "--run-root", str(tmp_path / "runs")]) == 0
    assert [s["name"] for s in json.loads(capsys.readouterr().out)["sessions"]] == [a.name, b.name, c.name]
    assert vast_cli.main(["--root", str(root), "history", "--run-root", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "RTX 5080" in out and "35,000" in out and "合計" in out
    assert vast_cli.main(["--root", str(tmp_path / "empty"), "history", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["sessions"] == []


def test_worker_id_is_passed_and_continued(tmp_path):
    """2 台目のワーカーを別の --root で動かすとき、ワーカー名（対局ファイル名・seed・ls の by_worker）を vast1 と分ける。
    借り直しの鎖でも同じ名前を引き継ぐ。"""
    import argparse

    from libra_cloud.vast_cli import continue_settings, launch_argv

    a = argparse.Namespace(gpu="RTX 5090", max_dph=0.6, hours=1.5, min_rel=0.94, min_cpu_ghz=4.4, min_cores=16, max_inet_cost=0.04,
                           n_games=512, rent="on-demand", bid_margin=0.1, worker_id="vast2")
    cmd = launch_argv(a, tmp_path / "ls", tmp_path / "s")[-1]
    assert "--id vast2" in cmd
    prev = tmp_path / "prev"
    prev.mkdir()
    now = time.time()
    (prev / "session.json").write_text(json.dumps({"run": "ls", "gpu": "RTX 5090", "hours": 1.5, "started": now, "worker_id": "vast2"}))
    cont, why = continue_settings(prev, now)
    assert cont is not None and cont["worker_id"] == "vast2", why
    # 名前の無い古いセッションからの借り直しは既定（vast1）のまま
    (prev / "session.json").write_text(json.dumps({"run": "ls", "gpu": "RTX 5090", "hours": 1.5, "started": now}))
    cont, _ = continue_settings(prev, now)
    assert "worker_id" not in cont


def test_offers_report_shows_the_expected_speed_and_cost_and_ranks_by_value():
    """「候補を見る」の表に見込みの局/日と 100 万局あたりの費用を出し、借りる順をその費用の安い順にする。"""
    from libra_cloud import hosts
    from libra_cloud.vast_cli import offers_report

    def off(i, dph, gpu, cpu, mid):
        return {"id": i, "dph_total": dph, "dph_eff": dph, "num_gpus": 1, "cpu_cores_effective": 16, "cpu_ghz": 5.0, "inet_down": 500.0,
                "reliability2": 0.99, "cuda_max_good": 12.9, "inet_up_cost": 0.004, "inet_down_cost": 0.004,
                "gpu_name": gpu, "cpu_name": cpu, "machine_id": mid}

    table = hosts.speed_table([{"name": "a", "gpu": "RTX 5070 Ti", "cpu": "Intel Xeon Gold 6130", "machine_id": 4242,
                                "games_per_day": 300_000, "span_h": 1.0},
                               {"name": "b", "gpu": "RTX 5080", "cpu": "AMD Ryzen 9 7900", "machine_id": 13,
                                "games_per_day": 877_037, "span_h": 1.0}])
    offers = hosts.annotate([off(1, 0.262, "RTX 5080", "AMD Ryzen 9 7900 12-Core Processor", 13),
                             off(2, 0.150, "RTX 5070 Ti", "Intel Xeon Gold 6130 16-Core Processor", 4242)], table, "dph_eff")
    cond = {"max_dph": 0.28, "min_cores": 16, "min_cpu_ghz": 4.4, "min_rel": 0.94, "max_inet_cost": 0.02, "min_cuda": 12.9,
            "price_key": "dph_eff"}
    rep = offers_report("RTX 5070 Ti", offers, cond, 30)
    assert [r["id"] for r in rep["top"]] == [1, 2]  # 安いほうが遅いので、100 万局あたりでは高い GPU が勝つ（$7.17 対 $12.00）
    assert [r["id"] for r in rep["all"]] == [2, 1]  # 表そのものは値段の安い順のまま
    assert rep["top"][0]["est_usd_per_1m"] is not None and rep["top"][0]["est_from"] == "machine"
    assert "◎87.7万" in rep["text"] and "$7.17" in rep["text"] and "$12.00" in rep["text"]
    assert "見込みは過去に借りたホストの実測" in rep["text"]


def test_history_counts_what_each_interruption_cost(tmp_path: Path, capsys):
    """打ち切り（入札で止められる）と借り直しの損を history が数える（コンソールの「クラウド履歴」の要約）。"""
    root = tmp_path / "cloud"
    t0 = time.mktime(time.strptime("2026-09-20 09:00:00", "%Y-%m-%d %H:%M:%S"))
    offer = {"dph_eff": 0.20, "dph_total": 0.26, "bid": 0.20, "gpu_name": "RTX 5070 Ti", "cpu_name": "AMD Ryzen 9 7900",
             "inet_down_cost": 0.0, "inet_up_cost": 0.0}
    # 1 回目: 入札で 3 時間の予定が、打ち始めて 1 時間で止められた（最後の回収は止められる 2 分前）
    a = root / "ls-20260920-090000"
    _write(a / "session.json", {"run": "ls", "gpu": "RTX 5070 Ti", "hours": 3.0, "started": t0, "pid": None, "rent": "bid"})
    _write(a / "launcher.log", "credit $9\ncreate #1\nssh ready\nbridge pid 7\ninstance 5 lost: exited / running\nrelaunched: ls-20260920-102000\n")
    _write(a / "instance.json", {"instance": 5, "offer": offer, "t_rent": t0, "t_bridge": t0 + 600})
    _write(a / "result.json", {"worker": "vast1", "offer": offer, "lost": True, "rented_h": 1.2, "est_cost_usd": 0.24,
                               "bridge": {"time": t0 + 4200, "last_pull": t0 + 4080, "games": 20000, "files": 200,
                                          "push_bytes": 0, "pull_bytes": 0}})
    # 2 回目: 残りの 1.83 時間で借り直した（準備に 12 分。この準備代が打ち切りのぶん余分に増えた費用）
    b = root / "ls-20260920-102000"
    t_b = t0 + 4200
    _write(b / "session.json", {"run": "ls", "gpu": "RTX 5070 Ti", "hours": 1.83, "started": t_b, "pid": None, "rent": "bid",
                                "continues": a.name})
    _write(b / "launcher.log", "credit $9\ncreate #1\nssh ready\nbridge pid 8\nbridge: stopping\ndestroyed instance 6 (show_instance after: none)\n")
    _write(b / "instance.json", {"instance": 6, "offer": offer, "t_rent": t_b, "t_bridge": t_b + 720})
    _write(b / "result.json", {"worker": "vast1", "offer": offer, "rented_h": 2.03, "est_cost_usd": 0.406,
                               "bridge": {"time": t_b + 720 + 6588, "last_pull": t_b + 720 + 6588, "games": 36000, "files": 360,
                                          "push_bytes": 0, "pull_bytes": 0}})
    _write(tmp_path / "runs" / "ls" / "log.txt", "")

    h = vast_cli.history(root, tmp_path / "runs", now=t_b + 8000)
    s1, s2 = h["sessions"]
    assert s1["lost"] and not s2["lost"] and s2["continues"] == a.name
    # 止められた 2 分前の回収から、借り直しが打ち始めるまでの 14 分（0.233 時間）は局が 1 つも増えない
    assert s1["dark_h"] == 0.233 and s1["unused_h"] == 0.0 and s1["lost_games"] == round(0.233 / 24 * s1["games_per_day"])
    assert s1["setup_h"] == 0.167 and s1["extra_setup_usd"] == 0.0  # 1 回目の準備は打ち切りが無くても払う
    assert s2["setup_h"] == 0.2 and s2["extra_setup_usd"] == 0.04   # 借り直しの準備 12 分 × $0.20/h
    w = h["interrupts"]
    assert w["interruptions"] == 1 and w["relaunches"] == 1 and w["not_relaunched"] == 0
    assert w["h_per_loss"] == w["bridge_h"] and w["extra_setup_usd"] == 0.04
    assert w["usd_per_1m"] > w["usd_per_1m_ideal"] and w["waste_pct"] > 0
    assert [g["name"] for g in w["by_rent"]] == ["入札"] and w["by_rent"][0]["lost"] == 1
    assert vast_cli.main(["--root", str(root), "history", "--run-root", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "打ち切り 1 回" in out and "借り直せず 0 回" in out and "打ち切りが無ければ" in out
