# SPDX-License-Identifier: Apache-2.0
"""`libra run` の監視役: 異常終了したら起動し直し、利用者が止めたときは起動し直さない。"""
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from libra_league.state import StateDir
from libra_league.supervise import EXIT_ALREADY_RUNNING, acquire_lock, child_argv, exit_code, running_pid, should_restart, supervise


def _child(tmp_path: Path, body: str) -> list[str]:
    """起動回数を count に数えてから body を実行する子。"""
    code = textwrap.dedent(f"""
        import os, signal, sys, time
        from pathlib import Path
        root = Path({str(tmp_path / "x")!r})
        c = root / "count"
        n = int(c.read_text()) + 1 if c.exists() else 1
        c.write_text(str(n))
        print(f"child start {{n}}", flush=True)
    """) + textwrap.dedent(body)
    return [sys.executable, "-c", code]


def _count(sd: StateDir) -> int:
    return int((sd.root / "count").read_text())


def test_exit_code_and_should_restart():
    assert exit_code(-6) == 134 and exit_code(-9) == 137 and exit_code(1) == 1
    assert should_restart(0, 10.0, 0) == (False, 0)  # STOP フラグで正常終了
    assert should_restart(EXIT_ALREADY_RUNNING, 1.0, 0) == (False, 0)
    for rc in (-int(signal.SIGTERM), -int(signal.SIGINT), -int(signal.SIGHUP), 143, 130):
        assert should_restart(rc, 1.0, 0)[0] is False  # 利用者や OS が止めた
    assert should_restart(-6, 5 * 3600.0, 4) == (True, 0)  # 長く動いた後の abort は数え直し
    assert should_restart(-9, 5.0, 0) == (True, 1)
    assert should_restart(1, 5.0, 3, max_quick=5) == (True, 4)
    assert should_restart(1, 5.0, 4, max_quick=5) == (False, 5)  # 短命な失敗が続いたら諦める


def test_supervise_restarts_after_crash(tmp_path):
    sd = StateDir(tmp_path / "x")
    argv = _child(tmp_path, """
        if n == 1:
            os.kill(os.getpid(), signal.SIGKILL)
        sys.exit(0)
    """)
    assert supervise(sd, argv, delay=0.0) == 0
    assert _count(sd) == 2
    log = sd.log.read_text(encoding="utf-8")
    assert "rc=137" in log and "restarting" in log
    out = (sd.root / "stdout.log").read_text(encoding="utf-8")
    assert "child start 1" in out and "child start 2" in out  # 子の出力は stdout.log に残る


def test_supervise_gives_up_on_repeated_quick_failures(tmp_path):
    sd = StateDir(tmp_path / "x")
    argv = _child(tmp_path, "sys.exit(1)\n")
    assert supervise(sd, argv, delay=0.0, healthy=3600.0, max_quick=3) == 1
    assert _count(sd) == 3
    assert "giving up" in sd.log.read_text(encoding="utf-8")


def test_supervise_does_not_restart_when_already_running(tmp_path):
    sd = StateDir(tmp_path / "x")
    argv = _child(tmp_path, f"sys.exit({EXIT_ALREADY_RUNNING})\n")
    assert supervise(sd, argv, delay=0.0) == EXIT_ALREADY_RUNNING
    assert _count(sd) == 1


def test_supervise_stop_flag_during_wait(tmp_path):
    sd = StateDir(tmp_path / "x")
    argv = _child(tmp_path, """
        (root / "STOP").write_text("1")
        sys.exit(1)
    """)
    t0 = time.monotonic()
    assert supervise(sd, argv, delay=30.0) == 0
    assert time.monotonic() - t0 < 10.0
    assert _count(sd) == 1 and not sd.flag("STOP")


def test_supervise_forwards_sigterm_and_exits(tmp_path):
    sd = StateDir(tmp_path / "x")
    sd.create()
    argv = _child(tmp_path, """
        (root / "child.pid").write_text(str(os.getpid()))
        time.sleep(60)
    """)
    code = f"import sys; from pathlib import Path; from libra_league.state import StateDir; from libra_league.supervise import supervise; sys.exit(supervise(StateDir(Path({str(sd.root)!r})), {argv!r}, delay=0.0))"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    sup = subprocess.Popen([sys.executable, "-c", code], env=env)
    pidf = sd.root / "child.pid"
    deadline = time.monotonic() + 20.0
    while not pidf.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pidf.exists()
    child_pid = int(pidf.read_text())
    sup.send_signal(signal.SIGTERM)
    assert sup.wait(timeout=10) == 128 + int(signal.SIGTERM)
    assert _count(sd) == 1
    assert not Path(f"/proc/{child_pid}").exists() or not Path(f"/proc/{child_pid}/cmdline").read_bytes()


def test_lock_ignores_dead_and_reused_pids(tmp_path):
    lock = tmp_path / "run.lock"
    assert running_pid(lock) is None
    assert acquire_lock(lock) is None and lock.read_text() == str(os.getpid())
    lock.write_text("999999999")  # 存在しない pid
    assert acquire_lock(lock) is None
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])  # libra ではないプロセス（pid の使い回し）
    libra = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "libra_league"])
    try:
        time.sleep(0.2)
        lock.write_text(str(other.pid))
        assert running_pid(lock) is None and acquire_lock(lock) is None
        lock.write_text(str(libra.pid))
        assert running_pid(lock) == libra.pid and acquire_lock(lock) == libra.pid
    finally:
        other.kill()
        libra.kill()
        other.wait()
        libra.wait()


def test_child_argv():
    argv = child_argv("/r", "lx", None)
    assert argv[:3] == [sys.executable, "-m", "libra_league.cli"]
    assert argv[3:] == ["--root", "/r", "--run", "lx", "run", "--no-supervise"]
    assert child_argv("/r", "ls", "c.toml")[-2:] == ["--config", "c.toml"]
