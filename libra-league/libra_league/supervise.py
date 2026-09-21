# SPDX-License-Identifier: Apache-2.0
"""`libra run` の監視役: ランナー本体を子プロセスで回し、異常終了したら 1 分後に起動し直す。

CUDA の illegal memory access はプロセスごと abort するのでプロセス内では復帰できない。タスク スケジューラの
「失敗時に再起動」も起動後の異常終了には効かなかった（docs/decisions.md 2026-09-13）。監視役は torch を読み込まない。
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from .state import StateDir

EXIT_ALREADY_RUNNING = 3
# 設定がどこにも無い run を既定値で始めようとした。起動し直さない（設定を置くまで何度試しても同じ）
EXIT_NO_CONFIG = 4
# 利用者や OS が止めた（Ctrl+C、kill、wsl --shutdown、ログオフ）。起動し直さない
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
RESTART_DELAY = 60.0
HEALTHY_SECONDS = 900.0  # これより長く動いてから落ちたら連続失敗の数を数え直す（チェックポイント間隔 10 分より長く）
MAX_QUICK_FAILURES = 5
STOP_WAIT = 180.0  # 停止処理中の run に起動が来たら終わるのをこれだけ待つ（チェックポイントと ONNX の書き出しで数秒）
LEFTOVER_FLAGS = ("STOP", "PAUSE")  # 止まっている run に残っていたら起動時に消す（PAUSE は廃止した一時停止の名残）


def exit_code(rc: int) -> int:
    """Popen の returncode（シグナルで死ぬと負）をシェルの終了コードに直す。"""
    return 128 - rc if rc < 0 else rc


def should_restart(rc: int, uptime: float, quick_failures: int, healthy: float = HEALTHY_SECONDS,
                   max_quick: int = MAX_QUICK_FAILURES) -> tuple[bool, int]:
    """(起動し直すか, 更新後の連続失敗数)。0（STOP）、already running、設定が無い、止めるシグナルでは起動し直さない。"""
    code = exit_code(rc)
    if code in (0, EXIT_ALREADY_RUNNING, EXIT_NO_CONFIG) or code in {128 + int(s) for s in STOP_SIGNALS}:
        return False, 0
    n = 0 if uptime >= healthy else quick_failures + 1
    return n < max_quick, n


def running_pid(lock: Path) -> int | None:
    """run.lock の pid が生きている libra_league のプロセスならその pid。使い回された pid は別プロセスとみなす。

    自己対局ワーカー（`libra worker`）は run.lock を取らないので、pid がワーカーに使い回されていても持ち主とみなさない。"""
    try:
        pid = int(lock.read_text().strip())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return None
    return pid if b"libra_league" in cmdline and b"worker" not in cmdline.split(b"\0") else None


def acquire_lock(lock: Path) -> int | None:
    """run.lock を取る。取れたら None、別の libra が持っていればその pid。"""
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return None
    except FileExistsError:
        pid = running_pid(lock)
        if pid is not None and pid != os.getpid():
            return pid
        lock.write_text(str(os.getpid()))
        return None


def prepare_start(sd: StateDir, log: Callable[[str], None], stop_wait: float = STOP_WAIT) -> int | None:
    """起動の前処理。起動してよければ None、起動しないなら終了コード（二重起動）。

    操作は「起動」と「停止」だけなので（docs/decisions.md 2026-09-14）、どの順に押されても起動が空振りしないようにする:
    停止処理中（STOP があり lock の持ち主が生きている）なら終わるのを待ってから起動し、
    止まっている run に残った STOP・PAUSE は消す（残っていると起動直後に止まる／待機する）。STOP の無い稼働中の run には起動しない。
    """
    lock = sd.root / "run.lock"
    pid = running_pid(lock)
    if pid is not None:
        if not sd.flag("STOP"):
            log(f"start: already running (pid {pid})")
            return EXIT_ALREADY_RUNNING
        log(f"start: waiting for the running process to stop (pid {pid})")
        end = time.monotonic() + stop_wait
        while running_pid(lock) is not None:
            if time.monotonic() >= end:
                log(f"start: pid {pid} did not stop within {stop_wait:.0f} s; not starting")
                return EXIT_ALREADY_RUNNING
            time.sleep(0.5)
    left = [f for f in LEFTOVER_FLAGS if sd.flag(f)]
    for f in left:
        sd.clear_flag(f)
    if left:
        log(f"start: cleared leftover flags {' '.join(left)}")
    return None


def child_argv(root: str, run: str, config: str | None) -> list[str]:
    argv = [sys.executable, "-m", "libra_league.cli", "--root", root, "--run", run, "run", "--no-supervise"]
    return argv + (["--config", config] if config else [])


def supervise(sd: StateDir, argv: list[str], *, delay: float = RESTART_DELAY, healthy: float = HEALTHY_SECONDS,
              max_quick: int = MAX_QUICK_FAILURES, log: Callable[[str], None] | None = None, stop_wait: float = STOP_WAIT) -> int:
    """argv を子として回す。子の標準出力・標準エラーは <run>/stdout.log に追記する。戻り値は終了コード。"""
    def _log(msg: str) -> None:
        print(msg, flush=True)
        sd.append_log(msg)

    log = log or _log
    sd.create()
    code = prepare_start(sd, log, stop_wait)
    if code is not None:
        return code
    stopping: list[int] = []
    child: list[subprocess.Popen] = []

    def on_signal(signum, frame):
        stopping.append(signum)
        if child and child[0].poll() is None:
            child[0].send_signal(signum)

    old = {s: signal.signal(s, on_signal) for s in STOP_SIGNALS}
    quick = 0
    try:
        while True:
            t0 = time.monotonic()
            with open(sd.root / "stdout.log", "ab") as out:
                out.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(argv)}\n".encode())
                out.flush()
                p = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            child[:] = [p]
            if stopping:  # Popen の最中に届いたシグナル
                p.send_signal(stopping[0])
            rc = p.wait()
            child.clear()
            code = exit_code(rc)
            if stopping:
                return code
            uptime = time.monotonic() - t0
            restart, quick = should_restart(rc, uptime, quick, healthy, max_quick)
            if not restart:
                # 設定が無い（EXIT_NO_CONFIG）は子が理由を出しているので、「異常終了」と重ねて書かない
                if code not in (0, EXIT_ALREADY_RUNNING, EXIT_NO_CONFIG):
                    log(f"supervisor: run exited abnormally rc={code} after {uptime / 3600:.2f} h; giving up after {quick} quick failures")
                return code
            log(f"supervisor: run exited abnormally rc={code} after {uptime / 3600:.2f} h; restarting in {delay:.0f} s (quick failures {quick}/{max_quick})")
            end = time.monotonic() + delay
            while True:
                if stopping:
                    return code
                if sd.flag("STOP"):
                    sd.clear_flag("STOP")
                    log("supervisor: STOP flag while waiting to restart: exit")
                    return 0
                left = end - time.monotonic()
                if left <= 0:
                    break
                time.sleep(min(1.0, left))
    finally:
        for s, h in old.items():
            signal.signal(s, h)
