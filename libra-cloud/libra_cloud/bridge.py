# SPDX-License-Identifier: Apache-2.0
"""vast.ai の自己対局ワーカーと手元の学習側をつなぐ同期ループ（手元で動かす。プロジェクトの .venv）。

    PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-cloud \\
      .venv/bin/python -m libra_cloud.bridge --run-dir ~/libra-run/ls --host <ip> --port <port> --out <dir>

- 学習側が weights/latest.pt を書き換えたら（学習のたび。ls で約 69 秒ごと）ホストへ送る（.tmp に送ってから mv）。
- 布石（[selfplay] openings）が変わったらホストの <run>/openings.json へ送る。
- ホストの inbox/*.npz を取ってきてホストから消し、手元で手を再生して検査（workers.verify_games_file）してから
  学習側の inbox/ に置く（.tmp に書いてから os.replace）。不正なものは <out>/rejected/ に残す。
- 学習側の inbox/ が無い（[workers] enabled でない）間は取ってこない（ホストに溜まる）。
- 局を打った重みが古すぎる局は学習側（Inbox、max_lag_steps）が捨てる。ここでは見ない。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Callable, Protocol

from libra_league.config import load_config
from libra_league.workers import MAX_FILE_BYTES, GamesFileError, verify_games_file

NAME = re.compile(r"^[A-Za-z0-9_]{1,32}-\d{1,16}-\d{1,9}\.npz$")  # write_games_file の名前
KEEP_REJECTED = 20
KEY = Path.home() / ".ssh" / "id_ed25519_vast"


class Transport(Protocol):
    def list_inbox(self) -> list[str]: ...
    def fetch(self, names: list[str], dst: Path) -> None: ...
    def delete(self, names: list[str]) -> None: ...
    def push(self, src: Path, rel: str) -> None: ...
    def stop_worker(self, timeout: float) -> None: ...


class LocalTransport:
    """ホストの run ディレクトリの代わりに手元のディレクトリを使う（テスト用）。"""

    def __init__(self, remote_run: Path):
        self.root = remote_run

    def list_inbox(self) -> list[str]:
        d = self.root / "inbox"
        return sorted(p.name for p in d.glob("*.npz")) if d.is_dir() else []

    def fetch(self, names: list[str], dst: Path) -> None:
        for n in names:
            shutil.copyfile(self.root / "inbox" / n, dst / n)

    def delete(self, names: list[str]) -> None:
        for n in names:
            (self.root / "inbox" / n).unlink(missing_ok=True)

    def push(self, src: Path, rel: str) -> None:
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)

    def stop_worker(self, timeout: float) -> None:
        pass


class SSHTransport:
    def __init__(self, host: str, port: int, remote_run: str, key: Path = KEY, pid_file: str = "/root/out/worker.pid"):
        self.host, self.port, self.root, self.key, self.pid_file = host, int(port), remote_run.rstrip("/"), key, pid_file

    def _opts(self, flag: str) -> list[str]:
        return ["-i", str(self.key), flag, str(self.port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "BatchMode=yes"]

    def ssh(self, cmd: str, timeout: float = 60) -> bytes:
        p = subprocess.run(["ssh", *self._opts("-p"), f"root@{self.host}", cmd], capture_output=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"ssh rc={p.returncode}: {cmd[:60]}: {p.stderr.decode(errors='replace')[:200]}")
        return p.stdout

    def list_inbox(self) -> list[str]:
        out = self.ssh(f"cd {shlex.quote(self.root + '/inbox')} 2>/dev/null && ls -1 | grep '\\.npz$' || true")
        return sorted(out.decode(errors="replace").split())

    def fetch(self, names: list[str], dst: Path) -> None:
        """tar で 1 回の ssh にまとめて取る。中身の名前と大きさは信用しない（取りに行った名前の通常ファイルだけ書く）。"""
        want = set(names)
        raw = self.ssh(f"cd {shlex.quote(self.root + '/inbox')} && tar cf - -- {' '.join(shlex.quote(n) for n in names)}", timeout=600)
        with tarfile.open(fileobj=io.BytesIO(raw)) as t:
            for m in t.getmembers():
                if not (m.isfile() and m.name in want and NAME.match(m.name) and m.size <= MAX_FILE_BYTES):
                    raise RuntimeError(f"unexpected tar member {m.name!r} ({m.size} bytes)")
                f = t.extractfile(m)
                assert f is not None
                (dst / m.name).write_bytes(f.read())

    def delete(self, names: list[str]) -> None:
        self.ssh(f"cd {shlex.quote(self.root + '/inbox')} && rm -f -- {' '.join(shlex.quote(n) for n in names)}")

    def push(self, src: Path, rel: str) -> None:
        dst = f"{self.root}/{rel}"
        self.ssh(f"mkdir -p {shlex.quote(os.path.dirname(dst))}")
        subprocess.run(["scp", *self._opts("-P"), str(src), f"root@{self.host}:{dst}.tmp"], check=True, timeout=600,
                       capture_output=True)
        self.ssh(f"mv -f {shlex.quote(dst + '.tmp')} {shlex.quote(dst)}")

    def stop_worker(self, timeout: float) -> None:
        """ワーカーに SIGTERM を送り（残りの局を書いてから抜ける）、抜けるまで待つ。"""
        self.ssh(f"P=$(cat {shlex.quote(self.pid_file)} 2>/dev/null) && kill -TERM $P 2>/dev/null; "
                 f"for i in $(seq {int(timeout)}); do kill -0 $P 2>/dev/null || exit 0; sleep 1; done; kill -KILL $P 2>/dev/null; true",
                 timeout=timeout + 60)


class Bridge:
    def __init__(self, run_dir: Path, transport: Transport, out: Path, *, max_files: int = 20, log: Callable[[str], None] = print):
        self.cfg = load_config(run_dir / "config.toml")
        self.run_dir = run_dir
        self.t = transport
        self.out = out
        self.max_files = max_files
        self.log = log
        self.inbox = run_dir / "inbox"
        self.weights = run_dir / "weights" / "latest.pt"
        op = self.cfg["selfplay"].get("openings") or ""
        self.openings = Path(op).expanduser() if op else None
        self.staging = out / "staging"
        self.rejected = out / "rejected"
        self.staging.mkdir(parents=True, exist_ok=True)
        self.pushed: dict[str, int] = {}
        self.waiting_logged = False
        self.stats: dict = {"pushes": {}, "files": 0, "games": 0, "rejected_files": 0, "verify_s": 0.0, "verify_ms_per_game": None,
                            "last_pull": None, "last_push": None, "errors": 0}

    def push_if_changed(self, src: Path, rel: str) -> None:
        try:
            mt = src.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if self.pushed.get(rel) == mt:
            return
        self.t.push(src, rel)
        self.pushed[rel] = mt
        self.stats["pushes"][rel] = self.stats["pushes"].get(rel, 0) + 1
        self.stats["last_push"] = time.time()

    def pull(self) -> int:
        """ホストの対局ファイルを取ってきて検査し、学習側の inbox に置く。置いた局数を返す。"""
        if not self.inbox.is_dir():
            if not self.waiting_logged:
                self.log(f"bridge: {self.inbox} does not exist ([workers] enabled でない): not pulling")
                self.waiting_logged = True
            return 0
        self.waiting_logged = False
        names = [n for n in self.t.list_inbox() if NAME.match(n)][:self.max_files]
        if names:
            self.t.fetch(names, self.staging)
            self.t.delete(names)
            self.stats["last_pull"] = time.time()
        n = 0
        for p in sorted(self.staging.glob("*.npz")):
            n += self.accept(p)
        return n

    def accept(self, p: Path) -> int:
        t0 = time.perf_counter()
        try:
            meta, games = verify_games_file(p, self.cfg["search"])
        except GamesFileError as e:
            self.reject(p, str(e))
            return 0
        dt = time.perf_counter() - t0
        tmp = self.inbox / (p.name + ".tmp")
        shutil.copyfile(p, tmp)
        os.replace(tmp, self.inbox / p.name)
        p.unlink()
        s = self.stats
        s["files"] += 1
        s["games"] += len(games)
        s["verify_s"] += dt
        s["verify_ms_per_game"] = round(s["verify_s"] * 1000 / s["games"], 2)
        return len(games)

    def reject(self, p: Path, why: str) -> None:
        self.stats["rejected_files"] += 1
        self.log(f"bridge: rejected {p.name}: {why}")
        self.rejected.mkdir(exist_ok=True)
        os.replace(p, self.rejected / p.name)
        for old in sorted(self.rejected.iterdir(), key=lambda q: q.stat().st_mtime)[:-KEEP_REJECTED]:
            old.unlink(missing_ok=True)

    def cycle(self) -> int:
        self.push_if_changed(self.weights, "weights/latest.pt")
        if self.openings is not None:
            self.push_if_changed(self.openings, "openings.json")
        return self.pull()

    def write_status(self) -> None:
        tmp = self.out / "bridge.json.tmp"
        tmp.write_text(json.dumps({"time": time.time(), **self.stats}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.out / "bridge.json")


def serve(bridge: Bridge, *, interval: float, deadline: float, stop: Callable[[], bool], max_error_s: float,
          drain_timeout: float = 180.0) -> int:
    """deadline（time.time()）か stop() まで cycle を回す。終わるときはワーカーを止めて残りを取る。
    ssh が max_error_s 秒続けて失敗したら 3 を返す（ホストが落ちた）。"""
    first_error: float | None = None
    rc = 0
    while time.time() < deadline and not stop():
        t0 = time.time()
        try:
            n = bridge.cycle()
            if n:
                bridge.log(f"bridge: placed {n} games (total {bridge.stats['games']}, verify {bridge.stats['verify_ms_per_game']} ms/game)")
            first_error = None
        except (RuntimeError, OSError, subprocess.SubprocessError, tarfile.TarError) as e:
            bridge.stats["errors"] += 1
            first_error = first_error or t0
            bridge.log(f"bridge: error: {type(e).__name__}: {str(e)[:300]}")
            if t0 - first_error > max_error_s:
                bridge.log(f"bridge: errors for {max_error_s:.0f} s: giving up")
                rc = 3
                break
        bridge.write_status()
        time.sleep(max(0.0, interval - (time.time() - t0)))
    if rc == 0:
        bridge.log("bridge: stopping the worker and pulling the rest")
        try:
            bridge.t.stop_worker(drain_timeout)
            while bridge.cycle():
                pass
        except (RuntimeError, OSError, subprocess.SubprocessError, tarfile.TarError) as e:
            bridge.log(f"bridge: drain failed: {type(e).__name__}: {str(e)[:300]}")
    bridge.write_status()
    bridge.log(f"bridge: exit: {json.dumps(bridge.stats, ensure_ascii=False)}")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="libra_cloud.bridge")
    ap.add_argument("--run-dir", required=True, help="学習側の run（例: ~/libra-run/ls）。読むのは config.toml・weights/・布石、書くのは inbox/ だけ")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--remote-run", default=None, help="ホストの run ディレクトリ（既定: /root/libra/run/<run_id>）")
    ap.add_argument("--out", required=True, help="ログ・状態（bridge.json）・検査で弾いたファイルの置き場所")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--hours", type=float, default=3.0, help="この時間が過ぎたらワーカーを止めて残りを取って抜ける")
    ap.add_argument("--max-error-minutes", type=float, default=10.0)
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir).expanduser()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(run_dir / "config.toml")
    remote = a.remote_run or f"/root/libra/run/{cfg['run_id']}"
    logf = open(out / "bridge.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    stopping: list[int] = []
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda signum, frame: stopping.append(signum))
    bridge = Bridge(run_dir, SSHTransport(a.host, a.port, remote), out, log=log)
    log(f"bridge: {run_dir} <-> {a.host}:{a.port}:{remote} for {a.hours} h")
    return serve(bridge, interval=a.interval, deadline=time.time() + a.hours * 3600, stop=lambda: bool(stopping) or (out / "STOP").exists(),
                 max_error_s=a.max_error_minutes * 60)


if __name__ == "__main__":
    sys.exit(main())
