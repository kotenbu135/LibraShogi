# SPDX-License-Identifier: Apache-2.0
"""vast.ai の検証対局ワーカー（libra-scale seq worker）と手元の seq の run をつなぐ同期ループ（手元の .venv で動かす）。

    PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-scale:libra-cloud \\
      .venv/bin/python -m libra_cloud.scale_bridge --scale-dir ~/libra-run/ls/scale/seq-v0.1 --host <ip> --port <port> --out <dir>

- 手元の <scale-dir>/active.json（打ち切っていない組）が変わったらホストへ送る（.tmp に送ってから mv）。
  全部の組が止まったら（done）ワーカーを止めて残りを取って抜ける。
- ホストの inbox/*.jsonl.gz を取ってきてホストから消し、全局を玉の配置から再生して検査（seqrun.check_record）してから
  手元の inbox/ に置く（知っている欄だけを書き直し、worker はファイル名の先頭にする）。1 局でも合わなければファイルごと <out>/rejected/。
  数えるのは手元の seq run（Coordinator.ingest）。探索の中身（読みの回数など）の改ざんは再生からは確かめられない。
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shlex
import shutil
import signal
import sys
import tarfile
import time
from pathlib import Path
from typing import Callable

from libra_scale.seqrun import check_record, read_chunk, read_json

from .bridge import ERRORS, KEEP_REJECTED, LocalTransport, SSHTransport, serve

NAME = re.compile(r"^([A-Za-z0-9_]{1,32})-\d{1,16}-\d{1,9}\.jsonl\.gz$")  # seqrun.write_chunk の名前
MAX_CHUNK_BYTES = 4 * 2**20
FIELDS = ("kb", "kw", "result", "reason", "plies", "moves")


class ScaleLocalTransport(LocalTransport):
    def __init__(self, remote_run: Path, alive: Callable[[], bool] = lambda: True):
        super().__init__(remote_run)
        self.alive = alive

    def list_inbox(self) -> list[str]:
        d = self.root / "inbox"
        return sorted(p.name for p in d.glob("*.jsonl.gz")) if d.is_dir() else []


class ScaleSSHTransport(SSHTransport):
    def alive(self) -> bool:
        """ホストのワーカーのプロセスが生きているか。"""
        out = self.ssh(f"P=$(cat {shlex.quote(self.pid_file)} 2>/dev/null) && kill -0 $P 2>/dev/null && echo alive || echo dead")
        return out.decode(errors="replace").strip() == "alive"

    def list_inbox(self) -> list[str]:
        out = self.ssh(f"cd {shlex.quote(self.root + '/inbox')} 2>/dev/null && ls -1 | grep '\\.jsonl\\.gz$' || true")
        return sorted(out.decode(errors="replace").split())

    def fetch(self, names: list[str], dst: Path) -> None:
        want = set(names)
        raw = self.ssh(f"cd {shlex.quote(self.root + '/inbox')} && tar cf - -- {' '.join(shlex.quote(n) for n in names)}", timeout=600)
        with tarfile.open(fileobj=io.BytesIO(raw)) as t:
            for m in t.getmembers():
                if not (m.isfile() and m.name in want and NAME.match(m.name) and m.size <= MAX_CHUNK_BYTES):
                    raise RuntimeError(f"unexpected tar member {m.name!r} ({m.size} bytes)")
                f = t.extractfile(m)
                assert f is not None
                (dst / m.name).write_bytes(f.read())


class ScaleBridge:
    def __init__(self, scale_dir: Path, transport, out: Path, *, max_files: int = 20, log: Callable[[str], None] = print,
                 alive_check_s: float = 300.0, clock: Callable[[], float] = time.monotonic):
        self.alive_check_s = alive_check_s  # ワーカーが落ちたまま課金が続かないよう、この間隔で生存を確かめ、2 回続けて死んでいたら抜ける
        self.clock = clock
        self.last_alive_check = clock()
        self.dead_checks = 0
        self.dead = False
        self.d = Path(scale_dir)
        self.search = read_json(self.d / "config.json")["search"]
        self.t = transport
        self.out = Path(out)
        self.max_files = max_files
        self.log = log
        self.staging = self.out / "staging"
        self.rejected = self.out / "rejected"
        self.staging.mkdir(parents=True, exist_ok=True)
        self.pushed_mt: int | None = None
        self.stats: dict = {"pushes": {}, "files": 0, "games": 0, "rejected_files": 0, "verify_s": 0.0, "verify_ms_per_game": None,
                            "last_pull": None, "last_push": None, "errors": 0, "push_bytes": 0, "pull_bytes": 0}

    def done(self) -> bool:
        try:
            return bool(read_json(self.d / "active.json").get("done"))
        except (OSError, ValueError):
            return False

    def push_active(self) -> None:
        src = self.d / "active.json"
        mt = src.stat().st_mtime_ns
        if mt == self.pushed_mt:
            return
        self.t.push(src, "active.json")
        self.pushed_mt = mt
        self.stats["pushes"]["active.json"] = self.stats["pushes"].get("active.json", 0) + 1
        self.stats["push_bytes"] += src.stat().st_size
        self.stats["last_push"] = time.time()

    def pull(self) -> int:
        names = [n for n in self.t.list_inbox() if NAME.match(n)][:self.max_files]
        if names:
            self.t.fetch(names, self.staging)
            self.stats["pull_bytes"] += sum((self.staging / n).stat().st_size for n in names if (self.staging / n).exists())
            self.t.delete(names)
            self.stats["last_pull"] = time.time()
        return sum(self.accept(p) for p in sorted(self.staging.glob("*.jsonl.gz")))

    def accept(self, p: Path) -> int:
        t0 = time.perf_counter()
        worker = NAME.match(p.name).group(1)
        try:
            recs = read_chunk(p)
            clean = []
            for i, r in enumerate(recs):
                c = {k: r[k] for k in FIELDS}
                c["worker"] = worker
                try:
                    check_record(c, self.search)
                except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                    raise ValueError(f"game {i}: {type(e).__name__}: {str(e)[:200]}") from None
                clean.append(c)
        except (OSError, EOFError, ValueError, KeyError, TypeError, AttributeError) as e:
            self.reject(p, f"{type(e).__name__}: {str(e)[:300]}")
            return 0
        dt = time.perf_counter() - t0
        tmp = self.d / "inbox" / (p.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for c in clean:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        os.replace(tmp, self.d / "inbox" / p.name)
        p.unlink()
        s = self.stats
        s["files"] += 1
        s["games"] += len(clean)
        s["verify_s"] += dt
        s["verify_ms_per_game"] = round(s["verify_s"] * 1000 / max(s["games"], 1), 2)
        return len(clean)

    def reject(self, p: Path, why: str) -> None:
        self.stats["rejected_files"] += 1
        self.log(f"bridge: rejected {p.name}: {why}")
        self.rejected.mkdir(exist_ok=True)
        os.replace(p, self.rejected / p.name)
        for old in sorted(self.rejected.iterdir(), key=lambda q: q.stat().st_mtime)[:-KEEP_REJECTED]:
            old.unlink(missing_ok=True)
        if self.stats["files"] == 0 and self.stats["rejected_files"] >= 3:  # 1 つも通らないまま課金を続けない
            self.log("bridge: every file so far was rejected; giving up")
            self.dead = True

    def cycle(self) -> int:
        try:
            self.push_active()
        except ERRORS as e:
            self.stats["errors"] += 1
            self.log(f"bridge: push active.json failed: {type(e).__name__}: {str(e)[:300]}")
        n = self.pull()
        if self.clock() - self.last_alive_check >= self.alive_check_s:
            self.last_alive_check = self.clock()
            if self.t.alive():
                self.dead_checks = 0
            else:
                self.dead_checks += 1
                self.log(f"bridge: worker process is not running ({self.dead_checks})")
                self.dead = self.dead_checks >= 2
        return n

    def write_status(self) -> None:
        tmp = self.out / "bridge.json.tmp"
        tmp.write_text(json.dumps({"time": time.time(), **self.stats}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.out / "bridge.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="libra_cloud.scale_bridge")
    ap.add_argument("--scale-dir", required=True, help="手元の seq の run（読むのは config.json・active.json、書くのは inbox/ だけ）")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--remote-dir", default=None, help="ホストの seq のディレクトリ（既定: /root/libra/scale/<scale-dir の名前>）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--max-error-minutes", type=float, default=10.0)
    a = ap.parse_args(argv)
    d = Path(a.scale_dir).expanduser()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    remote = a.remote_dir or f"/root/libra/scale/{d.name}"
    logf = open(out / "bridge.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    stopping: list[int] = []
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda signum, frame: stopping.append(signum))
    bridge = ScaleBridge(d, ScaleSSHTransport(a.host, a.port, remote), out, log=log)
    log(f"bridge: {d} <-> {a.host}:{a.port}:{remote} for {a.hours} h")
    return serve(bridge, interval=a.interval, deadline=time.time() + a.hours * 3600,
                 stop=lambda: bool(stopping) or (out / "STOP").exists() or bridge.done() or bridge.dead, max_error_s=a.max_error_minutes * 60)


if __name__ == "__main__":
    sys.exit(main())
