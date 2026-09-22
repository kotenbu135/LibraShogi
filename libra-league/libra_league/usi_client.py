# SPDX-License-Identifier: Apache-2.0
"""USI エンジンを子プロセスとして駆動する（ハーネス用）。"""
from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path


class UsiEngine:
    def __init__(self, name: str, cmd: list[str], cwd: str | None = None, options: dict | None = None, log=None):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.options = options or {}
        self.log = log
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue[str | None] = queue.Queue()
        self.id_name = ""
        self.declared: dict[str, str] = {}
        # 相手の標準エラーの末尾。2026-09-21 まで捨てていたので、相手が起動できずに落ちた理由がどこにも
        # 残らなかった（外部計測が 3 回続けて 1 局も記録せず、原因はホストでしか分からなかった）
        self.stderr_tail: list[str] = []

    def _reader(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.rstrip("\r\n")
            if self.log:
                self.log(f"[{self.name}] < {line}")
            self.q.put(line)
        self.q.put(None)

    def _stderr_reader(self) -> None:
        """標準エラーは USI の行ではないので待ち行列に混ぜず、末尾だけ持っておく（落ちた理由になる）。"""
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            if self.log:
                self.log(f"[{self.name}] ! {line}")
            self.stderr_tail = (self.stderr_tail + [line[:200]])[-10:]

    def send(self, line: str) -> None:
        assert self.proc and self.proc.stdin
        if self.log:
            self.log(f"[{self.name}] > {line}")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _why(self) -> str:
        """落ちた理由として添える標準エラーの末尾（`cmd` は呼ぶ側が持っているので出さない）。"""
        time.sleep(0.05)  # 読み取りの糸が最後の行を拾うのを待つ
        return (" | stderr: " + " / ".join(self.stderr_tail)) if self.stderr_tail else ""

    def wait_for(self, pred, timeout: float) -> list[str]:
        """pred(line) が真になるまでの行を返す（最後の行を含む）。時間切れは TimeoutError。"""
        lines: list[str] = []
        end = time.time() + timeout
        while True:
            remain = end - time.time()
            if remain <= 0:
                raise TimeoutError(f"{self.name}: no response within {timeout}s (last: {lines[-3:]})" + self._why())
            try:
                line = self.q.get(timeout=remain)
            except queue.Empty:
                continue
            if line is None:
                # 標準出力の末尾も添える。やねうら王は起動の失敗を標準出力に書いて終わるので、
                # stderr だけだと「process exited」しか残らない（2026-09-22 の外部計測で実際にそうなった）
                raise RuntimeError(f"{self.name}: process exited (last: {lines[-3:]})" + self._why())
            lines.append(line)
            if pred(line):
                return lines

    def start(self, ready_timeout: float = 1200.0, usi_timeout: float = 60.0) -> None:
        self.proc = subprocess.Popen(self.cmd, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._stderr_reader, daemon=True).start()
        self.send("usi")
        for line in self.wait_for(lambda l: l == "usiok", usi_timeout):
            t = line.split()
            if line.startswith("id name "):
                self.id_name = line[len("id name "):]
            if line.startswith("option name ") and "type" in t:
                self.declared[t[2]] = line
        for k, v in self.options.items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait_for(lambda l: l == "readyok", ready_timeout)

    def new_game(self) -> None:
        self.send("usinewgame")

    def go(self, position_line: str, go_args: str, timeout: float = 600.0) -> tuple[str, dict]:
        """bestmove と、最後に受けた multipv 1 の info（winrate/cp/pv）を返す。"""
        self.send(position_line)
        self.send(f"go {go_args}".strip())
        info: dict = {}
        lines = self.wait_for(lambda l: l.startswith("bestmove"), timeout)
        for line in lines:
            if not line.startswith("info "):
                continue
            t = line.split()
            if "string" in t:
                continue
            if "multipv" in t and t[t.index("multipv") + 1] != "1":
                continue
            d: dict = {}
            if "winrate" in t:
                d["winrate"] = float(t[t.index("winrate") + 1])
            if "score" in t:
                i = t.index("score")
                if t[i + 1] == "cp":
                    d["cp"] = int(t[i + 2])
                elif t[i + 1] == "mate":
                    d["mate"] = t[i + 2]
            if "pv" in t:
                d["pv"] = t[t.index("pv") + 1 :]
            if d:
                info = d
        bm = lines[-1].split()
        return (bm[1] if len(bm) > 1 else "resign"), info

    def quit(self) -> None:
        if not self.proc:
            return
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()
