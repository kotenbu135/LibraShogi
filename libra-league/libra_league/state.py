# SPDX-License-Identifier: Apache-2.0
"""状態ディレクトリ（docs/libra-local.md §7.1）。すべての状態は 1 ディレクトリに置き、書き込みは原子的に行う。"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path.home() / "libra-run"


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj: Any) -> None:
    write_atomic(path, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


class StateDir:
    FLAGS = ("PAUSE", "STOP", "THROTTLE", "EVAL_NOW", "MATCH_NOW")

    def __init__(self, root: Path):
        self.root = root
        self.checkpoints = root / "checkpoints"
        self.replay = root / "replay"
        self.games = root / "games"
        self.state_json = root / "state.json"
        self.status_json = root / "status.json"
        self.config_toml = root / "config.toml"
        self.log = root / "log.txt"

    def create(self) -> None:
        for d in (self.root, self.checkpoints, self.replay, self.games):
            d.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.state_json.exists()

    def flag(self, name: str) -> bool:
        return (self.root / name).exists()

    def set_flag(self, name: str, text: str = "") -> None:
        write_atomic(self.root / name, (text or str(time.time())).encode())

    def clear_flag(self, name: str) -> None:
        try:
            (self.root / name).unlink()
        except FileNotFoundError:
            pass

    def throttle_value(self) -> int | None:
        p = self.root / "THROTTLE"
        if not p.exists():
            return None
        try:
            return int(p.read_text().strip())
        except ValueError:
            return None

    def read_state(self) -> dict:
        return read_json(self.state_json, {})

    def write_state(self, st: dict) -> None:
        write_json_atomic(self.state_json, st)

    def append_log(self, line: str) -> None:
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
