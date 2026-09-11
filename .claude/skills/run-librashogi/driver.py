#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LibraShogi を動かすための駆動スクリプト（エージェント用）。リポジトリ root から実行する:

  .venv/bin/python .claude/skills/run-librashogi/driver.py engine [--model X.onnx] [--provider cpu|cuda|auto] [--go "nodes 200"] [--repl]
      C++ USI エンジン build/libra-engine/libra を子プロセスで起動し、usi → isready → 布石 2 局面 → 本将棋 1 局面に go を送って
      bestmove を合法性つきで表示する。--model 省略時は乱数の小さなネットを ONNX にして使う（学習済みモデル不要）。
      --repl で標準入力の USI 行をそのまま流し、応答を表示する（tmux から使う）。
  .venv/bin/python .claude/skills/run-librashogi/driver.py runner [--seconds 60] [--device cpu|cuda]
      一時ディレクトリに小さなネットで自己対局＋学習のランナー（bin/libra run と同じ Runner）を起こし、
      最初の対局が終わるまで待って status.json を表示し、STOP フラグで止める。学習パイプライン全体の煙テスト。
  .venv/bin/python .claude/skills/run-librashogi/driver.py sim
      Python バインディング librashogi の最小確認（合法手・perft・裁定）。

終了コードは 0 = 成功、1 = 失敗。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for sub in ("libra-sim/python", "libra-search/python", "libra-net", "libra-league", "libra-scale"):
    sys.path.insert(0, str(ROOT / sub))

import librashogi as ls  # noqa: E402

ENGINE_BIN = ROOT / "build" / "libra-engine" / "libra"


def tiny_onnx(path: Path) -> Path:
    """学習済みモデルが無くても動かすための、乱数初期化の小さなネット（d=32、2 層）。"""
    import torch

    from libra_net.export_onnx import export_model
    from libra_net.model import LibraNet, NetConfig

    torch.manual_seed(0)
    export_model(LibraNet(NetConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64)), path)
    return path


class Usi:
    """USI エンジンの子プロセス。送った行と受けた行を [>] [<] で標準エラーに写す。"""

    def __init__(self, cmd: list[str], env: dict | None = None):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        self.lines: list[str] = []
        self.lock = threading.Condition()
        self.cursor = 0  # wait が読み進めた位置（呼ぶたびに続きから探す）
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        assert self.p.stdout
        for line in self.p.stdout:
            line = line.rstrip("\r\n")
            print(f"[<] {line}", file=sys.stderr, flush=True)
            with self.lock:
                self.lines.append(line)
                self.lock.notify_all()
        with self.lock:
            self.lines.append(None)  # type: ignore[arg-type]
            self.lock.notify_all()

    def send(self, line: str) -> None:
        print(f"[>] {line}", file=sys.stderr, flush=True)
        assert self.p.stdin
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def wait(self, prefix: str, timeout: float = 120) -> str:
        end = time.time() + timeout
        with self.lock:
            while True:
                while self.cursor < len(self.lines):
                    l = self.lines[self.cursor]
                    self.cursor += 1
                    if l is None:
                        raise RuntimeError("engine exited")
                    if l.startswith(prefix):
                        return l
                remain = end - time.time()
                if remain <= 0:
                    raise TimeoutError(f"no '{prefix}' within {timeout}s")
                self.lock.wait(remain)

    def quit(self) -> None:
        try:
            self.send("quit")
            self.p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.p.kill()


def engine_env() -> dict:
    """CUDA EP が使う cuDNN などを PyTorch の pip 配布物から見せる（bin/libra-usi と同じ）。"""
    env = dict(os.environ)
    nv = [str(p) for p in ROOT.glob(".venv/lib/python3.*/site-packages/nvidia/*/lib")]
    env["LD_LIBRARY_PATH"] = ":".join(nv + [env.get("LD_LIBRARY_PATH", "")]).strip(":")
    return env


def cmd_engine(a: argparse.Namespace) -> int:
    if not ENGINE_BIN.exists():
        print(f"engine not built: {ENGINE_BIN} (see SKILL.md Build)", file=sys.stderr)
        return 1
    tmp = Path(tempfile.mkdtemp(prefix="libra-driver-"))
    model = Path(a.model) if a.model else tiny_onnx(tmp / "tiny.onnx")
    e = Usi([str(ENGINE_BIN)], engine_env())
    try:
        e.send("usi")
        e.wait("usiok")
        e.send(f"setoption name DNN_Model value {model}")
        e.send(f"setoption name DNN_Provider value {a.provider}")
        e.send("setoption name MultiPV value 2")
        e.send("isready")
        e.wait("readyok", timeout=600)
        if a.repl:
            for line in sys.stdin:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                e.send(line)
                if line.startswith("go"):
                    e.wait("bestmove")
                if line == "quit":
                    return 0
            return 0
        ok = True
        # 1) 布石の開始局面（置く側の 1 手目）、2) 両玉のあと（選ぶ側は winrate を見る）、3) 本将棋（平手 SFEN）
        for label, line in (
            ("fuseki ply0", "position fuseki"),
            ("fuseki ply2", "position fuseki moves K*5i K*5a"),
            ("normal", "position sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"),
        ):
            pos = ls.Position()
            pos.set_position(line)
            e.send(line)
            e.send(f"go {a.go}")
            t0 = time.time()
            bm = e.wait("bestmove", timeout=600).split()[1]
            legal = bm in ("win", "resign") or pos.is_legal(bm)
            ok &= legal
            print(f"{label:12s} bestmove {bm:6s} legal={legal} ({time.time() - t0:.2f}s)")
        # stop が効くこと
        e.send("position fuseki moves K*5i K*5a")
        e.send("go infinite")
        time.sleep(0.5)
        e.send("stop")
        t0 = time.time()
        e.wait("bestmove", timeout=30)
        print(f"go infinite + stop -> bestmove in {time.time() - t0:.2f}s")
        print("engine: OK" if ok else "engine: FAILED (illegal bestmove)")
        return 0 if ok else 1
    finally:
        e.quit()


def cmd_runner(a: argparse.Namespace) -> int:
    import torch

    from libra_league.config import load_config
    from libra_league.runner import Runner
    from libra_league.state import StateDir, read_json

    device = torch.device(a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(tempfile.mkdtemp(prefix="libra-run-")) / "smoke"
    cfg = load_config(None)
    cfg["run_id"] = "smoke"
    cfg["net"] = {"d_model": 32, "n_layers": 1, "n_heads": 4, "d_ff": 64, "dropout": 0.0}
    cfg["search"].update({"full_sims": 8, "fast_sims": 4, "proof_nodes": 0, "mate_nodes_root": 0, "max_moves_per_game": 60})
    cfg["selfplay"].update({"n_games": 8, "threads": 2, "infer_dtype": "float32" if device.type == "cpu" else "float16"})
    cfg["train"].update({"batch_size": 16, "min_window_games": 8, "train_every_games": 8, "window_games": 200})
    cfg["run"].update({"status_seconds": 1, "checkpoint_minutes": 100, "chunk_games": 8})
    sd = StateDir(root)
    sd.create()
    r = Runner(sd, cfg, device=device)
    th = threading.Thread(target=r.run, daemon=True)
    th.start()
    t0 = time.time()
    st: dict = {}
    while time.time() - t0 < a.seconds:
        st = read_json(sd.status_json, {}) or {}
        if st.get("games_total", 0) >= 8 and st.get("step", 0) >= 1:
            break
        time.sleep(0.5)
    sd.set_flag("STOP")
    th.join(timeout=120)
    st = read_json(sd.status_json, {}) or {}
    print(json.dumps({k: st.get(k) for k in ("step", "games_total", "games_per_day_1h", "train")}, ensure_ascii=False))
    print(f"state dir: {root}")
    ok = st.get("games_total", 0) >= 1 and (sd.checkpoints / "latest.pt").exists() and (sd.checkpoints / "latest.onnx").exists()
    print("runner: OK" if ok else "runner: FAILED")
    return 0 if ok else 1


def cmd_sim(_: argparse.Namespace) -> int:
    p = ls.Position()
    n0 = len(p.legal_moves())
    p.do_move("K*5i")
    p.do_move("K*5a")
    n2 = len(p.legal_moves())
    q = ls.Position()
    q.set_position("position sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
    perft3 = q.perft(3)
    print(f"fuseki legal moves: ply0 {n0} (36 king squares), ply2 {n2}; startpos perft(3) = {perft3}")
    ok = n0 == 36 and perft3 == 25470
    print("sim: OK" if ok else "sim: FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("engine")
    e.add_argument("--model", default=None)
    e.add_argument("--provider", default="cpu", choices=["cpu", "cuda", "dml", "auto"])
    e.add_argument("--go", default="nodes 200")
    e.add_argument("--repl", action="store_true")
    r = sub.add_parser("runner")
    r.add_argument("--seconds", type=int, default=120)
    r.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    sub.add_parser("sim")
    a = ap.parse_args()
    return {"engine": cmd_engine, "runner": cmd_runner, "sim": cmd_sim}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
