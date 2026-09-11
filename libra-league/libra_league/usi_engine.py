# SPDX-License-Identifier: Apache-2.0
"""Libra の USI 拡張エンジン（Python 版。docs/protocol.md）。C++ の探索（librasearch、外部駆動）＋ PyTorch 推論。

  python -m libra_league.usi_engine            # 標準入出力で USI

公開版の libra.exe（ONNX Runtime）ができるまでの暫定。desktop には
`wsl.exe -d Ubuntu-24.04 -- /home/sakis/LibraShogi/bin/libra-usi` を登録して使える。
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import librasearch
import librashogi as ls
from libra_net.model import LibraNet, NetConfig

from .config import DEFAULTS
from .state import DEFAULT_ROOT

VERSION = "0.0.1"
DEFAULT_MODEL = str(DEFAULT_ROOT / "ls" / "checkpoints" / "latest.pt")


class Engine:
    def __init__(self) -> None:
        self.opts = {
            "Fuseki_Mode": "tenbin",
            "MultiPV": 1,
            "Threads": 4,
            "DNN_Model": DEFAULT_MODEL,
            "DNN_Batch_Size": 1,
            "Sims_Fuseki": 400,
            "Sims_Normal": 800,
            "Scale_Table": "",
            "USI_Ponder": False,
            "Declare_Win": False,  # GUI は本将棋の `bestmove win` を投了として扱う（docs/protocol.md §4）
            "Mate_Nodes": 2000,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: LibraNet | None = None
        self.eng: librasearch.SelfPlay | None = None
        self.position_line = "position fuseki"
        self.pos = ls.Position()
        self.stop_flag = threading.Event()
        self.inbox: queue.Queue[str] = queue.Queue()

    # ---- 出力 ----
    @staticmethod
    def out(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def declare_options(self) -> None:
        o = self.opts
        self.out("option name Fuseki_Mode type combo default tenbin var tenbin var fuseki")
        self.out("option name MultiPV type spin default 1 min 1 max 300")
        self.out("option name Threads type spin default 4 min 1 max 64")
        self.out(f"option name DNN_Model type string default {o['DNN_Model']}")
        self.out("option name DNN_Batch_Size type spin default 1 min 1 max 1024")
        self.out("option name Sims_Fuseki type spin default 400 min 1 max 1000000")
        self.out("option name Sims_Normal type spin default 800 min 1 max 1000000")
        self.out("option name Scale_Table type string default <empty>")
        self.out("option name USI_Ponder type check default false")
        self.out("option name Declare_Win type check default false")
        self.out("option name Mate_Nodes type spin default 2000 min 0 max 10000000")

    # ---- 準備 ----
    def ready(self) -> None:
        if self.model is None:
            path = Path(self.opts["DNN_Model"])
            sd = torch.load(path, map_location=self.device, weights_only=False)
            cfg = sd.get("config", {}).get("net", {})
            m = LibraNet(NetConfig.from_dict(cfg)).to(self.device)
            m.load_state_dict(sd["model"])
            m.eval()
            if self.device.type == "cuda":
                m = m.half()
            for p in m.parameters():
                p.requires_grad_(False)
            self.model = m
            self.out(f"info string model {path.name} step {sd.get('step', '?')} params {m.n_params() / 1e6:.1f}M device {self.device}")
        if self.eng is None:
            scfg = dict(DEFAULTS["search"])
            scfg.update({"external": True, "mate_nodes_root": int(self.opts["Mate_Nodes"]), "policy_topk": 300})
            self.eng = librasearch.SelfPlay(scfg, 1, int(time.time()) & 0xFFFF, int(self.opts["Threads"]))
            self.sq = np.zeros((1, 81, ls.SQ_FEATS), np.float32)
            self.glob = np.zeros((1, ls.GLOB_FEATS), np.float32)

    # ---- 探索 ----
    @torch.no_grad()
    def evaluate(self) -> None:
        assert self.eng is not None and self.model is not None
        self.eng.collect(self.sq, self.glob)
        dt = torch.float16 if self.device.type == "cuda" else torch.float32
        sq = torch.from_numpy(self.sq).to(self.device).to(dt)
        gl = torch.from_numpy(self.glob).to(self.device).to(dt)
        p, w, _ = self.model(sq, gl)
        self.eng.apply(np.ascontiguousarray(p.float().cpu().numpy()), np.ascontiguousarray(F.softmax(w.float(), dim=-1).cpu().numpy()))

    def info_lines(self, res: dict, elapsed: float, phase: str, ply: int, method: str) -> None:
        from librashogi.usi import winrate_to_cp

        k = max(1, int(self.opts["MultiPV"]))
        cands = res["cands"][:k]
        for i, c in enumerate(cands):
            p = (c["q"] + 1) / 2
            pv = " ".join(res["pv"]) if i == 0 else c["move"]
            self.out(f"info depth 1 seldepth {len(res['pv'])} multipv {i + 1} score cp {winrate_to_cp(p)} winrate {p:.4f} "
                     f"nodes {res['sims']} time {int(elapsed * 1000)} pv {pv}")
        self.out(f"info string phase {phase} ply {ply} method {method}")

    def go(self, args: list[str]) -> None:
        self.ready()
        assert self.eng is not None
        pos = ls.Position()
        try:
            pos.set_position(self.position_line, self.opts["Fuseki_Mode"])
        except ValueError as e:
            self.out(f"info string bad position: {e}")
            self.out("bestmove resign")
            return
        phase, ply = pos.phase, pos.ply
        turn = pos.turn
        result, reason = pos.outcome()
        # 40 手完了時の裁定: 手番（先手）が後手玉を取れる
        if result != "ongoing":
            if reason == "ruling41" and turn == "sente":
                self.out("info string phase normal ply 40 method ruling41")
                self.out("bestmove win")
            else:
                self.out(f"info string game over: {result} {reason}")
                self.out("bestmove resign")
            return
        if phase == "normal" and self.opts["Declare_Win"] and pos.can_declare(turn):
            self.out(f"info string phase normal ply {ply} method declaration")
            self.out("bestmove win")
            return
        if not pos.legal_moves():
            self.out("bestmove resign")
            return
        # 思考量
        sims = int(self.opts["Sims_Fuseki"] if phase == "fuseki" else self.opts["Sims_Normal"])
        infinite = "infinite" in args
        deadline = None
        if "nodes" in args:
            sims = int(args[args.index("nodes") + 1])
        elif "movetime" in args:
            deadline = time.time() + int(args[args.index("movetime") + 1]) / 1000.0
        elif "btime" in args or "wtime" in args:
            key = "btime" if turn == "sente" else "wtime"
            remain = int(args[args.index(key) + 1]) if key in args else 0
            byo = int(args[args.index("byoyomi") + 1]) if "byoyomi" in args else 0
            inc = 0
            ik = "binc" if turn == "sente" else "winc"
            if ik in args:
                inc = int(args[args.index(ik) + 1])
            budget_ms = byo + inc + remain / 30.0
            deadline = time.time() + max(0.05, budget_ms / 1000.0 * 0.9)
        if infinite:
            sims = 10**9
        elif deadline is not None:
            sims = 10**9  # 時間で止める
        self.stop_flag.clear()
        t0 = time.time()
        ok = self.eng.set_position(0, self.position_line, sims, True)
        if not ok:
            self.out("bestmove resign")
            return
        last_info = 0.0
        while not self.eng.idle(0):
            if self.stop_flag.is_set() or (deadline is not None and time.time() >= deadline):
                self.eng.finish_now(0)
                if not self.eng.idle(0):
                    self.evaluate()
                    self.eng.finish_now(0)
                break
            self.evaluate()
            now = time.time()
            if infinite and now - last_info > 1.0:
                last_info = now
        while not self.eng.idle(0):
            self.evaluate()
        res = self.eng.result(0)
        if infinite:
            # stop まで待つ（結果は出ている）
            while not self.stop_flag.is_set():
                time.sleep(0.02)
        if not res["ready"] or res["best"] == "none":
            self.out("bestmove resign")
            return
        method = "mcgs" if phase == "fuseki" else "mcts"
        if len(res["cands"]) == 1 and res["sims"] == 0:
            method = "proof"
        self.info_lines(res, time.time() - t0, phase, ply, method)
        self.out(f"bestmove {res['best']}")

    # ---- 主ループ ----
    def reader(self) -> None:
        for line in sys.stdin:
            line = line.rstrip("\r\n")
            if line == "stop":
                self.stop_flag.set()
            self.inbox.put(line)
        self.inbox.put("quit")

    def run(self) -> None:
        threading.Thread(target=self.reader, daemon=True).start()
        while True:
            line = self.inbox.get()
            t = line.split()
            if not t:
                continue
            cmd = t[0]
            if cmd == "usi":
                self.out(f"id name LibraShogi {VERSION}")
                self.out("id author kotenbu")
                self.declare_options()
                self.out("usiok")
            elif cmd == "isready":
                try:
                    self.ready()
                except Exception as e:  # noqa: BLE001
                    self.out(f"info string failed to load model: {e}")
                self.out("readyok")
            elif cmd == "setoption":
                if "name" in t and "value" in t:
                    name = t[t.index("name") + 1]
                    value = " ".join(t[t.index("value") + 1 :])
                    if name in self.opts:
                        cur = self.opts[name]
                        if isinstance(cur, bool):
                            self.opts[name] = value.lower() == "true"
                        elif isinstance(cur, int):
                            try:
                                self.opts[name] = int(value)
                            except ValueError:
                                pass
                        else:
                            self.opts[name] = value
                        if name in ("Threads", "Mate_Nodes"):
                            self.eng = None
                        if name == "DNN_Model":
                            self.model = None
            elif cmd == "usinewgame":
                pass
            elif cmd == "position":
                self.position_line = line
            elif cmd == "go":
                self.go(t[1:])
            elif cmd == "stop":
                pass
            elif cmd == "gameover":
                pass
            elif cmd == "quit":
                return


def main() -> None:
    Engine().run()


if __name__ == "__main__":
    main()
