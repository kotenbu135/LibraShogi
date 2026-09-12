# SPDX-License-Identifier: Apache-2.0
"""`libra run`: 自己対局と学習を 1 プロセスで時分割し、10 分ごとにチェックポイントを原子的に書く（docs/libra-local.md §7）。"""
from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from pathlib import Path

import numpy as np
import torch

from libra_net.model import LibraNet, NetConfig

from .auto import AutoJobs, append_metrics
from .config import dump_toml, load_config
from .replay import ReplayBuffer
from .selfplay import SelfPlayLoop
from .state import StateDir, write_json_atomic
from .trainer import Trainer


class Runner:
    def __init__(self, sd: StateDir, cfg: dict, device: torch.device | None = None):
        self.sd = sd
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LibraNet(NetConfig.from_dict(cfg["net"])).to(self.device)
        self.trainer = Trainer(self.model, cfg["train"], self.device)
        tr, sr, rr = cfg["train"], cfg["search"], cfg["run"]
        self.replay = ReplayBuffer(sd.replay, sd.games, tr["window_games"], rr["chunk_games"], sr["max_ply"], sr["count_from_41"])
        self.state = {
            "run_id": cfg["run_id"], "created": time.time(), "generation": 0, "step": 0, "games_total": 0, "moves_total": 0,
            "chunk_index": 0, "seed": cfg["seed"], "restarts": [], "last_checkpoint": None, "elapsed": 0.0,
        }
        self.rng = np.random.default_rng(cfg["seed"])
        self.loop: SelfPlayLoop | None = None
        self.started = time.time()
        self.session_games = 0
        self.session_elapsed_offset = 0.0
        self.paused = False
        self.rate_hist: list[tuple[float, int]] = []
        self.last_train: dict = {}
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.exploiter_stats = {"games": 0, "wins": 0, "draws": 0, "losses": 0}
        self.openings_mtime: float | None = None
        self.openings_checked = 0.0
        self.auto = AutoJobs(sd, cfg, self.state, self.log)
        self.last_metrics = 0.0

    # ---- 永続化 ----
    def log(self, msg: str) -> None:
        print(msg, flush=True)
        self.sd.append_log(msg)

    def load(self) -> None:
        st = self.sd.read_state()
        if st:
            self.state.update(st)
        ck = self.sd.checkpoints / "latest.pt"
        if ck.exists():
            sd = torch.load(ck, map_location=self.device, weights_only=False)
            self.trainer.load_state_dict(sd)
            self.state["step"] = self.trainer.step_count
            if "rng" in sd:
                self.rng.bit_generator.state = sd["rng"]
        self.replay.load(self.state["chunk_index"], self.state["games_total"])
        # 索引に無いチャンク（書きかけ）は捨てる。書き終えたチャンクだけを索引に載せる設計
        for p in self.sd.replay.glob("chunk_*.tmp"):
            p.unlink()
        self.state["restarts"] = (self.state.get("restarts") or [])[-20:] + [time.strftime("%Y-%m-%d %H:%M:%S")]
        self.session_elapsed_offset = float(self.state.get("elapsed", 0.0))
        if self.state.get("exploiter_stats"):
            self.exploiter_stats.update(self.state["exploiter_stats"])
        self.log(f"resume: step={self.state['step']} games={self.state['games_total']} window={self.replay.n_games()} chunks={self.state['chunk_index']}")

    def checkpoint(self) -> None:
        t0 = time.time()
        sd = self.trainer.state_dict()
        sd["rng"] = self.rng.bit_generator.state
        sd["config"] = self.cfg
        sd["state"] = self.state
        step = self.trainer.step_count
        path = self.sd.checkpoints / f"ckpt_{step:09d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(sd, tmp)
        os.replace(tmp, path)
        latest = self.sd.checkpoints / "latest.pt"
        tmp2 = latest.with_suffix(".tmp")
        shutil.copyfile(path, tmp2)
        os.replace(tmp2, latest)
        # 古いものを消す（長期保管は auto が checkpoints/archive/ に写す）
        keep = self.cfg["run"]["keep_checkpoints"]
        cks = sorted(self.sd.checkpoints.glob("ckpt_*.pt"))
        for p in cks[:-keep]:
            p.unlink()
        self.state["step"] = step
        self.state["chunk_index"] = self.replay.chunk_index
        self.state["games_total"] = self.replay.total_games
        self.state["generation"] = self.state.get("generation", 0) + 1
        self.state["last_checkpoint"] = time.time()
        self.state["elapsed"] = self.elapsed()
        if self.exploiter_stats["games"]:
            self.state["exploiter_stats"] = dict(self.exploiter_stats)
        self.sd.write_state(self.state)
        self.log(f"checkpoint step={step} games={self.state['games_total']} ({time.time() - t0:.1f}s)")
        if self.cfg["run"].get("export_onnx", False):
            self.export_onnx(latest)
        self.auto.on_checkpoint(path)
        self.auto.poll()
        self.sd.write_state(self.state)

    def export_onnx(self, ckpt: Path) -> None:
        """latest.pt → latest.onnx（原子的に置き換え）。失敗してもランは止めない。"""
        try:
            import copy

            from libra_net.export_onnx import export_model

            t0 = time.time()
            m = copy.deepcopy(self.model).float().cpu().eval()
            out = ckpt.with_suffix(".onnx")
            tmp = out.with_suffix(".onnx.tmp")
            export_model(m, tmp, {"libra_step": str(self.trainer.step_count), "libra_net": self.cfg["net"], "libra_source": ckpt.name, "license": "Apache-2.0"})
            os.replace(tmp, out)
            self.log(f"export: {out.name} ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            self.log(f"export failed: {e}")

    def elapsed(self) -> float:
        return self.session_elapsed_offset + (time.time() - self.started)

    def load_opponent(self) -> None:
        """搾取者モード: 凍結した本体を読む（cfg.exploiter.main_ckpt）。"""
        ex = self.cfg.get("exploiter", {})
        path = ex.get("main_ckpt") or ""
        if not path:
            return
        sd = torch.load(Path(path).expanduser(), map_location=self.device, weights_only=False)
        m = LibraNet(NetConfig.from_dict(sd.get("config", {}).get("net", {}))).to(self.device)
        m.load_state_dict(sd["model"])
        assert self.loop is not None
        self.loop.set_opponent(m)
        self.log(f"exploiter: opponent {path} step {sd.get('step', '?')} params {m.n_params()/1e6:.1f}M (even slots: exploiter sente)")

    def reload_openings(self, force: bool = False) -> None:
        """cfg.selfplay.openings（openings.json）が更新されていればエンジンに渡す。"""
        sp = self.cfg["selfplay"]
        path = sp.get("openings") or ""
        if not path or self.loop is None:
            return
        now = time.time()
        if not force and now - self.openings_checked < sp.get("openings_reload_seconds", 600):
            return
        self.openings_checked = now
        p = Path(path).expanduser()
        if not p.exists():
            return
        mt = p.stat().st_mtime
        if mt == self.openings_mtime:
            return
        from .openings import load_openings

        ops = load_openings(p)
        self.loop.engine.set_openings(ops, float(sp.get("openings_prob", 0.0)))
        self.openings_mtime = mt
        self.log(f"openings: {len(ops)} lines from {p} (prob {sp.get('openings_prob', 0.0)})")

    def write_status(self) -> None:
        now = time.time()
        self.rate_hist.append((now, self.replay.total_games))
        self.rate_hist = [(t, g) for (t, g) in self.rate_hist if now - t <= 3600]
        rate = 0.0
        if len(self.rate_hist) >= 2:
            (t0, g0), (t1, g1) = self.rate_hist[0], self.rate_hist[-1]
            rate = (g1 - g0) / max(1e-6, t1 - t0) * 86400
        st = self.loop.stats() if self.loop else {}
        gpu = None
        if self.device.type == "cuda":
            gpu = {"mem_alloc_mb": torch.cuda.memory_allocated() // 2**20, "mem_reserved_mb": torch.cuda.memory_reserved() // 2**20}
        status = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "paused": self.paused,
            "active_games": self.loop.engine.active if self.loop else 0,
            "step": self.trainer.step_count,
            "generation": self.state.get("generation", 0),
            "games_total": self.replay.total_games,
            "games_session": self.session_games,
            "games_per_day_1h": round(rate),
            "window_games": self.replay.n_games(),
            "elapsed_h": round(self.elapsed() / 3600, 2),
            "engine": st,
            "train": self.last_train,
            "gpu": gpu,
            "restarts": self.state.get("restarts", [])[-5:],
        }
        if self.loop and self.loop.opponent is not None:
            es = dict(self.exploiter_stats)
            es["winrate"] = round((es["wins"] + 0.5 * es["draws"]) / max(1, es["games"]), 4)
            status["exploiter"] = es
        write_json_atomic(self.sd.status_json, status)
        if now - self.last_metrics >= float(self.cfg["run"].get("metrics_minutes", 5)) * 60 and self.loop is not None and not self.paused:
            append_metrics(self.sd, status)
            self.last_metrics = now

    # ---- メインループ ----
    def run(self) -> None:
        self.sd.create()
        if not self.sd.config_toml.exists():
            self.sd.config_toml.write_text(dump_toml(self.cfg), encoding="utf-8")
        self.load()
        self.sd.write_state(self.state)
        sp = self.cfg["selfplay"]
        self.loop = SelfPlayLoop(self.cfg["search"], sp["n_games"], sp["threads"], int(self.rng.integers(0, 2**63)), self.device, sp["infer_dtype"])
        self.loop.set_model(self.model)
        self.load_opponent()
        self.reload_openings(force=True)
        tr, rr = self.cfg["train"], self.cfg["run"]
        last_ck = time.time()
        last_status = 0.0
        new_games = 0
        self.log(f"run: device={self.device} params={self.model.n_params()/1e6:.1f}M n_games={sp['n_games']} threads={sp['threads']}")
        while True:
            # フラグ
            if self.sd.flag("STOP"):
                self.log("STOP flag: checkpoint and exit")
                self.auto.stop()
                self.checkpoint()
                self.write_status()
                self.sd.clear_flag("STOP")
                return
            if self.sd.flag("PAUSE"):
                if not self.paused:
                    self.paused = True
                    self.checkpoint()
                    self.write_status()
                    torch.cuda.empty_cache() if self.device.type == "cuda" else None
                    self.log("PAUSE flag: waiting")
                time.sleep(1.0)
                if time.time() - last_status > rr["status_seconds"]:
                    self.write_status()
                    last_status = time.time()
                continue
            if self.paused:
                self.paused = False
                self.log("resume")
            th = self.sd.throttle_value()
            want = min(sp["n_games"], max(1, th)) if th else sp["n_games"]
            if self.loop.engine.active != want:
                self.loop.set_active(want)
                self.log(f"active games -> {want}")
            # 自己対局
            finished = self.loop.round()
            if finished:
                for g in finished:
                    if "exploiter_result" in g:
                        r = int(g["exploiter_result"])
                        self.exploiter_stats["games"] += 1
                        self.exploiter_stats["wins" if r > 0 else "draws" if r == 0 else "losses"] += 1
                self.replay.add_games(finished)
                new_games += len(finished)
                self.session_games += len(finished)
                self.state["games_total"] = self.replay.total_games
                self.state["chunk_index"] = self.replay.chunk_index
            # 学習: 新規 N 局ごとに、局面数 × replay_ratio / batch ステップ
            if new_games >= tr["train_every_games"] and self.replay.n_games() >= tr["min_window_games"]:
                positions = sum(len(g["moves"]) for g in finished) if finished else 0
                avg_len = self.replay.n_positions() / max(1, self.replay.n_games())
                steps = max(1, int(round(new_games * avg_len * tr["replay_ratio"] / tr["batch_size"])))
                t0 = time.time()
                # バッチ作成（CPU、replay_features は GIL を離す）と学習ステップ（GPU）を重ねる
                sample = lambda: self.replay.sample(tr["batch_size"], self.rng, tr["mirror_prob"], tr["lambda_z"], self.cfg["search"]["policy_topk"])  # noqa: E731
                fut = self.pool.submit(sample)
                for _ in range(steps):
                    batch = fut.result()
                    fut = self.pool.submit(sample)
                    self.last_train = self.trainer.step(batch)
                fut.result()
                self.last_train["steps"] = steps
                self.last_train["sec"] = round(time.time() - t0, 1)
                self.loop.set_model(self.model)
                self.state["step"] = self.trainer.step_count
                new_games = 0
            now = time.time()
            self.reload_openings()
            if now - last_status > rr["status_seconds"]:
                self.write_status()
                self.auto.poll()
                last_status = now
            if now - last_ck > rr["checkpoint_minutes"] * 60:
                self.checkpoint()
                last_ck = time.time()


def main_run(root: Path, config_path: Path | None) -> None:
    sd = StateDir(root)
    cfg_path = config_path if config_path else (sd.config_toml if sd.config_toml.exists() else None)
    cfg = load_config(cfg_path)
    sd.create()
    lock = sd.root / "run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        pid = lock.read_text().strip()
        if pid and Path(f"/proc/{pid}").exists():
            print(f"already running (pid {pid})", file=sys.stderr)
            sys.exit(1)
        lock.write_text(str(os.getpid()))
    try:
        Runner(sd, cfg).run()
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
