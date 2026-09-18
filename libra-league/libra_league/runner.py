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

from .auto import AutoJobs, append_metrics, crossed_games_multiple
from .progress import Publisher
from .config import dump_toml, load_config
from .league import add_result, list_pool, main_winrate, pfsp_pick, pool_name, prune_pool, tag_league_game
from .looptime import LoopTimer
from .replay import ReplayBuffer, add_target_stats, summarize_target_stats
from .selfplay import SelfPlayLoop
from .state import StateDir, write_json_atomic
from .supervise import EXIT_ALREADY_RUNNING, acquire_lock
from .trainer import Trainer
from .workers import Inbox, load_weights, publish_weights


def train_steps(new_games: int, avg_len: float, tr: dict) -> int:
    """新規 new_games 局ぶんの学習ステップ数（局面数 × replay_ratio / batch）。手元の自己対局かワーカーの局かは区別しない。"""
    return max(1, int(round(new_games * avg_len * tr["replay_ratio"] / tr["batch_size"])))


class Runner:
    def __init__(self, sd: StateDir, cfg: dict, device: torch.device | None = None):
        self.sd = sd
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LibraNet(NetConfig.from_dict(cfg["net"])).to(self.device)
        self.trainer = Trainer(self.model, cfg["train"], self.device)
        tr, sr, rr = cfg["train"], cfg["search"], cfg["run"]
        self.replay = ReplayBuffer(sd.replay, sd.games, tr["window_games"], rr["chunk_games"], sr["max_ply"], sr["count_from_41"],
                                   window_frac=float(tr.get("window_frac", 0.0)), window_games_max=int(tr.get("window_games_max", 0)),
                                   heldout_every_chunks=int(rr.get("heldout_every_chunks", 0)), heldout_games=int(rr.get("heldout_games", 20000)))
        self.state = {
            "run_id": cfg["run_id"], "created": time.time(), "generation": 0, "step": 0, "games_total": 0, "moves_total": 0,
            "chunk_index": 0, "seed": cfg["seed"], "restarts": [], "last_checkpoint": None, "elapsed": 0.0,
        }
        self.rng = np.random.default_rng(cfg["seed"])
        self.loop: SelfPlayLoop | None = None
        self.started = time.time()
        self.session_games = 0
        self.session_elapsed_offset = 0.0
        self.rate_hist: list[tuple[float, int]] = []
        self.last_train: dict = {}
        self.timer = LoopTimer()  # ループの処理時間の内訳（metrics の 1 行ごとに閉じる。looptime.py）
        self.last_timing: dict | None = None
        self.train_mode = ""  # 学習の compile の状態（変わったらログに出す）
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.exploiter_stats = {"games": 0, "wins": 0, "draws": 0, "losses": 0}
        self.opponent_step: int | None = None
        self.last_openings_write = 0.0
        self.last_source_check = 0.0
        self.openings_mtime: float | None = None
        self.openings_checked = 0.0
        self.auto = AutoJobs(sd, cfg, self.state, self.log)
        self.progress = Publisher(sd, cfg, self.log)  # 進捗の要約をリポジトリへ（[progress] enabled のとき）
        self.last_metrics = 0.0
        self.last_calib = 0.0
        self.last_gen = 0.0
        self.last_gen_games = -1
        self.gen: dict | None = None  # 一般化の物差し（genprof.py。窓の中と held-out）
        # 自己対局ワーカー（[workers] enabled）: 重みを weights/ に配り、inbox/ に届いた局を取り込む。搾取者の run では使わない
        self.inbox: Inbox | None = None
        wk = cfg.get("workers", {})
        if wk.get("enabled") and not cfg.get("exploiter", {}).get("main_ckpt"):
            self.inbox = Inbox(sd.inbox, cfg["run_id"], int(wk.get("max_lag_steps", 0)), self.log)
        # 本体と過去の搾取者の対局（[league] enabled）: 自己対局とは別のエンジンで打つ。搾取者の run では使わない
        self.league_enabled = bool(cfg.get("league", {}).get("enabled")) and not cfg.get("exploiter", {}).get("main_ckpt")
        self.league_loop: SelfPlayLoop | None = None
        self.league_opponent: int | None = None
        self.league_since_switch = 0
        self.league_checked = 0.0
        self.league_waiting_logged = False

    # ---- 永続化 ----
    def log(self, msg: str) -> None:
        print(msg, flush=True)
        self.sd.append_log(msg)

    def infer_modes(self) -> str:
        """推論の捕獲の状態（例: model=compile(max-autotune)+cudagraph）。変わったらログに出す。"""
        if self.loop is None:
            return ""
        nets = [("model", self.loop.model), ("opponent", self.loop.opponent)]
        if self.league_loop is not None:
            nets += [("league_model", self.league_loop.model), ("league_opponent", self.league_loop.opponent)]
        return " ".join(f"{k}={n.mode_used}" for k, n in nets if n is not None)

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
        self.log(f"resume: step={self.state['step']} games={self.state['games_total']} window={self.replay.n_games()}/{self.replay.window()} heldout={self.replay.n_heldout()} chunks={self.state['chunk_index']}")

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

    def publish_weights(self) -> None:
        """ワーカー用の重み（fp16 のモデルだけ）を配る。手元の推論用ネットを更新するのと同じ時点で書く。"""
        if self.inbox is None:
            return
        try:
            publish_weights(self.sd.weights / "latest.pt", self.model, self.trainer.step_count, self.cfg["net"], self.cfg["run_id"])
        except OSError as e:
            self.log(f"workers: publish failed: {e}")

    def ingest_workers(self) -> int:
        """inbox に届いた局をリプレイに足し、足した局数を返す。"""
        if self.inbox is None:
            return 0
        games = self.inbox.poll(self.trainer.step_count)
        if games:
            self.replay.add_games(games)
            self.state["games_total"] = self.replay.total_games
            self.state["chunk_index"] = self.replay.chunk_index
            self.log(f"workers: ingested {len(games)} games (total {self.inbox.stats['games']})")
        return len(games)

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
        prior = bool(ex.get("opponent_prior", True))
        self.loop.set_opponent(m, opponent_prior=prior)
        self.opponent_step = sd.get("step")
        self.exploiter_state()["main_step"] = self.opponent_step
        self.log(f"exploiter: opponent {path} step {sd.get('step', '?')} params {m.n_params()/1e6:.1f}M (even slots: exploiter sente; "
                 f"opponent_prior {'on' if prior else 'off'})")

    # ---- 凍結相手の作り直しと布石の書き出し（搾取者の run だけ） ----
    def exploiter_state(self) -> dict:
        es = self.state.setdefault("exploiter", {})
        for k, v in (("main_step", None), ("refreshed_at", None), ("from_chunk", 0), ("history", [])):
            es.setdefault(k, v)
        return es

    def refresh_main(self) -> None:
        """凍結相手を本体ランの最新で作り直す。今までの成績は履歴に移し、布石は空にして区切る。"""
        ex = self.cfg.get("exploiter", {})
        src = Path(str(ex.get("main_source") or "")).expanduser()
        dst = Path(str(ex.get("main_ckpt") or "")).expanduser()
        if not src.exists() or not str(dst):
            self.log(f"exploiter: main_source not found: {src}")
            return
        self.save_pool_snapshot()  # 作り直す前の自分（この相手の穴を突くよう学んだ搾取者）を本体の対局相手に残す
        es = self.exploiter_state()
        if self.exploiter_stats["games"]:
            g = max(1, self.exploiter_stats["games"])
            es["history"] = (es["history"] + [{
                "t": time.time(), "main_step": es.get("main_step"), "games": self.exploiter_stats["games"],
                "winrate": round((self.exploiter_stats["wins"] + 0.5 * self.exploiter_stats["draws"]) / g, 4),
            }])[-40:]
        tmp = dst.with_suffix(".pt.tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        self.exploiter_stats = {"games": 0, "wins": 0, "draws": 0, "losses": 0}
        self.state.pop("exploiter_stats", None)
        self.load_opponent()
        es["main_step"] = self.opponent_step
        es["refreshed_at"] = time.time()
        es["from_chunk"] = self.replay.chunk_index
        out = str(ex.get("openings_out") or "")
        if out:
            from .openings import write_openings

            write_openings(Path(out).expanduser(), [], str(self.sd.root))  # 古い相手の穴なので本体には渡さない
        self.last_openings_write = time.time()
        self.sd.write_state(self.state)
        self.log(f"exploiter: main refreshed from {src} (step {self.opponent_step}); stats and openings reset at chunk {es['from_chunk']}")

    def source_step(self) -> int | None:
        """main_source の step だけを読む（mmap なので重みは読まない。123 MB で 0.01 秒）。"""
        src = Path(str(self.cfg.get("exploiter", {}).get("main_source") or "")).expanduser()
        try:
            step = torch.load(src, map_location="cpu", mmap=True, weights_only=False).get("step")
            return None if step is None else int(step)
        except Exception:  # noqa: BLE001  書き換え中・無いときは次の確認に回す
            return None

    def maybe_refresh_main(self) -> None:
        """refresh_hours が過ぎたか、main_source が凍結相手より refresh_steps 以上進んだら作り直す。"""
        ex = self.cfg.get("exploiter", {})
        if not ex.get("main_source") or self.loop is None or self.loop.opponent is None:
            return
        es = self.exploiter_state()
        hours = float(ex.get("refresh_hours", 0.0) or 0.0)
        last = es.get("refreshed_at")
        if hours > 0 and (last is None or time.time() - float(last) >= hours * 3600):
            self.refresh_main()
            return
        steps = int(ex.get("refresh_steps", 0) or 0)
        now = time.time()
        if steps <= 0 or now - self.last_source_check < float(ex.get("refresh_check_minutes", 5.0)) * 60:
            return
        self.last_source_check = now
        src_step = self.source_step()
        if src_step is None:
            return
        es["source_step"] = src_step
        if self.opponent_step is None or src_step - int(self.opponent_step) >= steps:
            self.log(f"exploiter: main_source step {src_step} is {steps}+ steps ahead of opponent step {self.opponent_step}")
            self.refresh_main()

    def maybe_write_openings(self) -> None:
        """搾取者が勝った布石を openings_out に書く（凍結相手を作り直してからの対局だけ）。"""
        ex = self.cfg.get("exploiter", {})
        out = str(ex.get("openings_out") or "")
        mins = float(ex.get("openings_minutes", 0.0) or 0.0)
        if not out or mins <= 0 or self.loop is None or self.loop.opponent is None:
            return
        now = time.time()
        if now - self.last_openings_write < mins * 60:
            return
        self.last_openings_write = now
        from .openings import openings_from_replay, write_openings

        es = self.exploiter_state()
        lines = openings_from_replay(self.sd.replay, int(ex.get("openings_chunks", 50)), int(ex.get("openings_moves", 12)),
                                     min_chunk=int(es.get("from_chunk", 0)))
        write_openings(Path(out).expanduser(), lines, str(self.sd.root))
        self.log(f"exploiter: wrote {len(lines)} openings to {out} (chunks >= {es.get('from_chunk', 0)})")

    # ---- 搾取者のスナップショット（搾取者の run）と、本体と過去の搾取者の対局（本体の run） ----
    def save_pool_snapshot(self) -> None:
        """今の重みを [exploiter] pool_out に lx-<step>.pt（workers.publish_weights の形式）で保存し、新しい pool_keep 個を残す。"""
        ex = self.cfg.get("exploiter", {})
        out = str(ex.get("pool_out") or "")
        if not out:
            return
        pool = Path(out).expanduser()
        step = self.trainer.step_count
        try:
            pool.mkdir(parents=True, exist_ok=True)
            publish_weights(pool / pool_name(step), self.model, step, self.cfg["net"], self.cfg["run_id"])
            prune_pool(pool, int(ex.get("pool_keep", 10)))
            self.log(f"exploiter: saved snapshot step {step} to {pool}")
        except OSError as e:
            self.log(f"exploiter: snapshot failed: {e}")

    def ensure_pool_snapshot(self) -> None:
        """プールが空なら今の自分を保存する（起動時。本体の run がすぐ相手にできるように）。"""
        out = str(self.cfg.get("exploiter", {}).get("pool_out") or "")
        if out and not list_pool(Path(out).expanduser(), 1):
            self.save_pool_snapshot()

    def start_league(self) -> None:
        lg, sp = self.cfg["league"], self.cfg["selfplay"]
        # eval_cache は切る: 1 ラウンドで進む手数が増えると、自己対局と同じラウンド数だけ進めるリーグの局の割合が変わるため
        # （2026-09-17、割合を変えるかはユーザーの判断待ち。それまでの搾取者モードのキャッシュ無しと同じ）
        self.league_loop = SelfPlayLoop({**self.cfg["search"], "eval_cache": False}, int(lg["n_games"]), int(lg["threads"]), int(self.rng.integers(0, 2**63)), self.device,
                                        sp["infer_dtype"], sp.get("compile", "none"))
        self.league_loop.set_model(self.model)
        self.state.setdefault("league", {}).setdefault("stats", {})
        self.league_switch()

    def league_switch(self) -> None:
        """プールの新しい recent 体から PFSP で相手を選び、相手のネットに重みを入れる。
        対局中の局は新しい相手で続き、結果は終局時の相手に数える。"""
        lg = self.cfg["league"]
        self.league_checked = time.time()
        pool = list_pool(Path(str(lg.get("pool") or "")).expanduser(), int(lg.get("recent", 5)))
        if not pool:
            if not self.league_waiting_logged:
                self.log(f"league: no exploiter snapshots in {lg.get('pool')}; checking every {lg.get('pool_check_minutes', 10.0)} min")
                self.league_waiting_logged = True
            return
        self.league_waiting_logged = False
        st = self.state.setdefault("league", {})
        steps = [s for s, _ in pool]
        keep = {str(s) for s in steps} | {str(self.league_opponent)}
        stats = st["stats"] = {k: v for k, v in st.get("stats", {}).items() if k in keep}
        step = pfsp_pick(steps, stats, self.rng)
        self.league_since_switch = 0
        if step == self.league_opponent:
            return
        path = dict(pool)[step]
        try:
            m, step, _ = load_weights(path)
        except Exception as e:  # noqa: BLE001  消された・書きかけなら次の選び直しに回す
            self.log(f"league: failed to load {path}: {type(e).__name__}: {e}")
            return
        assert self.league_loop is not None
        self.league_loop.set_opponent(m, opponent_prior=False)  # 相手（lx）も木を丸ごと自分のネットで読む
        self.league_opponent = step
        st["opponent_step"] = step
        self.log(f"league: opponent lx step {step} (pool {steps}, main winrate {[round(main_winrate(stats.get(str(s))), 3) for s in steps]})")

    def league_round(self) -> list[dict]:
        """対 lx の対局を 1 ラウンド進め、終局した局に印を付けて返す。相手がまだ無ければプールを見に行くだけ。"""
        if self.league_loop is None:
            return []
        lg = self.cfg["league"]
        if self.league_opponent is None:
            if time.time() - self.league_checked >= float(lg.get("pool_check_minutes", 10.0)) * 60:
                self.league_switch()
            return []
        games = self.league_loop.round()
        st = self.state["league"]
        for g in games:
            tag_league_game(g, self.league_opponent)
            add_result(st["stats"], self.league_opponent, g["league_result"])
        st["games"] = st.get("games", 0) + len(games)
        self.league_since_switch += len(games)
        if self.league_since_switch >= int(lg.get("switch_games", 256)):
            self.league_switch()
        return games

    def league_status(self) -> dict:
        lg = self.cfg["league"]
        st = self.state.get("league", {})
        stats = st.get("stats", {})
        pool = list_pool(Path(str(lg.get("pool") or "")).expanduser(), int(lg.get("recent", 5)))
        return {"opponent_step": self.league_opponent, "games": st.get("games", 0), "n_games": int(lg["n_games"]),
                "pool": [{"step": s, **(stats.get(str(s)) or {"games": 0, "wins": 0, "draws": 0, "losses": 0}),
                          "main_winrate": round(main_winrate(stats.get(str(s))), 4)} for s, _ in pool]}

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

    def take_timing(self) -> dict:
        """処理時間の窓を閉じる（自己対局とリーグのループの段ごとの累計も取って空にする）。"""
        sp = self.loop.timing if self.loop is not None else None
        lg = self.league_loop.timing if self.league_loop is not None else None
        out = self.timer.take(sp, lg)
        for loop in (self.loop, self.league_loop):
            if loop is not None:
                loop.timing = {}
        return out

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
            "active_games": self.loop.engine.active if self.loop else 0,
            "step": self.trainer.step_count,
            "generation": self.state.get("generation", 0),
            "games_total": self.replay.total_games,
            "games_session": self.session_games,
            "games_per_day_1h": round(rate),
            "window_games": self.replay.n_games(),
            "window_target": self.replay.window(),
            "heldout_games": self.replay.n_heldout(),
            "gen": self.gen,
            "elapsed_h": round(self.elapsed() / 3600, 2),
            "engine": st,
            "train": self.last_train,
            "gpu": gpu,
            "restarts": self.state.get("restarts", [])[-5:],
        }
        if self.inbox is not None:
            status["workers"] = self.inbox.stats
        if self.loop and self.loop.opponent is not None:
            es = dict(self.exploiter_stats)
            es["winrate"] = round((es["wins"] + 0.5 * es["draws"]) / max(1, es["games"]), 4)
            xs = self.state.get("exploiter", {})
            es["main_step"] = xs.get("main_step")
            es["refreshed_at"] = xs.get("refreshed_at")
            es["source_step"] = xs.get("source_step")  # refresh_steps が有効なときだけ入る（本体の最新 step）
            es["history"] = xs.get("history", [])[-10:]
            status["exploiter"] = es
        if self.league_loop is not None:
            status["league"] = self.league_status()
        take_timing = now - self.last_metrics >= float(self.cfg["run"].get("metrics_minutes", 5)) * 60 and self.loop is not None
        if take_timing:
            self.last_timing = self.take_timing()
        if self.last_timing is not None:
            status["timing"] = self.last_timing
        write_json_atomic(self.sd.status_json, status)
        if take_timing:
            append_metrics(self.sd, status)
            self.last_metrics = now
        self.maybe_write_calib(now)
        self.maybe_measure_gen(now)

    def maybe_measure_gen(self, now: float) -> None:
        """gen_minutes ごとに、窓の中と held-out の局面で一般化の物差し（genprof.py）を測る。held-out が無い run では何もしない。"""
        rc, tr = self.cfg["run"], self.cfg["train"]
        minutes = float(rc.get("gen_minutes", 60))
        every_games = int(rc.get("gen_games", 0))
        games = self.replay.total_games
        if every_games > 0:  # 局数で（PC の利用状況で局/日が変わっても同じ局数ごと）
            due = crossed_games_multiple(self.last_gen_games, games, every_games)  # 倍数（2 万・4 万…）で
        else:
            due = minutes > 0 and now - self.last_gen >= minutes * 60
        if not due or self.replay.n_heldout() <= 0 or self.replay.n_games() < int(tr["min_window_games"]):
            return
        from .genprof import format_gen, generalization

        self.last_gen = now
        self.last_gen_games = games
        t0 = time.time()
        self.gen = generalization(self.model, self.replay, int(rc.get("gen_positions", 4000)), self.rng, self.device, tr["lambda_z"], self.cfg["search"]["policy_topk"])
        if self.gen is not None:
            self.gen["t"] = round(now, 1)
            self.gen["step"] = self.trainer.step_count
            self.gen["games"] = games
            self.log(format_gen(self.gen) + f" ({time.time() - t0:.1f}s)")

    def maybe_write_calib(self, now: float) -> None:
        """calib_minutes ごとに、窓の最新 calib_games 局で同じネットの較正を calib.jsonl に足す（docs/decisions.md 2026-09-17）。
        搾取者の run では書かない: 記録が全部本体との対局で、自己対局の較正と同じ意味にならない（手で `libra --run lx calib` は回せる）。"""
        rc = self.cfg["run"]
        minutes = float(rc.get("calib_minutes", 60))
        if self.cfg.get("exploiter", {}).get("main_ckpt") or minutes <= 0 or now - self.last_calib < minutes * 60 or not self.replay.games:
            return
        from .calibrate import append_calib, make_row

        games = self.replay.window_games_list()[-int(rc.get("calib_games", 20000)):]
        append_calib(self.sd, make_row(games, self.state.get("step"), self.state.get("generation"), now=now))
        self.last_calib = now

    # ---- メインループ ----
    def run(self) -> None:
        self.sd.create()
        if not self.sd.config_toml.exists():
            self.sd.config_toml.write_text(dump_toml(self.cfg), encoding="utf-8")
        self.load()
        self.auto.recover()  # 前回が abort で終わっていれば、残った計測ジョブを止めて積み直す
        self.auto.repair_anchor_chain()  # 基準比の行を結果のファイルから数え直す（2026-09-18 の offset の誤り）
        self.sd.write_state(self.state)
        sp = self.cfg["selfplay"]
        self.loop = SelfPlayLoop(self.cfg["search"], sp["n_games"], sp["threads"], int(self.rng.integers(0, 2**63)), self.device, sp["infer_dtype"],
                                 sp.get("compile", "none"))
        self.loop.set_model(self.model)
        self.loop.timing = {}
        self.load_opponent()
        self.ensure_pool_snapshot()
        self.reload_openings(force=True)
        if self.league_enabled:
            self.start_league()
            if self.league_loop is not None:
                self.league_loop.timing = {}
        self.timer = LoopTimer()  # 立ち上げ（読み込み・compile の前段）は窓に入れない
        tr, rr = self.cfg["train"], self.cfg["run"]
        last_ck = time.time()
        last_status = 0.0
        last_ingest = 0.0
        new_games = 0
        if self.inbox is not None:
            self.sd.inbox.mkdir(exist_ok=True)
            self.sd.weights.mkdir(exist_ok=True)
            self.publish_weights()
        elif self.cfg.get("workers", {}).get("enabled"):
            self.log("workers: disabled (exploiter runs do not take worker games)")
        self.log(f"run: device={self.device} params={self.model.n_params()/1e6:.1f}M n_games={sp['n_games']} threads={sp['threads']}"
                 + (f" workers=inbox(max_lag_steps={self.inbox.max_lag_steps})" if self.inbox else "")
                 + (f" league=n_games {self.cfg['league']['n_games']} vs {self.cfg['league']['pool']}" if self.league_loop is not None else ""))
        while True:
            # フラグ
            if self.sd.flag("STOP"):
                self.log("STOP flag: checkpoint and exit")
                self.auto.stop()
                self.checkpoint()
                self.write_status()
                self.sd.clear_flag("STOP")
                return
            # 一時停止は廃止した（止めるときは STOP。docs/decisions.md 2026-09-14）
            # 自己対局
            modes = self.infer_modes()
            finished = self.loop.round()
            with self.timer.phase("league"):
                league_games = self.league_round()  # 対 lx（[league] enabled のときだけ。自己対局と同じ回数だけ進める）
            if self.infer_modes() != modes:
                self.log(f"selfplay: inference {self.infer_modes()}")
            if league_games:
                with self.timer.phase("replay"):
                    self.replay.add_games(league_games)
                new_games += len(league_games)
                self.session_games += len(league_games)
                self.state["games_total"] = self.replay.total_games
                self.state["chunk_index"] = self.replay.chunk_index
            if finished:
                for g in finished:
                    if "exploiter_result" in g:
                        r = int(g["exploiter_result"])
                        self.exploiter_stats["games"] += 1
                        self.exploiter_stats["wins" if r > 0 else "draws" if r == 0 else "losses"] += 1
                with self.timer.phase("replay"):
                    self.replay.add_games(finished)
                new_games += len(finished)
                self.session_games += len(finished)
                self.state["games_total"] = self.replay.total_games
                self.state["chunk_index"] = self.replay.chunk_index
            if self.inbox is not None and time.time() - last_ingest >= float(self.cfg["workers"].get("ingest_seconds", 10)):
                with self.timer.phase("ingest"):
                    n = self.ingest_workers()
                new_games += n
                self.session_games += n  # games_session と局/日（total_games から数える）を揃える
                last_ingest = time.time()
            # 学習: 新規 N 局ごとに、局面数 × replay_ratio / batch ステップ
            if new_games >= tr["train_every_games"] and self.replay.n_games() >= tr["min_window_games"]:
                avg_len = self.replay.n_positions() / max(1, self.replay.n_games())
                steps = train_steps(new_games, avg_len, tr)
                t0 = time.time()
                # バッチ作成（CPU、replay_features は GIL を離す）と学習ステップ（GPU）を重ねる
                sample = lambda: self.replay.sample(tr["batch_size"], self.rng, tr["mirror_prob"], tr["lambda_z"], self.cfg["search"]["policy_topk"])  # noqa: E731
                fut = self.pool.submit(sample)
                target_acc: dict = {}
                for _ in range(steps):
                    with self.timer.phase("train_sample"):
                        batch = fut.result()
                    fut = self.pool.submit(sample)
                    add_target_stats(target_acc, batch["target_stats"])
                    with self.timer.phase("train_step"):
                        self.last_train = self.trainer.step(batch)
                with self.timer.phase("train_sample"):
                    fut.result()
                self.timer.count("train_steps", steps)
                self.last_train["steps"] = steps
                if self.trainer.mode_used != self.train_mode:
                    self.train_mode = self.trainer.mode_used
                    self.log(f"train: model={self.train_mode}")
                self.last_train["sec"] = round(time.time() - t0, 1)
                self.last_train["target"] = summarize_target_stats(target_acc)  # 学習目標と結果の差（metrics.jsonl・コンソール）
                with self.timer.phase("train_publish"):
                    self.loop.set_model(self.model)
                    if self.league_loop is not None:
                        self.league_loop.set_model(self.model)
                    self.publish_weights()
                self.state["step"] = self.trainer.step_count
                new_games = 0
            now = time.time()
            with self.timer.phase("housekeeping"):
                self.maybe_refresh_main()
                self.maybe_write_openings()
                self.reload_openings()
            if now - last_status > rr["status_seconds"]:
                with self.timer.phase("status"):
                    self.write_status()
                    changed = self.auto.poll()
                    if changed:
                        self.sd.write_state(self.state)  # コンソールの「自動計測」欄が実行中/待機を追えるように
                    self.progress.maybe_publish(now, milestone=changed)
                last_status = now
            if now - last_ck > rr["checkpoint_minutes"] * 60 or self.auto.games_due(self.replay.total_games):  # 計測の区切り（10 万局など）は待たない
                with self.timer.phase("checkpoint"):
                    self.checkpoint()
                last_ck = time.time()


def main_run(root: Path, config_path: Path | None) -> None:
    sd = StateDir(root)
    cfg_path = config_path if config_path else (sd.config_toml if sd.config_toml.exists() else None)
    cfg = load_config(cfg_path)
    sd.create()
    lock = sd.root / "run.lock"
    pid = acquire_lock(lock)
    if pid is not None:
        print(f"already running (pid {pid})", file=sys.stderr)
        sys.exit(EXIT_ALREADY_RUNNING)  # 監視役はこれを見て起動し直さない
    try:
        Runner(sd, cfg).run()
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
