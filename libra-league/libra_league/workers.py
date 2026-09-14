# SPDX-License-Identifier: Apache-2.0
"""自己対局ワーカーと学習側の分離（docs/libra-design.md §6.1、docs/libra-local.md §9）。

ワーカーは自己対局だけを回す: 学習側が配る重み（<run>/weights/latest.pt、fp16 のモデルだけ）を読み、
終局した対局を chunk_games 局ずつ対局ファイル（<run>/inbox/<id>-<時刻>-<連番>.npz）に置く。
学習側は inbox を取り込んで手元の自己対局と同じようにリプレイに足す。チャンクの採番と棋譜 JSONL は学習側だけが書く。

対局ファイルは pickle を使わない（別マシンのワーカーは信用しない。np.load は allow_pickle=False）。
read_games_file が検査するのは型・形・値域・整合性まで。verify_games_file はさらに手を再生して合法性・終局の判定・方策の添字を確かめる
（別マシンのワーカーの局に使う。libra-cloud のブリッジ）。探索の出力（方策の確率・価値）の改ざんは検出できない。
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import librashogi as ls
from libra_net.model import LibraNet, NetConfig

FORMAT = 1
MAX_FILE_BYTES = 256 << 20  # 展開後。100 局で数 MB
MAX_GAMES = 10000
MAX_MOVES = 4096
WORKER_ID = re.compile(r"^[A-Za-z0-9_]{1,32}$")
REASONS = frozenset({"none", "no_legal_move", "ruling41", "sennichite", "perpetual_check", "declaration", "illegal_declaration",
                     "illegal_move", "max_ply", "resign", "timeout"})
GAME_KEYS = ("slot", "kb", "kw", "result", "reason", "v41", "plies", "sfen41")
# 配列（1 次元）と dtype。moves・full・root_q は局ごとに手数ぶん、policy_off は手数 + 1、policy_* は方策の要素数ぶんを連結する
ARRAYS = {"meta": np.uint8, "n_moves": np.int32, "n_policy": np.int32, "moves": np.uint32, "full": np.uint8, "root_q": np.float32,
          "policy_idx": np.int16, "policy_p": np.float32, "policy_off": np.int32}
KEEP_REJECTED = 20


class GamesFileError(ValueError):
    pass


# ---- 対局ファイル ----
def write_games_file(inbox: Path, worker_id: str, weights_step: int, run_id: str, games: list[dict]) -> Path:
    """games を 1 ファイルに書く（tmp → os.replace）。書いたパスを返す。"""
    if not WORKER_ID.match(worker_id):
        raise ValueError(f"worker id must match {WORKER_ID.pattern}: {worker_id!r}")
    meta = {"format": FORMAT, "worker": worker_id, "weights_step": int(weights_step), "run_id": run_id, "created": time.time(),
            "games": [{"slot": int(g["slot"]), "kb": int(g["kb"]), "kw": int(g["kw"]), "result": int(g["result"]), "reason": str(g["reason"]),
                       "v41": float(g["v41"]), "plies": int(g["plies"]), "sfen41": str(g["sfen41"])} for g in games]}

    def cat(key: str) -> np.ndarray:
        return np.concatenate([np.asarray(g[key], ARRAYS[key]) for g in games]) if games else np.zeros(0, ARRAYS[key])

    arrs = {"meta": np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), np.uint8),
            "n_moves": np.array([len(g["moves"]) for g in games], np.int32),
            "n_policy": np.array([len(g["policy_idx"]) for g in games], np.int32),
            **{k: cat(k) for k in ("moves", "full", "root_q", "policy_idx", "policy_p", "policy_off")}}
    name = f"{worker_id}-{int(time.time() * 1000)}-{write_games_file.seq:06d}.npz"
    write_games_file.seq += 1
    path = inbox / name
    tmp = inbox / (name + ".tmp")
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    with open(tmp, "wb") as f:
        f.write(buf.getbuffer())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


write_games_file.seq = 0  # type: ignore[attr-defined]


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise GamesFileError(msg)


def _int(v, lo: int, hi: int, what: str) -> int:
    _check(isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi, f"{what} out of range: {v!r}")
    return v


def read_games_file(path: Path, max_bytes: int = MAX_FILE_BYTES) -> tuple[dict, list[dict]]:
    """対局ファイルを検査して (meta, games) を返す。games は libra-search の record_to_dict と同じ形。不正なら GamesFileError。"""
    try:
        _check(path.stat().st_size <= max_bytes, "file too large")
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            _check(sum(i.file_size for i in infos) <= max_bytes, "uncompressed size too large")
            _check(sorted(i.filename for i in infos) == sorted(k + ".npy" for k in ARRAYS), f"unexpected members {[i.filename for i in infos]}")
        with np.load(path, allow_pickle=False) as z:
            arrs = {k: z[k] for k in ARRAYS}
    except GamesFileError:
        raise
    except Exception as e:  # noqa: BLE001  壊れた zip・pickle を含む配列など
        raise GamesFileError(f"unreadable: {type(e).__name__}: {str(e)[:200]}") from e
    for k, dt in ARRAYS.items():
        _check(arrs[k].dtype == np.dtype(dt) and arrs[k].ndim == 1, f"{k}: dtype {arrs[k].dtype} shape {arrs[k].shape}")
    try:
        meta = json.loads(arrs["meta"].tobytes().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise GamesFileError(f"meta: {type(e).__name__}") from e
    _check(isinstance(meta, dict) and meta.get("format") == FORMAT, "meta: format")
    _check(isinstance(meta.get("worker"), str) and bool(WORKER_ID.match(meta["worker"])), "meta: worker")
    _int(meta.get("weights_step"), 0, 2**62, "weights_step")
    _check(isinstance(meta.get("run_id"), str) and len(meta["run_id"]) <= 64, "meta: run_id")
    gm = meta.get("games")
    _check(isinstance(gm, list) and 1 <= len(gm) <= MAX_GAMES, "meta: games")
    n_moves, n_policy = arrs["n_moves"].astype(np.int64), arrs["n_policy"].astype(np.int64)
    _check(len(n_moves) == len(gm) and len(n_policy) == len(gm), "game count mismatch")
    _check(bool(((n_moves >= 1) & (n_moves <= MAX_MOVES)).all() and (n_policy >= 0).all()), "n_moves / n_policy out of range")
    tm, tp = int(n_moves.sum()), int(n_policy.sum())
    _check(len(arrs["moves"]) == tm and len(arrs["full"]) == tm and len(arrs["root_q"]) == tm, "moves length")
    _check(len(arrs["policy_idx"]) == tp and len(arrs["policy_p"]) == tp, "policy length")
    _check(len(arrs["policy_off"]) == tm + len(gm), "policy_off length")
    _check(bool((arrs["full"] <= 1).all()), "full not 0/1")
    rq, pp, pi = arrs["root_q"], arrs["policy_p"], arrs["policy_idx"]
    _check(bool(np.isfinite(rq).all() and (np.abs(rq) <= 1.0 + 1e-4).all()), "root_q out of range")
    _check(bool(np.isfinite(pp).all() and (pp >= 0).all() and (pp <= 1.0 + 1e-4).all()), "policy_p out of range")
    _check(bool(((pi >= 0) & (pi < ls.POLICY_SIZE)).all()), "policy_idx out of range")
    games: list[dict] = []
    mo = po = oo = 0
    for i, g in enumerate(gm):
        _check(isinstance(g, dict) and sorted(g) == sorted(GAME_KEYS), f"game {i}: keys")
        n, k = int(n_moves[i]), int(n_policy[i])
        off = arrs["policy_off"][oo:oo + n + 1]
        _check(int(off[0]) == 0 and int(off[-1]) == k and bool((np.diff(off) >= 0).all()), f"game {i}: policy_off")
        _check(isinstance(g["reason"], str) and g["reason"] in REASONS, f"game {i}: reason")
        _check(isinstance(g["sfen41"], str) and len(g["sfen41"]) <= 512, f"game {i}: sfen41")
        _check(isinstance(g["v41"], (int, float)) and math.isfinite(g["v41"]) and abs(g["v41"]) <= 1.0 + 1e-4, f"game {i}: v41")
        games.append({
            "slot": _int(g["slot"], 0, 1 << 20, "slot"), "kb": _int(g["kb"], 0, 80, "kb"), "kw": _int(g["kw"], 0, 80, "kw"),
            "result": _int(g["result"], -1, 1, "result"), "reason": g["reason"], "v41": float(g["v41"]),
            "plies": _int(g["plies"], 0, MAX_MOVES + 2, "plies"), "sfen41": g["sfen41"],
            "moves": arrs["moves"][mo:mo + n].copy(), "full": arrs["full"][mo:mo + n].copy(), "root_q": arrs["root_q"][mo:mo + n].copy(),
            "policy_idx": pi[po:po + k].copy(), "policy_p": pp[po:po + k].copy(), "policy_off": off.copy(),
        })
        mo, po, oo = mo + n, po + k, oo + n + 1
    return meta, games


# ---- 手の再生による検査（別マシンのワーカーの局。libra-cloud のブリッジが学習側の inbox に置く前に呼ぶ） ----
_RESULT = {"sente": 1, "gote": -1, "draw": 0}


def verify_game(g: dict, search: dict) -> None:
    """read_games_file が返した 1 局を玉の配置から再生し、記録の骨格が自己対局の規則と合うかを確かめる。合わなければ GamesFileError。

    確かめるもの: 玉の配置と全手の合法性、終局より後に手が無いこと、方策の添字がその局面の合法手であること、
    終局の判定（result・reason・plies）、sfen41（本将棋に入った最初の局面。入らずに終わった局は最終局面）。
    手数の上限（max_moves_per_game）で打ち切った局は、再生した局面は終局していないが reason が timeout で手番側の負け
    （selfplay.cpp の安全弁）。root_q・v41・policy_p は探索の出力で、再生からは確かめられない（値域は read_games_file が見る）。"""
    p = ls.Position()
    p.set_max_ply(int(search["max_ply"]), bool(search["count_from_41"]))
    for key in ("kb", "kw"):
        u = "K*" + ls.sq_to_usi(g[key])
        _check(p.is_legal(u), f"{key}: illegal king placement {u}")
        p.do_move(u)
    moves, off, pidx = g["moves"], g["policy_off"], g["policy_idx"]
    sfen41 = None
    for j in range(len(moves)):
        _check(not p.is_over(), f"move {j}: after the end")
        codes = p.legal_move_codes()
        m = int(moves[j])
        _check(m in codes, f"move {j}: illegal")
        if sfen41 is None and p.phase == "normal":
            sfen41 = p.sfen()
        o0, o1 = int(off[j]), int(off[j + 1])
        if o1 > o0:
            legal = {p.move_index(ls.move_to_usi(c)) for c in codes}
            _check(all(int(i) in legal for i in pidx[o0:o1]), f"move {j}: policy index is not a legal move")
        p.do_move_code(m)
    if sfen41 is None:
        sfen41 = p.sfen()
    res, reason = p.outcome()
    if res == "ongoing":
        _check(len(moves) >= int(search["max_moves_per_game"]) and g["reason"] == "timeout"
               and g["result"] == (-1 if p.turn == "sente" else 1), f"unfinished game ({len(moves)} moves, {g['reason']})")
    else:
        _check(g["reason"] == reason and g["result"] == _RESULT.get(res), f"outcome {res}/{reason} != {g['result']}/{g['reason']}")
    _check(g["plies"] == p.ply, f"plies {g['plies']} != {p.ply}")
    _check(g["sfen41"] == sfen41, "sfen41")


def verify_games_file(path: Path, search: dict, max_bytes: int = MAX_FILE_BYTES) -> tuple[dict, list[dict]]:
    """read_games_file に加えて全局を verify_game で確かめる。1 局でも合わなければファイルごと GamesFileError。"""
    meta, games = read_games_file(path, max_bytes)
    for i, g in enumerate(games):
        try:
            verify_game(g, search)
        except GamesFileError as e:
            raise GamesFileError(f"game {i}: {e}") from None
        except (ValueError, RuntimeError, IndexError) as e:  # librashogi が不正な値で投げる
            raise GamesFileError(f"game {i}: {type(e).__name__}: {str(e)[:200]}") from None
    return meta, games


# ---- 重みの配布 ----
def publish_weights(path: Path, model: LibraNet, step: int, net_cfg: dict, run_id: str) -> None:
    """推論に要るものだけ（fp16 の重み、step、ネットの形）を原子的に書く。10M のネットで約 20 MB。"""
    sd = {k: (v.detach().to("cpu", torch.float16) if v.is_floating_point() else v.detach().cpu()) for k, v in model.state_dict().items()}
    obj = {"format": FORMAT, "model": sd, "step": int(step), "net": dict(net_cfg), "run_id": run_id}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_weights(path: Path) -> tuple[LibraNet, int, str]:
    w = torch.load(path, map_location="cpu", weights_only=True)
    m = LibraNet(NetConfig.from_dict(w["net"]))
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in w["model"].items()})
    return m.eval(), int(w["step"]), str(w.get("run_id", ""))


# ---- 学習側: inbox の取り込み ----
class Inbox:
    def __init__(self, path: Path, run_id: str, max_lag_steps: int, log: Callable[[str], None]):
        self.path = path
        self.run_id = run_id
        self.max_lag_steps = max_lag_steps
        self.log = log
        self.stats: dict = {"games": 0, "files": 0, "stale_games": 0, "rejected_files": 0, "by_worker": {}, "last": None}

    def reject(self, p: Path, why: str) -> None:
        self.stats["rejected_files"] += 1
        self.log(f"workers: rejected {p.name}: {why}")
        rej = self.path / "rejected"
        rej.mkdir(exist_ok=True)
        try:
            os.replace(p, rej / p.name)
        except OSError:
            p.unlink(missing_ok=True)
        for old in sorted(rej.iterdir(), key=lambda q: q.stat().st_mtime)[:-KEEP_REJECTED]:
            old.unlink(missing_ok=True)

    def poll(self, step: int, max_files: int = 64) -> list[dict]:
        """届いた対局ファイルを読み、使える局を返す。読んだファイルは消す（不正なものは rejected/ に移す）。"""
        out: list[dict] = []
        now = time.time()
        for p in sorted(self.path.glob("*.npz"))[:max_files]:
            try:
                meta, games = read_games_file(p)
            except GamesFileError as e:
                self.reject(p, str(e))
                continue
            except FileNotFoundError:
                continue
            if meta["run_id"] != self.run_id:
                self.reject(p, f"run_id {meta['run_id']!r} != {self.run_id!r}")
                continue
            lag = step - int(meta["weights_step"])
            p.unlink(missing_ok=True)
            if self.max_lag_steps > 0 and lag > self.max_lag_steps:
                self.stats["stale_games"] += len(games)
                self.log(f"workers: dropped {len(games)} stale games from {meta['worker']} (weights step {meta['weights_step']}, {lag} steps behind)")
                continue
            out += games
            self.stats["files"] += 1
            self.stats["games"] += len(games)
            bw = self.stats["by_worker"]
            bw[meta["worker"]] = bw.get(meta["worker"], 0) + len(games)
            self.stats["last"] = now
        for p in self.path.glob("*.tmp"):  # 落ちたワーカーの書きかけ
            try:
                if now - p.stat().st_mtime > 3600:
                    p.unlink()
            except FileNotFoundError:
                pass
        return out


# ---- ワーカー ----
def worker_seed(worker_id: str, entropy: int) -> int:
    """同じ設定から起動したワーカーが同じ対局列を打たないように、id と起動ごとの乱数から seed を作る。"""
    ss = np.random.SeedSequence([int(entropy), *worker_id.encode("utf-8")])
    return int(ss.generate_state(1, np.uint64)[0] >> np.uint64(1))


class Worker:
    """自己対局だけを回す。lock を渡すと学習側（run.lock の持ち主）が動いている間だけ打ち、止まったら抜ける（同じマシンのワーカー）。
    stop_root の STOP でも抜ける（学習側が動いていないときに残った STOP は無視する）。別マシンでは lock と stop_root を渡さない。"""

    def __init__(self, cfg: dict, worker_id: str, weights: Path, inbox: Path, device: torch.device, *, n_games: int | None = None,
                 threads: int | None = None, stop_root: Path | None = None, lock: Path | None = None, entropy: int | None = None,
                 poll_seconds: float = 2.0, reload_seconds: float = 10.0, lock_seconds: float = 5.0, perf_seconds: float = 60.0,
                 log: Callable[[str], None] = print):
        if not WORKER_ID.match(worker_id):
            raise ValueError(f"worker id must match {WORKER_ID.pattern}: {worker_id!r}")
        self.cfg = cfg
        self.id = worker_id
        self.weights = weights
        self.inbox = inbox
        self.device = device
        self.n_games = n_games or cfg["selfplay"]["n_games"]
        self.threads = threads or cfg["selfplay"]["threads"]
        self.stop_root = stop_root
        self.lock = lock
        self.seed = worker_seed(worker_id, entropy if entropy is not None else int.from_bytes(os.urandom(8), "little"))
        self.poll_seconds = poll_seconds
        self.reload_seconds = reload_seconds
        self.lock_seconds = lock_seconds
        self.perf_seconds = perf_seconds  # 段ごとの時間をログに出す間隔（0 で出さない）
        self.log = log
        self.stopping = False  # シグナルで立てる
        self.step: int | None = None
        self.weights_mtime: int | None = None
        self.files = 0
        self.games = 0
        self._lock_checked = 0.0
        self._learner = True

    def learner_running(self, force: bool = False) -> bool:
        if self.lock is None:
            return True
        now = time.monotonic()
        if force or now - self._lock_checked >= self.lock_seconds:
            from .supervise import running_pid

            self._lock_checked = now
            self._learner = running_pid(self.lock) is not None
        return self._learner

    def stop_flag(self) -> bool:
        return self.stop_root is not None and (self.stop_root / "STOP").exists()

    def _load(self) -> LibraNet | None:
        try:
            mt = self.weights.stat().st_mtime_ns
            if mt == self.weights_mtime:
                return None
            model, step, run_id = load_weights(self.weights)
        except (OSError, RuntimeError, KeyError, EOFError) as e:
            self.log(f"worker {self.id}: cannot read weights {self.weights}: {type(e).__name__}: {str(e)[:200]}")
            return None
        if run_id != self.cfg["run_id"]:
            raise RuntimeError(f"weights are for run {run_id!r}, not {self.cfg['run_id']!r}")
        self.weights_mtime = mt
        if step == self.step:
            return None
        self.step = step
        return model

    def run(self) -> int:
        if self.cfg.get("exploiter", {}).get("main_ckpt"):
            self.log(f"worker {self.id}: exploiter runs are not supported")
            return 2
        from .selfplay import SelfPlayLoop

        waited = False
        while True:  # 重みと学習側を待つ
            if self.stopping:
                return 0
            running = self.learner_running(force=True)
            if running and self.stop_flag():
                self.log(f"worker {self.id}: STOP flag before start: exit")
                return 0
            if running and self.weights.exists():
                break
            if not waited:
                self.log(f"worker {self.id}: waiting for the learner and {self.weights}")
                waited = True
            time.sleep(self.poll_seconds)
        model = None
        while model is None:
            model = self._load()
            if model is None:
                if self.stopping:
                    return 0
                time.sleep(self.poll_seconds)
        sp = self.cfg["selfplay"]
        loop = SelfPlayLoop(self.cfg["search"], self.n_games, self.threads, self.seed, self.device, sp["infer_dtype"], sp.get("compile", "none"))
        loop.set_model(model)
        del model
        self.log(f"worker {self.id}: start weights step {self.step} n_games {self.n_games} threads {self.threads} device {self.device} seed {self.seed}")
        openings = OpeningsReloader(sp, loop, self.log)
        chunk = int(self.cfg["run"]["chunk_games"])
        pending: list[dict] = []
        file_step = self.step
        last_reload = time.monotonic()
        last_perf = time.monotonic()
        if self.perf_seconds > 0:
            loop.timing = {}
        modes = ""
        reason = ""
        while True:
            if self.stopping:
                reason = "signal"
            elif self.stop_flag():
                reason = "STOP flag"
            elif not self.learner_running():
                reason = "learner not running"
            if reason:
                break
            pending += loop.round()
            if loop.model is not None and loop.model.mode_used != modes:
                modes = loop.model.mode_used
                self.log(f"worker {self.id}: inference {modes}")
            while len(pending) >= chunk:
                self._flush(pending[:chunk], file_step)
                pending = pending[chunk:]
                file_step = self.step
            now = time.monotonic()
            if loop.timing is not None and now - last_perf >= self.perf_seconds and loop.timing.get("rounds"):
                tm, dt = loop.timing, now - last_perf
                r = tm["rounds"]
                self.log(f"worker {self.id}: perf rounds/s {r / dt:.1f} evals/s {r * self.n_games / dt:.0f} per round: "
                         f"collect {tm['collect'] / r * 1000:.1f} ms eval {tm['eval'] / r * 1000:.1f} ms "
                         f"proof {tm.get('proof', 0.0) / r * 1000:.1f} ms apply {tm['apply'] / r * 1000:.1f} ms "
                         f"other {max(0.0, dt / r - (tm['collect'] + tm['eval'] + tm.get('proof', 0.0) + tm['apply']) / r) * 1000:.1f} ms "
                         f"games {self.games + len(pending)}")
                loop.timing = {}
                last_perf = now
            if now - last_reload >= self.reload_seconds:
                last_reload = now
                m = self._load()
                if m is not None:
                    loop.set_model(m)
                    file_step = min(int(file_step), int(self.step))  # 途中まで古い重みで打った局を含む
                    self.log(f"worker {self.id}: weights step {self.step}")
                openings.poll()
        if pending:
            self._flush(pending, file_step)
        self.log(f"worker {self.id}: {reason}: exit after {self.games} games in {self.files} files")
        loop.release()
        return 0

    def _flush(self, games: list[dict], step: int | None) -> None:
        write_games_file(self.inbox, self.id, int(step or 0), self.cfg["run_id"], games)
        self.files += 1
        self.games += len(games)


class OpeningsReloader:
    """cfg.selfplay.openings が更新されていればエンジンに渡す（Runner.reload_openings と同じ規則）。"""

    def __init__(self, sp: dict, loop, log: Callable[[str], None]):
        self.sp = sp
        self.loop = loop
        self.log = log
        self.mtime: float | None = None
        self.checked = 0.0
        self.poll(force=True)

    def poll(self, force: bool = False) -> None:
        path = self.sp.get("openings") or ""
        now = time.monotonic()
        if not path or (not force and now - self.checked < self.sp.get("openings_reload_seconds", 600)):
            return
        self.checked = now
        p = Path(path).expanduser()
        if not p.exists() or p.stat().st_mtime == self.mtime:
            return
        from .openings import load_openings

        ops = load_openings(p)
        self.loop.engine.set_openings(ops, float(self.sp.get("openings_prob", 0.0)))
        self.mtime = p.stat().st_mtime
        self.log(f"openings: {len(ops)} lines from {p} (prob {self.sp.get('openings_prob', 0.0)})")
