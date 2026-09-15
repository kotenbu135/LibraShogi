# SPDX-License-Identifier: Apache-2.0
"""逐次の検証対局の実行（`libra-scale seq`。規則は seqrule.py）。状態はディレクトリ 1 つに置く:

  config.json   規則・探索の設定・対象の組・モデル（初回に書く。置く側の精度だけ再開時に変えられる）
  state.json    組ごとの局数と打ち切り、取り込んだ棋譜ファイル、アラート
  status.json   進み具合（30 秒ごと）
  log.txt       打ち切り・段階・アラート（`ALERT` 行）  ALERT.txt  アラートだけ
  active.json   打ち切っていない組（ワーカーが読む）
  inbox/        ワーカーが置く棋譜（100 局ごとの *.jsonl.gz。1 局 1 行: kb, kw, result, reason, plies, moves, worker）
  games/        取り込んだ棋譜（CC0）
  STOP          置くと止まる（次の run が消す）

局は自己対局エンジン（SelfPlayLoop: compile＋CUDA Graphs、評価のキャッシュ、根の証明探索の先送り。棋譜は変わらない）で打ち、
打ち切っていない組の玉 2 手を布石として確率 1 で渡す（対局の合間に差し替えられるので C++ を変えずに組を絞れる）。
"""
from __future__ import annotations

import gzip
import json
import math
import os
import secrets
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

import librashogi as ls
from libra_league.config import DEFAULTS

from . import seqrule as R
from .pairs import from_usi, unique_pairs, usi

CHUNK_GAMES = 100
CYCLE_S = 10.0
FLUSH_S = 120.0
STATUS_S = 30.0


def now_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def read_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p: Path, obj) -> None:
    p = Path(p)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def search_config(sims: int, overrides: dict | None = None) -> dict:
    """2026-09-11 の verify と同じ探索（全手を読み sims 回、方策の記録は上位 8 手）。"""
    cfg = dict(DEFAULTS["search"])
    cfg.update({"full_prob": 1.0, "full_sims": sims, "policy_topk": 8})
    cfg.update(overrides or {})
    return cfg


def parse_pairs(spec: str) -> list[str]:
    """`5i5a,5h5b` → 代表の組の鍵。剪定後の組でなければ ValueError。"""
    allk = {R.key_of(a, b) for a, b in unique_pairs()}
    out = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if len(tok) != 4:
            raise ValueError(f"pair must look like 5i5a: {tok}")
        k = R.key_of(from_usi(tok[:2]), from_usi(tok[2:]))
        if k not in allk:
            raise ValueError(f"not a pruned pair: {tok}")
        out.append(k)
    return sorted(set(out))


def init_dir(d: Path, base_table: Path, sims: int, rule: R.Rule, keys: list[str] | None = None, model: Path | None = None,
             search_overrides: dict | None = None) -> dict:
    d = Path(d)
    if (d / "config.json").exists():
        raise FileExistsError(f"{d}/config.json exists")
    (d / "inbox").mkdir(parents=True, exist_ok=True)
    (d / "games").mkdir(exist_ok=True)
    base = read_json(base_table)
    keys = sorted(set(keys)) if keys else [R.key_of(a, b) for a, b in unique_pairs()]
    cfg = {"created_at": now_str(), "base_table": str(base_table), "model": str(model or base["model"]["path"]),
           "model_step": base["model"].get("step"), "sims": sims, "search": search_config(sims, search_overrides),
           "rule": asdict(rule), "pairs": keys}
    write_json(d / "config.json", cfg)
    write_json(d / "state.json", {"pairs": {k: R.new_state(rule) for k in keys}, "ingested": [], "alerts": [], "fired": [],
                                  "games": 0, "workers": {}, "phase": None})
    return cfg


def record_from_game(g: dict, worker: str) -> dict:
    return {"kb": usi(int(g["kb"])), "kw": usi(int(g["kw"])), "result": int(g["result"]), "reason": str(g["reason"]),
            "plies": int(g["plies"]), "moves": " ".join(ls.move_to_usi(int(m)) for m in g["moves"]), "worker": worker}


def write_chunk(inbox: Path, worker: str, seq: int, records: list[dict]) -> Path:
    name = f"{worker}-{int(time.time() * 1000)}-{seq:06d}.jsonl.gz"
    tmp = Path(inbox) / (name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, Path(inbox) / name)
    return Path(inbox) / name


def read_chunk(p: Path) -> list[dict]:
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_RESULT = {"sente": 1, "gote": -1, "draw": 0}


def check_record(r: dict, search: dict) -> None:
    """1 局を玉の配置から再生し、全手の合法性と終局（result・reason・plies）を確かめる（別マシンの局を取り込む前）。合わなければ ValueError。
    手数の上限（max_moves_per_game）で打ち切った局は、終局していなくても reason が timeout で手番側の負け（selfplay.cpp の安全弁）。"""
    p = ls.Position()
    p.set_max_ply(int(search["max_ply"]), bool(search["count_from_41"]))
    for k in ("kb", "kw"):
        u = "K*" + str(r[k])
        if not p.is_legal(u):
            raise ValueError(f"{k}: illegal king placement {u}")
        p.do_move(u)
    moves = str(r["moves"]).split()
    for j, u in enumerate(moves):
        if p.is_over():
            raise ValueError(f"move {j}: after the end")
        if not p.is_legal(u):
            raise ValueError(f"move {j}: illegal {u}")
        p.do_move(u)
    res, reason = p.outcome()
    if res == "ongoing":
        if not (len(moves) >= int(search["max_moves_per_game"]) and r["reason"] == "timeout"
                and r["result"] == (-1 if p.turn == "sente" else 1)):
            raise ValueError(f"unfinished game ({len(moves)} moves, {r['reason']})")
    elif r["reason"] != reason or r["result"] != _RESULT.get(res):
        raise ValueError(f"outcome {res}/{reason} != {r['result']}/{r['reason']}")
    if int(r["plies"]) != p.ply:
        raise ValueError(f"plies {r['plies']} != {p.ply}")


class Coordinator:
    """inbox の棋譜を取り込み、規則を当て、アラート・active.json・status.json を書く。"""

    def __init__(self, d: Path, log: Callable[[str], None] = print):
        self.d = Path(d)
        self.log = log
        self.config = read_json(self.d / "config.json")
        self.state = read_json(self.d / "state.json")
        self.rule = R.Rule.from_dict(self.config["rule"])
        self.groups = R.symmetric_groups()
        self.symmetric = {k for ks in self.groups.values() for k in ks}
        self.rate: deque[tuple[float, int]] = deque(maxlen=200)

    @property
    def states(self) -> dict[str, dict]:
        return self.state["pairs"]

    def set_eps_place(self, eps_place: float) -> None:
        if float(self.config["rule"].get("eps_place", 0.0)) != eps_place:
            self.log(f"seq: eps_place {self.config['rule'].get('eps_place')} -> {eps_place}")
            self.config["rule"]["eps_place"] = eps_place
            write_json(self.d / "config.json", self.config)
            self.rule = R.Rule.from_dict(self.config["rule"])

    def save(self) -> None:
        write_json(self.d / "state.json", self.state)

    def ingest(self) -> int:
        """inbox の *.jsonl.gz を数える。状態を書いてから games/ へ移す（間で落ちても数え直さない）。"""
        inbox, games = self.d / "inbox", self.d / "games"
        games.mkdir(exist_ok=True)
        done = set(self.state["ingested"])
        new, n, ignored = [], 0, 0
        for f in sorted(inbox.glob("*.jsonl.gz")):
            if f.name in done:
                os.replace(f, games / f.name)
                continue
            try:
                recs = read_chunk(f)
            except (OSError, EOFError, ValueError) as e:  # 壊れたファイルは脇へ
                (self.d / "rejected").mkdir(exist_ok=True)
                os.replace(f, self.d / "rejected" / f.name)
                self.log(f"seq: rejected {f.name}: {type(e).__name__}: {str(e)[:120]}")
                continue
            for r in recs:
                st = self.states.get(R.key_of(from_usi(r["kb"]), from_usi(r["kw"])))
                if st is None:
                    ignored += 1
                    continue
                R.add_result(st, int(r["result"]))
                n += 1
                w = str(r.get("worker", "?"))
                self.state["workers"][w] = self.state["workers"].get(w, 0) + 1
            self.state["ingested"].append(f.name)
            new.append(f)
        if new:
            self.state["games"] += n
            self.save()
            for f in new:
                os.replace(f, games / f.name)
        if ignored:
            self.log(f"seq: ignored {ignored} games of pairs outside the run")
        self.rate.append((time.time(), self.state["games"]))
        return n

    def update(self) -> list[dict]:
        for k, st in self.states.items():
            why = R.update(st, self.rule, k in self.symmetric)
            if why:
                m, se = R.stats(st)
                self.log(f"seq: {k} {why} at {st['n']} games: w {m:.4f} ±{R.Z * se:.4f}")
        alerts = R.check_alerts(self.states, self.groups, self.rule, self.state["fired"])
        for a in alerts:
            self.state["alerts"].append(a)
            self.log("ALERT " + a["message"])
            with open(self.d / "ALERT.txt", "a", encoding="utf-8") as f:
                f.write(f"{a['time']} {a['message']}\n")
        phase = self.phase()
        if phase != self.state.get("phase"):
            if self.state.get("phase") == "symmetric":
                for line in self.symmetric_lines():
                    self.log("seq: " + line)
            self.log(f"seq: phase {phase}")
            self.state["phase"] = phase
        return alerts

    def phase(self) -> str:
        if not self.active():
            return "done"
        return "symmetric" if any(self.states[k]["stop"] is None for k in self.symmetric if k in self.states) else "all"

    def active(self) -> list[str]:
        return R.active_keys(self.states, self.rule, self.symmetric)

    def symmetric_lines(self) -> list[str]:
        out = []
        for g, s in R.symmetric_summary(self.states, self.groups).items():
            p = s["pooled"]
            out.append(f"{s['label']}: {p['games']:,} games w {p['winrate']:.4f} ci {p['ci95']}")
            for row in s["pairs"]:
                out.append(f"  ▲{row['pair'].split()[0]} △{row['pair'].split()[1]}: {row['games']:,} games w {row['winrate']:.4f} "
                           f"ci {row['ci95']} stop {row['stop']}")
        return out

    def write_active(self, keys: list[str]) -> None:
        write_json(self.d / "active.json", {"time": now_str(), "pairs": keys, "done": not keys})

    def write_status(self) -> None:
        stops, place = {}, {}
        for st in self.states.values():
            stops[st["stop"] or "running"] = stops.get(st["stop"] or "running", 0) + 1
            if st.get("place"):
                place[st["place"]] = place.get(st["place"], 0) + 1
        per_min = None
        old = [x for x in self.rate if x[0] >= time.time() - 600]
        if len(old) >= 2 and old[-1][0] > old[0][0]:
            per_min = round((old[-1][1] - old[0][1]) / (old[-1][0] - old[0][0]) * 60, 1)
        write_json(self.d / "status.json", {"time": now_str(), "phase": self.phase(), "games": self.state["games"], "games_per_min": per_min,
                                            "active": len(self.active()), "stops": stops, "place": place, "alerts": len(self.state["alerts"]),
                                            "workers": self.state["workers"], "rule": asdict(self.rule)})


class Player:
    """SelfPlayLoop で局を打つ。set_pairs で打つ組（布石の玉 2 手）を差し替える（対局の合間に呼ぶ）。"""

    def __init__(self, search: dict, keys: list[str], model, n_games: int, threads: int, seed: int, device: torch.device, compile: str):
        from libra_league.selfplay import SelfPlayLoop

        cfg = dict(search, openings=self._codes(keys), openings_prob=1.0)  # 最初の対局から布石で始める
        cuda = device.type == "cuda"
        self.loop = SelfPlayLoop(cfg, n_games, threads, seed, device, "float16" if cuda else "float32", compile if cuda else "none")
        self.loop.set_model(model)

    @staticmethod
    def _codes(keys: list[str]) -> list[list[int]]:
        return [[int(ls.move_from_usi(m)) for m in line] for line in R.opening_lines(keys)]

    def set_pairs(self, keys: list[str]) -> None:
        self.loop.engine.set_openings(self._codes(keys), 1.0)

    def round(self) -> list[dict]:
        return self.loop.round()


def run_local(d: Path, device: torch.device, n_games: int = 512, threads: int = 12, compile: str = "max-autotune", worker: str = "local",
              log: Callable[[str], None] = print, should_stop: Callable[[], bool] = lambda: False, seed: int | None = None,
              chunk_games: int = CHUNK_GAMES, cycle_s: float = CYCLE_S, flush_s: float = FLUSH_S) -> int:
    """手元の GPU で打ちながら取り込みと規則も回す（inbox に別マシンの局があれば一緒に数える）。打つ組が無くなるか STOP で抜ける。"""
    from .table import load_model

    d = Path(d)
    if (d / "STOP").exists():
        (d / "STOP").unlink()
        log("seq: removed a stale STOP")
    co = Coordinator(d, log)
    co.ingest()
    co.update()
    co.save()
    keys = co.active()
    co.write_active(keys)
    co.write_status()
    if not keys:
        log("seq: nothing to play")
        return 0
    model, info = load_model(Path(co.config["model"]), torch.device("cpu"), torch.float32)
    seed = secrets.randbits(62) if seed is None else seed  # 起動ごとに変える（同じ seed で再開すると同じ対局を繰り返す）
    log(f"seq: start {worker} model step {info['step']} seed {seed} n_games {n_games} threads {threads} device {device} "
        f"phase {co.phase()} active {len(keys)} games {co.state['games']}")
    pl = Player(co.config["search"], keys, model, n_games, threads, seed, device, compile)
    buf: list[dict] = []
    seq = 0
    t_flush = t_cycle = t_status = time.time()
    try:
        while True:
            for g in pl.round():
                buf.append(record_from_game(g, worker))
            now = time.time()
            if len(buf) >= chunk_games or (buf and now - t_flush >= flush_s):
                write_chunk(d / "inbox", worker, seq, buf)
                seq, buf, t_flush = seq + 1, [], now
            if now - t_cycle < cycle_s:
                continue
            t_cycle = now
            if co.ingest() or cycle_s == 0:
                co.update()
                co.save()
            new = co.active()
            if new != keys:
                keys = new
                co.write_active(keys)
                if not keys:
                    log("seq: all pairs stopped")
                    break
                pl.set_pairs(keys)
            if now - t_status >= STATUS_S or cycle_s == 0:
                t_status = now
                co.write_status()
            if should_stop() or (d / "STOP").exists():
                log("seq: stop requested")
                break
    finally:
        if buf:
            write_chunk(d / "inbox", worker, seq, buf)
        co.ingest()
        co.update()
        co.save()
        co.write_active(co.active())
        co.write_status()
        log(f"seq: exit games {co.state['games']} phase {co.phase()}")
    return 0


def run_worker(d: Path, device: torch.device, worker: str, n_games: int = 512, threads: int = 12, compile: str = "max-autotune",
               log: Callable[[str], None] = print, should_stop: Callable[[], bool] = lambda: False, seed: int | None = None,
               chunk_games: int = CHUNK_GAMES, flush_s: float = FLUSH_S, poll_s: float = 30.0, max_games: int | None = None) -> int:
    """別マシン（vast.ai）で打つだけ。取り込みと規則は手元の run が行う。d に config.json・weights.pt（fp16）・active.json を置き、
    棋譜を d/inbox に書く。active.json（手元のブリッジが送る）を poll_s ごとに見て打つ組を差し替え、done か STOP か SIGTERM で残りを書いて抜ける。"""
    from libra_league.workers import load_weights

    d = Path(d)
    (d / "inbox").mkdir(parents=True, exist_ok=True)
    cfg = read_json(d / "config.json")
    model, step, _ = load_weights(d / "weights.pt")
    act = read_json(d / "active.json")
    keys = list(act["pairs"])
    if act.get("done") or not keys:
        log("seq worker: nothing to play")
        return 0
    mt = (d / "active.json").stat().st_mtime_ns
    seed = secrets.randbits(62) if seed is None else seed
    log(f"seq worker {worker}: step {step} seed {seed} n_games {n_games} threads {threads} device {device} pairs {len(keys)}")
    pl = Player(cfg["search"], keys, model, n_games, threads, seed, device, compile)
    buf: list[dict] = []
    seq = played = 0
    t_flush = t_poll = time.time()
    try:
        while True:
            for g in pl.round():
                buf.append(record_from_game(g, worker))
            now = time.time()
            if len(buf) >= chunk_games or (buf and now - t_flush >= flush_s):
                write_chunk(d / "inbox", worker, seq, buf)
                seq, played, buf, t_flush = seq + 1, played + len(buf), [], now
            if (max_games is not None and played >= max_games) or should_stop() or (d / "STOP").exists():
                log("seq worker: stop")
                break
            if now - t_poll < poll_s:
                continue
            t_poll = now
            try:
                m2 = (d / "active.json").stat().st_mtime_ns
            except FileNotFoundError:
                continue
            if m2 == mt:
                continue
            mt, act = m2, read_json(d / "active.json")
            if act.get("done") or not act["pairs"]:
                log("seq worker: all pairs stopped")
                break
            if list(act["pairs"]) != keys:
                keys = list(act["pairs"])
                pl.set_pairs(keys)
                log(f"seq worker: {len(keys)} pairs")
    finally:
        if buf:
            write_chunk(d / "inbox", worker, seq, buf)
            played += len(buf)
        log(f"seq worker: exit games {played}")
    return 0


def status_lines(d: Path) -> list[str]:
    co = Coordinator(d, log=lambda s: None)
    st = read_json(d / "status.json") if (d / "status.json").exists() else {}
    out = [f"phase {co.phase()}  games {co.state['games']:,}  per min {st.get('games_per_min')}  active {len(co.active())}  "
           f"rule {asdict(co.rule)}  status at {st.get('time')}"]
    stops, place = {}, {}
    for s in co.states.values():
        stops[s["stop"] or "running"] = stops.get(s["stop"] or "running", 0) + 1
        if s.get("place"):
            place[s["place"]] = place.get(s["place"], 0) + 1
    out.append(f"stops {stops}  place {place}  workers {co.state['workers']}")
    out.append(f"alerts {len(co.state['alerts'])}")
    out += ["  " + a["time"] + " " + a["message"] for a in co.state["alerts"]]
    out += co.symmetric_lines()
    return out


def eta_games(co: Coordinator) -> int:
    """残りの局数の粗い見込み（打ち切っていない組が 0.5 に近いと仮定した上限寄りの値）。"""
    left = 0
    for k, st in co.states.items():
        if st["stop"] is None:
            m, _ = R.stats(st)
            need = min(co.rule.max_games, math.ceil((R.Z * 0.5 / co.rule.eps) ** 2))
            if k not in co.symmetric and st["n"] >= co.rule.min_games and abs(m - 0.5) > 0:
                need = min(need, max(co.rule.min_games, math.ceil((R.Z * 0.5 / abs(m - 0.5)) ** 2)))
            left += max(0, need - st["n"])
    return left
