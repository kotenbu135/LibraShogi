# SPDX-License-Identifier: Apache-2.0
"""リプレイ: 対局記録（libra-search の GameRecord）を窓に保持し、学習バッチを作る。チャンクは 100 局単位で追記。

窓の大きさ（docs/restart-plan.md §4、decisions.md 2026-09-17）: `window_games` は最小の窓で、`window_frac` > 0 なら総局数 × window_frac
まで広げる（上限 `window_games_max`、0 で無制限）。KataGo [Wu19] が総数に応じて窓を広げるのに倣う。
held-out（同 §3 M1）: `heldout_every_chunks` > 0 なら、チャンク番号がその倍数のチャンクの局は学習に使わず `heldout` に持つ
（新しい `heldout_games` 局まで）。窓の中と held-out で同じ物差し（genprof.py）を測り、差が「窓の記憶」の量になる。
"""
from __future__ import annotations

import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np

import librashogi as ls

MIRROR_TABLE = np.array([ls.mirror_index(i) for i in range(ls.POLICY_SIZE)], dtype=np.int64)


def game_to_jsonl(g: dict) -> str:
    """公開用の棋譜行（CC0）。玉 2 手、choose は自己対局では置く側が自動なので省く。"""
    tokens = [f"K*{ls.sq_to_usi(int(g['kb']))}", f"K*{ls.sq_to_usi(int(g['kw']))}"] + [ls.move_to_usi(int(m)) for m in g["moves"]]
    return json.dumps(
        {
            "tokens": " ".join(tokens),
            "result": {1: "sente", -1: "gote", 0: "draw"}[int(g["result"])],
            "reason": g["reason"],
            "plies": int(g["plies"]),
            "sfen41": g["sfen41"],
            "v41": round(float(g["v41"]), 4),
            **({"exploiter": g["exploiter_side"]} if "exploiter_side" in g else {}),
            **({"league": {"opponent_step": int(g["league_opponent"]), "main": g["league_main_side"]}} if "league_opponent" in g else {}),
        },
        ensure_ascii=False,
    )


def soft_wdl(t: np.ndarray) -> np.ndarray:
    """期待値 t ∈ [−1, 1] を (勝, 分, 負) の分布に。t=±1 は one-hot、0 は引き分け。"""
    w = np.clip(t, 0, 1)
    l = np.clip(-t, 0, 1)
    return np.stack([w, 1 - w - l, l], axis=1).astype(np.float32)


def add_target_stats(acc: dict, s: dict) -> None:
    """sample の target_stats（和と局面数）を学習 1 回ぶん足し合わせる。"""
    for k, v in s.items():
        acc[k] = acc.get(k, 0.0) + float(v)


def summarize_target_stats(acc: dict) -> dict | None:
    """学習目標・v41・探索値と実際の結果の差を、得点の尺度（値の差 / 2）の平均にする（docs/method-evidence.md §4.4 (a)）。
    target_minus_z・v41_minus_z は布石の局面を先手から見た値、rootq_minus_z_* は手番側から見た値。
    draw_target は布石の目標の引き分けの確率、draw_actual は同じ局面の実際の引き分けの割合。"""
    nf, nn = acc.get("fuseki_n", 0.0), acc.get("normal_n", 0.0)
    if nf + nn <= 0:
        return None

    def mean(k: str, n: float, scale: float = 1.0) -> float | None:
        return round(acc[k] / n * scale, 4) if n > 0 else None

    return {
        "fuseki_n": int(nf), "normal_n": int(nn),
        "target_minus_z": mean("fuseki_t_minus_z", nf, 0.5), "v41_minus_z": mean("fuseki_v41_minus_z", nf, 0.5),
        "draw_target": mean("fuseki_draw_target", nf), "draw_actual": mean("fuseki_draw_actual", nf),
        "rootq_minus_z_fuseki": mean("fuseki_rootq_minus_z", nf, 0.5), "rootq_minus_z_normal": mean("normal_rootq_minus_z", nn, 0.5),
    }


class ReplayBuffer:
    def __init__(self, replay_dir: Path, games_dir: Path, window_games: int, chunk_games: int, max_ply: int, count_from_41: bool,
                 window_frac: float = 0.0, window_games_max: int = 0, heldout_every_chunks: int = 0, heldout_games: int = 20000):
        self.replay_dir = replay_dir
        self.games_dir = games_dir
        self.window_games = int(window_games)
        self.window_frac = float(window_frac)
        self.window_games_max = int(window_games_max)
        self.heldout_every_chunks = int(heldout_every_chunks)
        self.heldout_games = int(heldout_games)
        self.chunk_games = chunk_games
        self.max_ply = max_ply
        self.count_from_41 = count_from_41
        # 学習に使う局（古い → 新しい）。窓より古い分は _trim でまとめて落とす（1 局ずつ popleft しない: 100 万局でも足す・数えるが O(1)）
        self.games: list[dict] = []
        self._lens: list[int] = []
        self.heldout: list[dict] = []
        self.pending: list[dict] = []
        self.chunk_index = 0
        self.total_games = 0
        self._cum: np.ndarray | None = None  # sample 用の局面数の累積和（窓の分だけ）。games が変わったら捨てる

    # ---- 窓の大きさ ----
    def window(self) -> int:
        """今の窓の大きさ（局）。"""
        w = self.window_games
        if self.window_frac > 0:
            w = max(w, int(self.total_games * self.window_frac))
        if self.window_games_max > 0:
            w = min(w, self.window_games_max)
        return max(1, w)

    def is_heldout_chunk(self, index: int) -> bool:
        return self.heldout_every_chunks > 0 and index % self.heldout_every_chunks == 0

    # ---- 永続化 ----
    def load(self, chunk_index: int, total_games: int) -> None:
        """索引にあるチャンクのうち、窓（と held-out）に入る分だけ新しい側から読む。"""
        self.chunk_index = chunk_index
        self.total_games = total_games
        w = self.window()
        loaded: list[dict] = []
        held: list[dict] = []
        for i in range(chunk_index - 1, -1, -1):
            if len(loaded) >= w and (self.heldout_every_chunks <= 0 or len(held) >= self.heldout_games):
                break
            p = self.replay_dir / f"chunk_{i:06d}.pkl"
            if not p.exists():
                continue
            dst = held if self.is_heldout_chunk(i) else loaded
            if dst is held and len(held) >= self.heldout_games:
                continue
            if dst is loaded and len(loaded) >= w:
                continue
            with open(p, "rb") as f:
                dst.extend(reversed(pickle.load(f)))
        loaded.reverse()
        held.reverse()
        self.games = loaded[-w:]
        self._lens = [len(g["moves"]) for g in self.games]
        self.heldout = held[-self.heldout_games:] if self.heldout_games > 0 else []
        self._cum = None

    def add_games(self, games: list[dict]) -> int:
        """終局した記録を足す。chunk_games 局たまるごとにチャンクと棋譜 JSONL を書き、書いたチャンク数を返す。
        held-out のチャンクに入る局は学習の窓には入れない（どのチャンクに入るかは pending の位置で決まり、load と同じ振り分けになる）。"""
        written = 0
        for g in games:
            index = self.chunk_index + len(self.pending) // self.chunk_games
            self.pending.append(g)
            self.total_games += 1
            if self.is_heldout_chunk(index):
                self.heldout.append(g)
            else:
                self.games.append(g)
                self._lens.append(len(g["moves"]))
        self._cum = None
        self._trim()
        while len(self.pending) >= self.chunk_games:
            chunk, self.pending = self.pending[: self.chunk_games], self.pending[self.chunk_games :]
            self._write_chunk(chunk)
            written += 1
        return written

    def _trim(self) -> None:
        w = self.window()
        if len(self.games) > w + max(self.chunk_games, w // 20):  # 5% 超えたらまとめて落とす
            k = len(self.games) - w
            del self.games[:k]
            del self._lens[:k]
        if self.heldout_games > 0 and len(self.heldout) > self.heldout_games + max(self.chunk_games, self.heldout_games // 20):
            del self.heldout[: len(self.heldout) - self.heldout_games]

    def _write_chunk(self, chunk: list[dict]) -> None:
        p = self.replay_dir / f"chunk_{self.chunk_index:06d}.pkl"
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)
        j = self.games_dir / f"games_{self.chunk_index:06d}.jsonl"
        tmpj = j.with_suffix(".tmp")
        with open(tmpj, "w", encoding="utf-8") as f:
            for g in chunk:
                f.write(game_to_jsonl(g) + "\n")
        tmpj.replace(j)
        self.chunk_index += 1

    def _start(self) -> int:
        return max(0, len(self.games) - self.window())

    def n_games(self) -> int:
        return len(self.games) - self._start()

    def n_heldout(self) -> int:
        return min(len(self.heldout), self.heldout_games) if self.heldout_games > 0 else 0

    def n_positions(self) -> int:
        return sum(self._lens[self._start():])

    def window_games_list(self) -> list[dict]:
        """窓の中の局（古い → 新しい）。"""
        return self.games[self._start():]

    def heldout_games_list(self) -> list[dict]:
        return self.heldout[-self.heldout_games:] if self.heldout_games > 0 else []

    # ---- サンプリング ----
    def sample(self, batch: int, rng: np.random.Generator, mirror_prob: float, lambda_z: float, topk: int = 32) -> dict:
        start = self._start()
        if self._cum is None or len(self._cum) != len(self.games) - start:
            self._cum = np.cumsum(np.array(self._lens[start:], dtype=np.int64))
        return sample_batch(self.games, start, self._cum, batch, rng, mirror_prob, lambda_z, topk, self.max_ply, self.count_from_41)

    def sample_heldout(self, batch: int, rng: np.random.Generator, mirror_prob: float, lambda_z: float, topk: int = 32) -> dict:
        held = self.heldout_games_list()
        cum = np.cumsum(np.array([len(g["moves"]) for g in held], dtype=np.int64))
        return sample_batch(held, 0, cum, batch, rng, mirror_prob, lambda_z, topk, self.max_ply, self.count_from_41)


def sample_batch(games: list[dict], start: int, cum: np.ndarray, batch: int, rng: np.random.Generator, mirror_prob: float, lambda_z: float,
                 topk: int, max_ply: int, count_from_41: bool) -> dict:
    """games[start:] の全局面から一様に batch 局面を取り、学習バッチを作る。cum は games[start:] の局面数の累積和。"""
    if len(cum) == 0 or cum[-1] <= 0:
        raise ValueError("sample_batch: no positions")
    pick = rng.integers(0, cum[-1], size=batch)
    gi = np.searchsorted(cum, pick, side="right")
    lens = np.diff(np.concatenate([[0], cum]))
    mi = pick - (cum[gi] - lens[gi])
    gi = gi + start
    kb = np.array([games[i]["kb"] for i in gi], np.int32)
    kw = np.array([games[i]["kw"] for i in gi], np.int32)
    plies = (mi + 2).astype(np.int32)
    mirror = (rng.random(batch) < mirror_prob).astype(np.uint8)
    sq = np.empty((batch, 81, ls.SQ_FEATS), np.float32)
    glob = np.empty((batch, ls.GLOB_FEATS), np.float32)
    side = np.empty(batch, np.uint8)
    fuseki = np.empty(batch, np.uint8)
    ls.replay_features(kb, kw, [games[i]["moves"] for i in gi], plies, mirror, sq, glob, side, fuseki, max_ply, count_from_41)
    sign = np.where(side == 1, 1.0, -1.0).astype(np.float32)
    z = np.array([games[i]["result"] for i in gi], np.float32) * sign
    v41 = np.array([games[i]["v41"] for i in gi], np.float32) * sign
    fu = fuseki.astype(bool)
    t = np.where(fu, lambda_z * z + (1 - lambda_z) * v41, z).astype(np.float32)
    wdl_t = soft_wdl(t)
    # 学習目標と実際の結果の差（和）。z・t・v41 は手番側、sign を掛けると先手から見た値
    rq = np.array([games[i]["root_q"][m] for i, m in zip(gi, mi)], np.float32)
    nu = ~fu
    target_stats = {
        "fuseki_n": int(fu.sum()), "normal_n": int(nu.sum()),
        "fuseki_t_minus_z": float(((t - z) * sign)[fu].sum()), "fuseki_v41_minus_z": float(((v41 - z) * sign)[fu].sum()),
        "fuseki_draw_target": float(wdl_t[fu, 1].sum()), "fuseki_draw_actual": float((z[fu] == 0).sum()),
        "fuseki_rootq_minus_z": float((rq - z)[fu].sum()), "normal_rootq_minus_z": float((rq - z)[nu].sum()),
    }
    pidx = np.full((batch, topk), -1, np.int64)
    pp = np.zeros((batch, topk), np.float32)
    valid = np.zeros(batch, bool)
    for b in range(batch):
        g = games[gi[b]]
        j = mi[b]
        if not g["full"][j]:
            continue
        o0, o1 = g["policy_off"][j], g["policy_off"][j + 1]
        k = min(topk, o1 - o0)
        if k <= 0:
            continue
        idx = g["policy_idx"][o0 : o0 + k].astype(np.int64)
        if mirror[b]:
            idx = MIRROR_TABLE[idx]
        pidx[b, :k] = idx
        pp[b, :k] = g["policy_p"][o0 : o0 + k]
        valid[b] = True
    return {
        "sq": sq,
        "glob": glob,
        "wdl": wdl_t,
        "v41": soft_wdl(v41),
        "z": z,
        "target_stats": target_stats,
        "fuseki": fu,
        "policy_idx": pidx,
        "policy_p": pp,
        "policy_valid": valid,
    }
