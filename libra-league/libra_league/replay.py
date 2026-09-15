# SPDX-License-Identifier: Apache-2.0
"""リプレイ: 対局記録（libra-search の GameRecord）を窓に保持し、学習バッチを作る。チャンクは 100 局単位で追記。"""
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
    def __init__(self, replay_dir: Path, games_dir: Path, window_games: int, chunk_games: int, max_ply: int, count_from_41: bool):
        self.replay_dir = replay_dir
        self.games_dir = games_dir
        self.window_games = window_games
        self.chunk_games = chunk_games
        self.max_ply = max_ply
        self.count_from_41 = count_from_41
        self.games: deque[dict] = deque()
        self.pending: list[dict] = []
        self.chunk_index = 0
        self.total_games = 0
        self._lens_cache: np.ndarray | None = None

    # ---- 永続化 ----
    def load(self, chunk_index: int, total_games: int) -> None:
        """索引にあるチャンクのうち、窓に入る分だけ新しい側から読む。"""
        self.chunk_index = chunk_index
        self.total_games = total_games
        loaded: list[dict] = []
        for i in range(chunk_index - 1, -1, -1):
            p = self.replay_dir / f"chunk_{i:06d}.pkl"
            if not p.exists():
                continue
            with open(p, "rb") as f:
                loaded.extend(reversed(pickle.load(f)))
            if len(loaded) >= self.window_games:
                break
        loaded.reverse()
        self.games = deque(loaded[-self.window_games :])

    def add_games(self, games: list[dict]) -> int:
        """終局した記録を足す。chunk_games 局たまるごとにチャンクと棋譜 JSONL を書き、書いたチャンク数を返す。"""
        written = 0
        for g in games:
            self.pending.append(g)
            self.games.append(g)
            self.total_games += 1
        while len(self.games) > self.window_games:
            self.games.popleft()
        self._lens_cache = None
        while len(self.pending) >= self.chunk_games:
            chunk, self.pending = self.pending[: self.chunk_games], self.pending[self.chunk_games :]
            self._write_chunk(chunk)
            written += 1
        return written

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

    def n_games(self) -> int:
        return len(self.games)

    def n_positions(self) -> int:
        return sum(len(g["moves"]) for g in self.games)

    # ---- サンプリング ----
    def sample(self, batch: int, rng: np.random.Generator, mirror_prob: float, lambda_z: float, topk: int = 32) -> dict:
        games = list(self.games)
        if self._lens_cache is None or len(self._lens_cache) != len(games):
            self._lens_cache = np.array([len(g["moves"]) for g in games], dtype=np.int64)
        lens = self._lens_cache
        cum = np.cumsum(lens)
        pick = rng.integers(0, cum[-1], size=batch)
        gi = np.searchsorted(cum, pick, side="right")
        mi = pick - (cum[gi] - lens[gi])
        kb = np.array([games[i]["kb"] for i in gi], np.int32)
        kw = np.array([games[i]["kw"] for i in gi], np.int32)
        plies = (mi + 2).astype(np.int32)
        mirror = (rng.random(batch) < mirror_prob).astype(np.uint8)
        sq = np.empty((batch, 81, ls.SQ_FEATS), np.float32)
        glob = np.empty((batch, ls.GLOB_FEATS), np.float32)
        side = np.empty(batch, np.uint8)
        fuseki = np.empty(batch, np.uint8)
        ls.replay_features(kb, kw, [games[i]["moves"] for i in gi], plies, mirror, sq, glob, side, fuseki, self.max_ply, self.count_from_41)
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
            "target_stats": target_stats,
            "fuseki": fu,
            "policy_idx": pidx,
            "policy_p": pp,
            "policy_valid": valid,
        }
