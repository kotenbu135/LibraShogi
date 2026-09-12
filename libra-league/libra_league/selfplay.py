# SPDX-License-Identifier: Apache-2.0
"""自己対局ループ: C++ エンジンが葉を集め、PyTorch がバッチ評価し、エンジンが進める。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import librasearch
import librashogi as ls
from libra_net.model import LibraNet


class SelfPlayLoop:
    def __init__(self, search_cfg: dict, n_games: int, threads: int, seed: int, device: torch.device, infer_dtype: str = "float16"):
        self.engine = librasearch.SelfPlay(search_cfg, n_games, seed, threads)
        self.n_games = n_games
        self.device = device
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[infer_dtype]
        pin = device.type == "cuda"
        self.sq = torch.empty((n_games, 81, ls.SQ_FEATS), dtype=torch.float32, pin_memory=pin)
        self.glob = torch.empty((n_games, ls.GLOB_FEATS), dtype=torch.float32, pin_memory=pin)
        self.sq_np = self.sq.numpy()
        self.glob_np = self.glob.numpy()
        self.model: LibraNet | None = None
        self.opponent: LibraNet | None = None  # 搾取者モード: 凍結した本体。奇数枠では本体が先手

    def set_model(self, model: LibraNet) -> None:
        """学習中のモデルから推論用の写しを作る（半精度・eval）。"""
        import copy

        m = copy.deepcopy(model).to(self.device).eval()
        if self.dtype != torch.float32:
            m = m.to(self.dtype)
        for p in m.parameters():
            p.requires_grad_(False)
        self.model = m

    def set_opponent(self, model: LibraNet | None) -> None:
        if model is None:
            self.opponent = None
            return
        m = model.to(self.device).eval()
        if self.dtype != torch.float32:
            m = m.to(self.dtype)
        for p in m.parameters():
            p.requires_grad_(False)
        self.opponent = m

    @staticmethod
    def exploiter_is_sente(slot: int) -> bool:
        return slot % 2 == 0

    @torch.no_grad()
    def round(self) -> list[dict]:
        assert self.model is not None
        self.engine.collect(self.sq_np, self.glob_np)
        sq = self.sq.to(self.device, non_blocking=True).to(self.dtype)
        glob = self.glob.to(self.device, non_blocking=True).to(self.dtype)
        if self.opponent is None:
            policy, wdl, _ = self.model(sq, glob)
            logits = policy.float().cpu().numpy()
            wdl_p = F.softmax(wdl.float(), dim=-1).cpu().numpy()
        else:
            # 手番が搾取者側なら自分のネット、相手側なら凍結した本体（偶数枠は搾取者が先手）
            turns = self.engine.root_turns()  # 0 先手、1 後手
            slot_swap = (np.arange(self.n_games) % 2).astype(np.int8)
            who = turns ^ slot_swap  # 0 なら搾取者
            logits = np.zeros((self.n_games, ls.POLICY_SIZE), np.float32)
            wdl_p = np.zeros((self.n_games, 3), np.float32)
            for k, model in ((0, self.model), (1, self.opponent)):
                idx = np.flatnonzero(who == k)
                if idx.size == 0:
                    continue
                it = torch.from_numpy(idx).to(self.device)
                p, w, _ = model(sq[it], glob[it])
                logits[idx] = p.float().cpu().numpy()
                wdl_p[idx] = F.softmax(w.float(), dim=-1).cpu().numpy()
        self.engine.apply(np.ascontiguousarray(logits), np.ascontiguousarray(wdl_p))
        games = self.engine.take_finished()
        if self.opponent is not None:
            for g in games:
                mask_opponent_moves(g, self.exploiter_is_sente(int(g["slot"])))
        return games

    def stats(self) -> dict:
        return self.engine.stats()


def mask_opponent_moves(g: dict, exploiter_is_sente: bool) -> dict:
    """搾取者の記録: 相手（本体）の手は方策ターゲットにしない（full=0）。3 手目（添字 0）は先手の手。"""
    full = np.asarray(g["full"]).copy()
    n = len(full)
    idx = np.arange(n)
    sente_move = idx % 2 == 0
    full[sente_move != exploiter_is_sente] = 0
    g["full"] = full
    g["exploiter_side"] = "sente" if exploiter_is_sente else "gote"
    r = int(g["result"])
    g["exploiter_result"] = r if exploiter_is_sente else -r
    return g
