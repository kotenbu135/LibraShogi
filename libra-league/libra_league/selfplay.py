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

    def set_model(self, model: LibraNet) -> None:
        """学習中のモデルから推論用の写しを作る（半精度・eval）。"""
        import copy

        m = copy.deepcopy(model).to(self.device).eval()
        if self.dtype != torch.float32:
            m = m.to(self.dtype)
        for p in m.parameters():
            p.requires_grad_(False)
        self.model = m

    @torch.no_grad()
    def round(self) -> list[dict]:
        assert self.model is not None
        self.engine.collect(self.sq_np, self.glob_np)
        sq = self.sq.to(self.device, non_blocking=True).to(self.dtype)
        glob = self.glob.to(self.device, non_blocking=True).to(self.dtype)
        policy, wdl, _ = self.model(sq, glob)
        logits = policy.float().cpu().numpy()
        wdl_p = F.softmax(wdl.float(), dim=-1).cpu().numpy()
        self.engine.apply(np.ascontiguousarray(logits), np.ascontiguousarray(wdl_p))
        return self.engine.take_finished()

    def stats(self) -> dict:
        return self.engine.stats()

    def set_active(self, n: int) -> None:
        self.engine.set_active(n)
