# SPDX-License-Identifier: Apache-2.0
"""単一・フェーズ条件付き Transformer（docs/libra-design.md §3.2 をローカル向けに縮めたもの）。

入力: 81 マスのトークン（libra-sim の SQ_FEATS 次元）＋グローバルトークン（GLOB_FEATS 次元）。
幹: エンコーダ層 × L、d、2D 相対位置バイアス（筋差・段差ごとの学習パラメータ、ヘッド別）。
ヘッド: 方策（81 × 28 = 2268、マスごとのトークンから）、価値 WDL（グローバルトークンから）、
       補助 V̂41（布石局面から 41 手目の探索値を予測、WDL と同じ 3 値）。
       opp_head なら補助の相手の次の手（KataGo [Wu19] §3.4）、own_head なら補助の「盤上の駒が最後まで残るか」（同 §4.1 の
       陣地の予測の将棋版）。どちらも学習だけで使い、forward と ONNX には出さない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SQ_NB = 81
SQ_FEATS = 32
GLOB_FEATS = 32
POLICY_CLASSES = 28
POLICY_SIZE = SQ_NB * POLICY_CLASSES


@dataclass
class NetConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.0
    opp_head: bool = False  # 補助方策「相手の次の手」の頭を持つか。幹の形は変わらないので、無い重みから引き継げる
    own_head: bool = False  # 補助「盤上の駒が最後まで残るか」（マスごとに 1 値）の頭を持つか。同上

    @classmethod
    def from_dict(cls, d: dict) -> "NetConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RelPosBias(nn.Module):
    """2D 相対位置バイアス。81 マス同士は (筋差, 段差) ∈ [-8,8]^2 の 289 通り、グローバルトークンとの間は別の 1 値。"""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.table = nn.Parameter(torch.zeros(n_heads, 17 * 17 + 3))
        idx = torch.zeros(SQ_NB + 1, SQ_NB + 1, dtype=torch.long)
        for a in range(SQ_NB):
            fa, ra = divmod(a, 9)
            for b in range(SQ_NB):
                fb, rb = divmod(b, 9)
                idx[a, b] = (fa - fb + 8) * 17 + (ra - rb + 8)
        idx[SQ_NB, :SQ_NB] = 17 * 17
        idx[:SQ_NB, SQ_NB] = 17 * 17 + 1
        idx[SQ_NB, SQ_NB] = 17 * 17 + 2
        self.register_buffer("idx", idx, persistent=False)

    def forward(self) -> torch.Tensor:  # [1, H, T, T]
        return self.table[:, self.idx].unsqueeze(0)


class Block(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))
        self.n_heads = cfg.n_heads
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        x = x + self.ff(self.ln2(x))
        return x


class LibraNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        cfg = cfg or NetConfig()
        self.cfg = cfg
        d = cfg.d_model
        self.sq_embed = nn.Linear(SQ_FEATS, d)
        self.pos_embed = nn.Parameter(torch.randn(SQ_NB, d) * 0.02)
        self.glob_embed = nn.Linear(GLOB_FEATS, d)
        self.glob_pos = nn.Parameter(torch.randn(1, d) * 0.02)
        self.rel = RelPosBias(cfg.n_heads)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.policy_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, POLICY_CLASSES))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        self.v41_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        if cfg.opp_head:  # 最後に作る（パラメータの並びの末尾に来るので、頭の無い AdamW の状態を引き継げる。Trainer.load_state_dict）
            self.opp_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, POLICY_CLASSES))
        if cfg.own_head:  # 同上（opp_head の後ろ）
            self.own_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, sq: torch.Tensor, glob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """sq: [B, 81, SQ_FEATS]、glob: [B, GLOB_FEATS] → (policy logits [B, 2268], wdl logits [B, 3], v41 logits [B, 3])"""
        return self._heads(self._trunk(sq, glob))

    def forward_aux(self, sq: torch.Tensor, glob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """学習用: forward に補助の頭の出力を足す。"opp" は相手の次の手の logits [B, 2268]、"own" は駒が残るかの logits [B, 81]
        （持っている頭だけ）。"""
        x = self._trunk(sq, glob)
        policy, wdl, v41 = self._heads(x)
        aux = {}
        if self.cfg.opp_head:
            aux["opp"] = self.opp_head(x[:, :SQ_NB]).reshape(x.shape[0], POLICY_SIZE)
        if self.cfg.own_head:
            aux["own"] = self.own_head(x[:, :SQ_NB]).squeeze(-1)
        return policy, wdl, v41, aux

    def _trunk(self, sq: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        x_sq = self.sq_embed(sq) + self.pos_embed
        x_g = (self.glob_embed(glob) + self.glob_pos).unsqueeze(1)
        x = torch.cat([x_sq, x_g], dim=1)
        bias = self.rel().to(x.dtype)
        for blk in self.blocks:
            x = blk(x, bias)
        return self.ln_f(x)

    def _heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy = self.policy_head(x[:, :SQ_NB]).reshape(x.shape[0], POLICY_SIZE)
        g = x[:, SQ_NB]
        return policy, self.value_head(g), self.v41_head(g)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def wdl_to_value(wdl_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(wdl_logits.float(), dim=-1)
    return p[:, 0] - p[:, 2]
