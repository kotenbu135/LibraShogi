# SPDX-License-Identifier: Apache-2.0
"""損失: 方策（全読み局面だけ、疎な改善方策とのクロスエントロピー）、価値 WDL、補助 V̂41（布石局面だけ）。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def policy_loss(logits: torch.Tensor, target_idx: torch.Tensor, target_p: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """logits [B, P]、target_idx [B, K]（無効は -1）、target_p [B, K]、valid [B]（全読み局面）。"""
    logp = F.log_softmax(logits.float(), dim=-1)
    idx = target_idx.clamp(min=0)
    gathered = logp.gather(1, idx)
    ce = -(gathered * target_p).sum(dim=1)
    w = valid.float()
    return (ce * w).sum() / w.sum().clamp(min=1.0)


def wdl_loss(logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """z ∈ {+1, 0, −1}（手番側）→ クラス {0: 勝, 1: 分, 2: 負}。"""
    cls = (1 - z.long()).clamp(0, 2)
    return F.cross_entropy(logits.float(), cls)


def v41_loss(logits: torch.Tensor, v41: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """v41 ∈ [−1, 1]（手番側の連続値）→ 3 値の分布 (p_win, p_draw, p_loss) = ((1+v)/2 の一部…) ではなく、
    単純に期待値 v に合わせる: E[wdl] = p_w − p_l を v に近づける（MSE）。valid は布石局面。"""
    p = F.softmax(logits.float(), dim=-1)
    v = p[:, 0] - p[:, 2]
    w = valid.float()
    return (((v - v41) ** 2) * w).sum() / w.sum().clamp(min=1.0)
