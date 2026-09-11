# SPDX-License-Identifier: Apache-2.0
"""学習: AdamW、bf16 autocast、方策・価値・V̂41 の損失。"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from libra_net.model import LibraNet


def soft_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    ce = -(logp * target).sum(dim=-1)
    if weight is None:
        return ce.mean()
    return (ce * weight).sum() / weight.sum().clamp(min=1.0)


class Trainer:
    def __init__(self, model: LibraNet, cfg: dict, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            (no_decay if p.ndim < 2 or n.endswith("bias") else decay).append(p)
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-8, fused=device.type == "cuda",
        )
        self.step_count = 0

    def lr_at(self, step: int) -> float:
        w = max(1, self.cfg["warmup_steps"])
        return self.cfg["lr"] * min(1.0, (step + 1) / w)

    def step(self, batch: dict) -> dict:
        m = self.model
        m.train()
        dev = self.device
        sq = torch.from_numpy(batch["sq"]).to(dev, non_blocking=True)
        glob = torch.from_numpy(batch["glob"]).to(dev, non_blocking=True)
        wdl_t = torch.from_numpy(batch["wdl"]).to(dev)
        v41_t = torch.from_numpy(batch["v41"]).to(dev)
        fuseki = torch.from_numpy(batch["fuseki"]).to(dev)
        pidx = torch.from_numpy(batch["policy_idx"]).to(dev)
        pp = torch.from_numpy(batch["policy_p"]).to(dev)
        pvalid = torch.from_numpy(batch["policy_valid"]).to(dev)
        for g in self.opt.param_groups:
            g["lr"] = self.lr_at(self.step_count)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            policy, wdl, v41 = m(sq, glob)
        logp = F.log_softmax(policy.float(), dim=-1)
        gathered = logp.gather(1, pidx.clamp(min=0))
        ce = -(gathered * pp).sum(dim=1)
        pw = pvalid.float()
        l_policy = (ce * pw).sum() / pw.sum().clamp(min=1.0)
        l_value = soft_ce(wdl, wdl_t)
        l_v41 = soft_ce(v41, v41_t, fuseki.float())
        loss = self.cfg["policy_weight"] * l_policy + self.cfg["value_weight"] * l_value + self.cfg["v41_weight"] * l_v41
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), self.cfg["grad_clip"])
        self.opt.step()
        self.step_count += 1
        with torch.no_grad():
            acc = (policy.float().argmax(1) == pidx[:, 0]).float()
            acc = (acc * pw).sum() / pw.sum().clamp(min=1.0)
        return {
            "loss": loss.item(), "policy": l_policy.item(), "value": l_value.item(), "v41": l_v41.item(),
            "policy_acc": acc.item(), "grad_norm": float(gn), "lr": self.lr_at(self.step_count - 1),
        }

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "step": self.step_count}

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd["model"])
        if "opt" in sd:
            self.opt.load_state_dict(sd["opt"])
        self.step_count = int(sd.get("step", 0))
