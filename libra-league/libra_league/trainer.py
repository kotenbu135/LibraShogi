# SPDX-License-Identifier: Apache-2.0
"""学習: AdamW、bf16 autocast、方策・価値・V̂41 の損失。"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from libra_net.model import LibraNet

COMPILE_MODES = ("none", "default", "max-autotune")


def soft_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    ce = -(logp * target).sum(dim=-1)
    if weight is None:
        return ce.mean()
    return (ce * weight).sum() / weight.sum().clamp(min=1.0)


class Trainer:
    """compile が none 以外で CUDA なら、学習の forward（と AOTAutograd で逆伝播）を torch.compile する。損失・目標・最適化・ステップ数は
    同じで、変わるのはカーネルの選び方による浮動小数点の最下位の桁だけ（GPU 単独の実測で max-autotune は eager の 1.30 倍、
    docs/measurements.md 2026-09-16）。max-autotune は推論と同じく CUDA Graphs を使わない形にする。最初のステップで失敗したら eager に落として続ける。
    self.model は compile しない（state_dict の鍵・推論用の写し・ONNX の書き出しはそのまま使う）。"""

    def __init__(self, model: LibraNet, cfg: dict, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        mode = cfg.get("compile", "none")
        if mode not in COMPILE_MODES:
            raise ValueError(f"train.compile must be one of {COMPILE_MODES}: {mode!r}")
        self.forward = model
        self.mode_used = "eager"
        self._fallback_ok = False  # 1 ステップでも compile で通ったら、それ以降の例外は落とさずに上げる
        if device.type == "cuda" and mode != "none":
            from .selfplay import _quiet_inductor

            _quiet_inductor()
            self.forward = torch.compile(model, mode="max-autotune-no-cudagraphs" if mode == "max-autotune" else "default", fullgraph=True, dynamic=False)
            self.mode_used = f"compile({mode})"
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
        if self.forward is self.model or self._fallback_ok:
            return self._step(batch, self.forward)
        try:
            out = self._step(batch, self.forward)
        except Exception as e:  # noqa: BLE001  compile できない環境では eager に落として学習を続ける（zero_grad から同じバッチでやり直す）
            print(f"train: compile failed, falling back to eager: {type(e).__name__}: {str(e)[:300]}", flush=True)
            self.forward, self.mode_used = self.model, "eager"
            return self._step(batch, self.forward)
        self._fallback_ok = True
        return out

    def _step(self, batch: dict, fwd) -> dict:
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
            policy, wdl, v41 = fwd(sq, glob)
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
