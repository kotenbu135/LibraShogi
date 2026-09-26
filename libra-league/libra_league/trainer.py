# SPDX-License-Identifier: Apache-2.0
"""学習: AdamW、bf16 autocast、方策・価値・V̂41（と補助の相手の次の手）の損失。"""
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
        # 補助の頭（KataGo [Wu19] §3.4 の相手の次の手、§4.1 の陣地の将棋版「駒が最後まで残るか」）。重み > 0 のときだけ forward_aux で学習する
        self.opp = float(cfg.get("opp_weight", 0.0)) > 0
        self.own = float(cfg.get("own_weight", 0.0)) > 0
        self.aux = self.opp or self.own
        if self.opp and not getattr(model, "opp_head", None):
            raise ValueError("train.opp_weight > 0 には net.opp_head = true が要る")
        if self.own and not getattr(model, "own_head", None):
            raise ValueError("train.own_weight > 0 には net.own_head = true が要る")
        self._eager = model.forward_aux if self.aux else model
        self.forward = self._eager
        self.mode_used = "eager"
        self._fallback_ok = False  # 1 ステップでも compile で通ったら、それ以降の例外は落とさずに上げる
        if device.type == "cuda" and mode != "none":
            from .selfplay import _quiet_inductor

            _quiet_inductor()
            self.forward = torch.compile(self._eager, mode="max-autotune-no-cudagraphs" if mode == "max-autotune" else "default", fullgraph=True, dynamic=False)
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
        if self.forward is self._eager or self._fallback_ok:
            return self._step(batch, self.forward)
        try:
            out = self._step(batch, self.forward)
        except Exception as e:  # noqa: BLE001  compile できない環境では eager に落として学習を続ける（zero_grad から同じバッチでやり直す）
            print(f"train: compile failed, falling back to eager: {type(e).__name__}: {str(e)[:300]}", flush=True)
            self.forward, self.mode_used = self._eager, "eager"
            return self._step(batch, self.forward)
        self._fallback_ok = True
        return out

    def _step(self, batch: dict, fwd) -> dict:
        """1 step。train.accum_steps = k > 1 なら batch を k 個に分けて勾配を足し合わせる（大きいネットでバッチがメモリに載らないとき用。
        損失は分けた小さいバッチごとの平均の平均で、k = 1 と同じ値にはならない）。k = 1 は分けずに今までと同じ計算。"""
        m = self.model
        m.train()
        for g in self.opt.param_groups:
            g["lr"] = self.lr_at(self.step_count)
        k = int(self.cfg.get("accum_steps", 1))
        self.opt.zero_grad(set_to_none=True)
        parts = []
        for part in ([batch] if k <= 1 else split_batch(batch, k)):
            loss, st = self._loss(part, fwd)
            (loss if k <= 1 else loss / k).backward()
            parts.append(st)
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), self.cfg["grad_clip"])
        self.opt.step()
        self.step_count += 1
        out = {key: sum(float(p[key]) for p in parts) / len(parts) for key in parts[0] if key not in ("acc_n", "acc_d")}
        out["policy_acc"] = sum(float(p["acc_n"]) for p in parts) / max(1.0, sum(float(p["acc_d"]) for p in parts))
        return {**out, "grad_norm": float(gn), "lr": self.lr_at(self.step_count - 1)}

    def _loss(self, batch: dict, fwd) -> tuple[torch.Tensor, dict]:
        dev = self.device
        sq = torch.from_numpy(batch["sq"]).to(dev, non_blocking=True)
        glob = torch.from_numpy(batch["glob"]).to(dev, non_blocking=True)
        wdl_t = torch.from_numpy(batch["wdl"]).to(dev)
        v41_t = torch.from_numpy(batch["v41"]).to(dev)
        fuseki = torch.from_numpy(batch["fuseki"]).to(dev)
        pidx = torch.from_numpy(batch["policy_idx"]).to(dev)
        pp = torch.from_numpy(batch["policy_p"]).to(dev)
        pvalid = torch.from_numpy(batch["policy_valid"]).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            outs = fwd(sq, glob)
        policy, wdl, v41 = outs[:3]
        l_policy, pw = masked_ce(policy, pidx, pp, pvalid)
        l_value = soft_ce(wdl, wdl_t)
        l_v41 = soft_ce(v41, v41_t, fuseki.float())
        loss = self.cfg["policy_weight"] * l_policy + self.cfg["value_weight"] * l_value + self.cfg["v41_weight"] * l_v41
        st = {"loss": loss, "policy": l_policy, "value": l_value, "v41": l_v41}
        if self.opp:
            oidx = torch.from_numpy(batch["opp_idx"]).to(dev)
            op = torch.from_numpy(batch["opp_p"]).to(dev)
            ovalid = torch.from_numpy(batch["opp_valid"]).to(dev)
            l_opp, _ = masked_ce(outs[3]["opp"], oidx, op, ovalid)
            loss = loss + float(self.cfg["opp_weight"]) * l_opp
            st.update({"loss": loss, "opp": l_opp})
        if self.own:
            # 盤上の駒（玉を除く）ごとの 2 値の交差エントロピーの、駒のあるマスの平均（目標 -1 のマスは数えない）
            own_t = torch.from_numpy(batch["own"]).to(dev)
            w = (own_t >= 0).float()
            bce = F.binary_cross_entropy_with_logits(outs[3]["own"].float(), own_t.clamp(min=0), reduction="none")
            l_own = (bce * w).sum() / w.sum().clamp(min=1.0)
            loss = loss + float(self.cfg["own_weight"]) * l_own
            st.update({"loss": loss, "own": l_own})
        with torch.no_grad():
            acc = (policy.float().argmax(1) == pidx[:, 0]).float()
            st.update({"acc_n": (acc * pw).sum(), "acc_d": pw.sum()})
        return loss, {key: v.detach() for key, v in st.items()}

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "step": self.step_count}

    def load_state_dict(self, sd: dict) -> None:
        """重み・AdamW の状態・step を戻す。保存した側に補助の頭（opp_head・own_head）が無ければ、その頭だけ初期値のまま・AdamW の状態も空で始める
        （頭は最後に作るので、パラメータの並びの末尾に足されるだけ。幹と他の頭の状態はそのまま引き継ぐ）。"""
        missing, unexpected = self.model.load_state_dict(sd["model"], strict=False)
        if unexpected or any(not k.startswith(("opp_head.", "own_head.")) for k in missing):
            raise RuntimeError(f"重みが合わない: missing {missing[:5]} unexpected {unexpected[:5]}")
        if "opt" in sd:
            self.opt.load_state_dict(extend_opt_state(sd["opt"], self.opt) if missing else sd["opt"])
        self.step_count = int(sd.get("step", 0))


def masked_ce(logits: torch.Tensor, idx: torch.Tensor, p: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """方策の目標（上位 topk 手の添字 idx と確率 p、無い行は valid = False）との交差エントロピーの、目標のある行の平均。"""
    logp = F.log_softmax(logits.float(), dim=-1)
    ce = -(logp.gather(1, idx.clamp(min=0)) * p).sum(dim=1)
    w = valid.float()
    return (ce * w).sum() / w.sum().clamp(min=1.0), w


def split_batch(batch: dict, k: int) -> list[dict]:
    """学習バッチを先頭の次元で k 個に分ける（1 局面 1 行の配列だけ。target_stats などはそのまま渡す）。"""
    n = len(batch["sq"])
    if n % k:
        raise ValueError(f"train.batch_size {n} は accum_steps {k} で割り切れない")
    step = n // k
    return [{key: (v[i * step:(i + 1) * step] if isinstance(v, np.ndarray) and len(v) == n else v) for key, v in batch.items()}
            for i in range(k)]


def extend_opt_state(saved: dict, opt: torch.optim.Optimizer) -> dict:
    """頭を足す前の AdamW の状態を、頭を足した後の optimizer に合う形にする。各グループの末尾に足りない数だけ新しい番号を足し、
    その番号には状態を持たせない（初めての step で 0 から作られる）。"""
    out = {"state": dict(saved["state"]), "param_groups": [dict(g) for g in saved["param_groups"]]}
    if len(out["param_groups"]) != len(opt.param_groups):
        raise RuntimeError("AdamW のグループの数が合わない")
    nxt = 1 + max((i for g in saved["param_groups"] for i in g["params"]), default=-1)
    for g_saved, g_now in zip(out["param_groups"], opt.param_groups):
        extra = len(g_now["params"]) - len(g_saved["params"])
        if extra < 0:
            raise RuntimeError("AdamW のパラメータが保存した側より少ない")
        g_saved["params"] = list(g_saved["params"]) + list(range(nxt, nxt + extra))
        nxt += extra
    return out
