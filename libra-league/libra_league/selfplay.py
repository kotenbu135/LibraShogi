# SPDX-License-Identifier: Apache-2.0
"""自己対局ループ: C++ エンジンが葉を集め、PyTorch がバッチ評価し、エンジンが進める。"""
from __future__ import annotations

import copy
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

import librasearch
import librashogi as ls
from libra_net.model import LibraNet

COMPILE_MODES = ("none", "default", "max-autotune")


class InferenceNet:
    """推論用の写し。1 回だけ作り、学習後の重みは load でその場に書き込む（番地を変えない）。

    CUDA では固定バッチ n の forward（方策ロジットと WDL 確率、どちらも float32）を CUDA Graphs で捕獲する。
    compile が none 以外なら torch.compile した forward を捕獲する（max-autotune は行列積のカーネルも選ぶ。
    GPU 単独の実測で eager の 1.14 倍、docs/measurements.md 2026-09-14）。捕獲に失敗したら段を落として続ける。
    """

    def __init__(self, model: LibraNet, n: int, device: torch.device, dtype: torch.dtype, compile: str = "none"):
        if compile not in COMPILE_MODES:
            raise ValueError(f"selfplay.compile must be one of {COMPILE_MODES}: {compile!r}")
        m = copy.deepcopy(model).to(device).eval()
        if dtype != torch.float32:
            m = m.to(dtype)
        for p in m.parameters():
            p.requires_grad_(False)
        self.net = m
        self.n = n
        self.device = device
        self.dtype = dtype
        self.compile = compile
        self.compiled = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.mode_used = "eager"
        self._static: tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | None = None
        self._broken: set[str] = set()  # 捕獲に失敗した段（起動し直すまで再挑戦しない）

    @torch.no_grad()
    def load(self, model: LibraNet) -> bool:
        """重みを書き込む。ネットの形が違えば何もせず False（呼び出し側が作り直す）。"""
        if model.cfg != self.net.cfg:
            return False
        src = dict(model.named_parameters())
        for name, p in self.net.named_parameters():
            p.copy_(src[name].detach())
        return True

    def release(self) -> None:
        """捕獲したグラフと固定バッファを捨てて GPU メモリを空ける。次の呼び出しで捕獲し直す。"""
        self.graph = None
        self._static = None

    def _forward(self, fn, sq: torch.Tensor, glob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p, w, _ = fn(sq, glob)
        return p.float(), F.softmax(w.float(), dim=-1)

    def _capture(self) -> None:
        for stage in ("compile", "graph"):
            if stage in self._broken or (stage == "compile" and self.compile == "none"):
                continue
            try:
                fn = self.net
                if stage == "compile":
                    if self.compiled is None:
                        _quiet_inductor()
                        mode = "max-autotune-no-cudagraphs" if self.compile == "max-autotune" else "default"
                        self.compiled = torch.compile(self.net, mode=mode, fullgraph=True)
                    fn = self.compiled
                sq = torch.zeros((self.n, 81, ls.SQ_FEATS), device=self.device, dtype=self.dtype)
                glob = torch.zeros((self.n, ls.GLOB_FEATS), device=self.device, dtype=self.dtype)
                for _ in range(2):  # compile と autotune はここで済ませる
                    self._forward(fn, sq, glob)
                side = torch.cuda.Stream(self.device)
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self._forward(fn, sq, glob)
                torch.cuda.current_stream(self.device).wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self._forward(fn, sq, glob)
                self.graph, self._static = graph, (sq, glob, out)
                self.mode_used = (f"compile({self.compile})+" if stage == "compile" else "eager+") + "cudagraph"
                return
            except Exception as e:  # noqa: BLE001  捕獲できない環境では段を落として自己対局を続ける
                self._broken.add(stage)
                self.graph, self._static = None, None
                print(f"selfplay: {stage} capture failed, falling back: {type(e).__name__}: {str(e)[:300]}", flush=True)
        self.mode_used = "eager"

    @torch.no_grad()
    def __call__(self, sq: torch.Tensor, glob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """sq [n, 81, SQ_FEATS]・glob [n, GLOB_FEATS]（CPU、float32。CUDA なら pin 済み）→ (ロジット, WDL 確率)。

        CUDA Graphs のときは戻り値が固定バッファなので、次の呼び出しの前に写し取ること。"""
        if self.device.type == "cuda" and self.graph is None and len(self._broken) < 2:
            self._capture()
        if self._static is not None:
            s_sq, s_glob, out = self._static
            s_sq.copy_(sq, non_blocking=True)
            s_glob.copy_(glob, non_blocking=True)
            self.graph.replay()
            return out
        return self._forward(self.net, sq.to(self.device, non_blocking=True).to(self.dtype), glob.to(self.device, non_blocking=True).to(self.dtype))


def _quiet_inductor() -> None:
    """max-autotune が候補ごとの計時表と失敗した候補を標準エラーに出すので止める（stdout.log を埋めるため）。"""
    import warnings

    import torch._inductor.config as ic

    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
    ic.max_autotune_report_choices_stats = False
    ic.autotune_num_choices_displayed = 0
    logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.CRITICAL)


class SelfPlayLoop:
    def __init__(self, search_cfg: dict, n_games: int, threads: int, seed: int, device: torch.device, infer_dtype: str = "float16",
                 compile: str = "none"):
        # 根の証明探索は round() で GPU が評価している間に解く（CPU の apply から外す。棋譜は変わらない）。設定の false で切れる。
        # 対局ごとのネットの出力のキャッシュ（eval_cache）: 直近 3 手の探索で評価した局面を評価に出さない（棋譜は変わらない）。
        # 重みを替えるたびに捨てる（set_model・set_opponent）。搾取者モードでは両方のネットの出力を持つ（set_two_nets）。[search] eval_cache = false で切れる
        self.engine = librasearch.SelfPlay({"defer_root_proof": True, "eval_cache": True, **search_cfg}, n_games, seed, threads)
        self.n_games = n_games
        self.device = device
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[infer_dtype]
        if compile not in COMPILE_MODES:
            raise ValueError(f"selfplay.compile must be one of {COMPILE_MODES}: {compile!r}")
        self.compile = compile
        pin = device.type == "cuda"
        self.sq = torch.empty((n_games, 81, ls.SQ_FEATS), dtype=torch.float32, pin_memory=pin)
        self.glob = torch.empty((n_games, ls.GLOB_FEATS), dtype=torch.float32, pin_memory=pin)
        self.sq_np = self.sq.numpy()
        self.glob_np = self.glob.numpy()
        self.logits = torch.empty((n_games, ls.POLICY_SIZE), dtype=torch.float32, pin_memory=pin)
        self.wdl = torch.empty((n_games, 3), dtype=torch.float32, pin_memory=pin)
        self.o_logits: torch.Tensor | None = None  # 搾取者モードの凍結した本体の出力（set_opponent で作る）
        self.o_wdl: torch.Tensor | None = None
        self.model: InferenceNet | None = None
        self.opponent: InferenceNet | None = None  # 搾取者モード: 凍結した本体。奇数枠では本体が先手
        self.opponent_prior = False
        # 段ごとの時間（秒の累計）。dict を入れたときだけ測る: collect（CPU）、eval（H2D・forward・D2H、CUDA は同期まで。proof を除く）、
        # proof（根の証明探索、CPU。GPU の評価と重なる）、apply（CPU）
        self.timing: dict | None = None

    def _install(self, cur: InferenceNet | None, model: LibraNet) -> InferenceNet:
        if cur is not None and cur.load(model):
            return cur
        return InferenceNet(model, self.n_games, self.device, self.dtype, self.compile)

    def set_model(self, model: LibraNet) -> None:
        """学習中のモデルの重みを推論用の写しに移す（写しは 1 回だけ作る）。"""
        self.model = self._install(self.model, model)
        self.engine.clear_eval_cache()

    def set_opponent(self, model: LibraNet | None, opponent_prior: bool = True) -> None:
        """opponent_prior: 搾取者の手の探索木の中で、本体の手番の葉の方策を本体のネットから取る（価値は搾取者のネット）。

        行ごとのネットの選び分けはエンジン（set_two_nets、偶数枠は搾取者が先手）が行い、両方のネットの出力を eval_cache に持つ。"""
        self.opponent = None if model is None else self._install(self.opponent, model)
        self.opponent_prior = opponent_prior
        self.engine.set_two_nets(model is not None, opponent_prior)  # キャッシュも捨てる
        if model is not None and self.o_logits is None:
            pin = self.device.type == "cuda"
            self.o_logits = torch.empty((self.n_games, ls.POLICY_SIZE), dtype=torch.float32, pin_memory=pin)
            self.o_wdl = torch.empty((self.n_games, 3), dtype=torch.float32, pin_memory=pin)

    def release(self) -> None:
        for net in (self.model, self.opponent):
            if net is not None:
                net.release()

    @staticmethod
    def exploiter_is_sente(slot: int) -> bool:
        return slot % 2 == 0


    @torch.no_grad()
    def round(self) -> list[dict]:
        assert self.model is not None
        tm = self.timing
        t0 = time.perf_counter() if tm is not None else 0.0
        self.engine.collect(self.sq_np, self.glob_np)
        t1 = time.perf_counter() if tm is not None else 0.0
        logits, wdl = self.model(self.sq, self.glob)
        if self.opponent is not None:
            # CUDA Graphs が固定バッチなので、両方のネットで全枠を評価し、エンジンが行ごとに選ぶ（set_opponent の説明）。
            # 戻り値はネットごとの固定バッファなので、続けて呼んでから写し取ってよい
            o_logits, o_wdl = self.opponent(self.sq, self.glob)
            self.o_logits.copy_(o_logits, non_blocking=True)
            self.o_wdl.copy_(o_wdl, non_blocking=True)
        self.logits.copy_(logits, non_blocking=True)
        self.wdl.copy_(wdl, non_blocking=True)
        tp = time.perf_counter() if tm is not None else 0.0
        self.engine.proof()  # GPU の評価を待つ間に根の証明探索を解く
        tq = time.perf_counter() if tm is not None else 0.0
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        t2 = time.perf_counter() if tm is not None else 0.0
        if self.opponent is not None:
            self.engine.apply2(self.logits.numpy(), self.wdl.numpy(), self.o_logits.numpy(), self.o_wdl.numpy())
        else:
            self.engine.apply(self.logits.numpy(), self.wdl.numpy())
        games = self.engine.take_finished()
        if tm is not None:
            t3 = time.perf_counter()
            tm["rounds"] = tm.get("rounds", 0) + 1
            tm["collect"] = tm.get("collect", 0.0) + (t1 - t0)
            tm["proof"] = tm.get("proof", 0.0) + (tq - tp)
            tm["eval"] = tm.get("eval", 0.0) + (t2 - t1) - (tq - tp)
            tm["apply"] = tm.get("apply", 0.0) + (t3 - t2)
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
