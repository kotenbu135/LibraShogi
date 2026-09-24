# SPDX-License-Identifier: Apache-2.0
"""評価ハーネス: 2 つのネットを同じ探索設定で対局させ、先後別の得点と Elo 差、較正を出す（docs/libra-design.md §4.3）。

同時進行枠の偶奇で A の先後を入れ替える（偶数枠は A が先手）。各手の探索木はルートの手番のネットで丸ごと評価する。
`search_cfg_b` を渡すと B 側だけ別の探索設定で読む（同じ重みで σ の形を比べるときに使う。docs/ls2-settings.md §2）。
較正は、対局記録の root_q（手番側の探索値）を勝率に直したものと実際の結果を区間ごとに比べる。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import librasearch
import librashogi as ls
from libra_net.model import LibraNet, NetConfig

from .calibrate import reliability


def load_model(path: Path, device: torch.device, dtype: torch.dtype = torch.float16) -> LibraNet:
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = sd.get("config", {}).get("net", {})
    m = LibraNet(NetConfig.from_dict(cfg)).to(device)
    m.load_state_dict(sd["model"])
    m.eval()
    if dtype != torch.float32:
        m = m.to(dtype)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def elo_diff(score: float) -> float:
    s = min(max(score, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / s - 1.0)


def calibration(games: list[dict], side_of: "callable", bins: int = 10) -> list[dict]:
    """side_of(game, ply_index) が True の手だけを使う。root_q は手番側の値、結果も手番側に直す。"""
    qs, zs = [], []
    for g in games:
        res = int(g["result"])
        for j, q in enumerate(g["root_q"]):
            if not side_of(g, j):
                continue
            sente = j % 2 == 0  # 添字 0 は 3 手目（先手）
            z = res if sente else -res
            qs.append((float(q) + 1) / 2)
            zs.append((z + 1) / 2)
    return reliability(np.array(qs), np.array(zs), bins).get("bins", [])


PARTIAL_KEYS = ("slot", "result", "reason", "plies", "root_q")  # 途中経過に残す対局の項目（集計と較正に使うものだけ）


def _load_partial(path: Path, header: dict) -> list[dict]:
    """途中経過（1 行目が条件、以降 1 行 1 局）を読む。条件が違う・壊れているときは使わずに .stale へ退ける。
    最後の行は書きかけで止められたことがあるので、読めない行は捨てる。"""
    if not path.exists():
        return []
    games: list[dict] = []
    ok = False
    try:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if i == 0:
                ok = row == header
                if not ok:
                    break
            elif isinstance(row, dict) and all(k in row for k in PARTIAL_KEYS):
                games.append(row)
    except OSError:
        return []
    if not ok:
        path.replace(path.with_name(path.name + ".stale"))
        return []
    return games


@torch.no_grad()
def play_match(model_a: LibraNet, model_b: LibraNet, search_cfg: dict, n_games: int, concurrent: int, threads: int, seed: int,
               device: torch.device, dtype: torch.dtype = torch.float16, log=None, search_cfg_b: dict | None = None,
               partial: Path | None = None, partial_tag: dict | None = None) -> dict:
    """A と B を n_games 局打つ。A の先手（偶数枠）と後手（奇数枠）はちょうど半分ずつ（奇数なら先手が 1 局多い）。

    局数ちょうどで止めるときに打ちかけの対局を捨てない: 枠ごとに「次の対局を始めてよいか」を始める前に決め
    （`SelfPlay.retire`）、始めた対局はすべて数える。先に終わった n_games 局を取る形だと、打ちかけで残るのは長い対局なので
    短い対局に偏り、同時に打つ数（concurrent）を増やすほど偏りが大きくなる（2026-09-24 まではこの形で、64 枠・2,000 局で約 3%）。

    partial を渡すと、数えた対局を 1 局ずつ追記し、次に同じ条件で呼ばれたらその続きから打つ（ランの停止・起動で
    計測ジョブが止められても、打ち終えた対局を捨てない）。続きは種を変えて打つ（同じ種だと同じ対局をもう一度打つため）。"""
    cfg = dict(search_cfg)
    cfg["full_prob"] = 1.0  # 評価は全読みで固定
    cfg["resign_threshold"] = 0.0  # 計測の対局では投了しない（誤投了が Elo に乗ると物差しが狂う。自己対局だけで使う）
    # 計測の対局は玉を 36×36 から一様に置く（後手玉四段目 25%）。四段目の局は 41 手目の裁定で先手が勝つので得点は両者 0.5 に寄り、
    # その割合が節目ごとに変わると同じ強さの差でも Elo の出方が変わる。自己対局の偏り（gote_rank4_prob）は持ち込まない
    cfg["gote_rank4_prob"] = -1.0
    # 根の証明探索は GPU の評価中に解く（自己対局と同じ。棋譜は変わらない: test_selfplay_threads.py）
    cfg["defer_root_proof"] = True
    header = {**(partial_tag or {}), "n_games": n_games, "search": {k: cfg[k] for k in sorted(cfg) if isinstance(cfg[k], (int, float, str, bool))},
              "search_b": search_cfg_b or None}
    games: list[dict] = _load_partial(partial, header) if partial is not None else []
    resumed = len(games)
    # 残りの局数（枠の偶奇ごと。偶数枠は A が先手）
    need = [(n_games + 1) // 2, n_games // 2]
    for g in games:
        need[int(g["slot"]) % 2] -= 1
    need = [max(0, x) for x in need]
    per_side = max(1, min(max(1, concurrent // 2), max(need)))
    n_slots = 2 * per_side
    eng = librasearch.SelfPlay(cfg, n_slots, seed + resumed, threads)
    if search_cfg_b is not None:
        cfg_b = {**cfg, **search_cfg_b, "full_prob": 1.0}
        eng.set_side_config(cfg_b)  # B 側の探索設定（変えられるのは読む手の選び方だけ。ほかが違えば例外）
    # 要る局数より枠が多い側は、始めから止める枠を決めておき、その最初の対局は数えない（打つ前に決めるので偏らない）
    ignore: set[int] = set()
    retired = np.zeros(n_slots, bool)  # 今の対局が終わったら止める枠（eng.retire と同じ）

    def retire(s: int) -> None:
        retired[s] = True
        eng.retire(s)

    for p in (0, 1):
        for r in range(need[p], per_side):
            ignore.add(2 * r + p)
            retire(2 * r + p)
    started = [min(per_side, need[p]) for p in (0, 1)]  # 始めた（数える）対局の数
    idle = np.zeros(n_slots, bool)  # 止めた枠で対局も終わった（評価に出さない）
    pf = None
    if partial is not None:
        partial.parent.mkdir(parents=True, exist_ok=True)
        if resumed == 0:
            partial.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        pf = open(partial, "a", encoding="utf-8")
        if resumed and log:
            log(f"eval: resume {resumed}/{n_games} games from {partial.name}")
    sq = np.zeros((n_slots, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((n_slots, ls.GLOB_FEATS), np.float32)
    slot_swap = (np.arange(n_slots) % 2).astype(np.int8)  # 奇数枠は B が先手
    t0 = time.time()
    last_log = 0
    try:
        while len(games) < n_games:
            if idle.all():
                raise RuntimeError(f"eval: all slots stopped at {len(games)}/{n_games} games")
            eng.collect(sq, glob)
            who = eng.root_turns() ^ slot_swap  # 0 なら A のネット、1 なら B
            logits = np.zeros((n_slots, ls.POLICY_SIZE), np.float32)
            wdl = np.zeros((n_slots, 3), np.float32)
            sq_t = torch.from_numpy(sq).to(device).to(dtype)
            gl_t = torch.from_numpy(glob).to(device).to(dtype)
            outs = []
            for k, model in ((0, model_a), (1, model_b)):
                idx = np.flatnonzero((who == k) & ~idle)
                if idx.size == 0:
                    continue
                it = torch.from_numpy(idx).to(device)
                p, w, _ = model(sq_t[it], gl_t[it])
                outs.append((idx, p, w))
            eng.proof()  # GPU が評価している間に根の証明探索を解く
            for idx, p, w in outs:
                logits[idx] = p.float().cpu().numpy()
                wdl[idx] = F.softmax(w.float(), dim=-1).cpu().numpy()
            eng.apply(logits, wdl)
            for g in eng.take_finished():
                s = int(g["slot"])
                if retired[s]:
                    idle[s] = True
                else:
                    started[s % 2] += 1  # 終わった枠はすぐ次の対局を始めている
                if s in ignore:
                    ignore.discard(s)
                    continue
                row = {k: g[k] for k in PARTIAL_KEYS}
                row["root_q"] = [round(float(q), 4) for q in row["root_q"]]
                games.append(row)
                if pf is not None:
                    pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    pf.flush()
            # 次の回に止めずにおく枠がすべて終局しても、始めた対局が要る局数を超えないようにする（番号の大きい枠から止める）
            for p in (0, 1):
                live = [s for s in range(p, n_slots, 2) if not retired[s]]
                for s in sorted(live, reverse=True)[:max(0, started[p] + len(live) - need[p])]:
                    retire(s)
            if log and len(games) - last_log >= 20:
                last_log = len(games)
                log(f"eval: {len(games)}/{n_games} games ({time.time() - t0:.0f}s)")
    finally:
        if pf is not None:
            pf.close()
    # 集計: A が先手なのは偶数枠
    a_sente = {"w": 0, "d": 0, "l": 0}
    a_gote = {"w": 0, "d": 0, "l": 0}
    reasons: dict[str, int] = {}
    for g in games:
        res = int(g["result"])
        a_is_sente = g["slot"] % 2 == 0
        a_res = res if a_is_sente else -res
        key = "w" if a_res > 0 else "l" if a_res < 0 else "d"
        (a_sente if a_is_sente else a_gote)[key] += 1
        reasons[g["reason"]] = reasons.get(g["reason"], 0) + 1
    n = len(games)
    pts = a_sente["w"] + a_gote["w"] + 0.5 * (a_sente["d"] + a_gote["d"])
    score = pts / max(1, n)
    se = math.sqrt(max(score * (1 - score), 1e-6) / max(1, n))

    def a_moved(g: dict, j: int) -> bool:
        return (j % 2 == 0) == (g["slot"] % 2 == 0)

    return {
        "n": n,
        "score_a": round(score, 4),
        "score_se": round(se, 4),
        "elo_a_minus_b": round(elo_diff(score), 1),
        "elo_ci95": [round(elo_diff(max(1e-4, score - 1.96 * se)), 1), round(elo_diff(min(1 - 1e-4, score + 1.96 * se)), 1)],
        "a_as_sente": a_sente,
        "a_as_gote": a_gote,
        "reasons": reasons,
        "avg_plies": round(sum(int(g["plies"]) for g in games) / max(1, n), 1),
        "calibration_a": calibration(games, a_moved),
        "calibration_b": calibration(games, lambda g, j: not a_moved(g, j)),
        "sims": cfg.get("full_sims"),
        "gumbel_noise": bool(cfg.get("gumbel_noise", True)),
        "search_b": {k: v for k, v in (search_cfg_b or {}).items() if cfg.get(k) != v} or None,
        "seconds": round(time.time() - t0, 1),
        "concurrent": n_slots,
        "resumed_games": resumed,
    }


def main_eval(a: Path, b: Path, search_cfg: dict, n_games: int, concurrent: int, threads: int, seed: int, out: Path | None,
              search_cfg_b: dict | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ma = load_model(a, device)
    mb = load_model(b, device)
    # 打ち終えた対局は <out>.partial.jsonl に 1 局ずつ残し、止められて同じ引数で起動し直されたら続きから打つ（auto.py の積み直し）
    partial = out.with_name(out.name + ".partial.jsonl") if out else None
    res = play_match(ma, mb, search_cfg, n_games, concurrent, threads, seed, device, log=lambda s: print(s, flush=True),
                     search_cfg_b=search_cfg_b, partial=partial, partial_tag={"a": str(a), "b": str(b), "seed": seed})
    res["a"] = str(a)
    res["b"] = str(b)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        partial.unlink(missing_ok=True)
    return res
