# SPDX-License-Identifier: Apache-2.0
"""「仮」の設定値のオフライン比較（docs/restart-plan.md §0 の手順、docs/ls2-settings.md §5.2）。

保存済みの重み（ふつうは archive の節目）と、その重みが持っていた窓から、**設定だけを変えた 2 つ以上の「腕」**を
同じ窓・同じ step 数・同じバッチで学習し、腕どうしと元の重みを対局させて Elo 差を出す。稼働中の run の
ディレクトリには書かない（リプレイと重みを読むだけ。出力は --out のディレクトリ）。

同じ seed のサンプラを腕ごとに作るので、**腕の間で学習する局面と鏡映は 1 バッチずつ同じ**になり、違うのは
設定だけになる（対になった比較。§0 の確認実験は窓だけを変えた 1 対 1 で、同じ考え方）。

一般化の物差し（genprof）は 2 通り出す:
- `gen_z`: 目標を λ = 1.0（実際の勝敗 z）にして測る。**腕の間で同じ物差し**になるので比較に使う。
- `gen_own`: 腕自身の λ で測る。学習の損失と同じ物差し（自分の目標にどれだけ当たっているか）。

扱うのは学習側の設定（[train] の値、例 `lambda_z`）。探索の設定（[search]）は窓の中の棋譜と方策の目標を
作り直さないと比べられないので、この命令では変えても意味がない（σ の形の比較は `libra eval --b-set`）。
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from libra_net.model import LibraNet, NetConfig

from .config import DEFAULTS
from .genprof import PHASES, generalization
from .replay import ReplayBuffer
from .state import StateDir, write_json_atomic
from .trainer import Trainer

# 腕ごとに変えてはいけない鍵（窓と held-out の作り方はグループ分けで扱う。ネットの形を変えると重みを引き継げない）
FIXED_SECTIONS = ("net",)
WINDOW_KEYS = ("window_games", "window_frac", "window_games_max")


def parse_set(spec: str) -> tuple[str, str, Any]:
    """`train.lambda_z=1.0` → `("train", "lambda_z", 1.0)`。型は DEFAULTS の同じ鍵に合わせる。無い鍵は誤りにする
    （打ち間違いが黙って無視されると、比べたつもりの設定が変わっていない結果になるため）。"""
    if "=" not in spec or "." not in spec.split("=", 1)[0]:
        raise ValueError(f"--set/--arm の上書きは <節>.<鍵>=<値> の形で書く: {spec!r}")
    path, value = spec.split("=", 1)
    section, key = path.split(".", 1)
    if section not in DEFAULTS or not isinstance(DEFAULTS[section], dict) or key not in DEFAULTS[section]:
        raise ValueError(f"設定に無い鍵: {section}.{key}")
    cur = DEFAULTS[section][key]
    if isinstance(cur, bool):
        if value.lower() not in ("true", "false"):
            raise ValueError(f"{section}.{key} は true / false: {value!r}")
        return section, key, value.lower() == "true"
    if isinstance(cur, int) and not isinstance(cur, bool):
        return section, key, int(value)
    if isinstance(cur, float):
        return section, key, float(value)
    return section, key, value


def parse_arm(spec: str) -> tuple[str, list[str]]:
    """`lambda1:train.lambda_z=1.0,train.mirror_prob=0.0` → 名前と上書きの一覧。`:` が無ければ上書きなし（元の設定のまま）。"""
    name, _, rest = spec.partition(":")
    name = name.strip()
    if not name:
        raise ValueError(f"腕の名前が空: {spec!r}")
    sets = [s for s in (x.strip() for x in rest.split(",")) if s]
    return name, sets


def apply_sets(cfg: dict, sets: list[str]) -> dict:
    out = copy.deepcopy(cfg)
    for spec in sets:
        section, key, value = parse_set(spec)
        if section in FIXED_SECTIONS:
            raise ValueError(f"[{section}] は腕ごとに変えられない（重みを引き継げない）: {spec}")
        out.setdefault(section, {})[key] = value
    return out


def diff_of(base: dict, cfg: dict) -> dict:
    """base との差（節.鍵 → 値）。報告と JSON に残す。"""
    out: dict[str, Any] = {}
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if base.get(section, {}).get(key) != value:
                out[f"{section}.{key}"] = value
    return out


def window_key(cfg: dict) -> tuple:
    tr, rr, sr = cfg["train"], cfg["run"], cfg["search"]
    return (tuple(tr[k] for k in WINDOW_KEYS), rr["chunk_games"], rr["heldout_every_chunks"], rr["heldout_games"],
            sr["max_ply"], sr["count_from_41"])


def load_replay(sd: StateDir, cfg: dict, chunk_index: int, games_total: int, log) -> ReplayBuffer:
    tr, rr, sr = cfg["train"], cfg["run"], cfg["search"]
    rb = ReplayBuffer(sd.replay, sd.games, tr["window_games"], rr["chunk_games"], sr["max_ply"], sr["count_from_41"],
                      window_frac=float(tr.get("window_frac", 0.0)), window_games_max=int(tr.get("window_games_max", 0)),
                      heldout_every_chunks=int(rr.get("heldout_every_chunks", 0)), heldout_games=int(rr.get("heldout_games", 20000)))
    t0 = time.time()
    rb.load(chunk_index, games_total)
    log(f"replay: window {rb.n_games()}/{rb.window()} games, held-out {rb.n_heldout()}, {rb.n_positions()} positions ({time.time() - t0:.0f}s)")
    return rb


def mean_rows(rows: list[dict], keys: tuple[str, ...]) -> dict:
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}


LOSS_KEYS = ("loss", "policy", "value", "v41", "policy_acc")


def train_arm(base_sd: dict, cfg: dict, rb: ReplayBuffer, steps: int, seed: int, device: torch.device, positions: int,
              out_ckpt: Path, every: int, log) -> dict:
    """base_sd（重み＋AdamW の状態）から steps だけ学習し、腕の重みを out_ckpt に保存して物差しを返す。"""
    tr = cfg["train"]
    model = LibraNet(NetConfig.from_dict(cfg["net"])).to(device)
    trainer = Trainer(model, tr, device)
    trainer.load_state_dict(base_sd)
    rng = np.random.default_rng(seed)  # 腕の間で同じ局面・同じ鏡映になる（対になった比較）
    t0 = time.time()
    curve, acc = [], []
    for i in range(steps):
        batch = rb.sample(tr["batch_size"], rng, tr["mirror_prob"], tr["lambda_z"], cfg["search"]["policy_topk"], bool(tr.get("full_only", False)))
        acc.append(trainer.step(batch))
        if len(acc) >= every or i + 1 == steps:
            row = {"step": trainer.step_count, **mean_rows(acc, LOSS_KEYS)}
            curve.append(row)
            log(f"  {i + 1}/{steps} steps ({time.time() - t0:.0f}s): loss {row['loss']} policy {row['policy']} value {row['value']} v41 {row['v41']}")
            acc = []
    topk = cfg["search"]["policy_topk"]
    gen_z = generalization(model, rb, positions, np.random.default_rng(seed + 1), device, 1.0, topk)
    gen_own = generalization(model, rb, positions, np.random.default_rng(seed + 1), device, tr["lambda_z"], topk)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": trainer.opt.state_dict(), "step": trainer.step_count, "config": cfg}, out_ckpt)
    del model, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"steps": steps, "seconds": round(time.time() - t0, 1), "curve": curve, "gen_z": gen_z, "gen_own": gen_own, "ckpt": str(out_ckpt)}


def play_pairs(ckpts: dict[str, Path], pairs: list[tuple[str, str]], scfg: dict, games: int, concurrent: int, threads: int,
               seed: int, device: torch.device, log) -> list[dict]:
    """ckpts の中の組を対局させる（評価ハーネス。`libra eval` と同じ条件）。"""
    from .evaluate import load_model, play_match

    rows = []
    for k, (a, b) in enumerate(pairs):
        ma, mb = load_model(ckpts[a], device), load_model(ckpts[b], device)
        log(f"match {a} vs {b}: {games} games")
        res = play_match(ma, mb, scfg, games, concurrent, threads, seed + k, device, log=log)
        del ma, mb
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rows.append({"a": a, "b": b, **{x: res[x] for x in ("n", "score_a", "elo_a_minus_b", "elo_ci95", "a_as_sente", "a_as_gote",
                                                            "avg_plies", "reasons")}})
        log(f"  {a} score {res['score_a']} Elo {res['elo_a_minus_b']} {res['elo_ci95']}")
    return rows


def run_abtest(sd: StateDir, base_cfg: dict, ckpt: Path, arms: list[str], base_sets: list[str], steps: int, games: int, sims: int,
               concurrent: int, threads: int, seed: int, positions: int, every: int, vs_base: bool, out_dir: Path,
               device: torch.device, chunk_index: int | None, games_total: int | None, log, config_from: str = "ckpt") -> dict:
    base_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck_state = base_sd.get("state", {}) or {}
    ck_cfg = base_sd.get("config") or {}
    if config_from == "ckpt" and ck_cfg.get("train") and ck_cfg.get("search"):
        log("config: チェックポイントに保存された設定を使う（--config-from run で config.toml に切り替え）")
        base_cfg = ck_cfg
    cfg0 = apply_sets(base_cfg, base_sets)
    ci = chunk_index if chunk_index is not None else int(ck_state.get("chunk_index", 0))
    gt = games_total if games_total is not None else int(ck_state.get("games_total", 0))
    if ci <= 0:
        raise ValueError("窓の位置が分からない（チェックポイントに state が無い）。--chunk-index と --games-total を渡す")
    parsed = [parse_arm(a) for a in arms]
    if len({n for n, _ in parsed}) != len(parsed):
        raise ValueError("腕の名前が重なっている")
    cfgs = {name: apply_sets(cfg0, sets) for name, sets in parsed}
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {
        "schema": 1, "run": sd.root.name, "ckpt": str(ckpt), "ckpt_step": int(base_sd.get("step", 0)),
        "chunk_index": ci, "games_total": gt, "steps": steps, "seed": seed, "device": str(device),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_sets": base_sets, "arms": {name: {"diff": diff_of(cfg0, cfgs[name])} for name, _ in parsed},
    }
    log(f"abtest: ckpt {ckpt} (step {res['ckpt_step']}, chunk {ci}, games {gt}) → {out_dir}")
    for name, _ in parsed:
        log(f"  arm {name}: {res['arms'][name]['diff'] or '(base)'}")
    # 窓の作り方が同じ腕はまとめて学習し、窓は 1 つだけ持つ（RAM）
    groups: dict[tuple, list[str]] = {}
    for name, _ in parsed:
        groups.setdefault(window_key(cfgs[name]), []).append(name)
    ckpts: dict[str, Path] = {}
    for key, names in groups.items():
        rb = load_replay(sd, cfgs[names[0]], ci, gt, log)
        for name in names:
            log(f"train arm {name} ({steps} steps)")
            out_ckpt = out_dir / f"{name}.pt"
            res["arms"][name].update(train_arm(base_sd, cfgs[name], rb, steps, seed, device, positions, out_ckpt, every, log))
            res["arms"][name]["window_games"] = rb.n_games()
            ckpts[name] = out_ckpt
            write_json_atomic(out_dir / "abtest.json", res)
        del rb
    if games > 0:
        scfg = dict(cfg0["search"])
        scfg["full_sims"] = sims
        names = [n for n, _ in parsed]
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
        if vs_base:
            ckpts["base"] = ckpt
            pairs += [(n, "base") for n in names]
        res["matches"] = play_pairs(ckpts, pairs, scfg, games, concurrent, threads, seed, device, log)
        res["match"] = {"games": games, "sims": sims, "gumbel_noise": scfg.get("gumbel_noise", True)}
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json_atomic(out_dir / "abtest.json", res)
    return res


def publish_result(res: dict, name: str, repo: Path | None, branch: str, subdir: str, log, push: bool = True) -> str | None:
    """結果を `progress` ブランチ（`<subdir>/<name>.json` と `.md`）へ push する。`~/libra-run` は手元の PC にしか
    無いので、外（クラウドのセッション、別の端末）から結果を読めるようにするため（docs/runbook.md §設定の管理の
    「進捗の書き出し」と同じ形: main と作業ツリーには触れない）。絶対パスは消し、ホームは `~` に直す。"""
    from .progress import publish, repo_root, scrub

    d = subdir.strip("/")
    pre = f"{d}/" if d else ""
    body = json.dumps(scrub(res), ensure_ascii=False, indent=1) + "\n"
    text = format_abtest(res) if "arms" in res else format_side_eval(res)
    files = {f"{pre}{name}.json": body, f"{pre}{name}.md": _scrub_text(text) + "\n"}
    return publish(repo or repo_root(), branch, files, f"experiment: {name}", push=push, log=log)


def format_side_eval(res: dict) -> str:
    """`libra eval`（--b-set で側ごとに探索設定を変えた対局）の 1 行。"""
    return (f"A {res.get('a')}\nB {res.get('b')}\nB 側の探索設定: {res.get('search_b')}\n"
            f"読み {res.get('sims')}・{res.get('n')} 局: A の得点 {res.get('score_a')}、Elo {res.get('elo_a_minus_b')} "
            f"{res.get('elo_ci95')}、平均 {res.get('avg_plies')} 手")


def _scrub_text(text: str) -> str:
    from .progress import scrub

    return "\n".join(scrub(line) for line in text.split("\n"))


def format_abtest(res: dict) -> str:
    lines = [f"ckpt {res['ckpt']} step {res['ckpt_step']} / window at chunk {res['chunk_index']} ({res['games_total']} games)"
             f" / {res['steps']} steps / seed {res['seed']}"]
    for name, arm in res["arms"].items():
        lines.append(f"[{name}] {arm.get('diff') or '(base)'}")
        curve = arm.get("curve") or []
        if curve:
            f, l = curve[0], curve[-1]
            lines.append(f"  loss {f['loss']}→{l['loss']} (policy {f['policy']}→{l['policy']}, value {f['value']}→{l['value']},"
                         f" v41 {f['v41']}→{l['v41']}, acc {f['policy_acc']}→{l['policy_acc']})")
        for which in ("gen_z", "gen_own"):
            g = arm.get(which)
            if not g:
                continue
            parts = [f"{ph} corr {g['window'][ph]['corr_v']}→{g['heldout'][ph]['corr_v']} mse {g['window'][ph]['mse_v']}→{g['heldout'][ph]['mse_v']}"
                     for ph in PHASES]
            lines.append(f"  {which} (window→heldout): " + " | ".join(parts))
    for m in res.get("matches", []):
        lines.append(f"{m['a']} vs {m['b']}: score {m['score_a']} Elo {m['elo_a_minus_b']} {m['elo_ci95']} ({m['n']} games,"
                     f" avg {m['avg_plies']} plies)")
    return "\n".join(lines)
