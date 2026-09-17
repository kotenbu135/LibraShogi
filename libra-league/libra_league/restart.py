# SPDX-License-Identifier: Apache-2.0
"""搾取者の run を、プールに残した過去の自分（lx-<step>.pt）の重みから始め直す（docs/decisions.md 2026-09-17）。

止まっている搾取者の run に 1 回だけ使う。何も消さない:
- 置き換える前の latest.pt・state.json・openings.json を backup-<時刻>/ に写す
- 今の重みをプールに lx-<今の step>.pt で残す（本体のリーグの相手になる）
- latest.pt の重みを過去の自分に置き換える。step の数えと乱数は引き継ぎ、最適化の内部状態は捨てる
- リプレイのチャンクを replay/pre-restart-<step>/ に移し、窓を空から数え直す（今の重みの対局で引き戻さないため）
- 対本体の成績を履歴に移して 0 から数え、布石を空にする（凍結相手の作り直しと同じ扱い）
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch

from libra_net.model import LibraNet, NetConfig

from .league import pool_name
from .openings import write_openings
from .state import StateDir
from .supervise import running_pid
from .workers import load_weights, publish_weights


def restart_exploiter(sd: StateDir, cfg: dict, snapshot: Path, apply: bool, log=print) -> dict:
    ex = cfg.get("exploiter", {})
    if not ex.get("main_ckpt"):
        raise SystemExit(f"{sd.root} は搾取者の run ではない（[exploiter] main_ckpt が無い）")
    if running_pid(sd.root / "run.lock") is not None:
        raise SystemExit(f"{sd.root} は動いている。止めてから実行する")
    ck = sd.checkpoints / "latest.pt"
    cur = torch.load(ck, map_location="cpu", weights_only=False)
    step = int(cur["step"])
    net_cfg = cur["config"]["net"]
    model, snap_step, _ = load_weights(snapshot)
    if model.cfg != NetConfig.from_dict(net_cfg):
        raise SystemExit(f"ネットの形が違う: {snapshot} {model.cfg} / latest.pt {net_cfg}")
    state = sd.read_state()
    chunks = sorted(sd.replay.glob("chunk_*.pkl"))
    pool_out = Path(str(ex.get("pool_out") or "")).expanduser()
    openings_out = Path(str(ex.get("openings_out") or "")).expanduser()
    plan = {"step": step, "snapshot": str(snapshot), "snapshot_step": snap_step, "chunks": len(chunks),
            "keep_current_as": str(pool_out / pool_name(step)) if str(pool_out) not in ("", ".") else None,
            "exploiter_stats": state.get("exploiter_stats")}
    log(f"restart-exploiter: latest.pt（step {step}）の重みを {snapshot.name}（step {snap_step}）に置き換える。"
        f"チャンク {len(chunks)} 個を replay/pre-restart-{step}/ に移す。対本体の成績と布石を 0 から")
    if not apply:
        log("restart-exploiter: --apply が無いので何もしない")
        return plan

    backup = sd.root / f"backup-{time.strftime('%Y%m%d-%H%M%S')}"
    backup.mkdir()
    for p in (ck, sd.state_json, openings_out):
        if str(p) not in ("", ".") and p.is_file():
            shutil.copy2(p, backup / p.name)
    if plan["keep_current_as"]:
        pool_out.mkdir(parents=True, exist_ok=True)
        old = LibraNet(NetConfig.from_dict(net_cfg))
        old.load_state_dict(cur["model"])
        publish_weights(pool_out / pool_name(step), old, step, net_cfg, str(cfg.get("run_id", "")))
    new = {"model": model.state_dict(), "step": step, "rng": cur.get("rng"), "config": cur.get("config"), "state": cur.get("state")}
    tmp = ck.with_suffix(".tmp")
    torch.save(new, tmp)
    tmp.replace(ck)
    moved = sd.replay / f"pre-restart-{step}"
    moved.mkdir(exist_ok=True)
    for p in chunks:
        p.rename(moved / p.name)
    es = state.setdefault("exploiter", {})
    st = state.pop("exploiter_stats", None) or {}
    if st.get("games"):
        es["history"] = (es.get("history", []) + [{
            "t": time.time(), "main_step": es.get("main_step"), "games": st["games"],
            "winrate": round((st.get("wins", 0) + 0.5 * st.get("draws", 0)) / st["games"], 4),
            "note": f"restart weights from {snapshot.name}"}])[-40:]
    es["from_chunk"] = int(state.get("chunk_index", 0))  # 布石は始め直した後のチャンクだけから作る
    es["restarts_from_snapshot"] = (es.get("restarts_from_snapshot", []) + [
        {"t": time.time(), "step": step, "snapshot": snapshot.name, "snapshot_step": snap_step}])[-20:]
    sd.write_state(state)
    if str(openings_out) not in ("", "."):
        write_openings(openings_out, [], str(sd.root))
    sd.append_log(f"restart-exploiter: weights from {snapshot.name} (step {snap_step}) at step {step}; "
                  f"{len(chunks)} chunks moved to {moved.name}; stats and openings reset; backup {backup.name}")
    log(f"restart-exploiter: 済み（backup {backup}）")
    return plan
