# SPDX-License-Identifier: Apache-2.0
"""`libra review`: 物差し M1〜M4 の生の値（metrics.jsonl の gen、eval/best.jsonl・anchor.jsonl・reference.jsonl）を読み、
閾値で「続ける／注意／見直し」を出す（docs/restart-plan.md §3 M6・§7 P4、合否の目安は docs/ls2-settings.md §5）。
計測は保存済みの値だけを読むので、閾値を変えても再計測は要らない。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .auto import collect_anchor, collect_best, collect_reference, load_metrics
from .state import StateDir

DEFAULT_THRESHOLDS = {
    "gen_hours": 4.0,        # gen の伸びを見る窓（時間）
    "gen_min_rise": 0.0,     # held-out の本将棋の価値の相関が、窓の間にこれ以上上がれば「続ける」
    "gen_max_gap": 0.1,      # 窓の中 − held-out の相関の差がこれを超えたら「注意」（窓の記憶）
    "best_stall_alert": 3,   # 最強を更新できない回数がこれ以上なら「見直し」
    "reference_hours": 24.0, # 参照との Elo の伸びを見る窓（時間）
    "gpd_min": 0,            # 局/日の下限（0 で見ない）
}

OK, WARN, REVIEW, NA = "続ける", "注意", "見直し", "まだ無い"


def _gen_rows(sd: StateDir, hours: float, now: float) -> list[dict]:
    rows = [r for r in load_metrics(sd, 100000) if r.get("gen") and (r["gen"].get("heldout") or {}).get("normal")]
    return [r for r in rows if float(r["t"]) >= now - hours * 3600]


def review(sd: StateDir, thresholds: dict | None = None, now: float | None = None) -> dict:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    now = now or time.time()
    out: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), "run": sd.root.name, "items": []}

    # M1 一般化: held-out の本将棋の価値の相関の伸びと、窓の中との差
    rows = _gen_rows(sd, th["gen_hours"], now)
    if len(rows) >= 2:
        first, last = rows[0]["gen"], rows[-1]["gen"]
        h0, h1 = first["heldout"]["normal"].get("corr_v"), last["heldout"]["normal"].get("corr_v")
        w1 = last["window"]["normal"].get("corr_v")
        gap = (w1 - h1) if (w1 is not None and h1 is not None) else None
        rise = (h1 - h0) if (h0 is not None and h1 is not None) else None
        if gap is not None and gap > th["gen_max_gap"]:
            verdict, why = WARN, f"窓の中との差 {gap:+.3f} が {th['gen_max_gap']} を超えた（窓の記憶）"
        elif rise is not None and rise < th["gen_min_rise"]:
            verdict, why = WARN, f"held-out の相関が {th['gen_hours']:.0f} 時間で {rise:+.3f}（上がっていない）"
        else:
            verdict, why = OK, f"held-out の相関 {h0} → {h1}（{th['gen_hours']:.0f} 時間）、窓の中との差 {gap:+.3f}" if gap is not None else OK
        out["items"].append({"name": "M1 一般化（本将棋の価値の相関）", "verdict": verdict, "why": why,
                             "values": {"heldout_corr_first": h0, "heldout_corr_last": h1, "window_corr_last": w1, "gap": gap, "n_rows": len(rows)}})
    else:
        out["items"].append({"name": "M1 一般化（本将棋の価値の相関）", "verdict": NA, "why": f"gen の行が {len(rows)} 個（{th['gen_hours']:.0f} 時間の窓）", "values": {}})

    # M2 最強比
    best = collect_best(sd)
    if best:
        lb = best[-1]
        stall = int(lb.get("stall", 0))
        if stall >= th["best_stall_alert"]:
            verdict, why = REVIEW, f"最強（step {lb.get('best_step')}）を {stall} 回続けて更新できない"
        elif lb.get("improved"):
            verdict, why = OK, f"step {lb.get('step')} が最強を更新（{lb.get('elo_vs_best'):+.1f} Elo、区間 {lb.get('ci95')}）"
        else:
            verdict, why = WARN, f"step {lb.get('step')} は最強 step {lb.get('best_step')} を更新できず（{lb.get('elo_vs_best'):+.1f} Elo、区間 {lb.get('ci95')}、足踏み {stall}）"
        out["items"].append({"name": "M2 最強比", "verdict": verdict, "why": why, "values": {k: lb.get(k) for k in ("step", "best_step", "elo_vs_best", "ci95", "improved", "stall")}})
    else:
        out["items"].append({"name": "M2 最強比", "verdict": NA, "why": "eval/best.jsonl がまだ無い", "values": {}})

    # M3 基準比
    anc = collect_anchor(sd)
    if anc:
        la = anc[-1]
        out["items"].append({"name": "M3 基準比", "verdict": OK if (la.get("ci95") or [None])[0] is not None and la["ci95"][0] > 0 else WARN,
                             "why": f"step {la.get('step')} は基準 step {la.get('anchor_step')} に {la.get('elo_vs_anchor'):+.1f} Elo（累積 {la.get('elo'):+.1f}、区間 {la.get('ci95')}）",
                             "values": {k: la.get(k) for k in ("step", "anchor_step", "elo_vs_anchor", "elo", "ci95")}})
    else:
        out["items"].append({"name": "M3 基準比", "verdict": NA, "why": "eval/anchor.jsonl がまだ無い", "values": {}})

    # M4 固定の参照: 参照ごとに、reference_hours の窓での伸び
    refs = collect_reference(sd)
    by_ref: dict[str, list[dict]] = {}
    for r in refs:
        by_ref.setdefault(str(r.get("ref")), []).append(r)
    for name, rs in sorted(by_ref.items()):
        recent = [r for r in rs if float(r.get("t", 0)) >= now - th["reference_hours"] * 3600]
        last = rs[-1]
        if len(recent) >= 2:
            rise = float(recent[-1]["elo"]) - float(recent[0]["elo"])
            verdict = OK if rise > 0 else WARN
            why = f"対 {name}: {recent[0]['elo']:+.1f} → {recent[-1]['elo']:+.1f} Elo（{th['reference_hours']:.0f} 時間で {rise:+.1f}）"
        else:
            verdict, why = NA, f"対 {name}: {last['elo']:+.1f} Elo（step {last.get('step')}、比べる点がまだ 1 つ）"
        out["items"].append({"name": f"M4 参照 {name}", "verdict": verdict, "why": why, "values": {"last": last, "n_recent": len(recent)}})
    if not by_ref:
        out["items"].append({"name": "M4 参照", "verdict": NA, "why": "eval/reference.jsonl がまだ無い", "values": {}})

    # 局/日
    st = json.loads(sd.status_json.read_text(encoding="utf-8")) if sd.status_json.exists() else {}
    gpd = st.get("games_per_day_1h")
    if gpd is not None:
        verdict = OK if th["gpd_min"] <= 0 or gpd >= th["gpd_min"] else WARN
        out["items"].append({"name": "局/日（1 時間平均）", "verdict": verdict, "why": f"{gpd:,}" + (f"（下限 {th['gpd_min']:,}）" if th["gpd_min"] > 0 else ""), "values": {"gpd": gpd}})

    order = {REVIEW: 3, WARN: 2, OK: 1, NA: 0}
    out["verdict"] = max((i["verdict"] for i in out["items"]), key=lambda v: order[v], default=NA)
    out["thresholds"] = th
    return out


def format_review(r: dict) -> str:
    lines = [f"review {r['run']} {r['time']}: {r['verdict']}"]
    for i in r["items"]:
        lines.append(f"  [{i['verdict']}] {i['name']}: {i['why']}")
    return "\n".join(lines)
