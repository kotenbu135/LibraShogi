# SPDX-License-Identifier: Apache-2.0
"""`libra scaling`: 「局を何倍にすると何 Elo 伸びるか」を、保存済みの計測から出す（docs/scaling-2026-09-18.md §4）。

読むのは `eval/reference.jsonl`（固定の参照との Elo）と `metrics.jsonl`（step → 総局数）だけで、
対局はし直さない。局数は対数で効く（AlphaZero 系の伸びは局数の対数にほぼ直線）ので、区間の伸びは
**2 倍あたりの Elo** を主に出し、ユーザーの問い「100 万局あたり何 Elo」も同じ区間から併記する。

固定の参照は run をまたいで同じ相手なので絶対の物差しになるが、勝率が 0 か 1 に寄ると Elo が縮み、
伸びが止まったように見える（実際 9/18 の 80 万局の点で ls-v1 の参照は勝率 90% まで来ていた）。
そこで得点が BAND の中にある点だけを傾きの計算に使い、外れた点は「天井／床」と印を付ける。
参照が複数あるときは、両方が BAND の中にある点での差の平均で目盛りを合わせて 1 本の曲線にする
（差が測れない参照は目盛りを合わせずに別の曲線として出す）。
"""
from __future__ import annotations

import math

from .auto import ckpt_step, collect_reference, load_metrics
from .state import StateDir

# 得点がこの範囲の外だと Elo が縮む（200 局で得点 0.8 の区間は既に ±60 Elo）。出所は自分の判断で、
# 9/18 の実測で確かめた: 2 つの参照の差は、両方が 0.2〜0.8 に入る点（20 万・30 万局）では +274・+306 Elo
# だが、片方が 0.83・0.90 まで来た 40 万・80 万局では +244・+146 に潰れる。ls2-settings.md §5 の
# 「win1m は旧 ls に +297」とも合う。docs/method-evidence.md §5
BAND = (0.2, 0.8)
# 100 万局あたりの費用（ドル）。measurements.md 2026-09-17 の RTX 5090 2 台の実測 $7.87・$8.03
COST_PER_1M_USD = 8.0


def games_of_step(metrics: list[dict]):
    """step → 総局数。metrics.jsonl の (step, games_total) を直線でつなぐ。

    計測の記録の `games` は「打ち終わった時刻の総局数」で、その重みを保存した時点より後（9/18 の
    80 万局の点で約 5 万局ぶん）なので、x 軸には使わずに step から引き直す。"""
    rows = sorted(({"step": int(r["step"]), "games": int(r["games_total"])}
                   for r in metrics if r.get("step") is not None and r.get("games_total") is not None),
                  key=lambda r: r["step"])

    def f(step: int | None) -> int | None:
        if step is None or not rows:
            return None
        if step <= rows[0]["step"]:
            return rows[0]["games"]
        for a, b in zip(rows, rows[1:]):
            if a["step"] <= step <= b["step"]:
                if b["step"] == a["step"]:
                    return a["games"]
                w = (step - a["step"]) / (b["step"] - a["step"])
                return round(a["games"] + w * (b["games"] - a["games"]))
        return rows[-1]["games"]

    return f


def time_of_step(metrics: list[dict]):
    """step → その step だったときの時刻（Unix 秒）。metrics.jsonl の (step, t) を直線でつなぐ。

    `games_of_step` と同じ引き直し。管理コンソールの Elo のグラフで横軸を「時間」にしたときに、
    step でしか分からない点（Bradley-Terry の目盛り）を置く場所を出すために使う。"""
    rows = sorted(({"step": int(r["step"]), "t": float(r["t"])}
                   for r in metrics if r.get("step") is not None and r.get("t") is not None),
                  key=lambda r: r["step"])

    def f(step: int | None) -> float | None:
        if step is None or not rows:
            return None
        if step <= rows[0]["step"]:
            return rows[0]["t"]
        for a, b in zip(rows, rows[1:]):
            if a["step"] <= step <= b["step"]:
                if b["step"] == a["step"]:
                    return a["t"]
                w = (step - a["step"]) / (b["step"] - a["step"])
                return round(a["t"] + w * (b["t"] - a["t"]), 1)
        return rows[-1]["t"]

    return f


def _point(r: dict, g_of, band: tuple[float, float]) -> dict:
    score = r.get("score_new")
    games = g_of(r.get("step"))
    return {
        "step": r.get("step"), "games": games, "elo": r.get("elo"), "ci95": r.get("ci95"),
        "n": r.get("n"), "score": score,
        "in_band": (score is not None and band[0] <= float(score) <= band[1]),
    }


def _self_point(sd: StateDir, ref: str, g_of) -> dict | None:
    """参照がこの run 自身の archive なら、その局数での「自分対自分 ＝ 0 Elo」の点。

    測らなくても正しい点なので、参照を入れ替えたときに**新しい参照を古い目盛りに必ずつなげる**。
    新しい参照は足した時点では強すぎて（古い参照が天井に着いている）、他の参照と帯の中で重なる
    点が取れないことがあるため（9/18 の 80 万局の archive を 3 つ目の参照に足したときの実例）。"""
    f = sd.root / "checkpoints" / "archive" / ref
    step = ckpt_step(ref)
    if step is None or not f.exists():
        return None
    games = g_of(step)
    if games is None:
        return None
    return {"step": step, "games": games, "elo": 0.0, "ci95": [0.0, 0.0], "n": 0,
            "score": 0.5, "in_band": True, "self": True}


def reference_points(sd: StateDir, band: tuple[float, float] = BAND) -> dict[str, list[dict]]:
    """参照ごとの点（総局数の順）。"""
    g_of = games_of_step(load_metrics(sd, 100000))
    out: dict[str, list[dict]] = {}
    for r in collect_reference(sd):
        ref = r.get("ref")
        if ref is None or r.get("elo") is None:
            continue
        p = _point(r, g_of, band)
        if p["games"] is None:
            continue
        out.setdefault(ref, []).append(p)
    for ref, pts in out.items():
        sp = _self_point(sd, ref, g_of)
        if sp is not None and all(q["games"] != sp["games"] for q in pts):
            pts.append(sp)
        pts.sort(key=lambda p: p["games"])
    return out


def intervals(pts: list[dict]) -> list[dict]:
    """隣り合う点の間の伸び。得点が BAND の外の点をまたぐ区間には `in_band` False を付ける。"""
    out = []
    for a, b in zip(pts, pts[1:]):
        if not a["games"] or not b["games"] or b["games"] <= a["games"]:
            continue
        d_elo = float(b["elo"]) - float(a["elo"])
        dbl = math.log2(b["games"] / a["games"])
        out.append({
            "games_from": a["games"], "games_to": b["games"], "d_elo": round(d_elo, 1),
            "doublings": round(dbl, 3),
            "elo_per_doubling": round(d_elo / dbl, 1) if dbl > 0 else None,
            "elo_per_1m": round(d_elo / (b["games"] - a["games"]) * 1e6, 1),
            "in_band": bool(a["in_band"] and b["in_band"]),
        })
    return out


def fit(pts: list[dict]) -> dict | None:
    """log2(局数) に対する Elo の直線当てはめ。傾きが「2 倍あたりの Elo」。"""
    xy = [(math.log2(p["games"]), float(p["elo"])) for p in pts if p["games"]]
    n = len(xy)
    if n < 2:
        return None
    sx = sum(x for x, _ in xy)
    sy = sum(y for _, y in xy)
    sxx = sum(x * x for x, _ in xy)
    sxy = sum(x * y for x, y in xy)
    den = n * sxx - sx * sx
    if den == 0:
        return None
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    resid = [y - (a + b * x) for x, y in xy]
    return {"elo_per_doubling": round(b, 1), "intercept": round(a, 1), "n": n,
            "rms_resid": round(math.sqrt(sum(r * r for r in resid) / n), 1),
            "games_from": min(p["games"] for p in pts), "games_to": max(p["games"] for p in pts)}


def _pair_offset(a_pts: list[dict], b_pts: list[dict]) -> float | None:
    """2 つの参照の目盛りの差（a − b）。**両方が帯の中にある**同じ局数の点の差の平均。

    片方が天井・床の点は使わない（縮んだ Elo から出した差は小さく出る。9/18 の実測で +290 が +146 に潰れた）。
    重なりが無ければ None。"""
    by = {p["games"]: p for p in a_pts if p["in_band"]}
    ds = [float(by[p["games"]]["elo"]) - float(p["elo"]) for p in b_pts if p["in_band"] and p["games"] in by]
    return sum(ds) / len(ds) if ds else None


def stitch(refs: dict[str, list[dict]]) -> dict:
    """複数の参照を 1 本の曲線にする。BAND の中の点がいちばん多い参照を目盛りの土台にし、
    他の参照はその差だけずらす。

    差は**間の参照をたどって**求める（A と C が直に重ならなくても、A−B と B−C が測れていれば C は乗る）。
    参照は強くなるたびに入れ替える（弱くなったものは天井に着いて外す）ので、古い参照と新しい参照が
    同じ局数で両方とも帯の中に入ることは無くなる。どこにもつながらない参照だけ落とす。"""
    if not refs:
        return {"base": None, "points": [], "offsets": {}, "dropped": [], "via": {}}
    base = max(sorted(refs), key=lambda k: (sum(1 for p in refs[k] if p["in_band"]), len(refs[k])))
    offsets, via, frontier = {base: 0.0}, {base: None}, [base]
    while frontier:
        cur = frontier.pop(0)
        for ref in sorted(refs):
            if ref in offsets:
                continue
            d = _pair_offset(refs[cur], refs[ref])  # cur − ref
            if d is None:
                continue
            offsets[ref] = round(offsets[cur] + d, 1)
            via[ref] = cur
            frontier.append(ref)
    dropped = [r for r in sorted(refs) if r not in offsets]
    merged: dict[int, dict] = {}
    for ref, pts in refs.items():
        if ref in dropped:
            continue
        for p in pts:
            if not p["in_band"]:
                continue
            # 土台の目盛りに乗せる（ずらしは「土台 − この参照」なので足す）
            q = {**p, "elo": round(float(p["elo"]) + offsets[ref], 1), "ref": ref}
            # 同じ局数に 2 つあれば、得点が 0.5 に近いほう（Elo の縮みが小さいほう）を採る。
            # 同点なら土台の参照を採る（参照の並び順で結果が変わらないように）
            rank = (abs(float(p["score"]) - 0.5), ref != base)
            old = merged.get(p["games"])
            if old is None or rank < (abs(float(old["score"]) - 0.5), old["ref"] != base):
                merged[p["games"]] = q
    return {"base": base, "points": [merged[g] for g in sorted(merged)], "offsets": offsets,
            "dropped": dropped, "via": via}


def scaling(sd: StateDir, cost_per_1m: float = COST_PER_1M_USD, band: tuple[float, float] = BAND) -> dict:
    refs = reference_points(sd, band)
    st = stitch(refs)
    curve = st["points"]
    out: dict = {
        "band": list(band),
        "cost_per_1m_usd": cost_per_1m,
        "references": {ref: {"points": pts, "intervals": intervals(pts), "fit": fit([p for p in pts if p["in_band"]])}
                       for ref, pts in refs.items()},
        "curve": {"base": st["base"], "offsets": st["offsets"], "dropped": st["dropped"], "via": st["via"],
                  "points": curve, "intervals": intervals(curve), "fit": fit(curve)},
        "outlook": None,
        "notes": [],
    }
    f = out["curve"]["fit"]
    if f and curve:
        g = curve[-1]["games"]
        slope = f["elo_per_doubling"]
        out["outlook"] = {
            "games_now": g,
            "elo_per_doubling": slope,
            # 今から 100 万局買い足したときの見込み（対数なので局数が増えるほど下がる）
            "next_1m_elo": round(slope * math.log2((g + 1_000_000) / g), 1),
            "double_games": g, "double_elo": slope,
            "double_cost_usd": round(g / 1e6 * cost_per_1m, 2),
            "usd_per_elo": (round(g / 1e6 * cost_per_1m / slope, 3) if slope > 0 else None),
        }
    ib = [iv for iv in out["curve"]["intervals"] if iv["in_band"]]
    if len(ib) < 3:
        out["notes"].append(f"天井・床に触れていない区間が {len(ib)} 本しかない（3 本以上で傾きの向きを見る）")
    elif ib[-1]["elo_per_doubling"] is not None and ib[0]["elo_per_doubling"] is not None:
        if ib[-1]["elo_per_doubling"] < ib[0]["elo_per_doubling"] * 0.6:
            out["notes"].append("直近の 2 倍あたりの Elo が最初の区間の 6 割を下回った（局を足す効きが落ちている）")
    for ref in st["dropped"]:
        out["notes"].append(
            f"参照 {ref} は他のどの参照とも帯の中で重ならないので曲線に乗せられない"
            "（重なる局数で両方を測るか、間をつなぐ参照を残す）")
    for ref, r in out["references"].items():
        last = r["points"][-1] if r["points"] else None
        if last and not last["in_band"]:
            out["notes"].append(f"参照 {ref} は最新の点が得点 {last['score']} で天井・床（この参照の伸びは実際より小さく出る）")
    # 次の節目で測れなくなる合図: 最新の局数で帯の中に残っている参照が、どれも 0.5 から遠い
    if curve:
        last_g = curve[-1]["games"]
        room = [p for p in curve if p["games"] == last_g and abs(float(p["score"]) - 0.5) <= 0.25]
        if not room:
            out["notes"].append(
                f"最新の {last_g:,} 局で得点が 0.25〜0.75 に収まる参照が無い（次の節目では全部の参照が天井に着く）。"
                "今の重みを新しい固定の参照に足すと曲線を続けられる")
    return out


def render(res: dict) -> str:
    L = []
    for ref, r in res["references"].items():
        L.append(f"== 参照 {ref} ==")
        L.append("     総局数 |     Elo | 得点 |  局数 | 帯")
        for p in r["points"]:
            ci = p["ci95"] or [None, None]
            ci_s = f" [{ci[0]:+.0f}, {ci[1]:+.0f}]" if ci[0] is not None else ""
            L.append(f"{p['games']:>11,} | {float(p['elo']):+7.1f}{ci_s:>18} | {p['score']} | {p['n']} | {'' if p['in_band'] else '天井/床'}")
        for iv in r["intervals"]:
            L.append(f"  {iv['games_from']:>9,} → {iv['games_to']:>9,}  {iv['d_elo']:+7.1f} Elo"
                     f"  2 倍あたり {iv['elo_per_doubling']:+7.1f}  100 万局あたり {iv['elo_per_1m']:+8.1f}"
                     f"{'' if iv['in_band'] else '  （天井/床をまたぐ）'}")
        if r["fit"]:
            f = r["fit"]
            L.append(f"  帯の中の {f['n']} 点の当てはめ: {f['elo_per_doubling']:+.1f} Elo / 2 倍（残差 {f['rms_resid']}）")
        L.append("")
    c = res["curve"]
    if c["points"]:
        L.append(f"== 1 本にした曲線（目盛りは参照 {c['base']}。ずらし {c['offsets']}）==")
        chain = {k: v for k, v in (c.get("via") or {}).items() if v is not None and v != c["base"]}
        if chain:
            L.append("  間の参照をたどったずらし: " + "、".join(f"{k} ← {v}" for k, v in chain.items()))
        for p in c["points"]:
            L.append(f"{p['games']:>11,} | {float(p['elo']):+7.1f} | 得点 {p['score']} | {p['ref']}")
        for iv in c["intervals"]:
            L.append(f"  {iv['games_from']:>9,} → {iv['games_to']:>9,}  {iv['d_elo']:+7.1f} Elo"
                     f"  2 倍あたり {iv['elo_per_doubling']:+7.1f}  100 万局あたり {iv['elo_per_1m']:+8.1f}")
        if c["fit"]:
            f = c["fit"]
            L.append(f"  {f['n']} 点の当てはめ: {f['elo_per_doubling']:+.1f} Elo / 2 倍（残差 {f['rms_resid']}）")
    o = res["outlook"]
    if o:
        L.append("")
        L.append(f"見込み: 今 {o['games_now']:,} 局。局を 2 倍（+{o['double_games']:,} 局、${o['double_cost_usd']}）にすると "
                 f"{o['double_elo']:+.1f} Elo、1 Elo あたり ${o['usd_per_elo']}。"
                 f"100 万局の買い足しは {o['next_1m_elo']:+.1f} Elo の見込み")
    for n in res["notes"]:
        L.append(f"注意: {n}")
    return "\n".join(L)
