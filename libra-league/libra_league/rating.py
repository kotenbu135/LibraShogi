# SPDX-License-Identifier: Apache-2.0
"""全部の対局から一度に Elo を出す（Bradley-Terry）。

鎖（`anchor.jsonl` の累積）や「1 つの参照との差」は、相手が替わるたびに誤差と**非推移性**が積み上がる。
2026-09-18 の 120 万局の点では、直前の自分に +166.7 Elo なのに古い相手には伸びず、足し算が 126〜181 Elo
合わなかった（docs/measurements.md 同日）。そこで**記録した対局を全部まとめて 1 つの目盛りに当てはめる**。

- 手法: Bradley-Terry の最尤推定を Hunter (2004) の MM 反復で解く。式は Caron & Doucet の式（Hunter の
  MM を引用）どおり `λ_i ← w_i / Σ_{j≠i} n_ij / (λ_i + λ_j)`（a=1, b=0 が最尤）。Elo = (400/ln10)·ln λ。
  Hunter の原典は Annals of Statistics 32(1) だが本文が読めなかった（Incapsula）ので、式は Caron &
  Doucet (2012) の引用で確かめた（**原典は未確認**）。docs/method-evidence.md §2.9。
- **引き分けは半勝**（得点をそのまま小数の勝ち数にする）。Hunter の模型は勝ち負けだけで、引き分けを
  正しく扱うには Davidson (1970) の追加の母数が要る。ここでは Elo の慣習どおり半勝にした。理由は
  引き分けが全体の 0.3% ほど（1,000 局で千日手 2・320 手 1）で、追加の母数を置くだけの情報が無いため。
- **区間**は Fisher 情報行列（重み n_ij·p_ij·p_ji のラプラシアン）の逆から出す。土台の 1 点を固定して解く。
- **非推移性**は当てはめの残差で測る。z_ij = (w_ij − n_ij p_ij) / sqrt(n_ij p_ij p_ji) の二乗和が自由度
  （組の数 − (点の数 − 1)）より大きければ、Elo の 1 本の目盛りでは説明しきれない（じゃんけん）。
"""
from __future__ import annotations

import math
import re

import numpy as np

from .auto import collect_anchor, collect_best, collect_matches, collect_reference
from .state import StateDir

# Elo = SCALE * ln λ
SCALE = 400.0 / math.log(10.0)
# 同じ組・同じ局数・同じ得点の行がこの秒数の中に 2 つあれば、基準比が最強比の結果を写したもの（auto.py の
# reuse）とみなして 1 つだけ数える。別々に打った結果がぴったり同じ得点になることは 1,000 局ではまず無い
DEDUP_WINDOW_S = 6 * 3600.0


_STEP_NODE_RE = re.compile(r"^step ([\d,]+)$")


def step_node(step: int | None) -> str | None:
    """この run の archive の点の名前。step だけで決まるので、同じ重みは必ず同じ点にまとまる。"""
    return None if step is None else f"step {int(step):,}"


def node_step(name: str) -> int | None:
    """点の名前から step を読む（外の相手なら None）。"""
    m = _STEP_NODE_RE.match(str(name))
    return int(m.group(1).replace(",", "")) if m else None


def _ref_node(sd: StateDir, name: str, ref_step: int | None) -> str:
    """参照の名前を点の名前にする。この run 自身の archive なら step の点と同じ点にまとめる
    （そうしないと「80 万局の自分」が別人として二重に並び、目盛りがつながらない）。"""
    if ref_step is not None and (sd.root / "checkpoints" / "archive" / name).exists():
        return step_node(ref_step) or name
    return name


def pairs_of(sd: StateDir) -> list[dict]:
    """記録した対局を (a, b, 局数, a の得点) の組にする。a は測った側（新しい重み）。"""
    out: list[dict] = []

    def add(a, b, n, score_a, t, src):
        if a is None or b is None or a == b or not n or score_a is None:
            return
        out.append({"a": a, "b": b, "n": int(n), "score_a": float(score_a), "t": float(t or 0.0), "src": src})

    for r in collect_best(sd):
        add(step_node(r.get("step")), step_node(r.get("best_step")), r.get("n"), r.get("score_new"), r.get("t"), "best")
    for r in collect_anchor(sd):
        add(step_node(r.get("step")), step_node(r.get("anchor_step")), r.get("n"), r.get("score_new"), r.get("t"), "anchor")
    for r in collect_reference(sd):
        ref = str(r.get("ref") or "")
        if ref:
            add(step_node(r.get("step")), _ref_node(sd, ref, r.get("ref_step")), r.get("n"), r.get("score_new"),
                r.get("t"), "reference")
    for m in collect_matches(sd):
        n = m.get("n")
        pts = m.get("a_points")
        score = (float(pts) / float(n)) if (n and pts is not None) else m.get("winrate")
        add(step_node(m.get("libra_step")), str(m.get("opponent") or "外部の相手"), n, score, m.get("time"), "match")

    out.sort(key=lambda p: p["t"])
    # 基準比が最強比の結果を写した行（同じ組・同じ局数・同じ得点）を 1 つにする
    seen: dict[tuple, float] = {}
    uniq = []
    for p in out:
        key = (p["a"], p["b"], p["n"], round(p["score_a"], 6))
        prev = seen.get(key)
        if prev is not None and abs(p["t"] - prev) <= DEDUP_WINDOW_S:
            continue
        seen[key] = p["t"]
        uniq.append(p)
    return uniq


def _strong_components(nodes: list[str], beat: set[tuple[int, int]]) -> list[list[int]]:
    """i が j に 1 局でも勝っていれば i→j の有向辺。最尤推定が定まる条件は強連結（Hunter 2004）。"""
    k = len(nodes)
    adj: list[list[int]] = [[] for _ in range(k)]
    rev: list[list[int]] = [[] for _ in range(k)]
    for i, j in beat:
        adj[i].append(j)
        rev[j].append(i)
    order, seen = [], [False] * k
    for s in range(k):                       # Kosaraju の 1 段目（再帰を避けて明示の山で回す）
        if seen[s]:
            continue
        stack = [(s, 0)]
        seen[s] = True
        while stack:
            v, idx = stack.pop()
            if idx < len(adj[v]):
                stack.append((v, idx + 1))
                w = adj[v][idx]
                if not seen[w]:
                    seen[w] = True
                    stack.append((w, 0))
            else:
                order.append(v)
    comp, done = [], [False] * k
    for v in reversed(order):                # 2 段目
        if done[v]:
            continue
        group, stack = [], [v]
        done[v] = True
        while stack:
            x = stack.pop()
            group.append(x)
            for y in rev[x]:
                if not done[y]:
                    done[y] = True
                    stack.append(y)
        comp.append(sorted(group))
    return comp


def fit(pairs: list[dict], anchor: str | None = None, max_iter: int = 10000, tol: float = 1e-12) -> dict:
    """Bradley-Terry の最尤推定（Hunter 2004 の MM）。土台の点を 0 Elo に置いた Elo を返す。"""
    empty = {"nodes": {}, "anchor": None, "pairs": [], "dropped": [], "fit": None, "converged": True, "iters": 0}
    if not pairs:
        return empty
    names = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    N = np.zeros((k, k))
    W = np.zeros((k, k))                     # W[i][j] = i が j から取った得点（局数 × 得点）
    for p in pairs:
        i, j = idx[p["a"]], idx[p["b"]]
        n, s = float(p["n"]), float(p["score_a"])
        N[i, j] += n
        N[j, i] += n
        W[i, j] += n * s
        W[j, i] += n * (1.0 - s)
    beat = {(i, j) for i in range(k) for j in range(k) if i != j and W[i, j] > 0}
    comps = _strong_components(names, beat)
    # 最尤推定が定まるのは強連結な塊の中だけ。土台を含む（無ければいちばん大きい）塊だけを解く
    if anchor in idx:
        keep = next(c for c in comps if idx[anchor] in c)
    else:
        keep = max(comps, key=lambda c: (len(c), -min(c)))
    dropped = [names[i] for i in range(k) if i not in set(keep)]
    sel = sorted(keep)
    names = [names[i] for i in sel]
    N, W = N[np.ix_(sel, sel)], W[np.ix_(sel, sel)]
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    if k < 2:
        return {**empty, "dropped": dropped}
    if anchor not in idx:
        anchor = names[0]

    w = W.sum(axis=1)                        # 各点の総得点
    lam = np.ones(k)
    converged, iters = False, 0
    for iters in range(1, max_iter + 1):
        den = np.zeros(k)
        for i in range(k):
            nz = N[i] > 0
            den[i] = np.sum(N[i][nz] / (lam[i] + lam[nz]))
        new = np.where(den > 0, w / np.maximum(den, 1e-300), lam)
        new = np.maximum(new, 1e-300)
        new /= new[idx[anchor]]              # 土台を 1（＝0 Elo）に置く
        d = float(np.max(np.abs(np.log(new) - np.log(lam))))
        lam = new
        if d < tol:
            converged = True
            break

    theta = np.log(lam)
    elo = SCALE * (theta - theta[idx[anchor]])

    # 区間: Fisher 情報行列（重み n_ij p_ij p_ji のラプラシアン）の、土台を除いた部分の逆
    P = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j and N[i, j] > 0:
                P[i, j] = lam[i] / (lam[i] + lam[j])
    Wt = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j and N[i, j] > 0:
                Wt[i, j] = N[i, j] * P[i, j] * P[j, i]
    L = np.diag(Wt.sum(axis=1)) - Wt
    free = [i for i in range(k) if i != idx[anchor]]
    sd_elo = np.zeros(k)
    try:
        cov = np.linalg.inv(L[np.ix_(free, free)])
        for pos, i in enumerate(free):
            sd_elo[i] = SCALE * math.sqrt(max(float(cov[pos, pos]), 0.0))
    except np.linalg.LinAlgError:
        sd_elo[:] = float("nan")

    # 当てはまり（じゃんけん度）: 標準化した残差
    resid, chi2 = [], 0.0
    for i in range(k):
        for j in range(i + 1, k):
            n = N[i, j]
            if n <= 0:
                continue
            exp = n * P[i, j]
            var = n * P[i, j] * P[j, i]
            z = (W[i, j] - exp) / math.sqrt(var) if var > 0 else 0.0
            chi2 += float(z * z)
            resid.append({"a": names[i], "b": names[j], "n": int(round(n)),
                          "score_obs": round(float(W[i, j]) / float(n), 4), "score_fit": round(float(P[i, j]), 4), "z": round(float(z), 2)})
    n_pairs = len(resid)
    df = max(n_pairs - (k - 1), 0)
    resid.sort(key=lambda r: -abs(r["z"]))

    nodes = {}
    for i, nm in enumerate(names):
        g = float(N[i].sum())
        nodes[nm] = {"elo": round(float(elo[i]), 1),
                     "ci95": None if math.isnan(sd_elo[i]) else [round(float(elo[i] - 1.96 * sd_elo[i]), 1),
                                                                 round(float(elo[i] + 1.96 * sd_elo[i]), 1)],
                     "games": int(round(g)), "opponents": int((N[i] > 0).sum()), "step": node_step(nm)}
    return {"nodes": nodes, "anchor": anchor, "pairs": resid, "dropped": dropped,
            "fit": {"n_nodes": k, "n_pairs": n_pairs, "chi2": round(float(chi2), 1), "df": df,
                    "chi2_per_df": (round(float(chi2) / df, 2) if df > 0 else None),
                    "rms_z": round(math.sqrt(float(chi2) / n_pairs), 2) if n_pairs else None},
            "converged": converged, "iters": iters}


def rating(sd: StateDir, anchor: str | None = None) -> dict:
    """この run の記録から Elo を出す。土台は既定でいちばん古い step の点。"""
    ps = pairs_of(sd)
    if anchor is None:
        steps = sorted({node_step(p[x]) for p in ps for x in ("a", "b") if node_step(p[x]) is not None})
        anchor = step_node(steps[0]) if steps else None
    r = fit(ps, anchor=anchor)
    r["n_evals"] = len(ps)
    return r


def render(r: dict) -> str:
    L = ["== 全部の対局から出した Elo（Bradley-Terry。土台 " + str(r.get("anchor")) + " を 0 とする）=="]
    if not r.get("nodes"):
        return "対局の記録がまだ無い"
    rows = sorted(r["nodes"].items(), key=lambda kv: (kv[1]["step"] is None, kv[1]["step"] or 0, kv[1]["elo"]))
    L.append("            点 |     Elo | 95% 区間          | 局数 | 相手")
    for nm, v in rows:
        ci = v["ci95"] or [None, None]
        ci_s = f"[{ci[0]:+.0f}, {ci[1]:+.0f}]" if ci[0] is not None else "—"
        L.append(f"{nm:>14} | {v['elo']:+7.1f} | {ci_s:>17} | {v['games']:>4} | {v['opponents']}")
    f = r.get("fit") or {}
    if f:
        L.append("")
        L.append(f"  当てはまり: {f['n_pairs']} 組・{f['n_nodes']} 点、χ²/自由度 {f.get('chi2_per_df')}"
                 f"（1 なら Elo の 1 本の目盛りで説明できる。大きいほどじゃんけんが強い）")
    worst = (r.get("pairs") or [])[:3]
    for p in worst:
        L.append(f"  ずれの大きい組: {p['a']} 対 {p['b']} 実測 {p['score_obs']} / 当てはめ {p['score_fit']}"
                 f"（{p['n']} 局、z {p['z']:+.1f}）")
    if r.get("dropped"):
        L.append("  つながらない点（全勝か全敗で Elo が決まらない）: " + "、".join(r["dropped"]))
    if not r.get("converged"):
        L.append("  ※ 収束しなかった（反復 {} 回）".format(r.get("iters")))
    return "\n".join(L)


def curve(sd: StateDir, r: dict | None = None) -> dict:
    """この run の archive の Elo を総局数の順に並べ、伸びの傾き（2 倍あたりの Elo）を出す。

    目盛りは `fit` が全部の対局から一度に決めたもので、参照を入れ替えても鎖を継ぎ足さない。"""
    from .auto import load_metrics
    from .scaling import fit as line_fit
    from .scaling import games_of_step, intervals

    r = rating(sd) if r is None else r
    g_of = games_of_step(load_metrics(sd, 100000))
    pts = []
    for nm, v in (r.get("nodes") or {}).items():
        st = v.get("step")
        if st is None:
            continue
        g = g_of(st)
        if g is None:
            continue
        pts.append({"step": st, "games": g, "elo": v["elo"], "ci95": v["ci95"], "n": v["games"],
                    "opponents": v["opponents"], "score": 0.5, "in_band": True, "node": nm})
    pts.sort(key=lambda p: p["games"])
    return {"points": pts, "intervals": intervals(pts), "fit": line_fit(pts),
            "anchor": r.get("anchor"), "misfit": (r.get("fit") or {}).get("chi2_per_df")}
