# SPDX-License-Identifier: Apache-2.0
"""打ち切り（借りたホストを途中で失うこと）でいくら損したかを数える（管理コンソールの「クラウド履歴」）。

入札（bid）で借りると同じホストでも on-demand より 16〜32% 安いが、ほかの利用者に競り落とされると途中で止められる
（decisions.md 2026-09-15）。止められると vast_worker.py が残りの時間で次のセッションを起動する（借り直し）。
その 1 回ごとにかかるのは 2 つ:

1. **余分な準備代**（`extra_setup_usd`）: 借りてからブリッジが動き出すまで（イメージの取得とビルドで 5〜15 分）は
   課金されるのに局が 1 つも出ない。最初の 1 回は打ち切られなくても払うので、**借り直したセッションのぶんだけ**を数える。
2. **打てなかった時間**（`lost_h`）: 最後に回収できた時刻から、次のセッションが打ち始めるまで。ホストの上で打っていた
   途中の局も回収できずに消えるので、この幅にそのセッションの局/日を掛けたものを失った局（`lost_games`）とみなす。
   借り直せなかったとき（残りが `MIN_CONTINUE_H` 未満、締め切り超過、候補なし）は予定の残り時間も丸ごと足す。

この 2 つから「打ち切りが無ければ 100 万局あたりいくらだったか」（`usd_per_1m_ideal`）が出る。実際の値との差が
入札の値引きより大きければ、入札で借りるのは損になる。`by_rent` はその比較（入札 / on-demand ごとの実効費用）。

使う記録はすべて既にあるもの（session.json の continues・hours、instance.json の t_rent・t_bridge、
result.json の lost、bridge.json の last_pull・time）で、新しく取るものは無い。
"""
from __future__ import annotations

import statistics

# 打ち切りの穴・余分な準備をこれ以上の長さだと見なさない（記録の取りこぼしで桁外れの値が出ないように）
MAX_HOLE_H = 6.0

RENT_LABELS = {"bid": "入札"}  # session.json の rent はコンソールの「借り方」の表示と綴りが違う


def _f(v) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _rate(rows: list[dict]) -> float:
    """局/日の代表値（そのセッションで測れなかったときに当てる中央値）。"""
    known = [_f(r.get("games_per_day")) for r in rows]
    known = [v for v in known if v]
    return statistics.median(known) if known else 0.0


def annotate(rows: list[dict]) -> list[dict]:
    """history() のセッションの行に、打ち切りの損の内訳を足した写しを返す（古い順に渡すこと）。

    足す欄: lost（打ち切られたか）・continues（借り直しの元）・setup_h / setup_usd（準備に払った時間と額）・
    extra_setup_usd（そのうち打ち切りのせいで余分に払ったぶん）・dark_h（回収が止まっていた時間）・
    unused_h（借り直せずに捨てた予定の残り）・lost_h（合計）・lost_games（打てなかった局の見積もり）。"""
    med = _rate(rows)
    nxt_of = {r["continues"]: r for r in rows if r.get("continues")}  # 打ち切られた回 → その借り直し
    out = []
    for r in rows:
        r = dict(r)
        t_rent, t_bridge, dph = _f(r.get("t_rent")), _f(r.get("t_bridge")), _f(r.get("dph"))
        both = t_rent is not None and t_bridge is not None
        r["setup_h"] = round(min(max((t_bridge - t_rent) / 3600, 0.0), MAX_HOLE_H), 3) if both else None
        r["setup_usd"] = round(r["setup_h"] * dph, 4) if r["setup_h"] is not None and dph else None
        r["extra_setup_usd"] = r["setup_usd"] if r.get("continues") and r["setup_usd"] else 0.0
        r["dark_h"] = r["unused_h"] = r["lost_h"] = 0.0
        r["lost_games"] = 0
        if r.get("lost"):
            t_end = _f(r.get("t_end"))
            frm = _f(r.get("last_pull"))
            frm = t_end if frm is None else frm
            nxt = nxt_of.get(r["name"])
            to = _f((nxt or {}).get("t_bridge"))
            to = t_end if to is None else to
            if frm is not None and to is not None:
                r["dark_h"] = round(min(max((to - frm) / 3600, 0.0), MAX_HOLE_H), 3)
            if nxt is None and t_bridge is not None and t_end is not None and _f(r.get("hours")):
                r["unused_h"] = round(max(_f(r["hours"]) - (t_end - t_bridge) / 3600, 0.0), 3)
            r["lost_h"] = round(r["dark_h"] + r["unused_h"], 3)
            r["lost_games"] = round(r["lost_h"] / 24 * (_f(r.get("games_per_day")) or med))
        out.append(r)
    return out


def _group(rows: list[dict], key: str) -> list[dict]:
    """借り方（rent）や GPU ごとに、打ち切りの割合と実効費用をまとめる（多い順）。"""
    by: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("bridge_h"):  # 借りて打ち始めた回だけ（オファーが無くて借りなかった回は数えない）
            name = str(r.get(key) or "-")
            by.setdefault(RENT_LABELS.get(name, name) if key == "rent" else name, []).append(r)
    out = []
    for name, rs in by.items():
        bridge_h = round(sum(_f(r["bridge_h"]) or 0.0 for r in rs), 2)
        lost = sum(1 for r in rs if r.get("lost"))
        dphs = [_f(r.get("dph")) for r in rs]
        out.append({"name": name, "sessions": len(rs), "bridge_h": bridge_h, "lost": lost,
                    "h_per_loss": round(bridge_h / lost, 2) if lost else None,
                    "dph": round(statistics.median([d for d in dphs if d]), 3) if any(dphs) else None,
                    **_time_lost(rs, bridge_h), **_cost(rs)})
    return sorted(out, key=lambda g: -g["sessions"])


def _time_lost(rows: list[dict], bridge_h: float) -> dict:
    """打てなかった時間と、それが経過時間（打った時間＋打てなかった時間）に占める割合。

    **お金の損とは別**（止められている間は課金されないので費用にはほとんど出ない）。局/日で見るとここが効く:
    入札で止められるたびに、借り直しが打ち始めるまでは 1 局も増えない。「同じ予算でいくつ買えるか」ではなく
    「今日いくつ稼げるか」を気にするときの物差し。"""
    lost_h = round(sum(_f(r.get("lost_h")) or 0.0 for r in rows), 2)
    total = bridge_h + lost_h
    return {"lost_h": lost_h, "lost_time_pct": round(lost_h / total * 100, 1) if total > 0 else None}


def _cost(rows: list[dict]) -> dict:
    """実際の 100 万局あたりの費用と、打ち切りが無かったときの値。"""
    usd = sum(_f(r.get("total_usd")) or 0.0 for r in rows)
    extra = sum(_f(r.get("extra_setup_usd")) or 0.0 for r in rows)
    net = sum(int(r.get("net_games") or 0) for r in rows)
    lost = sum(int(r.get("lost_games") or 0) for r in rows)
    actual = round(usd / net * 1e6, 2) if net else None
    ideal = round((usd - extra) / (net + lost) * 1e6, 2) if net + lost else None
    return {"total_usd": round(usd, 4), "extra_setup_usd": round(extra, 4), "net_games": net, "lost_games": lost,
            "usd_per_1m": actual, "usd_per_1m_ideal": ideal,
            "waste_pct": round((actual / ideal - 1) * 100, 1) if actual and ideal else None}


def summary(rows: list[dict]) -> dict:
    """annotate を済ませた行から、打ち切りの回数・頻度・損と、借り方ごと・GPU ごとの比較を出す。"""
    rented = [r for r in rows if r.get("bridge_h")]
    bridge_h = round(sum(_f(r["bridge_h"]) or 0.0 for r in rented), 2)
    lost = [r for r in rows if r.get("lost")]
    return {"sessions": len(rented), "bridge_h": bridge_h, "interruptions": len(lost),
            "relaunches": sum(1 for r in rows if r.get("continues")),
            "not_relaunched": sum(1 for r in lost if r.get("unused_h")),
            "h_per_loss": round(bridge_h / len(lost), 2) if lost else None,
            **_time_lost(rows, bridge_h),
            "unused_h": round(sum(_f(r.get("unused_h")) or 0.0 for r in rows), 2),
            **_cost(rows), "by_rent": _group(rows, "rent"), "by_gpu": _group(rows, "gpu")}
