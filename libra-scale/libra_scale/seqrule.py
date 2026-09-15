# SPDX-License-Identifier: Apache-2.0
"""逐次の検証対局の規則（docs/decisions.md 2026-09-15）。エンジンを使わない関数だけを置く（実行は seqrun.py）。

組は鏡映の代表で数え、局は代表と鏡映の両方の向きで打つ（5 筋どうしの組は同じ向きが 2 回）。得点は先手から見て勝ち 1・引き分け 0.5。

選ぶ側（1 段目）: 組ごとに min_games 局打ってから look_every 局ごとに見て、|w − 0.5| > 1.96·se なら「有意」（sig）、
  1.96·se ≤ eps なら「精度」（eps）で打ち切る。毎局見ると真の w が 0.5 の組の 46% が有意で止まる（名目は 5%）ので、
  見る間隔を粗くした（100 局から 50 局ごとで 29%。w 0.52 で逆を選ぶのは 5.1%、補正して名目どおりにすると局数が 1.6 倍で損は同じ）。
  対称な組（線対称・点対称）は有意では止めず eps まで打つ（後手に傾くかを同じ精度で見るため）。
置く側（2 段目、eps_place > 0 のとき）: eps で止まり区間が 0.5 を含んだ組（候補）だけを打ち足し、place_look_every 局ごとに
  |w − 0.5| − 1.96·se > eps_place なら外し（out）、1.96·se ≤ eps_place で打ち切る（done）。
アラート: 対称な組が後手に傾いたら出す。途中は z 3.0（見る回数が多いので厳しく）、eps で止まった組と群は 95% 区間で確定。
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

from .pairs import GOTE_SQUARES_ALL, SENTE_SQUARES, canonical, from_usi, is_immediate_loss, mirror_sq, sq, usi
from .table import expand_mirrors

Z = 1.96
Z_EARLY = 3.0
GROUP_MIN_GAMES = 500
GROUP_LABELS = {"five": "5 筋の同じ高さ", "line": "線対称", "point": "点対称", "all": "対称な組すべて"}


@dataclass
class Rule:
    eps: float = 0.02           # 選ぶ側: 95% 区間の半幅がこれ以下で打ち切る
    min_games: int = 100        # 最初に見る局数
    look_every: int = 50        # 以後に見る間隔
    max_games: int = 3000       # 1 段目の安全弁（eps 0.02 は約 2,401 局で届く）
    eps_place: float = 0.0      # 置く側の精度（0 なら 2 段目なし）
    place_look_every: int = 500

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def place_max(self) -> int:
        return int(math.ceil((Z * 0.5 / self.eps_place) ** 2 * 1.25)) if self.eps_place > 0 else 0


def key_of(kb: int, kw: int) -> str:
    kb, kw = canonical(kb, kw)
    return f"{usi(kb)} {usi(kw)}"


def parse_key(k: str) -> tuple[int, int]:
    a, b = k.split()
    return from_usi(a), from_usi(b)


def new_state(rule: Rule) -> dict:
    return {"n": 0, "sente": 0, "draw": 0, "gote": 0, "next_look": rule.min_games,
            "stop": None, "stop_n": 0, "stop_w": None, "stop_se": None, "place": None, "place_n": 0, "place_next": 0}


def add_result(st: dict, result: int) -> None:
    st["n"] += 1
    st["sente" if result > 0 else "draw" if result == 0 else "gote"] += 1


def stats_counts(n: int, sente: int, draw: int) -> tuple[float, float]:
    """(先手の得点, 標準誤差)。分散は勝ち・引き分け・負けの 3 値の標本分散 w(1−w) − d/4 に、端の組で 0 にならないよう
    (S+2)/(n+4) で作った下限を掛ける。n = 0 なら (0.5, inf)。"""
    if n <= 0:
        return 0.5, math.inf
    m = (sente + 0.5 * draw) / n
    dq = draw / n / 4
    pt = (sente + 0.5 * draw + 2) / (n + 4)
    return m, math.sqrt(max(m * (1 - m) - dq, pt * (1 - pt) - dq, 1e-12) / n)


def stats(st: dict) -> tuple[float, float]:
    return stats_counts(st["n"], st["sente"], st["draw"])


def ci95(m: float, se: float) -> list[float] | None:
    return None if math.isinf(se) else [round(m - Z * se, 4), round(m + Z * se, 4)]


def is_candidate(st: dict) -> bool:
    """置く側の候補: 精度で止まり、そのときの 95% 区間が 0.5 を含んだ組。"""
    return st["stop"] == "eps" and abs(st["stop_w"] - 0.5) <= Z * st["stop_se"]


def is_active(st: dict, rule: Rule) -> bool:
    return st["stop"] is None or (rule.eps_place > 0 and is_candidate(st) and st["place"] is None)


def _next(look: int, n: int, every: int) -> int:
    while look <= n:
        look += every
    return look


def update(st: dict, rule: Rule, symmetric: bool) -> str | None:
    """見る局数に達していれば規則を当てる。打ち切ったら理由（sig / eps / cap / place_out / place_done / place_cap）を返す。
    打ち切った後に届いた局（同時に進んでいた対局）も局数には足すが、打ち切りの判定は変えない。"""
    n = st["n"]
    if st["stop"] is None:
        if n < st["next_look"]:
            return None
        m, se = stats(st)
        if not symmetric and abs(m - 0.5) > Z * se:
            why = "sig"
        elif Z * se <= rule.eps:
            why = "eps"
        elif n >= rule.max_games:
            why = "cap"
        else:
            st["next_look"] = _next(st["next_look"], n, rule.look_every)
            return None
        st.update(stop=why, stop_n=n, stop_w=m, stop_se=se)
        return why
    if rule.eps_place > 0 and is_candidate(st) and st["place"] is None:
        if st["place_next"] <= st["stop_n"]:
            st["place_next"] = st["stop_n"] + rule.place_look_every
        if n < st["place_next"]:
            return None
        m, se = stats(st)
        if abs(m - 0.5) - Z * se > rule.eps_place:
            why = "out"
        elif Z * se <= rule.eps_place:
            why = "done"
        elif n >= rule.place_max():
            why = "cap"
        else:
            st["place_next"] = _next(st["place_next"], n, rule.place_look_every)
            return None
        st.update(place=why, place_n=n)
        return "place_" + why
    return None


def symmetric_groups() -> dict[str, list[str]]:
    """対称な置き方の代表の組。線対称は同じ筋で段が逆（▲2八 △2二）、点対称は盤の中心について対称（▲2八 △8二）、
    5 筋の同じ高さ（▲5九 △5一 など）は両方に入る。後手玉が四段目の組（先手玉が六段目）は剪定済みで含まない。"""
    five, line, point = set(), set(), set()
    for f in range(9):
        for r in (6, 7, 8):  # 七〜九段目。後手玉は三〜一段目
            line.add(key_of(sq(f, r), sq(f, 8 - r)))
            point.add(key_of(sq(f, r), sq(8 - f, 8 - r)))
            if f == 4:
                five.add(key_of(sq(f, r), sq(f, 8 - r)))
    return {"five": sorted(five), "line": sorted(line), "point": sorted(point)}


def active_keys(states: dict[str, dict], rule: Rule, symmetric: set[str]) -> list[str]:
    """打つ組。対称な組の 1 段目が残っていればそれだけを打つ（後手に傾くかを先に知るため。全体の局数は変わらない）。"""
    act = [k for k, st in states.items() if is_active(st, rule)]
    sym = [k for k in act if k in symmetric and states[k]["stop"] is None]
    return sorted(sym or act)


def opening_lines(keys: list[str]) -> list[list[str]]:
    """自己対局エンジンの布石（玉 2 手だけの手順）。代表と鏡映を 1 本ずつ入れ、どの組も同じ確率で当たるようにする。"""
    out = []
    for k in keys:
        kb, kw = parse_key(k)
        out.append([f"K*{usi(kb)}", f"K*{usi(kw)}"])
        out.append([f"K*{usi(mirror_sq(kb))}", f"K*{usi(mirror_sq(kw))}"])
    return out


def _pooled(states: dict[str, dict], keys: list[str]) -> tuple[int, int, int]:
    c = [0, 0, 0]
    for k in keys:
        st = states.get(k)
        if st:
            c = [c[0] + st["n"], c[1] + st["sente"], c[2] + st["draw"]]
    return c[0], c[1], c[2]


def check_alerts(states: dict[str, dict], groups: dict[str, list[str]], rule: Rule, fired: list[str]) -> list[dict]:
    """対称な組・群が後手に傾いたときのアラート（同じ対象と段階は 1 回だけ。fired に印を足す）。
    早期: 途中の局数で 先手の得点 + 3.0·se < 0.5（組は min_games 局、群は 500 局から）。
    確定: 組は eps で止まった時点の 95% 区間の上限が 0.5 未満、群は全部の組が止まった後の合算の上限が 0.5 未満。"""
    out: list[dict] = []
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    def emit(tag: str, level: str, target: str, label: str, n: int, m: float, se: float, z: float) -> None:
        if tag in fired:
            return
        fired.append(tag)
        lo, hi = m - Z * se, m + Z * se
        out.append({"time": stamp, "level": level, "target": target, "games": n, "winrate": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                    "z": z, "message": f"{level}: {label} で後手に傾いている（{n:,} 局、先手の得点 {m:.4f}、95% 区間 {lo:.4f}〜{hi:.4f}、"
                                       f"判定は 先手の得点 + {z}·se < 0.5）"})

    sym = sorted({k for ks in groups.values() for k in ks if k in states})
    for k in sym:
        st = states[k]
        label = "▲{} △{}".format(*k.split())
        if st["stop"] in ("eps", "cap") and st["stop_w"] + Z * st["stop_se"] < 0.5:
            emit(f"{k}:final", "確定", k, label, st["stop_n"], st["stop_w"], st["stop_se"], Z)
        m, se = stats(st)
        # 止まった組は確定の判定だけにする（止まった後に届いた局で早期だけが出ると、確定と食い違って見える）
        if st["stop"] is None and st["n"] >= rule.min_games and m + Z_EARLY * se < 0.5:
            emit(f"{k}:early", "早期", k, label, st["n"], m, se, Z_EARLY)
    for g, ks in {**groups, "all": sym}.items():
        ks = [k for k in ks if k in states]
        if not ks:
            continue
        n, s, d = _pooled(states, ks)
        if n == 0:
            continue
        m, se = stats_counts(n, s, d)
        if n >= GROUP_MIN_GAMES and m + Z_EARLY * se < 0.5:
            emit(f"group:{g}:early", "早期", f"group:{g}", GROUP_LABELS[g], n, m, se, Z_EARLY)
        if all(states[k]["stop"] for k in ks) and m + Z * se < 0.5:
            emit(f"group:{g}:final", "確定", f"group:{g}", GROUP_LABELS[g], n, m, se, Z)
    return out


def symmetric_summary(states: dict[str, dict], groups: dict[str, list[str]]) -> dict:
    out = {}
    for g, ks in groups.items():
        ks = [k for k in ks if k in states]
        rows = []
        for k in ks:
            st = states[k]
            m, se = stats(st)
            rows.append({"pair": k, "games": st["n"], "winrate": round(m, 4), "ci95": ci95(m, se), "stop": st["stop"]})
        n, s, d = _pooled(states, ks)
        m, se = stats_counts(n, s, d)
        out[g] = {"label": GROUP_LABELS[g], "pairs": rows, "pooled": {"games": n, "winrate": round(m, 4), "ci95": ci95(m, se)}}
    return out


def balanced_keys(states: dict[str, dict], rule: Rule) -> tuple[list[str], str]:
    """置く側の集合（decisions.md 2026-09-15。設計 §5 の「区間が最小値と重なる組」は、逐次に打ち切ったデータでは
    有意の境目で止まった組まで拾うので、最小値を 0 と置いた「区間が 0.5 を含む組」にした）。"""
    done = {k: st for k, st in states.items() if st["stop"] is not None}
    if rule.eps_place > 0:
        keys = [k for k, st in done.items() if st["place"] in ("done", "cap") and abs(stats(st)[0] - 0.5) <= rule.eps_place]
        desc = (f"置く側: 区間が 0.5 を含んだまま半幅 {rule.eps} 以下で止まった組を半幅 {rule.eps_place} まで打ち足し、"
                f"|w − 0.5| ≤ {rule.eps_place} の組")
    else:
        keys = [k for k, st in done.items() if is_candidate(st)]
        desc = f"置く側: 95% 区間が 0.5 を含んだまま半幅 {rule.eps} 以下で止まった組"
    if not keys and done:
        keys = [min(done, key=lambda k: abs(stats(done[k])[0] - 0.5))]
        desc += "（該当なしのため |w − 0.5| が最小の組）"
    return sorted(keys), desc


def build_table(base: dict, config: dict, state: dict) -> dict:
    """build の表（V̂・自己対局の集計）に逐次の検証の結果を入れた scale.json（version 1）。エンジンは今は balanced だけを読む。"""
    rule = Rule.from_dict(config["rule"])
    states = state["pairs"]
    pairs_out, diffs, stops = [], [], {}
    for e0 in base["pairs"]:
        e = {k: v for k, v in e0.items() if k not in ("verify", "choose")}
        st = states.get(key_of(from_usi(e["kb"]), from_usi(e["kw"])))
        if st and st["n"] > 0:
            m, se = stats(st)
            e["verify"] = {"games": st["n"], "sente": st["sente"], "draw": st["draw"], "gote": st["gote"], "winrate": round(m, 4),
                           "ci95": ci95(m, se), "sims": config["sims"], "stop": st["stop"], "stop_games": st["stop_n"]}
            if st.get("place"):
                e["verify"]["place"] = st["place"]
            e["choose"] = "sente" if m >= 0.5 else "gote"
            stops[st["stop"] or "running"] = stops.get(st["stop"] or "running", 0) + 1
            if "v_hat" in e:
                diffs.append(float(e["v_hat"]) - m)
        pairs_out.append(e)
    forced, seen = [], set()
    for kb in SENTE_SQUARES:
        for kw in GOTE_SQUARES_ALL:
            c = canonical(kb, kw)
            if is_immediate_loss(kb, kw) and c not in seen:
                seen.add(c)
                forced.append({"kb": usi(c[0]), "kw": usi(c[1]), "mirror": [usi(mirror_sq(c[0])), usi(mirror_sq(c[1]))], "choose": "sente"})
    bal, desc = balanced_keys(states, rule)
    verify = {"complete": all(not is_active(st, rule) for st in states.values()), "pairs": sum(1 for st in states.values() if st["n"]),
              "games": sum(st["n"] for st in states.values()), "sims": config["sims"], "model_step": config.get("model_step"),
              "rule": asdict(rule), "search": config["search"], "stops": stops}
    if len(diffs) > 1:
        mu = sum(diffs) / len(diffs)
        var = sum((x - mu) ** 2 for x in diffs) / (len(diffs) - 1)
        verify.update(v_hat_minus_w=round(mu, 4), v_hat_minus_w_se=round((var / len(diffs)) ** 0.5, 4))
    return {
        "version": 1,
        "license": "CC0-1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": base.get("model"),
        "sims": base.get("sims"),
        "n_pairs_pruned": 972,
        "n_pairs_unique": len(pairs_out),
        "pruned": base.get("pruned"),
        "choose_rule": "選ぶ側: 検証対局の先手の得点 w（引き分け 0.5）が 0.5 以上なら先手。後手玉が四段目の組（forced）は先手",
        "balance_rule": desc,
        "pairs": pairs_out,
        "forced": forced,
        "balanced": expand_mirrors([parse_key(k) for k in bal]),
        "symmetric": symmetric_summary(states, symmetric_groups()),
        "verify": verify,
    }
