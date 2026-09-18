# SPDX-License-Identifier: Apache-2.0
"""借りたホストの実測から、次に借りるオファーの見込み（局/日と 100 万局あたりの費用）を出す。

選別（bench.pick_offers）はこれまで「実効単価の安い順」だったが、同じ GPU でも CPU によって局/日が 1.2 倍違い、
高い GPU ほど 100 万局あたりでは損をすることがある（measurements.md 2026-09-16: 同じ 5070 Ti で 566k / 576k / 675k 局/日、
5080 は局/日 1.3 倍でも $/h が 1.6 倍で 100 万局あたりは負ける）。そこで過去のセッションの実測を引いて
**見込みの 100 万局あたりの費用**で並べる。実測が無いオファーは実測のあるホストの中央値を当てるので、未知のホストは真ん中の扱いになる
（前に出しすぎて外れを引くことも、後ろに回して良いホストを試さないこともない）。

セッションの速さは「定常状態の回収の傾き」で測る（measurements.md 2026-09-16 と同じ測り方）。起動直後は 512 局を
同時に打ち始めるので最初の回収がまとまって遅れる。その分（WARMUP_S）を除いた区間の傾きを使う。
"""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

WARMUP_S = 600.0    # ブリッジの起動からこの秒数は数えない（最初の回収のまとまりを外す）
MIN_SPAN_S = 900.0  # 傾きを取る区間がこれより短いセッションは実測として使わない
MIN_GAMES = 500     # その区間の回収局がこれより少ないセッションも使わない

PLACED = re.compile(r"^(\d{2}):(\d{2}):(\d{2}) bridge: placed \d+ games \(total (\d+)", re.M)


def short_cpu(name) -> str:
    """CPU 名から型番に要らない語を落とす（"AMD EPYC 7B13 64-Core Processor" → "AMD EPYC 7B13"）。同じ CPU をまとめる鍵にも使う。"""
    return re.sub(r"\s+", " ", re.sub(r"®|\(R\)|\(TM\)|\bCPU\b|\d+-Core|\bProcessor\b|with Radeon.*$|@.*$", "", str(name or ""))).strip()


def placed_points(log_text: str, t_bridge: float) -> list[tuple[float, int]]:
    """bridge.log の「placed … (total N」の行を (epoch 秒, 累計局数) にする。
    行の時刻は HH:MM:SS しかないので、ブリッジの起動時刻から進む向きに日付を補う。"""
    out: list[tuple[float, int]] = []
    prev = t_bridge
    for h, m, s, total in PLACED.findall(log_text):
        lt = time.localtime(prev)
        t = prev - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec) + int(h) * 3600 + int(m) * 60 + int(s)
        while t < prev - 1:  # 日付をまたいだ
            t += 86400
        prev = t
        out.append((t, int(total)))
    return out


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def session_speed(d: Path) -> dict | None:
    """セッションのディレクトリから定常状態の局/日と、そのホストの素性を出す。測れなければ None。"""
    inst = read_json(d / "instance.json")
    if not inst or not inst.get("t_bridge"):
        return None
    try:
        text = (d / "bridge" / "bridge.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    pts = placed_points(text, float(inst["t_bridge"]))
    win = [p for p in pts if p[0] >= float(inst["t_bridge"]) + WARMUP_S]
    if len(win) < 2:
        return None
    span = win[-1][0] - win[0][0]
    games = win[-1][1] - win[0][1]  # 最初の点までの局は暖機の分なので数えない
    if span < MIN_SPAN_S or games < MIN_GAMES:
        return None
    offer = inst.get("offer") or {}
    return {"name": d.name, "games_per_day": games / span * 86400, "span_h": span / 3600, "games": games,
            "machine_id": offer.get("machine_id"), "gpu": offer.get("gpu_name"), "cpu": short_cpu(offer.get("cpu_name")),
            "dph": offer.get("dph_eff") or offer.get("dph_total")}


def scan_sessions(roots) -> list[dict]:
    """セッションの置き場所（~/libra-run/cloud など）から、速さを測れたセッションを古い順に返す。"""
    out = []
    for root in [roots] if isinstance(roots, (str, Path)) else roots:
        root = Path(root).expanduser()
        for d in sorted(root.glob("*-*")):
            if d.is_dir() and (s := session_speed(d)) is not None:
                out.append(s)
    return out


def keys_of(offer: dict) -> list[tuple]:
    """そのオファーが当たる鍵を、確かな順（同じ機械 → 同じ GPU と CPU → 同じ GPU）に返す。"""
    gpu, cpu, mid = offer.get("gpu_name"), short_cpu(offer.get("cpu_name")), offer.get("machine_id")
    ks: list[tuple] = []
    if mid is not None:
        ks.append(("machine", mid))
    if gpu and cpu:
        ks.append(("cpu", gpu, cpu))
    if gpu:
        ks.append(("gpu", gpu))
    return ks


def speed_table(sessions: list[dict]) -> dict[tuple, dict]:
    """セッションの実測を鍵ごとにまとめる（局/日は中央値。1 回だけ良かった・悪かったに引きずられないため）。"""
    by: dict[tuple, list[dict]] = {}
    for s in sessions:
        for k in keys_of({"gpu_name": s["gpu"], "cpu_name": s["cpu"], "machine_id": s["machine_id"]}):
            by.setdefault(k, []).append(s)
    return {k: {"games_per_day": statistics.median(x["games_per_day"] for x in v), "sessions": len(v),
                "span_h": round(sum(x["span_h"] for x in v), 2)} for k, v in by.items()}


SOURCE_LABELS = {"machine": "同じ機械", "cpu": "同じ GPU と CPU", "gpu": "同じ GPU"}


def estimate(offer: dict, table: dict[tuple, dict]) -> dict | None:
    """オファーの見込みの局/日。実測が無ければ None。"""
    for k in keys_of(offer):
        if k in table:
            e = table[k]
            return {"games_per_day": e["games_per_day"], "from": k[0], "sessions": e["sessions"]}
    return None


def usd_per_1m(dph, games_per_day) -> float | None:
    """1 時間あたり $dph のホストで 100 万局を打つ費用。"""
    if not dph or not games_per_day or games_per_day <= 0:
        return None
    return float(dph) * 24 / float(games_per_day) * 1e6


def annotate(offers: list[dict], table: dict[tuple, dict], price_key: str = "dph_total") -> list[dict]:
    """オファーの写しに見込み（est_games_per_day・est_usd_per_1m・est_from）を付ける。
    実測が無いオファーには、渡した中で実測のあるものの中央値を当てる（est_from は None のまま）。"""
    out = []
    for o in offers:
        e = estimate(o, table)
        out.append({**o, "est_games_per_day": e["games_per_day"] if e else None, "est_from": e["from"] if e else None,
                    "est_usd_per_1m": usd_per_1m(o.get(price_key), e["games_per_day"]) if e else None})
    known = [o["est_games_per_day"] for o in out if o["est_games_per_day"] is not None]
    if known:
        mid = statistics.median(known)
        for o in out:
            if o["est_games_per_day"] is None:
                o["est_usd_per_1m"] = usd_per_1m(o.get(price_key), mid)
    return out


def rank(offers: list[dict], price_key: str = "dph_total") -> list[dict]:
    """annotate を済ませたオファーを、見込みの 100 万局あたりの費用の安い順に並べる（見込みが無ければ値段の安い順）。"""
    return sorted(offers, key=lambda o: (o.get("est_usd_per_1m") is None, o.get("est_usd_per_1m") or 0.0,
                                         o[price_key], -(o.get("cpu_cores_effective") or 0)))
