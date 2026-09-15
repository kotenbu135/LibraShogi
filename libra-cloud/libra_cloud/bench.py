# SPDX-License-Identifier: Apache-2.0
"""vast.ai の実測ベンチの純粋な部分: オファーの選別、ベンチ用の設定、inbox の対局ファイルからの局/日。

ホストでは `python -m libra_cloud.bench report --inbox DIR --start EPOCH --warmup S --out report.json` で結果を書く。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

# 転送料（$/GB）の上限。対局の書き出しと重みの取得で月に数十〜数百 GB になるので高いホストは避ける
MAX_INET_COST = 0.02


def pick_offers(offers: list[dict], *, max_dph: float, price_key: str = "dph_total", min_cores: int = 8, min_down: float = 200.0,
                min_rel: float = 0.98, min_cuda: float = 12.8, max_inet_cost: float = MAX_INET_COST,
                min_cpu_ghz: float = 0.0) -> list[dict]:
    """条件を満たす 1 GPU のオファーを安い順（同じ値段なら CPU の多い順）に返す。

    min_cpu_ghz: CPU の最大周波数（vast.ai の cpu_ghz）の下限。自己対局の探索の反映（apply）は 1 スレッドの速さで決まり、
    2016 年ごろのサーバー CPU（2.4 GHz 前後）では GPU が半分遊んだ（measurements.md 2026-09-14）。"""
    ok = [o for o in offers if not offer_rejects(o, max_dph=max_dph, price_key=price_key, min_cores=min_cores, min_down=min_down, min_rel=min_rel,
                                                 min_cuda=min_cuda, max_inet_cost=max_inet_cost, min_cpu_ghz=min_cpu_ghz)]
    return sorted(ok, key=lambda o: (o[price_key], -(o.get("cpu_cores_effective") or 0)))


def annotate_price(offers: list[dict], rent: str, bid_margin: float) -> list[dict]:
    """オファーの写しに入札額（bid）と実効単価（dph_eff）を付ける。rent が "bid" なら最低入札（min_bid）の bid_margin 増しで入札し、
    実効単価は dph_total − min_bid + bid（dph_total は最低入札にストレージ代などを足した値）。on-demand と min_bid の無いオファーは dph_total。
    選別（pick_offers）と表示は price_key="dph_eff" で行う。"""
    out = []
    for o in offers:
        mb = o.get("min_bid")
        if rent == "bid" and mb is not None:
            bid = round(float(mb) * (1 + bid_margin), 4)
            out.append({**o, "bid": bid, "dph_eff": round(float(o["dph_total"]) - float(mb) + bid, 4)})
        else:
            out.append({**o, "bid": None, "dph_eff": float(o["dph_total"])})
    return out


def worker_threads(offer: dict, cap: int = 12) -> int:
    """ワーカーの自己対局のスレッド数。ホストの nproc は割り当てより多く見える（16 コア割り当てで 64）ので、オファーの実効コア数を使い、
    12 で頭打ちにする（12 を超えても apply は速くならない、measurements.md 2026-09-14 05:47）。コア数が分からなければ 12。"""
    cores = offer.get("cpu_cores_effective")
    return cap if not cores else max(1, min(cap, int(cores)))


# 管理コンソールの入力欄で緩められる条件（offer_rejects の key）
ADJUSTABLE = ("max_dph", "min_cores", "min_cpu_ghz", "min_rel", "max_inet_cost")


def _ceil(x: float, digits: int) -> float:
    return math.ceil(round(x * 10 ** digits, 6)) / 10 ** digits


def _floor(x: float, digits: int) -> float:
    return math.floor(round(x * 10 ** digits, 6)) / 10 ** digits


def offer_rejects(o: dict, *, max_dph: float, price_key: str = "dph_total", min_cores: int = 8, min_down: float = 200.0,
                  min_rel: float = 0.98, min_cuda: float = 12.8, max_inet_cost: float = MAX_INET_COST, min_cpu_ghz: float = 0.0) -> list[dict]:
    """pick_offers の条件のうちオファーが満たさないものを返す（満たせば空）。各要素は {key, text, need}。
    need はその条件の値をいくつにすれば通るか（ADJUSTABLE の条件だけ。表示の桁に丸めて、丸めても通る側に寄せる）。"""
    out: list[dict] = []

    def add(key: str, text: str, need=None) -> None:
        out.append({"key": key, "text": text, "need": need})

    price = o.get(price_key)
    if price is None:
        add("price", f"{price_key} の欄なし")
    elif price > max_dph:
        add("max_dph", f"${price:.3f}/h > 上限 ${max_dph:.2f}", _ceil(price, 2))
    if o.get("num_gpus") != 1:
        add("num_gpus", f"GPU {o.get('num_gpus')} 枚")
    cores = o.get("cpu_cores_effective") or 0
    if cores < min_cores:
        add("min_cores", f"コア {cores:g} < {min_cores}", int(cores))
    ghz = float(o.get("cpu_ghz") or 0)
    if ghz < min_cpu_ghz:
        add("min_cpu_ghz", f"CPU {ghz:.2f} GHz < {min_cpu_ghz:.1f}", _floor(ghz, 1))
    down = o.get("inet_down") or 0
    if down < min_down:
        add("min_down", f"下り {down:.0f} Mbps < {min_down:.0f}")
    rel = o.get("reliability2") or o.get("reliability") or 0
    if rel < min_rel:
        add("min_rel", f"信頼度 {rel:.3f} < {min_rel:.2f}", _floor(rel, 2))
    cuda = float(o.get("cuda_max_good") or 0)
    if cuda < min_cuda:
        add("min_cuda", f"CUDA {cuda:g} < {min_cuda:g}")
    inet = max(o.get("inet_up_cost") or 0, o.get("inet_down_cost") or 0)
    if inet > max_inet_cost:
        add("max_inet_cost", f"転送料 ${inet:.3f}/GB > 上限 ${max_inet_cost:.3f}", _ceil(inet, 3))
    return out


def near_misses(offers: list[dict], **cond) -> list[dict]:
    """条件を 1 つだけ緩めれば通るオファーを、緩める条件ごとに借りる順で先頭のもの 1 件ずつ返す（そのオファーの値段の順）。
    各要素は {key, need, text, offer}。"""
    best: dict[str, tuple] = {}
    price_key = cond.get("price_key", "dph_total")
    for o in offers:
        r = offer_rejects(o, **cond)
        if len(r) != 1 or r[0]["key"] not in ADJUSTABLE:
            continue
        rank = (o[price_key], -(o.get("cpu_cores_effective") or 0))
        if r[0]["key"] not in best or rank < best[r[0]["key"]][0]:
            best[r[0]["key"]] = (rank, {**r[0], "offer": o})
    return [h for _, h in sorted(best.values(), key=lambda x: x[0])]


def offer_query(gpu: str, min_rel: float, min_cuda: float, disk: float) -> str:
    """vast.ai のオファー検索の条件（1 GPU、verified、下り 200 Mbps 以上、イメージの CUDA 以上のドライバー、ディスク）。"""
    return (f"gpu_name={gpu.replace(' ', '_')} num_gpus=1 rentable=true verified=true reliability>{min_rel} inet_down>=200 "
            f"cuda_max_good>={min_cuda} disk_space>={disk}")


def bench_config(cfg: dict, n_games: int, threads: int) -> dict:
    """本番の設定からベンチ用を作る: 探索とネットはそのまま、手元のファイルを参照する項目と学習側の機能は外す。"""
    b = copy.deepcopy(cfg)
    b["selfplay"]["openings"] = ""
    b["selfplay"]["n_games"] = int(n_games)
    b["selfplay"]["threads"] = int(threads)
    b["exploiter"]["main_ckpt"] = ""
    b["exploiter"]["main_source"] = ""
    b["exploiter"]["openings_out"] = ""
    b["auto"]["enabled"] = False
    b["workers"]["enabled"] = False
    return b


def games_per_day(files: list[tuple[float, int, int]], start: float, warmup_s: float) -> dict:
    """files: (書いた時刻, 局数, 手数) の列。start + warmup_s 以降に書いたファイルのうち、最初のファイルの時刻から
    最後のファイルの時刻までに書かれた局数で局/日を出す（最初のファイルの局はそれ以前に打った分なので数えない）。"""
    win = sorted(f for f in files if f[0] >= start + warmup_s)
    games = sum(n for _, n, _ in win)
    moves = sum(m for _, _, m in win)
    out = {"files": len(win), "games": games, "avg_moves": moves / games if games else None, "window_s": None, "games_per_day": None}
    if len(win) >= 2:
        span = win[-1][0] - win[0][0]
        out["window_s"] = span
        if span > 0:
            out["games_per_day"] = sum(n for _, n, _ in win[1:]) / span * 86400
    return out


def scan_inbox(inbox: Path) -> list[tuple[float, int, int]]:
    """対局ファイルを検査して読み、(書いた時刻, 局数, 手数) を時刻順に返す。読めないファイルは飛ばす。"""
    from libra_league.workers import GamesFileError, read_games_file

    out = []
    for p in inbox.glob("*.npz"):
        try:
            meta, games = read_games_file(p)
        except GamesFileError:
            continue
        out.append((float(meta["created"]), len(games), sum(len(g["moves"]) for g in games)))
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="libra_cloud.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="inbox の対局ファイルから局/日を出す")
    r.add_argument("--inbox", required=True)
    r.add_argument("--start", type=float, required=True, help="ワーカーを起動した時刻（epoch 秒）")
    r.add_argument("--warmup", type=float, default=300.0)
    r.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    files = scan_inbox(Path(a.inbox))
    rep = games_per_day(files, a.start, a.warmup)
    rep["all_files"] = len(files)
    rep["all_games"] = sum(n for _, n, _ in files)
    if files:
        rep["first_file_after_s"] = files[0][0] - a.start
        rep["last_file_after_s"] = files[-1][0] - a.start
    text = json.dumps(rep, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
