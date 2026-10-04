#!/usr/bin/env python3
"""深読みの合図の下調べ（docs/deep-read-signals-2026-10-04.md）。

自己対局の局面を本番と同じ探索の設定（config/ls.toml の [search]）で 24 回（候補 8）と 96 回（候補 16）読み、
「深く読むと答えが変わる局面」を 24 回までで分かる合図でどれだけ拾えるかを数える。自己対局の仕組みは変えない。

  # 1. 読む（CPU。ONNX Runtime。3,100 局面＋520 局面で約 50 分、4 コア）
  PYTHONPATH=libra-sim/python:libra-search/python .venv/bin/python tools/deep_read_probe.py probe \\
      libra-v0.3.onnx libra-v0.3-selfplay-sample.jsonl.gz out.jsonl --positions 3100 --repeat 520
  # 2. 数える（表を Markdown で標準出力に）
  .venv/bin/python tools/deep_read_probe.py analyze out.jsonl

局面は標本の対局からくじで 1 局 1 局面（玉 2 手の後から終局の手前まで。くじの種は --seed）。
読みの種は局面の番号と読み方（a24・b96・c96）から決める。
読みの途中の 12 回目の状態は SelfPlay.set_snapshot で取る（探索は変えない。libra-search/tests/test_external.py）。
"""
import argparse
import gzip
import hashlib
import json
import math
import random
import sys
import time
import tomllib

import numpy as np

TOP = 0.25  # 深読みする割合（今のくじ引き full_prob と同じ）
EPS = 1e-12


# ---------------------------------------------------------------- 読む

def probe(a):
    import onnxruntime as ort
    import librasearch
    import librashogi as ls

    search = tomllib.load(open(a.config, "rb"))["search"]
    cfg = dict(search, external=True)
    print("search", {k: search[k] for k in ("full_sims", "fast_sims", "full_prob", "gumbel_m_full", "gumbel_m_fast",
                                             "c_visit", "c_scale", "gumbel_rescale", "cpuct", "mate_nodes_root",
                                             "proof_nodes")}, flush=True)
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])

    def net(sq, gl):
        pol, wdl, _ = sess.run(None, {"sq": sq, "glob": gl})
        w = np.exp(wdl - wdl.max(1, keepdims=True))  # wdl はロジット
        return np.ascontiguousarray(pol, np.float32), np.ascontiguousarray(w / w.sum(1, keepdims=True), np.float32)

    games = [json.loads(line) for line in gzip.open(a.sample, "rt")]
    rng = random.Random(a.seed)
    positions = []
    for gi in rng.sample(range(len(games)), a.positions):
        toks = games[gi]["tokens"].split()
        k = rng.randrange(2, len(toks))  # k 手を指した後の局面（手番の側が読む）。k < 40 なら布石
        positions.append({"game": gi, "k": k, "line": "position fuseki moves " + " ".join(toks[:k])})

    def seed_of(i, tag):
        return int.from_bytes(hashlib.blake2b(f"{tag}:{i}".encode(), digest_size=8).digest(), "little")

    def run(jobs, sims, full, snap, tag):
        # 局面ごとに種を決めた 1 枠のエンジンで読み、葉をまとめて評価する（ネットの評価は特徴量だけで決まるので、まとめ方に依らない）
        out, todo, act = {}, list(jobs), []
        sq = np.zeros((a.batch, 81, ls.SQ_FEATS), np.float32)
        gl = np.zeros((a.batch, ls.GLOB_FEATS), np.float32)
        t0 = time.time()
        while todo or act:
            while todo and len(act) < a.batch:
                i = todo.pop(0)
                e = librasearch.SelfPlay(cfg, 1, seed_of(i, tag), 1)
                e.set_snapshot(0, snap)
                assert e.set_position(0, positions[i]["line"], sims, full)
                act.append((i, e))
            live = []
            for i, e in act:
                if e.idle(0):
                    out[i] = e.result(0)
                else:
                    live.append((i, e))
            act = live
            if not act:
                continue
            for r, (_, e) in enumerate(act):
                e.collect(sq[r:r + 1], gl[r:r + 1])
            pol, wdl = net(sq[:len(act)], gl[:len(act)])
            for r, (_, e) in enumerate(act):
                e.apply(pol[r:r + 1], wdl[r:r + 1])
        print(f"{tag}: {len(out)} positions, {time.time() - t0:.0f}s", flush=True)
        return out

    def phase(jobs, sims, full, snap, tag):
        # 読み方ごとに <out>.<tag>.json に残し、やり直したときは読んだ分を使う（長い計算が途中で止まったとき用）
        path = f"{a.out}.{tag}.json"
        try:
            got = {int(k): v for k, v in json.load(open(path)).items()}
            if set(got) == set(jobs):
                print(f"{tag}: reuse {path}", flush=True)
                return got
        except FileNotFoundError:
            pass
        got = run(jobs, sims, full, snap, tag)
        json.dump(got, open(path, "w"))
        return got

    idx = list(range(a.positions))
    ra = phase(idx, search["fast_sims"], False, a.snapshot, "a24")  # 速読み: 候補 gumbel_m_fast
    rb = phase(idx, search["full_sims"], True, a.snapshot, "b96")   # 全読み: 候補 gumbel_m_full
    ok = [i for i in idx if ra[i]["policy"] and rb[i]["policy"]]  # 証明済み（読まずに証明手。policy が空）を除く
    rc = phase(ok[:a.repeat], search["full_sims"], True, 0, "c96")  # 雑音の床: 別の種でもう一度 96 回
    with open(a.out, "w") as f:
        for i in idx:
            f.write(json.dumps(dict(positions[i], a=ra[i], b=rb[i], c=rc.get(i))) + "\n")
    print("done", len(ok), "searched of", a.positions)


# ---------------------------------------------------------------- 数える

def rankdata(x):
    """平均順位（1 始まり、同点は平均）。"""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(score, y):
    """Mann-Whitney（同点は 0.5）。y は 0/1。"""
    y = np.asarray(y, int)
    r = rankdata(score)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kl(p, q):  # KL(p || q)。p, q は手 → 確率
    return sum(pp * math.log((pp + EPS) / (q.get(m, 0.0) + EPS)) for m, pp in p.items() if pp > 0)


def argmax(p):
    return max(p.items(), key=lambda x: x[1])[0]


def features(rec):
    a, b, c = rec["a"], rec["b"], rec["c"]
    pa, pb = dict(map(tuple, a["policy"])), dict(map(tuple, b["policy"]))
    prior = {x["move"]: x["prior"] for x in a["cands"]}
    z = sum(prior.values())
    prior = {m: p / z for m, p in prior.items()}
    qa = {x["move"]: x["q"] for x in a["cands"]}
    top2 = sorted(pa.items(), key=lambda x: -x[1])[:2]
    row = {
        "phase": "fuseki" if rec["k"] < 40 else "normal",
        # 答えの変化（24 回 → 96 回）
        "y1": kl(pb, pa),                          # ① 方策の目標のずれ KL(π96 || π24)
        "y2": int(argmax(pb) != argmax(pa)),       # ② 最善手（方策の目標の最大）が変わった
        "y2play": int(b["best"] != a["best"]),     # （参考）指す手（Gumbel ノイズ込み）が変わった
        "y3": abs(b["root_q"] - a["root_q"]),      # ③ 局面の評価の差
        # 24 回までで分かる合図
        "s1": kl(pa, prior),                       # 驚き: KL(π24 || ネットの方策)
        "s2": int(a["snapshot"]["best"] != a["best"]),  # 入れ替わり: 12 回目と 24 回目で指す手が違う
        "s3": abs(a["net_value"] - a["root_q"]),   # 評価の動き: |ネットの値 − 24 回後の root の値|
        "s4": int(len(top2) == 2 and qa[top2[0][0]] * qa[top2[1][0]] < 0),  # 境目: 第 1・第 2 候補の評価の符号が違う
        "s5": -sum(p * math.log(p + EPS) for p in pa.values()),  # 多様さ: π24 のエントロピー
        "skip": int(len(pa) == 1 or max(prior.values()) > 0.95),  # 深読みしない側
    }
    if c is not None:
        pc = dict(map(tuple, c["policy"]))
        row.update(n1=kl(pc, pb), n2=int(argmax(pc) != argmax(pb)), n2play=int(c["best"] != b["best"]),
                   n3=abs(c["root_q"] - b["root_q"]), y1c=kl(pc, pa), y2c=int(argmax(pc) != argmax(pa)))
    return row


def col(rs, k):
    return np.array([x[k] for x in rs], float)


def label_top(v):
    thr = np.quantile(v, 1 - TOP)
    return (v >= thr).astype(int), thr


def select(score, g):
    """合図の強い順に TOP を選ぶ（同点は乱数で崩す）。"""
    n = len(score)
    order = np.lexsort((g.random(n), -np.asarray(score, float)))
    sel = np.zeros(n, bool)
    sel[order[:int(round(TOP * n))]] = True
    return sel


def with_lottery(score_fn, frac_lot=0.1):
    """深読みの枠の 9 割を合図の順で、1 割をくじで（合図で選ばなかった局面から）。"""
    def f(rs, g):
        n = len(rs)
        k = int(round(TOP * n))
        k_sig = int(round(k * (1 - frac_lot)))
        order = np.lexsort((g.random(n), -score_fn(rs)))
        sel = np.zeros(n, bool)
        sel[order[:k_sig]] = True
        sel[g.choice(np.flatnonzero(~sel), k - k_sig, replace=False)] = True
        return sel
    return f


def skipped(rs, s):
    return np.where(col(rs, "skip") == 1, -1e9, s)


SCORES = {
    "s1": lambda rs: col(rs, "s1"),
    "s2": lambda rs: col(rs, "s2"),
    "s3": lambda rs: col(rs, "s3"),
    "s4": lambda rs: col(rs, "s4"),
    "s5": lambda rs: col(rs, "s5"),
    "s1skip": lambda rs: skipped(rs, col(rs, "s1")),
    "c12": lambda rs: col(rs, "s2") * 1e6 + col(rs, "s1"),
    "c12rank": lambda rs: rankdata(col(rs, "s1")) + rankdata(col(rs, "s2")),
    "c12skip": lambda rs: skipped(rs, col(rs, "s2") * 1e6 + col(rs, "s1")),
}
METHODS = [
    ("くじ引き（今の方式）", lambda rs, g: select(np.zeros(len(rs)), g), None),
    ("合図1 驚き", lambda rs, g: select(SCORES["s1"](rs), g), "s1"),
    ("合図2 入れ替わり", lambda rs, g: select(SCORES["s2"](rs), g), "s2"),
    ("合図3 評価の動き", lambda rs, g: select(SCORES["s3"](rs), g), "s3"),
    ("合図4 境目", lambda rs, g: select(SCORES["s4"](rs), g), "s4"),
    ("合図5 多様さ", lambda rs, g: select(SCORES["s5"](rs), g), "s5"),
    ("合図1（深読みしない側を除く）", lambda rs, g: select(SCORES["s1skip"](rs), g), None),
    ("合図1と2（2 が立った局面を先に、その中は 1 の順）", lambda rs, g: select(SCORES["c12"](rs), g), None),
    ("合図1と2（順位の和）", lambda rs, g: select(SCORES["c12rank"](rs), g), None),
    ("合図1（除く）＋1 割くじ", with_lottery(SCORES["s1skip"]), None),
    ("合図1と2（除く）＋1 割くじ", with_lottery(SCORES["c12skip"]), None),
]


def evaluate(rs, fn, reps=200, boot=1000):
    """拾えた割合: ①の上位 25%・②の変化・③の上位 25% のうち、選んだ 25% に入った割合。①の量の取り分は KL の合計のうち選んだ分。
    乱数（同点の崩し・くじ）は reps 回の平均。区間は局面を引き直すブートストラップの 2.5〜97.5%。"""
    y1v = col(rs, "y1")
    y1, _ = label_top(y1v)
    y2 = col(rs, "y2")
    y3, _ = label_top(col(rs, "y3"))
    p = np.mean([fn(rs, np.random.default_rng(s)) for s in range(reps)], axis=0)  # 局面ごとに選ばれる確率
    res = {"rec1": (p * y1).sum() / y1.sum(), "rec2": (p * y2).sum() / max(1, y2.sum()),
           "rec3": (p * y3).sum() / y3.sum(), "mass1": (p * y1v).sum() / y1v.sum()}
    br = np.random.default_rng(1)
    bs1, bs2 = [], []
    n = len(rs)
    for _ in range(boot):
        ix = br.integers(0, n, n)
        yy, _ = label_top(y1v[ix])
        bs1.append((p[ix] * yy).sum() / yy.sum())
        bs2.append((p[ix] * y2[ix]).sum() / max(1, y2[ix].sum()))
    res["rec1_ci"] = np.quantile(bs1, [0.025, 0.975])
    res["rec2_ci"] = np.quantile(bs2, [0.025, 0.975])
    return res


def table(rs, title):
    y1, thr = label_top(col(rs, "y1"))
    print(f"\n### {title}（{len(rs)} 局面。①の上位 25% の境は KL ≥ {thr:.4f}、②の最善手が変わった局面 {col(rs, 'y2').mean():.1%}）\n")
    print("| 選び方 | ① 拾えた割合 [95%] | ① の量の取り分 | ② 拾えた割合 [95%] | ③ 拾えた割合 | AUC ① | AUC ② | 合図が立った局面 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, fn, key in METHODS:
        e = evaluate(rs, fn)
        if key:
            sc = col(rs, key)
            a1, a2 = f"{auc(sc, y1):.3f}", f"{auc(sc, col(rs, 'y2')):.3f}"
            fire = f"{(sc > 0).mean():.1%}" if key in ("s2", "s4") else "—"
        else:
            a1 = a2 = fire = "—"
        print(f"| {name} | {e['rec1']:.1%} [{e['rec1_ci'][0]:.1%}〜{e['rec1_ci'][1]:.1%}] | {e['mass1']:.1%} | "
              f"{e['rec2']:.1%} [{e['rec2_ci'][0]:.1%}〜{e['rec2_ci'][1]:.1%}] | {e['rec3']:.1%} | {a1} | {a2} | {fire} |")


def analyze(a):
    recs = [json.loads(line) for line in open(a.data)]
    n_all = len(recs)
    recs = [r for r in recs if r["a"]["policy"] and r["b"]["policy"]]
    rows = [features(r) for r in recs]
    print(f"局面 {n_all}、証明済みで除いた {n_all - len(rows)}、残り {len(rows)}"
          f"（布石 {sum(x['phase'] == 'fuseki' for x in rows)}・本将棋 {sum(x['phase'] == 'normal' for x in rows)}）")

    print("\n### 答えの変化の大きさ（24 回 → 96 回）\n")
    print("| 段階 | 局面 | ① KL 平均 | ① KL 中央 | ② 最善手が変わる | （参考）指す手が変わる | ③ 評価の差 平均 | 深読みしない側 |")
    print("|---|---|---|---|---|---|---|---|")
    for ph, sub in phases(rows):
        print(f"| {ph} | {len(sub)} | {col(sub, 'y1').mean():.4f} | {np.median(col(sub, 'y1')):.4f} | {col(sub, 'y2').mean():.1%} | "
              f"{col(sub, 'y2play').mean():.1%} | {col(sub, 'y3').mean():.4f} | {col(sub, 'skip').mean():.1%} |")

    rep = [x for x in rows if "n1" in x]
    print(f"\n### 雑音の床（同じ局面を 96 回で 2 回、別の種。{len(rep)} 局面）\n")
    print("| 段階 | 局面 | ① KL 平均（96↔96 / 24→96） | ① KL 中央 | ② 最善手が変わる | （参考）指す手が変わる | ③ 評価の差 平均 |")
    print("|---|---|---|---|---|---|---|")
    for ph, sub in phases(rep):
        print(f"| {ph} | {len(sub)} | {col(sub, 'n1').mean():.4f} / {col(sub, 'y1').mean():.4f} | "
              f"{np.median(col(sub, 'n1')):.4f} / {np.median(col(sub, 'y1')):.4f} | {col(sub, 'n2').mean():.1%} / {col(sub, 'y2').mean():.1%} | "
              f"{col(sub, 'n2play').mean():.1%} / {col(sub, 'y2play').mean():.1%} | {col(sub, 'n3').mean():.4f} / {col(sub, 'y3').mean():.4f} |")
    y1a, y1b = col(rep, "y1"), col(rep, "y1c")
    la, _ = label_top(y1a)
    lb, _ = label_top(y1b)
    both2 = (col(rep, "y2") * col(rep, "y2c")).sum()
    any2 = ((col(rep, "y2") + col(rep, "y2c")) > 0).sum()
    print(f"\n- ①の再現性: 同じ 24 回の読みとの KL を、別の種の 96 回 2 本で測った順位相関 {spearman(y1a, y1b):.3f}、"
          f"上位 25% の重なり {(la & lb).sum() / la.sum():.1%}（くじなら 25%）")
    print(f"- ②の再現性: 2 本の 96 回のどちらかで最善手が変わった局面のうち、両方で変わった割合 {both2 / max(1, any2):.1%}（{int(both2)}/{int(any2)}）")
    g = np.random.default_rng(0)
    sel = select(y1b, g)
    print(f"- 天井の目安: もう 1 本の 96 回の KL（24 回からは分からない）を合図にしたときの①の拾えた割合 {(sel * la).sum() / la.sum():.1%}、"
          f"AUC {auc(y1b, la):.3f}")

    keys = ["s1", "s2", "s3", "s4", "s5", "skip", "y1", "y2", "y3"]
    print("\n### 合図と答えの変化の順位相関（全局面）\n")
    print("| | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for k1 in keys:
        print(f"| {k1} | " + " | ".join(f"{spearman(col(rows, k1), col(rows, k2)):.2f}" for k2 in keys) + " |")

    for ph, sub in phases(rows):
        table(sub, {"全体": "全局面", "布石": "布石（1〜40 手目）", "本将棋": "本将棋（41 手目から）"}[ph])


def phases(rows):
    return [("全体", rows), ("布石", [x for x in rows if x["phase"] == "fuseki"]),
            ("本将棋", [x for x in rows if x["phase"] == "normal"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="局面を集めて 24 回・96 回で読む")
    p.add_argument("onnx")
    p.add_argument("sample", help="自己対局の標本（games_*.jsonl を 1 つにした jsonl.gz。Release の selfplay-sample）")
    p.add_argument("out")
    p.add_argument("--positions", type=int, default=3100)
    p.add_argument("--repeat", type=int, default=520, help="雑音の床のためにもう一度 96 回で読む局面の数")
    p.add_argument("--snapshot", type=int, default=12, help="途中の状態を取る回数")
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--config", default="config/ls.toml")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    q = sub.add_parser("analyze", help="合図ごとの拾えた割合を数える")
    q.add_argument("data")
    a = ap.parse_args()
    if a.cmd == "probe":
        sys.path[:0] = ["libra-sim/python", "libra-search/python"]
        probe(a)
    else:
        analyze(a)


if __name__ == "__main__":
    main()
