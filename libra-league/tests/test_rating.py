# SPDX-License-Identifier: Apache-2.0
"""`libra rating`（記録した対局を全部まとめて 1 つの Elo の目盛りにする。Bradley-Terry）。"""
import json
import math

from libra_league.rating import curve, fit, node_step, pairs_of, rating, step_node
from libra_league.state import StateDir


def _w(sd, name, rows):
    (sd.root / "eval").mkdir(exist_ok=True)
    with open(sd.root / "eval" / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _pair(a, b, n, s, t=0.0):
    return {"a": a, "b": b, "n": n, "score_a": s, "t": t, "src": "x"}


def test_two_nodes_match_the_plain_elo_formula():
    # 2 点だけなら Bradley-Terry の最尤推定は Elo の式そのもの
    r = fit([_pair("A", "B", 1000, 0.75)], anchor="B")
    assert r["converged"] and r["anchor"] == "B"
    assert abs(r["nodes"]["A"]["elo"] - 400 * math.log10(0.75 / 0.25)) < 0.05
    lo, hi = r["nodes"]["A"]["ci95"]
    assert lo < r["nodes"]["A"]["elo"] < hi and 40 < hi - lo < 60   # 1,000 局で ±25 Elo ほど
    assert r["nodes"]["B"]["elo"] == 0.0


def test_perfectly_consistent_results_fit_exactly():
    # 強さが 1 : 2 : 4（＝ 0・+120.4・+240.8 Elo）の 3 点を、ちょうど期待どおりの得点で打たせる
    lam = {"A": 1.0, "B": 2.0, "C": 4.0}
    ps = [_pair(x, y, 900, lam[x] / (lam[x] + lam[y])) for x, y in (("A", "B"), ("A", "C"), ("B", "C"))]
    r = fit(ps, anchor="A")
    assert abs(r["nodes"]["B"]["elo"] - 400 * math.log10(2)) < 0.05
    assert abs(r["nodes"]["C"]["elo"] - 400 * math.log10(4)) < 0.05
    assert r["fit"]["chi2"] < 1e-6 and r["fit"]["rms_z"] == 0.0   # ずれ無し
    assert r["fit"]["n_pairs"] == 3 and r["fit"]["n_nodes"] == 3


def test_janken_shows_up_as_misfit():
    """じゃんけん（A が B に、B が C に、C が A に勝つ）は 1 本の Elo では説明できない。

    これが 2026-09-18 の 120 万局で起きたこと（直前の自分には +167 なのに古い相手には伸びない）の形。"""
    ps = [_pair("A", "B", 1000, 0.75), _pair("B", "C", 1000, 0.75), _pair("C", "A", 1000, 0.75)]
    r = fit(ps, anchor="A")
    # 対称なので 3 点とも同じ Elo になり、どの組も「五分のはず」と当てはめられる
    assert all(abs(v["elo"]) < 1.0 for v in r["nodes"].values())
    assert r["fit"]["chi2_per_df"] > 100          # 当てはまらない＝じゃんけん
    worst = r["pairs"][0]
    assert abs(worst["z"]) > 10 and worst["score_obs"] in (0.75, 0.25)


def test_all_losses_node_is_dropped():
    # 1 局も勝っていない点は Elo が −∞ に発散する（Hunter 2004 の存在条件）ので外す
    ps = [_pair("A", "B", 100, 0.6), _pair("A", "Z", 100, 1.0), _pair("B", "Z", 100, 1.0)]
    r = fit(ps, anchor="A")
    assert r["dropped"] == ["Z"] and set(r["nodes"]) == {"A", "B"}


def test_pairs_are_read_from_every_record_and_deduplicated(tmp_path):
    sd = StateDir(tmp_path / "ls")
    sd.create()
    (sd.root / "checkpoints" / "archive").mkdir(parents=True, exist_ok=True)
    (sd.root / "checkpoints" / "archive" / "ckpt_000000800.pt").write_bytes(b"")
    _w(sd, "best.jsonl", [{"t": 100.0, "step": 1600, "best_step": 800, "n": 1000, "score_new": 0.7}])
    # 基準比が最強比の結果を写した行（同じ組・同じ局数・同じ得点）は 1 つだけ数える
    _w(sd, "anchor.jsonl", [{"t": 130.0, "step": 1600, "anchor_step": 800, "n": 1000, "score_new": 0.7},
                            {"t": 140.0, "step": 1600, "anchor_step": 400, "n": 1000, "score_new": 0.9}])
    _w(sd, "reference.jsonl", [
        {"t": 150.0, "step": 1600, "ref": "win1m.pt", "ref_step": None, "n": 200, "score_new": 0.6},
        # この run 自身の archive を参照にした行は、step 800 の点と同じ点にまとめる
        {"t": 160.0, "step": 1600, "ref": "ckpt_000000800.pt", "ref_step": 800, "n": 200, "score_new": 0.72}])
    (sd.root / "matches").mkdir(exist_ok=True)
    (sd.root / "matches" / "a.summary.json").write_text(json.dumps(
        {"n": 40, "a_points": 28.0, "b": "外部エンジン",
         "libra_options": {"DNN_Model": "/x/ckpt_000001600.onnx"}}), encoding="utf-8")

    ps = pairs_of(sd)
    got = sorted((p["a"], p["b"], p["n"], p["score_a"], p["src"]) for p in ps)
    assert got == [
        ("step 1,600", "step 400", 1000, 0.9, "anchor"),
        ("step 1,600", "step 800", 200, 0.72, "reference"),     # 参照も step 800 の点にまとまる
        ("step 1,600", "step 800", 1000, 0.7, "best"),          # anchor の写しは落ちて 1 つ
        ("step 1,600", "win1m.pt", 200, 0.6, "reference"),
        ("step 1,600", "外部エンジン", 40, 0.7, "match"),
    ]


def test_rating_and_curve_over_a_run(tmp_path):
    sd = StateDir(tmp_path / "ls")
    sd.create()
    with open(sd.root / "metrics.jsonl", "w", encoding="utf-8") as f:
        for step, games in [(400, 100000), (800, 200000), (1600, 400000)]:
            f.write(json.dumps({"t": float(step), "step": step, "games_total": games}) + "\n")
    _w(sd, "best.jsonl", [{"t": 1.0, "step": 800, "best_step": 400, "n": 1000, "score_new": 0.75},
                          {"t": 2.0, "step": 1600, "best_step": 800, "n": 1000, "score_new": 0.75}])
    r = rating(sd)
    assert r["anchor"] == step_node(400) and r["n_evals"] == 2
    assert abs(r["nodes"]["step 1,600"]["elo"] - 2 * 400 * math.log10(3)) < 1.0

    c = curve(sd, r)
    assert [(p["games"], p["step"]) for p in c["points"]] == [(100000, 400), (200000, 800), (400000, 1600)]
    # 局数が 2 倍になるたびに +190.8 Elo
    assert all(abs(iv["elo_per_doubling"] - 400 * math.log10(3)) < 1.0 for iv in c["intervals"])
    assert abs(c["fit"]["elo_per_doubling"] - 400 * math.log10(3)) < 1.0
    assert node_step(c["points"][0]["node"]) == 400
    assert c["thin"] == [] and c["min_games"] == 200
    # 点には総局数と時刻の両方を付ける（管理コンソールの Elo のグラフは横軸を切り替えられる）
    assert [p["t"] for p in c["points"]] == [400.0, 800.0, 1600.0]

    # 呼ぶ側が既に metrics を読んでいれば渡せる（status --history が二度読みしないため）
    c2 = curve(sd, r, g_of=lambda st: 7, t_of=lambda st: 9.0)
    assert [(p["games"], p["t"]) for p in c2["points"]] == [(7, 9.0)] * 3


def test_curve_fits_the_recent_stretch_separately(tmp_path):
    """「最近の伸び」は最後の点から 4 回の倍化ぶんだけに当てはめる。

    学習の初めは基本を覚えるぶん伸び方が違うので、全部の点に 1 本の直線を当てはめると形が合わない。
    コンソールの Elo のグラフの「目安の線」はこちらを使う（2026-09-19 のユーザーの「初期に比べると
    伸びが緩やかなので頭打ちなのかグラフからわかりにくい」）。"""
    sd = StateDir(tmp_path / "ls")
    sd.create()
    # 2026-09-19 の ls の実データの形: 初めの 2 点だけ傾きが違い、120 万局以降は log2 に対して直線
    real = [(402, 4738, 0.0), (1416, 18720, 163.5), (8144, 120171, 375.5), (13807, 200361, 470.7),
            (21270, 300500, 624.5), (28908, 400021, 744.3), (62426, 800281, 916.2),
            (98604, 1200126, 1029.8), (117436, 1402864, 1091.9), (135740, 1600085, 1106.0),
            (172923, 2000241, 1151.8)]
    with open(sd.root / "metrics.jsonl", "w", encoding="utf-8") as f:
        for step, games, _ in real:
            f.write(json.dumps({"t": float(step), "step": step, "games_total": games}) + "\n")
    # 目盛りそのものは fit が決めるので、ここは curve の当てはめだけを見る
    nodes = {step_node(st): {"elo": elo, "ci95": [elo - 30, elo + 30], "games": 1000,
                             "opponents": 3, "step": st} for st, _, elo in real}
    c = curve(sd, {"nodes": nodes, "anchor": step_node(402)})

    assert [p["games"] for p in c["points"]] == [g for _, g, _ in real]
    assert c["recent_doublings"] == 4
    # 全部の点: 初めの 2 点に引っ張られて傾きが浅く、残差が大きい
    assert abs(c["fit"]["elo_per_doubling"] - 140.1) < 1.0
    assert c["fit"]["rms_resid"] > 80
    assert c["fit"]["n"] == 11
    # 最近の伸び: 200 万局 / 2^4 = 12.5 万局以降の 8 点。傾きは倍、残差は 4 分の 1 以下
    assert c["fit_recent"]["n"] == 8
    assert c["fit_recent"]["games_from"] == 200361
    assert abs(c["fit_recent"]["elo_per_doubling"] - 203.1) < 1.0
    assert c["fit_recent"]["rms_resid"] < 25
    # **最新の点は目安の線の下**（全部の点だと +62 上に出て「加速している」と誤読させる）
    import math
    pred = c["fit_recent"]["intercept"] + c["fit_recent"]["elo_per_doubling"] * math.log2(2000241)
    assert -40 < 1151.8 - pred < 0


def test_curve_fit_recent_falls_back_while_points_are_few(tmp_path):
    """点が 4 つに満たないうちは全部の点に当てはめる（学習の初め）。"""
    sd = StateDir(tmp_path / "ls")
    sd.create()
    with open(sd.root / "metrics.jsonl", "w", encoding="utf-8") as f:
        for step, games in [(400, 100000), (800, 200000), (1600, 400000)]:
            f.write(json.dumps({"t": float(step), "step": step, "games_total": games}) + "\n")
    _w(sd, "best.jsonl", [{"t": 1.0, "step": 800, "best_step": 400, "n": 1000, "score_new": 0.75},
                          {"t": 2.0, "step": 1600, "best_step": 800, "n": 1000, "score_new": 0.75}])
    c = curve(sd)
    assert c["fit_recent"] == c["fit"] and c["fit"]["n"] == 3


def test_curve_drops_points_with_too_few_games(tmp_path):
    """対局が少なすぎる点は曲線から外す。

    2026-09-19 の step 74,627 は外部計測の 10 局しか無く、区間 ±287 Elo だったのに曲線に載り、
    「2 倍あたり −190 Elo」という区間を作っていた。"""
    sd = StateDir(tmp_path / "ls")
    sd.create()
    with open(sd.root / "metrics.jsonl", "w", encoding="utf-8") as f:
        for step, games in [(400, 100000), (600, 150000), (800, 200000)]:
            f.write(json.dumps({"t": float(step), "step": step, "games_total": games}) + "\n")
    _w(sd, "best.jsonl", [{"t": 1.0, "step": 800, "best_step": 400, "n": 1000, "score_new": 0.75}])
    (sd.root / "matches").mkdir(exist_ok=True)
    (sd.root / "matches" / "a.summary.json").write_text(json.dumps(
        {"n": 10, "a_points": 3.0, "b": "外部エンジン", "libra_options": {"DNN_Model": "/x/ckpt_000000600.onnx"}}),
        encoding="utf-8")
    (sd.root / "matches" / "b.summary.json").write_text(json.dumps(
        {"n": 10, "a_points": 8.0, "b": "外部エンジン", "libra_options": {"DNN_Model": "/x/ckpt_000000800.onnx"}}),
        encoding="utf-8")
    c = curve(sd)
    assert c["thin"] == ["step 600"]                      # 10 局しか無いので外す
    assert [p["step"] for p in c["points"]] == [400, 800]
    # 少ない点を入れない指定にすれば戻る
    assert [p["step"] for p in curve(sd, min_games=1)["points"]] == [400, 600, 800]


def _match(sd, name, step, n, pts, *, go="movetime 1000", go_opp=None, opp_opt=None, libra_opt=None):
    (sd.root / "matches").mkdir(exist_ok=True)
    r = {"n": n, "a_points": pts, "b": "外部エンジン", "go": go,
         "libra_options": {"DNN_Model": f"/x/ckpt_{step:09d}.onnx", "Declare_Win": "true", **(libra_opt or {})}}
    if go_opp:
        r["go_opp"] = go_opp
    if opp_opt:
        r["opponent_options"] = opp_opt
    (sd.root / "matches" / name).write_text(json.dumps(r), encoding="utf-8")


def test_external_opponent_is_a_separate_node_per_setting(tmp_path):
    """相手の設定（スレッド数・持ち時間）を変えたら別の点にする。同じ設定どうしは 1 点にまとまる。

    相手を強くすると強さが変わるのに `id name` は同じままなので、1 点にまとめると強弱の違う相手が
    混ざって目盛りが狂う。ルールの版（Fuseki_Rules）は強さではないので名前に入れない。
    """
    sd = StateDir(tmp_path / "ls")
    sd.create()
    _match(sd, "a.summary.json", 800, 40, 36.0, opp_opt={"Threads": "2", "Fuseki_Rules": "2"})
    _match(sd, "b.summary.json", 1600, 40, 36.0, opp_opt={"Threads": "2", "Fuseki_Rules": "2"})
    _match(sd, "c.summary.json", 1600, 40, 20.0, opp_opt={"Threads": "8", "Fuseki_Rules": "2"})
    opps = sorted({p["b"] for p in pairs_of(sd)})
    assert opps == ["外部エンジン [Threads=2, movetime 1000]", "外部エンジン [Threads=8, movetime 1000]"]


def test_handicapped_libra_is_a_separate_node(tmp_path):
    """Libra 側だけ読む量を減らした対局は、全読みの自分と別の点にする。"""
    sd = StateDir(tmp_path / "ls")
    sd.create()
    _match(sd, "a.summary.json", 1600, 40, 36.0)
    _match(sd, "b.summary.json", 1600, 40, 20.0, go="nodes 400", go_opp="movetime 1000")
    got = sorted((p["a"], p["b"]) for p in pairs_of(sd))
    assert got == [("step 1,600", "外部エンジン [movetime 1000]"),
                   ("step 1,600 [nodes 400]", "外部エンジン [movetime 1000]")]


def test_old_match_records_without_conditions_keep_the_plain_name(tmp_path):
    """条件を記録していない古い結果は素の名前のまま（点が分かれて鎖が切れないように）。"""
    sd = StateDir(tmp_path / "ls")
    sd.create()
    (sd.root / "matches").mkdir(exist_ok=True)
    (sd.root / "matches" / "a.summary.json").write_text(json.dumps(
        {"n": 40, "a_points": 28.0, "b": "外部エンジン",
         "libra_options": {"DNN_Model": "/x/ckpt_000001600.onnx"}}), encoding="utf-8")
    assert [(p["a"], p["b"]) for p in pairs_of(sd)] == [("step 1,600", "外部エンジン")]
