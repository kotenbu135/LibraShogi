# SPDX-License-Identifier: Apache-2.0
"""`libra scaling`（局を何倍にすると何 Elo 伸びるか。docs/scaling-2026-09-18.md §4）。"""
import json
import math

from libra_league.cli import main
from libra_league.scaling import fit, games_of_step, intervals, reference_points, scaling, stitch, time_of_step
from libra_league.state import StateDir


def _metrics(sd, rows):
    with open(sd.root / "metrics.jsonl", "w", encoding="utf-8") as f:
        for step, games in rows:
            f.write(json.dumps({"t": float(step), "step": step, "games_total": games}) + "\n")


def _reference(sd, rows):
    (sd.root / "eval").mkdir(exist_ok=True)
    with open(sd.root / "eval" / "reference.jsonl", "w", encoding="utf-8") as f:
        for t, step, ref, elo, score in rows:
            f.write(json.dumps({"t": t, "step": step, "ref": ref, "n": 200, "score_new": score,
                                "elo": elo, "ci95": [elo - 50, elo + 50]}) + "\n")


def test_games_of_step_interpolates():
    g = games_of_step([{"step": 100, "games_total": 1000}, {"step": 200, "games_total": 3000}])
    assert g(100) == 1000 and g(200) == 3000
    assert g(150) == 2000            # 間は直線
    assert g(50) == 1000 and g(999) == 3000  # 外は端の値
    assert g(None) is None
    assert games_of_step([])(100) is None
    # 同じ step が 2 行あっても割り算で落ちない
    assert games_of_step([{"step": 5, "games_total": 10}, {"step": 5, "games_total": 20}])(5) == 10


def test_time_of_step_interpolates():
    """step → 時刻。管理コンソールの Elo のグラフで横軸を「時間」にしたときに使う。"""
    t = time_of_step([{"step": 100, "t": 1000.0}, {"step": 200, "t": 3000.0}])
    assert t(100) == 1000.0 and t(200) == 3000.0
    assert t(150) == 2000.0          # 間は直線
    assert t(50) == 1000.0 and t(999) == 3000.0  # 外は端の値
    assert t(None) is None
    assert time_of_step([])(100) is None
    assert time_of_step([{"step": 5, "t": 1.0}, {"step": 5, "t": 2.0}])(5) == 1.0   # 同じ step が 2 行でも落ちない
    assert time_of_step([{"step": 1, "games_total": 2}])(1) is None                 # t が無い行は使わない


def test_intervals_and_fit():
    # 局数が 2 倍になるたびに +100 Elo ちょうどの点
    pts = [{"games": 100000 * 2 ** i, "elo": 100.0 * i, "in_band": True, "score": 0.5, "ci95": None, "n": 200, "step": i}
           for i in range(4)]
    ivs = intervals(pts)
    assert len(ivs) == 3
    assert all(abs(iv["elo_per_doubling"] - 100.0) < 1e-6 and iv["in_band"] for iv in ivs)
    # 100 万局あたりは区間ごとに下がる（局数は対数で効くので当然）
    assert ivs[0]["elo_per_1m"] > ivs[-1]["elo_per_1m"]
    assert abs(ivs[0]["elo_per_1m"] - 100.0 / 100000 * 1e6) < 1e-6
    f = fit(pts)
    assert abs(f["elo_per_doubling"] - 100.0) < 1e-6 and f["n"] == 4 and f["rms_resid"] == 0.0
    assert fit(pts[:1]) is None
    # 局数が減る・同じ点は区間にしない
    assert intervals([pts[1], pts[0]]) == [] and intervals([pts[0], pts[0]]) == []


def test_stitch_puts_two_references_on_one_scale():
    # 参照 A は弱く（点が天井に寄る）、B は強い。真の差は 300 Elo
    refs = {
        "A.pt": [{"games": 100000, "elo": 0.0, "score": 0.50, "in_band": True, "ci95": None, "n": 200, "step": 1},
                 {"games": 200000, "elo": 150.0, "score": 0.70, "in_band": True, "ci95": None, "n": 200, "step": 2},
                 {"games": 400000, "elo": 260.0, "score": 0.90, "in_band": False, "ci95": None, "n": 200, "step": 3}],
        "B.pt": [{"games": 100000, "elo": -300.0, "score": 0.15, "in_band": True, "ci95": None, "n": 200, "step": 1},
                 {"games": 200000, "elo": -150.0, "score": 0.30, "in_band": True, "ci95": None, "n": 200, "step": 2},
                 {"games": 400000, "elo": 0.0, "score": 0.50, "in_band": True, "ci95": None, "n": 200, "step": 3}],
    }
    st = stitch(refs)
    assert st["base"] == "B.pt" and st["dropped"] == []      # 帯の中の点が多いほうが土台
    assert st["offsets"] == {"B.pt": 0.0, "A.pt": -300.0}    # B の目盛りでは A の値は 300 低い
    # 天井の点（A の 40 万局）は落ち、B の 40 万局が残る。同じ局数では得点が 0.5 に近いほうを採るので、
    # 10 万局は A（得点 0.50、B は 0.15）を B の目盛りに直した値になり、どちらも同じ曲線の上に乗る
    assert [(p["games"], p["elo"], p["ref"]) for p in st["points"]] == [
        (100000, -300.0, "A.pt"), (200000, -150.0, "A.pt"), (400000, 0.0, "B.pt")]
    # 帯の中で重なりが無い参照は目盛りを合わせられないので落とす
    lone = {"A.pt": refs["A.pt"], "C.pt": [{"games": 800000, "elo": 10.0, "score": 0.5, "in_band": True, "ci95": None, "n": 200, "step": 9}]}
    assert stitch(lone)["dropped"] == ["C.pt"]
    assert stitch({}) == {"base": None, "points": [], "offsets": {}, "dropped": [], "via": {}}


def test_stitch_chains_through_a_middle_reference():
    """古い参照 A と新しい参照 C が直に重ならなくても、間の B を経由して 1 本の目盛りに乗る。

    参照は強くなるたびに入れ替えるので、A（天井に着いて外す）と C（新しく足す）が同じ局数で
    両方とも帯の中に入ることは無い。ここで落としてしまうと、参照を入れ替えた時点で曲線が伸びなくなる。"""
    def pt(games, elo, score, step):
        return {"games": games, "elo": elo, "score": score, "in_band": 0.2 <= score <= 0.8,
                "ci95": None, "n": 200, "step": step}
    refs = {
        # A は古く弱い参照。40 万局で天井（0.90）に着く
        "A.pt": [pt(100000, 0.0, 0.50, 1), pt(200000, 150.0, 0.70, 2), pt(400000, 260.0, 0.90, 3)],
        # B は真ん中。A の 20 万・40 万局と帯の中で重なる（A − B = 300）
        "B.pt": [pt(200000, -150.0, 0.30, 2), pt(400000, -40.0, 0.44, 3), pt(800000, 90.0, 0.62, 4)],
        # C は新しく強い参照。A とは重ならず、B の 80 万局とだけ重なる（B − C = 250）
        "C.pt": [pt(800000, -160.0, 0.28, 4), pt(1600000, -20.0, 0.47, 5)],
    }
    st = stitch(refs)
    assert st["dropped"] == []                       # C は B をたどって乗る
    assert st["base"] == "B.pt"                      # 帯の中の点が 3 つでいちばん多い
    assert st["offsets"] == {"B.pt": 0.0, "A.pt": -300.0, "C.pt": 250.0}
    assert st["via"] == {"B.pt": None, "A.pt": "B.pt", "C.pt": "B.pt"}
    # C の 160 万局の点が B の目盛りで +230 として曲線の先に乗る
    assert [(p["games"], p["elo"], p["ref"]) for p in st["points"]] == [
        (100000, -300.0, "A.pt"), (200000, -150.0, "A.pt"), (400000, -40.0, "B.pt"),
        (800000, 90.0, "B.pt"), (1600000, 230.0, "C.pt")]


def test_scaling_reads_the_run_and_warns_on_ceiling(tmp_path):
    sd = StateDir(tmp_path / "ls")
    sd.create()
    _metrics(sd, [(0, 0), (1000, 100000), (2000, 200000), (4000, 400000), (8000, 800000)])
    # 弱い参照 A は 80 万局で得点 0.92（天井）、強い参照 B は 0.75 で帯の中
    _reference(sd, [
        (1.0, 1000, "A.pt", -100.0, 0.36), (1.1, 1000, "B.pt", -400.0, 0.08),
        (2.0, 2000, "A.pt", 0.0, 0.50), (2.1, 2000, "B.pt", -300.0, 0.15),
        (3.0, 4000, "A.pt", 150.0, 0.70), (3.1, 4000, "B.pt", -150.0, 0.30),
        (4.0, 8000, "A.pt", 230.0, 0.92), (4.1, 8000, "B.pt", 0.0, 0.75),
    ])
    pts = reference_points(sd)
    assert [p["games"] for p in pts["A.pt"]] == [100000, 200000, 400000, 800000]
    assert [p["in_band"] for p in pts["A.pt"]] == [True, True, True, False]

    r = scaling(sd, cost_per_1m=8.0)
    c = r["curve"]
    assert c["base"] == "A.pt"                     # 帯の中の点は A が 3、B が 2
    assert c["offsets"]["B.pt"] == 300.0           # 重なる 2 点（20 万・40 万局）の差の平均
    assert [p["games"] for p in c["points"]] == [100000, 200000, 400000, 800000]
    assert [p["ref"] for p in c["points"]] == ["A.pt", "A.pt", "A.pt", "B.pt"]  # 天井の A ではなく B を採る
    assert c["points"][-1]["elo"] == 300.0  # B の 0.0 を A の目盛りに直すと +300
    # 40 万 → 80 万局は +150 Elo ＝ 2 倍あたり +150、100 万局あたり +375
    last = c["intervals"][-1]
    assert last["elo_per_doubling"] == 150.0 and last["elo_per_1m"] == 375.0
    # 見込み: 2 倍（+80 万局）は $6.4、1 Elo あたり $6.4 / 傾き
    o = r["outlook"]
    assert o["games_now"] == 800000 and o["double_cost_usd"] == 6.4
    assert abs(o["usd_per_elo"] - 6.4 / o["elo_per_doubling"]) < 1e-3
    assert abs(o["next_1m_elo"] - o["elo_per_doubling"] * math.log2(1800000 / 800000)) < 0.1
    # 天井に触れた参照は注意に出る
    assert any("A.pt" in n and "天井" in n for n in r["notes"])
    # 最新の点は B の得点 0.75 で 0.25〜0.75 に収まるので、参照を足せという注意はまだ出ない
    assert not any("新しい固定の参照" in n for n in r["notes"])
    # 局を足す効きが落ちていないので、その注意は出ない
    assert not any("6 割" in n for n in r["notes"])


def test_band_can_be_narrowed(tmp_path):
    sd = StateDir(tmp_path / "ls")
    sd.create()
    _metrics(sd, [(0, 0), (1000, 100000), (2000, 200000)])
    _reference(sd, [(1.0, 1000, "A.pt", 0.0, 0.50), (2.0, 2000, "A.pt", 150.0, 0.70)])
    assert [p["in_band"] for p in scaling(sd)["references"]["A.pt"]["points"]] == [True, True]
    assert [p["in_band"] for p in scaling(sd, band=(0.4, 0.6))["references"]["A.pt"]["points"]] == [True, False]


def test_scaling_asks_for_a_new_reference_before_the_ceiling(tmp_path):
    sd = StateDir(tmp_path / "ls")
    sd.create()
    _metrics(sd, [(0, 0), (1000, 100000), (2000, 200000), (4000, 400000)])
    # どの参照も最新の点が 0.25〜0.75 の外（帯の中ではあるが 0.5 から遠い）
    _reference(sd, [(1.0, 1000, "A.pt", -100.0, 0.36), (2.0, 2000, "A.pt", 0.0, 0.50), (3.0, 4000, "A.pt", 190.0, 0.76)])
    assert any("新しい固定の参照" in n for n in scaling(sd)["notes"])


def test_scaling_warns_when_the_slope_falls(tmp_path):
    sd = StateDir(tmp_path / "ls")
    sd.create()
    _metrics(sd, [(0, 0), (1000, 100000), (2000, 200000), (4000, 400000), (8000, 800000)])
    _reference(sd, [(1.0, 1000, "A.pt", -200.0, 0.25), (2.0, 2000, "A.pt", 0.0, 0.50),
                    (3.0, 4000, "A.pt", 80.0, 0.60), (4.0, 8000, "A.pt", 100.0, 0.64)])
    r = scaling(sd)
    assert [iv["elo_per_doubling"] for iv in r["curve"]["intervals"]] == [200.0, 80.0, 20.0]
    assert any("6 割" in n for n in r["notes"])


def test_scaling_is_quiet_before_there_is_anything(tmp_path, capsys):
    sd = StateDir(tmp_path / "ls")
    sd.create()
    r = scaling(sd)
    assert r["references"] == {} and r["curve"]["points"] == [] and r["outlook"] is None
    assert any("3 本以上" in n for n in r["notes"])
    assert main(["--root", str(tmp_path), "--run", "ls", "scaling", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["curve"]["points"] == []


def test_cli_prints_a_table(tmp_path, capsys):
    sd = StateDir(tmp_path / "ls")
    sd.create()
    _metrics(sd, [(0, 0), (1000, 100000), (2000, 200000), (4000, 400000)])
    _reference(sd, [(1.0, 1000, "A.pt", -100.0, 0.36), (2.0, 2000, "A.pt", 0.0, 0.50), (3.0, 4000, "A.pt", 150.0, 0.70)])
    assert main(["--root", str(tmp_path), "--run", "ls", "scaling"]) == 0
    out = capsys.readouterr().out
    assert "参照 A.pt" in out and "2 倍あたり" in out and "100 万局あたり" in out and "1 Elo あたり" in out


def test_progress_snapshot_carries_the_curve(tmp_path):
    from libra_league.progress import format_md, snapshot

    sd = StateDir(tmp_path / "ls")
    sd.create()
    sd.write_state({"step": 8000, "games_total": 800000})
    _metrics(sd, [(0, 0), (1000, 100000), (2000, 200000), (4000, 400000), (8000, 800000)])
    _reference(sd, [(1.0, 1000, "A.pt", -200.0, 0.25), (2.0, 2000, "A.pt", -50.0, 0.43),
                    (3.0, 4000, "A.pt", 100.0, 0.64), (4.0, 8000, "A.pt", 250.0, 0.79)])
    s = snapshot(sd)
    assert s["scaling"]["curve"]["fit"]["elo_per_doubling"] == 150.0
    assert s["scaling"]["outlook"]["games_now"] == 800000
    md = format_md(s)
    assert "局を 2 倍にしたときの伸び" in md and "+150.0 Elo / 2 倍" in md and "100 万局の買い足しの見込み" in md


def test_new_reference_from_own_archive_joins_the_scale(tmp_path):
    """新しく足した参照（この run 自身の archive）は、古い参照が天井に着いていても目盛りに乗る。

    足した時点の重み ＝ その参照なので「自分対自分 ＝ 0 Elo」の点が測らずに分かる。これが無いと、
    古い参照 B（既に得点 0.8 超）と新しい参照 C が帯の中で重ならず、C が曲線から落ちる。"""
    sd = StateDir(tmp_path / "ls")
    sd.create()
    _metrics(sd, [(0, 0), (1000, 100000), (2000, 200000), (4000, 400000), (8000, 800000), (16000, 1600000)])
    (sd.root / "checkpoints" / "archive").mkdir(parents=True, exist_ok=True)
    (sd.root / "checkpoints" / "archive" / "ckpt_000008000.pt").write_bytes(b"")  # 80 万局の archive
    _reference(sd, [
        (1.0, 2000, "B.pt", -150.0, 0.30), (2.0, 4000, "B.pt", 0.0, 0.50),
        (3.0, 8000, "B.pt", 240.0, 0.80),
        # 160 万局で初めて C を測る。このとき B は得点 0.88 で天井に着いている
        (4.0, 16000, "B.pt", 350.0, 0.88), (4.1, 16000, "ckpt_000008000.pt", 90.0, 0.62),
    ])
    pts = reference_points(sd)
    me = pts["ckpt_000008000.pt"]
    assert [(p["games"], p["elo"], p.get("self")) for p in me] == [(800000, 0.0, True), (1600000, 90.0, None)]
    c = scaling(sd, cost_per_1m=8.0)["curve"]
    assert c["base"] == "B.pt" and c["dropped"] == []
    assert c["offsets"]["ckpt_000008000.pt"] == 240.0      # 80 万局で B は +240、C は 0
    assert [(p["games"], p["elo"], p["ref"]) for p in c["points"]] == [
        (200000, -150.0, "B.pt"), (400000, 0.0, "B.pt"),
        (800000, 240.0, "ckpt_000008000.pt"), (1600000, 330.0, "ckpt_000008000.pt")]
