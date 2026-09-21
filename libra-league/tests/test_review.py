# SPDX-License-Identifier: Apache-2.0
"""`libra review`（docs/restart-plan.md §3 M6・§7 P4）。"""
import json

from libra_league.cli import main
from libra_league.review import NA, OK, REVIEW, WARN, review
from libra_league.state import StateDir


def _gen(t, win, held, games=None):
    games = games if games is not None else int(t)
    return {"t": t, "games_total": games, "gen": {"t": t, "games": games, "window": {"normal": {"corr_v": win}, "fuseki": {"corr_v": 0.5}},
                                                  "heldout": {"normal": {"corr_v": held}, "fuseki": {"corr_v": 0.5}}}}


def _write(sd, name, rows):
    (sd.root / "eval").mkdir(exist_ok=True)
    with open(sd.root / "eval" / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_review_verdicts(tmp_path, capsys):
    sd = StateDir(tmp_path / "x")
    sd.create()
    now = 100000.0
    # まだ何も無い
    r = review(sd, now=now)
    assert r["verdict"] == NA and all(i["verdict"] == NA for i in r["items"])
    # M1: held-out が上がり、差が小さい → 続ける
    with open(sd.root / "metrics.jsonl", "w", encoding="utf-8") as f:
        for t, w, h, g in ((now - 3 * 3600, 0.5, 0.40, 100000), (now - 2 * 3600, 0.55, 0.45, 200000), (now - 3600, 0.6, 0.52, 300000)):
            f.write(json.dumps(_gen(t, w, h, g)) + "\n")
            f.write(json.dumps(_gen(t, w, h, g)) + "\n")  # 同じ gen を繰り返す行（5 分ごとの status）は 1 つに数える
    _write(sd, "best.jsonl", [{"t": now - 100, "step": 2000, "games": 300000, "best_step": 1000, "elo_vs_best": 60.0, "ci95": [30.0, 90.0], "improved": True, "stall": 0}])
    _write(sd, "anchor.jsonl", [{"t": now - 100, "step": 2000, "games": 300000, "anchor_step": 1000, "elo_vs_anchor": 60.0, "elo": 60.0, "ci95": [30.0, 90.0]}])
    _write(sd, "reference.jsonl", [{"t": now - 7200, "step": 1000, "games": 100000, "ref": "ckpt_000646699.pt", "elo": -400.0, "ci95": [-450.0, -350.0]},
                                   {"t": now - 100, "step": 2000, "games": 300000, "ref": "ckpt_000646699.pt", "elo": -300.0, "ci95": [-350.0, -250.0]}])
    sd.status_json.write_text(json.dumps({"games_per_day_1h": 400000}))
    r = review(sd, now=now)
    assert r["verdict"] == OK and [i["verdict"] for i in r["items"]] == [OK, OK, OK, OK, OK]
    assert abs(r["items"][0]["values"]["gap"] - 0.08) < 1e-9 and r["items"][0]["values"]["n_rows"] == 3
    # 窓を 15 万局にすると gen の行は最後の 2 つ
    assert review(sd, {"gen_games": 150000}, now=now)["items"][0]["values"]["n_rows"] == 2
    # 参照は最後の 2 回を比べ、実際の間隔を書く
    assert "200,000 局で +100.0" in review(sd, now=now)["items"][3]["why"]
    # 窓の記憶（差 0.3）→ 注意。局/日の下限 → 注意
    with open(sd.root / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(_gen(now - 10, 0.9, 0.6, 320000)) + "\n")
    r = review(sd, {"gpd_min": 500000}, now=now)
    assert r["items"][0]["verdict"] == WARN and "窓の記憶" in r["items"][0]["why"] and r["items"][-1]["verdict"] == WARN and r["verdict"] == WARN
    # 最強の足踏み 3 回 → 見直し（全体も見直し）
    _write(sd, "best.jsonl", [{"t": now - 100, "step": 5000, "best_step": 2000, "elo_vs_best": -10.0, "ci95": [-40.0, 20.0], "improved": False, "stall": 3}])
    r = review(sd, now=now)
    assert r["items"][1]["verdict"] == REVIEW and r["verdict"] == REVIEW
    # CLI
    assert main(["--root", str(tmp_path), "--run", "x", "review", "--set", "best_stall_alert=5"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("review x") and "[注意] M2" in out
    assert main(["--root", str(tmp_path), "--run", "x", "review", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == REVIEW


def test_m1_flat_is_ok_and_fall_warns(tmp_path):
    """M1 は「横ばい」を咎めない（2026-09-19 の切り分け。docs/gen-metric-2026-09-19.md）。
    見るのは窓の中との差（丸暗記）と、下がっていないかだけ。"""
    sd = StateDir(tmp_path / "flat")
    sd.create()
    now = 100000.0

    def write(series):
        with open(sd.root / "metrics.jsonl", "w", encoding="utf-8") as f:
            for i, (w, h) in enumerate(series):
                g = 1000000 + i * 20000
                f.write(json.dumps(_gen(now - (len(series) - i) * 600, w, h, g)) + "\n")

    # 1 点ずつのばらつき（±0.02）はあるが横ばい → 続ける。以前の「上がっていない」は出さない
    flat = [(0.66, 0.655), (0.68, 0.645), (0.65, 0.668), (0.67, 0.648), (0.66, 0.662), (0.65, 0.651),
            (0.67, 0.659), (0.66, 0.646), (0.68, 0.664), (0.66, 0.653), (0.67, 0.661), (0.66, 0.657)]
    write(flat)
    r = review(sd, now=now)["items"][0]
    assert r["verdict"] == OK and "横ばい" in r["why"], r
    assert abs(r["values"]["fall"]) < 0.01

    # 端の 1 点だけ低くても裏返らない（中央値で見る）
    write(flat[:-1] + [(0.66, 0.630)])
    assert review(sd, now=now)["items"][0]["verdict"] == OK

    # 本当に下がった（-0.05）→ 注意
    write(flat[:6] + [(0.66, 0.605), (0.65, 0.611), (0.67, 0.603), (0.66, 0.608), (0.67, 0.606), (0.66, 0.604)])
    r = review(sd, now=now)["items"][0]
    assert r["verdict"] == WARN and "下がった" in r["why"], r

    # 一度 0.6 を超えた run が 0.6 を割った → 見直し
    write(flat[:6] + [(0.60, 0.55), (0.59, 0.56), (0.60, 0.55), (0.58, 0.54), (0.59, 0.55), (0.60, 0.56)])
    r = review(sd, now=now)["items"][0]
    assert r["verdict"] == REVIEW and "下回った" in r["why"], r

    # まだ一度も 0.6 に届いていない run（学習の初め）は、0.6 未満でも見直しにしない
    write([(0.30, 0.28), (0.32, 0.30), (0.34, 0.31), (0.36, 0.33), (0.38, 0.35), (0.40, 0.37)])
    assert review(sd, now=now)["items"][0]["verdict"] == OK

    # 丸暗記（差 0.3）は今までどおり注意
    write(flat[:6] + [(0.95, 0.65), (0.95, 0.65), (0.95, 0.65), (0.95, 0.65), (0.95, 0.65), (0.95, 0.65)])
    r = review(sd, now=now)["items"][0]
    assert r["verdict"] == WARN and "窓の記憶" in r["why"], r


def test_m4_reference_uses_the_last_two_points(tmp_path):
    """M4（固定の参照）は最後の 2 回を比べる。

    2026-09-20 まで「reference_games 局の窓」で点を選んでいたが、窓（40 万局）が節目の間隔
    （every_games = 40 万局）と同じで、直前の点が数百局ぶん窓からはみ出していた。活きている参照は
    いつまでも「比べる点がまだ 1 つ」になり、もう測っていない参照だけが判定を出していた。"""
    sd = StateDir(tmp_path / "m4")
    sd.create()
    now = 100000.0
    sd.status_json.write_text(json.dumps({"games_total": 2800000}))

    def refs(rows):
        _write(sd, "reference.jsonl", rows)
        return {i["name"]: i for i in review(sd, now=now)["items"] if i["name"].startswith("M4")}

    # 節目が 40 万局ごとでも、数百局のはみ出しで判定が消えない（実測 400,430 局差）
    live = [{"t": now - 7200, "step": 1000, "games": 2415686, "ref": "live.pt", "elo": 200.0, "ci95": [150.0, 250.0], "score_new": 0.6},
            {"t": now - 100, "step": 2000, "games": 2816116, "ref": "live.pt", "elo": 240.0, "ci95": [190.0, 290.0], "score_new": 0.65}]
    it = refs(live)["M4 参照 live.pt"]
    assert it["verdict"] == OK and "400,430 局で +40.0" in it["why"] and it["values"]["gap_games"] == 400430

    # 下がったら注意
    down = [live[0], {**live[1], "elo": 180.0}]
    assert refs(down)["M4 参照 live.pt"]["verdict"] == WARN

    # 天井（得点が 0.2〜0.8 の外）なら、下がっていても弱くなった証拠にならないので判定しない
    ceil = [{**live[0], "ref": "top.pt", "score_new": 0.84}, {**live[1], "ref": "top.pt", "elo": 180.0, "score_new": 0.87}]
    it = refs(ceil)["M4 参照 top.pt"]
    assert it["verdict"] == NA and "天井" in it["why"]

    # もう測っていない参照（最後が 160 万局前）は run 全体の判定を動かさない
    old = [*live, {"t": now - 9000, "step": 500, "games": 800000, "ref": "old.pt", "elo": 380.0, "ci95": [340.0, 420.0], "score_new": 0.9},
           {"t": now - 8000, "step": 600, "games": 1200000, "ref": "old.pt", "elo": 370.0, "ci95": [330.0, 410.0], "score_new": 0.895}]
    items = refs(old)
    assert items["M4 参照 old.pt"]["verdict"] == NA and "もう測っていない" in items["M4 参照 old.pt"]["why"]
    assert items["M4 参照 old.pt"]["values"]["behind_games"] == 1600000


def test_review_reports_a_failed_measurement_job(tmp_path):
    """計測ジョブが失敗したら判定に出す。2026-09-21 まで、外部計測が 3 回続けて 1 局も記録を
    残していないのに判定はずっと「続ける」だった（理由は auto.log にしか出ていなかった）。"""
    sd = StateDir(tmp_path / "x")
    sd.create()
    sd.write_state({"auto": {"history": [
        {"kind": "best", "rc": 0},
        {"kind": "match", "rc": 3, "tail": ["opening ~/fuseki-shogi-ai/vendor/yaneuraou_eval failed"]},
    ]}})
    r = review(sd)
    item = next(i for i in r["items"] if i["name"] == "計測ジョブ")
    assert item["verdict"] == WARN
    assert "1 件が失敗（match）" in item["why"] and "yaneuraou_eval" in item["why"]
    assert item["values"]["n_failed"] == 1
    assert r["verdict"] == WARN          # run 全体の判定にも出る


def test_review_says_the_jobs_are_fine_when_they_are(tmp_path):
    sd = StateDir(tmp_path / "x")
    sd.create()
    sd.write_state({"auto": {"history": [{"kind": "best", "rc": 0}, {"kind": "match", "rc": 0}]}})
    item = next(i for i in review(sd)["items"] if i["name"] == "計測ジョブ")
    assert item["verdict"] == OK and item["values"]["n_failed"] == 0


def test_review_has_no_job_item_before_anything_ran(tmp_path):
    """履歴がまだ無い run では項目を出さない（「まだ無い」が増えるだけなので）。"""
    sd = StateDir(tmp_path / "x")
    sd.create()
    assert not any(i["name"] == "計測ジョブ" for i in review(sd)["items"])
