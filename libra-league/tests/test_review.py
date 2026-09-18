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
    # 参照の窓を 10 万局にすると比べる点が 1 つ
    assert review(sd, {"reference_games": 100000}, now=now)["items"][3]["verdict"] == NA
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
