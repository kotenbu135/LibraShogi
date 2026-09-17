# SPDX-License-Identifier: Apache-2.0
"""`libra review`（docs/restart-plan.md §3 M6・§7 P4）。"""
import json

from libra_league.cli import main
from libra_league.review import NA, OK, REVIEW, WARN, review
from libra_league.state import StateDir


def _gen(t, win, held):
    return {"t": t, "games_total": 1, "gen": {"window": {"normal": {"corr_v": win}, "fuseki": {"corr_v": 0.5}},
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
        for t, w, h in ((now - 3 * 3600, 0.5, 0.40), (now - 2 * 3600, 0.55, 0.45), (now - 3600, 0.6, 0.52)):
            f.write(json.dumps(_gen(t, w, h)) + "\n")
    _write(sd, "best.jsonl", [{"t": now - 100, "step": 2000, "best_step": 1000, "elo_vs_best": 60.0, "ci95": [30.0, 90.0], "improved": True, "stall": 0}])
    _write(sd, "anchor.jsonl", [{"t": now - 100, "step": 2000, "anchor_step": 1000, "elo_vs_anchor": 60.0, "elo": 60.0, "ci95": [30.0, 90.0]}])
    _write(sd, "reference.jsonl", [{"t": now - 7200, "step": 1000, "ref": "ckpt_000646699.pt", "elo": -400.0, "ci95": [-450.0, -350.0]},
                                   {"t": now - 100, "step": 2000, "ref": "ckpt_000646699.pt", "elo": -300.0, "ci95": [-350.0, -250.0]}])
    sd.status_json.write_text(json.dumps({"games_per_day_1h": 400000}))
    r = review(sd, now=now)
    assert r["verdict"] == OK and [i["verdict"] for i in r["items"]] == [OK, OK, OK, OK, OK]
    assert abs(r["items"][0]["values"]["gap"] - 0.08) < 1e-9
    # 窓の記憶（差 0.3）→ 注意。局/日の下限 → 注意
    with open(sd.root / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(_gen(now - 10, 0.9, 0.6)) + "\n")
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
