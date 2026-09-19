# SPDX-License-Identifier: Apache-2.0
"""`libra status --json`（Windows の管理コンソールが読む形）。"""
import json

from libra_league.cli import main
from libra_league.state import StateDir


def test_status_json_no_run(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "--run", "x", "status", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["exists"] is False and out["process"] == "not running" and out["status"] is None


def test_status_json_with_run(tmp_path, capsys):
    sd = StateDir(tmp_path / "x")
    sd.create()
    sd.write_state({"step": 5, "generation": 1, "games_total": 42, "last_checkpoint": 1.0, "elapsed": 2.0})
    sd.status_json.write_text(json.dumps({"time": "t", "step": 5, "generation": 1, "games_total": 42, "games_per_day_1h": 100, "window_games": 10, "elapsed_h": 0.1, "active_games": 8}), encoding="utf-8")
    sd.set_flag("EVAL_NOW")
    sd.append_log("hello")
    sd.append_log("world")
    (sd.root / "run.lock").write_text("999999999")  # 存在しない pid → not running
    assert main(["--root", str(tmp_path), "--run", "x", "status", "--json", "--tail", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["exists"] and out["process"] == "not running"
    assert out["flags"] == ["EVAL_NOW"] and "throttle" not in out
    assert out["state"]["games_total"] == 42 and out["status"]["games_per_day_1h"] == 100
    assert len(out["log_tail"]) == 1 and out["log_tail"][0].endswith("world")
    # 従来の表示も壊れていない
    assert main(["--root", str(tmp_path), "--run", "x", "status"]) == 0
    assert "not running" in capsys.readouterr().out


def test_default_match_model_prefers_onnx(tmp_path):
    from libra_league.cli import default_match_model

    sd = StateDir(tmp_path / "x")
    sd.create()
    assert default_match_model(sd).name == "latest.pt"
    (sd.checkpoints / "latest.onnx").write_bytes(b"x")
    assert default_match_model(sd).name == "latest.onnx"


def test_status_history_adds_games_at(tmp_path, capsys):
    """計測の行に「その重みを保存した時点の総局数」を足す（管理コンソールの Elo グラフの横軸）。

    行の `games` は打ち終わった時刻の総局数で、archive より後（実測で約 5 万局ぶん）なので使えない。
    step から metrics.jsonl を引き直す（scaling.games_of_step と同じ）。"""
    sd = StateDir(tmp_path / "x")
    sd.create()
    sd.write_state({"step": 300, "games_total": 300000})
    (sd.root / "metrics.jsonl").write_text(
        "".join(json.dumps({"t": float(s), "step": s, "games_total": s * 1000}) + "\n" for s in (0, 100, 200, 300)),
        encoding="utf-8")
    ev = sd.root / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    # step 100 の重みで打ち、終わったのは総局数 150,000 のとき（games は 150000 でも archive は 100,000 局の時点）
    (ev / "anchor.jsonl").write_text(json.dumps(
        {"t": 1.0, "step": 100, "games": 150000, "n": 100, "score_new": 0.6, "anchor_step": 50,
         "offset": 0.0, "elo_vs_anchor": 70.4, "elo": 70.4, "ci95": [20.0, 121.0]}) + "\n", encoding="utf-8")
    (ev / "best.jsonl").write_text(json.dumps(
        {"t": 2.0, "step": 200, "games": 260000, "best_step": 100, "n": 100, "score_new": 0.6,
         "elo_vs_best": 70.4, "ci95": [20.0, 121.0], "improved": True, "stall": 0}) + "\n", encoding="utf-8")
    (ev / "reference.jsonl").write_text(json.dumps(
        {"t": 3.0, "step": 250, "games": 280000, "ref": "old.pt", "ref_step": 9, "n": 200,
         "score_new": 0.6, "elo": 70.4, "ci95": [20.0, 121.0]}) + "\n", encoding="utf-8")
    md = sd.root / "matches"
    md.mkdir(parents=True, exist_ok=True)
    # マッチの step は打った重みのファイル名から読む（collect_matches）
    (md / "auto-1.summary.json").write_text(json.dumps(
        {"n": 10, "a_points": 7.0, "b": "opp", "go": "movetime 1000",
         "libra_options": {"DNN_Model": "/x/ckpt_000000300.onnx"}}), encoding="utf-8")

    assert main(["--root", str(tmp_path), "--run", "x", "status", "--json", "--history", "100"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["anchor"][0]["games_at"] == 100000   # step 100 の時点。行の games（150,000）ではない
    assert out["best"][0]["games_at"] == 200000
    assert out["reference"][0]["games_at"] == 250000  # metrics の点の間は直線で引く
    assert out["matches"][0]["games_at"] == 300000    # マッチの step は libra_step
    assert out["anchor"][0]["games"] == 150000        # 元の値は消さない
    # 鎖（eval/*.json）も同じ。相手ではなく新しい方の step で引く
    (ev / "anchor-1-50-100.json").write_text(json.dumps(
        {"a": "/x/ckpt_000000050.pt", "b": "/x/ckpt_000000100.pt", "n": 100, "score_a": 0.4,
         "elo_a_minus_b": -70.4, "elo_ci95": [-121.0, -20.0]}), encoding="utf-8")
    assert main(["--root", str(tmp_path), "--run", "x", "status", "--json", "--history", "100"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["evals"][0]["step_b"] == 100 and out["evals"][0]["games_at"] == 100000


def test_status_history_adds_the_rating_scale(tmp_path, capsys):
    """`status --history` は「強さの目盛り」（全部の対局をまとめた Bradley-Terry の Elo）も返す。

    管理コンソールの Elo のグラフはこれを主役にする。相手ごとの線は 1 点 200 局で幅が広く、
    練習相手の入れ替えと相性で上下するので、下がっていないのに下がって見えるため
    （2026-09-19 のユーザーの「Elo さがってませんか？線がいっぱいあってよくわからない」）。"""
    sd = StateDir(tmp_path / "x")
    sd.create()
    sd.write_state({"step": 1600, "games_total": 400000})
    (sd.root / "metrics.jsonl").write_text(
        "".join(json.dumps({"t": float(s), "step": s, "games_total": g}) + "\n"
                for s, g in ((400, 100000), (800, 200000), (1600, 400000))), encoding="utf-8")
    ev = sd.root / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "best.jsonl").write_text(
        json.dumps({"t": 1.0, "step": 800, "best_step": 400, "n": 1000, "score_new": 0.75,
                    "elo_vs_best": 190.8, "ci95": [160.0, 220.0], "improved": True, "stall": 0}) + "\n"
        + json.dumps({"t": 2.0, "step": 1600, "best_step": 800, "n": 1000, "score_new": 0.75,
                      "elo_vs_best": 190.8, "ci95": [160.0, 220.0], "improved": True, "stall": 0}) + "\n",
        encoding="utf-8")

    assert main(["--root", str(tmp_path), "--run", "x", "status", "--json", "--history", "100"]) == 0
    out = json.loads(capsys.readouterr().out)
    rt = out["rating"]
    assert "error" not in rt
    assert rt["anchor"] == "step 400"
    # 点には step・総局数・時刻・区間が付く（グラフは横軸を総局数と時間で切り替える）
    assert [(p["step"], p["games"], p["t"]) for p in rt["points"]] == [
        (400, 100000, 400.0), (800, 200000, 800.0), (1600, 400000, 1600.0)]
    assert rt["points"][0]["elo"] == 0.0 and rt["points"][2]["elo"] > rt["points"][1]["elo"] > 0
    assert all(p["ci95"] is not None for p in rt["points"])
    assert rt["curve_fit"]["elo_per_doubling"] > 0 and rt["fit"]["n_nodes"] == 3
    # 「目安の線」はこちらを使う。点が 4 つに満たないうちは全部の点と同じ
    assert rt["recent_doublings"] == 4 and rt["curve_fit_recent"] == rt["curve_fit"]


def test_status_history_rating_survives_a_run_with_no_games(tmp_path, capsys):
    """対局の記録がまだ無くても status は返る（目盛りは空）。"""
    sd = StateDir(tmp_path / "x")
    sd.create()
    sd.write_state({"step": 1})
    assert main(["--root", str(tmp_path), "--run", "x", "status", "--json", "--history", "100"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["rating"]["points"] == [] and "error" not in out["rating"]


def test_status_history_games_at_without_metrics(tmp_path, capsys):
    """metrics.jsonl がまだ無い run では games_at は null（横軸は時間に落ちる）。"""
    sd = StateDir(tmp_path / "x")
    sd.create()
    sd.write_state({"step": 1})
    ev = sd.root / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "best.jsonl").write_text(json.dumps(
        {"t": 1.0, "step": 10, "best_step": 5, "n": 100, "score_new": 0.6, "elo_vs_best": 70.4,
         "ci95": [20.0, 121.0], "improved": True, "stall": 0}) + "\n", encoding="utf-8")
    assert main(["--root", str(tmp_path), "--run", "x", "status", "--json", "--history", "100"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["best"][0]["games_at"] is None
