# SPDX-License-Identifier: Apache-2.0
"""進捗の時系列・archive・自動ジョブ（libra_league.auto）。"""
import json
import sys
import time

from libra_league.auto import AutoJobs, append_metrics, archive_checkpoint, collect_evals, collect_matches, list_archives, load_metrics
from libra_league.cli import main
from libra_league.config import load_config
from libra_league.state import StateDir


def _status(step, games):
    return {"time": "2026-09-12 00:00:00", "step": step, "generation": 1, "games_total": games, "games_per_day_1h": 1000,
            "active_games": 8, "window_games": 100,
            "train": {"loss": 1.0, "policy": 0.5, "value": 0.2, "v41": 0.3, "policy_acc": 0.4, "lr": 1e-4},
            "engine": {"games": games, "moves": games * 10, "sims": games * 100, "sente_wins": 1, "draws": 0, "gote_wins": 1,
                       "ruling41": 0, "no_legal_move": 2, "sennichite": 0, "perpetual_check": 0, "max_ply": 0, "plies_sum": 80.0,
                       "mate_found": 0, "proof_found": 0},
            "gpu": {"mem_alloc_mb": 1, "mem_reserved_mb": 2}}


def test_metrics_append_and_downsample(tmp_path):
    sd = StateDir(tmp_path / "x")
    sd.create()
    for i in range(50):
        append_metrics(sd, _status(i, i * 2))
    rows = load_metrics(sd, 10)
    assert len(rows) <= 11 and rows[0]["step"] == 0 and rows[-1]["step"] == 49
    assert rows[-1]["engine"]["games"] == 98 and rows[-1]["train"]["loss"] == 1.0 and rows[-1]["gpu_mb"] == 2
    assert len(load_metrics(sd, 1000)) == 50


def test_archive_and_eval_chain(tmp_path):
    sd = StateDir(tmp_path / "x")
    sd.create()
    for step in (0, 100, 200):
        (sd.checkpoints / f"ckpt_{step:09d}.pt").write_bytes(b"x")
        archive_checkpoint(sd, sd.checkpoints / f"ckpt_{step:09d}.pt")
    assert archive_checkpoint(sd, sd.checkpoints / "ckpt_000000100.pt") is None  # 重複は写さない
    assert [p.name for p in list_archives(sd)] == ["ckpt_000000000.pt", "ckpt_000000100.pt", "ckpt_000000200.pt"]
    ev = sd.root / "eval"
    ev.mkdir()
    a = str(sd.checkpoints / "archive")
    (ev / "auto-1-0-100.json").write_text(json.dumps({"a": f"{a}/ckpt_000000000.pt", "b": f"{a}/ckpt_000000100.pt", "n": 10, "elo_a_minus_b": -50.0, "elo_ci95": [-80, -20], "score_a": 0.4}))
    (ev / "auto-2-100-200.json").write_text(json.dumps({"a": f"{a}/ckpt_000000100.pt", "b": f"{a}/ckpt_000000200.pt", "n": 10, "elo_a_minus_b": -30.0, "elo_ci95": [-60, 0], "score_a": 0.45}))
    (ev / "manual.json").write_text(json.dumps({"a": "/tmp/old.pt", "b": "/tmp/new.pt", "n": 4, "elo_a_minus_b": 10.0}))
    evs = collect_evals(sd)
    assert [e["cumulative"] for e in evs] == [None, 50.0, 80.0]  # 手動（archive 以外）は鎖に入れない
    md = sd.root / "matches"
    md.mkdir()
    (md / "auto-1.summary.json").write_text(json.dumps({"n": 10, "a_points": 2.5, "b": "Opp 0.1", "go": "movetime 1000", "libra_options": {"DNN_Model": "/x/ckpt_000000200.pt"}}))
    ms = collect_matches(sd)
    assert ms[0]["winrate"] == 0.25 and ms[0]["libra_step"] == 200 and ms[0]["auto"]


def test_autojobs_runs_subprocess(tmp_path):
    sd = StateDir(tmp_path / "x")
    sd.create()
    fake = tmp_path / "fake.py"
    fake.write_text("import sys, json, pathlib\no = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
                    "o.parent.mkdir(parents=True, exist_ok=True)\no.write_text(json.dumps({'kind': sys.argv[1]}))\n")
    cfg = load_config(None)
    cfg["auto"].update({"enabled": True, "every_hours": 1.0, "match_games": 2})
    state = {}
    logs = []
    jobs = AutoJobs(sd, cfg, state, logs.append, cmd_prefix=[sys.executable, str(fake)])
    c0 = sd.checkpoints / "ckpt_000000010.pt"
    c0.write_bytes(b"a")
    jobs.on_checkpoint(c0, now=1000.0)
    # 最初は archive だけ（相手がいない）。match は last_match が無いので積まれる
    assert [j["kind"] for j in state["auto"]["queue"]] == ["match"]
    c1 = sd.checkpoints / "ckpt_000000020.pt"
    c1.write_bytes(b"b")
    jobs.on_checkpoint(c1, now=1000.0 + 1800)  # 期限前: 何もしない
    assert len(list_archives(sd)) == 1
    sd.set_flag("EVAL_NOW")
    jobs.on_checkpoint(c1, now=1000.0 + 1800)  # フラグで前倒し
    assert not sd.flag("EVAL_NOW") and len(list_archives(sd)) == 2
    assert [j["kind"] for j in state["auto"]["queue"]] == ["match", "eval"]
    for _ in range(200):
        jobs.poll()
        if not state["auto"]["queue"] and jobs.proc is None:
            break
        time.sleep(0.05)
    hist = state["auto"]["history"]
    assert [h["kind"] for h in hist] == ["match", "eval"] and all(h["rc"] == 0 for h in hist)
    assert json.loads(open(hist[1]["out"]).read())["kind"] == "eval"
    assert (sd.root / "auto.log").exists()
    jobs.stop()


def test_status_json_history(tmp_path, capsys):
    sd = StateDir(tmp_path / "x")
    sd.create()
    sd.write_state({"step": 1, "auto": {"last_archive": 1.0, "queue": [], "history": []}})
    append_metrics(sd, _status(1, 2))
    assert main(["--root", str(tmp_path), "--run", "x", "status", "--json", "--history", "100"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["metrics"]) == 1 and out["evals"] == [] and out["matches"] == [] and out["archives"] == []
    assert out["auto"]["last_archive"] == 1.0
    assert main(["--root", str(tmp_path), "--run", "x", "eval-now"]) == 0
    assert sd.flag("EVAL_NOW")
