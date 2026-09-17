# SPDX-License-Identifier: Apache-2.0
"""進捗の時系列・archive・自動ジョブ（libra_league.auto）。"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from libra_league.auto import AutoJobs, append_metrics, collect_anchor, archive_checkpoint, collect_evals, collect_matches, list_archives, load_metrics
from libra_league.cli import main
from libra_league.config import load_config
from libra_league.state import StateDir


def _status(step, games):
    return {"time": "2026-09-12 00:00:00", "step": step, "generation": 1, "games_total": games, "games_per_day_1h": 1000,
            "active_games": 8, "window_games": 100,
            "train": {"loss": 1.0, "policy": 0.5, "value": 0.2, "v41": 0.3, "policy_acc": 0.4, "lr": 1e-4,
                      "target": {"target_minus_z": -0.018, "draw_target": 0.32}},
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
    assert rows[-1]["train"]["target"]["target_minus_z"] == -0.018  # 学習目標と結果の差もコンソールに渡す
    assert len(load_metrics(sd, 1000)) == 50


def test_metrics_gpd_5m(tmp_path):
    # 局/日の 5 分平均は間引く前の隣り合う行の差で出す。一時停止などで間が空いた行と局数が戻った行は出さない
    sd = StateDir(tmp_path / "x")
    sd.create()
    rows = [(0, 0), (300, 100), (600, 300), (3600, 400), (3900, 350), (4200, 450)]
    with open(sd.root / "metrics.jsonl", "w", encoding="utf-8") as f:
        for t, g in rows:
            f.write(json.dumps({"t": t, "games_total": g}) + "\n")
        f.write("{broken\n")
    got = [r["gpd_5m"] for r in load_metrics(sd, 1000)]
    assert got == [None, 28800, 57600, None, None, 28800]
    assert [r["gpd_5m"] for r in load_metrics(sd, 3)] == [None, 57600, None, 28800]  # 間引いても各点は 5 分の値


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
    # 最初の archive が基準になるので、2 個目は基準との対局 1 回だけ（鎖も兼ねる）
    assert [j["kind"] for j in state["auto"]["queue"]] == ["match", "anchor"]
    for _ in range(200):
        jobs.poll()
        if not state["auto"]["queue"] and jobs.proc is None:
            break
        time.sleep(0.05)
    hist = state["auto"]["history"]
    assert [h["kind"] for h in hist] == ["match", "anchor"] and all(h["rc"] == 0 for h in hist)
    assert json.loads(open(hist[1]["out"]).read())["kind"] == "eval"
    assert state["auto"]["anchor"]["step"] == 10  # 最初の archive が基準
    assert (sd.root / "auto.log").exists()
    assert not (sd.root / "auto_job.json").exists()  # 終わったジョブの記録は消える
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


def _anchor_jobs(tmp_path, **acfg):
    """eval を「結果 JSON を書くだけ」の偽プロセスに差し替えた AutoJobs。"""
    sd = StateDir(tmp_path / "x")
    sd.create()
    fake = tmp_path / "fake_eval.py"
    fake.write_text(
        "import sys, json, pathlib\n"
        "a = sys.argv[sys.argv.index('--a') + 1]; b = sys.argv[sys.argv.index('--b') + 1]\n"
        "o = pathlib.Path(sys.argv[sys.argv.index('--out') + 1]); o.parent.mkdir(parents=True, exist_ok=True)\n"
        "elo, score = float(pathlib.Path(b + '.elo').read_text()), float(pathlib.Path(b + '.score').read_text())\n"
        "o.write_text(json.dumps({'a': a, 'b': b, 'n': 100, 'score_a': score, 'elo_a_minus_b': elo,\n"
        "                         'elo_ci95': [elo - 70, elo + 70]}))\n")
    cfg = load_config(None)
    cfg["auto"].update({"enabled": True, "every_hours": 1.0, "match_games": 0, "chain_eval": False})
    cfg["auto"].update(acfg)
    state = {}
    jobs = AutoJobs(sd, cfg, state, print, cmd_prefix=[sys.executable, str(fake)])
    return sd, jobs, state


def _archive(sd, jobs, step, now, elo=-100.0, score=0.3):
    """step の世代を archive に足して on_checkpoint を呼ぶ。elo/score は「その世代を基準にしたときの結果」。"""
    p = sd.checkpoints / f"ckpt_{step:09d}.pt"
    p.write_bytes(b"x" * 8)
    (sd.checkpoints / "archive").mkdir(parents=True, exist_ok=True)
    (sd.checkpoints / "archive" / f"ckpt_{step:09d}.pt.elo").write_text(str(elo))
    (sd.checkpoints / "archive" / f"ckpt_{step:09d}.pt.score").write_text(str(score))
    jobs.on_checkpoint(p, now=now)
    for _ in range(400):
        jobs.poll()
        if not jobs.state["auto"]["queue"] and jobs.proc is None:
            return
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_anchor_chain_accumulates_offset_only_on_rebaseline(tmp_path):
    sd, jobs, state = _anchor_jobs(tmp_path)
    _archive(sd, jobs, 1000, 1000.0)  # 最初の世代が基準になる
    assert state["auto"]["anchor"]["step"] == 1000 and state["auto"]["anchor"]["offset"] == 0.0
    # 基準より 100 Elo 強く、勝率 70%: 基準は据え置き
    _archive(sd, jobs, 2000, 1000.0 + 3600, elo=-100.0, score=0.3)
    rows = collect_anchor(sd)
    assert len(rows) == 1
    assert rows[0]["step"] == 2000 and rows[0]["elo_vs_anchor"] == 100.0 and rows[0]["elo"] == 100.0
    assert rows[0]["ci95"] == [30.0, 170.0] and rows[0]["anchor_step"] == 1000
    assert state["auto"]["anchor"]["step"] == 1000
    # 基準に 90% 勝つ世代: 基準を置き換えて差を offset に足す
    _archive(sd, jobs, 3000, 1000.0 + 7200, elo=-380.0, score=0.1)
    rows = collect_anchor(sd)
    assert rows[-1]["elo"] == 380.0 and state["auto"]["anchor"] == {
        "file": str(sd.checkpoints / "archive" / "ckpt_000003000.pt"), "step": 3000, "offset": 380.0,
        "since": state["auto"]["anchor"]["since"]}
    # 置き換え後は offset が足される
    _archive(sd, jobs, 4000, 1000.0 + 10800, elo=-50.0, score=0.35)
    rows = collect_anchor(sd)
    assert rows[-1]["anchor_step"] == 3000 and rows[-1]["elo_vs_anchor"] == 50.0 and rows[-1]["elo"] == 430.0


def test_anchor_doubles_as_chain_when_it_is_the_previous_generation(tmp_path):
    sd, jobs, state = _anchor_jobs(tmp_path, chain_eval=True)
    _archive(sd, jobs, 1000, 1000.0)                      # 基準になる
    _archive(sd, jobs, 2000, 1000.0 + 3600, elo=-100.0)   # 基準 = 直前の世代 → 対局は 1 回だけ
    assert [h["kind"] for h in state["auto"]["history"]] == ["anchor"]
    ev = collect_evals(sd)
    assert len(ev) == 1 and ev[0]["cumulative"] == 100.0  # 鎖にも使う
    _archive(sd, jobs, 3000, 1000.0 + 7200, elo=-150.0)   # 基準は 1000 のまま → 鎖と基準で 2 回
    assert [h["kind"] for h in state["auto"]["history"]][-2:] == ["eval", "anchor"]


def test_anchor_disabled(tmp_path):
    sd, jobs, state = _anchor_jobs(tmp_path, anchor_games=0)
    _archive(sd, jobs, 1000, 1000.0)
    _archive(sd, jobs, 2000, 1000.0 + 3600)
    assert state["auto"]["anchor"] is None and collect_anchor(sd) == []


def test_no_spawn_while_suspended(tmp_path):
    """停止・一時停止中に新しいジョブを起動しない（積むのは良い。再開後に走る）。

    ランナーは stop / pause でも checkpoint() を通り、その中で on_checkpoint → poll を呼ぶ。
    ここで起動すると start_new_session=True の子が親の終了後も GPU を使い続ける。
    """
    sd = StateDir(tmp_path / "x")
    sd.create()
    fake = tmp_path / "fake.py"
    fake.write_text("import sys, json, pathlib, time\ntime.sleep(30)\n")
    cfg = load_config(None)
    cfg["auto"].update({"enabled": True, "every_hours": 1.0, "match_games": 2})
    state: dict = {}
    jobs = AutoJobs(sd, cfg, state, lambda _m: None, cmd_prefix=[sys.executable, str(fake)])

    # (1) 走っているジョブが無い状態で stop → checkpoint が積んだジョブを起動しない
    jobs.stop()
    c0 = sd.checkpoints / "ckpt_000000010.pt"
    c0.write_bytes(b"a")
    jobs.on_checkpoint(c0, now=1000.0)
    assert [j["kind"] for j in state["auto"]["queue"]] == ["match"]
    assert jobs.poll() is False and jobs.proc is None
    assert [j["kind"] for j in state["auto"]["queue"]] == ["match"]  # 積んだまま残る

    # (2) 再開すると走る
    jobs.resume()
    assert jobs.poll() is True and jobs.proc is not None
    assert not state["auto"]["queue"] and state["auto"]["running"]["kind"] == "match"

    # (3) 走っているジョブの途中で stop → 止めて積み直し、直後の poll でも再起動しない
    pid = jobs.proc.pid
    jobs.stop()
    assert jobs.proc is None and [j["kind"] for j in state["auto"]["queue"]] == ["match"]
    assert jobs.poll() is False and jobs.proc is None
    try:
        os.kill(pid, 0)
        assert False, "子プロセスが残っている"
    except (ProcessLookupError, PermissionError):
        pass
    assert not (sd.root / "auto_job.json").exists()


def _crash_setup(tmp_path):
    """途中まで出力を書いて眠る計測ジョブ。ランナーの abort は stop() を通らないので、子は孤児として残る。"""
    sd = StateDir(tmp_path / "x")
    sd.create()
    fake = tmp_path / "fake.py"
    fake.write_text("import sys, pathlib, time\no = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
                    "o.parent.mkdir(parents=True, exist_ok=True)\no.write_text('partial\\n')\ntime.sleep(60)\n")
    cfg = load_config(None)
    cfg["auto"].update({"enabled": True, "every_hours": 1.0, "match_games": 2})
    return sd, cfg, [sys.executable, str(fake)]


def _wait_file(p, timeout=15.0):
    deadline = time.monotonic() + timeout
    while not p.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert p.exists()


def test_recover_kills_orphan_and_requeues(tmp_path):
    sd, cfg, prefix = _crash_setup(tmp_path)
    state1: dict = {}
    jobs1 = AutoJobs(sd, cfg, state1, lambda _m: None, cmd_prefix=prefix)
    jobs1.enqueue_match()
    assert jobs1.poll() is True
    out = Path(state1["auto"]["running"]["out"])
    _wait_file(out)
    assert (sd.root / "auto_job.json").exists()
    saved = json.loads(json.dumps(state1))  # 起動後のチェックポイントで保存された state

    logs: list = []
    state2 = saved
    jobs2 = AutoJobs(sd, cfg, state2, logs.append, cmd_prefix=prefix)
    jobs2.recover()
    assert jobs1.proc.wait(timeout=15) is not None  # 孤児は止まる
    assert state2["auto"]["running"] is None
    assert [j["out"] for j in state2["auto"]["queue"]] == [str(out)]
    assert not out.exists() and Path(str(out) + ".interrupted").exists()  # 途中の棋譜に追記しない
    assert not (sd.root / "auto_job.json").exists()
    assert any("orphan" in m for m in logs)
    assert jobs2.poll() is True and jobs2.proc is not None  # 積み直した分が走る
    jobs2.stop()


def test_recover_job_started_after_last_save(tmp_path):
    """最後の保存より後に起動したジョブ: state では待ち行列に残っているので、二重に積まない。"""
    sd, cfg, prefix = _crash_setup(tmp_path)
    state1: dict = {}
    jobs1 = AutoJobs(sd, cfg, state1, lambda _m: None, cmd_prefix=prefix)
    jobs1.enqueue_match()
    saved = json.loads(json.dumps(state1))
    assert jobs1.poll() is True
    _wait_file(Path(state1["auto"]["running"]["out"]))

    state2 = saved
    jobs2 = AutoJobs(sd, cfg, state2, lambda _m: None, cmd_prefix=prefix)
    jobs2.recover()
    assert jobs1.proc.wait(timeout=15) is not None
    assert len(state2["auto"]["queue"]) == 1 and state2["auto"]["running"] is None


def test_recover_legacy_running_without_args(tmp_path):
    sd, cfg, prefix = _crash_setup(tmp_path)
    state = {"auto": {"queue": [], "history": [], "running": {"kind": "eval", "started": 1.0, "out": str(sd.root / "eval" / "a.json")}}}
    jobs = AutoJobs(sd, cfg, state, lambda _m: None, cmd_prefix=prefix)
    jobs.recover()
    assert state["auto"]["running"] is None and state["auto"]["queue"] == []


def test_recover_does_not_kill_reused_pid(tmp_path):
    sd, cfg, prefix = _crash_setup(tmp_path)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        (sd.root / "auto_job.json").write_text(json.dumps({"pid": other.pid, "kind": "match", "args": ["match", "--out", "/nonexistent/x.jsonl"],
                                                            "out": "/nonexistent/x.jsonl", "started": 1.0}))
        state: dict = {}
        jobs = AutoJobs(sd, cfg, state, lambda _m: None, cmd_prefix=prefix)
        jobs.recover()
        time.sleep(0.3)
        assert other.poll() is None  # 別のプロセスには触らない
        assert not (sd.root / "auto_job.json").exists()
        assert [j["kind"] for j in state["auto"]["queue"]] == ["match"]
    finally:
        other.kill()
        other.wait()


def test_best_tracks_strongest_and_counts_stall(tmp_path):
    """最強比（docs/restart-plan.md §3 M2）: 95% 区間の下限が 0 を超えたときだけ最強を置き換え、それ以外は足踏みを数える。"""
    from libra_league.auto import collect_best

    sd, jobs, state = _anchor_jobs(tmp_path, anchor_games=0, best_games=1000, best_stall_alert=2)
    _archive(sd, jobs, 1000, 1000.0)  # 最初の世代が最強
    assert state["auto"]["best"]["step"] == 1000 and state["auto"]["best_stall"] == 0
    # +100 Elo（区間 +30〜+170）: 置き換え
    _archive(sd, jobs, 2000, 1000.0 + 3600, elo=-100.0, score=0.3)
    rows = collect_best(sd)
    assert rows[-1]["improved"] and rows[-1]["elo_vs_best"] == 100.0 and rows[-1]["ci95"] == [30.0, 170.0] and rows[-1]["best_step"] == 1000
    assert state["auto"]["best"]["step"] == 2000
    # +20 Elo（区間 −50〜+90）: 据え置き、足踏み 1
    _archive(sd, jobs, 3000, 1000.0 + 7200, elo=-20.0, score=0.47)
    assert state["auto"]["best"]["step"] == 2000 and state["auto"]["best_stall"] == 1 and not collect_best(sd)[-1]["improved"]
    # −40 Elo: 足踏み 2（WARNING）
    _archive(sd, jobs, 4000, 1000.0 + 10800, elo=40.0, score=0.55)
    assert state["auto"]["best_stall"] == 2 and collect_best(sd)[-1]["stall"] == 2
    # 最強と同じ世代は積まない
    jobs.enqueue_best(sd.checkpoints / "archive" / "ckpt_000002000.pt")
    assert not state["auto"]["queue"]
    assert [h["kind"] for h in state["auto"]["history"]] == ["best", "best", "best"]


def test_reference_evals_against_fixed_checkpoints(tmp_path):
    """固定の参照（同 M4）: run をまたいだ同じ相手との差を eval/reference.jsonl に残す。無い参照は飛ばす。"""
    from libra_league.auto import collect_reference
    from libra_league.config import dump_toml

    (tmp_path / "refs").mkdir()
    ref = tmp_path / "refs" / "ckpt_000000500.pt"  # step が名前から読めない参照（win1m.pt など）は ref_step が None になる
    ref.write_bytes(b"r")
    sd, jobs, state = _anchor_jobs(tmp_path, anchor_games=0, reference_games=200, reference_ckpts=[str(ref), str(tmp_path / "missing.pt")])
    _archive(sd, jobs, 1000, 1000.0, elo=-150.0, score=0.3)
    rows = collect_reference(sd)
    assert len(rows) == 1 and rows[0]["ref"] == ref.name and rows[0]["ref_step"] == 500 and rows[0]["step"] == 1000
    assert rows[0]["elo"] == 150.0 and rows[0]["ci95"] == [80.0, 220.0] and rows[0]["score_new"] == 0.7
    assert [h["kind"] for h in state["auto"]["history"]] == ["reference"]
    # 設定の写し（TOML）は文字列のリストを書ける
    assert 'reference_ckpts = ["' in dump_toml(jobs.cfg)
    import tomllib

    assert tomllib.loads(dump_toml(jobs.cfg))["auto"]["reference_ckpts"] == [str(ref), str(tmp_path / "missing.pt")]


def test_every_games_triggers_by_games_not_time(tmp_path):
    """every_games > 0 なら、総局数がその分たまるごとに archive と計測を積む（時間は見ない）。行には games が入る。"""
    from libra_league.auto import collect_best

    sd, jobs, state = _anchor_jobs(tmp_path, anchor_games=0, best_games=100, every_games=1000, every_hours=0.0)
    state["games_total"] = 0
    _archive(sd, jobs, 100, 1000.0)                  # 最初は必ず
    assert state["auto"]["best"]["step"] == 100 and state["auto"]["last_archive_games"] == 0
    state["games_total"] = 900
    _archive(sd, jobs, 200, 1000.0 + 86400)          # 1 日たっても 900 局では積まない
    assert [p.name for p in list_archives(sd)] == ["ckpt_000000100.pt"]
    state["games_total"] = 1000
    _archive(sd, jobs, 300, 1000.0 + 86400 + 1, elo=-100.0, score=0.3)  # 1,000 局で積む
    assert [p.name for p in list_archives(sd)] == ["ckpt_000000100.pt", "ckpt_000000300.pt"]
    rows = collect_best(sd)
    assert rows[-1]["games"] == 1000 and rows[-1]["improved"]
    # 区切りは every_games の倍数（1,000・2,000…）。前回が半端（1,870）でも次は 2,000 局で積む（2026-09-17 のユーザーの希望）
    state["auto"]["last_archive_games"] = 1870
    state["games_total"] = 1999
    _archive(sd, jobs, 400, 1000.0 + 86400 + 2)
    assert len(list_archives(sd)) == 2
    state["games_total"] = 2003
    _archive(sd, jobs, 500, 1000.0 + 86400 + 3, elo=-100.0, score=0.3)
    assert len(list_archives(sd)) == 3 and state["auto"]["last_archive_games"] == 2003
    # every_hours だけの run は今までどおり時間で
    sd2, jobs2, state2 = _anchor_jobs(tmp_path / "h", anchor_games=0, best_games=100, every_hours=1.0)
    state2["games_total"] = 0
    _archive(sd2, jobs2, 100, 1000.0)
    _archive(sd2, jobs2, 200, 1000.0 + 1800)
    assert len(list_archives(sd2)) == 1
    _archive(sd2, jobs2, 300, 1000.0 + 3600, elo=-100.0, score=0.3)
    assert len(list_archives(sd2)) == 2
    # ランナーは倍数を越えたら 10 分を待たずにチェックポイントを取る（games_due）。前回が無い・時間区切りの run では見ない
    assert not jobs.games_due(2999) and jobs.games_due(3000)
    state["auto"]["last_archive_games"] = None
    assert not jobs.games_due(3000)
    assert not jobs2.games_due(10**6)



def test_anchor_reuses_best_result_when_opponent_is_the_same(tmp_path):
    """最強と基準が同じ重みで局数も同じなら、基準比は最強比の結果を写して打たない（2026-09-18 のユーザーの指示「重複を省く」）。"""
    sd, jobs, state = _anchor_jobs(tmp_path, anchor_games=100, best_games=100)
    _archive(sd, jobs, 1000, 1000.0)                                  # 最強・基準とも 1000
    _archive(sd, jobs, 2000, 1000.0 + 3600, elo=-100.0, score=0.3)    # 最強比 1 回だけ打ち、基準比は写す
    kinds = [(h["kind"], bool(h.get("reused"))) for h in state["auto"]["history"]]
    assert kinds == [("best", False), ("anchor", True)]
    assert (sd.root / "auto.log").read_text().count("anchor: eval") == 0
    rows = collect_anchor(sd)
    assert len(rows) == 1 and rows[0]["anchor_step"] == 1000 and rows[0]["elo"] == 100.0 and rows[0]["n"] == 100
    assert len(list((sd.root / "eval").glob("anchor-*-1000-2000.json"))) == 1
    # 最強は 2000 に替わり、基準は 1000 のまま → 次は相手が違うので両方打つ
    _archive(sd, jobs, 3000, 1000.0 + 7200, elo=-150.0, score=0.28)
    assert [(h["kind"], bool(h.get("reused"))) for h in state["auto"]["history"]][-2:] == [("best", False), ("anchor", False)]
    # 局数が違えば写さない
    sd2, jobs2, state2 = _anchor_jobs(tmp_path / "n", anchor_games=100, best_games=50)
    _archive(sd2, jobs2, 1000, 1000.0)
    _archive(sd2, jobs2, 2000, 1000.0 + 3600, elo=-100.0, score=0.3)
    assert [h["kind"] for h in state2["auto"]["history"]] == ["best", "anchor"] and not state2["auto"]["history"][1].get("reused")


def test_anchor_offset_uses_the_opponent_actually_played_when_jobs_back_up(tmp_path):
    """計測が積み上がり、先の結果で基準が替わっても、後の結果は実際に打った相手（積んだときの基準）の offset で数える。
    基準を置き換えるのは、今の基準と打った結果のときだけ（2026-09-18 に ls で累積 +965.9 と誤記録した不具合）。"""
    sd, jobs, state = _anchor_jobs(tmp_path)
    _archive(sd, jobs, 1000, 1000.0)                                   # 基準 1000
    for step, now, elo, score in ((2000, 3600.0, -380.0, 0.1), (3000, 7200.0, -300.0, 0.12)):
        p = sd.checkpoints / f"ckpt_{step:09d}.pt"
        p.write_bytes(b"x" * 8)
        (sd.checkpoints / "archive").mkdir(parents=True, exist_ok=True)
        (sd.checkpoints / "archive" / f"ckpt_{step:09d}.pt.elo").write_text(str(elo))
        (sd.checkpoints / "archive" / f"ckpt_{step:09d}.pt.score").write_text(str(score))
        jobs.on_checkpoint(p, now=1000.0 + now)                        # 2 つとも基準 1000 で積む
    for _ in range(400):
        jobs.poll()
        if not state["auto"]["queue"] and jobs.proc is None:
            break
        time.sleep(0.05)
    rows = collect_anchor(sd)
    assert [(r["step"], r["anchor_step"], r["offset"], r["elo"]) for r in rows] == [(2000, 1000, 0.0, 380.0), (3000, 1000, 0.0, 300.0)]
    assert state["auto"]["anchor"]["step"] == 2000 and state["auto"]["anchor"]["offset"] == 380.0  # 3000 は古い基準との結果なので置き換えない


def test_repair_anchor_chain_from_result_files(tmp_path):
    """起動時に eval/anchor-*.json から基準比の行と今の基準を数え直し、違っていれば古い anchor.jsonl を残して書き直す。
    値は 2026-09-17〜18 の ls の実際の 4 つの結果。"""
    sd, jobs, state = _anchor_jobs(tmp_path, anchor_games=1000)
    ev = sd.root / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    arch = sd.checkpoints / "archive"
    arch.mkdir(parents=True, exist_ok=True)
    for s in (402, 1416, 8144, 13807, 21270):
        (arch / f"ckpt_{s:09d}.pt").write_bytes(b"x")
    for ts, a, b, elo, score in (("20260917-202955", 402, 1416, -273.0, 0.172), ("20260917-220428", 402, 8144, -354.5, 0.115),
                                 ("20260917-224425", 402, 13807, -339.6, 0.124), ("20260917-232228", 8144, 21270, -271.8, 0.173)):
        (ev / f"anchor-{ts}-{a}-{b}.json").write_text(json.dumps({"a": str(arch / f"ckpt_{a:09d}.pt"), "b": str(arch / f"ckpt_{b:09d}.pt"), "n": 1000,
                                                                  "score_a": score, "elo_a_minus_b": elo, "elo_ci95": [elo - 30, elo + 30]}))
    wrong = [{"t": 1.0, "step": 1416, "games": 91053, "n": 1000, "score_new": 0.828, "anchor_step": 402, "offset": 0.0, "elo_vs_anchor": 273.0, "elo": 273.0, "ci95": [243.0, 303.0]},
             {"t": 2.0, "step": 8144, "games": 288985, "n": 1000, "score_new": 0.885, "anchor_step": 402, "offset": 0.0, "elo_vs_anchor": 354.5, "elo": 354.5, "ci95": [324.5, 384.5]},
             {"t": 3.0, "step": 13807, "games": 414962, "n": 1000, "score_new": 0.876, "anchor_step": 8144, "offset": 354.5, "elo_vs_anchor": 339.6, "elo": 694.1, "ci95": [664.1, 724.1]},
             {"t": 4.0, "step": 21270, "games": 440084, "n": 1000, "score_new": 0.827, "anchor_step": 13807, "offset": 694.1, "elo_vs_anchor": 271.8, "elo": 965.9, "ci95": [935.9, 995.9]}]
    (ev / "anchor.jsonl").write_text("".join(json.dumps(r) + "\n" for r in wrong))
    state.setdefault("auto", {})["anchor"] = {"file": str(arch / "ckpt_000013807.pt"), "step": 13807, "offset": 694.1, "since": 5.0}
    assert jobs.repair_anchor_chain() is True
    rows = collect_anchor(sd)
    assert [(r["step"], r["anchor_step"], r["offset"], r["elo"]) for r in rows] == [(1416, 402, 0.0, 273.0), (8144, 402, 0.0, 354.5),
                                                                                 (13807, 402, 0.0, 339.6), (21270, 8144, 354.5, 626.3)]
    assert rows[3]["games"] == 440084 and rows[3]["t"] == 4.0 and rows[3]["ci95"] == [596.3, 656.3]
    assert state["auto"]["anchor"]["step"] == 8144 and state["auto"]["anchor"]["offset"] == 354.5
    assert len(list(ev.glob("anchor.jsonl.bak-*"))) == 1
    assert jobs.repair_anchor_chain() is False                          # 2 回目は直すものが無い
