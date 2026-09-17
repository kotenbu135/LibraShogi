# SPDX-License-Identifier: Apache-2.0
"""自己対局ワーカーと学習側の分離: 対局ファイルの書式と検査、重みの配布、inbox の取り込み、ワーカーの停止、別プロセスでの結合。"""
import io
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

import librasearch
import librashogi as ls
from libra_league.config import dump_toml, load_config
from libra_league.replay import ReplayBuffer
from libra_league.runner import train_steps
from libra_league.state import StateDir, read_json
from libra_league.workers import (GamesFileError, Inbox, Worker, load_weights, publish_weights, read_games_file, verify_game,
                                  verify_games_file, worker_seed, write_games_file)
from libra_net.model import LibraNet, NetConfig

NET = {"d_model": 32, "n_layers": 1, "n_heads": 4, "d_ff": 64, "dropout": 0.0}


def _games(n: int, seed: int, cfg: dict | None = None, openings: tuple[list, float] | None = None) -> list[dict]:
    cfg = cfg or {"full_sims": 8, "fast_sims": 4, "full_prob": 0.5, "max_ply": 320}
    sp = librasearch.SelfPlay(cfg, 8, seed=seed, threads=2)
    if openings is not None:
        sp.set_openings(*openings)
    sq = np.zeros((8, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((8, ls.GLOB_FEATS), np.float32)
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    while len(out) < n:
        sp.collect(sq, glob)
        sp.apply(rng.standard_normal((8, ls.POLICY_SIZE), dtype=np.float32), np.tile(np.array([0.4, 0.2, 0.4], np.float32), (8, 1)))
        out += sp.take_finished()
    return out[:n]


def _tiny_cfg(tmp_path: Path) -> dict:
    cfg = load_config(None)
    cfg["run_id"] = "t"
    cfg["net"] = dict(NET)
    cfg["search"].update({"full_sims": 4, "fast_sims": 4, "proof_nodes": 0, "mate_nodes_root": 0, "max_moves_per_game": 20})
    cfg["selfplay"].update({"n_games": 4, "threads": 1, "infer_dtype": "float32"})
    cfg["train"].update({"batch_size": 8, "min_window_games": 4, "train_every_games": 4, "window_games": 100})
    cfg["run"].update({"status_seconds": 0.5, "checkpoint_minutes": 100, "chunk_games": 2, "export_onnx": False})
    cfg["workers"].update({"enabled": True, "ingest_seconds": 0.2})
    return cfg


def _same_game(a: dict, b: dict) -> bool:
    keys = ("slot", "kb", "kw", "result", "reason", "plies", "sfen41")
    arrays = ("moves", "full", "root_q", "policy_idx", "policy_p", "policy_off")
    return (all(a[k] == b[k] for k in keys) and abs(a["v41"] - b["v41"]) < 1e-6
            and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) and np.asarray(a[k]).dtype == np.asarray(b[k]).dtype for k in arrays))


def test_games_file_roundtrip(tmp_path: Path):
    games = _games(5, 3)
    p = write_games_file(tmp_path, "w1", 7, "ls", games)
    assert p.parent == tmp_path and p.suffix == ".npz" and p.name.startswith("w1-")
    assert list(tmp_path.glob("*.tmp")) == []
    meta, got = read_games_file(p)
    assert meta["worker"] == "w1" and meta["weights_step"] == 7 and meta["run_id"] == "ls"
    assert len(got) == 5 and all(_same_game(a, b) for a, b in zip(games, got))
    # 読み戻した記録でリプレイが組める
    rb = ReplayBuffer(tmp_path / "replay", tmp_path / "games", 100, 100, 320, True)
    rb.add_games(got)
    b = rb.sample(16, np.random.default_rng(0), 0.5, 0.5)
    assert b["sq"].shape[0] == 16
    # 名前は衝突しない
    assert write_games_file(tmp_path, "w1", 7, "ls", games) != p


def _rewrite(src: Path, dst: Path, edit) -> Path:
    """npz を読み、edit(arrays, meta) で書き換えて保存し直す（検査の否定例を作る）。"""
    with np.load(src, allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    meta = json.loads(arrs["meta"].tobytes().decode("utf-8"))
    edit(arrs, meta)
    if "meta" in arrs:
        arrs["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    dst.write_bytes(buf.getvalue())
    return dst


def test_games_file_rejects_bad_input(tmp_path: Path):
    good = write_games_file(tmp_path, "w1", 1, "ls", _games(3, 4))

    def put(k, v):
        return lambda a, m: a.__setitem__(k, v)

    cases = {
        "not_zip": None,
        "object_array": put("moves", np.array([{"x": 1}], dtype=object)),
        "wrong_dtype": lambda a, m: a.__setitem__("moves", a["moves"].astype(np.int64)),
        "missing_key": lambda a, m: a.pop("root_q"),
        "extra_key": put("extra", np.zeros(3, np.uint8)),
        "bad_offsets": lambda a, m: a.__setitem__("policy_off", a["policy_off"][::-1].copy()),
        "policy_out_of_range": lambda a, m: a.__setitem__("policy_idx", np.full_like(a["policy_idx"], ls.POLICY_SIZE)),
        "nan_prob": lambda a, m: a.__setitem__("policy_p", np.full_like(a["policy_p"], np.nan)),
        "bad_result": lambda a, m: m["games"][0].__setitem__("result", 2),
        "bad_reason": lambda a, m: m["games"][0].__setitem__("reason", "x" * 100),
        "bad_square": lambda a, m: m["games"][0].__setitem__("kb", 81),
        "count_mismatch": lambda a, m: m["games"].pop(),
        "moves_length": lambda a, m: a.__setitem__("n_moves", a["n_moves"] + 1),
        "bad_worker": lambda a, m: m.__setitem__("worker", "../x"),
        "extra_game_key": lambda a, m: m["games"][0].__setitem__("exploiter_result", 1),
    }
    for name, edit in cases.items():
        p = tmp_path / f"bad-{name}.npz"
        if edit is None:
            p.write_bytes(b"not a zip")
        else:
            _rewrite(good, p, edit)
        with pytest.raises(GamesFileError):
            read_games_file(p)
    # 展開後の大きさで弾く（圧縮爆弾）
    big = tmp_path / "big.npz"
    with zipfile.ZipFile(big, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("moves.npy", b"\0" * (4 << 20))
    with pytest.raises(GamesFileError):
        read_games_file(big, max_bytes=1 << 20)


def _search(**kw) -> dict:
    s = {"max_ply": 320, "count_from_41": True, "max_moves_per_game": 400}
    s.update(kw)
    return s


def test_verify_game_accepts_selfplay_records(tmp_path: Path):
    """自己対局が作る局は、終局した局・手数の上限で打ち切った局・布石から始めた局のどれも再生の検査を通る。"""
    games = _games(20, 11)
    for g in games:
        verify_game(g, _search())
    cut = _games(12, 12, {"full_sims": 8, "fast_sims": 4, "full_prob": 0.5, "max_ply": 320, "max_moves_per_game": 30})
    assert any(g["reason"] == "timeout" for g in cut)
    for g in cut:
        verify_game(g, _search(max_moves_per_game=30))
    ops = [[ls.move_from_usi("K*" + ls.sq_to_usi(g["kb"])), ls.move_from_usi("K*" + ls.sq_to_usi(g["kw"]))]
           + [int(x) for x in g["moves"][:6]] for g in games[:4]]
    opened = _games(12, 14, openings=(ops, 1.0))
    assert any([int(x) for x in g["moves"][:6]] == op[2:] for g in opened for op in ops)
    for g in opened:
        verify_game(g, _search())
    # ファイルに書いて読み戻しても通る
    meta, got = verify_games_file(write_games_file(tmp_path, "w1", 1, "ls", games), _search())
    assert len(got) == 20


def test_verify_game_rejects_tampered_records(tmp_path: Path):
    games = _games(8, 15)
    g0 = next(g for g in games if g["result"] != 0 and len(g["moves"]) > 10 and int(g["policy_off"][-1]) > 0)
    search = _search()

    def edited(edit) -> dict:
        g = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in g0.items()}
        edit(g)
        return g

    def after_end(g):
        g["moves"] = np.append(g["moves"], g["moves"][-1])
        g["full"] = np.append(g["full"], np.uint8(0))
        g["root_q"] = np.append(g["root_q"], np.float32(0))
        g["policy_off"] = np.append(g["policy_off"], g["policy_off"][-1])

    # 方策の添字を、その局面で合法でない手の添字に変える
    j = next(i for i in range(len(g0["moves"])) if g0["policy_off"][i + 1] > g0["policy_off"][i])
    p = ls.Position()
    p.do_move("K*" + ls.sq_to_usi(g0["kb"]))
    p.do_move("K*" + ls.sq_to_usi(g0["kw"]))
    for m in g0["moves"][:j]:
        p.do_move_code(int(m))
    legal = {p.move_index(ls.move_to_usi(c)) for c in p.legal_move_codes()}
    not_legal = next(i for i in range(ls.POLICY_SIZE) if i not in legal)
    cases = {
        "illegal_move": (lambda g: g["moves"].__setitem__(3, ls.move_from_usi("K*5e")), "move 3: illegal"),
        "after_end": (after_end, "after the end"),
        "result": (lambda g: g.__setitem__("result", -g["result"]), "outcome"),
        "reason": (lambda g: g.__setitem__("reason", "sennichite"), "outcome"),
        "plies": (lambda g: g.__setitem__("plies", g["plies"] + 2), "plies"),
        "sfen41": (lambda g: g.__setitem__("sfen41", games[-1]["sfen41"] + " "), "sfen41"),
        "king": (lambda g: g.__setitem__("kb", g["kw"]), "king placement"),
        "policy_index": (lambda g: g["policy_idx"].__setitem__(int(g["policy_off"][j]), not_legal), "policy index"),
    }
    for name, (edit, msg) in cases.items():
        with pytest.raises(GamesFileError, match=msg):
            verify_game(edited(edit), search)
    # 打ち切りの局は、打ち切りの手数に届いていなければ弾く
    cut = next(g for g in _games(12, 12, {"full_sims": 8, "fast_sims": 4, "full_prob": 0.5, "max_ply": 320, "max_moves_per_game": 30})
               if g["reason"] == "timeout")
    with pytest.raises(GamesFileError, match="unfinished"):
        verify_game(cut, search)
    # 1 局でも合わなければファイルごと弾く（どの局かを理由に書く）
    bad = write_games_file(tmp_path, "w1", 1, "ls", [games[0], edited(lambda g: g.__setitem__("plies", g["plies"] + 2))])
    with pytest.raises(GamesFileError, match="game 1: plies"):
        verify_games_file(bad, search)


def test_weights_publish_and_load(tmp_path: Path):
    m = LibraNet(NetConfig.from_dict(NET))
    p = tmp_path / "weights" / "latest.pt"
    p.parent.mkdir()
    publish_weights(p, m, 42, NET, "ls")
    assert list(p.parent.glob("*.tmp")) == []
    m2, step, run_id = load_weights(p)
    assert step == 42 and run_id == "ls" and m2.cfg == m.cfg
    for (k, a), (_, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert a.dtype == b.dtype and torch.allclose(a, b, atol=1e-3), k
    # 配布物は fp16（学習用の optimizer は入れない）で、weights_only で読める
    raw = torch.load(p, map_location="cpu", weights_only=True)
    assert set(raw) == {"format", "model", "step", "net", "run_id"}
    assert all(v.dtype == torch.float16 for v in raw["model"].values() if v.is_floating_point())


def test_inbox_ingest(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    g = _games(6, 5)
    write_games_file(inbox, "a", 100, "ls", g[:2])
    write_games_file(inbox, "b", 95, "ls", g[2:4])
    write_games_file(inbox, "a", 10, "ls", g[4:5])      # 古い重みで打った局
    write_games_file(inbox, "c", 100, "other", g[5:6])  # 別の run
    (inbox / "broken.npz").write_bytes(b"junk")
    (inbox / "a-writing.npz.tmp").write_bytes(b"partial")  # 書きかけは読まない
    logs: list[str] = []
    ib = Inbox(inbox, "ls", max_lag_steps=50, log=logs.append)
    got = ib.poll(step=100)
    assert len(got) == 4
    assert ib.stats["games"] == 4 and ib.stats["files"] == 2 and ib.stats["stale_games"] == 1 and ib.stats["rejected_files"] == 2
    assert ib.stats["by_worker"] == {"a": 2, "b": 2}
    assert sorted(p.name for p in inbox.iterdir() if p.is_file()) == ["a-writing.npz.tmp"]
    assert len(list((inbox / "rejected").iterdir())) == 2
    assert any("rejected" in s for s in logs) and any("stale" in s for s in logs)
    assert ib.poll(step=100) == []


def test_train_steps_counts_games_not_sources():
    """学習量は新規局数だけで決まる（手元の自己対局とワーカーの局を区別しない。replay_ratio の意味を変えない）。"""
    tr = {"replay_ratio": 4.0, "batch_size": 1024}
    assert train_steps(256, 200.0, tr) == round(256 * 200.0 * 4.0 / 1024)
    assert train_steps(1, 1.0, tr) == 1


def test_worker_seed_depends_on_id():
    assert worker_seed("a", 1) != worker_seed("b", 1)
    assert worker_seed("a", 1) != worker_seed("a", 2)
    assert worker_seed("a", 1) == worker_seed("a", 1) and 0 <= worker_seed("a", 1) < 2**63


def _wait(cond, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, what
        time.sleep(0.1)


def test_worker_writes_games_reloads_weights_and_stops(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    root = tmp_path / "t"
    root.mkdir()
    weights = root / "weights" / "latest.pt"
    weights.parent.mkdir()
    publish_weights(weights, LibraNet(NetConfig.from_dict(NET)), 5, NET, "t")
    inboxes = {}
    workers = {}
    threads = []
    for wid in ("a", "b"):
        inboxes[wid] = tmp_path / f"inbox-{wid}"
        inboxes[wid].mkdir()
        workers[wid] = Worker(cfg, wid, weights, inboxes[wid], torch.device("cpu"), stop_root=root, lock=None, entropy=7,
                              reload_seconds=0.2, log=lambda s: None)
        threads.append(threading.Thread(target=workers[wid].run, daemon=True))
    for t in threads:
        t.start()
    _wait(lambda: all(list(d.glob("*.npz")) for d in inboxes.values()), 120, "worker files")
    publish_weights(weights, LibraNet(NetConfig.from_dict(NET)), 9, NET, "t")
    _wait(lambda: workers["a"].step == 9, 60, "reload")
    n_before = len(list(inboxes["a"].glob("*.npz")))
    _wait(lambda: len(list(inboxes["a"].glob("*.npz"))) > n_before + 1, 120, "files after reload")
    (root / "STOP").write_text("1")
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive()
    files = {wid: sorted(d.glob("*.npz"), key=lambda p: p.stat().st_mtime) for wid, d in inboxes.items()}
    metas = [read_games_file(p)[0] for p in files["a"]]
    assert metas[0]["weights_step"] == 5 and metas[-1]["weights_step"] == 9 and all(m["worker"] == "a" for m in metas)
    # id が違えば同じ entropy でも別の対局列になる
    ga = read_games_file(files["a"][0])[1]
    gb = read_games_file(files["b"][0])[1]
    assert not all(_same_game(x, y) for x, y in zip(ga, gb))


def test_worker_waits_for_learner_and_exits_when_it_stops(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path)
    root = tmp_path / "t"
    root.mkdir()
    weights = root / "weights" / "latest.pt"
    weights.parent.mkdir()
    publish_weights(weights, LibraNet(NetConfig.from_dict(NET)), 1, NET, "t")
    (root / "STOP").write_text("1")  # 止まっている run に残った STOP では抜けない（学習側の起動で消える）
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    logs: list[str] = []
    w = Worker(cfg, "a", weights, inbox, torch.device("cpu"), stop_root=root, lock=root / "run.lock", poll_seconds=0.1,
               lock_seconds=0.1, perf_seconds=0.2, log=logs.append)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    _wait(lambda: any("waiting for the learner" in s for s in logs), 30, "waiting log")
    assert t.is_alive() and not list(inbox.iterdir())
    (root / "STOP").unlink()
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "libra_league"])
    try:
        (root / "run.lock").write_text(str(holder.pid))
        # 段ごとの時間の行も待つ（速い CPU では perf_seconds より先に対局ファイルが出て、学習側を止めると 1 行も出ずに抜ける）
        _wait(lambda: list(inbox.glob("*.npz")) and any(": perf " in s for s in logs), 120, "games and perf after learner started")
    finally:
        holder.kill()
        holder.wait()
    t.join(timeout=60)
    assert not t.is_alive() and any("learner not running" in s for s in logs)
    # 段ごとの時間（クラウドのホストで CPU と GPU のどちらが遅いかを分ける）
    perf = [s for s in logs if ": perf " in s]
    assert perf and all(k in perf[0] for k in ("evals/s", "collect", "eval", "apply"))


def test_selfplay_round_timing():
    """timing に dict を入れたときだけ段ごとの時間を累計する（既定の None では測らない）。"""
    from libra_league.selfplay import SelfPlayLoop

    cfg = load_config(None)
    cfg["search"].update({"full_sims": 4, "fast_sims": 4, "proof_nodes": 0, "mate_nodes_root": 0, "max_moves_per_game": 20})
    loop = SelfPlayLoop(cfg["search"], 4, 1, 3, torch.device("cpu"), "float32")
    loop.set_model(LibraNet(NetConfig.from_dict(NET)))
    loop.round()
    assert loop.timing is None
    loop.timing = {}
    for _ in range(5):
        loop.round()
    tm = loop.timing
    assert tm["rounds"] == 5 and all(tm[k] > 0 for k in ("collect", "eval", "apply")) and tm["proof"] >= 0


def test_selfplay_loop_eval_cache_on_for_self_play_and_off_for_exploiter():
    """自己対局のループはネットの出力のキャッシュを使い（重みを替えるたびに捨てる）、搾取者モード（手番でネットが替わる）でも使う。"""
    from libra_league.selfplay import SelfPlayLoop

    torch.manual_seed(0)
    cfg = load_config(None)
    cfg["search"].update({"full_sims": 8, "fast_sims": 8, "proof_nodes": 0, "mate_nodes_root": 0, "max_moves_per_game": 30})
    loop = SelfPlayLoop(cfg["search"], 4, 1, 3, torch.device("cpu"), "float32")
    assert loop.engine.eval_cache_enabled()
    loop.set_model(LibraNet(NetConfig.from_dict(NET)))
    for _ in range(200):
        loop.round()
    assert loop.stats()["cache_hits"] > 0
    loop.set_opponent(LibraNet(NetConfig.from_dict(NET)))  # 両方のネットの出力を持つので切らない
    assert loop.engine.eval_cache_enabled() and loop.engine.two_nets()
    hits = loop.stats()["cache_hits"]
    for _ in range(200):
        loop.round()
    assert loop.stats()["cache_hits"] > hits
    off = SelfPlayLoop({**cfg["search"], "eval_cache": False}, 4, 1, 3, torch.device("cpu"), "float32")
    assert not off.engine.eval_cache_enabled()


def test_runner_ingests_games_from_worker_process(tmp_path: Path):
    """学習側（run --no-supervise）とワーカーを別プロセスで回し、ワーカーの局が学習側の窓に入り、停止で両方抜けること。"""
    cfg = _tiny_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(dump_toml(cfg), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), CUDA_VISIBLE_DEVICES="")
    base = [sys.executable, "-m", "libra_league.cli", "--root", str(tmp_path), "--run", "t"]
    sd = StateDir(tmp_path / "t")
    learner = subprocess.Popen(base + ["run", "--no-supervise", "--config", str(cfg_path)], env=env,
                               stdout=open(tmp_path / "learner.out", "wb"), stderr=subprocess.STDOUT)
    worker = None
    try:
        _wait(lambda: (sd.root / "weights" / "latest.pt").exists(), 120, "published weights")
        worker = subprocess.Popen(base + ["worker", "--id", "w1", "--n-games", "4", "--threads", "1"], env=env,
                                  stdout=open(tmp_path / "worker.out", "wb"), stderr=subprocess.STDOUT)
        _wait(lambda: (read_json(sd.status_json, {}) or {}).get("workers", {}).get("games", 0) >= 2, 240, "ingested games")
        sd.set_flag("STOP")
        assert learner.wait(timeout=120) == 0
        assert worker.wait(timeout=120) == 0
    finally:
        for p in (learner, worker):
            if p is not None and p.poll() is None:
                p.kill()
                p.wait()
    st = read_json(sd.status_json)
    assert st["workers"]["by_worker"].get("w1", 0) >= 2 and st["games_total"] >= st["workers"]["games"]
    assert "workers: ingested" in sd.log.read_text(encoding="utf-8")
    assert (tmp_path / "worker.out").read_text().count("exit after") == 1
