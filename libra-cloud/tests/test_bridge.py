# SPDX-License-Identifier: Apache-2.0
"""ブリッジ（libra_cloud.bridge）: 重みと布石の送信、ホストの対局ファイルの回収・検査・学習側の inbox への配置、停止と打ち切り。
ssh の代わりに手元のディレクトリをホストの run に見立てる（LocalTransport）。"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

import librasearch
import librashogi as ls
from libra_cloud.bridge import NAME, Bridge, LocalTransport, SSHTransport, push_timeout, serve
from libra_league.config import dump_toml, load_config
from libra_league.workers import read_games_file, write_games_file


def _games(n: int, seed: int) -> list[dict]:
    sp = librasearch.SelfPlay({"full_sims": 8, "fast_sims": 4, "full_prob": 0.5, "max_ply": 320}, 8, seed=seed, threads=2)
    sq = np.zeros((8, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((8, ls.GLOB_FEATS), np.float32)
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    while len(out) < n:
        sp.collect(sq, glob)
        sp.apply(rng.standard_normal((8, ls.POLICY_SIZE), dtype=np.float32), np.tile(np.array([0.4, 0.2, 0.4], np.float32), (8, 1)))
        out += sp.take_finished()
    return out[:n]


def _learner(tmp_path: Path, inbox: bool = True) -> Path:
    """学習側の run（config.toml、weights/latest.pt、布石、[workers] enabled なら inbox/）。"""
    run = tmp_path / "ls"
    (run / "weights").mkdir(parents=True)
    cfg = load_config(None)
    cfg["run_id"] = "ls"
    op = tmp_path / "openings.json"
    op.write_text("[]", encoding="utf-8")
    cfg["selfplay"]["openings"] = str(op)
    (run / "config.toml").write_text(dump_toml(cfg), encoding="utf-8")
    (run / "weights" / "latest.pt").write_bytes(b"w1")
    if inbox:
        (run / "inbox").mkdir()
    return run


def _quiet(s: str) -> None:
    pass


def test_bridge_pushes_weights_and_openings_when_changed(tmp_path: Path):
    run = _learner(tmp_path)
    host = tmp_path / "host"
    b = Bridge(run, LocalTransport(host), tmp_path / "out", log=_quiet)
    b.cycle()
    assert (host / "weights" / "latest.pt").read_bytes() == b"w1" and (host / "openings.json").read_text() == "[]"
    b.cycle()
    assert b.stats["pushes"] == {"weights/latest.pt": 1, "openings.json": 1}  # 変わっていなければ送らない
    w = run / "weights" / "latest.pt"
    w.write_bytes(b"w2")
    st = w.stat()
    os.utime(w, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    b.cycle()
    assert (host / "weights" / "latest.pt").read_bytes() == b"w2" and b.stats["pushes"]["weights/latest.pt"] == 2
    assert list((host / "weights").glob("*.tmp")) == []


def test_bridge_pulls_verifies_and_places_games(tmp_path: Path):
    run = _learner(tmp_path)
    host = tmp_path / "host"
    (host / "inbox").mkdir(parents=True)
    games = _games(6, 21)
    good = write_games_file(host / "inbox", "vast1", 5, "ls", games[:3])
    tampered = [games[3], {**games[4], "plies": games[4]["plies"] + 2}, games[5]]
    bad = write_games_file(host / "inbox", "vast1", 5, "ls", tampered)
    (host / "inbox" / "vast1-1-000009.npz.tmp").write_bytes(b"partial")  # ワーカーの書きかけは取らない
    assert NAME.match(good.name) and NAME.match(bad.name)
    logs: list[str] = []
    b = Bridge(run, LocalTransport(host), tmp_path / "out", log=logs.append)
    assert b.cycle() == 3
    placed = sorted((run / "inbox").iterdir())
    assert [p.name for p in placed] == [good.name] and len(read_games_file(placed[0])[1]) == 3
    assert sorted(p.name for p in (host / "inbox").iterdir()) == ["vast1-1-000009.npz.tmp"]
    assert [p.name for p in (tmp_path / "out" / "rejected").iterdir()] == [bad.name]
    assert any("game 1" in s and "plies" in s for s in logs)
    assert b.stats["games"] == 3 and b.stats["files"] == 1 and b.stats["rejected_files"] == 1 and b.stats["verify_ms_per_game"] > 0
    assert list((tmp_path / "out" / "staging").iterdir()) == []
    assert b.cycle() == 0


def test_bridge_does_not_pull_until_learner_takes_worker_games(tmp_path: Path):
    run = _learner(tmp_path, inbox=False)
    host = tmp_path / "host"
    (host / "inbox").mkdir(parents=True)
    write_games_file(host / "inbox", "vast1", 5, "ls", _games(2, 22))
    logs: list[str] = []
    b = Bridge(run, LocalTransport(host), tmp_path / "out", log=logs.append)
    assert b.cycle() == 0 and b.cycle() == 0
    assert len(list((host / "inbox").glob("*.npz"))) == 1 and (host / "weights" / "latest.pt").exists()  # 重みは送る
    assert sum("not pulling" in s for s in logs) == 1
    (run / "inbox").mkdir()  # 学習側が [workers] enabled で起動した
    assert b.cycle() == 2


def test_push_failure_does_not_stop_pulling_and_is_retried(tmp_path: Path):
    """重みの送信が詰まって失敗しても、同じ周回で局を回収する。送れなかった重みは次の周回で送り直す。"""
    run = _learner(tmp_path)
    host = tmp_path / "host"
    (host / "inbox").mkdir(parents=True)
    write_games_file(host / "inbox", "vast1", 5, "ls", _games(2, 26))

    class StalledPush(LocalTransport):
        fail = True

        def push(self, src: Path, rel: str) -> None:
            if self.fail and rel == "weights/latest.pt":
                raise subprocess.TimeoutExpired(["scp"], 163.0)
            super().push(src, rel)

    t = StalledPush(host)
    logs: list[str] = []
    b = Bridge(run, t, tmp_path / "out", log=logs.append)
    assert b.cycle() == 2 and b.stats["errors"] == 1 and not (host / "weights" / "latest.pt").exists()
    assert (host / "openings.json").exists() and any("push weights/latest.pt failed" in s for s in logs)
    t.fail = False
    assert b.cycle() == 0 and (host / "weights" / "latest.pt").read_bytes() == b"w1" and b.stats["pushes"]["weights/latest.pt"] == 1


def test_push_records_duration_and_age(tmp_path: Path):
    """送信ごとに、かかった時間と学習側が書いてからホストに届くまでの時間（age）をログと状態に残す。"""
    run = _learner(tmp_path)
    host = tmp_path / "host"

    class SlowPush(LocalTransport):
        def push(self, src: Path, rel: str) -> None:
            time.sleep(0.05)
            super().push(src, rel)

    w = run / "weights" / "latest.pt"
    st = w.stat()
    os.utime(w, ns=(st.st_atime_ns, st.st_mtime_ns - 2 * 10**9))  # 学習側が 2 秒前に書いた
    logs: list[str] = []
    b = Bridge(run, SlowPush(host), tmp_path / "out", log=logs.append)
    b.cycle()
    ps = b.stats["push_s"]["weights/latest.pt"]
    assert ps["n"] == 1 and 0.05 <= ps["max"] < 1.0 and ps["total"] == ps["max"] and 2.0 <= ps["max_age"] < 5.0
    assert any(s.startswith("bridge: pushed weights/latest.pt 0.0 MB in 0.1 s (age 2.") for s in logs)
    b.cycle()
    assert b.stats["push_s"]["weights/latest.pt"]["n"] == 1  # 変わっていなければ送らないので記録もしない
    b.write_status()
    assert json.loads((tmp_path / "out" / "bridge.json").read_text())["push_s"]["openings.json"]["n"] == 1


def test_weights_push_waits_for_min_interval_and_bytes_are_counted(tmp_path: Path):
    """重みは前の送信から min_push_s 経つまで送らない（学習のたびに 20 MB を送ると 3 時間で 6.7 GB。max_lag_steps 2000 は約 24 分）。
    布石は変わったらすぐ送る。送った・取ってきたバイト数を数える（転送料の計上）。"""
    run = _learner(tmp_path)
    host = tmp_path / "host"
    (host / "inbox").mkdir(parents=True)
    now = [1000.0]
    b = Bridge(run, LocalTransport(host), tmp_path / "out", min_push_s=300, clock=lambda: now[0], log=_quiet)
    b.cycle()
    assert b.stats["pushes"]["weights/latest.pt"] == 1 and b.stats["push_bytes"] == 2 + 2  # b"w1" と "[]"
    w = run / "weights" / "latest.pt"

    def rewrite(data: bytes) -> None:
        w.write_bytes(data)
        st = w.stat()
        os.utime(w, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))

    rewrite(b"w2")
    now[0] += 100
    b.cycle()
    assert b.stats["pushes"]["weights/latest.pt"] == 1 and (host / "weights" / "latest.pt").read_bytes() == b"w1"
    op = Path(b.openings)
    op.write_text("[1]", encoding="utf-8")
    st = op.stat()
    os.utime(op, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    b.cycle()
    assert b.stats["pushes"]["openings.json"] == 2 and (host / "openings.json").read_text() == "[1]"  # 布石は待たない
    now[0] += 201
    rewrite(b"w3!")
    b.cycle()
    assert b.stats["pushes"]["weights/latest.pt"] == 2 and (host / "weights" / "latest.pt").read_bytes() == b"w3!"
    assert b.stats["push_bytes"] == 2 + 2 + 3 + 3
    f = write_games_file(host / "inbox", "vast1", 5, "ls", _games(2, 27))
    size = f.stat().st_size
    assert b.cycle() == 2 and b.stats["pull_bytes"] == size


def test_push_timeout_scales_with_size():
    assert push_timeout(0) == 60.0
    assert 150 < push_timeout(20_501_387) < 170  # ls の fp16 の重み。通常は 10 秒で送れる


def test_serve_stops_worker_and_drains(tmp_path: Path):
    run = _learner(tmp_path)
    host = tmp_path / "host"
    (host / "inbox").mkdir(parents=True)

    class StoppingHost(LocalTransport):
        stopped = False

        def stop_worker(self, timeout: float) -> None:  # 止めると残りの局を書いてから抜けるワーカー
            self.stopped = True
            write_games_file(self.root / "inbox", "vast1", 5, "ls", _games(2, 23))

    t = StoppingHost(host)
    b = Bridge(run, t, tmp_path / "out", log=_quiet)
    write_games_file(host / "inbox", "vast1", 5, "ls", _games(2, 24))
    calls: list[int] = []
    rc = serve(b, interval=0.0, deadline=time.time() + 60, stop=lambda: calls.append(1) is None and len(calls) > 2, max_error_s=10)
    assert rc == 0 and t.stopped and b.stats["games"] == 4
    assert json.loads((tmp_path / "out" / "bridge.json").read_text(encoding="utf-8"))["games"] == 4


def test_serve_gives_up_when_host_is_gone(tmp_path: Path):
    class DeadHost(LocalTransport):
        def list_inbox(self) -> list[str]:
            raise RuntimeError("ssh rc=255")

    run = _learner(tmp_path)
    b = Bridge(run, DeadHost(tmp_path / "host"), tmp_path / "out", log=_quiet)
    rc = serve(b, interval=0.05, deadline=time.time() + 30, stop=lambda: False, max_error_s=0.2)
    assert rc == 3 and b.stats["errors"] >= 2


def test_ssh_transport_commands_with_fake_ssh(tmp_path: Path, monkeypatch):
    """SSHTransport が送るシェルのコマンド（一覧・tar での取得・削除・原子的な送信・ワーカーの停止）を、
    最後の引数を手元の bash で実行する偽の ssh と、手元でコピーする偽の scp で確かめる。"""
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "ssh").write_text('#!/usr/bin/env bash\nfor a; do last="$a"; done\nexec bash -c "$last"\n')
    (fake / "scp").write_text('#!/usr/bin/env bash\nargs=("$@"); n=${#args[@]}\nexec cp "${args[$((n-2))]}" "${args[$((n-1))]#*:}"\n')
    for f in fake.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    remote = tmp_path / "host" / "ls"
    (remote / "inbox").mkdir(parents=True)
    games = _games(4, 25)
    a = write_games_file(remote / "inbox", "vast1", 5, "ls", games[:2])
    b = write_games_file(remote / "inbox", "vast1", 5, "ls", games[2:])
    (remote / "inbox" / "vast1-1-000009.npz.tmp").write_bytes(b"partial")
    # 止める相手のワーカー（孫プロセスにして、止まったら init が回収するようにする）
    pid = int(subprocess.run(["bash", "-c", "sleep 120 > /dev/null 2>&1 & echo $!"], capture_output=True, text=True, check=True).stdout)
    pid_file = tmp_path / "worker.pid"
    pid_file.write_text(str(pid))
    t = SSHTransport("h", 22, str(remote), key=tmp_path / "key", pid_file=str(pid_file))
    try:
        assert t.list_inbox() == sorted([a.name, b.name])
        staging = tmp_path / "staging"
        staging.mkdir()
        t.fetch([a.name, b.name], staging)
        assert (staging / a.name).read_bytes() == a.read_bytes() and (staging / b.name).read_bytes() == b.read_bytes()
        t.delete([a.name, b.name])
        assert t.list_inbox() == [] and (remote / "inbox" / "vast1-1-000009.npz.tmp").exists()
        w = tmp_path / "latest.pt"
        w.write_bytes(b"w")
        t.push(w, "weights/latest.pt")
        assert (remote / "weights" / "latest.pt").read_bytes() == b"w" and list((remote / "weights").glob("*.tmp")) == []
        t.stop_worker(20)
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("worker was not stopped")
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
