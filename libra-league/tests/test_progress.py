# SPDX-License-Identifier: Apache-2.0
"""進捗の書き出しと別ブランチへの push（libra_league.progress、docs/runbook.md §6）。"""
import json
import subprocess
from pathlib import Path

from libra_league.cli import main
from libra_league.progress import Publisher, files_for, format_md, message_for, publish, scrub, snapshot
from libra_league.state import StateDir


def _run(sd: StateDir, home: Path) -> StateDir:
    """最強比・基準比・参照・外部計測の結果が一通りある run を作る。"""
    sd.create()
    (sd.root / "eval").mkdir(exist_ok=True)
    (sd.root / "matches").mkdir(exist_ok=True)
    ck = str(home / "libra-run" / "ls" / "checkpoints" / "archive")
    sd.write_state({"step": 28908, "generation": 5, "games_total": 400000,
                    "auto": {"anchor": {"file": f"{ck}/ckpt_000008144.pt", "step": 8144, "offset": 354.5},
                             "best": {"file": f"{ck}/ckpt_000021270.pt", "step": 21270}, "best_stall": 0,
                             "queue": [], "running": None, "last_archive_games": 400000}})
    (sd.status_json).write_text(json.dumps({"time": "2026-09-18 02:23:00", "step": 28908, "games_total": 400000,
                                            "games_per_day_1h": 620000, "window_games": 400000, "active_games": 512,
                                            "elapsed_h": 6.1, "train": {"loss": 2.1, "policy_acc": 0.42, "lr": 2e-4}}), encoding="utf-8")
    (sd.root / "eval" / "best.jsonl").write_text(json.dumps(
        {"t": 1, "step": 28908, "games": 400000, "best_step": 21270, "n": 1000, "score_new": 0.797,
         "elo_vs_best": 237.6, "ci95": [211.9, 265.7], "improved": True, "stall": 0}) + "\n", encoding="utf-8")
    (sd.root / "eval" / "anchor.jsonl").write_text(json.dumps(
        {"t": 1, "step": 28908, "games": 400000, "n": 1000, "score_new": 0.852, "anchor_step": 8144,
         "offset": 354.5, "elo_vs_anchor": 304.1, "elo": 658.6, "ci95": [629.9, 691.0]}) + "\n", encoding="utf-8")
    (sd.root / "eval" / "reference.jsonl").write_text(json.dumps(
        {"t": 1, "step": 28908, "games": 400000, "ref": "ckpt_000646699.pt", "ref_step": 646699, "n": 200,
         "score_new": 0.83, "elo": 275.5, "ci95": [217.8, 349.5]}) + "\n", encoding="utf-8")
    (sd.root / "matches" / "auto-1.summary.json").write_text(json.dumps(
        {"a": "libra", "b": "fuseki-usi", "n": 10, "a_points": 7.0, "go": "movetime 1000", "reasons": {"no_legal_move": 10},
         "libra_options": {"DNN_Model": f"{ck}/ckpt_000028908.onnx"},
         "opponent_cmd": [str(home / "fuseki-shogi-ai" / ".venv" / "bin" / "python"), "scripts/fuseki_usi_server.py"]}), encoding="utf-8")
    with open(sd.root / "metrics.jsonl", "w", encoding="utf-8") as f:
        for i in range(30):
            f.write(json.dumps({"t": i * 300, "step": i * 100, "games_total": i * 2000, "gpd": 620000, "window": 400000,
                                "train": {"loss": 2.5 - i * 0.01, "policy_acc": 0.4}}) + "\n")
    return sd


def test_scrub_removes_paths_and_home(tmp_path):
    home = "/home/kotenbu"
    got = scrub({"a": f"{home}/libra-run/ls/checkpoints/archive/ckpt_000028908.pt",
                 "b": [f"{home}/fuseki-shogi-ai/.venv/bin/python", "C:\\Users\\k\\libra\\engine\\libra.exe"],
                 "c": f"{home}/libra-run", "d": 3}, home=home)
    assert got["a"] == "ckpt_000028908.pt"
    # 拡張子のある置き場はファイル名だけにし、残りはホームを ~ にする（絶対パスを外に出さない）
    assert got["b"] == ["~/fuseki-shogi-ai/.venv/bin/python", "libra.exe"]
    assert got["c"] == "~/libra-run" and got["d"] == 3


def test_snapshot_has_the_numbers_and_no_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sd = _run(StateDir(tmp_path / "libra-run" / "ls"), tmp_path)
    s = snapshot(sd, {"auto": {"enabled": True, "every_games": 400000, "best_games": 1000}}, points=10)
    assert s["schema"] == 1 and s["run"] == "ls" and s["process"] == "not running"
    assert s["now"]["step"] == 28908 and s["now"]["games_total"] == 400000 and s["now"]["games_per_day_1h"] == 620000
    assert s["auto"]["best_step"] == 21270 and s["auto"]["anchor_step"] == 8144 and s["auto"]["anchor_offset"] == 354.5
    assert s["best"][-1]["elo_vs_best"] == 237.6 and s["anchor"][-1]["elo"] == 658.6
    assert s["reference"][-1]["ref"] == "ckpt_000646699.pt" and s["reference"][-1]["elo"] == 275.5
    assert s["matches"][-1]["winrate"] == 0.7 and s["matches"][-1]["libra_step"] == 28908
    assert s["auto_cfg"]["every_games"] == 400000 and 0 < len(s["metrics"]) <= 11
    assert s["review"]["verdict"] and [i["name"] for i in s["review"]["items"]]
    # 絶対パスとホームがどこにも残らない（公開リポジトリに置くため）
    blob = json.dumps(s, ensure_ascii=False)
    assert str(tmp_path) not in blob and "/libra-run/" not in blob and ".venv" not in blob
    assert "step 28,908" in message_for(s) and "最強の step" in format_md(s)


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--quiet", "--bare", "-b", "main", str(bare)], check=True)
    repo = tmp_path / "repo"
    subprocess.run(["git", "clone", "--quiet", str(bare), str(repo)], check=True)
    for k, v in (("user.name", "kotenbu"), ("user.email", "k@example.invalid"), ("commit.gpgsign", "false")):
        subprocess.run(["git", "-C", str(repo), "config", k, v], check=True)
    (repo / "README.md").write_text("main\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "--quiet", "-m", "init"], check=True)
    subprocess.run(["git", "-C", str(repo), "push", "--quiet", "origin", "main"], check=True)
    return bare, repo


def _show(bare: Path, ref: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(bare), "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout


def test_publish_pushes_to_its_own_branch_without_touching_the_worktree(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    bare, repo = _repo(tmp_path)
    sd = _run(StateDir(tmp_path / "libra-run" / "ls"), tmp_path)
    files, snap = files_for(sd, None, 10, "progress")
    assert sorted(files) == ["progress/ls.json", "progress/ls.md"]
    head_before = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    assert publish(repo, "progress", files, message_for(snap), log=lambda s: None)
    # 作業ツリー・HEAD・main は動かない
    assert subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout == ""
    assert subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout == head_before
    assert _show(bare, "main", "README.md") == "main\n"
    # progress ブランチに数値が入り、main の履歴は混ざらない（親なしの 1 コミット）
    got = json.loads(_show(bare, "progress", "progress/ls.json"))
    assert got["now"]["games_total"] == 400000 and got["best"][-1]["elo_vs_best"] == 237.6
    assert "最強の step | 21,270" in _show(bare, "progress", "progress/ls.md")
    log = subprocess.run(["git", "-C", str(bare), "log", "--format=%s%n%b", "progress"], capture_output=True, text=True).stdout
    assert "進捗 ls: step 28,908" in log and "Signed-off-by: kotenbu <k@example.invalid>" in log
    assert subprocess.run(["git", "-C", str(bare), "rev-list", "--count", "progress"], capture_output=True, text=True).stdout.strip() == "1"


def test_publish_is_a_noop_when_nothing_changed_and_keeps_history(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    bare, repo = _repo(tmp_path)
    sd = _run(StateDir(tmp_path / "libra-run" / "ls"), tmp_path)
    files, snap = files_for(sd, None, 10, "progress")
    assert publish(repo, "progress", files, "1", log=lambda s: None)
    assert publish(repo, "progress", files, "2", log=lambda s: None) is None  # 同じ中身なら積まない
    files2 = {**files, "progress/ls.md": files["progress/ls.md"] + "\n変わった\n"}
    assert publish(repo, "progress", files2, "3", log=lambda s: None)
    assert subprocess.run(["git", "-C", str(bare), "rev-list", "--count", "progress"], capture_output=True, text=True).stdout.strip() == "2"
    assert "変わった" in _show(bare, "progress", "progress/ls.md")
    assert json.loads(_show(bare, "progress", "progress/ls.json"))["run"] == "ls"  # 書かなかったファイルは残る


def test_publish_recovers_from_a_rejected_push(tmp_path, monkeypatch):
    """別のところから同じブランチに push があっても、読み直して積み直す。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    bare, repo = _repo(tmp_path)
    sd = _run(StateDir(tmp_path / "libra-run" / "ls"), tmp_path)
    files, snap = files_for(sd, None, 10, "progress")
    assert publish(repo, "progress", files, "1", log=lambda s: None)
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "--quiet", "--branch", "progress", str(bare), str(other)], check=True)
    for k, v in (("user.name", "x"), ("user.email", "x@example.invalid")):
        subprocess.run(["git", "-C", str(other), "config", k, v], check=True)
    (other / "progress" / "other.md").write_text("other\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(other), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(other), "commit", "--quiet", "-m", "other"], check=True)
    subprocess.run(["git", "-C", str(other), "push", "--quiet", "origin", "progress"], check=True)
    files2 = {**files, "progress/ls.md": files["progress/ls.md"] + "\n新しい\n"}
    assert publish(repo, "progress", files2, "2", log=lambda s: None)
    assert _show(bare, "progress", "progress/other.md") == "other\n"  # 相手の変更を消さない
    assert "新しい" in _show(bare, "progress", "progress/ls.md")


def test_cli_progress_writes_files_and_json(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sd = _run(StateDir(tmp_path / "libra-run" / "ls"), tmp_path)
    assert main(["--root", str(tmp_path / "libra-run"), "--run", "ls", "progress"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0])["now"]["step"] == 28908
    out = tmp_path / "out"
    assert main(["--root", str(tmp_path / "libra-run"), "--run", "ls", "progress", "--out", str(out)]) == 0
    capsys.readouterr()
    assert json.loads((out / "ls.json").read_text(encoding="utf-8"))["run"] == "ls"
    assert "進み具合" in (out / "ls.md").read_text(encoding="utf-8")


def test_cli_progress_publish(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    bare, repo = _repo(tmp_path)
    _run(StateDir(tmp_path / "libra-run" / "ls"), tmp_path)
    assert main(["--root", str(tmp_path / "libra-run"), "--run", "ls", "progress", "--publish",
                 "--repo", str(repo), "--branch", "p2", "--dir", "x"]) == 0
    assert json.loads(_show(bare, "p2", "x/ls.json"))["now"]["step"] == 28908


def test_publisher_runs_on_a_milestone_and_on_the_heartbeat(tmp_path):
    """節目（自動計測が動いた回）と heartbeat ごとだけ起動し、前の書き出しが走っていれば見送る。"""
    sd = StateDir(tmp_path / "ls")
    sd.create()

    class P(Publisher):
        """起動は `true` で代える（cmd_prefix）。走っているかは alive で決める。"""

        def __init__(self, cfg):
            super().__init__(sd, cfg, log=lambda s: None, cmd_prefix=["true"])
            self.alive = False

        def poll(self):
            self.proc = self if self.alive else None

    assert not P({"progress": {"enabled": False}}).maybe_publish(1000.0, milestone=True)
    p = P({"progress": {"enabled": True, "min_seconds": 120, "heartbeat_minutes": 60}})
    assert p.maybe_publish(1000.0, milestone=True)
    assert not p.maybe_publish(1010.0, milestone=True)       # min_seconds の内
    assert not p.maybe_publish(2000.0, milestone=False)      # 節目でなく heartbeat の前
    assert p.maybe_publish(1000.0 + 3600, milestone=False)   # heartbeat
    p.alive = True
    assert not p.maybe_publish(1000.0 + 7200 * 2, milestone=True)  # 前の書き出しが走っている


def test_runner_publishes_at_a_milestone(tmp_path):
    """ランナーは自動計測が動いた回に Publisher を呼ぶ（runner.py の組み込み）。"""
    src = (Path(__file__).resolve().parents[1] / "libra_league" / "runner.py").read_text(encoding="utf-8")
    assert "from .progress import Publisher" in src and "self.progress = Publisher(sd, cfg, self.log)" in src
    assert "self.progress.maybe_publish(now, milestone=changed)" in src
    assert "changed = self.auto.poll()" in src
