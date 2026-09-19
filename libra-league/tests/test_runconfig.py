# SPDX-License-Identifier: Apache-2.0
"""稼働中のランの設定をリポジトリで管理する（docs/runbook.md §設定の管理、2026-09-18 のユーザーの依頼）。"""
import json
import subprocess
from pathlib import Path

import pytest

from libra_league.cli import main
from libra_league.runconfig import adopt, changed_keys, home_to_tilde, read_ref_config, repo_config_path, resolve
from libra_league.state import StateDir


def _sd(tmp_path, name="ls"):
    sd = StateDir(tmp_path / "run" / name)
    sd.create()
    return sd


def _repo_config(repo, run_id, text):
    p = repo_config_path(run_id, repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_changed_keys_walks_sections():
    old = {"seed": 1, "train": {"lr": 0.1, "batch_size": 8}, "auto": {"enabled": False}}
    new = {"seed": 1, "train": {"lr": 0.2, "batch_size": 8}, "auto": {"enabled": True}, "net": {"d": 3}}
    assert changed_keys(old, new) == ["auto.enabled: False → True", "net.d: None → 3", "train.lr: 0.1 → 0.2"]
    assert changed_keys(old, old) == []


def test_home_to_tilde():
    assert home_to_tilde("/home/who/libra-run/ls/x.pt", home="/home/who") == "~/libra-run/ls/x.pt"
    assert home_to_tilde("relative/x.pt", home="/home/who") == "relative/x.pt"
    assert home_to_tilde("/home/who/x", home="") == "/home/who/x"  # ホームが取れなければ何もしない


def test_repo_config_becomes_the_source(tmp_path):
    sd, repo = _sd(tmp_path), tmp_path / "repo"
    sd.config_toml.write_text('seed = 1\n[train]\nlr = 0.001\nreplay_ratio = 4.0\n', encoding="utf-8")
    _repo_config(repo, "ls", '[train]\nlr = 0.002\n')
    cfg, info = resolve(sd, None, repo, ref="none", now=1000.0, log=lambda m: None)
    assert cfg["train"]["lr"] == 0.002
    assert info["source"].endswith("config/ls.toml") and info["ref"] is None
    # 変わったキーが記録され、前の config.toml は控えに残る
    assert any("train.lr" in c for c in info["changed"])
    assert info["backup"] and (sd.root / info["backup"]).read_text(encoding="utf-8").startswith("seed = 1")
    # <run>/config.toml は作り直され、ブリッジやワーカーの束はこれまで通りそれを読める
    body = sd.config_toml.read_text(encoding="utf-8")
    assert "直接編集しても次の起動で上書きされる" in body and "lr = 0.002" in body
    # 2 回目は中身が同じなので控えを作らない
    _, info2 = resolve(sd, None, repo, ref="none", now=2000.0, log=lambda m: None)
    assert info2["backup"] is None and info2["changed"] == []


def test_local_overlay_wins_for_this_machine(tmp_path):
    sd, repo = _sd(tmp_path), tmp_path / "repo"
    _repo_config(repo, "ls", '[train]\nlr = 0.002\n[auto]\nreference_games = 200\n')
    (sd.root / "config.local.toml").write_text('[auto]\nreference_games = 20\n', encoding="utf-8")
    cfg, info = resolve(sd, None, repo, ref="none", log=lambda m: None)
    assert cfg["train"]["lr"] == 0.002 and cfg["auto"]["reference_games"] == 20 and info["local"] is True


def test_keeps_the_live_config_when_the_repo_has_none(tmp_path):
    sd, repo = _sd(tmp_path), tmp_path / "repo"
    sd.config_toml.write_text('[auto]\nreference_ckpts = ["/home/who/libra-run/ls/a.pt"]\n', encoding="utf-8")
    cfg, info = resolve(sd, None, repo, ref="none", log=lambda m: None)
    # 今動いている設定のまま（中身を推測して変えない）。ランナーはリポジトリの作業ツリーに書かない
    assert cfg["auto"]["reference_ckpts"] == ["/home/who/libra-run/ls/a.pt"]
    assert info["source"] == str(sd.config_toml) and info["adopt_hint"] == str(repo_config_path("ls", repo))
    assert not repo_config_path("ls", repo).exists()
    assert info["changed"] == [] and info["backup"] is None
    assert sd.config_toml.read_text(encoding="utf-8") == '[auto]\nreference_ckpts = ["/home/who/libra-run/ls/a.pt"]\n'


def test_adopt_copies_the_live_config_and_scrubs_the_home_path(tmp_path, monkeypatch):
    sd, repo = _sd(tmp_path), tmp_path / "repo"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    sd.config_toml.write_text(f'[auto]\nreference_ckpts = ["{tmp_path / "home"}/libra-run/ls/a.pt"]\n', encoding="utf-8")
    p, _ = adopt(sd, repo)
    assert p.read_text(encoding="utf-8") == '[auto]\nreference_ckpts = ["~/libra-run/ls/a.pt"]\n'
    with pytest.raises(FileExistsError):
        adopt(sd, repo)                      # 既にあるものは黙って上書きしない
    sd.config_toml.write_text('[train]\nlr = 0.5\n', encoding="utf-8")
    assert adopt(sd, repo, force=True)[0].read_text(encoding="utf-8") == '[train]\nlr = 0.5\n'
    with pytest.raises(FileNotFoundError):
        adopt(_sd(tmp_path, "lx"), repo)
    # 写したあとは、次の起動からリポジトリのほうが正になる
    _, info = resolve(sd, None, repo, ref="none", log=lambda m: None)
    assert info["source"].endswith("config/ls.toml") and info["adopt_hint"] is None


def test_cli_adopt(tmp_path, capsys):
    sd, repo = _sd(tmp_path), tmp_path / "repo"
    sd.config_toml.write_text('[train]\nlr = 0.001\n', encoding="utf-8")
    assert main(["--root", str(tmp_path / "run"), "--run", "ls", "config", "--adopt", "--repo", str(repo)]) == 0
    assert "写した" in capsys.readouterr().out
    assert repo_config_path("ls", repo).read_text(encoding="utf-8") == '[train]\nlr = 0.001\n'
    assert main(["--root", str(tmp_path / "run"), "--run", "ls", "config", "--adopt", "--repo", str(repo)]) == 1


def test_nothing_anywhere_gives_the_defaults(tmp_path):
    sd, repo = _sd(tmp_path), tmp_path / "repo"
    cfg, info = resolve(sd, None, repo, ref="none", log=lambda m: None)
    assert info["source"] == "DEFAULTS" and info["adopt_hint"] is None and cfg["train"]["batch_size"] == 1024
    assert not sd.config_toml.exists()  # 書き出しはランナーに任せる（これまで通り）


def test_explicit_config_is_untouched(tmp_path):
    sd, repo = _sd(tmp_path), tmp_path / "repo"
    _repo_config(repo, "ls", '[train]\nlr = 0.002\n')
    one = tmp_path / "one.toml"
    one.write_text('[train]\nlr = 0.009\n', encoding="utf-8")
    cfg, info = resolve(sd, one, repo, ref="none", log=lambda m: None)
    assert cfg["train"]["lr"] == 0.009 and info["source"] == str(one)
    assert not sd.config_toml.exists()  # 1 回きりの起動なので写しを作り直さない


def test_dry_run_writes_nothing(tmp_path):
    sd, repo = _sd(tmp_path), tmp_path / "repo"
    sd.config_toml.write_text('[train]\nlr = 0.001\n', encoding="utf-8")
    _repo_config(repo, "ls", '[train]\nlr = 0.002\n')
    cfg, info = resolve(sd, None, repo, ref="none", apply=False, log=lambda m: None)
    assert cfg["train"]["lr"] == 0.002 and any("train.lr" in c for c in info["changed"])
    assert sd.config_toml.read_text(encoding="utf-8") == '[train]\nlr = 0.001\n'  # まだ効いていない
    assert info["backup"] is None
    # 写しもしない
    sd2 = _sd(tmp_path, "lx")
    sd2.config_toml.write_text('[train]\nlr = 0.003\n', encoding="utf-8")
    _, i2 = resolve(sd2, None, repo, ref="none", apply=False, log=lambda m: None)
    assert i2["adopt_hint"] == str(repo_config_path("lx", repo)) and not repo_config_path("lx", repo).exists()


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


@pytest.mark.skipif(subprocess.run(["git", "--version"], capture_output=True).returncode != 0, reason="git が無い")
def test_reads_the_config_from_a_git_ref_without_touching_the_work_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _repo_config(repo, "ls", '[train]\nlr = 0.002\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "c")
    # コミットしたあとで作業ツリーを変えても、ref の中身が読める
    _repo_config(repo, "ls", '[train]\nlr = 0.777\n')
    assert read_ref_config(repo, "main", "ls", log=lambda m: None) == '[train]\nlr = 0.002\n'
    assert read_ref_config(repo, "main", "lx", log=lambda m: None) is None   # そのファイルが無い
    assert read_ref_config(repo, "origin/main", "ls", log=lambda m: None) is None  # リモートが無い

    sd = _sd(tmp_path)
    cfg, info = resolve(sd, None, repo, ref="main", log=lambda m: None)
    assert cfg["train"]["lr"] == 0.002 and info["ref"] == "main"
    assert repo_config_path("ls", repo).read_text(encoding="utf-8") == '[train]\nlr = 0.777\n'  # 作業ツリーは触らない
    # ref が読めなければ作業ツリーのファイルに戻る
    cfg2, info2 = resolve(sd, None, repo, ref="none", log=lambda m: None)
    assert cfg2["train"]["lr"] == 0.777 and info2["ref"] is None


def test_cli_config_shows_the_source_and_what_is_pending(tmp_path, capsys):
    sd, repo = _sd(tmp_path), tmp_path / "repo"
    sd.config_toml.write_text('[train]\nlr = 0.001\n', encoding="utf-8")
    _repo_config(repo, "ls", '[train]\nlr = 0.002\n')
    assert main(["--root", str(tmp_path / "run"), "--run", "ls", "config", "--ref", "none", "--repo", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "設定の正" in out and "反映待ちの違い 1 件" in out and "train.lr: 0.001 → 0.002" in out
    assert sd.config_toml.read_text(encoding="utf-8") == '[train]\nlr = 0.001\n'  # 見るだけ
    assert main(["--root", str(tmp_path / "run"), "--run", "ls", "config", "--ref", "none", "--repo", str(repo), "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["config"]["train"]["lr"] == 0.002 and len(r["info"]["changed"]) == 1


def test_progress_publishes_the_effective_config(tmp_path):
    from libra_league.progress import files_for

    sd = _sd(tmp_path)
    sd.write_state({"step": 1, "games_total": 2})
    sd.config_toml.write_text(f'[auto]\nreference_ckpts = ["{tmp_path}/libra-run/ls/a.pt"]\n', encoding="utf-8")
    files, _ = files_for(sd, None, 10, "progress")
    assert set(files) == {"progress/ls.json", "progress/ls.md", "progress/ls-config.toml"}
    assert "reference_ckpts" in files["progress/ls-config.toml"]


def test_round_trip_keeps_every_value(tmp_path):
    """作り直した <run>/config.toml を読み直すと、元の設定と同じになる（黙って値が変わらない）。"""
    from libra_league.config import load_config

    sd, repo = _sd(tmp_path), tmp_path / "repo"
    real = Path(__file__).resolve().parents[2] / "docs" / "ls2-config.toml"
    _repo_config(repo, "ls", real.read_text(encoding="utf-8"))
    cfg, _ = resolve(sd, None, repo, ref="none", log=lambda m: None)
    assert load_config(sd.config_toml) == cfg
    # 逆斜線や引用符が入っても壊れない
    _repo_config(repo, "lx", '[exploiter]\nmain_ckpt = "C:\\\\libra\\\\a.pt"\n[auto]\nmatch_go = "movetime 1000"\n')
    sdx = _sd(tmp_path, "lx")
    cfgx, _ = resolve(sdx, None, repo, ref="none", log=lambda m: None)
    assert cfgx["exploiter"]["main_ckpt"] == "C:\\libra\\a.pt"
    assert load_config(sdx.config_toml) == cfgx



def test_unknown_config_keys_warn_instead_of_being_ignored_silently(tmp_path):
    """今のプログラムが知らない鍵は警告に出す。

    設定は `origin/main` から読むのにプログラムは手元の作業ツリーなので、`git pull` を忘れると
    新しい鍵が黙って無視される。2026-09-19 に `match_go_opp` がこれで効かず、相手まで 1 手 400 回に
    なった 40 局を「勝率 100%」として記録してしまった（docs/measurements.md 同日）。
    """
    sd, repo = _sd(tmp_path), tmp_path / "repo"
    _repo_config(repo, "ls", '[auto]\nmatch_games = 40\nmatch_go_opp_typo = "x"\n[nosuch]\nk = 1\n')
    (sd.root / "config.local.toml").write_text('[train]\nnope = 1\n', encoding="utf-8")
    logs: list[str] = []
    cfg, info = resolve(sd, None, repo, ref="none", log=logs.append)
    assert info["unknown"] == ["auto.match_go_opp_typo", "nosuch", "train.nope"]
    assert any("WARNING" in m and "git pull" in m for m in logs)
    assert cfg["auto"]["match_games"] == 40   # 知っている鍵はそのまま効く


def test_no_warning_when_every_key_is_known(tmp_path):
    sd, repo = _sd(tmp_path), tmp_path / "repo"
    _repo_config(repo, "ls", '[auto]\nmatch_go = "nodes 400"\nmatch_go_opp = "movetime 1000"\n')
    logs: list[str] = []
    _cfg, info = resolve(sd, None, repo, ref="none", log=logs.append)
    assert info["unknown"] == [] and not any("WARNING" in m for m in logs)
