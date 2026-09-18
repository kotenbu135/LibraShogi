# SPDX-License-Identifier: Apache-2.0
"""稼働中のランの設定をリポジトリで管理する（docs/runbook.md §設定の管理、2026-09-18 のユーザーの依頼）。

これまでは `~/libra-run/<run>/config.toml` を手で直すのがユーザーの仕事だった。手で直すのは面倒で、
PC を入れ替えるときに設定が失われる。そこで**設定の正をリポジトリの `config/<run-id>.toml` にし**、
起動のたびにランナーがそこから `~/libra-run/<run>/config.toml` を作り直す。ユーザーの操作は
管理コンソールの停止と起動だけになる。

読む順（後のものが勝つ）:

1. `libra_league.config.DEFAULTS`
2. **リポジトリの `config/<run-id>.toml`**。まず `origin/main` の中身を読み（`git fetch` → `git show`。
   作業ツリーにも HEAD にも触れない）、取れなければ作業ツリーのファイルを読む。手元のチェックアウトが
   古くても、main に入れた設定がそのまま効く（`LIBRA_CONFIG_REF=none` で無効、`--config` でも無効）
3. `~/libra-run/<run>/config.local.toml`（あれば）。その PC だけの上書き（置き場所など）。
   リポジトリには入れない

`--config <path>` を付けたときは、これまで通りそのファイルだけを読む（1 回きりの起動用）。

リポジトリにまだ `config/<run-id>.toml` が無いときは、**今動いている `config.toml` をそこへ写す**
（`adopt`）。中身を推測して書くと学習の設定を黙って変えてしまうので、写して引き継ぐ。ホームの絶対パスは
`~` に直す（公開リポジトリに個人のユーザー名を残さない。置き場所の設定は `expanduser` を通る）。
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .config import DEFAULTS, _merge, dump_toml, load_config
from .state import StateDir

CONFIG_DIR = "config"
DEFAULT_REF = "origin/main"
FETCH_TIMEOUT_S = 60.0
HEADER = ("# このファイルは `libra run` が作り直す。直接編集しても次の起動で上書きされる。\n"
          "# 設定の正はリポジトリの {src}（docs/runbook.md §設定の管理）。\n"
          "# その PC だけの上書きは {local} に置く。\n")


def repo_root() -> Path:
    """このコードの入っているチェックアウト。"""
    return Path(__file__).resolve().parents[2]


def repo_config_path(run_id: str, repo: Path | None = None) -> Path:
    return (repo or repo_root()) / CONFIG_DIR / f"{run_id}.toml"


def local_path(sd: StateDir) -> Path:
    return sd.root / "config.local.toml"


def home_to_tilde(text: str, home: str | None = None) -> str:
    """ホームの絶対パスを `~` に直す（公開リポジトリに個人のユーザー名を残さない）。"""
    home = home if home is not None else str(Path.home())
    return text.replace(home, "~") if home else text


def _git(repo: Path, *args: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout)


def read_ref_config(repo: Path, ref: str, run_id: str, timeout: float = FETCH_TIMEOUT_S, log=print) -> str | None:
    """`<ref>:config/<run-id>.toml` の中身。作業ツリー・HEAD・ローカルのブランチには触れない。

    取れなければ None（git が無い、リモートに繋がらない、そのファイルがまだ無い）。"""
    if "/" in ref:
        remote, branch = ref.split("/", 1)
        try:
            r = _git(repo, "fetch", "--quiet", remote, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}", timeout=timeout)
            if r.returncode != 0:
                log(f"config: git fetch {ref} に失敗（手元のファイルを使う）: {(r.stderr or r.stdout).strip()[:200]}")
        except (OSError, subprocess.SubprocessError) as e:
            log(f"config: git fetch {ref} に失敗（手元のファイルを使う）: {type(e).__name__}")
    try:
        r = _git(repo, "show", f"{ref}:{CONFIG_DIR}/{run_id}.toml", timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 and r.stdout.strip() else None


def changed_keys(old: dict, new: dict, prefix: str = "") -> list[str]:
    """2 つの設定で値が違うキー（`train.replay_ratio` の形）。"""
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        name = f"{prefix}{k}"
        if isinstance(a, dict) or isinstance(b, dict):
            # 片側にしか無い節も、キーごとに並べる（「節が丸ごと」では何が変わったか読めない）
            out += changed_keys(a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {}, f"{name}.")
        elif a != b:
            out.append(f"{name}: {a!r} → {b!r}")
    return out


def resolve(sd: StateDir, explicit: Path | None = None, repo: Path | None = None, ref: str | None = None,
            now: float | None = None, log=print, apply: bool = True) -> tuple[dict, dict]:
    """起動時に効かせる設定を決め、`<run>/config.toml` を作り直す。(設定, 経過) を返す。

    `apply=False` なら何も書かない（`libra config` の下見。「反映待ちの違い」を数えるだけ）。"""
    run_id = sd.root.name
    repo = repo or repo_root()
    ref = ref if ref is not None else os.environ.get("LIBRA_CONFIG_REF", DEFAULT_REF)
    info: dict = {"run": run_id, "source": None, "ref": None, "local": False, "adopt_hint": None,
                  "changed": [], "backup": None}

    if explicit is not None:
        info["source"] = str(explicit)
        return load_config(explicit), info

    live = load_config(sd.config_toml) if sd.config_toml.exists() else None
    text = None if ref in ("", "none", "off") else read_ref_config(repo, ref, run_id, log=log)
    if text is not None:
        info["source"], info["ref"] = f"{CONFIG_DIR}/{run_id}.toml", ref
    else:
        p = repo_config_path(run_id, repo)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            info["source"] = str(p)

    if text is None:
        # リポジトリにまだ無い: これまで通り今の config.toml で動かし、取り込み方をログに出す。
        # ランナーはリポジトリの作業ツリーに書かない（あとで git pull とぶつかるため。取り込みは `libra config --adopt`）
        if live is not None:
            info["source"], info["adopt_hint"] = str(sd.config_toml), str(repo_config_path(run_id, repo))
            log(f"config: リポジトリにまだ {CONFIG_DIR}/{run_id}.toml が無い。今の config.toml で動かす"
                f"（取り込みは `libra --run {run_id} config --adopt`、または progress ブランチの {run_id}-config.toml から）")
            return live, info
        info["source"] = "DEFAULTS"
        return load_config(None), info

    cfg = _merge(DEFAULTS, _parse(text))
    lp = local_path(sd)
    if lp.exists():
        cfg = _merge(cfg, _parse(lp.read_text(encoding="utf-8")))
        info["local"] = True

    info["changed"] = changed_keys(live, cfg) if live is not None else []
    src = info["source"] if not info["ref"] else f"{info['ref']}:{CONFIG_DIR}/{run_id}.toml"
    body = HEADER.format(src=src, local=lp.name) + dump_toml(cfg)
    info["body"] = body
    if apply and (not sd.config_toml.exists() or sd.config_toml.read_text(encoding="utf-8") != body):
        sd.root.mkdir(parents=True, exist_ok=True)
        if sd.config_toml.exists():
            bak = sd.config_toml.with_name(f"config.toml.bak-{time.strftime('%Y%m%d-%H%M%S', time.localtime(now or time.time()))}")
            bak.write_text(sd.config_toml.read_text(encoding="utf-8"), encoding="utf-8")
            info["backup"] = bak.name
        sd.config_toml.write_text(body, encoding="utf-8")
        if info["changed"]:
            log(f"config: {src} で {len(info['changed'])} 件変えた（前の設定は {info['backup']}）")
            for line in info["changed"]:
                log(f"config:   {line}")
    return cfg, info


def _parse(text: str) -> dict:
    import tomllib

    return tomllib.loads(text)


def adopt(sd: StateDir, repo: Path | None = None, force: bool = False) -> tuple[Path, str]:
    """今動いている `<run>/config.toml` をリポジトリの `config/<run-id>.toml` に写す（`libra config --adopt`）。

    中身を推測して書くと学習の設定を黙って変えてしまうので、写して引き継ぐ。ホームの絶対パスは `~` にする。
    既にあるファイルは `force` でないと上書きしない。書いたパスと中身を返す。"""
    run_id = sd.root.name
    p = repo_config_path(run_id, repo)
    if p.exists() and not force:
        raise FileExistsError(f"{p} は既にある（上書きするなら --force）")
    if not sd.config_toml.exists():
        raise FileNotFoundError(f"{sd.config_toml} が無い")
    body = home_to_tilde(sd.config_toml.read_text(encoding="utf-8"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p, body

