# SPDX-License-Identifier: Apache-2.0
"""コンパイル済みの部品（pybind11 の `.so`）より新しいソースがあれば知らせる。

設定（`config/<run-id>.toml`）は `git pull` だけで新しくなるが、**C++ の部品は作り直さないと変わらない**。
食い違っても何も起きないので、設定だけが新しくなって動きが古いままになる。

2026-09-21 に実害が出た: 自己対局の投了（`[search] resign_threshold` を 0.0 → 0.9）を入れたのに、
動いていた `_search.so` が投了の入る前（9/19 19:54 より前）のものだったため、**バインディングがその設定を
黙って読み飛ばし**、7 時間ぶん投了なしで打ち続けた。設定の名前は Python 側の既定にあるので
`config.unknown_keys` の警告にも掛からない。終局内訳（裁定 ＋ 詰み ＝ ちょうど 100%）で初めて気付いた。

判定は更新時刻の比較だけ（作り直したかは分からないので、「作り直していない疑い」を出す）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# (パッケージ名, そのパッケージの C++ のソースの場所)
MODULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("librasearch", ("libra-search/src", "libra-search/include", "libra-search/python")),
    ("librashogi", ("libra-sim/src", "libra-sim/include", "libra-sim/python")),
)
SRC_SUFFIXES = (".cpp", ".h", ".hpp", ".cc")
BUILT_SUFFIXES = (".so", ".pyd")
SLACK_S = 1.0  # 同じ秒に並んだだけで疑わない


def _newest(d: Path, suffixes: tuple[str, ...]) -> tuple[float, Path] | None:
    """d の下で suffixes に当たるいちばん新しいファイルの (更新時刻, パス)。"""
    best: tuple[float, Path] | None = None
    for p in d.rglob("*"):
        if p.suffix in suffixes and p.is_file():
            t = p.stat().st_mtime
            if best is None or t > best[0]:
                best = (t, p)
    return best


def stale(pkg_dir: Path, src_dirs: list[Path], slack: float = SLACK_S) -> str | None:
    """部品より新しいソースがあれば、その説明。無ければ None。

    部品がまだ無いときは None（読み込みのところで別に失敗する）。"""
    built = _newest(pkg_dir, BUILT_SUFFIXES)
    if built is None:
        return None
    newest_src: tuple[float, Path] | None = None
    for d in src_dirs:
        if not d.is_dir():
            continue
        s = _newest(d, SRC_SUFFIXES)
        if s is not None and (newest_src is None or s[0] > newest_src[0]):
            newest_src = s
    if newest_src is None or newest_src[0] <= built[0] + slack:
        return None
    return f"{newest_src[1].name} のほうが {built[1].name} より新しい"


def stale_modules(repo: Path | None = None, modules=MODULES) -> list[str]:
    """作り直していない疑いのある部品の説明の一覧（無ければ空）。"""
    if repo is None:
        from .runconfig import repo_root

        repo = repo_root()
    out: list[str] = []
    for name, srcs in modules:
        try:
            spec = importlib.util.find_spec(name)  # 読み込まずに置き場所だけ見る
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.origin:
            continue
        msg = stale(Path(spec.origin).parent, [repo / s for s in srcs])
        if msg is not None:
            out.append(f"{name}: {msg}")
    return out


def warn_lines(repo: Path | None = None) -> list[str]:
    """ログに出す行（無ければ空）。"""
    st = stale_modules(repo)
    if not st:
        return []
    return [f"build: WARNING C++ の部品を作り直していない疑い {len(st)} 件。"
            "設定を変えても動きが変わらないことがある（2026-09-21 の投了がこれ）"] + \
           [f"build:   {s}" for s in st] + \
           ["build:   `cd <repo> && export PATH=$PWD/.venv/bin:$PATH && cmake --build build` で作り直す"]
