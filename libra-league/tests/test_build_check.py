# SPDX-License-Identifier: Apache-2.0
"""コンパイル済みの部品より新しいソースがあれば知らせる（2026-09-21 の投了の見落とし）。"""
import os

from libra_league.build_check import stale, stale_modules, warn_lines


def _touch(p, mtime):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")
    os.utime(p, (mtime, mtime))


def test_source_newer_than_module_is_stale(tmp_path):
    pkg, src = tmp_path / "pkg", tmp_path / "src"
    _touch(pkg / "_search.so", 1000.0)
    _touch(src / "selfplay.cpp", 900.0)
    assert stale(pkg, [src]) is None  # 作り直したあと
    _touch(src / "selfplay.cpp", 2000.0)  # git pull がソースを新しくした
    msg = stale(pkg, [src])
    assert msg is not None and "selfplay.cpp" in msg and "_search.so" in msg


def test_same_second_and_missing_pieces_are_quiet(tmp_path):
    pkg, src = tmp_path / "pkg", tmp_path / "src"
    _touch(pkg / "_search.so", 1000.0)
    _touch(src / "selfplay.cpp", 1000.5)
    assert stale(pkg, [src]) is None  # 同じ秒に並んだだけ（SLACK_S）
    assert stale(tmp_path / "none", [src]) is None  # 部品がまだ無い
    assert stale(pkg, [tmp_path / "none"]) is None  # ソースの場所が無い
    _touch(src / "README.md", 9000.0)
    assert stale(pkg, [src]) is None  # C++ 以外は見ない


def test_ignores_unrelated_suffixes_and_reports_the_newest(tmp_path):
    pkg, src = tmp_path / "pkg", tmp_path / "src"
    _touch(pkg / "_search.so", 1000.0)
    _touch(src / "old.cpp", 500.0)
    _touch(src / "sub" / "new.h", 3000.0)
    assert "new.h" in (stale(pkg, [src]) or "")


def test_real_tree_does_not_raise():
    """本物の置き場所で呼んでも落ちない（部品が無い環境でも空を返す）。"""
    assert isinstance(stale_modules(), list)
    assert isinstance(warn_lines(), list)
