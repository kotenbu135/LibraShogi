# SPDX-License-Identifier: Apache-2.0
"""搾取者が見つけた布石（openings.json）: 搾取者が勝った対局の玉 2 手＋布石の先頭 K 手。本体の自己対局の一部をここから始める。"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import librashogi as ls


def extract_openings(games, k_moves: int = 12, min_plies: int = 0) -> list[list[str]]:
    """搾取者が勝った記録から手順（USI）を取り出す。重複は除く。"""
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for g in games:
        if int(g.get("exploiter_result", 0)) <= 0:
            continue
        if int(g.get("plies", 0)) < min_plies:
            continue
        if int(g["kw"]) % 9 == 3:  # 後手玉が四段目: 桂打ちで先手の裁定勝ちが決まる自明なペア（libra-scale の剪定）
            continue
        moves = [f"K*{ls.sq_to_usi(int(g['kb']))}", f"K*{ls.sq_to_usi(int(g['kw']))}"]
        moves += [ls.move_to_usi(int(m)) for m in list(g["moves"])[:k_moves]]
        key = tuple(moves)
        if key in seen:
            continue
        seen.add(key)
        out.append(moves)
    return out


def write_openings(path: Path, lines: list[list[str]], source: str) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source, "license": "CC0-1.0",
                               "openings": lines}, ensure_ascii=False))
    tmp.replace(path)


def load_openings(path: Path) -> list[list[int]]:
    """openings.json → 手コードの列（libra-search の SearchConfig.openings 用）。非合法な手順は除く。"""
    d = json.loads(Path(path).read_text())
    out: list[list[int]] = []
    for line in d.get("openings", []):
        pos = ls.Position()
        codes: list[int] = []
        ok = True
        for m in line:
            if not pos.is_legal(m) or pos.is_over():
                ok = False
                break
            codes.append(int(ls.move_from_usi(m)))
            pos.do_move(m)
        if ok and len(codes) >= 2:
            out.append(codes)
    return out


def openings_from_replay(replay_dir: Path, last_chunks: int, k_moves: int, min_chunk: int = 0) -> list[list[str]]:
    """新しい側から last_chunks 個のチャンクを見る。min_chunk 未満のチャンク（凍結相手を作り直す前の対局）は使わない。"""
    import pickle

    def idx(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    chunks = [p for p in sorted(replay_dir.glob("chunk_*.pkl")) if idx(p) >= min_chunk][-last_chunks:]
    games: deque = deque()
    for c in chunks:
        with open(c, "rb") as f:
            games.extend(pickle.load(f))
    return extract_openings(games, k_moves)
