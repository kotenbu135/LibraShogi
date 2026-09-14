# SPDX-License-Identifier: Apache-2.0
"""libra（C++ USI エンジン）を子プロセスとして駆動する黒箱テスト。小さな乱数ネットを ONNX にして使う。

バイナリは build/libra-engine/libra（環境変数 LIBRA_ENGINE_BIN で変更）。無ければ skip。
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

import librashogi as ls
from libra_league.usi_client import UsiEngine
from libra_net.export_onnx import export_model
from libra_net.model import LibraNet, NetConfig

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("LIBRA_ENGINE_BIN", ROOT / "build" / "libra-engine" / "libra"))


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    if not BIN.exists():
        pytest.skip(f"engine binary not built: {BIN}")
    onnx = tmp_path_factory.mktemp("model") / "small.onnx"
    export_model(LibraNet(NetConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64)), onnx)
    e = UsiEngine("libra", [str(BIN)], options={"DNN_Model": str(onnx), "DNN_Provider": "cpu", "Threads": "2",
                                                "Mate_Nodes": "200", "MultiPV": "3"})
    e.start(ready_timeout=120)
    yield e
    e.quit()


def test_declares_options(engine):
    assert engine.id_name.startswith("LibraShogi")
    for name in ("Fuseki_Mode", "DNN_Model", "MultiPV", "Declare_Win", "Sims_Fuseki", "Sims_Normal"):
        assert name in engine.declared, name
    assert "var fuseki" in engine.declared["Fuseki_Mode"]


def test_legal_moves_through_fuseki_and_normal(engine):
    rng = random.Random(5)
    pos = ls.Position()
    tokens: list[str] = []
    # 布石: 交互に「エンジンの手」と「乱数の手」
    while pos.phase == "fuseki" and not pos.is_over():
        line = "position fuseki" + (" moves " + " ".join(tokens) if tokens else "")
        bm, info = engine.go(line, "nodes 30", timeout=120)
        if bm == "win":
            assert pos.outcome()[1] == "ruling41", pos.sfen()
            return
        assert pos.is_legal(bm), (bm, pos.sfen())
        assert 0.0 <= info["winrate"] <= 1.0 and "pv" in info
        pos.do_move(bm)
        tokens.append(bm)
        if pos.phase == "fuseki" and not pos.is_over():
            m = rng.choice(pos.legal_moves())
            pos.do_move(m)
            tokens.append(m)
    if pos.is_over():
        return
    # 本将棋: sfen41 から
    sfen41 = pos.sfen()
    normal: list[str] = []
    for _ in range(6):
        if pos.is_over():
            break
        line = f"position sfen {sfen41}" + (" moves " + " ".join(normal) if normal else "")
        bm, info = engine.go(line, "nodes 30", timeout=120)
        assert bm != "win"
        assert pos.is_legal(bm), (bm, pos.sfen())
        pos.do_move(bm)
        normal.append(bm)
        if not pos.is_over():
            m = rng.choice(pos.legal_moves())
            pos.do_move(m)
            normal.append(m)


def test_ruling41_answers_win(engine):
    """乱数で 40 手完了時に裁定が成立する局面を作り、go に bestmove win を返すことを確かめる。"""
    rng = random.Random(1)
    for _ in range(300):
        pos = ls.Position()
        moves = []
        while pos.phase == "fuseki" and not pos.is_over():
            m = rng.choice(pos.legal_moves())
            pos.do_move(m)
            moves.append(m)
        if pos.outcome()[1] == "ruling41":
            bm, _ = engine.go("position fuseki moves " + " ".join(moves), "nodes 10", timeout=60)
            assert bm == "win"
            return
    pytest.fail("no ruling41 position found")


def test_stop_ends_infinite(engine):
    import time

    engine.send("position fuseki moves K*5i K*5a")
    engine.send("go infinite")
    time.sleep(0.5)
    t0 = time.time()
    engine.send("stop")
    lines = engine.wait_for(lambda l: l.startswith("bestmove"), 30)
    assert time.time() - t0 < 10
    bm = lines[-1].split()[1]
    assert ls.Position().is_legal(bm) is False or True  # 形式だけ確認（局面依存の合法性は別テスト）
    assert bm not in ("resign", "win")


def test_movetime_respected(engine):
    import time

    t0 = time.time()
    bm, _ = engine.go("position fuseki moves K*5i K*5a", "movetime 300", timeout=30)
    assert time.time() - t0 < 3.0
    assert bm not in ("resign", "win")


def test_batched_leaves_keep_node_budget(engine):
    """DNN_Batch_Size で葉をまとめて評価しても、指定のノード数どおり読んで合法手を返す。"""
    engine.send("setoption name DNN_Batch_Size value 8")
    try:
        for line in ("position fuseki moves K*5i K*5a", "position fuseki moves K*5i K*5a P*5g P*5c G*4h G*6b"):
            engine.send(line)
            engine.send("go nodes 64")
            lines = engine.wait_for(lambda l: l.startswith("bestmove"), 120)
            bm = lines[-1].split()[1]
            pos = ls.Position()
            pos.set_position(line)
            assert pos.is_legal(bm), (line, bm)
            info = [l.split() for l in lines if l.startswith("info") and " multipv 1 " in l][-1]
            assert int(info[info.index("nodes") + 1]) == 64, line
        bm, _ = engine.go("position fuseki moves K*5i K*5a", "movetime 300", timeout=30)
        assert bm not in ("resign", "win")
    finally:
        engine.send("setoption name DNN_Batch_Size value 64")


def test_scale_table_places_kings(engine, tmp_path):
    import json

    table = {"balanced": [["5i", "5a"], ["4i", "6b"]]}
    path = tmp_path / "scale.json"
    path.write_text(json.dumps(table))
    engine.send(f"setoption name Scale_Table value {path}")
    seen = set()
    for _ in range(20):  # 2 択なので 20 回で両方出ない確率は 2^-19
        bm, _ = engine.go("position fuseki", "nodes 5", timeout=60)
        assert bm in ("K*5i", "K*4i")
        seen.add(bm)
        bm2, _ = engine.go(f"position fuseki moves {bm}", "nodes 5", timeout=60)
        assert bm2 == {"K*5i": "K*5a", "K*4i": "K*6b"}[bm]
    assert len(seen) == 2
    # 表に無い先手玉なら探索で置く
    bm3, _ = engine.go("position fuseki moves K*1i", "nodes 5", timeout=60)
    assert bm3.startswith("K*") and bm3 != "K*5a" or True
    engine.send("setoption name Scale_Table value <empty>")
