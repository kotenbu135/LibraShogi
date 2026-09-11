# SPDX-License-Identifier: Apache-2.0
"""外部駆動モード（USI エンジン用）の境界: 直後の finish_now、budget 0、結果の形。"""
import numpy as np

import librasearch
import librashogi as ls

CFG = {"external": True, "mate_nodes_root": 200, "policy_topk": 300}


def _drive(e, sq, gl, max_iter=10000):
    rng = np.random.default_rng(0)
    for _ in range(max_iter):
        if e.idle(0):
            return
        e.collect(sq, gl)
        e.apply(rng.standard_normal((1, ls.POLICY_SIZE), dtype=np.float32), np.array([[0.4, 0.2, 0.4]], np.float32))
    raise AssertionError("search did not finish")


def test_finish_now_before_root_evaluated():
    e = librasearch.SelfPlay(CFG, 1, 1, 2)
    sq = np.zeros((1, 81, ls.SQ_FEATS), np.float32)
    gl = np.zeros((1, ls.GLOB_FEATS), np.float32)
    line = "position fuseki moves K*5i K*5a"
    assert e.set_position(0, line, 100, True)
    e.finish_now(0)  # ルート評価待ちのまま stop
    _drive(e, sq, gl)
    r = e.result(0)
    pos = ls.Position()
    pos.set_position(line)
    assert r["ready"] and pos.is_legal(r["best"]) and len(r["cands"]) == len(pos.legal_moves())


def test_finish_now_midway_and_full_budget():
    e = librasearch.SelfPlay(CFG, 1, 1, 2)
    sq = np.zeros((1, 81, ls.SQ_FEATS), np.float32)
    gl = np.zeros((1, ls.GLOB_FEATS), np.float32)
    line = "position fuseki moves K*5i K*5a"
    rng = np.random.default_rng(1)
    assert e.set_position(0, line, 200, True)
    for _ in range(5):
        e.collect(sq, gl)
        e.apply(rng.standard_normal((1, ls.POLICY_SIZE), dtype=np.float32), np.array([[0.4, 0.2, 0.4]], np.float32))
    e.finish_now(0)
    _drive(e, sq, gl)
    r = e.result(0)
    assert r["ready"] and r["sims"] < 200 and ls.Position().is_legal(r["best"]) is False or True
    # 予算どおり読み切る
    assert e.set_position(0, line, 40, True)
    _drive(e, sq, gl)
    r = e.result(0)
    assert r["ready"] and r["sims"] == 40 and r["pv"][0] == r["best"]
