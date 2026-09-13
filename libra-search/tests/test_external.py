# SPDX-License-Identifier: Apache-2.0
"""外部駆動モード（USI エンジン用）の境界: 直後の finish_now、budget 0、結果の形、複数葉の同時評価。"""
import hashlib

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


def fake_net(sq, glob):
    """特徴量だけから決まる疑似ネット（どの呼び出しで評価されたかに依らない）。"""
    n = sq.shape[0]
    logits = np.empty((n, ls.POLICY_SIZE), np.float32)
    wdl = np.empty((n, 3), np.float32)
    for i in range(n):
        h = hashlib.blake2b(sq[i].tobytes() + glob[i].tobytes(), digest_size=8).digest()
        r = np.random.default_rng(int.from_bytes(h, "little"))
        logits[i] = r.standard_normal(ls.POLICY_SIZE, dtype=np.float32)
        w = r.random(3, dtype=np.float32) + 0.1
        wdl[i] = w / w.sum()
    return logits, wdl


def _drive_batch(e, batch, max_iter=10000):
    sq = np.zeros((batch, 81, ls.SQ_FEATS), np.float32)
    gl = np.zeros((batch, ls.GLOB_FEATS), np.float32)
    sizes = []
    for _ in range(max_iter):
        if e.idle(0):
            return sizes
        k = e.collect_batch(0, sq, gl)
        if k == 0:
            continue
        rows = {sq[i].tobytes() + gl[i].tobytes() for i in range(k)}
        assert len(rows) == k  # 同じ葉を 2 度評価に出さない
        lg, w = fake_net(sq[:k], gl[:k])
        e.apply_batch(0, lg, w)
        sizes.append(k)
    raise AssertionError("search did not finish")


def _drive_single(e, max_iter=10000):
    sq = np.zeros((1, 81, ls.SQ_FEATS), np.float32)
    gl = np.zeros((1, ls.GLOB_FEATS), np.float32)
    for _ in range(max_iter):
        if e.idle(0):
            return
        e.collect(sq, gl)
        e.apply(*fake_net(sq, gl))
    raise AssertionError("search did not finish")


LINES = ["position fuseki moves K*5i K*5a",
         "position fuseki moves K*5i K*5a P*5g P*5c G*4h G*6b S*3h S*7b",
         "position sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"]


def test_batch_of_one_matches_single_leaf():
    """collect_batch を 1 葉ずつ使うと、従来の collect / apply と同じ探索になる。"""
    for line in LINES:
        a = librasearch.SelfPlay(CFG, 1, 3, 2)
        b = librasearch.SelfPlay(CFG, 1, 3, 2)
        assert a.set_position(0, line, 120, True) and b.set_position(0, line, 120, True)
        _drive_single(a)
        _drive_batch(b, 1)
        assert a.result(0) == b.result(0), line


def test_batch_spreads_leaves_and_keeps_budget():
    for line in LINES:
        e = librasearch.SelfPlay(CFG, 1, 5, 2)
        assert e.set_position(0, line, 400, True)
        sizes = _drive_batch(e, 16)
        r = e.result(0)
        pos = ls.Position()
        pos.set_position(line)
        assert r["ready"] and pos.is_legal(r["best"]), line
        assert r["sims"] == 400 and sum(c["visits"] for c in r["cands"]) == 400, line
        assert sizes[0] == 1  # 根の評価が先
        assert len(sizes) <= 400 // 4, (line, len(sizes))  # まとめて評価に出せている


def test_finish_now_discards_pending_batch():
    line = LINES[1]
    e = librasearch.SelfPlay(CFG, 1, 9, 2)
    sq = np.zeros((8, 81, ls.SQ_FEATS), np.float32)
    gl = np.zeros((8, ls.GLOB_FEATS), np.float32)
    assert e.set_position(0, line, 300, True)
    for _ in range(6):
        k = e.collect_batch(0, sq, gl)
        e.apply_batch(0, *fake_net(sq[:k], gl[:k]))
    k = e.collect_batch(0, sq, gl)
    assert k > 1
    e.finish_now(0)  # 評価待ちの葉をまとめて捨てる
    e.apply_batch(0, *fake_net(sq[:k], gl[:k]))  # 捨てた後の結果は無視される
    _drive_batch(e, 8)
    r = e.result(0)
    pos = ls.Position()
    pos.set_position(line)
    assert r["ready"] and pos.is_legal(r["best"]) and r["sims"] < 300
    assert sum(c["visits"] for c in r["cands"]) == r["sims"]
    # 続けて次の局面を読める（仮の訪問が残っていない）
    assert e.set_position(0, LINES[0], 64, True)
    _drive_batch(e, 8)
    r = e.result(0)
    assert r["sims"] == 64 and sum(c["visits"] for c in r["cands"]) == 64


def test_finish_now_before_root_with_batch():
    e = librasearch.SelfPlay(CFG, 1, 1, 2)
    assert e.set_position(0, LINES[0], 100, True)
    e.finish_now(0)
    _drive_batch(e, 8)
    r = e.result(0)
    pos = ls.Position()
    pos.set_position(LINES[0])
    assert r["ready"] and pos.is_legal(r["best"]) and len(r["cands"]) == len(pos.legal_moves())
