# SPDX-License-Identifier: Apache-2.0
"""自己対局エンジンの並列の割り当てを変えても棋譜が変わらないこと（対局ごとの乱数と状態は独立）。
評価は特徴量だけから決まる疑似ネットにして、どのラウンドで評価されたかに依らないようにする。"""
import hashlib

import numpy as np

import librasearch
import librashogi as ls

CFG = {"full_sims": 16, "fast_sims": 8, "full_prob": 0.25, "gumbel_m_full": 8, "gumbel_m_fast": 4,
       "max_ply": 40, "proof_min_ply": 30, "proof_nodes": 300, "mate_nodes_root": 100}


def fake_net(sq, glob):
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


def play(threads, n_games=40, rounds=2000):
    sp = librasearch.SelfPlay(CFG, n_games, seed=7, threads=threads)
    sq = np.zeros((n_games, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((n_games, ls.GLOB_FEATS), np.float32)
    done = []
    for _ in range(rounds):
        sp.collect(sq, glob)
        sp.apply(*fake_net(sq, glob))
        done += sp.take_finished()
    st = sp.stats()
    return done, st


def canon(rec):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rec.items()}


def test_records_do_not_depend_on_threads():
    ref, st1 = play(1)
    assert len(ref) >= 40 and st1["evals"] > 0
    assert st1["mate_found"] + st1["proof_found"] > 0  # 証明探索の経路も通っている
    for threads in (4, 8):
        got, st = play(threads)
        assert [canon(r) for r in got] == [canon(r) for r in ref], threads
        assert st == st1, threads


def test_pool_survives_many_engines():
    # 常駐スレッドの生成と終了を繰り返しても止まらない（最初の並列呼び出しの前に捨てる場合も含む）
    for k in range(20):
        sp = librasearch.SelfPlay(CFG, 16, seed=k, threads=4)
        if k % 2:
            sq = np.zeros((16, 81, ls.SQ_FEATS), np.float32)
            glob = np.zeros((16, ls.GLOB_FEATS), np.float32)
            for _ in range(5):
                sp.collect(sq, glob)
                sp.apply(*fake_net(sq, glob))
        del sp


def play_deferred(threads, call_proof, n_games=40, rounds=2000):
    sp = librasearch.SelfPlay({**CFG, "defer_root_proof": True}, n_games, seed=7, threads=threads)
    sq = np.zeros((n_games, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((n_games, ls.GLOB_FEATS), np.float32)
    done = []
    for _ in range(rounds):
        sp.collect(sq, glob)
        logits, wdl = fake_net(sq, glob)
        if call_proof:
            sp.proof()  # 自己対局のループでは GPU が評価している間に呼ぶ
        sp.apply(logits, wdl)
        done += sp.take_finished()
    return done, sp.stats()


def by_slot(recs):
    out = {}
    for r in recs:
        out.setdefault(r["slot"], []).append(canon(r))
    return out


def test_deferred_root_proof_keeps_records():
    """根の証明探索を proof() の段に回しても各枠の棋譜は変わらない。証明できた手ではその根の評価を捨てるので 1 ラウンド遅れるだけ。
    proof() を呼ばなかったときは apply がその場で解く（同じ棋譜）。"""
    ref = by_slot(play(1)[0])
    n_ref = sum(len(v) for v in ref.values())
    for threads, call_proof in ((1, True), (8, True), (8, False)):
        got_list, st = play_deferred(threads, call_proof)
        got = by_slot(got_list)
        for slot in set(ref) | set(got):
            g, r = got.get(slot, []), ref.get(slot, [])
            k = min(len(g), len(r))
            assert g[:k] == r[:k], (threads, call_proof, slot)
            assert abs(len(g) - len(r)) <= 1, (threads, call_proof, slot)
        assert len(got_list) >= n_ref - len(ref)
        assert st["mate_found"] + st["proof_found"] > 0
