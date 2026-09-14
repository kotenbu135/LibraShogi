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
        assert st["mate_found"] > 0 and st["proof_found"] > 0  # 本将棋の詰み探索と布石の証明探索の両方の経路を通る


def play_cached(threads, call_proof, clear_every=0, n_games=40, rounds=2000):
    sp = librasearch.SelfPlay({**CFG, "defer_root_proof": True, "eval_cache": True}, n_games, seed=7, threads=threads)
    assert sp.eval_cache_enabled()
    sq = np.zeros((n_games, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((n_games, ls.GLOB_FEATS), np.float32)
    done = []
    for r in range(rounds):
        sp.collect(sq, glob)
        logits, wdl = fake_net(sq, glob)
        if call_proof:
            sp.proof()
        sp.apply(logits, wdl)
        done += sp.take_finished()
        if clear_every and r % clear_every == clear_every - 1:
            sp.clear_eval_cache()  # 重みを替えたときと同じ
    return done, sp.stats()


def test_eval_cache_keeps_records_and_saves_evals():
    """対局ごとのネットの出力のキャッシュ（eval_cache）: 特徴量が同じ葉を評価に出さずに展開しても各枠の棋譜は変わらない
    （疑似ネットは特徴量だけで決まるので、キャッシュの値は評価し直した値と同じ）。重みの入れ替え（clear_eval_cache）を挟んでも同じ。
    当たった分だけ 1 ラウンドで先へ進む。"""
    ref_list, ref_st = play(1)
    ref = by_slot(ref_list)
    for threads, call_proof, clear_every in ((1, True, 0), (8, True, 0), (8, False, 37)):
        got_list, st = play_cached(threads, call_proof, clear_every)
        got = by_slot(got_list)
        compared = 0
        for slot in set(ref) | set(got):
            g, r = got.get(slot, []), ref.get(slot, [])
            k = min(len(g), len(r))
            assert g[:k] == r[:k], (threads, call_proof, clear_every, slot)
            compared += k
        assert compared >= len(ref_list) - len(ref), (threads, compared)
        assert st["cache_hits"] > 0.05 * (st["evals"] + st["cache_hits"]), st
        assert st["moves"] > ref_st["moves"] and len(got_list) >= len(ref_list), (st["moves"], ref_st["moves"])
        assert st["mate_found"] > 0 and st["proof_found"] > 0


def test_gumbel_noise_off_makes_moves_independent_of_seed():
    """gumbel_noise = false（評価・計測用）: 玉配置を 1 つに固定し全読みにすると、seed が違っても同じ手順になる。
    既定（ノイズあり）では seed で手順が変わる。"""
    cfg = {**CFG, "full_prob": 1.0, "king_pairs": [[4 * 9 + 8, 4 * 9 + 0]]}  # 5i・5a

    def moves(seed, noise):
        sp = librasearch.SelfPlay({**cfg, "gumbel_noise": noise}, 4, seed=seed, threads=2)
        sq = np.zeros((4, 81, ls.SQ_FEATS), np.float32)
        glob = np.zeros((4, ls.GLOB_FEATS), np.float32)
        done = []
        while len(done) < 4:
            sp.collect(sq, glob)
            sp.apply(*fake_net(sq, glob))
            done += sp.take_finished()
        return [list(g["moves"]) for g in done[:4]]

    off = moves(1, False)
    assert all(m == off[0] for m in off) and moves(2, False)[0] == off[0]
    on1, on2 = moves(1, True), moves(2, True)
    assert on1 != on2 and any(m != on1[0] for m in on1 + on2)


def test_leaf_turns_report_the_side_to_move_of_each_row():
    """leaf_turns: 未評価の根を出す最初の collect では根の手番と同じ。探索が進むと木の中の相手の手番の葉も出る。"""
    n = 16
    sp = librasearch.SelfPlay(CFG, n, seed=5, threads=2)
    sq = np.zeros((n, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((n, ls.GLOB_FEATS), np.float32)
    sp.collect(sq, glob)
    assert (sp.leaf_turns() == sp.root_turns()).all()
    differ = 0
    for _ in range(200):
        sp.apply(*fake_net(sq, glob))
        sp.collect(sq, glob)
        leaf, root = sp.leaf_turns(), sp.root_turns()
        assert set(np.unique(leaf)) <= {0, 1}
        differ += int((leaf != root).sum())
    assert differ > 0


def test_eval_cache_is_off_by_default_and_can_be_switched_off():
    sp = librasearch.SelfPlay(CFG, 4, seed=1, threads=1)
    assert not sp.eval_cache_enabled()
    sp = librasearch.SelfPlay({**CFG, "eval_cache": True}, 4, seed=1, threads=1)
    sp.set_eval_cache(False)
    assert not sp.eval_cache_enabled()
    sq = np.zeros((4, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((4, ls.GLOB_FEATS), np.float32)
    for _ in range(300):
        sp.collect(sq, glob)
        sp.apply(*fake_net(sq, glob))
    assert sp.stats()["cache_hits"] == 0
