# SPDX-License-Identifier: Apache-2.0
"""自己対局エンジンの並列の割り当てを変えても棋譜が変わらないこと（対局ごとの乱数と状態は独立）。
評価は特徴量だけから決まる疑似ネットにして、どのラウンドで評価されたかに依らないようにする。"""
import hashlib

import numpy as np
import pytest

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


def fake_net2(sq, glob, net):
    """2 つのネットの疑似ネット: 特徴量とネットの番号で決まる。"""
    return fake_net(np.concatenate([sq, np.full_like(sq[:, :1], net)], axis=1), glob)


def play_two_nets(opponent_prior, how, threads=1, call_proof=True, clear_every=0, n_games=40, rounds=2000):
    """how: "python" は行の選び分けを外（旧来の搾取者のループと同じ式）で行って apply、"apply2" は set_two_nets と apply2。"""
    cache = how == "apply2+cache"
    sp = librasearch.SelfPlay({**CFG, "defer_root_proof": True, "eval_cache": cache}, n_games, seed=7, threads=threads)
    if how != "python":
        sp.set_two_nets(True, opponent_prior)
    sq = np.zeros((n_games, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((n_games, ls.GLOB_FEATS), np.float32)
    swap = (np.arange(n_games) % 2).astype(np.int8)
    done = []
    for r in range(rounds):
        sp.collect(sq, glob)
        (l0, w0), (l1, w1) = fake_net2(sq, glob, 0), fake_net2(sq, glob, 1)
        if call_proof:
            sp.proof()
        if how == "python":
            who = sp.root_turns() ^ swap
            prior_who = who | (sp.leaf_turns() ^ swap) if opponent_prior else who
            sp.apply(np.where(prior_who[:, None] != 0, l1, l0), np.where(who[:, None] != 0, w1, w0))
        else:
            sp.apply2(l0, w0, l1, w1)
        done += sp.take_finished()
        if clear_every and r % clear_every == clear_every - 1:
            sp.clear_eval_cache()
    return done, sp.stats()


@pytest.mark.parametrize("opponent_prior", [True, False])
def test_two_nets_select_rows_like_the_exploiter_loop_and_cache_keeps_records(opponent_prior):
    """set_two_nets + apply2 は、外で根の手番（と葉の手番）から行ごとにネットを選んで apply したのと同じ棋譜になる。
    eval_cache は両方のネットの出力を持つので、手番でネットが替わっても棋譜は変わらず、当たった分だけ先へ進む。"""
    ref_list, ref_st = play_two_nets(opponent_prior, "python")
    got_list, st = play_two_nets(opponent_prior, "apply2")
    assert by_slot(got_list) == by_slot(ref_list) and st == ref_st
    ref = by_slot(ref_list)
    for threads, call_proof, clear_every in ((1, True, 0), (8, False, 37)):
        got_list, st = play_two_nets(opponent_prior, "apply2+cache", threads, call_proof, clear_every)
        got = by_slot(got_list)
        compared = 0
        for slot in set(ref) | set(got):
            g, r = got.get(slot, []), ref.get(slot, [])
            k = min(len(g), len(r))
            assert g[:k] == r[:k], (threads, call_proof, clear_every, slot)
            compared += k
        assert compared >= len(ref_list) - len(ref), (threads, compared)
        assert st["cache_hits"] > 0.03 * (st["evals"] + st["cache_hits"]), st
        assert st["moves"] > ref_st["moves"], (st["moves"], ref_st["moves"])


def test_apply_and_apply2_match_the_two_nets_mode():
    sp = librasearch.SelfPlay(CFG, 4, seed=1, threads=1)
    sq = np.zeros((4, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((4, ls.GLOB_FEATS), np.float32)
    sp.collect(sq, glob)
    l, w = fake_net(sq, glob)
    with pytest.raises(ValueError):
        sp.apply2(l, w, l, w)
    sp.set_two_nets(True)
    assert sp.two_nets()
    with pytest.raises(RuntimeError):
        sp.apply(l, w)
    sp.apply2(l, w, l, w)
    sp.set_two_nets(False)
    assert not sp.two_nets()


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


def test_gumbel_rescale_completed_q():
    """gumbel_rescale（mctx の completed Q: 未訪問は v_mix、根の手の間で [0,1] に正規化、c_scale 0.1）でも対局が進み、方策ターゲットは
    正規化された分布になる。既定（false）は同じ seed で今までどおりの棋譜（この関数の中では rescale の有無で棋譜が変わることだけ見る）。"""
    import numpy as np

    def play(rescale: bool):
        cfg = {**CFG, "full_prob": 1.0, "gumbel_rescale": rescale, "c_scale": 0.1 if rescale else 1.0}
        sp = librasearch.SelfPlay(cfg, 4, seed=5, threads=2)
        sq = np.zeros((4, 81, ls.SQ_FEATS), np.float32)
        glob = np.zeros((4, ls.GLOB_FEATS), np.float32)
        rng = np.random.default_rng(5)
        out = []
        while len(out) < 4:
            sp.collect(sq, glob)
            sp.apply(rng.standard_normal((4, ls.POLICY_SIZE), dtype=np.float32) * 2, np.tile(np.array([0.5, 0.1, 0.4], np.float32), (4, 1)))
            out += sp.take_finished()
        return out[:4]

    a, b = play(True), play(False)
    for g in a:
        po, pp = np.asarray(g["policy_off"]), np.asarray(g["policy_p"], np.float64)
        for j in range(len(g["moves"])):
            p = pp[po[j]:po[j + 1]]
            assert len(p) >= 1 and abs(p.sum() - 1) < 1e-4 and (p >= 0).all()
    assert any(not np.array_equal(x["moves"], y["moves"]) for x, y in zip(a, b))


def _records(sp, n_games, rounds):
    sq = np.zeros((n_games, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((n_games, ls.GLOB_FEATS), np.float32)
    out = []
    for _ in range(rounds):
        sp.collect(sq, glob)
        sp.apply(*fake_net(sq, glob))
        out += sp.take_finished()
    return out


def test_side_config_does_not_change_anything_when_both_sides_are_equal():
    """set_side_config に同じ設定を渡したときは、渡さないときと棋譜が 1 ビット同じ（本番の自己対局の経路を変えない）。"""
    plain = _records(librasearch.SelfPlay(CFG, 16, seed=3, threads=2), 16, 600)
    sp = librasearch.SelfPlay(CFG, 16, seed=3, threads=2)
    sp.set_side_config(CFG)
    assert sp.side_configs()
    same = _records(sp, 16, 600)
    assert len(plain) == len(same) > 0
    for x, y in zip(plain, same):
        assert np.array_equal(x["moves"], y["moves"]) and np.array_equal(x["policy_p"], y["policy_p"])
        assert (x["result"], x["plies"], x["slot"], x["v41"]) == (y["result"], y["plies"], y["slot"], y["v41"])


def test_side_config_rejects_keys_that_would_change_the_game_or_the_record():
    sp = librasearch.SelfPlay(CFG, 2, seed=3, threads=1)
    for bad in ({"max_ply": 320}, {"policy_topk": 8}, {"mate_nodes_root": 0}, {"proof_nodes": 0}, {"draw_value": 0.5},
                {"count_from_41": False}, {"eval_cache": True}):
        with pytest.raises(ValueError):
            sp.set_side_config({**CFG, **bad})
    assert not sp.side_configs()
    sp.set_side_config({**CFG, "gumbel_rescale": True, "c_scale": 0.1, "gumbel_m_full": 4, "cpuct": 2.0, "full_sims": 8})
    assert sp.side_configs()


def test_side_config_applies_to_the_side_that_is_thinking():
    """外部駆動で同じ局面を偶数枠と奇数枠に与える。先手の手番なら偶数枠が A 側・奇数枠が B 側の設定で読み、
    後手の手番では入れ替わる（評価ハーネスの「偶数枠は A が先手」と同じ約束）。"""
    a = {**CFG, "external": True, "gumbel_noise": False, "policy_topk": 300}
    b = {**a, "gumbel_m_full": 2, "gumbel_rescale": True, "c_scale": 0.1}
    sente = "position fuseki moves K*5i K*5a"          # 先手の手番
    gote = "position fuseki moves K*5i K*5a P*5g"      # 後手の手番

    def visits(cfg_a, cfg_b, line, slots=(0, 1)):
        e = librasearch.SelfPlay(cfg_a, 2, 1, 1)
        if cfg_b is not None:
            e.set_side_config(cfg_b)
        sq = np.zeros((2, 81, ls.SQ_FEATS), np.float32)
        gl = np.zeros((2, ls.GLOB_FEATS), np.float32)
        for s in slots:
            assert e.set_position(s, line, 48, True)
        while not all(e.idle(s) for s in slots):
            n = e.collect(sq, gl)
            e.apply(*fake_net(sq[:n], gl[:n]))
        return [sorted((c["move"], c["visits"]) for c in e.result(s)["cands"]) for s in slots]

    only_a = visits(a, None, sente)
    only_b = visits(b, None, sente)
    assert only_a[0] != only_b[0]  # 設定で読みの分かれ方が変わる（変わらなければ以下の判定に意味が無い）
    mixed = visits(a, b, sente)
    assert mixed[0] == only_a[0] and mixed[1] == only_b[0]      # 先手の手番: 偶数枠は A 側、奇数枠は B 側
    mixed_gote = visits(a, b, gote)
    only_a_gote, only_b_gote = visits(a, None, gote), visits(b, None, gote)
    assert mixed_gote[0] == only_b_gote[0] and mixed_gote[1] == only_a_gote[0]  # 後手の手番では入れ替わる


def test_gote_rank4_prob_sets_the_share_of_rank4_gote_kings():
    """gote_rank4_prob: 後手玉が四段目になる確率（残りは一〜三段目から一様）。先手玉の置き方は変えない。
    prune_gote_rank4 が true ならそちらが優先（四段目は出ない）。"""

    def kings(extra, n_games=400):
        cfg = {"full_sims": 2, "fast_sims": 2, "proof_nodes": 0, "mate_nodes_root": 0, "max_moves_per_game": 1, **extra}
        sp = librasearch.SelfPlay(cfg, 64, seed=3, threads=2)
        sq = np.zeros((64, 81, ls.SQ_FEATS), np.float32)
        glob = np.zeros((64, ls.GLOB_FEATS), np.float32)
        rng = np.random.default_rng(0)
        wdl = np.tile(np.array([[0.4, 0.2, 0.4]], np.float32), (64, 1))
        done = []
        while len(done) < n_games:
            sp.collect(sq, glob)
            sp.apply(rng.standard_normal((64, ls.POLICY_SIZE), dtype=np.float32), wdl)
            done += sp.take_finished()
        done = done[:n_games]
        assert all(5 <= int(g["kb"]) % 9 <= 8 for g in done)
        return [int(g["kw"]) % 9 for g in done]

    assert all(r <= 2 for r in kings({"gote_rank4_prob": 0.0}))
    assert all(r == 3 for r in kings({"gote_rank4_prob": 1.0}))
    assert all(r <= 2 for r in kings({"gote_rank4_prob": 1.0, "prune_gote_rank4": True}))
    ranks = kings({"gote_rank4_prob": 0.2}, 1000)
    share = sum(r == 3 for r in ranks) / len(ranks)
    assert 0.15 < share < 0.25, share  # 1,000 局で標準誤差 0.013
    assert {r for r in ranks if r <= 2} == {0, 1, 2}
    # 既定（負）は 36 マスから一様で四段目は 4 分の 1
    share0 = sum(r == 3 for r in kings({}, 1000)) / 1000
    assert 0.2 < share0 < 0.3, share0


def test_retire_stops_a_slot_after_its_current_game_and_leaves_the_others_unchanged():
    """retire した枠は今の対局を最後まで打ってから止まる（次の対局を始めない）。ほかの枠の棋譜は変わらない
    （評価対局で局数ちょうどで打ち切るのに使う。evaluate.play_match）。"""
    n_games, rounds = 8, 1500

    def run(retire_at: int | None):
        sp = librasearch.SelfPlay(CFG, n_games, seed=3, threads=2)
        sq = np.zeros((n_games, 81, ls.SQ_FEATS), np.float32)
        glob = np.zeros((n_games, ls.GLOB_FEATS), np.float32)
        done = []
        for r in range(rounds):
            if r == retire_at:
                for s in (1, 6):
                    sp.retire(s)
            sp.collect(sq, glob)
            sp.apply(*fake_net(sq, glob))
            done += sp.take_finished()
        return sp, done

    _, ref = run(None)
    sp, got = run(5)
    assert sp.retired(1) and sp.retired(6) and not sp.retired(0)
    by_slot = lambda recs, s: [canon(r) for r in recs if r["slot"] == s]  # noqa: E731
    for s in range(n_games):
        if s in (1, 6):
            # 5 回目のラウンドでは 1 局目の途中なので、その 1 局だけ打って止まる
            assert by_slot(got, s) == by_slot(ref, s)[:1] and len(by_slot(ref, s)) > 1
            assert sp.idle(s)
        else:
            assert by_slot(got, s) == by_slot(ref, s)
    with pytest.raises(IndexError):
        sp.retire(n_games)
