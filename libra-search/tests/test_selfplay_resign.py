# SPDX-License-Identifier: Apache-2.0
"""自己対局の投了（AlphaGo Zero の形）。既定では無効で、無効なら棋譜は入れる前と同じであること。

評価は特徴量に依らない固定のネットにする。手番側から見て「必ず負け」と答えるネットでも、
木の中で手番が交互に入れ替わるので根の値は −1 にはならず −0.2 のあたりに落ち着く。
そこで投了のしきい値は 0.15 にして、「負けの側が続けて下回ったら投了する」ところを見る。
"""
import numpy as np
import pytest

import librasearch
import librashogi as ls

THR = 0.15  # 上の説明のとおり、この疑似ネットの根の値は −0.2 のあたり
BASE = {"full_sims": 8, "fast_sims": 4, "full_prob": 0.25, "gumbel_m_full": 4, "gumbel_m_fast": 2,
        "max_ply": 120, "proof_nodes": 0, "mate_nodes_root": 0, "resign_min_ply": 40}
N_GAMES, ROUNDS = 24, 1200


def play(cfg, rounds=ROUNDS, n_games=N_GAMES, seed=11, losing=True):
    """手番側が必ず負けだと答えるネットで打たせる（losing=False なら互角）。"""
    sp = librasearch.SelfPlay(cfg, n_games, seed=seed, threads=2)
    sq = np.zeros((n_games, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((n_games, ls.GLOB_FEATS), np.float32)
    logits = np.zeros((n_games, ls.POLICY_SIZE), np.float32)
    wdl = np.tile(np.array([0.0, 0.0, 1.0] if losing else [0.3, 0.4, 0.3], np.float32), (n_games, 1))
    done = []
    for _ in range(rounds):
        sp.collect(sq, glob)
        sp.apply(logits, wdl)
        done += sp.take_finished()
    return done, sp.stats()


def canon(games):
    return [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in g.items()} for g in games]


def replay(g):
    p = ls.Position()
    p.set_max_ply(BASE["max_ply"], True)
    p.do_move(f"K*{ls.sq_to_usi(g['kb'])}")
    p.do_move(f"K*{ls.sq_to_usi(g['kw'])}")
    for m in g["moves"]:
        p.do_move_code(int(m))
    return p


def test_off_by_default_leaves_the_record_unchanged():
    """投了の項目を書かない場合と 0 を書いた場合で棋譜が完全に一致する（乱数を引いていない）。"""
    a, sa = play(dict(BASE))
    b, sb = play({**BASE, "resign_threshold": 0.0, "resign_runs": 3, "resign_disable_prob": 0.5})
    assert canon(a) == canon(b)
    assert sa["resign"] == sb["resign"] == 0
    assert all(g["reason"] != "resign" for g in a)


def test_it_resigns_and_the_side_that_resigned_loses():
    done, st = play({**BASE, "resign_threshold": THR, "resign_runs": 1, "resign_disable_prob": 0.0})
    res = [g for g in done if g["reason"] == "resign"]
    assert res and st["resign"] == len(res)
    for g in res:
        p = replay(g)
        assert not p.is_over()            # 投了しなければまだ続いていた局面
        assert p.ply >= BASE["resign_min_ply"]
        # 投了したのは最後の手を指した側＝いま手番でない側。その側が負け
        loser = 1 if p.turn == "gote" else -1
        assert g["result"] == -loser


def test_it_does_not_resign_before_resign_min_ply():
    done, _ = play({**BASE, "resign_threshold": THR, "resign_runs": 1, "resign_disable_prob": 0.0,
                    "resign_min_ply": 60})
    res = [g for g in done if g["reason"] == "resign"]
    assert res
    assert min(g["plies"] for g in res) >= 60


def test_the_sample_that_never_resigns():
    """resign_disable_prob = 1 ならしきい値を割っても投了しない（誤投了を測り続けるための見本）。"""
    done, st = play({**BASE, "resign_threshold": THR, "resign_runs": 1, "resign_disable_prob": 1.0})
    assert st["resign"] == 0
    assert all(g["reason"] != "resign" for g in done)


def test_more_runs_means_later_resignation():
    """連続の手数を増やすと投了が遅くなる（＝対局が長くなる）。"""
    a, _ = play({**BASE, "resign_threshold": THR, "resign_runs": 1, "resign_disable_prob": 0.0})
    b, _ = play({**BASE, "resign_threshold": THR, "resign_runs": 3, "resign_disable_prob": 0.0})
    ra = [g["plies"] for g in a if g["reason"] == "resign"]
    rb = [g["plies"] for g in b if g["reason"] == "resign"]
    assert ra and rb
    assert np.mean(rb) > np.mean(ra)


def test_it_shortens_the_games():
    off, _ = play(dict(BASE))
    on, _ = play({**BASE, "resign_threshold": THR, "resign_runs": 1, "resign_disable_prob": 0.0})
    assert np.mean([g["plies"] for g in on]) < np.mean([g["plies"] for g in off])


def test_it_is_deterministic():
    cfg = {**BASE, "resign_threshold": THR, "resign_runs": 2, "resign_disable_prob": 0.1}
    assert canon(play(cfg)[0]) == canon(play(cfg)[0])


@pytest.mark.parametrize("losing", [True, False])
@pytest.mark.parametrize("thr", [THR, 0.9])
def test_it_only_resigns_after_a_run_below_the_threshold(thr, losing):
    """投了した局では、投了した側の直前の連続 K 手がすべてしきい値を下回っている。"""
    done, _ = play({**BASE, "resign_threshold": thr, "resign_runs": 2, "resign_disable_prob": 0.0}, losing=losing)
    for g in done:
        if g["reason"] != "resign":
            continue
        q = g["root_q"]
        assert len(q) >= 3
        assert q[-1] <= -thr and q[-3] <= -thr  # 投了した側の手は 1 つおき


def test_an_even_evaluation_hardly_ever_resigns():
    """負けだと答えるネットではほとんどの局が投了で終わるが、互角だと答えるネットではまず起きない
    （起きるのは木の中で本当に負けが見えた局面だけ）。"""
    cfg = {**BASE, "resign_threshold": THR, "resign_runs": 1, "resign_disable_prob": 0.0}
    losing, _ = play(cfg)
    even, _ = play(cfg, losing=False)
    f_losing = np.mean([g["reason"] == "resign" for g in losing])
    f_even = np.mean([g["reason"] == "resign" for g in even])
    assert f_losing > 0.5
    assert f_even < 0.1
