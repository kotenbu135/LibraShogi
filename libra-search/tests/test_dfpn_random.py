# SPDX-License-Identifier: Apache-2.0
"""df-pn の健全性: 自己対局の局面で「詰みが証明されたら、証明手を指した後、受け方のどの手にも再び証明がある」ことを深さ優先で確かめる。"""
import random

import numpy as np

import librasearch
import librashogi as ls


def verify_mate(pos: ls.Position, depth: int = 24) -> bool:
    """攻め方の手番。証明手を指し、受け方の全応手に対して再帰的に証明が続くか（詰みまで）。"""
    res, best, _ = librasearch.solve(pos, "mate", 20000)
    assert res == "proven", res
    assert pos.is_legal(best), best
    pos.do_move(best)
    try:
        assert pos.in_check(), "証明手は王手"
        replies = pos.legal_moves()
        if not replies:
            assert pos.outcome()[1] == "no_legal_move"
            return True
        assert depth > 0, "証明の深さが想定を超えた"
        for r in replies:
            pos.do_move(r)
            try:
                assert verify_mate(pos, depth - 1)
            finally:
                pos.undo()
        return True
    finally:
        pos.undo()


def test_mate_proofs_on_random_games():
    rng = random.Random(3)
    proven = 0
    checked = 0
    for g in range(60):
        p = ls.Position()
        while not p.is_over():
            moves = p.legal_moves()
            p.do_move(rng.choice(moves))
            if p.phase == "normal" and p.normal_ply > 10 and not p.is_over():
                res, best, nodes = librasearch.solve(p, "mate", 3000)
                checked += 1
                if res == "proven":
                    proven += 1
                    assert verify_mate(p)
    print(f"checked {checked} positions, proven {proven}")
    assert proven >= 5


def test_fuseki_proofs_consistent_with_rules():
    """39 手目直後の局面: 証明（先手の裁定）は ruling41_pending と一致し、後手の 40 手目全部に対し裁定が成立する。"""
    rng = random.Random(11)
    agree = 0
    for g in range(200):
        p = ls.Position()
        while p.ply < 39:
            moves = p.legal_moves()
            # 先手は後手玉に当てる手を優先
            if p.turn == "sente" and p.ply >= 2 and rng.random() < 0.7:
                att = []
                for m in moves:
                    p.do_move(m)
                    if p.king_attacked("gote"):
                        att.append(m)
                    p.undo()
                if att:
                    moves = att
            p.do_move(rng.choice(moves))
        pending = p.ruling41_pending()
        res, best, _ = librasearch.solve(p, "ruling41", 5000)
        assert (res == "proven") == pending, (res, pending, p.sfen())
        if pending:
            for m in p.legal_moves():
                p.do_move(m)
                assert p.outcome() == ("sente", "ruling41")
                p.undo()
            agree += 1
    assert agree >= 5
