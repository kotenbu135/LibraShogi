# SPDX-License-Identifier: Apache-2.0
"""librashogi の Python バインディングのテスト。ルールの本体は tests/test_rules.cpp と同じ規定（docs/rules.md）。"""
import math
import random

import pytest

import librashogi as ls


def test_initial_tenbin():
    p = ls.Position()
    assert p.mode == "tenbin" and p.phase == "fuseki" and p.turn == "sente" and p.ply == 0
    moves = p.legal_moves()
    assert len(moves) == 36 and all(m.startswith("K*") for m in moves)
    p.do_move("K*5i")
    assert p.turn == "gote" and len(p.legal_moves()) == 36
    p.do_move("K*5a")
    assert len(p.legal_moves()) == 7 * 35
    with pytest.raises(ValueError):
        p.do_move("K*4i")


def test_position_command_skips_choose():
    p = ls.Position()
    p.set_position("position fuseki moves K*5i K*5a choose:gote P*7g P*3c")
    assert p.ply == 4 and p.turn == "sente"
    assert p.sfen() == "4k4/9/6p2/9/9/9/2P6/9/4K4 b RB2G2S2N2L8Prb2g2s2n2l8p 5"
    q = ls.Position()
    q.set_sfen(p.sfen(), "fuseki")
    assert q.key == p.key and q.norm_key == p.norm_key


def test_order_independent_key_and_mirror():
    a, b, c = ls.Position(), ls.Position(), ls.Position()
    a.set_position("position fuseki moves K*5i K*5a P*7g P*3c S*6h G*4b")
    b.set_position("position fuseki moves K*5i K*5a S*6h G*4b P*7g P*3c")
    c.set_position("position fuseki moves K*5i K*5a P*3g P*7c S*4h G*6b")
    assert a.key == b.key
    assert a.key != c.key and a.mirror_key == c.key and a.norm_key == c.norm_key


def test_startpos_perft():
    p = ls.Position()
    p.set_position("position startpos")
    assert p.phase == "normal"
    assert [p.perft(d) for d in (1, 2, 3)] == [30, 900, 25470]


def test_random_game_ends():
    rng = random.Random(1)
    p = ls.Position()
    p.set_max_ply(320, True)
    n = 0
    while not p.is_over():
        moves = p.legal_moves()
        assert moves, p.sfen()
        p.do_move(rng.choice(moves))
        n += 1
        assert n < 400
    result, reason = p.outcome()
    assert result in ("sente", "gote", "draw")
    assert reason in ("no_legal_move", "ruling41", "sennichite", "perpetual_check", "max_ply")
    assert p.ply == n


def test_declaration_and_harness_results():
    p = ls.Position()
    p.set_sfen("BNSGKGSNR/L7L/4P4/9/9/9/9/9/4k4 b 9P 1", "normal")
    assert p.can_declare("sente") and p.declaration_points("sente") == 28 and p.declaration_pieces("sente") == 11
    p.declare("sente")
    assert p.outcome() == ("sente", "declaration")
    q = ls.Position()
    q.set_position("position startpos")
    q.resign("gote")
    assert q.outcome() == ("sente", "resign") and q.legal_moves() == []


def test_cp_winrate_roundtrip():
    # docs/protocol.md: cp = round(435·ln(p/(1−p)) + 34)。GUI の cpToWinrate(cp, 435, 34) の逆関数
    from librashogi.usi import cp_to_winrate, winrate_to_cp

    assert winrate_to_cp(0.5) == 34
    assert winrate_to_cp(0.0) == -5976 and winrate_to_cp(1.0) == 6044  # node で確認した値
    for cp in (-3000, -500, -34, 0, 34, 100, 500, 3000):
        assert abs(cp_to_winrate(cp) - 1 / (1 + math.exp(-(cp - 34) / 435))) < 1e-12
        assert winrate_to_cp(cp_to_winrate(cp)) == cp
    assert winrate_to_cp(0.6120) == 232 and winrate_to_cp(0.25) == -444 and winrate_to_cp(0.75) == 512


def _mirror_sfen(sfen):
    """SFEN の盤面を 1↔9 筋で鏡映する（テスト用）。"""
    board, rest = sfen.split(" ", 1)
    rows = []
    for row in board.split("/"):
        cells, i = [], 0
        while i < len(row):
            if row[i].isdigit():
                cells.extend([""] * int(row[i]))
                i += 1
            elif row[i] == "+":
                cells.append(row[i : i + 2])
                i += 2
            else:
                cells.append(row[i])
                i += 1
        cells.reverse()
        out, empty = "", 0
        for c in cells:
            if c == "":
                empty += 1
            else:
                if empty:
                    out += str(empty)
                    empty = 0
                out += c
        if empty:
            out += str(empty)
        rows.append(out)
    return "/".join(rows) + " " + rest


def test_policy_index_bijection_and_mirror():
    rng = random.Random(7)
    checked = 0
    for g in range(30):
        p = ls.Position("tenbin" if g % 2 else "fuseki")
        while not p.is_over() and p.ply < 140:
            moves = p.legal_moves()
            idx = [p.move_index(m) for m in moves]
            assert len(set(idx)) == len(idx), "policy index must be unique per legal move"
            assert all(0 <= i < ls.POLICY_SIZE for i in idx)
            assert [p.move_from_index(i) for i in idx] == moves
            # 鏡映局面の合法手の添字は mirror_index で写る
            q = ls.Position(p.mode)
            q.set_sfen(_mirror_sfen(p.sfen()), p.phase)
            assert q.turn == p.turn and q.ply == p.ply
            assert sorted(ls.mirror_index(i) for i in idx) == sorted(q.move_index(m) for m in q.legal_moves())
            assert q.key == p.mirror_key and q.norm_key == p.norm_key
            checked += len(moves)
            p.do_move(rng.choice(moves))
    assert checked > 10000


def test_features_shape_and_frame():
    p = ls.Position()
    p.set_position("position fuseki moves K*5i K*5a P*7g")
    sq, glob = p.features()
    assert sq.shape == (81, ls.SQ_FEATS) and glob.shape == (ls.GLOB_FEATS,)
    # 後手番: 180° 回転した座標系。後手玉 5a → 手番側の座標では 5i（添字 80-4=76）に自玉として立つ
    assert p.turn == "gote"
    k_own = 8 - 1  # KING=8 → 添字 7
    assert sq[80 - ls.sq_from_usi("5a"), k_own] == 1.0
    assert sq[80 - ls.sq_from_usi("5i"), 14 + k_own] == 1.0
    assert glob[16] == 1.0 and abs(glob[17] - 3 / 40) < 1e-6 and glob[25] == 0.0
    q = ls.Position()
    q.set_position("position startpos")
    sq, glob = q.features()
    assert glob[16] == 0.0 and glob[25] == 1.0
    assert sq[ls.sq_from_usi("5i"), k_own] == 1.0 and sq[ls.sq_from_usi("5a"), 14 + k_own] == 1.0
