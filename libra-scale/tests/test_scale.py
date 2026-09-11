# SPDX-License-Identifier: Apache-2.0
import json
import random

import torch

import librashogi as ls
from libra_net.model import LibraNet, NetConfig
from libra_scale import pairs as P
from libra_scale.table import build_table, measured_winrates, wilson
from libra_scale.verify import apply_verification, verify_pairs


def test_pair_counts_and_mirror():
    assert len(P.all_pairs()) == 1296
    assert len(P.pruned_pairs()) == 972
    u = P.unique_pairs()
    assert len(u) == 492
    for kb, kw in u:
        assert P.canonical(kb, kw) == (kb, kw)
        assert P.canonical(P.mirror_sq(kb), P.mirror_sq(kw)) == (kb, kw)
    assert P.usi(P.sq(4, 8)) == "5i" and P.from_usi("5a") == P.sq(4, 0)


def test_prune_rule_backed_by_sim():
    """後手玉が四段目: 先手の 3 手目の桂打ちが後手玉に当たり、40 手完了まで外せない（乱数の応手で確認）。"""
    rng = random.Random(0)
    for kw in [P.sq(f, 3) for f in (0, 4, 8)]:
        pos = ls.Position()
        pos.do_move(f"K*{P.usi(P.sq(4, 8))}")
        pos.do_move(f"K*{P.usi(kw)}")
        f = kw // 9
        nf = f + 1 if f + 1 <= 8 else f - 1
        knight = f"N*{P.usi(P.sq(nf, 5))}"  # 六段目、隣の筋
        assert pos.is_legal(knight), knight
        pos.do_move(knight)
        assert pos.king_attacked("gote")
        while pos.phase == "fuseki" and not pos.is_over():
            pos.do_move(rng.choice(pos.legal_moves()))
        assert pos.outcome() == ("sente", "ruling41"), pos.outcome()
    # 一〜三段目は即死ではない: 後手の 4 手目で桂の利きを外す手が無いのは四段目だけ
    assert not P.is_immediate_loss(P.sq(4, 8), P.sq(4, 2)) and P.is_immediate_loss(P.sq(4, 8), P.sq(4, 3))


def test_wilson():
    lo, hi = wilson(0.5, 100)
    assert 0.39 < lo < 0.42 and 0.58 < hi < 0.61
    assert wilson(0.5, 0) == (0.0, 1.0)


def test_measured_winrates(tmp_path):
    lines = [
        {"tokens": "K*5i K*5a P*7g", "result": "sente"},
        {"tokens": "K*5i K*5a P*7g", "result": "gote"},
        {"tokens": "K*1i K*9a P*7g", "result": "draw"},
        {"tokens": "K*9i K*1a P*7g", "result": "sente"},  # 鏡映で合算
    ]
    (tmp_path / "g.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    c = measured_winrates(tmp_path)
    assert c[(P.sq(4, 8), P.sq(4, 0))] == [2, 1, 0, 1]
    key = P.canonical(P.sq(0, 8), P.sq(8, 0))
    assert c[key] == [2, 1, 1, 0]


def test_build_and_verify_small(tmp_path):
    torch.manual_seed(0)
    model = LibraNet(NetConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64)).eval()
    pairs = P.unique_pairs()[:6]
    table = build_table(model, {"path": "x", "step": 0}, sims=8, concurrent=4, threads=2, seed=0, device=torch.device("cpu"),
                        games_dir=None, margin=0.05, dtype=torch.float32, pairs=pairs)
    assert table["n_pairs_unique"] == 6 and len(table["pairs"]) == 6
    assert all(0.0 <= e["v_hat"] <= 1.0 for e in table["pairs"])
    assert len(table["balanced"]) >= 1 and all(len(p) == 2 for p in table["balanced"])
    counts = verify_pairs(model, pairs[:2], games_per_pair=2, sims=4, concurrent=8, threads=2, seed=0,
                          device=torch.device("cpu"), dtype=torch.float32, search_overrides={"fast_sims": 4, "proof_nodes": 0, "mate_nodes_root": 0})
    assert all(c[0] >= 2 and c[0] == c[1] + c[2] + c[3] for c in counts.values())
    table = apply_verification(table, counts, sims=4, games_per_pair=2)
    assert "verify" in table and any("verify" in e for e in table["pairs"])
    assert len(table["balanced"]) >= 1
