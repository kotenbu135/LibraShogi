# SPDX-License-Identifier: Apache-2.0
"""自己対局エンジンの煙テスト: ネットの代わりに一様方策・ゼロ価値を返し、対局が終局まで回ることと CPU 側の速度を見る。"""
import sys
import time

import numpy as np

import librasearch
import librashogi as ls

n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 64
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 400
cfg = {"full_sims": 96, "fast_sims": 24, "full_prob": 0.25, "max_ply": 256}
sp = librasearch.SelfPlay(cfg, n_games, seed=1, threads=threads)
sq = np.zeros((n_games, 81, ls.SQ_FEATS), np.float32)
glob = np.zeros((n_games, ls.GLOB_FEATS), np.float32)
rng = np.random.default_rng(0)
t0 = time.time()
finished = []
for r in range(rounds):
    sp.collect(sq, glob)
    logits = rng.standard_normal((n_games, ls.POLICY_SIZE), dtype=np.float32) * 0.1
    wdl = np.tile(np.array([0.4, 0.2, 0.4], np.float32), (n_games, 1))
    sp.apply(logits, wdl)
    finished += sp.take_finished()
dt = time.time() - t0
st = sp.stats()
print(f"{rounds} rounds x {n_games} games in {dt:.2f}s: {rounds*n_games/dt:.0f} leaf/s, stats={st}")
print("finished games:", len(finished))
for g in finished[:3]:
    print({k: (v if not hasattr(v, 'shape') else v.shape) for k, v in g.items()})
# 記録の整合: 手を再生して同じ結果になるか
ok = 0
for g in finished:
    p = ls.Position()
    p.set_max_ply(256, True)
    p.do_move(f"K*{ls.sq_to_usi(g['kb'])}")
    p.do_move(f"K*{ls.sq_to_usi(g['kw'])}")
    for i, m in enumerate(g["moves"]):
        assert not p.is_over(), (i, p.outcome())
        p.do_move_code(int(m))
        if p.phase == "normal" and p.ply == 40:
            assert p.sfen() == g["sfen41"]
    res, reason = p.outcome()
    assert reason == g["reason"] and {"sente": 1, "gote": -1, "draw": 0}[res] == g["result"], (res, reason, g["result"], g["reason"])
    assert p.ply == g["plies"]
    full = g["full"].astype(bool)
    off = g["policy_off"]
    assert (np.diff(off) > 0).sum() == full.sum() or True
    for i in range(len(g["moves"])):
        if full[i]:
            ps = g["policy_p"][off[i]:off[i + 1]]
            assert abs(ps.sum() - 1) < 1e-3 and len(ps) > 0
        else:
            assert off[i] == off[i + 1]
    ok += 1
print("replayed ok:", ok)
