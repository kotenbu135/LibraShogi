# SPDX-License-Identifier: Apache-2.0
"""本体と過去の搾取者の対局（libra_league.league）: スナップショットの一覧と間引き、PFSP、対局の印、
搾取者が作り直しのたびに自分を保存すること、Runner のスモーク。"""
import json

import numpy as np
import torch

from libra_league.league import add_result, list_pool, main_winrate, pfsp_pick, pool_name, prune_pool, tag_league_game
from libra_league.selfplay import mask_opponent_moves
from libra_league.workers import load_weights, publish_weights
from libra_net.model import LibraNet, NetConfig

NET = {"d_model": 32, "n_layers": 1, "n_heads": 4, "d_ff": 64, "dropout": 0.0}


def test_list_and_prune_pool(tmp_path):
    for s in (10, 300, 20, 7):
        (tmp_path / pool_name(s)).write_bytes(b"x")
    (tmp_path / "lx-000000500.pt.tmp").write_bytes(b"x")  # 書きかけは数えない
    assert [s for s, _ in list_pool(tmp_path, 2)] == [300, 20]
    prune_pool(tmp_path, 3)
    assert [s for s, _ in list_pool(tmp_path, 10)] == [300, 20, 10]
    assert list_pool(tmp_path / "none", 3) == []


def test_pfsp_prefers_opponents_the_main_cannot_beat():
    stats: dict = {}
    for r, n in ((1, 90), (-1, 10)):
        for _ in range(n):
            add_result(stats, 1, r)
    for r, n in ((1, 20), (-1, 80)):
        for _ in range(n):
            add_result(stats, 2, r)
    assert abs(main_winrate(stats["1"]) - 91 / 102) < 1e-9 and main_winrate(None) == 0.5
    rng = np.random.default_rng(0)
    picks = [pfsp_pick([1, 2, 3], stats, rng) for _ in range(2000)]
    c = {s: picks.count(s) for s in (1, 2, 3)}
    assert c[2] > c[3] > c[1] > 0  # 勝てない相手 2 > 未対局の 3 > よく勝つ相手 1


def test_tag_league_game_keeps_main_moves_and_replaces_v41():
    g = {"full": np.ones(5, np.uint8), "result": -1, "slot": 1, "v41": 0.3}
    mask_opponent_moves(g, exploiter_is_sente=False)  # 奇数枠: 自分（本体）は後手
    tag_league_game(g, 42)
    assert list(g["full"]) == [0, 1, 0, 1, 0]
    assert g["league_main_side"] == "gote" and g["league_result"] == 1 and g["league_opponent"] == 42 and g["v41"] == -1.0
    assert "exploiter_side" not in g and "exploiter_result" not in g


class _StubLoop:
    def __init__(self):
        self.opponent = None

    def set_opponent(self, m, opponent_prior=True):
        self.opponent = m


def test_exploiter_saves_itself_to_the_pool(tmp_path):
    """搾取者は起動時にプールが空なら自分を保存し、凍結相手を作り直すたびに作り直す前の自分を保存する（新しい pool_keep 個を残す）。"""
    from libra_league.config import load_config
    from libra_league.runner import Runner
    from libra_league.state import StateDir

    for name, step in (("main.pt", 5), ("source.pt", 100)):
        torch.save({"model": LibraNet(NetConfig.from_dict(NET)).state_dict(), "config": {"net": NET}, "step": step}, tmp_path / name)
    cfg = load_config(None)
    cfg["run_id"] = "lx"
    cfg["net"] = NET
    cfg["exploiter"].update({"main_ckpt": str(tmp_path / "main.pt"), "main_source": str(tmp_path / "source.pt"),
                             "pool_out": str(tmp_path / "pool"), "pool_keep": 2})
    sd = StateDir(tmp_path / "lx")
    sd.create()
    r = Runner(sd, cfg, device=torch.device("cpu"))
    r.loop = _StubLoop()
    r.load_opponent()
    r.trainer.step_count = 3
    r.ensure_pool_snapshot()
    r.ensure_pool_snapshot()  # 既にあれば足さない
    assert [s for s, _ in list_pool(tmp_path / "pool", 10)] == [3]
    for step in (11, 12):
        r.trainer.step_count = step
        r.refresh_main()
    snaps = list_pool(tmp_path / "pool", 10)
    assert [s for s, _ in snaps] == [12, 11]
    m, step, run_id = load_weights(snaps[0][1])
    assert step == 12 and run_id == "lx" and m.cfg == NetConfig.from_dict(NET)


def test_runner_league_smoke(tmp_path):
    """小さなネットで本体の Runner を回し、自己対局とは別に過去の搾取者との対局が進み、印の付いた局がリプレイと棋譜に入ること。"""
    import threading
    import time

    from libra_league.config import load_config
    from libra_league.runner import Runner
    from libra_league.state import StateDir, read_json

    pool = tmp_path / "pool"
    pool.mkdir()
    publish_weights(pool / pool_name(7), LibraNet(NetConfig.from_dict(NET)), 7, NET, "lx")
    cfg = load_config(None)
    cfg["net"] = NET
    cfg["search"].update({"full_sims": 4, "fast_sims": 4, "proof_nodes": 0, "mate_nodes_root": 0, "max_moves_per_game": 20})
    cfg["selfplay"].update({"n_games": 4, "threads": 2, "infer_dtype": "float32"})
    cfg["league"].update({"enabled": True, "pool": str(pool), "n_games": 4, "threads": 2, "switch_games": 2})
    cfg["train"].update({"batch_size": 8, "min_window_games": 4, "train_every_games": 4, "window_games": 100})
    cfg["run"].update({"status_seconds": 0.5, "checkpoint_minutes": 100, "chunk_games": 4, "export_onnx": False})
    sd = StateDir(tmp_path / "ls")
    sd.create()
    r = Runner(sd, cfg, device=torch.device("cpu"))
    th = threading.Thread(target=r.run, daemon=True)
    th.start()
    t0 = time.time()
    while time.time() - t0 < 120:
        st = read_json(sd.status_json, {})
        if st.get("league", {}).get("games", 0) >= 4 and st.get("step", 0) >= 1:
            break
        time.sleep(0.5)
    sd.set_flag("STOP")
    th.join(timeout=60)
    st = read_json(sd.status_json)
    lg = st["league"]
    assert lg["opponent_step"] == 7 and lg["games"] >= 4 and lg["pool"][0]["step"] == 7 and lg["pool"][0]["games"] >= 1
    assert "exploiter" not in st
    rows = [json.loads(x) for p in sd.games.glob("games_*.jsonl") for x in p.read_text(encoding="utf-8").splitlines()]
    assert any(x.get("league", {}).get("opponent_step") == 7 for x in rows) and any("league" not in x for x in rows)
    assert any("league: opponent lx step 7" in line for line in (sd.root / "log.txt").read_text().splitlines())
