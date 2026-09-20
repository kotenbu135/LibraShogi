# SPDX-License-Identifier: Apache-2.0
"""progress ブランチに書き出す指標（libra_league.progress）。

2026-09-20 の点検で、要約に載っていたのは損失と方策の正解率だけ（しかも最後の 1 バッチ）で、
価値・布石の損失、学習率、勾配、自己対局の分布（引き分け・手数・千日手・投了）、学習目標と結果の差、
処理時間の内訳がクラウドから読めなかった。docs/runbook.md §6 が「train_sample が 1% 前後か見る」と
書いている物差しが、書き出しに無かった。"""
import json

from libra_league.auto import append_metrics
from libra_league.progress import format_md, snapshot
from libra_league.state import StateDir


def _status(step, games, *, eng_games, draws, sennichite, resign, plies_sum, sente, gote):
    return {"time": "2026-09-20 00:00:00", "step": step, "generation": 1, "games_total": games,
            "games_per_day_1h": 700000, "active_games": 512, "window_games": games // 2,
            # 最後の 1 バッチ（loss と正解率は使わない。target はここに入る）
            "train": {"loss": 9.9, "policy_acc": 0.1,
                      "target": {"target_minus_z": -0.004, "v41_minus_z": 0.002,
                                 "draw_target": 0.031, "draw_actual": 0.029}},
            "train_avg": {"loss": 2.1, "policy": 1.5, "value": 0.4, "v41": 0.2, "policy_acc": 0.63,
                          "grad_norm": 0.8, "lr": 0.0002, "steps": 300},
            "engine": {"games": eng_games, "moves": eng_games * 10, "sims": eng_games * 100,
                       "sente_wins": sente, "draws": draws, "gote_wins": gote, "ruling41": 0, "no_legal_move": 0,
                       "sennichite": sennichite, "perpetual_check": 0, "max_ply": 0, "resign": resign,
                       "plies_sum": plies_sum, "mate_found": 0, "proof_found": 0},
            "timing": {"window_s": 300.0, "rounds": 100, "train_steps": 300,
                       "sec": {"train_sample": 3.0, "train_step": 60.0, "sp_eval": 150.0}},
            "gpu": {"mem_reserved_mb": 9000}, "mem": {"rss_mb": 18000, "swap_mb": 0}}


def test_snapshot_carries_training_and_selfplay_metrics(tmp_path):
    sd = StateDir(tmp_path / "ls")
    sd.create()
    sd.write_state({"step": 2000, "games_total": 20000})
    append_metrics(sd, _status(1000, 10000, eng_games=10000, draws=30, sennichite=20, resign=0,
                               plies_sum=1_200_000, sente=4985, gote=4985))
    append_metrics(sd, _status(2000, 20000, eng_games=20000, draws=90, sennichite=60, resign=1000,
                               plies_sum=2_300_000, sente=9955, gote=9955))
    s = snapshot(sd)
    row = s["metrics"][-1]

    # 学習: 平均（最後の 1 バッチの 9.9 ではない）と、価値・布石・学習率・勾配
    assert row["loss"] == 2.1 and row["policy_acc"] == 0.63 and row["train_avg"] is True
    assert (row["value"], row["v41"], row["lr"], row["grad_norm"], row["steps"]) == (0.4, 0.2, 0.0002, 0.8, 300)
    # 学習目標と結果の差
    assert row["target_minus_z"] == -0.004 and row["draw_target"] == 0.031 and row["draw_actual"] == 0.029
    # 処理時間の内訳（窓に占める割合）
    assert row["t_train_sample"] == 1.0 and row["t_train_step"] == 20.0 and row["t_sp_eval"] == 50.0
    # 自己対局の分布は行の差から（この 1 万局で引き分け 60、千日手 40、投了 1,000、1 局 110 手）
    assert row["sp_games"] == 10000 and row["sp_draw"] == 0.006 and row["sp_sennichite"] == 0.004
    assert row["sp_resign"] == 0.1 and row["sp_plies"] == 110.0 and row["sp_sente"] == 0.5
    # 最初の行は比べる相手がいないので空
    assert s["metrics"][0].get("sp_draw") is None

    md = format_md(s)
    assert "学習（直近の平均）" in md and "損失 2.100" in md and "価値 0.400" in md
    assert "自己対局の中身" in md and "1 局 110.0 手" in md and "投了 10.00%" in md
    assert "布石の学習目標と結果の差" in md and "処理時間の内訳" in md and "バッチ作り 1.00%" in md


def test_selfplay_rates_skip_a_restart(tmp_path):
    """再起動で数え上げが 0 に戻った行は割合を出さない（負の差を割合にしない）。"""
    sd = StateDir(tmp_path / "ls")
    sd.create()
    sd.write_state({"step": 3000, "games_total": 30000})
    append_metrics(sd, _status(1000, 10000, eng_games=10000, draws=30, sennichite=20, resign=0,
                               plies_sum=1_200_000, sente=4985, gote=4985))
    append_metrics(sd, _status(2000, 20000, eng_games=100, draws=1, sennichite=0, resign=0,
                               plies_sum=11_000, sente=49, gote=50))     # 再起動して数え直し
    append_metrics(sd, _status(3000, 30000, eng_games=5100, draws=20, sennichite=10, resign=0,
                               plies_sum=561_000, sente=2540, gote=2540))
    rows = snapshot(sd)["metrics"]
    assert rows[1].get("sp_draw") is None          # 0 に戻った行
    assert rows[2]["sp_games"] == 5000             # その次の行からは出る


def test_games_per_day_is_absent_right_after_a_start(tmp_path):
    """起動直後で 1 時間の履歴が無いときは局/日を書かない（0 は「止まっている」に読める）。"""
    sd = StateDir(tmp_path / "ls")
    sd.create()
    sd.write_state({"step": 10, "games_total": 100})
    st = _status(10, 100, eng_games=100, draws=0, sennichite=0, resign=0, plies_sum=11000, sente=50, gote=50)
    st["games_per_day_1h"] = None
    sd.status_json.write_text(json.dumps(st), encoding="utf-8")
    md = format_md(snapshot(sd))
    assert "| 局/日（1 時間平均） | - |" in md
