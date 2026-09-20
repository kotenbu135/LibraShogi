# SPDX-License-Identifier: Apache-2.0
"""打ち切り（借りたホストを途中で失うこと）の損の勘定（libra_cloud.interrupts）。

筋書き: 入札で 3 時間借りて 1.25 時間で止められ（A）、残りの 1.8 時間で借り直して打ち切った（B）。
そのあと on-demand で 2 時間ふつうに打ち（C）、最後にまた入札で止められて今度は借り直せなかった（D）。
"""
from libra_cloud import interrupts

RATE = 480_000  # 局/日（1 時間あたり 20,000 局）
H = 3600.0


def _rows() -> list[dict]:
    a = {"name": "A", "rent": "bid", "gpu": "RTX 5070 Ti", "lost": True, "continues": None, "hours": 3.0, "dph": 0.25,
         "t_rent": 0.0, "t_bridge": 600.0, "last_pull": 600.0 + 1.25 * H, "t_end": 600.0 + 1.25 * H + 180,
         "bridge_h": 1.25, "games_per_day": RATE, "net_games": 24_000, "total_usd": 0.35}
    b = {"name": "B", "rent": "bid", "gpu": "RTX 5070 Ti", "lost": False, "continues": "A", "hours": 1.8, "dph": 0.25,
         "t_rent": a["t_end"], "t_bridge": 6000.0, "t_end": 6000.0 + 1.8 * H, "last_pull": 6000.0 + 1.8 * H,
         "bridge_h": 1.8, "games_per_day": RATE, "net_games": 36_000, "total_usd": 0.52}
    c = {"name": "C", "rent": "on-demand", "gpu": "RTX 5080", "lost": False, "continues": None, "hours": 2.0, "dph": 0.30,
         "t_rent": 20_000.0, "t_bridge": 20_360.0, "t_end": 20_360.0 + 2 * H, "last_pull": 20_360.0 + 2 * H,
         "bridge_h": 2.0, "games_per_day": RATE, "net_games": 40_000, "total_usd": 0.60}
    d = {"name": "D", "rent": "bid", "gpu": "RTX 5070 Ti", "lost": True, "continues": None, "hours": 3.0, "dph": 0.25,
         "t_rent": 40_000.0, "t_bridge": 40_600.0, "last_pull": 40_600.0 + 0.5 * H - 120, "t_end": 40_600.0 + 0.5 * H,
         "bridge_h": 0.5, "games_per_day": RATE, "net_games": 10_000, "total_usd": 0.20}
    return [a, b, c, d]


def test_a_lost_session_loses_the_time_until_the_relaunch_starts_playing():
    a, b, c, d = interrupts.annotate(_rows())
    # A: 最後に回収した 5,100 秒から、借り直した B が打ち始める 6,000 秒まで 0.25 時間、局が 1 つも増えない
    assert a["dark_h"] == 0.25 and a["unused_h"] == 0.0 and a["lost_h"] == 0.25
    assert a["lost_games"] == round(0.25 / 24 * RATE) == 5_000
    # A の準備（10 分）は打ち切られなくても払うので余分ではない。B の準備（12 分）は打ち切りのせいで増えたぶん
    assert a["setup_h"] == 0.167 and a["extra_setup_usd"] == 0.0
    assert b["setup_h"] == 0.2 and b["setup_usd"] == 0.05 and b["extra_setup_usd"] == 0.05
    assert b["lost_h"] == 0.0 and b["lost_games"] == 0
    assert c["lost_h"] == 0.0 and c["extra_setup_usd"] == 0.0


def test_a_session_that_could_not_be_relaunched_loses_the_rest_of_the_booking():
    *_, d = interrupts.annotate(_rows())
    # D: 回収が止まっていた 2 分に加えて、予定 3 時間のうち打てなかった 2.5 時間を丸ごと数える
    assert d["dark_h"] == 0.033 and d["unused_h"] == 2.5 and d["lost_h"] == 2.533
    assert d["lost_games"] == round(2.533 / 24 * RATE) == 50_660


def test_summary_shows_how_much_the_interruptions_cost():
    w = interrupts.summary(interrupts.annotate(_rows()))
    assert w["sessions"] == 4 and w["interruptions"] == 2 and w["relaunches"] == 1 and w["not_relaunched"] == 1
    assert w["bridge_h"] == 5.55 and w["h_per_loss"] == 2.77  # 平均 2.8 時間打つと 1 回止められる
    assert w["lost_h"] == 2.78 and w["unused_h"] == 2.5 and w["lost_games"] == 55_660
    # 時間の損は費用とは別。止まっている間は課金されないので、お金には出ないが局/日には効く
    assert w["lost_time_pct"] == 33.4  # 2.78 / (5.55 + 2.78)
    assert w["extra_setup_usd"] == 0.05 and w["net_games"] == 110_000 and w["total_usd"] == 1.67
    # 打ち切りが無ければ、同じ 1.67 ドルから余分な準備代を引いた額で 16.6 万局打てていた
    assert w["usd_per_1m"] == 15.18 and w["usd_per_1m_ideal"] == 9.78 and w["waste_pct"] == 55.2


def test_summary_compares_bids_with_on_demand():
    w = interrupts.summary(interrupts.annotate(_rows()))
    bid, od = w["by_rent"]
    # session.json の "bid" はコンソールの「借り方」と同じ「入札」で出す
    assert bid["name"] == "入札" and bid["sessions"] == 3 and bid["lost"] == 2 and bid["h_per_loss"] == 1.77
    assert bid["dph"] == 0.25 and bid["usd_per_1m"] == 15.29 and bid["usd_per_1m_ideal"] == 8.12
    assert bid["lost_h"] == 2.78 and bid["lost_time_pct"] == 43.9  # 入札だけで見ると経過時間の 4 割は打てていない
    assert od["name"] == "on-demand" and od["sessions"] == 1 and od["lost"] == 0 and od["h_per_loss"] is None
    assert od["lost_h"] == 0.0 and od["lost_time_pct"] == 0.0
    assert od["usd_per_1m"] == 15.0 and od["usd_per_1m_ideal"] == 15.0 and od["waste_pct"] == 0.0
    assert [g["name"] for g in w["by_gpu"]] == ["RTX 5070 Ti", "RTX 5080"]


def test_sessions_without_records_do_not_produce_absurd_numbers():
    """借りられなかった回（時刻の記録が無い）と、次のセッションまでが離れすぎている回で落ちない。"""
    zero = {"name": "Z", "lost": False, "bridge_h": None, "net_games": 0, "total_usd": None}
    # X は打ち切られ、借り直した Y が打ち始めるまでが桁外れ（記録の取りこぼし）。穴は MAX_HOLE_H で頭打ちにする
    x = {"name": "X", "lost": True, "t_rent": 0.0, "t_bridge": 600.0, "last_pull": 1000.0, "t_end": 2000.0,
         "hours": 3.0, "dph": 0.2, "bridge_h": 0.4, "net_games": 100, "total_usd": 0.1}
    y = {"name": "Y", "lost": False, "continues": "X", "t_bridge": 1e6, "t_end": 1e6 + 3600, "bridge_h": 1.0,
         "net_games": 200, "total_usd": 0.2}
    z2, x2, y2 = interrupts.annotate([zero, x, y])
    assert z2["setup_h"] is None and z2["lost_h"] == 0.0 and z2["lost_games"] == 0
    assert x2["setup_h"] == 0.167 and x2["dark_h"] == interrupts.MAX_HOLE_H and x2["unused_h"] == 0.0
    assert x2["lost_games"] == 0  # 局/日の記録がどこにも無いので局は数えない
    assert y2["setup_h"] is None and y2["extra_setup_usd"] == 0.0
    w = interrupts.summary([z2, x2, y2])
    assert w["sessions"] == 2 and w["interruptions"] == 1 and w["by_rent"][0]["name"] == "-"
