# SPDX-License-Identifier: Apache-2.0
"""借りたホストの実測（libra_cloud.hosts）: bridge.log からの定常状態の局/日、実績の鍵、見込みと並び順。"""
import json
import time
from pathlib import Path

from libra_cloud import hosts
from libra_cloud.bench import pick_offers


def _at(t_bridge: float, dt: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t_bridge + dt))


def _session(root: Path, name: str, *, t_bridge: float, offer: dict, points: list[tuple[float, int]]) -> Path:
    """points: (ブリッジ起動からの秒, 累計局数) の列。"""
    d = root / name
    (d / "bridge").mkdir(parents=True)
    (d / "instance.json").write_text(json.dumps({"instance": 1, "offer": offer, "t_bridge": t_bridge}), encoding="utf-8")
    (d / "bridge" / "bridge.log").write_text(
        f"{_at(t_bridge, 0)} bridge: start\n"
        + "".join(f"{_at(t_bridge, dt)} bridge: placed 100 games (total {n}, verify 0.7 ms/game)\n" for dt, n in points), encoding="utf-8")
    return d


def _offer(**kw):
    o = {"id": 1, "dph_total": 0.17, "dph_eff": 0.17, "num_gpus": 1, "cpu_cores_effective": 16, "cpu_ghz": 5.0, "inet_down": 500.0,
         "reliability2": 0.99, "cuda_max_good": 12.9, "inet_up_cost": 0.004, "inet_down_cost": 0.004,
         "gpu_name": "RTX 5070 Ti", "cpu_name": "AMD Ryzen 7 7800X3D 8-Core Processor", "machine_id": 4242}
    o.update(kw)
    return o


def test_placed_points_reads_the_log_and_carries_the_date_over_midnight():
    t0 = time.mktime((2026, 9, 16, 23, 50, 0, 0, 0, -1))
    text = (f"{_at(t0, 0)} bridge: placed 100 games (total 100, verify 0.7 ms/game)\n"
            "23:55:00 bridge: error: OSError: boom\n"
            f"{_at(t0, 1800)} bridge: placed 100 games (total 900, verify 0.7 ms/game)\n")  # 日付をまたぐ 00:20:00
    got = hosts.placed_points(text, t0)
    assert [n for _, n in got] == [100, 900]
    assert got[1][0] - got[0][0] == 1800  # 00:20 を翌日と読む（前の日に戻らない）


def test_session_speed_uses_the_steady_slope_and_skips_the_warmup(tmp_path):
    """起動直後は 512 局を同時に打ち始めて回収がまとまるので、最初の 10 分を外した区間の傾きで測る。"""
    t0 = time.mktime((2026, 9, 16, 15, 0, 0, 0, 0, -1))
    # 暖機の 10 分で 8,000 局がまとめて入り、その後は 20 分ごとに 9,375 局（1 時間に 28,125 局 = 675,000 局/日）
    pts = [(60.0, 8000), (300.0, 8000)] + [(600.0 + 1200 * i, 8000 + 9375 * i) for i in range(4)]
    d = _session(tmp_path, "ls-20260916-1500", t_bridge=t0, offer=_offer(), points=pts)
    s = hosts.session_speed(d)
    assert round(s["games_per_day"]) == 675000
    assert s["machine_id"] == 4242 and s["gpu"] == "RTX 5070 Ti" and s["cpu"] == "AMD Ryzen 7 7800X3D"
    assert round(s["span_h"], 2) == 1.0 and s["games"] == 28125


def test_session_speed_returns_none_when_it_cannot_be_measured(tmp_path):
    t0 = time.mktime((2026, 9, 16, 15, 0, 0, 0, 0, -1))
    short = _session(tmp_path, "ls-1", t_bridge=t0, offer=_offer(), points=[(700.0, 100), (900.0, 300)])  # 区間 200 秒
    assert hosts.session_speed(short) is None
    few = _session(tmp_path, "ls-2", t_bridge=t0, offer=_offer(), points=[(700.0, 100), (4000.0, 400)])   # 300 局だけ
    assert hosts.session_speed(few) is None
    warm = _session(tmp_path, "ls-3", t_bridge=t0, offer=_offer(), points=[(60.0, 100), (300.0, 9000)])   # 暖機の中だけ
    assert hosts.session_speed(warm) is None
    (tmp_path / "ls-4" / "bridge").mkdir(parents=True)
    assert hosts.session_speed(tmp_path / "ls-4") is None  # instance.json が無い


def _table(rows):
    """(gpu, cpu, machine_id, 局/日) から速さの表を作る。"""
    return hosts.speed_table([{"name": f"s{i}", "gpu": g, "cpu": hosts.short_cpu(c), "machine_id": m, "games_per_day": v, "span_h": 1.0}
                              for i, (g, c, m, v) in enumerate(rows)])


def test_estimate_prefers_the_same_machine_then_the_same_cpu_then_the_gpu():
    t = _table([("RTX 5070 Ti", "AMD Ryzen 7 7800X3D", 4242, 675_000),
                ("RTX 5070 Ti", "AMD Ryzen 9 5950X", 11, 576_000),
                ("RTX 5070 Ti", "AMD Ryzen Threadripper 3970X", 12, 566_000)])
    assert hosts.estimate(_offer(), t)["from"] == "machine"                                 # 同じ機械
    assert hosts.estimate(_offer(machine_id=99), t) == {"games_per_day": 675_000, "from": "cpu", "sessions": 1}
    other = hosts.estimate(_offer(machine_id=99, cpu_name="Intel Xeon E5-2680 v4"), t)      # 同じ GPU の中央値
    assert other["from"] == "gpu" and other["games_per_day"] == 576_000
    assert hosts.estimate(_offer(gpu_name="RTX 4090", machine_id=None, cpu_name="?"), t) is None


def test_estimate_takes_the_median_of_several_sessions_on_the_same_machine():
    t = _table([("RTX 5070 Ti", "AMD Ryzen 7 7800X3D", 4242, 664_615),
                ("RTX 5070 Ti", "AMD Ryzen 7 7800X3D", 4242, 674_932),
                ("RTX 5070 Ti", "AMD Ryzen 7 7800X3D", 4242, 300_000)])
    e = hosts.estimate(_offer(), t)
    assert e["games_per_day"] == 664_615 and e["sessions"] == 3  # 1 回の外れ（借り直しの直後など）に引きずられない


def test_usd_per_1m():
    assert round(hosts.usd_per_1m(0.168, 674_932), 2) == 5.97
    assert hosts.usd_per_1m(0.168, 0) is None and hosts.usd_per_1m(None, 1) is None


def test_annotate_fills_unknown_offers_with_the_median_of_the_known_ones():
    t = _table([("RTX 5070 Ti", "AMD Ryzen 7 7800X3D", 4242, 675_000)])
    known, unknown = _offer(id=1, dph_eff=0.168), _offer(id=2, dph_eff=0.168, machine_id=None, gpu_name="RTX 4090", cpu_name="?")
    a = {o["id"]: o for o in hosts.annotate([known, unknown], t, "dph_eff")}
    assert a[1]["est_from"] == "machine" and round(a[1]["est_usd_per_1m"], 2) == 5.97
    assert a[2]["est_from"] is None and a[2]["est_games_per_day"] is None
    assert round(a[2]["est_usd_per_1m"], 2) == 5.97  # 実測が無いものは実測のあるホストの中央値を当てる（前にも後ろにも寄せない）
    none = hosts.annotate([unknown], {}, "dph_eff")  # 実測がまったく無ければ見込みも無い
    assert none[0]["est_usd_per_1m"] is None


def test_pick_offers_ranks_by_expected_cost_per_million_games_not_by_price():
    """2026-09-16 の実測: 5080 は局/日が 1.3 倍でも $/h が 1.6 倍で 100 万局あたりでは 5070 Ti に負ける。
    同じ 5070 Ti でも CPU の遅いホストは 100 万局あたり $7.80 で、$/h は安くても後ろに回す。"""
    t = _table([("RTX 5070 Ti", "AMD Ryzen 7 7800X3D", 4242, 674_932),
                ("RTX 5070 Ti", "AMD Ryzen Threadripper 3970X", 12, 566_040),
                ("RTX 5080", "AMD Ryzen 9 7900", 13, 877_037)])
    offers = [_offer(id=1, dph_eff=0.184, machine_id=12, cpu_name="AMD Ryzen Threadripper 3970X"),   # 安いが遅い $7.80/100 万局
              _offer(id=2, dph_eff=0.262, machine_id=13, gpu_name="RTX 5080", cpu_name="AMD Ryzen 9 7900"),  # 速いが高い $7.17
              _offer(id=3, dph_eff=0.168, machine_id=4242)]                                          # 最良 $5.98
    cond = dict(max_dph=0.30, price_key="dph_eff", min_cpu_ghz=4.4, min_rel=0.94)
    assert [o["id"] for o in pick_offers(hosts.annotate(offers, t, "dph_eff"), **cond)] == [3, 2, 1]
    assert [o["id"] for o in pick_offers(offers, **cond)] == [3, 1, 2]  # 見込みを付けなければこれまで通り値段の安い順


def test_scan_sessions_reads_every_session_under_the_roots(tmp_path):
    t0 = time.mktime((2026, 9, 16, 12, 0, 0, 0, 0, -1))
    pts = [(700.0 + 600 * i, 1000 + 5000 * i) for i in range(6)]
    _session(tmp_path / "cloud", "ls-20260916-1200", t_bridge=t0, offer=_offer(), points=pts)
    _session(tmp_path / "cloud", "ls-20260916-1400", t_bridge=t0 + 7200, offer=_offer(machine_id=7), points=pts)
    (tmp_path / "cloud" / "current").write_text("ls-20260916-1400", encoding="utf-8")  # ディレクトリでないものは飛ばす
    got = hosts.scan_sessions([tmp_path / "cloud", tmp_path / "missing"])
    assert [s["machine_id"] for s in got] == [4242, 7]
    assert hosts.speed_table(got)[("gpu", "RTX 5070 Ti")]["sessions"] == 2
