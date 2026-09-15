# SPDX-License-Identifier: Apache-2.0
"""逐次の検証対局（libra_scale/seqrule.py・seqrun.py）のテスト。"""
import gzip
import json

import pytest
import torch

from libra_net.model import LibraNet, NetConfig
from libra_scale import pairs as P
from libra_scale import seqrule as R
from libra_scale import seqrun as S
from libra_scale.cli import main as cli_main


def st_with(rule, n, sente, draw=0):
    st = R.new_state(rule)
    st.update(n=n, sente=sente, draw=draw, gote=n - sente - draw)
    return st


def test_symmetric_groups():
    g = R.symmetric_groups()
    assert len(g["five"]) == 3 and len(g["line"]) == 15 and len(g["point"]) == 15
    assert set(g["five"]) == {"5g 5c", "5h 5b", "5i 5a"}
    assert set(g["five"]) <= set(g["line"]) and set(g["five"]) <= set(g["point"])
    allk = {R.key_of(a, b) for a, b in P.unique_pairs()}
    union = set(g["line"]) | set(g["point"])
    assert len(union) == 27 and union <= allk
    exp = set()
    for k in union:
        kb, kw = R.parse_key(k)
        exp |= {(kb, kw), (P.mirror_sq(kb), P.mirror_sq(kw))}
    assert len(exp) == 51  # measurements.md 2026-09-15 15:10 の「対称な 51 組」
    for k in g["line"]:  # 同じ筋で段が逆（sq = 筋 * 9 + 段）
        kb, kw = R.parse_key(k)
        assert kb // 9 == kw // 9 and kb % 9 + kw % 9 == 8
    for k in g["point"]:  # 盤の中心について対称
        kb, kw = R.parse_key(k)
        assert kb // 9 + kw // 9 == 8 and kb % 9 + kw % 9 == 8


def test_rule_choose_stage_looks():
    rule = R.Rule()
    st = st_with(rule, 99, 90)
    assert R.update(st, rule, False) is None and st["stop"] is None  # 100 局までは見ない
    st = st_with(rule, 100, 80)
    assert R.update(st, rule, False) == "sig" and st["stop_n"] == 100 and not R.is_active(st, rule)
    # 対称な組は有意では止めず、次に見る局数へ進む（間の局数は飛ばす）
    st = st_with(rule, 100, 80)
    assert R.update(st, rule, True) is None and st["next_look"] == 150
    st.update(n=170, sente=136)
    assert R.update(st, rule, True) is None and st["next_look"] == 200
    # 0.5 付近: 2,000 局では半幅 0.0219 で続け、2,450 局で半幅 0.0198 になり精度で止まる
    st = st_with(rule, 2000, 1000)
    st["next_look"] = 2000
    assert R.update(st, rule, False) is None and st["next_look"] == 2050
    st.update(n=2450, sente=1225, gote=1225)
    assert R.update(st, rule, False) == "eps" and R.is_candidate(st)
    # 上限
    r2 = R.Rule(eps=0.001, max_games=300)
    st = st_with(r2, 300, 150)
    st["next_look"] = 300
    assert R.update(st, r2, True) == "cap"


def test_rule_place_stage():
    rule = R.Rule(eps_place=0.01)
    st = st_with(rule, 2450, 1225)
    st["next_look"] = 2450
    assert R.update(st, rule, False) == "eps" and R.is_active(st, rule)
    st.update(n=2949, sente=1474)
    assert R.update(st, rule, False) is None  # 次に見るのは 2,950 局
    st.update(n=9700, sente=4850)
    assert R.update(st, rule, False) == "place_done" and not R.is_active(st, rule)
    # 0.5 から離れた候補は外す
    st2 = st_with(rule, 2450, 1250)
    st2["next_look"] = 2450
    assert R.update(st2, rule, False) == "eps" and R.is_candidate(st2)
    st2.update(n=6000, sente=3300)
    assert R.update(st2, rule, False) == "place_out"
    # 有意で止まった組は候補にしない。置く側の精度を後から有効にすると候補は再開する
    r0 = R.Rule()
    st3 = st_with(r0, 100, 80)
    R.update(st3, r0, False)
    assert not R.is_candidate(st3)
    st4 = st_with(r0, 2450, 1225)
    st4["next_look"] = 2450
    R.update(st4, r0, False)
    assert not R.is_active(st4, r0) and R.is_active(st4, R.Rule(eps_place=0.01))


def test_active_keys_symmetric_first_and_lines():
    rule = R.Rule()
    states = {"5i 5a": R.new_state(rule), "1f 1a": R.new_state(rule), "2g 6a": R.new_state(rule)}
    sym = {"5i 5a"}
    assert R.active_keys(states, rule, sym) == ["5i 5a"]
    states["5i 5a"].update(stop="eps", stop_n=2450, stop_w=0.6, stop_se=0.01)
    assert R.active_keys(states, rule, sym) == ["1f 1a", "2g 6a"]
    lines = R.opening_lines(["5i 5a", "2g 6a"])
    # 鏡映も同じ重みで入れ、鏡映が自分と同じ 5 筋の組は同じ手順を 2 回入れる
    assert len(lines) == 4 and lines.count(["K*5i", "K*5a"]) == 2
    assert ["K*2g", "K*6a"] in lines and ["K*8g", "K*4a"] in lines


def test_alerts_symmetric_gote_lean():
    rule = R.Rule()
    groups = R.symmetric_groups()
    states = {k: R.new_state(rule) for ks in groups.values() for k in ks}
    fired: list[str] = []
    assert R.check_alerts(states, groups, rule, fired) == []
    s = states["5i 5a"]
    s.update(n=2450, sente=1127, gote=1323, next_look=2450)  # 先手の得点 0.46
    assert R.update(s, rule, True) == "eps"
    alerts = R.check_alerts(states, groups, rule, fired)
    got = {(a["target"], a["level"]) for a in alerts}
    assert ("5i 5a", "確定") in got and ("5i 5a", "早期") in got
    assert {"group:five", "group:line", "group:point", "group:all"} <= {a["target"] for a in alerts}
    assert all(a["winrate"] < 0.5 and "後手" in a["message"] for a in alerts)
    assert R.check_alerts(states, groups, rule, fired) == []  # 同じアラートは 1 回だけ
    # 先手に傾くのはアラートにしない
    states2 = {k: R.new_state(rule) for ks in groups.values() for k in ks}
    states2["5h 5b"].update(n=2450, sente=1323, gote=1127, next_look=2450)
    R.update(states2["5h 5b"], rule, True)
    assert R.check_alerts(states2, groups, rule, []) == []


def write_tiny(tmp_path):
    torch.manual_seed(0)
    net = {"d_model": 32, "n_layers": 1, "n_heads": 4, "d_ff": 64}
    m = LibraNet(NetConfig.from_dict(net)).eval()
    ck = tmp_path / "tiny.pt"
    torch.save({"model": m.state_dict(), "config": {"net": net}, "step": 7}, ck)
    base = {"version": 0, "model": {"path": str(ck), "step": 7}, "sims": 8, "pruned": "x", "balanced": [["5i", "5a"]],
            "pairs": [{"kb": P.usi(kb), "kw": P.usi(kw), "mirror": [P.usi(P.mirror_sq(kb)), P.usi(P.mirror_sq(kw))], "v_hat": 0.5,
                       "selfplay": {"games": 0}} for kb, kw in P.unique_pairs()]}
    bt = tmp_path / "base.json"
    bt.write_text(json.dumps(base))
    return bt


FAST = {"fast_sims": 4, "proof_nodes": 0, "mate_nodes_root": 0, "max_ply": 40}


def test_seq_run_small(tmp_path):
    bt = write_tiny(tmp_path)
    d = tmp_path / "seq"
    rule = R.Rule(eps=0.45, min_games=2, look_every=2, max_games=6)
    keys = ["5i 5a", "1f 1a", "2g 6a"]
    cfg = S.init_dir(d, bt, sims=4, rule=rule, keys=keys, search_overrides=FAST)
    assert cfg["search"]["full_sims"] == 4 and cfg["search"]["full_prob"] == 1.0
    logs: list[str] = []
    assert S.run_local(d, torch.device("cpu"), n_games=6, threads=2, compile="none", seed=1, log=logs.append,
                       chunk_games=3, cycle_s=0.0, flush_s=0.0) == 0
    state = S.read_json(d / "state.json")
    assert all(state["pairs"][k]["stop"] for k in keys)
    assert not list((d / "inbox").glob("*.jsonl.gz"))
    recs = []
    for f in sorted((d / "games").glob("*.jsonl.gz")):
        recs += S.read_chunk(f)
    per: dict[str, int] = {}
    for r in recs:
        k = R.key_of(P.from_usi(r["kb"]), P.from_usi(r["kw"]))
        per[k] = per.get(k, 0) + 1
        S.check_record(r, cfg["search"])
    assert per == {k: state["pairs"][k]["n"] for k in keys} and state["games"] == len(recs)
    bad = dict(recs[0], result=-recs[0]["result"] if recs[0]["result"] else 1)
    with pytest.raises(ValueError):
        S.check_record(bad, cfg["search"])
    with pytest.raises(ValueError):
        S.check_record(dict(recs[0], moves=recs[0]["moves"] + " 5e5d"), cfg["search"])
    assert any("phase all" in s for s in logs)  # 対称な組（5i 5a）を先に終えてから残りへ進む
    assert (d / "status.json").exists() and S.read_json(d / "active.json")["done"]
    # 表: 検証した組に選ぶ側の手番、四段目の組は先手、置く側の集合
    table = R.build_table(S.read_json(bt), cfg, state)
    ver = [e for e in table["pairs"] if "verify" in e]
    assert len(ver) == 3 and all(e["choose"] in ("sente", "gote") for e in ver)
    # 四段目の 324 組は鏡映で 164 通り（両玉が 5 筋の 4 組は鏡映が自分）
    assert len(table["pairs"]) == 492 and len(table["forced"]) == 164 and all(e["choose"] == "sente" for e in table["forced"])
    assert len({(e["kb"], e["kw"]) for e in table["forced"]} | {tuple(e["mirror"]) for e in table["forced"]}) == 324
    assert table["balanced"] and table["verify"]["complete"] and table["verify"]["games"] == len(recs)
    assert table["symmetric"]["five"]["pooled"]["games"] == state["pairs"]["5i 5a"]["n"]
    # 再開しても打つ組が無ければすぐ終わる
    assert S.run_local(d, torch.device("cpu"), n_games=6, threads=2, compile="none", log=logs.append) == 0
    assert S.read_json(d / "state.json")["games"] == state["games"]
    out = tmp_path / "scale.json"
    assert cli_main(["seq", "table", "--dir", str(d), "--out", str(out)]) == 0 and json.loads(out.read_text())["version"] == 1
    assert cli_main(["seq", "status", "--dir", str(d)]) == 0


def test_ingest_counts_once(tmp_path):
    bt = write_tiny(tmp_path)
    d = tmp_path / "seq"
    S.init_dir(d, bt, sims=4, rule=R.Rule(), keys=["5i 5a"])
    recs = [{"kb": "5i", "kw": "5a", "result": 1, "reason": "no_legal_move", "plies": 90, "moves": "", "worker": "vast1"},
            {"kb": "5i", "kw": "5a", "result": -1, "reason": "no_legal_move", "plies": 91, "moves": "", "worker": "vast1"},
            {"kb": "1f", "kw": "1a", "result": 1, "reason": "no_legal_move", "plies": 91, "moves": "", "worker": "vast1"}]
    f = S.write_chunk(d / "inbox", "vast1", 0, recs)
    # 状態を書いた後、ファイルを games へ移す前に落ちた場合: 再開で数え直さない
    st = S.read_json(d / "state.json")
    st["ingested"].append(f.name)
    S.write_json(d / "state.json", st)
    co = S.Coordinator(d, log=lambda s: None)
    assert co.ingest() == 0 and co.states["5i 5a"]["n"] == 0 and (d / "games" / f.name).exists()
    S.write_chunk(d / "inbox", "vast1", 1, recs)
    assert co.ingest() == 2 and co.ingest() == 0  # 対象外の組（1f 1a）は数えない
    assert co.states["5i 5a"]["sente"] == 1 and co.states["5i 5a"]["gote"] == 1 and co.state["workers"]["vast1"] == 2
    with gzip.open(next((d / "games").glob("vast1-*-000001.jsonl.gz")), "rt") as fh:
        assert len(fh.readlines()) == 3
