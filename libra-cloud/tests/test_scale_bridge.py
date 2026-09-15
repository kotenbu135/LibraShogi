# SPDX-License-Identifier: Apache-2.0
"""全組の検証対局のクラウド用の配管: 束のディレクトリ → ホストの seq worker → ブリッジの検査 → 手元の seq run の取り込み。"""
import json

import torch

from libra_cloud.prepare import write_scale_dir
from libra_cloud.scale_bridge import ScaleBridge, ScaleLocalTransport
from libra_net.model import LibraNet, NetConfig
from libra_scale import pairs as P
from libra_scale import seqrule as R
from libra_scale import seqrun as S

FAST = {"fast_sims": 4, "proof_nodes": 0, "mate_nodes_root": 0, "max_ply": 40}


def make_seq(tmp_path, keys):
    torch.manual_seed(0)
    net = {"d_model": 32, "n_layers": 1, "n_heads": 4, "d_ff": 64}
    ck = tmp_path / "tiny.pt"
    torch.save({"model": LibraNet(NetConfig.from_dict(net)).state_dict(), "config": {"net": net}, "step": 3}, ck)
    base = {"model": {"path": str(ck), "step": 3}, "sims": 8,
            "pairs": [{"kb": P.usi(a), "kw": P.usi(b), "mirror": [P.usi(P.mirror_sq(a)), P.usi(P.mirror_sq(b))], "v_hat": 0.5}
                      for a, b in P.unique_pairs()]}
    bt = tmp_path / "base.json"
    bt.write_text(json.dumps(base))
    d = tmp_path / "seq-test"
    S.init_dir(d, bt, sims=4, rule=R.Rule(eps=0.45, min_games=2, look_every=2, max_games=6), keys=keys, search_overrides=FAST)
    co = S.Coordinator(d, log=lambda s: None)
    co.update()
    co.write_active(co.active())
    return d, co


def test_worker_bridge_roundtrip(tmp_path):
    d, co = make_seq(tmp_path, ["1f 1a", "2g 6a"])
    host = write_scale_dir(d, tmp_path / "bundle")
    assert host == tmp_path / "bundle" / "scale" / "seq-test" and (host / "weights.pt").exists()
    assert S.run_worker(host, torch.device("cpu"), "vast1", n_games=4, threads=2, compile="none", seed=3, chunk_games=2, flush_s=0.0,
                        max_games=2, log=lambda s: None) == 0
    good = sorted((host / "inbox").glob("*.jsonl.gz"))
    n_good = sum(len(S.read_chunk(p)) for p in good)
    assert good and n_good >= 2
    # 手順を書き換えた局を含むファイルはまるごと弾く
    rec = S.read_chunk(good[0])[0]
    S.write_chunk(host / "inbox", "evil", 0, [dict(rec, result=1 if rec["result"] <= 0 else -1)])
    logs: list[str] = []
    b = ScaleBridge(d, ScaleLocalTransport(host), tmp_path / "bridge", log=logs.append)
    assert b.cycle() == n_good
    assert b.stats["rejected_files"] == 1 and any("rejected evil-" in s for s in logs)
    assert not list((host / "inbox").glob("*.jsonl.gz"))
    placed = [r for p in sorted((d / "inbox").glob("*.jsonl.gz")) for r in S.read_chunk(p)]
    assert len(placed) == n_good and all(r["worker"] == "vast1" for r in placed)
    assert co.ingest() == n_good and co.state["workers"] == {"vast1": n_good}
    # 打つ組が変わったら active.json を送り直す。全部止まれば done
    for st in co.states.values():
        st.update(stop="sig", stop_n=st["n"], stop_w=1.0, stop_se=0.1)
    co.write_active(co.active())
    b.cycle()
    assert json.loads((host / "active.json").read_text())["done"] and b.done()
