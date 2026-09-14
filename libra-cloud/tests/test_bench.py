# SPDX-License-Identifier: Apache-2.0
"""vast.ai の実測ベンチ: オファーの選別、inbox からの局/日、ベンチ用の設定と束、ホスト側スクリプト。"""
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

import librasearch
import librashogi as ls
from libra_cloud.bench import bench_config, games_per_day, pick_offers, scan_inbox
from libra_cloud.prepare import BUNDLE_PATHS, make_bundle, write_run_dir
from libra_league.config import load_config
from libra_league.workers import load_weights, write_games_file
from libra_net.model import LibraNet, NetConfig

ROOT = Path(__file__).resolve().parents[2]


def _offer(i, dph, **kw):
    o = {"id": i, "dph_total": dph, "num_gpus": 1, "cpu_cores_effective": 16, "inet_down": 500.0, "reliability2": 0.99,
         "cuda_max_good": 12.9, "inet_up_cost": 0.004, "inet_down_cost": 0.004, "gpu_name": "RTX 3090"}
    o.update(kw)
    return o


def test_pick_offers_filters_and_sorts():
    offers = [
        _offer(1, 0.30),
        _offer(2, 0.12),
        _offer(3, 0.10, reliability2=0.90),        # 信頼度が低い
        _offer(4, 0.11, cpu_cores_effective=4),    # CPU が足りない
        _offer(5, 0.11, cuda_max_good=12.4),       # CUDA 12.8 未満のドライバ
        _offer(6, 0.11, inet_down=50.0),           # 回線が遅い
        _offer(7, 0.11, inet_up_cost=0.05),        # 転送料が高い
        _offer(8, 0.11, num_gpus=2),
        _offer(9, 0.60),                           # 上限を超える
        _offer(10, 0.12, cpu_cores_effective=32),  # 同じ値段なら CPU の多い方
    ]
    got = [o["id"] for o in pick_offers(offers, max_dph=0.5)]
    assert got == [10, 2, 1]
    assert pick_offers(offers, max_dph=0.5, price_key="min_bid") == []  # 価格の欄が無いオファーは選ばない
    # CPU の速さの下限（探索の反映は 1 スレッドの速さで決まる）。cpu_ghz が無いオファーは外す
    fast = [_offer(20, 0.17, cpu_ghz=5.76, cpu_name="AMD Ryzen 9 7950X3D"), _offer(21, 0.15, cpu_ghz=2.4, cpu_name="AMD EPYC 7532"),
            _offer(22, 0.14)]
    assert [o["id"] for o in pick_offers(fast, max_dph=0.5, min_cpu_ghz=4.4)] == [20]
    assert [o["id"] for o in pick_offers(fast, max_dph=0.5)] == [22, 21, 20]
    # 信頼度の下限は短いベンチなら下げられる（途中で落ちてもインスタンスは消す）
    shaky = [_offer(30, 0.245, reliability2=0.9426), _offer(31, 0.30, reliability2=0.99)]
    assert [o["id"] for o in pick_offers(shaky, max_dph=0.5)] == [31]
    assert [o["id"] for o in pick_offers(shaky, max_dph=0.5, min_rel=0.94)] == [30, 31]


def test_games_per_day_uses_window_after_warmup():
    start = 1000.0
    files = [(start + 100, 100, 9000), (start + 400, 100, 8800), (start + 460, 100, 9200), (start + 520, 100, 9000)]
    r = games_per_day(files, start=start, warmup_s=300)
    # 窓は 400 s のファイルから 520 s のファイルまで（120 s に 200 局）
    assert r["games_per_day"] == pytest.approx(200 / 120 * 86400)
    assert r["window_s"] == pytest.approx(120) and r["files"] == 3 and r["games"] == 300
    assert r["avg_moves"] == pytest.approx(27000 / 300)
    assert games_per_day(files[:1], start=start, warmup_s=0)["games_per_day"] is None  # 窓に 2 ファイル無ければ出さない


def _games(n: int, seed: int) -> list[dict]:
    sp = librasearch.SelfPlay({"full_sims": 8, "fast_sims": 4, "full_prob": 0.5, "max_ply": 320}, 8, seed=seed, threads=2)
    sq = np.zeros((8, 81, ls.SQ_FEATS), np.float32)
    glob = np.zeros((8, ls.GLOB_FEATS), np.float32)
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    while len(out) < n:
        sp.collect(sq, glob)
        sp.apply(rng.standard_normal((8, ls.POLICY_SIZE), dtype=np.float32), np.tile(np.array([0.4, 0.2, 0.4], np.float32), (8, 1)))
        out += sp.take_finished()
    return out[:n]


def test_scan_inbox(tmp_path: Path):
    g = _games(6, 1)
    write_games_file(tmp_path, "bench", 3, "ls", g[:4])
    write_games_file(tmp_path, "bench", 3, "ls", g[4:])
    (tmp_path / "junk.npz").write_bytes(b"x")
    files = scan_inbox(tmp_path)
    assert [n for _, n, _ in files] == [4, 2]
    assert sum(m for _, _, m in files) == sum(len(x["moves"]) for x in g)
    assert files[0][0] <= files[1][0]


def test_bench_config_is_standalone():
    cfg = load_config(None)
    cfg["selfplay"]["openings"] = "/home/sakis/libra-run/lx/openings.json"
    cfg["exploiter"]["main_ckpt"] = "/x/main.pt"
    cfg["auto"]["enabled"] = True
    cfg["workers"]["enabled"] = True
    b = bench_config(cfg, n_games=256, threads=8)
    assert b["selfplay"]["openings"] == "" and b["exploiter"]["main_ckpt"] == "" and not b["auto"]["enabled"] and not b["workers"]["enabled"]
    assert b["selfplay"]["n_games"] == 256 and b["selfplay"]["threads"] == 8
    assert b["search"] == cfg["search"] and b["net"] == cfg["net"] and b["run_id"] == cfg["run_id"]
    assert b["search"]["defer_root_proof"] is True  # ホストで on / off を切り替えられるよう設定に出ている
    assert cfg["selfplay"]["openings"] != ""  # 元の設定は変えない


def test_write_run_dir_and_bundle(tmp_path: Path):
    net = {"d_model": 32, "n_layers": 1, "n_heads": 4, "d_ff": 64, "dropout": 0.0}
    m = LibraNet(NetConfig.from_dict(net))
    cfg = load_config(None)
    cfg["net"] = net
    ck = tmp_path / "latest.pt"
    torch.save({"model": m.state_dict(), "opt": {}, "step": 77, "config": cfg}, ck)
    from libra_league.config import dump_toml

    (tmp_path / "config.toml").write_text(dump_toml(cfg), encoding="utf-8")
    run = write_run_dir(ck, tmp_path / "config.toml", tmp_path / "out", n_games=128, threads=4)
    assert run == tmp_path / "out" / "run" / "ls"
    m2, step, run_id = load_weights(run / "weights" / "latest.pt")
    assert step == 77 and run_id == "ls" and m2.cfg == m.cfg
    b = load_config(run / "config.toml")
    assert b["selfplay"]["n_games"] == 128 and b["net"] == net
    tar = make_bundle(ROOT, tmp_path / "out")
    with tarfile.open(tar) as t:
        names = t.getnames()
    for p in BUNDLE_PATHS:
        assert any(n == p or n.startswith(p + "/") for n in names), p
    assert "run/ls/weights/latest.pt" in names and "run/ls/config.toml" in names
    assert not any(n.startswith(("libra-engine", "third_party", ".venv", "build/")) for n in names)


def test_host_ssh_with_and_without_log(tmp_path: Path, monkeypatch):
    """到達確認（ログ無し）とセットアップ（ログあり）の両方で ssh を呼べる（9/14 にログ無しの経路で落ち、起動済みのホストを 2 台消した）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("vast_bench", ROOT / "libra-cloud" / "vast_bench.py")
    vb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vb)
    calls = []

    def fake_run(argv, stdout=None, stderr=None, timeout=None, **kw):
        calls.append((argv, stdout))
        if hasattr(stdout, "write"):
            stdout.write(b"ok\n")
        return subprocess.CompletedProcess(argv, 0 if argv[-1] != "false" else 1)

    monkeypatch.setattr(vb.subprocess, "run", fake_run)
    h = vb.Host("ssh1.example", 12345)
    assert h.ssh("true", timeout=5, check=False) == 0
    assert calls[-1][0][-2:] == ["root@ssh1.example", "true"] and "12345" in calls[-1][0]
    log = tmp_path / "setup.log"
    assert h.ssh("echo hi", timeout=5, log_path=log) == 0
    assert log.read_bytes() == b"ok\n"
    with pytest.raises(RuntimeError):
        h.ssh("false", timeout=5)


def test_wait_ssh_gives_up_on_stalled_instance(monkeypatch):
    """イメージの取得が止まったホストは全体の待ち時間を待たずに見切る（9/14 に 17 分待った）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("vast_bench", ROOT / "libra-cloud" / "vast_bench.py")
    vb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vb)
    clock = [0.0]
    monkeypatch.setattr(vb.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vb.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(vb, "log", lambda m: None)

    class Stuck:
        def show_instance(self, iid):
            return {"actual_status": "loading", "status_msg": "Pulling from pytorch/pytorch"}

    with pytest.raises(TimeoutError, match="stalled"):
        vb.wait_ssh(Stuck(), 1, timeout=1200, stall=480)
    assert 480 < clock[0] < 600  # 20 分待たずに 8 分強で見切る

    # running なのに鍵が拒まれ続ける（9/14、authorized_keys の権限）: 5 分で見切る。状態が変わらなくても stall では切らない
    class Denied:
        def show_instance(self, iid):
            return {"actual_status": "running", "status_msg": "success", "ssh_host": "ssh7.example", "ssh_port": 1}

    monkeypatch.setattr(vb.Host, "ssh", lambda self, cmd, timeout, log_path=None, check=True: 255)
    clock[0] = 0.0
    with pytest.raises(TimeoutError, match="ssh failed"):
        vb.wait_ssh(Denied(), 1, timeout=1200, stall=100, ssh_fail=300)
    assert 300 < clock[0] < 400
    assert "authorized_keys" in vb.ONSTART and "chmod 600" in vb.ONSTART


def test_ssh_prefers_direct_connection(monkeypatch):
    """中継（sshN.vast.ai）への逆向きトンネルが張れないホストでも、公開 IP とポートへの直接接続で入る（9/14）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("vast_bench", ROOT / "libra-cloud" / "vast_bench.py")
    vb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vb)
    inst = {"actual_status": "running", "status_msg": "success", "ssh_host": "ssh2.vast.ai", "ssh_port": 19638,
            "public_ipaddr": "190.111.198.202 ", "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "10299"}]}}
    hs = vb.ssh_hosts(inst)
    assert [(h.host, h.port) for h in hs] == [("190.111.198.202", 10299), ("ssh2.vast.ai", 19638)]
    assert [(h.host, h.port) for h in vb.ssh_hosts({"ssh_host": "ssh2.vast.ai", "ssh_port": 1})] == [("ssh2.vast.ai", 1)]
    monkeypatch.setattr(vb, "log", lambda m: None)
    monkeypatch.setattr(vb.time, "sleep", lambda s: None)
    monkeypatch.setattr(vb.Host, "ssh", lambda self, cmd, timeout, log_path=None, check=True: 0 if self.host != "ssh2.vast.ai" else 255)

    class Api:
        def show_instance(self, iid):
            return inst

    h = vb.wait_ssh(Api(), 1, timeout=60)
    assert (h.host, h.port) == ("190.111.198.202", 10299)


def test_image_cuda_sets_min_driver():
    """イメージの CUDA より古いドライバーのホストは選ばない（GeForce では前方互換が使えず Error 804、9/14）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("vast_bench", ROOT / "libra-cloud" / "vast_bench.py")
    vb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vb)
    assert vb.image_cuda("vastai/pytorch:2.11.0-cu128-cuda-12.9-mini-py312-2026-09-08") == 12.9
    assert vb.image_cuda("pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime") == 12.8
    assert vb.image_cuda(vb.IMAGE) == 12.9
    assert vb.image_cuda("ubuntu:24.04") == 12.8
    offers = [_offer(1, 0.17, cuda_max_good=12.8), _offer(2, 0.19, cuda_max_good=13.2)]
    assert [o["id"] for o in pick_offers(offers, max_dph=0.5, min_cuda=vb.image_cuda(vb.IMAGE))] == [2]


def test_try_create_reports_api_error_body():
    """作成の 400 で落ちず、API の本文を理由にして次のオファーへ移れる（9/14 に理由が見えないまま落ちた）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("vast_bench", ROOT / "libra-cloud" / "vast_bench.py")
    vb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vb)

    class Resp:
        text = '{"error": "invalid_args", "msg": "offer no longer available"}'

    class HTTPError(Exception):
        def __init__(self):
            super().__init__("400 Client Error: Bad Request")
            self.response = Resp()

    class Api:
        def __init__(self, result):
            self.result = result

        def create_instance(self, offer_id, **kw):
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    iid, why = vb.try_create(Api(HTTPError()), 1, image="x")
    assert iid is None and "400" in why and "offer no longer available" in why
    assert vb.try_create(Api({"success": True, "new_contract": 42}), 1) == (42, "")
    iid, why = vb.try_create(Api({"success": False, "msg": "no"}), 1)
    assert iid is None and "no" in why


def test_host_scripts_parse():
    for s in sorted((ROOT / "libra-cloud" / "bench").glob("*.sh")):
        subprocess.run(["bash", "-n", str(s)], check=True)
    assert (ROOT / "libra-cloud" / "bench" / "host_setup.sh").exists() and (ROOT / "libra-cloud" / "bench" / "host_bench.sh").exists()


def test_bench_command_defer_ab_on_same_host():
    """--defer-ab では同じホストで根の証明探索の先送りを off → on の順に続けて回し、結果を別のディレクトリに置く。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("vast_bench", ROOT / "libra-cloud" / "vast_bench.py")
    vb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vb)
    one = vb.bench_command(20, 6, 512, 300, [])
    assert one.count("host_bench.sh") == 1 and one.rstrip().endswith("host_bench.sh 20 $T 512 300")
    ab = vb.bench_command(15, 6, 512, 300, ["off", "on"])
    assert ab.count("host_bench.sh") == 2 and ab.index("300 off") < ab.index("300 on") and "T=6;" in ab
    sh = (ROOT / "libra-cloud" / "bench" / "host_bench.sh").read_text()
    assert "defer_root_proof" in sh and "/root/out${DEFER:+-$DEFER}" in sh
