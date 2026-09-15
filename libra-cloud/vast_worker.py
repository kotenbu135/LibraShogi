# SPDX-License-Identifier: Apache-2.0
"""vast.ai で GPU を 1 台借りて自己対局ワーカーを常駐させ、手元の学習側の run に局を足す。
終わったら（時間切れ・失敗・Ctrl+C・SIGTERM・SIGHUP のどれでも）ワーカーを止めて残りを取り、必ずインスタンスを消す。

    ~/.venvs/vastai/bin/python libra-cloud/vast_worker.py --gpu "RTX 5070 Ti" --max-dph 0.28 --hours 3 \\
      --run-dir ~/libra-run/ls --bundle <dir>/bundle.tar.gz --out <dir>

ふだんは bin/libra-vast start（管理コンソールのクラウドのタブ）から起動する。束は `python -m libra_cloud.prepare --worker` で作る。
重みの送信と局の回収・検査はブリッジ（libra_cloud.bridge、プロジェクトの .venv の Python で別プロセス）が行う。
学習側の run は [workers] enabled で起動しておく（inbox/ が無い間は取ってこない）。
<out>/bridge/STOP を置くとブリッジが止まり、ワーカーを止めて残りを取ってから抜ける。
<out>/instance.json に借りたインスタンスと時刻（t_rent、t_bridge）を書く（bin/libra-vast status が経過時間と費用を出す）。

--job scale は玉配置表の全組の検証対局（libra-scale seq）のワーカー: 束は `python -m libra_cloud.prepare --scale-dir <seq の run>`、
ホストで libra-scale seq worker（bench/host_scale.sh）、ブリッジは libra_cloud.scale_bridge。--run-dir の代わりに --scale-dir を渡す。
全部の組が止まるとブリッジが抜けてインスタンスを消す。ホストを失っても借り直さない。

    ~/.venvs/vastai/bin/python libra-cloud/vast_worker.py --job scale --scale-dir ~/libra-run/ls/scale/seq-v0.1 --gpu "RTX 4090" \\
      --max-dph 0.25 --hours 20 --rent bid --bundle <dir>/bundle.tar.gz --out ~/libra-run/cloud-scale/<名前>
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libra_cloud.bench import MAX_INET_COST, annotate_price, offer_query, offer_rejects, pick_offers, worker_threads  # noqa: E402
from vast_bench import IMAGE, KEY, ONSTART, image_cuda, log, try_create, wait_ssh  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROJECT_PY = REPO / ".venv" / "bin" / "python"
PYTHONPATH = ":".join(str(REPO / p) for p in ("libra-sim/python", "libra-search/python", "libra-net", "libra-league", "libra-scale", "libra-cloud"))
LABEL = "libra-worker"  # bin/libra-vast cleanup は libra- で始まるラベルのインスタンスを消す


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, help='例: "RTX 5070 Ti"')
    ap.add_argument("--max-dph", type=float, required=True, help="1 時間あたりの上限（$、ストレージ込み）")
    ap.add_argument("--hours", type=float, required=True, help="ワーカーを回す時間（借りてからの準備は含まない）")
    ap.add_argument("--job", choices=("selfplay", "scale"), default="selfplay", help="selfplay: 学習側の run に局を足す、scale: 玉配置表の全組の検証対局")
    ap.add_argument("--run-dir", default=None, help="学習側の run（例: ~/libra-run/ls）。--job selfplay で必須")
    ap.add_argument("--scale-dir", default=None, help="libra-scale seq の run。--job scale で必須")
    ap.add_argument("--bundle", required=True, help="libra_cloud.prepare --worker で作った bundle.tar.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--id", default="vast1", help="ワーカー名（対局ファイル名と seed）")
    ap.add_argument("--n-games", type=int, default=512)
    ap.add_argument("--threads", type=int, default=0, help="0 ならホストの CPU 数（12 まで）")
    ap.add_argument("--disk", type=float, default=30)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--min-credit", type=float, default=2.0)
    ap.add_argument("--min-cores", type=int, default=16)
    ap.add_argument("--min-cpu-ghz", type=float, default=0.0)
    ap.add_argument("--min-rel", type=float, default=0.98)
    ap.add_argument("--max-inet-cost", type=float, default=MAX_INET_COST, help="転送料（$/GB）の上限")
    ap.add_argument("--rent", choices=("bid", "on-demand"), default="on-demand", help="借り方（bid は割り込みあり。bin/libra-vast の既定は bid）")
    ap.add_argument("--bid-margin", type=float, default=0.1, help="入札額 = 最低入札 × (1 + この値)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from vastai.sdk import VastAI

    if a.job == "scale":
        if not a.scale_dir:
            ap.error("--job scale needs --scale-dir")
        run_dir = Path(a.scale_dir).expanduser()
        run_id = run_dir.name  # ホストの /root/libra/scale/<名前>
    else:
        if not a.run_dir:
            ap.error("--job selfplay needs --run-dir")
        run_dir = Path(a.run_dir).expanduser()
        run_id = subprocess.run([str(PROJECT_PY), "-c", "import sys, tomllib; print(tomllib.load(open(sys.argv[1], 'rb'))['run_id'])",
                                 str(run_dir / "config.toml")], check=True, capture_output=True, text=True).stdout.strip()
    label = "libra-scale" if a.job == "scale" else LABEL
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = VastAI(raw=True, quiet=True)
    credit = float((v.show_user() or {}).get("credit") or 0)
    log(f"credit ${credit:.2f}; run {run_dir} (run_id {run_id})")
    if credit < a.min_credit:
        log(f"credit below ${a.min_credit:.2f}; not renting")
        return 2
    min_cuda = image_cuda(a.image)
    offers = v.search_offers(query=offer_query(a.gpu, a.min_rel, min_cuda, a.disk), type=a.rent, order="dph_total", limit=100,
                             storage=a.disk) or []
    offers = annotate_price(offers, a.rent, a.bid_margin)
    cond = dict(max_dph=a.max_dph, min_cores=a.min_cores, min_cpu_ghz=a.min_cpu_ghz, min_cuda=min_cuda, min_rel=a.min_rel, max_inet_cost=a.max_inet_cost,
                price_key="dph_eff")
    cands = pick_offers(offers, **cond)
    log(f"{len(offers)} offers ({a.rent}), {len(cands)} usable; cheapest: "
        + ", ".join(f"#{o['id']} ${o['dph_eff']:.3f}/h{' bid $' + format(o['bid'], '.4f') if o.get('bid') else ''} "
                    f"cpu {o.get('cpu_cores_effective')} {str(o.get('cpu_name'))[:28]} "
                    f"{o.get('geolocation', '')}" for o in cands[:3]))
    if offers and not cands:  # 借りられなかった理由を launcher.log に残す（管理コンソールの「候補を見る」と同じ判定）
        counts: dict[str, int] = {}
        for o in offers:
            for r in offer_rejects(o, **cond):
                counts[r["key"]] = counts.get(r["key"], 0) + 1
        log("rejected by: " + ", ".join(f"{k} {n}" for k, n in counts.items()))
    if a.dry_run or not cands:
        return 0 if cands else 3
    pub = KEY.with_suffix(".pub").read_text().strip()
    try:
        v.create_ssh_key(pub)  # 既に登録済みならエラーが返るだけ
    except Exception as e:  # noqa: BLE001
        log(f"ssh key register: {type(e).__name__}: {str(e)[:120]}")

    iid = None
    bridge: subprocess.Popen | None = None
    stopping: list[int] = []

    def on_signal(signum, frame):
        stopping.append(signum)
        if bridge is not None and bridge.poll() is None:
            bridge.send_signal(signal.SIGTERM)  # ブリッジがワーカーを止めて残りを取ってから抜ける
        else:
            raise KeyboardInterrupt

    # SIGHUP も同じに扱う（管理コンソールから起動した wsl.exe が終わったとき。黙って死ぬとインスタンスが残って課金が続く）
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)
    t_rent = time.time()
    result: dict = {"gpu": a.gpu, "image": a.image, "hours": a.hours, "n_games": a.n_games, "run_id": run_id, "worker": a.id, "job": a.job}
    rc = 0
    try:
        for offer in cands[:3]:
            bid = {"price": offer["bid"]} if offer.get("bid") else {}  # 入札（SDK の price は入札額。bid_price と同じ）
            iid, why = try_create(v, offer["id"], image=a.image, disk=a.disk, label=label, ssh=True, direct=True,
                                  cancel_unavail=True, onstart_cmd=ONSTART, **bid)
            log(f"create #{offer['id']} ${offer['dph_eff']:.3f}/h{' bid $' + format(offer['bid'], '.4f') if offer.get('bid') else ''} -> instance {iid} {why}")
            if not iid:
                continue
            result["offer"] = {k: offer.get(k) for k in ("id", "dph_total", "dph_eff", "bid", "min_bid", "is_bid", "gpu_name", "cpu_name",
                                                        "cpu_cores_effective", "cpu_ghz", "reliability2", "geolocation", "cuda_max_good",
                                                        "inet_up_cost", "inet_down_cost")}
            result["rent"] = a.rent
            result["instance"] = iid
            t_rent = time.time()
            write_json(out / "instance.json", {"instance": iid, "offer": result["offer"], "t_rent": t_rent})
            try:
                v.attach_ssh(iid, pub)
                host = wait_ssh(v, iid, timeout=1200)
                break
            except Exception as e:  # noqa: BLE001  起動しないホストは消して次へ
                log(f"instance {iid} failed to start: {type(e).__name__}: {str(e)[:200]}")
                v.destroy_instance(iid)
                write_json(out / "instance.json", {"instance": iid, "offer": result["offer"], "t_rent": t_rent, "destroyed": time.time()})
                iid = None
        if iid is None:
            log("no instance started")
            return 4
        result["t_ready_s"] = round(time.time() - t_rent)
        log(f"ssh ready after {result['t_ready_s']} s; uploading bundle")
        host.ssh("mkdir -p /root/libra", timeout=60)
        host.put(Path(a.bundle), "/root/bundle.tar.gz")
        t0 = time.time()
        host.ssh("tar xzf /root/bundle.tar.gz -C /root/libra && bash /root/libra/libra-cloud/bench/host_setup.sh", timeout=1800,
                 log_path=out / "setup.log")
        result["t_setup_s"] = round(time.time() - t0)
        log(f"setup done in {result['t_setup_s']} s; starting worker {a.id}")
        threads = a.threads if a.threads > 0 else worker_threads(result["offer"])  # nproc は割り当てより多く見えるのでオファーの実効コア数
        result["threads"] = threads
        log(f"worker threads {threads} (cores {result['offer'].get('cpu_cores_effective')})")
        script = "host_scale.sh" if a.job == "scale" else "host_worker.sh"
        host.ssh(f"bash /root/libra/libra-cloud/bench/{script} {threads} {a.n_games} {a.id} {run_id}", timeout=120, log_path=out / "setup.log")
        if stopping:
            raise KeyboardInterrupt
        env = dict(os.environ, PYTHONPATH=PYTHONPATH)
        if a.job == "scale":
            cmd = ["-m", "libra_cloud.scale_bridge", "--scale-dir", str(run_dir)]
        else:
            cmd = ["-m", "libra_cloud.bridge", "--run-dir", str(run_dir)]
        bridge = subprocess.Popen([str(PROJECT_PY), *cmd, "--host", host.host, "--port", str(host.port), "--out", str(out / "bridge"),
                                   "--hours", str(a.hours)], env=env)
        write_json(out / "instance.json", {"instance": iid, "offer": result["offer"], "t_rent": t_rent, "t_bridge": time.time(),
                                           "host": f"{host.host}:{host.port}"})
        log(f"bridge pid {bridge.pid} for {a.hours} h (stop: touch {out / 'bridge' / 'STOP'})")
        from libra_cloud.bench import instance_lost

        def status_text(inst) -> str:
            return f"{(inst or {}).get('actual_status')} / {(inst or {}).get('intended_status')}" if inst else "gone"

        lost_polls = 0
        while True:  # ブリッジが終わるのを待ちながら、インスタンスを失っていないか（入札で止められた・ホストが落ちた）を 60 秒ごとに見る
            try:
                rc = bridge.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                pass
            except KeyboardInterrupt:  # ブリッジに SIGTERM を送った後も終わるまで待つ
                continue
            if stopping or result.get("lost"):
                continue
            try:
                inst = v.show_instance(iid)
            except Exception as e:  # noqa: BLE001  API の一時的な失敗は失ったとみなさない
                log(f"show_instance {iid} failed: {type(e).__name__}: {str(e)[:120]}")
                continue
            lost_polls = lost_polls + 1 if instance_lost(inst) else 0
            if lost_polls >= 2:
                result["lost"] = True
                log(f"instance {iid} lost: {status_text(inst)}")
                bridge.send_signal(signal.SIGTERM)  # ワーカーには届かないので、ブリッジは回収を試してから抜ける
        log(f"bridge exited rc={rc}")
        if rc == 3 and not stopping and not result.get("lost"):  # ssh が続けて失敗した: ホストを失ったかを確かめる
            try:
                inst = v.show_instance(iid)
                if instance_lost(inst):
                    result["lost"] = True
                    log(f"instance {iid} lost: {status_text(inst)}")
            except Exception as e:  # noqa: BLE001
                log(f"show_instance {iid} failed: {type(e).__name__}: {str(e)[:120]}")
        if not result.get("lost"):
            try:
                host.get("/root/out", out, timeout=120)
            except Exception as e:  # noqa: BLE001
                log(f"fetch /root/out failed: {type(e).__name__}: {str(e)[:120]}")
        return rc
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    finally:
        if bridge is not None and bridge.poll() is None:
            bridge.kill()
        if iid is not None:
            for _ in range(5):
                try:
                    v.destroy_instance(iid)
                    break
                except Exception as e:  # noqa: BLE001
                    log(f"destroy retry: {e}")
                    time.sleep(5)
            gone = v.show_instance(iid)
            log(f"destroyed instance {iid} (show_instance after: {'none' if not gone else gone.get('actual_status')})")
        hours = (time.time() - t_rent) / 3600
        if "offer" in result:
            result["rented_h"] = round(hours, 3)
            result["est_cost_usd"] = round(hours * float(result["offer"].get("dph_eff") or result["offer"]["dph_total"]), 3)
        bj = out / "bridge" / "bridge.json"
        if bj.exists():
            result["bridge"] = json.loads(bj.read_text(encoding="utf-8"))
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
        log(f"result {out / 'result.json'} est cost ${result.get('est_cost_usd', 0)}")
        # ホストを失った（入札で止められた・落ちた）なら、残りの時間で次のセッションを起動する（ユーザーの決定「残り時間で借り直す」）。
        # 停止（STOP・シグナル）なら借り直さない。残りの時間と締め切りの判定は bin/libra-vast start --continue-from が行う
        if a.job == "selfplay" and result.get("lost") and not stopping and not (out / "bridge" / "STOP").exists():
            try:
                p = subprocess.run([str(REPO / "bin" / "libra-vast"), "--root", str(out.parent), "start", "--continue-from", out.name,
                                    "--run-root", str(run_dir.parent)], capture_output=True, text=True, timeout=120)
                msg = (p.stdout + p.stderr).strip().replace("\n", " ")[:300]
                if p.returncode == 0:
                    nxt = (out.parent / "current").read_text(encoding="utf-8").strip()
                    log(f"relaunched: {nxt} ({msg})")
                else:
                    log(f"relaunch refused rc={p.returncode}: {msg}")
            except Exception as e:  # noqa: BLE001
                log(f"relaunch failed: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    sys.exit(main())
