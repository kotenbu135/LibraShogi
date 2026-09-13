# SPDX-License-Identifier: Apache-2.0
"""vast.ai で GPU を 1 台借りて、今のワーカーの局/日を実測する。終わったら（失敗しても、Ctrl+C でも）必ずインスタンスを消す。

    ~/.venvs/vastai/bin/python libra-cloud/vast_bench.py --gpu "RTX 3090" --max-dph 0.25 --minutes 20 --out <dir>

前もって手元で束を作る（libra_cloud.prepare、プロジェクトの .venv）。API キーは ~/.config/vastai/vast_api_key（リポジトリの外）、
SSH 鍵は ~/.ssh/id_ed25519_vast。オンデマンドで借りる（割込みで計測が途切れないように）。
"""
from __future__ import annotations

import argparse
import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libra_cloud.bench import pick_offers  # noqa: E402

# 手元と同じ torch 2.11.0+cu128。vast.ai の公式イメージはホストに取得済みのことが多い
# （pytorch/pytorch の 4.3 GB は 9/14 に 2 台で 15 分以上取得が終わらなかった）
IMAGE = "vastai/pytorch:2.11.0-cu128-cuda-12.9-mini-py312-2026-09-08"
KEY = Path.home() / ".ssh" / "id_ed25519_vast"
# vastai/pytorch では sshd が "bad ownership or modes for file /root/.ssh/authorized_keys" で鍵を拒んだ（9/14）。
# 起動時に /root と /root/.ssh の所有者と権限を正す（鍵が書かれるのが後になっても効くよう 2 分間繰り返す）
ONSTART = ("(for i in $(seq 60); do chown root:root /root /root/.ssh /root/.ssh/authorized_keys 2>/dev/null; "
           "chmod go-w /root 2>/dev/null; chmod 700 /root/.ssh 2>/dev/null; chmod 600 /root/.ssh/authorized_keys 2>/dev/null; "
           "sleep 2; done) &")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


class Host:
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port

    def _opts(self, flag: str) -> list[str]:
        return ["-i", str(KEY), flag, str(self.port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]

    def ssh(self, cmd: str, timeout: float, log_path: Path | None = None, check: bool = True) -> int:
        argv = ["ssh", *self._opts("-p"), f"root@{self.host}", cmd]
        if log_path is None:
            p = subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        else:
            with open(log_path, "ab") as out:
                p = subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
        if check and p.returncode != 0:
            raise RuntimeError(f"ssh failed rc={p.returncode}: {cmd[:80]}")
        return p.returncode

    def put(self, src: Path, dst: str, timeout: float = 900) -> None:
        subprocess.run(["scp", *self._opts("-P"), str(src), f"root@{self.host}:{dst}"], check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL)

    def get(self, src: str, dst: Path, timeout: float = 300) -> None:
        subprocess.run(["scp", "-r", *self._opts("-P"), f"root@{self.host}:{src}", str(dst)], check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL)


def wait_ssh(v, iid: int, timeout: float, stall: float = 480.0, ssh_fail: float = 300.0) -> Host:
    """ssh で入れるまで待つ。状態（イメージ取得の進み具合など）が stall 秒変わらないか、running になってから
    ssh_fail 秒入れなければ（鍵が拒まれるなど）見切って TimeoutError。"""
    end = time.monotonic() + timeout
    last = ""
    changed = time.monotonic()
    running_since: float | None = None
    while time.monotonic() < end:
        inst = v.show_instance(iid) or {}
        st = f"{inst.get('actual_status')} / {inst.get('status_msg') or ''}".strip()
        if st != last:
            log(f"instance {iid}: {st[:160]}")
            last = st
            changed = time.monotonic()
        elif inst.get("actual_status") != "running" and time.monotonic() - changed > stall:
            raise TimeoutError(f"instance {iid} stalled for {stall:.0f} s: {st[:120]}")
        if inst.get("actual_status") == "running" and inst.get("ssh_host") and inst.get("ssh_port"):
            running_since = running_since if running_since is not None else time.monotonic()
            h = Host(inst["ssh_host"], int(inst["ssh_port"]))
            try:
                if h.ssh("true", timeout=40, check=False) == 0:
                    return h
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() - running_since > ssh_fail:
                raise TimeoutError(f"instance {iid} running but ssh failed for {ssh_fail:.0f} s (see v.logs for sshd errors)")
        time.sleep(15)
    raise TimeoutError(f"instance {iid} not reachable in {timeout:.0f} s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, help='例: "RTX 3090"')
    ap.add_argument("--max-dph", type=float, required=True, help="1 時間あたりの上限（$、ストレージ込み）")
    ap.add_argument("--minutes", type=int, default=20, help="ワーカーを回す時間")
    ap.add_argument("--warmup", type=int, default=300, help="局/日の窓から外す起動直後の秒数")
    ap.add_argument("--n-games", type=int, default=512)
    ap.add_argument("--threads", type=int, default=0, help="0 ならホストの CPU 数（12 まで）")
    ap.add_argument("--bundle", required=True, help="libra_cloud.prepare で作った bundle.tar.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--disk", type=float, default=30)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--min-credit", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true", help="オファーを選ぶだけで借りない")
    a = ap.parse_args()
    from vastai.sdk import VastAI

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = VastAI(raw=True, quiet=True)
    credit = float((v.show_user() or {}).get("credit") or 0)
    log(f"credit ${credit:.2f}")
    if credit < a.min_credit:
        log(f"credit below ${a.min_credit:.2f}; not renting")
        return 2
    q = f"gpu_name={a.gpu.replace(' ', '_')} num_gpus=1 rentable=true verified=true reliability>0.98 inet_down>=200 cuda_max_good>=12.8 disk_space>={a.disk}"
    offers = v.search_offers(query=q, type="on-demand", order="dph_total", limit=100, storage=a.disk) or []
    cands = pick_offers(offers, max_dph=a.max_dph)
    log(f"{len(offers)} offers, {len(cands)} usable; cheapest: "
        + ", ".join(f"#{o['id']} ${o['dph_total']:.3f}/h cpu {o.get('cpu_cores_effective')} {o.get('geolocation', '')}" for o in cands[:3]))
    if a.dry_run or not cands:
        return 0 if cands else 3
    pub = KEY.with_suffix(".pub").read_text().strip()
    try:
        v.create_ssh_key(pub)  # 既に登録済みならエラーが返るだけ
    except Exception as e:  # noqa: BLE001
        log(f"ssh key register: {type(e).__name__}: {str(e)[:120]}")

    iid = None
    stopping = []

    def on_signal(signum, frame):
        stopping.append(signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)
    t_rent = time.time()
    result: dict = {"gpu": a.gpu, "image": a.image, "minutes": a.minutes, "n_games": a.n_games}
    try:
        for offer in cands[:3]:
            r = v.create_instance(offer["id"], image=a.image, disk=a.disk, label="libra-bench", ssh=True, direct=True, cancel_unavail=True,
                                  onstart_cmd=ONSTART)
            iid = (r or {}).get("new_contract")
            log(f"create #{offer['id']} ${offer['dph_total']:.3f}/h -> instance {iid} {'' if iid else str(r)[:200]}")
            if not iid:
                continue
            result["offer"] = {k: offer.get(k) for k in ("id", "dph_total", "gpu_name", "cpu_name", "cpu_cores_effective", "cpu_ram",
                                                        "inet_up", "inet_down", "inet_up_cost", "inet_down_cost", "reliability2",
                                                        "geolocation", "cuda_max_good", "pcie_bandwidth", "gpu_mem_bw")}
            t_rent = time.time()
            try:
                v.attach_ssh(iid, pub)
                host = wait_ssh(v, iid, timeout=1200)  # イメージ 4.3 GB の取得が遅いホストで 15 分を超えた
                break
            except Exception as e:  # noqa: BLE001  起動しないホストは消して次へ
                log(f"instance {iid} failed to start: {type(e).__name__}: {str(e)[:200]}")
                v.destroy_instance(iid)
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
        log(f"setup done in {result['t_setup_s']} s; running worker {a.minutes} min")
        threads = a.threads or 0
        cmd = (f"T={threads}; [ $T -gt 0 ] || T=$(( $(nproc) < 12 ? $(nproc) : 12 )); "
               f"bash /root/libra/libra-cloud/bench/host_bench.sh {a.minutes} $T {a.n_games} {a.warmup}")
        host.ssh(cmd, timeout=a.minutes * 60 + 900, log_path=out / "bench.log")
        host.get("/root/out", out)
        rep = json.loads((out / "out" / "report.json").read_text())
        result["report"] = rep
        log(f"report: {json.dumps(rep)}")
        return 0
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    finally:
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
            result["est_cost_usd"] = round(hours * float(result["offer"]["dph_total"]), 3)
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
        log(f"result {out / 'result.json'} est cost ${result.get('est_cost_usd', 0)}")


if __name__ == "__main__":
    sys.exit(main())
