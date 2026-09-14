#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# vast.ai のホストで libra worker（--detached）を常駐させる。重みの受け取りと対局ファイルの回収は手元のブリッジ（libra_cloud.bridge）が行う。
# 使い方: host_worker.sh THREADS N_GAMES WORKER_ID [RUN_ID]   ログは /root/out/worker.log、pid は /root/out/worker.pid、GPU は /root/out/gpu.csv
set -euo pipefail
cd "$(dirname "$0")/../.."
THREADS=${1:-12}
N_GAMES=${2:-512}
ID=${3:-vast1}
RUN_ID=${4:-ls}
RUN=$(pwd)/run
PY=$(cat /root/python_path)  # host_setup.sh が選んだ Python
export PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-cloud
export OMP_NUM_THREADS=4
mkdir -p /root/out "$RUN/$RUN_ID/inbox"
nohup bash -c 'while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,memory.used --format=csv,noheader >> /root/out/gpu.csv; sleep 60; done' \
  > /dev/null 2>&1 < /dev/null &
nohup "$PY" -m libra_league.cli --root "$RUN" --run "$RUN_ID" worker --id "$ID" --detached --threads "$THREADS" --n-games "$N_GAMES" \
  > /root/out/worker.log 2>&1 < /dev/null &
echo $! > /root/out/worker.pid
echo "worker pid $(cat /root/out/worker.pid)"
