#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# vast.ai のホストで libra-scale seq worker（全組の検証対局）を常駐させる。active.json の受け取りと棋譜の回収は手元の libra_cloud.scale_bridge が行う。
# 使い方: host_scale.sh THREADS N_GAMES WORKER_ID NAME   （束の scale/NAME を使う）ログは /root/out/worker.log、pid は /root/out/worker.pid、GPU は /root/out/gpu.csv
set -euo pipefail
cd "$(dirname "$0")/../.."
THREADS=${1:-12}
N_GAMES=${2:-512}
ID=${3:-vast1}
NAME=${4:?scale dir name}
DIR=$(pwd)/scale/$NAME
PY=$(cat /root/python_path)  # host_setup.sh が選んだ Python
export PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-scale:libra-cloud
export OMP_NUM_THREADS=4
mkdir -p /root/out "$DIR/inbox"
nohup bash -c 'while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,memory.used --format=csv,noheader >> /root/out/gpu.csv; sleep 60; done' \
  > /dev/null 2>&1 < /dev/null &
nohup "$PY" -m libra_scale.cli seq worker --dir "$DIR" --id "$ID" --threads "$THREADS" --n-games "$N_GAMES" \
  > /root/out/worker.log 2>&1 < /dev/null &
echo $! > /root/out/worker.pid
echo "worker pid $(cat /root/out/worker.pid)"
