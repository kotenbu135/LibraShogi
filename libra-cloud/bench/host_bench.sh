#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# vast.ai のホストで libra worker（--detached）を MINUTES 分回し、inbox の対局ファイルから局/日を出す。
# 使い方: host_bench.sh MINUTES THREADS N_GAMES WARMUP_S   結果は /root/out（report.json、worker.log、gpu.csv）
set -euo pipefail
cd "$(dirname "$0")/../.."
MINUTES=${1:-20}
THREADS=${2:-12}
N_GAMES=${3:-512}
WARMUP=${4:-300}
RUN=/root/libra/run
OUT=/root/out
export PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-cloud
export OMP_NUM_THREADS=4
mkdir -p "$OUT"
rm -rf "$RUN/ls/inbox"
mkdir -p "$RUN/ls/inbox"
(while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,memory.used --format=csv,noheader >> "$OUT/gpu.csv"; sleep 30; done) &
SMI=$!
START=$(date +%s.%N)
echo "$START" > "$OUT/start"
# 終わりは SIGTERM（ワーカーは残りの局を書いてから抜ける）。抜けなければ 2 分後に KILL
timeout -s TERM --kill-after=120 "${MINUTES}m" \
  python -m libra_league.cli --root "$RUN" --run ls worker --id bench --detached --threads "$THREADS" --n-games "$N_GAMES" \
  > "$OUT/worker.log" 2>&1 || true
kill "$SMI" 2>/dev/null || true
python -m libra_cloud.bench report --inbox "$RUN/ls/inbox" --start "$START" --warmup "$WARMUP" --out "$OUT/report.json"
cp /root/host_info.json "$OUT/" 2>/dev/null || true
