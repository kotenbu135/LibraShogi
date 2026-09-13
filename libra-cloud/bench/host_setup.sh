#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# vast.ai のホスト（pytorch/pytorch の CUDA 12.8 イメージ、root）で束を展開した /root/libra の中で実行する。
# ワーカーに要る C++ モジュール（librashogi、librasearch）だけをビルドし、環境を記録する。
set -euo pipefail
cd "$(dirname "$0")/../.."
t0=$(date +%s)
# torch を読める Python を探す（vastai/pytorch は /venv/main、pytorch/pytorch は /opt/conda で PATH 上）
PY=""
for c in /venv/main/bin/python /opt/conda/bin/python python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch" 2>/dev/null; then PY=$(command -v "$c"); break; fi
done
[ -n "$PY" ] || { echo "no python with torch"; exit 1; }
echo "$PY" > /root/python_path
export PATH="$(dirname "$PY"):$PATH"
if ! command -v g++ >/dev/null; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++ make >/dev/null
fi
"$PY" -m pip install -q cmake ninja pybind11 numpy
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLIBRA_BUILD_ENGINE=OFF -DLIBRA_BUILD_TESTS=OFF \
  -DPython_EXECUTABLE="$PY" -Dpybind11_DIR="$("$PY" -c 'import pybind11;print(pybind11.get_cmake_dir())')" > /root/cmake.log
cmake --build build >> /root/cmake.log
export PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-cloud
"$PY" - <<'EOF'
import json, os, platform
import torch, librashogi, librasearch
info = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0), "python": platform.python_version(), "nproc": os.cpu_count(),
        "sched_cpus": len(os.sched_getaffinity(0))}
print(json.dumps(info))
open("/root/host_info.json", "w").write(json.dumps(info))
EOF
nvidia-smi --query-gpu=name,driver_version,power.limit,memory.total --format=csv,noheader
echo "setup done in $(( $(date +%s) - t0 )) s"
