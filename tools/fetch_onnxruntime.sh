#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ONNX Runtime（MIT）の公式バイナリを third_party/onnxruntime/ に取得する（リポジトリには入れない）。
# 使い方: tools/fetch_onnxruntime.sh [linux-gpu|linux-cpu|win|win-dml|win-cuda]...   （省略時 linux-gpu と win-dml）
#   win-dml : Windows 配布物の既定。NuGet の Microsoft.ML.OnnxRuntime.DirectML と Microsoft.AI.DirectML を
#             include/ と lib/（onnxruntime.dll・DirectML.dll と各ライセンス）の形に並べ直す
#   win-cuda: NVIDIA の GPU 向けに差し替える DLL（CUDA 12・cuDNN 9 は含まない）
#   win     : CPU だけの版
# バージョンとハッシュはここで固定する。変えるときは LICENSES/README.md も更新する。
set -euo pipefail
V=1.30.0
V_DML=1.24.4        # DirectML 版の ONNX Runtime はこの版で更新が止まっている
V_DIRECTML=1.15.4
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/onnxruntime"
mkdir -p "$DEST"
GH="https://github.com/microsoft/onnxruntime/releases/download/v$V"
NUGET="https://api.nuget.org/v3-flatcontainer"
declare -A URL=(
  [linux-gpu]="$GH/onnxruntime-linux-x64-gpu_cuda12-$V.tgz"
  [linux-cpu]="$GH/onnxruntime-linux-x64-$V.tgz"
  [win]="$GH/onnxruntime-win-x64-$V.zip"
  [win-cuda]="$GH/onnxruntime-win-x64-gpu_cuda12-$V.zip"
  [win-dml]="$NUGET/microsoft.ml.onnxruntime.directml/$V_DML/microsoft.ml.onnxruntime.directml.$V_DML.nupkg"
  [directml]="$NUGET/microsoft.ai.directml/$V_DIRECTML/microsoft.ai.directml.$V_DIRECTML.nupkg"
)
declare -A SHA=(
  [linux-gpu]="f9886932ee7bb0b4d3fcab736a392d4ff5efaa0672b47f19f0cec03437cf64f1"
  [linux-cpu]="a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd"
  [win]="c6ba983baf5681af108599675d2a89c2d145512d02de28aed0bff177cd0ba949"
  [win-cuda]="d4667ea48eb0a10bc9b96b838f7b8975a6bf18de3bc5edd403a22e15c1458b23"
  [win-dml]="57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681"
  [directml]="4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9"
)

fetch() {  # fetch <target> → 検証済みのファイルのパスを出す
  local t="$1" f
  f="$DEST/$(basename "${URL[$t]}")"
  [ -f "$f" ] || curl -sSL -o "$f" "${URL[$t]}"
  local got
  got=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$got" != "${SHA[$t]}" ]; then echo "sha256 mismatch for $f: $got" >&2; exit 1; fi
  echo "$(basename "$f") sha256 $got" >&2
  echo "$f"
}

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(linux-gpu win-dml)
for t in "${targets[@]}"; do
  if [ "$t" = win-dml ]; then
    dir="$DEST/onnxruntime-win-x64-directml-$V_DML"
    if [ -f "$dir/lib/DirectML.dll" ]; then echo "ok: $dir"; continue; fi
    ort=$(fetch win-dml)
    dml=$(fetch directml)
    tmp=$(mktemp -d)
    unzip -q -o "$ort" -d "$tmp/ort"
    unzip -q -o "$dml" -d "$tmp/dml"
    mkdir -p "$dir/include" "$dir/lib"
    cp "$tmp"/ort/build/native/include/*.h "$dir/include/"
    cp "$tmp"/ort/runtimes/win-x64/native/onnxruntime.dll "$tmp"/ort/runtimes/win-x64/native/onnxruntime_providers_shared.dll "$dir/lib/"
    cp "$tmp"/dml/bin/x64-win/DirectML.dll "$dir/lib/"  # DirectML.Debug.dll は配らない
    cp "$tmp"/ort/LICENSE "$dir/LICENSE-onnxruntime.txt"
    cp "$tmp"/ort/ThirdPartyNotices.txt "$dir/ThirdPartyNotices-onnxruntime.txt"
    cp "$tmp"/dml/LICENSE.txt "$dir/LICENSE-DirectML.txt"
    cp "$tmp"/dml/ThirdPartyNotices.txt "$dir/ThirdPartyNotices-DirectML.txt"
    rm -rf "$tmp"
    echo "ok: $dir"
    continue
  fi
  [ -n "${URL[$t]:-}" ] && [ "$t" != directml ] || { echo "unknown target: $t" >&2; exit 2; }
  f="$(basename "${URL[$t]}")"
  dir="$DEST/${f%.tgz}"; dir="${dir%.zip}"
  if [ -d "$dir/include" ]; then echo "ok: $dir"; continue; fi
  path=$(fetch "$t")
  case "$f" in *.tgz) tar xzf "$path" -C "$DEST" ;; *.zip) unzip -q -o "$path" -d "$DEST" ;; esac
  echo "ok: $dir"
done
