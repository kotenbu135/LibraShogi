#!/usr/bin/env bash
# ONNX Runtime（MIT）の公式バイナリを third_party/onnxruntime/ に取得する（リポジトリには入れない）。
# 使い方: tools/fetch_onnxruntime.sh [linux-gpu|linux-cpu|win]...   （省略時 linux-gpu と win）
# バージョンとハッシュはここで固定する。変えるときは LICENSES/README.md も更新する。
set -euo pipefail
V=1.30.0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/onnxruntime"
mkdir -p "$DEST"
declare -A FILE=(
  [linux-gpu]="onnxruntime-linux-x64-gpu_cuda12-$V.tgz"
  [linux-cpu]="onnxruntime-linux-x64-$V.tgz"
  [win]="onnxruntime-win-x64-$V.zip"
)
declare -A SHA=(
  [linux-gpu]="f9886932ee7bb0b4d3fcab736a392d4ff5efaa0672b47f19f0cec03437cf64f1"
  [linux-cpu]="a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd"
  [win]="c6ba983baf5681af108599675d2a89c2d145512d02de28aed0bff177cd0ba949"
)
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(linux-gpu win)
for t in "${targets[@]}"; do
  f="${FILE[$t]}"
  dir="$DEST/${f%.tgz}"; dir="${dir%.zip}"
  if [ -d "$dir/include" ]; then echo "ok: $dir"; continue; fi
  [ -f "$DEST/$f" ] || curl -sSL -o "$DEST/$f" "https://github.com/microsoft/onnxruntime/releases/download/v$V/$f"
  got=$(sha256sum "$DEST/$f" | cut -d' ' -f1)
  if [ -n "${SHA[$t]}" ] && [ "$got" != "${SHA[$t]}" ]; then echo "sha256 mismatch for $f: $got" >&2; exit 1; fi
  echo "$f sha256 $got"
  case "$f" in *.tgz) tar xzf "$DEST/$f" -C "$DEST" ;; *.zip) unzip -q -o "$DEST/$f" -d "$DEST" ;; esac
  echo "ok: $dir"
done
