#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Windows 側の操作ファイル（bat / vbs / 管理コンソール）を %USERPROFILE%\libra とデスクトップへ置く。
# WSL から実行する。ファイルはすべて ASCII か UTF-8（BOM 付き .ps1）で、日本語は .ps1 の中だけ。
#
# bat / vbs は同じディレクトリの *.in をひな形にして、この環境の値を埋めて作る:
#   @DISTRO_ARG@ … wsl.exe に渡すディストロの指定（`-d <名前>`。既定のディストロなら空）
#   @LIBRA@      … WSL 内の bin/libra の絶対パス
# libra-console.ps1 は書き換えず、同じ場所に libra-paths.json（この環境の値）を書いて読ませる。
#
# 使い方: tools/windows/install.sh [Windows のユーザー名]
# 環境変数で上書きできる: LIBRA_DISTRO（既定は $WSL_DISTRO_NAME）、LIBRA_RUN_ROOT（既定は ~/libra-run）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WINUSER="${1:-$(powershell.exe -NoProfile -Command '$env:USERNAME' | tr -d '\r')}"
DEST="/mnt/c/Users/$WINUSER/libra"
DESKTOP="$(powershell.exe -NoProfile -Command '[Console]::OutputEncoding=[Text.Encoding]::UTF8; [Environment]::GetFolderPath("Desktop")' | tr -d '\r' | sed 's#^C:#/mnt/c#; s#\\#/#g')"
LOCALAPPDATA_WIN="$(powershell.exe -NoProfile -Command '[Console]::OutputEncoding=[Text.Encoding]::UTF8; $env:LOCALAPPDATA' | tr -d '\r')"

DISTRO="${LIBRA_DISTRO:-${WSL_DISTRO_NAME:-}}"
RUN_ROOT="${LIBRA_RUN_ROOT:-$HOME/libra-run}"
LIBRA="$ROOT/bin/libra"
LIBRA_VAST="$ROOT/bin/libra-vast"
[ -x "$LIBRA" ] || { echo "not executable: $LIBRA" >&2; exit 1; }
# WSL 内のパスに空白があると wsl.exe のコマンド行（bat / vbs）で壊れる
case "$ROOT" in *[[:space:]]*) echo "リポジトリの場所に空白が含まれている: $ROOT" >&2; exit 1;; esac
DISTRO_ARG=""
[ -n "$DISTRO" ] && DISTRO_ARG="-d $DISTRO"

mkdir -p "$DEST"
for t in "$HERE"/*.in; do
  out="$DEST/$(basename "${t%.in}")"
  sed -e "s#@DISTRO_ARG@#$DISTRO_ARG#g" -e "s#@LIBRA@#$LIBRA#g" "$t" > "$out"
done
cp "$HERE/libra-console.bat" "$HERE/libra-console.ps1" "$DEST/"

# 管理コンソールが読む、この環境の値（ps1 の param() の既定より優先。コマンド行の -Libra などが最優先）
python3 - "$DEST/libra-paths.json" "$DISTRO" "$LIBRA" "$RUN_ROOT" "$LIBRA_VAST" "$LOCALAPPDATA_WIN" <<'PY'
import json, sys
out, distro, libra, run_root, libra_vast, localappdata = sys.argv[1:7]
cfg = {"Distro": distro, "Libra": libra, "RunRoot": run_root, "LibraVast": libra_vast}
if localappdata:
    cfg["DesktopExe"] = localappdata + r"\天秤将棋GUI\tenbin-shogi-gui.exe"
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
PY

for f in libra-console libra-stop libra-status; do cp "$DEST/$f.bat" "$DESKTOP/"; done
# 一時停止・再開は廃止した（docs/decisions.md 2026-09-14）。前に写した bat を残さない
rm -f "$DEST/libra-pause.bat" "$DEST/libra-resume.bat" "$DESKTOP/libra-pause.bat" "$DESKTOP/libra-resume.bat"
echo "installed to $DEST and $DESKTOP"
echo "  distro=${DISTRO:-（既定）}  libra=$LIBRA  run-root=$RUN_ROOT"
