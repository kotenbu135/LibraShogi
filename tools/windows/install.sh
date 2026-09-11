#!/usr/bin/env bash
# Windows 側の操作ファイル（bat / vbs / 管理コンソール）を C:\Users\<user>\libra とデスクトップへ写す。
# WSL から実行する。ファイルはすべて ASCII か UTF-8（BOM 付き .ps1）で、日本語は .ps1 の中だけ。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WINUSER="${1:-$(powershell.exe -NoProfile -Command '$env:USERNAME' | tr -d '\r')}"
DEST="/mnt/c/Users/$WINUSER/libra"
DESKTOP="$(powershell.exe -NoProfile -Command '[Console]::OutputEncoding=[Text.Encoding]::UTF8; [Environment]::GetFolderPath("Desktop")' | tr -d '\r' | sed 's#^C:#/mnt/c#; s#\\#/#g')"
mkdir -p "$DEST"
cp "$HERE"/*.bat "$HERE"/*.vbs "$HERE"/libra-console.ps1 "$DEST/"
for f in libra-console libra-pause libra-resume libra-stop libra-status; do cp "$HERE/$f.bat" "$DESKTOP/"; done
echo "installed to $DEST and $DESKTOP"
