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
for f in libra-console libra-stop libra-status; do cp "$HERE/$f.bat" "$DESKTOP/"; done
# 一時停止・再開は廃止した（docs/decisions.md 2026-09-14）。前に写した bat を残さない
rm -f "$DEST/libra-pause.bat" "$DEST/libra-resume.bat" "$DESKTOP/libra-pause.bat" "$DESKTOP/libra-resume.bat"
echo "installed to $DEST and $DESKTOP"
