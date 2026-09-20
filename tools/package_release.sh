#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# GitHub Release に載せるファイルを作る。WSL から実行する（GPU は使わない）。
#
#   tools/package_release.sh <版> <重みの置き場>   例: tools/package_release.sh v0.1 ~/libra-run/releases/v0.1
#
# 出す先は <重みの置き場>/dist/:
#   libra-<版>.onnx / libra-<版>.pt / scale-<版>.json … 重みの置き場から写す（.pt 以外は必須）
#   libra-<版>-windows-x64.zip                        … libra.exe ＋ DirectML 版の DLL ＋ モデル ＋ 表 ＋ ライセンス
#   libra-<版>-selfplay-sample.jsonl.gz               … 第 3 引数を渡したときだけ（CC0 の自己対局の標本）
#   SHA256SUMS                                        … dist/ の全ファイル
#
# 第 3 引数 <中心の時刻> を渡すと、run の games/ からその前後 1 時間の chunk を 1 本にまとめる。
#   例: tools/package_release.sh v0.1 ~/libra-run/releases/v0.1 '2026-09-15 12:25:25'
# 選別はしない（搾取者の布石から始まる局・リーグの局・Gumbel ノイズの手が混ざる。data/README.md）。
#
# Windows 版 libra.exe は先に libra-engine/README.md の mingw クロスビルドで build-win/ に作っておく。
set -euo pipefail
VER="${1:?使い方: tools/package_release.sh <版> <重みの置き場>}"
SRC="$(cd "${2:?使い方: tools/package_release.sh <版> <重みの置き場>}" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORT="$ROOT/third_party/onnxruntime/onnxruntime-win-x64-directml-1.24.4"
WIN="$ROOT/build-win/libra-engine"
DIST="$SRC/dist"

[ -f "$WIN/libra.exe" ] || { echo "先にクロスビルドする（libra-engine/README.md）: $WIN/libra.exe" >&2; exit 1; }
[ -d "$ORT" ] || { echo "先に tools/fetch_onnxruntime.sh win-dml を実行する: $ORT" >&2; exit 1; }
# zip だけで指せることが配布の条件なので、モデル・玉配置表・モデルカードが欠けたら作らない。
# 以前は「あるものだけ」写していたので、名前を間違えるとモデルの入っていない zip が黙って出来上がった。
[ -f "$SRC/libra-$VER.onnx" ] || { echo "モデルが無い: $SRC/libra-$VER.onnx（bin/libra export で書き出す。docs/release.md §1）" >&2; exit 1; }
[ -f "$SRC/scale-$VER.json" ] || { echo "玉配置表が無い: $SRC/scale-$VER.json（libra-scale で作り直す。docs/release.md §2）" >&2; exit 1; }
[ -f "$ROOT/docs/model-card-$VER.md" ] || { echo "モデルカードが無い: docs/model-card-$VER.md（docs/release.md §4）" >&2; exit 1; }

rm -rf "$DIST"; mkdir -p "$DIST"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
PKG="$STAGE/libra-$VER-windows-x64"; mkdir -p "$PKG/LICENSES"

cp "$WIN/libra.exe" "$PKG/"
for d in onnxruntime.dll onnxruntime_providers_shared.dll DirectML.dll; do cp "$ORT/lib/$d" "$PKG/"; done
# 配布物だけで指せるよう、モデルと玉配置表も同梱する（同じものを単体でも Release に置く）
SCALE="$SRC/scale-$VER.json"
cp "$SRC/libra-$VER.onnx" "$PKG/libra.onnx"
cp "$SCALE" "$PKG/scale.json"
# ライセンス（LICENSES/README.md の台帳のとおり）
cp "$ROOT/LICENSE" "$PKG/LICENSES/Apache-2.0.txt"
cp "$ROOT/NOTICE" "$PKG/LICENSES/NOTICE"
cp "$ROOT/LICENSES/README.md" "$PKG/LICENSES/"
cp "$ROOT/LICENSES/CC0-1.0.txt" "$ROOT/LICENSES/CC-BY-4.0.txt" "$PKG/LICENSES/"
cp "$ORT"/LICENSE-*.txt "$ORT"/ThirdPartyNotices-*.txt "$PKG/LICENSES/"
cp "$ROOT/docs/model-card-$VER.md" "$PKG/MODEL-CARD.md"

cat > "$PKG/README.txt" <<TXT
LibraShogi $VER - Windows x64 (DirectML)

天秤将棋 (Tenbin Shogi) の USI 拡張エンジン。天秤将棋対応の GUI に libra.exe を登録する。
同じフォルダの libra.onnx をモデルとして読む (setoption name DNN_Model で変えられる)。
1-2 手目の両玉は scale.json (玉配置表) から置く (setoption name Scale_Table)。

実行プロバイダ (DNN_Provider, 既定 auto) は同梱の DLL で DirectML になる。
NVIDIA の GPU では CUDA 版の onnxruntime.dll に差し替えると速い (CUDA 12 / cuDNN 9 が別に要る)。

モデルの中身・学習・計測・既知の限界は MODEL-CARD.md。
ライセンス: コードと重みは Apache-2.0、玉配置表は CC0 1.0、同梱の DLL は LICENSES/ を見る。
https://github.com/kotenbu135/LibraShogi
TXT

# 同じ中身なら同じハッシュになるよう、名前順・固定の日時で詰める（zip は既定で mtime を埋め込む）
python3 - "$STAGE/libra-$VER-windows-x64" "$DIST/libra-$VER-windows-x64.zip" <<'PY'
import os, sys, zipfile
root, out = sys.argv[1:3]
base = os.path.basename(root)
paths = sorted(os.path.relpath(os.path.join(dp, f), root)
               for dp, _, fs in os.walk(root) for f in fs)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for rel in paths:
        info = zipfile.ZipInfo(f"{base}/{rel.replace(os.sep, '/')}", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        with open(os.path.join(root, rel), "rb") as f:
            z.writestr(info, f.read())
print(f"zip: {len(paths)} files, {os.path.getsize(out)/1e6:.1f} MB")
PY
for f in "$SRC/libra-$VER.onnx" "$SRC/libra-$VER.pt" "$SCALE"; do [ -f "$f" ] && cp "$f" "$DIST/"; done
if [ "${3:-}" != "" ]; then
  # 既定は <重みの置き場>/../../<run>/games。別の場所なら LIBRA_GAMES_DIR で渡す
  GAMES="${LIBRA_GAMES_DIR:-$(dirname "$(dirname "$SRC")")/${LIBRA_RUN_ID:-ls}/games}"
  [ -d "$GAMES" ] || { echo "自己対局の棋譜が見つかりません: $GAMES（LIBRA_GAMES_DIR で指す）" >&2; exit 1; }
  python3 - "$GAMES" "$3" "$DIST/libra-$VER-selfplay-sample.jsonl.gz" <<'PY'
import datetime, glob, gzip, io, os, sys
games, center, out = sys.argv[1:4]
t = datetime.datetime.strptime(center, "%Y-%m-%d %H:%M:%S").timestamp()
fs = sorted(f for f in glob.glob(os.path.join(games, "games_*.jsonl")) if abs(os.path.getmtime(f) - t) < 3600)
n = 0
# gzip はヘッダに mtime を書くので、同じ中身なら同じハッシュになるよう 0 に固定する
with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
     io.TextIOWrapper(gz, encoding="utf-8") as w:
    for f in fs:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                w.write(line); n += 1
print(f"selfplay sample: {len(fs)} chunks, {n} games, {os.path.getsize(out)/1e6:.1f} MB "
      f"({os.path.basename(fs[0])}..{os.path.basename(fs[-1])})")
PY
fi
( cd "$DIST" && sha256sum * > SHA256SUMS )
echo "→ $DIST"; ls -la "$DIST"
