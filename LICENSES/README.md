# 依存物のライセンス台帳

依存を追加するたびにここへ 1 行足し、ライセンス全文をこのディレクトリに置く（ファイル名は `<name>-<LICENSE>.txt`）。
GPL 系のコードは一行も入れない（docs/libra-design.md §7.3〜7.4、CONTRIBUTING.md）。

| 依存物 | ライセンス | 用途 | 全文 | 備考 |
|---|---|---|---|---|
| Apache License 2.0（本リポジトリのコード） | Apache-2.0 | — | `../LICENSE` | |
| pybind11 | BSD-3-Clause | libra-sim の Python バインディング | （追加時に置く） | |
| pytest | MIT | テスト | （追加時に置く） | 配布物に含めない |
| ONNX Runtime 1.30.0 | MIT | libra / libra.exe の推論（C API を実行時にロード。`tools/fetch_onnxruntime.sh` で公式バイナリを取得、SHA-256 固定） | `onnxruntime-MIT.txt` | Linux 版と、Windows で NVIDIA の GPU 向けに差し替える CUDA 版の DLL。CUDA・cuDNN は同梱しない |
| ONNX Runtime 1.24.4 DirectML 版（NuGet `Microsoft.ML.OnnxRuntime.DirectML`） | MIT | Windows 配布物の推論（`tools/fetch_onnxruntime.sh win-dml`、SHA-256 固定） | `onnxruntime-MIT.txt` | 配布物に `onnxruntime.dll`・`onnxruntime_providers_shared.dll` を同梱する。同梱物に NuGet の ThirdPartyNotices を添える |
| DirectML 1.15.4（NuGet `Microsoft.AI.DirectML`） | Microsoft Software License Terms（再配布可。Windows 向けのアプリの一部として配る。単体配布・リバースエンジニアリング・表示の削除は不可。テレメトリの条項あり） | Windows 配布物の DirectML EP | `DirectML-MSLT.txt` | 配布物に `DirectML.dll` を同梱する（`DirectML.Debug.dll` は配らない）。同梱物にこの全文と NuGet の ThirdPartyNotices を添える |
| onnx | Apache-2.0 | ONNX 書き出し時のメタデータ付与（libra-net） | `onnx-Apache-2.0.txt` | 配布物に含めない |

対戦相手（別プロセス・USI 経由でのみ使用。本リポジトリに同梱しない）:

| もの | ライセンス | 扱い |
|---|---|---|
| やねうら王 V9.00 / 水匠5 | GPL-3.0 / 水匠5 は配布条件を確認 | 計測ハーネスが外部プロセスとして起動するだけ。コード・評価値・棋譜を学習に使わない |
| fuseki-shogi-ai `scripts/fuseki_usi_server.py` | GPL-3.0（非公開） | 同上 |
| tenbin-shogi-desktop `public/wasm/fuseki.wasm`（dlshogi 由来） | GPL-3.0 | 合法手の黒箱テストの相手として実行するだけ。コードは読まない |
