# 依存物のライセンス台帳

依存を追加するたびにここへ 1 行足し、ライセンス全文をこのディレクトリに置く（ファイル名は `<name>-<LICENSE>.txt`）。
本リポジトリ自身の成果物（コード・文書・データ・重み）のライセンス全文もここに置く。
Creative Commons の全文は外部から取得した唯一のファイルで、「外部のファイルをリポジトリに入れない」方針（CONTRIBUTING.md）の明示した例外とする
（法文は改変せずそのまま置く必要があるため。取得元 URL と SHA-256 を下の表に記す。docs/decisions.md 2026-09-16）。
GPL 系のコードは一行も入れない（docs/libra-design.md §7.3〜7.4、CONTRIBUTING.md）。

| 依存物 | ライセンス | 用途 | 全文 | 備考 |
|---|---|---|---|---|
| Apache License 2.0（本リポジトリのコード） | Apache-2.0 | — | `../LICENSE` | |
| 本リポジトリの文書（`docs/` 配下、`*.md`） | CC BY 4.0 | — | `CC-BY-4.0.txt` | 全文は https://creativecommons.org/licenses/by/4.0/legalcode.txt から取得（2026-09-16、SHA-256 `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411`） |
| 自己対局データ・玉配置表（棋譜 JSONL、`scale.json`） | CC0 1.0 | — | `CC0-1.0.txt` | 全文は https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt から取得（2026-09-16、SHA-256 `a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499`） |
| pybind11 | BSD-3-Clause | libra-sim の Python バインディング | （追加時に置く） | |
| pytest | MIT | テスト | （追加時に置く） | 配布物に含めない |
| pytest-xdist | MIT | CI のテストを 2 並列で回す | （追加時に置く） | 配布物に含めない。CI だけで使う |
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
