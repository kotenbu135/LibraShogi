# 貢献のしかた

## クリーンルーム方針（最重要）

LibraShogi は天秤将棋の AI をゼロから作るプロジェクトで、コードを Apache-2.0 で公開する。次のものは**一切**持ち込まない。

- 既存の将棋 AI（やねうら王、Apery、dlshogi、Lc0、水匠、Stockfish、fuseki-shogi-ai、fuseki-shogi-web など）のコード断片。合法手生成・評価関数・探索ルーチン・定跡・ビットボード表を含む
- それらの評価関数・学習済み重み・棋譜・評価値・読み筋（学習信号としても使わない）
- GPL / AGPL / SSPL などコピーレフトのコード（ライブラリとしてのリンクも不可。別プロセスで USI 経由で動かすのは可）
- NVIDIA のバイナリ（TensorRT の `.so`、`.engine`）
- 秘密情報（vast.ai の API キー、SSH 鍵、ホストの IP）

手法・論文の参照は自由。KataGo（MIT）のコードを参考にした場合は `NOTICE` に著作権表示を足す。
依存を追加するときは `LICENSES/README.md` に 1 行足し、全文を `LICENSES/` に置く。

外部から取得したファイルをリポジトリに入れないのが原則（依存物の取得は `tools/` のスクリプトでハッシュを固定し、
`third_party/` は gitignore する）。唯一の例外が `LICENSES/` に置くライセンスの全文で、法文は改変せずそのまま
配る必要があるため、取得元 URL と SHA-256 を `LICENSES/README.md` に記したうえでコミットする。

## DCO（Developer Certificate of Origin）

コミットには `git commit -s` で Signed-off-by を付ける。これは https://developercertificate.org/ の DCO 1.1 に同意したことを意味する。
CLA は無い。将来ライセンスを変える（例: Apache-2.0 → 別ライセンス）には全貢献者の同意が要る。

## 進めかた

- 1 タスク 1 ブランチ。テストが通ってから `main` へマージ
- テストを先に書く。C++ と Python のテストは変更のたびに全部通す（コマンドは [docs/getting-started.md](docs/getting-started.md) §3、CI と同じ）
- コミットは小さく、メッセージは「何を・なぜ」
- 決定は `docs/decisions.md` に 1 行、実測値は `docs/measurements.md` に追記
- SPDX 識別子をソースのヘッダに付ける: `// SPDX-License-Identifier: Apache-2.0`（シェル・Python・CMake は `#`）
- 文書の置き場: 手順は docs/getting-started.md（入門）と docs/runbook.md（運用）、GUI との接続は docs/protocol.md、ルールは docs/rules.md。計画書（docs/libra-design.md・docs/libra-local.md）は書き換えず、差異は decisions.md に書く

## 質問・不具合・提案

[Issues](https://github.com/kotenbu135/LibraShogi/issues) へ。ルールの解釈は docs/rules.md の節を示す（天秤将棋のルールそのものの変更はルール設計者の判断）。
手法や設定値の変更の提案は、採用の経緯（docs/decisions.md、docs/method-evidence.md）を読んだうえで、根拠（文献・計測）を添える。

## Pull Request

外部からの PR は [.github/pull_request_template.md](.github/pull_request_template.md) のチェック項目を埋める。
メンテナ自身の作業は PR を作らず、ブランチを `main` に早送りでマージする（単独開発のため）。
