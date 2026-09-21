# LibraShogi

天秤将棋（[ルール](https://tenbinshogi.com/rules/)）の AI「Libra」。**自己対局だけで学習**し、既存の将棋 AI のコード・評価関数・重み・棋譜・評価値を内部にも学習信号にも使っていない。
コードは Apache-2.0、文書は CC BY 4.0、自己対局データと玉配置表は CC0。天秤将棋の AI を作る後続の人が、ルールの仕様・シミュレータ・対局ハーネス・学習の一式をそのまま使えるように公開している。

*LibraShogi is an AI for Tenbin Shogi (a shogi variant with a 40-ply piece-placement phase and a king-placement "balance" opening), trained from self-play only, with no code, weights or game records from existing shogi engines. Code is Apache-2.0, documents CC BY 4.0, self-play data CC0. Documentation is in Japanese; the rules specification ([docs/rules.md](docs/rules.md)) and the USI protocol extension ([docs/protocol.md](docs/protocol.md)) are the places to start for another implementation.*

## 天秤将棋とは

将棋の変種。1〜2 手目に両玉を置き（置く側）、もう一方が先手か後手かを選ぶ（選ぶ側）。3〜40 手目で残り 19 枚ずつを自陣に打って陣を作り（布石）、41 手目から本将棋を指す。
Libra の規定は [docs/rules.md](docs/rules.md) が唯一の正（布石の禁じ手、41 手目の裁定、終局規定は世界コンピュータ将棋選手権の大会ルールに揃える）。

## こんな人へ

| やりたいこと | 入口 |
|---|---|
| Libra と指す（ビルドしない） | [Releases](https://github.com/kotenbu135/LibraShogi/releases) の Windows 版 zip を GUI [tenbin-shogi-desktop](https://github.com/kotenbu135/tenbin-shogi-desktop) に登録する。手順は [docs/getting-started.md](docs/getting-started.md) §5 |
| ソースからビルドしてエンジンを動かす | [docs/getting-started.md](docs/getting-started.md) §1〜§4 |
| 自己対局と学習を回す（v0.1 の続きも） | [docs/getting-started.md](docs/getting-started.md) §6 → 運用は [docs/runbook.md](docs/runbook.md) |
| 天秤将棋の AI を自分で作る | [docs/rules.md](docs/rules.md)（perft 値つき）、[libra-sim](libra-sim/README.md)（シミュレータ）、[docs/protocol.md](docs/protocol.md)（GUI との接続）、対局ハーネス `bin/libra match`。まとめは [docs/getting-started.md](docs/getting-started.md) §7 |
| 設計と経緯を知る | 下の「文書」の読む順 |
| 貢献する | [CONTRIBUTING.md](CONTRIBUTING.md)（クリーンルーム方針、DCO） |

## クイックスタート

Ubuntu 24.04、g++ 13、Python 3.12 で確認。GPU は任意。apt を使わず venv の pip で揃える。詳細と Windows 版は [docs/getting-started.md](docs/getting-started.md)。

```bash
python3 -m venv .venv
.venv/bin/pip install cmake ninja pybind11 pytest numpy onnx onnxruntime
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128   # CPU だけなら https://download.pytorch.org/whl/cpu
tools/fetch_onnxruntime.sh linux-gpu    # ONNX Runtime（MIT）を third_party/ に取得。CPU だけなら linux-cpu
export PATH=$PWD/.venv/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python -c "import pybind11;print(pybind11.get_cmake_dir())") \
  -DLIBRA_ORT_DIR=$PWD/third_party/onnxruntime/onnxruntime-linux-x64-gpu_cuda12-1.30.0   # CPU だけなら onnxruntime-linux-x64-1.30.0
cmake --build build
ctest --test-dir build/libra-sim --output-on-failure && ctest --test-dir build/libra-search --output-on-failure
```

```bash
bin/libra-usi        # USI 拡張エンジン（モデルは環境変数 LIBRA_MODEL。Releases の libra-v0.1.onnx か、学習した latest.onnx）
bin/libra run        # 自己対局と学習（状態は ~/libra-run/ls）。stop / status / export
```

## 構成

| ディレクトリ | 役割 |
|---|---|
| `libra-sim/` | C++ シミュレータ（ビットボード、pybind11）、perft、ルールテスト |
| `libra-net/` | ネットワーク定義（Transformer）、ONNX 変換 |
| `libra-search/` | MCGS、df-pn、41 手目の裁定と配置詰みの証明探索 |
| `libra-engine/` | USI 拡張エンジン `libra` / `libra.exe`（ONNX Runtime） |
| `libra-league/` | 自己対局、学習、搾取者とリーグ、評価・計測ハーネス、`bin/libra` |
| `libra-scale/` | 玉配置表（天秤）の生成・検証、`scale.json` |
| `libra-cloud/` | vast.ai の GPU で自己対局ワーカーを動かす道具（`bin/libra-vast`、ブリッジ、ベンチ） |
| `bin/` | 起動スクリプト（`libra`、`libra-usi`、`libra-scale`、`libra-vast`） |
| `tools/` | ONNX Runtime の取得（ハッシュ固定）、配布物の作成、Windows 側の管理コンソールと bat |
| `docs/` | 設計書、ルール仕様、接続仕様、運用、決定と実測の記録 |
| `data/` | データセットのマニフェスト（SHA-256、取得元）のみ。実体は Releases |
| `.claude/skills/` | エージェント（Claude Code）向けの駆動手順と driver |

## 文書

読む順（新しく入る人向け）:

1. [docs/rules.md](docs/rules.md) — ルール仕様（唯一の正）。布石、41 手目の裁定、終局規定、perft 値、他の実装との差
2. [docs/getting-started.md](docs/getting-started.md) — ビルド・テスト・対局・学習の手順
3. [docs/libra-design.md](docs/libra-design.md) — 設計書（2026-09-11、ルール設計者の計画）。[docs/libra-local.md](docs/libra-local.md) — 実行計画（ローカル主体、年内の公開）。**この 2 つは作成時のまま書き換えない**。計画からの差異は [docs/protocol.md](docs/protocol.md) §4 と [docs/decisions.md](docs/decisions.md) に記録する。計画書の「v0.1」は年末の公開版（1.0）を指し、2026-09-15 に出した v0.1 とは別
4. [docs/protocol.md](docs/protocol.md) — GUI（tenbin-shogi-desktop）との接続仕様。USI の申告、布石の `position fuseki`、両玉の配置と先後の選択、計測用の相手の起動情報
5. [docs/decisions.md](docs/decisions.md) — 決定の記録（日付、内容、理由）。[docs/measurements.md](docs/measurements.md) — 実測値（日付、条件、値）
6. [docs/model-card-v0.1.md](docs/model-card-v0.1.md) — 配布した重みの説明（ネットの形、学習の設定、計測、既知の限界、再現に使う版）
7. [docs/runbook.md](docs/runbook.md) — 運用（状態ディレクトリ、停止と再開、Windows の自動起動と管理コンソール、自動計測、搾取者、クラウドのワーカー）
8. [docs/method-evidence.md](docs/method-evidence.md) — 採用した手法と設定値の裏取り（出所・文献・計測）。[docs/exploiter-literature.md](docs/exploiter-literature.md) — 搾取者の手法の文献調査

各パッケージの README（`libra-*/README.md`）にファイルごとの役割とテストの一覧がある。

## リリース

| 版 | 中身 | 取得 |
|---|---|---|
| v0.1（step 477,636、2026-09-15） | `libra-v0.1.onnx`（fp32、opset 17）・`libra-v0.1.pt`・`SHA256SUMS`、玉配置表 `scale-v0.1.json`（CC0）、自己対局の標本 56,300 局（CC0）、Windows 版 `libra.exe` の zip | [Releases](https://github.com/kotenbu135/LibraShogi/releases/tag/v0.1) |

重みの中身・学習のしかた・計測・既知の限界は [モデルカード](docs/model-card-v0.1.md)。v0.1 は「自己対局だけでどこまで指せるか」の最初の区切りで、外部エンジンに対する強さは測っていない。
年内に Libra-L を 1.0 として公開する予定（[docs/libra-local.md](docs/libra-local.md) §5）。

## ライセンス

| 範囲 | ライセンス | 全文 |
|---|---|---|
| コード（`libra-*/`、`bin/`、`tools/`、`cmake/`） | Apache-2.0 | [LICENSE](LICENSE) |
| 文書（`docs/` 配下と各 `*.md`） | CC BY 4.0 | [LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt) |
| 自己対局データ（棋譜 JSONL）・玉配置表（`scale.json`） | CC0 1.0 | [LICENSES/CC0-1.0.txt](LICENSES/CC0-1.0.txt) |
| 学習済みの重み（`.pt`・`.onnx`） | Apache-2.0（モデルカード付き） | [LICENSE](LICENSE) |

依存物は [LICENSES/README.md](LICENSES/README.md)、著作権表示は [NOTICE](NOTICE)。
対戦相手（水匠5、fuseki-shogi-ai、desktop の wasm）は別プロセスで USI 経由で動かすだけで、同梱せず、学習信号にも使わない。

## 質問・不具合

[Issues](https://github.com/kotenbu135/LibraShogi/issues) へ。ルールの解釈に関わるものは [docs/rules.md](docs/rules.md) の節を示す。天秤将棋のルールそのもの（tenbinshogi.com）の話はルール設計者の判断になる。
