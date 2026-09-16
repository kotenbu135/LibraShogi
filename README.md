# LibraShogi

天秤将棋（https://fusekishogi.com/rules/ ）の AI「Libra」。ゼロから開発する。

- 設計: [docs/libra-design.md](docs/libra-design.md)
- 実行計画: [docs/libra-local.md](docs/libra-local.md)
- ルール仕様（唯一の正）: [docs/rules.md](docs/rules.md)
- desktop との接続仕様: [docs/protocol.md](docs/protocol.md)
- 決定の記録: [docs/decisions.md](docs/decisions.md) ／ 実測値: [docs/measurements.md](docs/measurements.md)
- 運用（停止・再開、Task Scheduler）: [docs/runbook.md](docs/runbook.md)
- 配布した重みの説明: [docs/model-card-v0.1.md](docs/model-card-v0.1.md)（モデルカード）

## 構成

| ディレクトリ | 役割 |
|---|---|
| `libra-sim/` | C++ シミュレータ（ビットボード、pybind11）、perft、ルールテスト |
| `libra-net/` | ネットワーク定義、学習、ONNX 変換 |
| `libra-search/` | MCGS、df-pn、配置詰み探索 |
| `libra-engine/` | USI 拡張エンジン `libra` / `libra.exe`（ONNX Runtime） |
| `libra-league/` | 自己対局、リーグ、評価・計測ハーネス |
| `libra-scale/` | 玉配置表（天秤）の生成・検証、`scale.json` |
| `libra-cloud/` | vast.ai の GPU で自己対局ワーカーを動かす道具（`bin/libra-vast`、ブリッジ、ベンチ） |
| `bin/` | 起動スクリプト（`libra`、`libra-usi`、`libra-scale`、`libra-vast`） |
| `tools/` | ONNX Runtime の取得（ハッシュ固定）、Windows 側の管理コンソールと bat |
| `docs/` | 設計書、ルール仕様、runbook |
| `data/` | データセットのマニフェスト（SHA-256、取得スクリプト）のみ |

## ビルドと実行

Ubuntu 24.04、g++ 13、Python 3.12 で確認している。apt は使わず venv の pip で揃える。詳しい手順・エンジンの駆動・つまずきどころは
[.claude/skills/run-librashogi/SKILL.md](.claude/skills/run-librashogi/SKILL.md)、Windows 版 `libra.exe` のクロスビルドは
[libra-engine/README.md](libra-engine/README.md)。

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
```

`-DLIBRA_ORT_DIR` を渡さないと、エンジン（`build/libra-engine/libra`）は作られない（CMake は `libra-engine: skipped` と出すだけで成功する）。
Python のパッケージは pip install しない。`bin/` のスクリプトは自分で `PYTHONPATH` を通すが、pytest を直接呼ぶときは付ける。

テスト（CI の `.github/workflows/ci.yml` と同じ。結果行の `100% tests passed` と `N passed` で確かめる）:

```bash
ctest --test-dir build/libra-sim --output-on-failure
ctest --test-dir build/libra-search --output-on-failure
PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-scale:libra-cloud python -m pytest -q \
  libra-sim/tests/test_python.py libra-league/tests libra-net/tests libra-engine/tests libra-scale/tests \
  libra-search/tests/test_external.py libra-search/tests/test_selfplay_threads.py libra-cloud/tests
```

実行:

```bash
bin/libra run        # 自己対局と学習（状態は ~/libra-run/ls）。stop / status。運用は docs/runbook.md
bin/libra-usi        # USI エンジン（モデルは ~/libra-run/ls/checkpoints/latest.onnx）
```

## リリース

| 版 | 中身 | 取得 |
|---|---|---|
| v0.1（step 477,636、2026-09-15） | `libra-v0.1.onnx`（fp32、opset 17）・`libra-v0.1.pt`・`SHA256SUMS`、玉配置表 `scale-v0.1.json`（CC0）、Windows 版 `libra.exe` の zip | [Releases](https://github.com/kotenbu135/LibraShogi/releases) |

重みの中身・学習のしかた・計測・既知の限界は [モデルカード](docs/model-card-v0.1.md) を読む。
自己対局だけで学習しており、既存の将棋 AI のコード・重み・棋譜・評価値は内部にも学習信号にも使っていない。

## ライセンス

| 範囲 | ライセンス | 全文 |
|---|---|---|
| コード（`libra-*/`、`bin/`、`tools/`、`cmake/`） | Apache-2.0 | [LICENSE](LICENSE) |
| 文書（`docs/` 配下と各 `*.md`） | CC BY 4.0 | [LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt) |
| 自己対局データ（棋譜 JSONL）・玉配置表（`scale.json`） | CC0 1.0 | [LICENSES/CC0-1.0.txt](LICENSES/CC0-1.0.txt) |
| 学習済みの重み（`.pt`・`.onnx`） | Apache-2.0（モデルカード付き） | [LICENSE](LICENSE) |

依存物は [LICENSES/README.md](LICENSES/README.md)。貢献は [CONTRIBUTING.md](CONTRIBUTING.md)（DCO、クリーンルーム方針）。
