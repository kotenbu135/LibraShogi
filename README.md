# LibraShogi

天秤将棋（https://fusekishogi.com/rules/ ）の AI「Libra」。ゼロから開発する。

- 設計: [docs/libra-design.md](docs/libra-design.md)
- 実行計画: [docs/libra-local.md](docs/libra-local.md)
- ルール仕様（唯一の正）: [docs/rules.md](docs/rules.md)
- desktop との接続仕様: [docs/protocol.md](docs/protocol.md)
- 決定の記録: [docs/decisions.md](docs/decisions.md) ／ 実測値: [docs/measurements.md](docs/measurements.md)
- 運用（停止・再開、Task Scheduler）: [docs/runbook.md](docs/runbook.md)

## 構成

| ディレクトリ | 役割 |
|---|---|
| `libra-sim/` | C++ シミュレータ（ビットボード、pybind11）、perft、ルールテスト |
| `libra-net/` | ネットワーク定義、学習、ONNX 変換 |
| `libra-search/` | MCGS、df-pn、配置詰み探索 |
| `libra-engine/` | USI 拡張エンジン `libra` / `libra.exe`（ONNX Runtime） |
| `libra-league/` | 自己対局、リーグ、評価・計測ハーネス |
| `libra-scale/` | 玉配置表（天秤）の生成・検証、`scale.json` |
| `libra-cloud/` | vast.ai テンプレート、費用モデル |
| `docs/` | 設計書、ルール仕様、runbook |
| `data/` | データセットのマニフェスト（SHA-256、取得スクリプト）のみ |

## ビルドと実行

```bash
python3 -m venv .venv && .venv/bin/pip install cmake ninja pybind11 pytest numpy
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
export PATH=$PWD/.venv/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python -c "import pybind11;print(pybind11.get_cmake_dir())")
cmake --build build && ctest --test-dir build/libra-sim --output-on-failure
bin/libra run        # 自己対局と学習（~/libra-run/ls）。stop / status
```

## ライセンス

コード Apache-2.0（[LICENSE](LICENSE)）／文書 CC BY 4.0／自己対局データ CC0 1.0／重み Apache-2.0。
依存物は [LICENSES/README.md](LICENSES/README.md)。貢献は [CONTRIBUTING.md](CONTRIBUTING.md)（DCO、クリーンルーム方針）。
