# libra-engine — USI 拡張エンジン `libra` / `libra.exe`

Libra の対局用実行ファイル。`libra-search` の外部駆動 MCGS/MCTS（布石は証明探索、本将棋は df-pn 詰み探索つき）と、
ONNX Runtime（MIT、C API を実行時にロード）による推論を 1 つの実行ファイルにまとめる。プロトコルは
[docs/protocol.md](../docs/protocol.md)。

## ビルド

```bash
tools/fetch_onnxruntime.sh              # third_party/onnxruntime/ に公式バイナリ（Linux GPU、Windows）を取得
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python -c "import pybind11;print(pybind11.get_cmake_dir())") \
  -DLIBRA_ORT_DIR=$PWD/third_party/onnxruntime/onnxruntime-linux-x64-gpu_cuda12-1.30.0
cmake --build build                      # → build/libra-engine/libra（隣に libonnxruntime.so.1 などを写す）
```

Windows 版（WSL の mingw-w64 posix 版でクロスビルド。依存は静的リンク、`onnxruntime.dll` だけ同梱）:

```bash
cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-posix.cmake \
  -DLIBRA_BUILD_PYTHON=OFF -DLIBRA_BUILD_TESTS=OFF -DLIBRA_ORT_DIR=$PWD/third_party/onnxruntime/onnxruntime-win-x64-1.30.0
cmake --build build-win                  # → build-win/libra-engine/libra.exe + onnxruntime.dll
```

## モデル

`bin/libra export`（`libra_net.export_onnx`）でチェックポイントを `latest.onnx`（fp32、opset 17、バッチ可変）にする。
既定のモデルパスは実行ファイルの隣の `libra.onnx`。環境変数 `LIBRA_MODEL` か `setoption name DNN_Model` で変える。

## 実行プロバイダ

`DNN_Provider`（既定 `auto`）: CUDA → DirectML → CPU の順で使えるものを選ぶ。CUDA は `libonnxruntime_providers_cuda.so`
（Linux GPU 版に同梱）と cuDNN 9 / CUDA 12 のライブラリが必要（WSL では `bin/libra-usi` が PyTorch 同梱のものを
`LD_LIBRARY_PATH` に足す）。Windows 版の配布物は CPU 版 `onnxruntime.dll`。DirectML 版の同梱は別途判断。

## 複数葉の同時評価（`DNN_Batch_Size`）

探索木は 1 本のまま、読む先の局面（葉）を最大 `DNN_Batch_Size` 個選んで 1 回の推論にまとめる。評価待ちの枝には
仮の負け（virtual loss）を置いて同じ葉を選ばないようにし、根の逐次半減も評価待ちを訪問として数える。
`go nodes N` は評価待ちも含めて N を超えないように選ぶので、報告する `nodes` は 1 葉ずつのときと同じ意味。
既定は 64（RTX 5070 Ti で 1,600 ノードの思考が 1 葉ずつの約 20 倍速く、64 より大きくしてもほぼ伸びない。docs/measurements.md）。1 にすると従来どおり 1 葉ずつ評価し、探索も従来と同じになる。自己対局（`librasearch.SelfPlay` の collect / apply）には効かない。

## 環境変数

- `LIBRA_MODEL`: `DNN_Model` の既定値
- `LIBRA_PROVIDER`: `DNN_Provider` の既定値
- `LIBRA_ORT_LIB`: ONNX Runtime 共有ライブラリのパス（既定は実行ファイルの隣 → システム）

## テスト

`libra-engine/tests/test_usi.py`（pytest）が小さな乱数ネットを ONNX にしてバイナリを駆動し、合法手・`bestmove win`・
`stop`・`movetime` を確かめる。
