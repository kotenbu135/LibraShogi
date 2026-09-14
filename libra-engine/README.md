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

Windows で GPU を使うには `libra.exe` の隣の DLL を差し替える（`libra.exe` は同じもの）:
- DirectML: NuGet の `Microsoft.ML.OnnxRuntime.DirectML`（1.24.4 で更新が止まっている）の `onnxruntime.dll` と
  `Microsoft.AI.DirectML` の `DirectML.dll`。DirectML EP はメモリパターンと並列実行を切って開く。
- CUDA: `onnxruntime-win-x64-gpu_cuda12-*.zip` の DLL に加え、CUDA 12（cudart・cuBLAS・cuFFT）と cuDNN 9 の DLL を隣か PATH に置く。

見出し（1.30）より古い版の `onnxruntime.dll` でも動くよう、API は 1.16 の版まで下げて取る（使う関数はすべて 1.16 までにある）。

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
