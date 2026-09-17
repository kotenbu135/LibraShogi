# Libra の開発に要る GPU の機能

2026-09-18 時点。出典はコード（`libra-league/libra_league/trainer.py`・`selfplay.py`・`config.py`、`libra-engine/src/ort_infer.cpp`、`tools/fetch_onnxruntime.sh`、`libra-cloud/`）と docs/measurements.md の実測。

手元の環境: RTX 5070 Ti（compute capability 12.0、VRAM 16 GB）、ドライバー 610.62、torch 2.11.0+cu128、cuDNN 9.19。

## 1. 学習（PyTorch、`libra-league/libra_league/trainer.py`）

| 機能 | 使い方 | 無いとどうなるか |
|---|---|---|
| CUDA 12.8 | torch cu128。Blackwell（sm_120）にはこの版以上が要る | GPU を使えない |
| bfloat16 の autocast | 学習の forward | 動くが、fp32 で遅くメモリも多い。bf16 が遅い旧世代 GPU の実速度は未確認 |
| fused AdamW | `fused=True`（CUDA のときだけ） | foreach 版になり、少し遅い |
| torch.compile（Inductor/Triton） | `max-autotune-no-cudagraphs`。eager より ×1.30 | 設定の `compile=none` で eager に戻せる（失敗したら自動で戻る） |
| 非同期の転送 | `non_blocking` | 影響は小さい |
| VRAM | 学習だけで最大 7.9〜8.1 GiB | ls・lx・自動計測を合わせた実測は 14.7 GB で、16 GB ではぎりぎり |

## 2. 自己対局の推論（PyTorch、`libra-league/libra_league/selfplay.py`）

| 機能 | 使い方 |
|---|---|
| float16 の推論 | `infer_dtype = "float16"`（bf16 と fp32 も選べる） |
| CUDA Graphs | 同時 512 局の固定バッチを捕獲する。`torch.cuda.CUDAGraph` と、捕獲用の別の CUDA Stream を使う |
| torch.compile（max-autotune） | 捕獲する forward を compile して、行列積のカーネルを選ぶ |
| ピン留めメモリ | 入出力のバッファを `pin_memory` にして、CPU→GPU（0.3 ms）と GPU→CPU（0.1 ms）を速くする |
| ストリームの同期 | ラウンドの終わりに `synchronize` する |
| 計算の中身 | Transformer（d 320・8 層・8 ヘッド、10.2M）。時間の 56% が GEMM、16% が注意機構 |

## 3. 対局用エンジン（C++、ONNX Runtime、`libra-engine/src/ort_infer.cpp`）

| 機能 | 使い方 |
|---|---|
| ONNX Runtime の CUDA EP | CUDA 12 と cuDNN 9 が要る。Linux 版と、Windows で DLL を差し替えた版 |
| DirectML EP（DirectX 12） | Windows 配布物の既定（ORT 1.24.4 ＋ DirectML 1.15.4）。NVIDIA 以外の GPU でも動く |
| CPU へのフォールバック | 自動では CUDA → DML → CPU の順に試す。cuDNN が無ければ最初の Run で CPU に落ちる |
| モデルの形式 | ONNX opset 17、fp32、バッチの大きさは可変 |
| 予定（未実装） | 複数の葉を同時に評価する、fp16 の推論 |

## 4. 運用の環境

| 機能 | 内容 |
|---|---|
| WSL2 の CUDA（GPU の準仮想化） | 学習と自己対局は WSL 内で動かす。必要なのは Windows 側の NVIDIA ドライバーだけ |
| 複数プロセスでの GPU 共有 | ls・lx・自動計測・玉配置表の対局が 1 枚を時分割で使う（MPS は使っていない） |
| 電力の制御 | 電力上限 250 W に張り付いた実測がある。電力上限は設定しない（ユーザーの決定） |
| 監視 | `nvidia-smi`（使用率・電力・メモリ）と `torch.cuda.memory_allocated`（status.json に出す） |
| vast.ai | `pytorch:2.11.0-cuda12.8-cudnn9-runtime` のイメージ。オファーはドライバーが CUDA 12.8 以上、1 GPU のものに絞る |

## まとめ：GPU の条件

- **必須:** CUDA 12.8 以上のドライバー、PyTorch cu128 の対象（sm_75〜sm_120）、VRAM 約 8 GB 以上（学習 1 本）。
- **今の速さを出すのに要る:** bf16 と fp16、Triton、CUDA Graphs、ピン留めメモリ。さらに ls・lx を同時に回すには VRAM 16 GB 級。
- **配布版のエンジン:** DirectX 12 の DirectML があれば動く。NVIDIA では CUDA 12 と cuDNN 9 があると速い。

## 未確認

- Ampere より前の GPU で、bf16 と max-autotune の速さが出るか。
- VRAM 12 GB 以下で ls と lx を同時に回せるか。
