# libra-cloud

vast.ai で自己対局ワーカー（`libra worker`、libra-league/libra_league/workers.py）を動かすための道具。vast.ai の利用開始は 2026-09-14 のユーザーの決定（docs/decisions.md）。

**API キーと SSH 鍵は絶対にコミットしない。** API キーは `~/.config/vastai/vast_api_key`（権限 600）、SSH 鍵は `~/.ssh/id_ed25519_vast`。vast.ai の SDK（MIT）はプロジェクトの `.venv` ではなく `~/.venvs/vastai` に入れる（`python3 -m venv ~/.venvs/vastai && ~/.venvs/vastai/bin/pip install vastai`）。

| ファイル | 内容 |
|---|---|
| `libra_cloud/bench.py` | オファーの選別（1 GPU、信頼度 0.98 以上、下り 200 Mbps 以上、CUDA 12.8 以上、CPU 8 コア以上、転送料 $0.02/GB 以下）、ベンチ用の設定、inbox の対局ファイルからの局/日（`python -m libra_cloud.bench report`） |
| `libra_cloud/prepare.py` | 手元で束を作る: チェックポイントから fp16 の重み、ベンチ用の設定（openings・学習側の機能を外す）、git の HEAD にあるワーカー用ソースの tar.gz |
| `bench/host_setup.sh` | ホスト（`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`）で librashogi / librasearch だけをビルドし、GPU と CPU を記録 |
| `bench/host_bench.sh` | ホストでワーカーを `--detached` で N 分回し、`/root/out/report.json` に局/日 |
| `vast_bench.py` | GPU を 1 台オンデマンドで借りて上の 2 つを回し、結果を持ち帰る。失敗・中断でも必ずインスタンスを消す |

## 実測の手順

```bash
S=<scratchpad>
PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-cloud \
  .venv/bin/python -m libra_cloud.prepare --ckpt ~/libra-run/ls/checkpoints/latest.pt --config ~/libra-run/ls/config.toml --out $S/bench
~/.venvs/vastai/bin/python libra-cloud/vast_bench.py --gpu "RTX 3090" --max-dph 0.25 --bundle $S/bench/bundle.tar.gz --out $S/vast-3090 --dry-run
~/.venvs/vastai/bin/python libra-cloud/vast_bench.py --gpu "RTX 3090" --max-dph 0.25 --minutes 20 --bundle $S/bench/bundle.tar.gz --out $S/vast-3090
```

束は git の HEAD から作るので、libra-league や libra-cloud を変えたらコミットしてから作り直す。局/日は起動 300 秒後以降に書かれた対局ファイルの間隔で出す（512 局を同時に始めるので、最初の終局はまとまって遅れる）。借りた時間と見積もり費用は `result.json`。終わったら `show_instances` が空であることを確かめる。
