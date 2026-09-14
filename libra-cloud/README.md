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

## 本番の run にワーカーを足す（vast_worker.py）

| ファイル | 内容 |
|---|---|
| `libra_cloud/bridge.py` | 手元で回す同期ループ。学習側の `weights/latest.pt` と布石が変わったらホストへ送り、ホストの `inbox/*.npz` を取ってきて手を再生して検査（`libra_league.workers.verify_games_file`）してから学習側の `inbox/` に置く。不正なファイルは `<out>/bridge/rejected/` |
| `bench/host_worker.sh` | ホストでワーカーを `--detached` で常駐させる（pid は `/root/out/worker.pid`） |
| `vast_worker.py` | GPU を借り、セットアップ、ワーカーの起動、ブリッジを `--hours` 時間回す。終わり・失敗・Ctrl+C でワーカーを止めて残りを取り、必ずインスタンスを消す |

学習側の run は `[workers] enabled = true` で起動しておく（`inbox/` が無い間、ブリッジは取ってこない）。途中で止めるときは `<out>/bridge/STOP` を置く。

```bash
S=<scratchpad>
PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-cloud \
  .venv/bin/python -m libra_cloud.prepare --worker --ckpt ~/libra-run/ls/checkpoints/latest.pt --config ~/libra-run/ls/config.toml --out $S/worker
~/.venvs/vastai/bin/python libra-cloud/vast_worker.py --gpu "RTX 5070 Ti" --max-dph 0.28 --min-rel 0.94 --hours 3 \
  --run-dir ~/libra-run/ls --bundle $S/worker/bundle.tar.gz --out $S/vast-worker
```

検査で確かめるのは記録の骨格（玉の配置と全手の合法性、終局の判定と手数、sfen41、方策の添字が合法手であること）まで。探索の出力（方策の確率・価値）の改ざんは検出できない。

束は git の HEAD から作るので、libra-league や libra-cloud を変えたらコミットしてから作り直す。局/日は起動 300 秒後以降に書かれた対局ファイルの間隔で出す（512 局を同時に始めるので、最初の終局はまとまって遅れる）。借りた時間と見積もり費用は `result.json`。終わったら `show_instances` が空であることを確かめる。
