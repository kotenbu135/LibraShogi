# libra-cloud

vast.ai で自己対局ワーカー（`libra worker`、libra-league/libra_league/workers.py）を動かすための道具。vast.ai の利用開始は 2026-09-14 のユーザーの決定（docs/decisions.md）。

**API キーと SSH 鍵は絶対にコミットしない。** API キーは `~/.config/vastai/vast_api_key`（権限 600）、SSH 鍵は `~/.ssh/id_ed25519_vast`。vast.ai の SDK（MIT）はプロジェクトの `.venv` ではなく `~/.venvs/vastai` に入れる（`python3 -m venv ~/.venvs/vastai && ~/.venvs/vastai/bin/pip install vastai`）。

| ファイル | 内容 |
|---|---|
| `libra_cloud/hosts.py` | 借りたホストの実測（`bridge.log` の定常状態の局/日）から、次に借りるオファーの見込み（局/日、100 万局あたりの費用）を出す。選別の並び順に使う（docs/runbook.md §自己対局ワーカー「借りる順」） |
| `libra_cloud/bench.py` | オファーの選別（1 GPU、信頼度 0.98 以上、下り 200 Mbps 以上、CUDA 12.8 以上、CPU 8 コア以上、転送料 $0.02/GB 以下）、ベンチ用の設定、inbox の対局ファイルからの局/日（`python -m libra_cloud.bench report`） |
| `libra_cloud/prepare.py` | 手元で束を作る: チェックポイントから fp16 の重み、ベンチ用の設定（openings・学習側の機能を外す）、git の HEAD にあるワーカー用ソースの tar.gz |
| `bench/host_setup.sh` | ホスト（`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`）で librashogi / librasearch だけをビルドし、GPU と CPU を記録 |
| `bench/host_bench.sh` | ホストでワーカーを `--detached` で N 分回し、`/root/out/report.json` に局/日 |
| `vast_bench.py` | GPU を 1 台オンデマンドで借りて上の 2 つを回し、結果を持ち帰る。失敗・中断でも必ずインスタンスを消す |
| `libra_cloud/scale_bridge.py`, `bench/host_scale.sh` | 玉配置表の全組の検証対局（`libra-scale seq`）を vast.ai のホストで打たせるための同期ループとホスト側の常駐スクリプト（`libra-scale seq worker`）。手元の `active.json` を送り、棋譜を回収する |

## 実測の手順

```bash
S=<scratchpad>
PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-cloud \
  .venv/bin/python -m libra_cloud.prepare --ckpt ~/libra-run/ls/checkpoints/latest.pt --config ~/libra-run/ls/config.toml --out $S/bench
~/.venvs/vastai/bin/python libra-cloud/vast_bench.py --gpu "RTX 3090" --max-dph 0.25 --bundle $S/bench/bundle.tar.gz --out $S/vast-3090 --dry-run
~/.venvs/vastai/bin/python libra-cloud/vast_bench.py --gpu "RTX 3090" --max-dph 0.25 --minutes 20 --bundle $S/bench/bundle.tar.gz --out $S/vast-3090
```

## 本番の run にワーカーを足す（管理コンソール / bin/libra-vast）

ふだんは管理コンソールの **クラウド** タブから「起動」「停止」する（docs/runbook.md の自己対局ワーカーの節）。同じことをコマンドで行うとき:

```bash
bin/libra-vast offers --gpu RTX_5070_Ti --max-dph 0.28          # 全件に落ちた条件・1 つ緩めれば通る値・見込みの局/日と $/100 万局を付ける（借りない。--min-cores・--max-inet-cost も指定可）
bin/libra-vast start --run ls --gpu RTX_5070_Ti --max-dph 0.28 --hours 3   # すぐ返る。準備に 5〜15 分。--rent bid（既定、入札）| on-demand、--bid-margin 0.1
# ホストを失ったら launcher が bin/libra-vast start --continue-from <セッション名> で残りの時間の次のセッションを起動する
bin/libra-vast status --account                                  # 段階・借りた時間・費用・回収局数・残高・インスタンス
bin/libra-vast history                                           # 過去のセッションごとの費用（借りた時間＋転送料）・有効局（捨てた局を除く）・100 万局あたりの費用、月ごとの合計
bin/libra-vast stop                                              # 残りの局を取ってからインスタンスを消す
bin/libra-vast cleanup --yes                                     # libra- のラベルのインスタンスをすべて消す（残ったとき）
```

| ファイル | 内容 |
|---|---|
| `libra_cloud/vast_cli.py`（`bin/libra-vast`） | セッション（`~/libra-run/cloud/<run>-<時刻>/`）を作り、束の作成と `vast_worker.py` を setsid で切り離して起動する。同時に動かせるのは置き場所（`--root`）ごとに 1 つ。2 台目は別の `--root`（例 `~/libra-run/cloud2`）と別のワーカー名（`start --worker-id vast2`）で起動する（コンソールは `~/libra-run/cloud` しか見ないので、2 台目は 1 台目より先に終わるように `--hours` を決める。1 台目が終わった後に 2 台目のインスタンスが残ると、コンソールが「後始末」を促し、押すと 2 台目も消える）。停止はブリッジが動いていれば `bridge/STOP`、借りる途中ならプロセスグループに SIGTERM。`~/.venvs/vastai` の Python で動く |

GPU 名の空白は `_` でもよい（コンソールは wsl.exe に渡すので `_` を使う）。CPU GHz の下限の既定は 4.4（CPU が遅いホストでは探索が律速して GPU が遊ぶ）。

### 中身（vast_worker.py を直接使うとき）

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

## テスト（`tests/`）

| ファイル | 内容 |
|---|---|
| `test_bench.py` | オファーの選別と落ちた理由、入札の実効単価、局/日の計算、束の作成、ホストのスクリプトの構文、ssh の経路 |
| `test_hosts.py` | ホストの実測（bridge.log の傾き、日付をまたぐ時刻）、実績の鍵の優先、見込みと借りる順 |
| `test_bridge.py` | ブリッジ（重みと布石の送信、対局ファイルの回収と検査、送信の失敗と再試行、停止と回収） |
| `test_scale_bridge.py` | 検証対局のブリッジ（往復、ワーカーの死亡と回収の打ち切り） |
| `test_vast_cli.py` | `bin/libra-vast` の各サブコマンド（学習側の確認、二重起動の拒否、費用の見積もり、履歴） |

vast.ai の API には触れない（すべて偽の SDK・ssh で回す）。
