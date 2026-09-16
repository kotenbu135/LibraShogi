# libra-league

自己対局ループ、リプレイ、学習、停止・再開、搾取者とリーグ、評価・計測（`libra` コマンド）。運用手順は [docs/runbook.md](../docs/runbook.md)。

| ファイル | 内容 |
|---|---|
| `libra_league/cli.py` | `bin/libra`: `run` / `stop` / `status` / `eval-now` / `match-now` / `eval` / `match` / `export` / `worker` / `openings` |
| `libra_league/config.py` | 設定（TOML、既定は L-S） |
| `libra_league/state.py` | 状態ディレクトリ（`~/libra-run/<run-id>/`）、原子的な書き込み、フラグ |
| `libra_league/supervise.py` | `libra run` の監視役: ランナー本体を子プロセスで回し、異常終了したら起動し直す |
| `libra_league/runner.py` | ランナー本体: 自己対局と学習の時分割、フラグ、10 分ごとの原子的チェックポイント。搾取者の run では凍結相手の作り直しと布石の書き出し |
| `libra_league/selfplay.py` | C++ エンジン（librasearch）＋ PyTorch 推論（CUDA Graphs）のループ。搾取者モードの相手のネットと方策 |
| `libra_league/replay.py` | 100 局チャンク、棋譜 JSONL（CC0）、バッチ作成（鏡映増強、V̂41 混合） |
| `libra_league/trainer.py` | AdamW、bf16、方策・価値・V̂41 の損失 |
| `libra_league/openings.py` | 搾取者が勝った布石（`openings.json`）の抽出と読み込み。本体の自己対局の一部をここから始める |
| `libra_league/league.py` | 本体と過去の搾取者の対局（スナップショットのプール、PFSP で相手を選ぶ） |
| `libra_league/workers.py` | `libra worker`: 自己対局だけのワーカー、重みの配布、対局ファイル（npz、pickle なし）の検査と inbox の取り込み |
| `libra_league/auto.py` | `metrics.jsonl`、archive、24 時間ごとの自動計測ジョブ（`libra eval` / `libra match` を別プロセスで） |
| `libra_league/evaluate.py` | `libra eval`: 世代間 Elo と較正 |
| `libra_league/usi_client.py`, `harness.py` | `libra match`: 外部エンジンとの無人対局、libra-sim による裁定、JSONL 棋譜 |
| `libra_league/usi_engine.py` | `bin/libra-usi-py`: USI 拡張エンジン（Python 版、暫定。C++ 版は libra-engine） |

## テスト（`tests/`）

| ファイル | 内容 |
|---|---|
| `test_league.py` | 設定のマージ、状態のフラグ、価値の目標（soft WDL）、リプレイのチャンクとサンプリング |
| `test_supervise.py` | 監視役の起動し直しと、利用者が止めたときに起動し直さないこと |
| `test_cli_status.py` | `libra status --json`（管理コンソールが読む形） |
| `test_auto.py` | 進捗の時系列・archive・自動ジョブ |
| `test_exploiter.py` | 搾取者モード（相手の手のマスク）と布石 |
| `test_league_pool.py` | 過去の搾取者のプールと PFSP |
| `test_workers.py` | 対局ファイルの書式と検査、重みの配布、inbox の取り込み |
| `test_trainer.py` | 学習の torch.compile（CPU では eager のまま、state_dict の鍵が変わらない、CUDA では compile と eager の勾配が一致） |
| `test_infer.py` | 自己対局の推論の写し（重みの更新、形が変わったときの作り直し） |
| `test_harness.py` | 計測ハーネス（`random_usi.py` のランダム USI エンジンと対局） |
