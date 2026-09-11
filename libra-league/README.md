# libra-league

自己対局ループ、リプレイ、学習、停止・再開（`libra` コマンド）。運用手順は [docs/runbook.md](../docs/runbook.md)。

| ファイル | 内容 |
|---|---|
| `libra_league/config.py` | 設定（TOML、既定は L-S） |
| `libra_league/selfplay.py` | C++ エンジン（librasearch）＋ PyTorch 推論のループ |
| `libra_league/replay.py` | 100 局チャンク、棋譜 JSONL（CC0）、バッチ作成（鏡映増強、V̂41 混合） |
| `libra_league/trainer.py` | AdamW、bf16、方策・価値・V̂41 の損失 |
| `libra_league/runner.py` | `libra run`: 時分割、フラグ、10 分ごとの原子的チェックポイント |
| `libra_league/cli.py` | `libra run/pause/resume/stop/throttle/status` |
| `tests/` | 設定・状態・リプレイのテスト |

評価ハーネス（世代間 Elo、較正）と計測ハーネス（外部エンジンとの USI 対局）はこれから足す。
