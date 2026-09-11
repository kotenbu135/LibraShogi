# Runbook（ローカル運用）

docs/libra-local.md §7〜8 の実装。状態はすべて `~/libra-run/<run-id>/` にある。ライセンス: CC BY 4.0。

## 1. 状態ディレクトリ

| パス | 内容 |
|---|---|
| `config.toml` | 初回起動時の設定の写し。以後の `libra run` はこれを使う（変えるときはここを編集して stop → run） |
| `state.json` | 世代、ステップ数、総局数、チャンク索引、再起動履歴、累積稼働時間 |
| `status.json` | 30 秒ごとの状態（局/日、勝敗内訳、学習損失、GPU メモリ） |
| `checkpoints/ckpt_<step>.pt`, `latest.pt` | 10 分ごと。モデル・オプティマイザ・乱数状態・設定。直近 3 つと 50,000 ステップごとを残す |
| `replay/chunk_<n>.pkl` | 100 局ごとの対局記録（学習用）。書き終えてから名前を確定する（書きかけは `.tmp`） |
| `games/games_<n>.jsonl` | 同じ 100 局の棋譜（公開用、CC0）。1 局 1 行: `tokens`, `result`, `reason`, `plies`, `sfen41`, `v41` |
| `PAUSE` / `STOP` / `THROTTLE` | フラグファイル。`libra pause/stop/throttle` が置き、ランナーが読む |
| `run.lock` | 実行中の PID。二重起動を防ぐ |
| `log.txt` | ランナーのログ |

## 2. コマンド

```bash
~/LibraShogi/bin/libra run                 # 前回状態から再開（無ければ新規）。冪等
~/LibraShogi/bin/libra pause               # 現在のバッチを終えて待機（GPU メモリを解放）
~/LibraShogi/bin/libra resume
~/LibraShogi/bin/libra stop                # チェックポイントを書いて終了
~/LibraShogi/bin/libra throttle --games 64 # 同時進行局数を絞る（0 で解除）
~/LibraShogi/bin/libra status
```

`--run <id>` で run-id、`--root <dir>` で親ディレクトリを変えられる（既定 `~/libra-run/ls`）。
`run --config path.toml` は初回だけ有効。

Windows 側: `C:\Users\sakis\libra\` に `libra-run.bat` / `libra-pause.bat` / `libra-resume.bat` / `libra-stop.bat` / `libra-status.bat` / `libra-throttle.bat`。デスクトップに pause / resume / status / stop の写し。

## 3. Windows Update で再起動しても続くようにする

1. タスク スケジューラに「LibraShogi run」を登録済み（`C:\Users\sakis\libra\LibraShogi-run.xml`）。ログオン 1 分後に `wscript.exe libra-run-hidden.vbs` → `wsl.exe -d Ubuntu-24.04 -- /home/sakis/LibraShogi/bin/libra run` を非表示で起動し、失敗時は 1 分後に再起動（999 回まで）。実行時間の上限なし。
2. **自動ログオンは利用者が設定する**（`netplwiz`）。設定しないと再起動後にログオンするまで止まる。
3. 再起動後の損失はチェックポイント間隔（10 分）＋進行中の対局分。
4. Windows Update の「アクティブ時間」を広めに設定する。
5. `checkpoints/` の最新 2 世代を 1 日 1 回 Windows 側へ写す（未実装。cron か Task Scheduler で `cp`）。

再登録するとき:

```powershell
schtasks /Create /TN "LibraShogi run" /XML "C:\Users\sakis\libra\LibraShogi-run.xml" /F
```

## 4. 別作業でリソースを空けるとき

- GPU を使う作業: デスクトップの `libra-pause.bat` → 終わったら `libra-resume.bat`
- CPU だけ使う作業: `libra throttle --games 64`（ワーカーは `nice 10`）
- 数日止めても再開時のコストはゼロ

## 5. 設定を変えるとき

`~/libra-run/ls/config.toml` を編集し、`libra stop` → `libra run`。ネットの形（`[net]`）はチェックポイントと合わなくなるので変えない（変えるなら新しい run-id）。

## 6. 監視

`libra status` の `games/day(1h)` を docs/measurements.md に週 1 回記録する。`status.json` の `engine` に終局理由の内訳（ruling41、mate、sennichite、perpetual、max_ply）がある。

## 7. 計測（外部エンジンとの対局）

```bash
~/LibraShogi/bin/libra match --games 20 --go "movetime 3000"      # Libra（latest.pt）対 fuseki_usi_server.py（やねうら王＋水匠5 中継）
~/LibraShogi/bin/libra eval --a ckpt1.pt --b ckpt2.pt --games 200  # 世代間 Elo と較正
```

`match` は `~/libra-run/ls/matches/<時刻>.jsonl`（1 局 1 行）と `.summary.json`、USI ログ `.log` を書く。裁定は libra-sim（docs/rules.md）。相手のバージョンとハッシュは docs/protocol.md §5。
GPU を L-S と共有するので、計測中は `libra pause` するか、局/日が落ちることを承知で回す。

## 8. desktop で Libra を動かす（暫定・Python 版）

「エンジン」→「実行ファイルを選んで追加」で `C:\Windows\System32\wsl.exe` を選び、引数に `-d Ubuntu-24.04 -- /home/sakis/LibraShogi/bin/libra-usi` を入れる。`usi` の申告から「布石にも対応」「GPU で読む」が自動で付く。`bin/libra-usi` は C++ 版 `build/libra-engine/libra`（ONNX Runtime、CUDA EP）を起動し、モデルは `~/libra-run/ls/checkpoints/latest.onnx`。学習中の最新にするには `bin/libra export`（latest.pt → latest.onnx）を実行する。

Windows 単体版: `C:\Users\sakis\libra\engine\libra.exe`（`onnxruntime.dll` と `libra.onnx` を同じ場所に置く。CPU 実行）。`libra-engine/README.md` のクロスビルド手順で作る。desktop にはこの exe を直接登録できる。
