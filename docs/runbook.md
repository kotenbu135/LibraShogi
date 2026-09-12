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
| `PAUSE` / `STOP` / `EVAL_NOW` / `MATCH_NOW` | フラグファイル。`libra pause/stop/eval-now/match-now` が置き、ランナーが読む |
| `run.lock` | 実行中の PID。二重起動を防ぐ |
| `log.txt` | ランナーのログ |

## 2. コマンド

```bash
~/LibraShogi/bin/libra run                 # 前回状態から再開（無ければ新規）。冪等
~/LibraShogi/bin/libra pause               # 現在のバッチを終えて待機（GPU メモリを解放）
~/LibraShogi/bin/libra resume
~/LibraShogi/bin/libra stop                # チェックポイントを書いて終了
~/LibraShogi/bin/libra status
```

`--run <id>` で run-id、`--root <dir>` で親ディレクトリを変えられる（既定 `~/libra-run/ls`）。
`run --config path.toml` は初回だけ有効。

Windows 側: `C:\Users\sakis\libra\` に `libra-run.bat`（本体 ls）/ `libra-run-lx.bat`（搾取者 lx）と、**ls と lx の両方に効く** `libra-pause.bat` / `libra-resume.bat` / `libra-stop.bat` / `libra-status.bat`。デスクトップに pause / resume / status / stop の写し。

## 3. Windows Update で再起動しても続くようにする

1. タスク スケジューラに「LibraShogi run」（本体 ls、ログオン 1 分後）と「LibraShogi run lx」（搾取者 lx、ログオン 2 分後）を登録済み（`C:\Users\sakis\libra\LibraShogi-run.xml` / `LibraShogi-run-lx.xml`）。それぞれ `wscript.exe libra-run-hidden.vbs` / `libra-run-lx-hidden.vbs` → `wsl.exe -d Ubuntu-24.04 -- /home/sakis/LibraShogi/bin/libra [--run lx] run` を非表示で起動し、失敗時は 1 分後に再起動（999 回まで）。実行時間の上限なし。
2. **自動ログオンは利用者が設定する**（`netplwiz`）。設定しないと再起動後にログオンするまで止まる。
3. 再起動後の損失はチェックポイント間隔（10 分）＋進行中の対局分。
4. Windows Update の「アクティブ時間」を広めに設定する。
5. `checkpoints/` の最新 2 世代を 1 日 1 回 Windows 側へ写す（未実装。cron か Task Scheduler で `cp`）。

再登録するとき:

```powershell
schtasks /Create /TN "LibraShogi run" /XML "C:\Users\sakis\libra\LibraShogi-run.xml" /F
schtasks /Create /TN "LibraShogi run lx" /XML "C:\Users\sakis\libra\LibraShogi-run-lx.xml" /F
```

Windows 側のファイルの正は `tools/windows/`（`install.sh` で `C:\Users\<user>\libra` とデスクトップへ写す）。

**管理コンソール（GUI）**: デスクトップの `libra-console.bat`（`tools/windows/libra-console.ps1`、PowerShell 5.1 + WinForms、ビルド不要）。本体 ls と搾取者 lx の状態（稼働 / 一時停止 / 停止、step・世代・総局数、局/日の 1 時間平均と実測、終局内訳、loss、GPU メモリ、最終チェックポイント、搾取者の対本体勝率、log.txt の末尾）を 15 秒ごとに `wsl.exe -d Ubuntu-24.04 -- bin/libra --run <run> status --json --tail 8` で取り、局/日の推移を折れ線で出す。ボタンは bat と同じ操作（一時停止 / 再開 / 停止 / 起動 / 絞る、両 run 一括）。「起動」はタスク スケジューラの「LibraShogi run [lx]」を `schtasks /Run` で起動する（無ければ wsl.exe を直接起動）。局/日の履歴は `%LOCALAPPDATA%\LibraShogi\console-history.csv` に追記（7 日分を表示）。

再起動・一時停止の手順（bat 版）: 一時停止はデスクトップの `libra-pause.bat`（両 run が待機）、復帰は `libra-resume.bat`。PC を再起動するときは `libra-stop.bat` で両 run を止めて（`libra-status.bat` で `not running` を確認）から再起動し、ログオン後に両タスクが自動で再開する。搾取者を止めたままにしたいときは WSL で `bin/libra --run lx stop` だけ実行する（本体は openings を読むだけなので影響しない）。

## 4. 別作業でリソースを空けるとき

- GPU を使う作業: デスクトップの `libra-pause.bat` → 終わったら `libra-resume.bat`
- CPU だけ使う作業: そのままで良い（ワーカーは `nice 10`）。GPU が要る作業は管理コンソールの「一時停止」
- 数日止めても再開時のコストはゼロ

## 5. 設定を変えるとき

`~/libra-run/ls/config.toml` を編集し、`libra stop` → `libra run`。ネットの形（`[net]`）はチェックポイントと合わなくなるので変えない（変えるなら新しい run-id）。

## 6. 監視と自動計測

`libra status` の `games/day(1h)` を docs/measurements.md に週 1 回記録する（Windows では管理コンソール `libra-console.bat` で常時見える。§3）。`libra status --json [--tail N] [--history N]` は機械可読（`process`、`flags`、`state`、`status`、`log_tail`、`--history` で `metrics`・`evals`・`matches`・`archives`・`auto`・`auto_cfg`）。`status.json` の `engine` に終局理由の内訳（ruling41、mate、sennichite、perpetual、max_ply）がある。

**進捗の時系列**: ランナーは `metrics.jsonl` に 5 分ごと（`run.metrics_minutes`）に 1 行追記する（step、局数、局/日、loss 系、終局内訳の累積カウンタ、搾取者成績、GPU）。管理コンソールの「学習」「終局内訳」「手数」タブはこれを差分で割合にして描く。

**自動計測（`[auto]`、本体 ls のみ有効）**: `every_hours`（24）ごとにチェックポイントを `checkpoints/archive/` に残し、`libra eval`（`eval_sims`=96）で 2 通りの対局をする。

- **基準比（`anchor_games`=100、これが主）**: 固定の基準ネット（`state.json` の `auto.anchor`）と対局する。結果は `eval/anchor-*.json` と 1 行ずつの `eval/anchor.jsonl`（`elo` は基準の `offset` を足した値）。基準に `anchor_rebaseline`（0.85）以上勝ったら基準を新しい世代に置き換え、そこまでの差を `offset` に足す。連続世代どうしの差（1 日で +30 Elo 程度）は 100 局の測定幅 ±70 Elo に埋もれ、鎖にすると誤差が回数の平方根で積み上がるため、基準との大きな差で測る。
- **鎖（`chain_eval`、補助）**: 直前の archive との差を足した累積。基準が直前の archive と同じときは同じ対局になるので 1 回で兼ねる。

同じ周期で `libra match`（`match_games`=10 局、`match_go`="movetime 1000"、相手は fuseki_usi_server.py に `Threads=2`）も回す。どちらも別プロセス（`auto.log`）で GPU を共有し、結果は `eval/auto-*.json` と `matches/auto-*.summary.json`。前倒しは `libra eval-now` / `libra match-now`（フラグ EVAL_NOW / MATCH_NOW。次のチェックポイントで実行。コンソールの「今すぐ自己評価」「今すぐ対外対局」）。相手側は 41 手目以降を必ずやねうら王（水匠5 の評価関数）に中継するので「方策ネットだけの相手」は無い。ランナーを stop すると実行中のジョブは止め、再開後に積み直す。

## 7. 計測（外部エンジンとの対局）

```bash
~/LibraShogi/bin/libra match --games 20 --go "movetime 3000"      # Libra（latest.pt）対 fuseki_usi_server.py（やねうら王＋水匠5 中継）
~/LibraShogi/bin/libra eval --a ckpt1.pt --b ckpt2.pt --games 200  # 世代間 Elo と較正
```

`match` は `~/libra-run/ls/matches/<時刻>.jsonl`（1 局 1 行）と `.summary.json`、USI ログ `.log` を書く。裁定は libra-sim（docs/rules.md）。相手のバージョンとハッシュは docs/protocol.md §5。
GPU を L-S と共有するので、計測中は `libra pause` するか、局/日が落ちることを承知で回す。

## 8. desktop で Libra と指す（自分で体感する）

desktop（天秤将棋GUI 0.5.0、`C:\Users\sakis\AppData\Local\天秤将棋GUI\tenbin-shogi-gui.exe`）には Windows 版 `libra.exe` を「LibraShogi 0.0.2」として登録済み（`%APPDATA%\com.fusekishogi.tenbin\engines\libra\engine\`。モデルは同じフォルダの `libra.onnx`、CPU 実行）。

最新のネットで指すには管理コンソールの「desktop で対局」を押す。WSL の `~/libra-run/ls/checkpoints/latest.onnx` をそのフォルダの `libra.onnx` に写し（`libra.onnx.json` に step と時刻を残す）、desktop を起動する。desktop が既に起動しているときはモデルだけ更新するので、エンジンを立て直す（desktop を開き直す）と新しいネットになる。対局画面でエンジンに「LibraShogi」を選ぶ。無人で行うには `powershell -File libra-console.ps1 -UpdateDesktopModel`。

GPU（CUDA）で読ませたいときは「エンジン」→「実行ファイルを選んで追加」で `C:\Windows\System32\wsl.exe` を選び、引数に `-d Ubuntu-24.04 -- /home/sakis/LibraShogi/bin/libra-usi` を入れる（モデルは常に latest.onnx、学習中の GPU と共有）。

## 玉配置表（libra-scale）

`bin/libra-scale build --sims 1600` で `~/libra-run/ls/scale/scale.json` を作り、`bin/libra-scale verify --top 48 --games 100` で
釣り合い集合を検証対局の信頼区間で決め直す。エンジンには `setoption name Scale_Table value <path>` で渡す（`bin/libra match` は
`--libra-opt Scale_Table=<path>`）。表は世代ごとに作り直す（探索値は数分、検証対局は最終世代だけ本格的に）。

## 搾取者リーグ（Main exploiter）

搾取者 lx（`~/libra-run/lx`、1.9M、64 局同時）は凍結した本体（`lx/main.pt`）と対局し、自分の手だけを学習する。偶数枠で搾取者が先手。

**凍結相手の作り直し**: `[exploiter]` の `refresh_hours`（24）ごとに `main_source`（本体の `checkpoints/latest.pt`）を `main_ckpt` に写し、対本体成績を履歴（`state.json` の `exploiter.history`）へ移して 0 から数え直す。本体が強くなると古い相手への勝率が飽和し（2026-09-12 に 98.3%）、収束判定「対本体勝率が頭打ち」が意味を失うため。初回は起動直後に行う。

**布石**: `openings_minutes`（60）ごとに、作り直してからのチャンクだけから搾取者が勝った布石を `openings_out` に書く。本体 ls は `[selfplay] openings` でこれを読み、新規対局の 10% をそこから始める。相手を作り直した時点で布石は空にする（古い相手の穴なので本体に渡さない）。手動で書き出すときは `bin/libra --run lx openings`。

状態は `bin/libra --run lx status`（`exploiter` に勝率、`main_step`、`refreshed_at`）。管理コンソールの「対本体 勝率」行にも出る。
