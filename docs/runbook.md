# Runbook（ローカル運用）

docs/libra-local.md §7〜8 の実装。状態はすべて `~/libra-run/<run-id>/` にある。ライセンス: CC BY 4.0。

**系列**: 本体 `ls` と搾取者 `lx` は、二飛香（docs/rules.md §3.2、2026-09-13）を入れた系列。`ls` は旧ルールの ls の最終チェックポイント（step 229,590）の `latest.pt`（モデル・オプティマイザ・step）を `checkpoints/latest.pt` に置いて始め、`lx` はゼロから（凍結相手 `lx/main.pt` は同じ `latest.pt` を先に置き、起動直後に `main_source` から作り直す）。局数・チャンク・自動計測の基準（`auto.anchor`）・Elo の累積は 0 から数え直し、旧系列とはつながない。外部エンジンとの自動計測は、相手が二飛香を知らないあいだ `[auto] match_games = 0` で止め（布石中に二飛香の手を指すと非合法手で負けになり、勝率が意味を持たない）、相手が二飛香に追いついた（fuseki-shogi-ai `eb3870f`、`Fuseki_Rules` 既定 2）ので 2026-09-13 に `match_games = 10`、`match_opponent_opt = "Threads=2,Fuseki_Rules=2"` に戻した（反映はランの起動し直しから）。
旧ルールの系列は `~/libra-run/ls-v0`（step 229,590、710,483 局）と `~/libra-run/lx-v0`（step 129,608）に残してあり、再開しない。`config.toml` の参照先はそれぞれ `ls-v0` / `lx-v0` の中で閉じるように付け替えた（`bin/libra --run ls-v0 status` で読める）。棋譜の陣の多くが二飛香に当たるので、新しい規定では再生できない。

## 1. 状態ディレクトリ

| パス | 内容 |
|---|---|
| `config.toml` | 初回起動時の設定の写し。以後の `libra run` はこれを使う（変えるときはここを編集して stop → run） |
| `state.json` | 世代、ステップ数、総局数、チャンク索引、再起動履歴、累積稼働時間 |
| `status.json` | 30 秒ごとの状態（局/日、勝敗内訳、学習損失、GPU メモリ） |
| `checkpoints/ckpt_<step>.pt`, `latest.pt` | 10 分ごと。モデル・オプティマイザ・乱数状態・設定。直近 3 つと 50,000 ステップごとを残す |
| `replay/chunk_<n>.pkl` | 100 局ごとの対局記録（学習用）。書き終えてから名前を確定する（書きかけは `.tmp`） |
| `games/games_<n>.jsonl` | 同じ 100 局の棋譜（公開用、CC0）。1 局 1 行: `tokens`, `result`, `reason`, `plies`, `sfen41`, `v41` |
| `STOP` / `EVAL_NOW` / `MATCH_NOW` | フラグファイル。`libra stop/eval-now/match-now` が置き、ランナーが読む。止まっている run に残った `STOP`（と廃止した一時停止の `PAUSE`）は、次の `libra run` が起動前に消す |
| `run.lock` | 実行中のランナー本体の PID。二重起動を防ぐ（その pid が libra_league のプロセスでなければ無効） |
| `log.txt` | ランナーのログ（監視役の再起動の記録 `supervisor:` もここ） |
| `stdout.log` | ランナー本体の標準出力・標準エラー（監視役が追記。異常終了したときの CUDA のエラー文などはここに残る） |
| `auto_job.json` | 実行中の自動計測ジョブ（pid・引数）。ランナーが abort で落ちたとき、起動し直したランナーがこれを見て孤児のジョブを止めて積み直す |
| `weights/latest.pt` | `[workers] enabled` の run だけ。ワーカーに配る重み（fp16 のモデル、step、ネットの形。約 20 MB）。学習のたびに原子的に書き換える |
| `inbox/<worker>-<時刻>-<連番>.npz` | `[workers] enabled` の run だけ。ワーカーが置いた対局ファイル（pickle なし）。学習側が取り込んで消す。検査で弾いたものは `inbox/rejected/`（直近 20） |

## 2. コマンド

```bash
~/LibraShogi/bin/libra run                 # 起動。前回状態から再開（無ければ新規）。冪等
~/LibraShogi/bin/libra stop                # 停止。チェックポイントを書いて終了
~/LibraShogi/bin/libra status
```

**操作は起動（`run`）と停止（`stop`）だけ**（2026-09-14 のユーザーの決定。一時停止・再開は廃止）。GPU や CPU を空けるときも停止し、終わったら起動する。どの順に押しても起動が空振りしないよう、`libra run` は起動前に次をする（log.txt に `start:` 行）:
- 停止処理中（STOP があり、まだ動いている）なら、止まるのを最大 180 秒待ってから起動する
- 止まっている run に残った STOP（と PAUSE）を消す
- STOP の無い稼働中の run には起動しない（`start: already running`）

`--run <id>` で run-id、`--root <dir>` で親ディレクトリを変えられる（既定 `~/libra-run/ls`）。
`run --config path.toml` は初回だけ有効。

Windows 側: `C:\Users\sakis\libra\` に `libra-run.bat`（本体 ls）/ `libra-run-lx.bat`（搾取者 lx）と、**ls と lx の両方に効く** `libra-stop.bat` / `libra-status.bat`。デスクトップに status / stop の写し（`install.sh` は前に写した libra-pause.bat / libra-resume.bat を消す）。

## 3. Windows Update で再起動しても続くようにする

1. タスク スケジューラに「LibraShogi run」（本体 ls、ログオン 1 分後）と「LibraShogi run lx」（搾取者 lx、ログオン 2 分後）を登録済み（`C:\Users\sakis\libra\LibraShogi-run.xml` / `LibraShogi-run-lx.xml`）。それぞれ `wscript.exe libra-run-hidden.vbs` / `libra-run-lx-hidden.vbs` → `wsl.exe -d Ubuntu-24.04 -- /home/sakis/LibraShogi/bin/libra [--run lx] run` を非表示で起動する。実行時間の上限なし。タスクの「失敗時に 1 分後に再起動（999 回まで）」は起動後の異常終了には効かない（9/13 に lx が CUDA の abort で落ちたまま 4.7 h 止まった）。
   **`libra run` は監視役**で、ランナー本体（子の `run --no-supervise`）が異常終了したら 60 秒後に起動し直す。起動し直さないのは、停止（STOP フラグ、終了コード 0）、二重起動（終了コード 3）、Ctrl+C・kill・`wsl --shutdown`（SIGINT/SIGTERM/SIGHUP）、待機中に STOP が置かれたとき。15 分未満で落ちるのが 5 回続いたら諦めて止まる。記録は log.txt の `supervisor:` 行と `stdout.log`。待機中（最大 60 秒）は status が `not running` と出る。
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

**管理コンソール（GUI）**: デスクトップの `libra-console.bat`（`tools/windows/libra-console.ps1`、PowerShell 5.1 + WinForms、ビルド不要）。本体 ls と搾取者 lx の状態（稼働中 / 停止処理中 / 停止、step・世代・総局数、局/日の 1 時間平均と実測、終局内訳、loss、GPU メモリ、最終チェックポイント、搾取者の対本体勝率、log.txt の末尾）を 15 秒ごとに `wsl.exe -d Ubuntu-24.04 -- bin/libra --run <run> status --json --tail 40` で取り、局/日の推移を折れ線で出す。**縦長のウィンドウが前提**で、上のタブで **ls / lx / クラウド / クラウド履歴** を切り替える。タブの見出しに稼働状態（● 稼働中 / 停止処理中 / 停止、クラウドは段階、履歴は今月の費用）が色付きで出るので、裏のタブの run が止まっても分かる。run のタブは上から状態の表・ボタン・グラフ（局/日、Elo、対外対局、学習、終局内訳、手数、ログ。学習・終局内訳・手数・ログはそのタブの run）。Elo と対外対局は 1 日 1 回の計測なので「グラフの期間」にかかわらず全期間を描く。Elo の実線（点と 95% 区間の縦線）が主の基準比、薄い点線が補助の鎖で、同じ step を別の方法で測るので同じ時刻に 2 つの値が並ぶ（鎖が上でも基準比が下がったわけではない）。位置・大きさ・選んだタブは `%LOCALAPPDATA%\LibraShogi\console-layout.json` に残り、次に開いたときに戻る。ボタンは「起動」「停止」（run ごとと「全部 起動」「全部 停止」）と自動計測の前倒しだけ。稼働中は「起動」、止まっているときは「停止」を押せなくする。停止処理中は「起動」を押してよい（止まってから起動する）。「起動」はタスク スケジューラの「LibraShogi run [lx]」を `schtasks /Run` で起動し（無ければ wsl.exe を直接起動）、5 分以内に稼働を確かめられなければステータスバーに赤で出す。操作の結果はステータスバーに 1〜15 分残る。局/日の履歴は `%LOCALAPPDATA%\LibraShogi\console-history.csv` に追記（7 日分を表示）。**クラウド** タブは vast.ai の自己対局ワーカーの起動 / 停止 / 候補 / 後始末と、段階・費用・回収局数・残高を出す（`bin/libra-vast`。§「自己対局ワーカー」の 4）。起動の確認には同じ GPU の過去の実績（借りた 1 時間あたりの有効局、100 万局あたりの費用、見込みの局数）も出る。**クラウド履歴** タブは `bin/libra-vast history --json` でセッションごとの GPU・$/h・借りた時間・費用（借りた時間 × $/h ＋ 転送料。転送料は送った・取ってきたバイト数 × ホストの単価で、記録の無い古いセッションは見積もり）・有効局（回収局 − 学習側が古すぎて捨てた局。捨てた局は学習側の log.txt の `workers: dropped` の行をセッションの期間で数える）・局/日（ブリッジの時間で換算）・100 万局あたりの費用を一覧と棒グラフで出し、合計と今月の費用（1 ドル 150 円で円に直し、月 1 万円の上限との比）を出す。コンソールを直したら `tools/windows/install.sh` で写し直す。

再起動の手順（bat 版）: PC を再起動するときは `libra-stop.bat` で両 run を止めて（`libra-status.bat` で `not running` を確認）から再起動し、ログオン後に両タスクが自動で再開する。搾取者を止めたままにしたいときは WSL で `bin/libra --run lx stop` だけ実行する（本体は openings を読むだけなので影響しない）。

## 4. 別作業でリソースを空けるとき

- GPU を使う作業: 管理コンソールで「停止」→ 終わったら「起動」（一時停止は廃止）
- CPU だけ使う作業: そのままで良い（ワーカーは `nice 10`）
- 数日止めても再開時のコストはゼロ（損失は最後のチェックポイント以降の進行中の対局だけ）
- 推論は CUDA Graphs で捕獲している（`[selfplay] compile`、decisions.md 2026-09-14）。起動直後の最初のラウンドで捕獲するので、起動時は torch.compile の autotune に 5〜25 秒ほど掛かる（WSL の再起動で `/tmp` のキャッシュが消えた後は長め）。`log.txt` に `selfplay: inference model=compile(max-autotune)+cudagraph` が出れば有効、`eager` なら捕獲に失敗していて理由は `stdout.log`

## 5. 設定を変えるとき

`~/libra-run/ls/config.toml` を編集し、`libra stop` → `libra run`。ネットの形（`[net]`）はチェックポイントと合わなくなるので変えない（変えるなら新しい run-id）。

## 6. 監視と自動計測

`libra status` の `games/day(1h)` を docs/measurements.md に週 1 回記録する（Windows では管理コンソール `libra-console.bat` で常時見える。§3）。`libra status --json [--tail N] [--history N]` は機械可読（`process`、`flags`、`state`、`status`、`log_tail`、`--history` で `metrics`・`evals`・`matches`・`archives`・`auto`・`auto_cfg`）。`status.json` の `engine` に終局理由の内訳（ruling41、mate、sennichite、perpetual、max_ply）がある。

**進捗の時系列**: ランナーは `metrics.jsonl` に 5 分ごと（`run.metrics_minutes`）に 1 行追記する（step、局数、局/日、loss 系、終局内訳の累積カウンタ、搾取者成績、GPU）。管理コンソールの「学習」「終局内訳」「手数」タブはこれを差分で割合にして描く。「局/日」タブは隣り合う行の `games_total` の差から出した 5 分平均（`status --history` の `gpd_5m`。間が 15 分を超えた行は出さない）を描く。

**自動計測（`[auto]`、本体 ls のみ有効）**: `every_hours`（24）ごとにチェックポイントを `checkpoints/archive/` に残し、`libra eval`（`eval_sims`=96）で 2 通りの対局をする。

- **基準比（`anchor_games`=100、これが主）**: 固定の基準ネット（`state.json` の `auto.anchor`）と対局する。結果は `eval/anchor-*.json` と 1 行ずつの `eval/anchor.jsonl`（`elo` は基準の `offset` を足した値）。基準に `anchor_rebaseline`（0.85）以上勝ったら基準を新しい世代に置き換え、そこまでの差を `offset` に足す。連続世代どうしの差（1 日で +30 Elo 程度）は 100 局の測定幅 ±70 Elo に埋もれ、鎖にすると誤差が回数の平方根で積み上がるため、基準との大きな差で測る。
- **鎖（`chain_eval`、補助）**: 直前の archive との差を足した累積。基準が直前の archive と同じときは同じ対局になるので 1 回で兼ねる。

同じ周期で `libra match`（`match_games`=10 局、`match_go`="movetime 1000"、相手は fuseki_usi_server.py に `Threads=2`）も回す。どちらも別プロセス（`auto.log`）で GPU を共有し、結果は `eval/auto-*.json` と `matches/auto-*.summary.json`。前倒しは `libra eval-now` / `libra match-now`（フラグ EVAL_NOW / MATCH_NOW。次のチェックポイントで実行。コンソールの「今すぐ自己評価」「今すぐ対外対局」）。相手側は 41 手目以降を必ずやねうら王（水匠5 の評価関数）に中継するので「方策ネットだけの相手」は無い。ランナーを stop すると実行中のジョブは止め、再開後に積み直す。ランナーが異常終了した場合は、起動し直したランナーが残ったジョブ（孫の相手エンジンを含むプロセス グループ）を止めて積み直す。途中まで書いた出力は `<out>.interrupted` に改名する。

## 7. 計測（外部エンジンとの対局）

```bash
~/LibraShogi/bin/libra match --games 20 --go "movetime 3000"      # Libra（latest.pt）対 fuseki_usi_server.py（やねうら王＋水匠5 中継）
~/LibraShogi/bin/libra eval --a ckpt1.pt --b ckpt2.pt --games 200  # 世代間 Elo と較正
```

`match` は `~/libra-run/ls/matches/<時刻>.jsonl`（1 局 1 行）と `.summary.json`、USI ログ `.log` を書く。裁定は libra-sim（docs/rules.md）。相手のバージョンとハッシュは docs/protocol.md §5。
GPU を L-S と共有するので、計測中は ls・lx を停止するか、局/日が落ちることを承知で回す。

## 8. desktop で Libra と指す（自分で体感する）

desktop（天秤将棋GUI 0.5.0、`C:\Users\sakis\AppData\Local\天秤将棋GUI\tenbin-shogi-gui.exe`）には Windows 版 `libra.exe` を「LibraShogi 0.0.2」として登録済み（`%APPDATA%\com.fusekishogi.tenbin\engines\libra\engine\`。モデルは同じフォルダの `libra.onnx`）。2026-09-14 から DirectML 版の DLL（`onnxruntime.dll` 1.24.4・`DirectML.dll`）に差し替え、GPU で読む（`isready` で `info string … provider dml`）。以前の CPU 版は同じフォルダの `*.cpu-prev`、以前の exe は `libra.exe.prev`。学習中の ls・lx と GPU を共有するので、desktop で読ませている間は局/日が少し落ちる。

最新のネットで指すには管理コンソールの「desktop で対局」を押す。WSL の `~/libra-run/ls/checkpoints/latest.onnx` をそのフォルダの `libra.onnx` に写し（`libra.onnx.json` に step と時刻を残す）、desktop を起動する。desktop が既に起動しているときはモデルだけ更新するので、エンジンを立て直す（desktop を開き直す）と新しいネットになる。対局画面でエンジンに「LibraShogi」を選ぶ。無人で行うには `powershell -File libra-console.ps1 -UpdateDesktopModel`。

GPU（CUDA）で読ませたいときは「エンジン」→「実行ファイルを選んで追加」で `C:\Windows\System32\wsl.exe` を選び、引数に `-d Ubuntu-24.04 -- /home/sakis/LibraShogi/bin/libra-usi` を入れる（モデルは常に latest.onnx、学習中の GPU と共有）。

## 玉配置表（libra-scale）

`bin/libra-scale build --sims 1600` で `~/libra-run/ls/scale/scale.json` を作り、`bin/libra-scale verify --top 48 --games 100` で
釣り合い集合を検証対局の信頼区間で決め直す。エンジンには `setoption name Scale_Table value <path>` で渡す（`bin/libra match` は
`--libra-opt Scale_Table=<path>`）。表は世代ごとに作り直す（探索値は数分、検証対局は最終世代だけ本格的に）。

## 搾取者リーグ（Main exploiter）

搾取者 lx（`~/libra-run/lx`、1.9M、64 局同時）は凍結した本体（`lx/main.pt`）と対局し、自分の手だけを学習する。偶数枠で搾取者が先手。本体の手番の手は本体のネットで木を丸ごと読む。lx の手番の手は、価値を lx のネット、木の中の本体の手番の葉の方策を本体のネットから取る（`[exploiter] opponent_prior`、既定 true。本体の応手を本体の方策で予測する。docs/exploiter-literature.md）。

**凍結相手の作り直し**: `[exploiter]` の `refresh_hours`（24）が過ぎたとき、または `main_source`（本体の `checkpoints/latest.pt`）の step が凍結相手より `refresh_steps`（25,000。本体の約 4.9k step/h で約 5 h、vast.ai のワーカーを足すと約 3 h）以上進んだとき（`refresh_check_minutes` の 5 分ごとに step だけ読む）に、`main_source` を `main_ckpt` に写し、対本体成績を履歴（`state.json` の `exploiter.history`）へ移して 0 から数え直す。本体が強くなると古い相手への勝率が飽和し（2026-09-12 に 98.3%）、収束判定「対本体勝率が頭打ち」が意味を失うため。初回は起動直後に行う。

**布石**: `openings_minutes`（60）ごとに、作り直してからのチャンクだけから搾取者が勝った布石を `openings_out` に書く。本体 ls は `[selfplay] openings` でこれを読み、新規対局の 10%（`openings_prob`）をそこから始める。相手を作り直した時点で布石は空にする（古い相手の穴なので本体に渡さない）。手動で書き出すときは `bin/libra --run lx openings`。

**本体と過去の搾取者の対局**（`[league]`、2026-09-14 から）: 本体 ls は自己対局（512 局）とは別のエンジンで `n_games`（64）局を過去の lx と打ち、自分の手だけを方策の学習に使う（価値は結果から。lx の手は学習しない）。lx は起動時（プールが空のとき）と凍結相手を作り直すたびに、作り直す前の自分を `[exploiter] pool_out`（`~/libra-run/lx/pool/lx-<step>.pt`、新しい `pool_keep` 10 個）に保存する。ls は `[league] pool` の新しい `recent`（5）体から、本体が勝てていない相手ほど多く選び（PFSP、(1 − 勝率)²）、`switch_games`（256）局ごとに選び直す。成績は `status.json` の `league`（`pool` に相手ごとの本体の勝敗と勝率）と `log.txt` の `league: opponent lx step ...`。プールが空なら `pool_check_minutes`（10）ごとに見に行く。止めるときは ls の `[league] enabled = false` にして停止・起動する。

状態は `bin/libra --run lx status`（`exploiter` に勝率、`main_step`、`refreshed_at`）。管理コンソールの「対本体 勝率」行にも出る。

## 自己対局ワーカー（既定は無効。GPU を足すときの配管）

学習側（`libra run`）と自己対局だけのプロセス（`libra worker`）を分けられる（decisions.md 2026-09-14）。**同じ GPU で分けても局/日は増えない**（推論だけで GPU が埋まっているため）。別の GPU（vast.ai など、利用開始はユーザーの判断）の計算を同じ run に足すためのもの。本番の ls・lx は使っていない。

1. 学習側の `config.toml` に `[workers]` `enabled = true` を書き、停止 → 起動。学習側は今までどおり自分でも自己対局し、加えて `weights/latest.pt` を配り、`inbox/` の局を 10 秒ごとに取り込む（取り込んだ局も新規局数に数えるので、学習量の規則 `replay_ratio` は変わらない）。搾取者の run では無効。
2. ワーカーを起動: `bin/libra --run ls worker --id w1 [--n-games 512] [--threads 12]`。重みを読み、`chunk_games`（100）局ごとに `inbox/` にファイルを置き、学習側が新しい重みを配ると 10 秒以内に読み直す。同じマシンでは学習側が動いている間だけ打ち、停止（STOP）か学習側の終了で残りを書いて抜ける（学習側が止まっている間は待つ）。ワーカーは run.lock を取らないので、コンソールの状態と 起動 / 停止 は学習側だけを見る。別マシンで run ディレクトリの写し（`config.toml` と `weights/`）を使うときは `--detached`。
3. 学習側は、局を打った重みが `max_lag_steps`（2000）より古いファイルを捨て、型・形・値域が合わないファイルを `inbox/rejected/` に移す（手の合法性や方策・価値の改ざんは確かめない）。件数は `status` の `workers:` 行と status.json の `workers`。
4. **vast.ai の GPU を足す**（decisions.md 2026-09-14）: 管理コンソールの **クラウド** タブで GPU・上限 $/h・時間・信頼度の下限・CPU GHz の下限・コア数の下限・転送料の上限を選んで「起動」（確認に費用の見積もりと残高が出る）。準備に 5〜15 分。借りたホストでワーカーが打ち、手元のブリッジ（`libra_cloud.bridge`）が重みと布石を送り、局を取ってきて手を再生して検査してから `inbox/` に置く。
   - 表示（15 秒ごと。残高とインスタンスは 5 分ごと、「残高を更新」で今すぐ）: 状態（準備 → インスタンス作成 → セットアップ → 稼働 → 停止処理 → 終了）、借りた時間と残り、費用の見積もり、回収した局数と弾いた数、ls の取り込み、残高、借りているインスタンス、launcher.log の末尾。
   - 「停止」: ワーカーを止めて残りの局を取ってからインスタンスを消す（数分）。時間が来たときも同じ。「候補を見る」: 検索したオファーを全件、安い順に、落ちた条件（例「コア 6 < 16」「転送料 $0.026/GB > 上限 $0.020」）を付けて出す（借りない）。○1 から順に借りる。条件に合うものが無いときは「1 つ緩めれば借りられる」に、どの欄をいくつにすればどのホストが通るかが出て、そのボタンで欄の値を変えて検索し直せる。「この条件で起動…」で起動の確認へ進む（起動と同じ条件で判定する）。検索の時点で 1 GPU・verified・下り 200 Mbps 以上・イメージの CUDA 以上・ディスク 30 GB 以上に絞っているので、vast.ai のサイトより件数は少ない。借りられずに終わったセッションは launcher.log の `rejected by:` の行に落ちた条件の件数が残る。
   - **借り方**は既定で「入札」（割り込みあり。同じホストで on-demand より 16〜32% 安い。decisions.md 2026-09-15）。入札額は最低入札の 10% 増しで、上限 $/h・候補の順・費用は実効単価（dph_total − 最低入札 ＋ 入札）。高い入札が来るとホストが止められる。launcher が 60 秒ごとにインスタンスを見て、2 回続けて止まっていたら（またはブリッジが ssh の失敗で抜けて止まっていたら）状態が「打ち切り」になり、インスタンスを消して、残りの時間（ブリッジが動いた時間を引く）で次のセッションを自動で起動する（「終了（打ち切り・借り直し）」、履歴には別の行）。残りが 15 分未満・最初の開始から時間 + 30 分を過ぎた・停止を押したときは借り直さない。on-demand で落ちたときも同じ。on-demand で借りるときは「借り方」を切り替える。
   - **状態が「異常終了」か、セッションが動いていないのに「借りているインスタンス」が残っている（ステータスバーが赤）ときは課金が続いている。**「後始末」で libra- のラベルのインスタンスをすべて消す。
   - 同じ操作をコマンドで: `bin/libra-vast start [--gpu RTX_5070_Ti --max-dph 0.28 --hours 3]`・`stop`・`status [--account]`・`offers`・`cleanup --yes`。セッションの記録は `~/libra-run/cloud/<run>-<時刻>/`（launcher.log、setup.log、bridge/bridge.log、bridge/rejected/、result.json）。
   - 前提: ls の `config.toml` に `[workers] enabled = true`（無いと起動を断る）、vast.ai の API キーと SSH 鍵（libra-cloud/README.md）。
   - **ls を止めている間**: ホストのワーカーは `--detached` なので最後に配られた重みで打ち続け、ブリッジも `inbox/` があるので取り込み続ける（課金も続く）。ls の step が進まないので、再開後に取り込む局は `max_lag_steps` の判定でほとんど捨てられず、同じ重みの局がまとめて学習に回る。数分〜十数分の停止はそのままでよく、数時間止めるときは先にクラウドを「停止」する。「全部 停止」はクラウドを止めない。**クラウドを動かしたまま WSL の shutdown や PC の再起動をしない**（インスタンスを消すのは手元のランチャーで、ホスト側に自分で止まる仕組みが無いため、課金が続く。そうなったら「後始末」）。
   - ブリッジは重みを前の送信から 300 秒経つまで送らない（`libra_cloud.bridge --min-push-seconds`。転送料と回線の詰まりを減らすため。布石は変わったらすぐ送る）ので、bridge.log の `age` は最大で約 300 秒＋送信時間になる。
   - ls のログに `workers: dropped ... stale games` が多いときは、bridge.log の `bridge: pushed weights/latest.pt ... in X s (age Y s)` で重みがホストに届くまでの時間を見る（bridge.json の `push_s` に最大値）。
