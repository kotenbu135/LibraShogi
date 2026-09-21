# Runbook（ローカル運用）

docs/libra-local.md §7〜8 の実装。状態はすべて `~/libra-run/<run-id>/` にある。ライセンス: CC BY 4.0。

**系列（2026-09-17 夜から）**: 本体 `ls` は、やり直しの系列（ユーザーの決定。docs/decisions.md 2026-09-17、docs/restart-plan.md）。二飛香を最初から入れ、乱数初期化から、docs/ls2-config.toml の設定（窓を総局数の 50% まで広げる、replay_ratio 1、held-out 5%、基準比・最強比 1,000 局、固定の参照）で回す。最初の 8 時間は煙テスト（docs/ls2-settings.md §5）。搾取者 `lx` はまだ無く、本体が最強比で伸び始めてから足す。
9/13〜9/17 の系列（窓 10 万局を記憶して頭打ちになった旧系列）は `~/libra-run/ls-v1`（step 657,417、1,205,382 局）と `~/libra-run/lx-v1`（step 306,175）に改名して残し、再開しない。ls-v1 の archive の 646,699 は新しい系列の固定の参照に使う。

**旧・旧系列**: 9/13 までの本体 `ls`（ls-v1 の元）と搾取者 `lx` は、二飛香（docs/rules.md §3.2、2026-09-13）を入れた系列。`ls` は旧ルールの ls の最終チェックポイント（step 229,590）の `latest.pt`（モデル・オプティマイザ・step）を `checkpoints/latest.pt` に置いて始め、`lx` はゼロから（凍結相手 `lx/main.pt` は同じ `latest.pt` を先に置き、起動直後に `main_source` から作り直す）。局数・チャンク・自動計測の基準（`auto.anchor`）・Elo の累積は 0 から数え直し、旧系列とはつながない。外部エンジンとの自動計測は、相手が二飛香を知らないあいだ `[auto] match_games = 0` で止め（布石中に二飛香の手を指すと非合法手で負けになり、勝率が意味を持たない）、相手が二飛香に追いついた（fuseki-shogi-ai `eb3870f`、`Fuseki_Rules` 既定 2）ので 2026-09-13 に `match_games = 10`、`match_opponent_opt = "Threads=2,Fuseki_Rules=2"` に戻した（反映はランの起動し直しから）。
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
~/LibraShogi/bin/libra progress            # 進み具合の要約（--publish で GitHub の progress ブランチへ）
```

**操作は起動（`run`）と停止（`stop`）だけ**（2026-09-14 のユーザーの決定。一時停止・再開は廃止）。GPU や CPU を空けるときも停止し、終わったら起動する。どの順に押しても起動が空振りしないよう、`libra run` は起動前に次をする（log.txt に `start:` 行）:
- 停止処理中（STOP があり、まだ動いている）なら、止まるのを最大 180 秒待ってから起動する
- 止まっている run に残った STOP（と PAUSE）を消す
- STOP の無い稼働中の run には起動しない（`start: already running`）

`--run <id>` で run-id、`--root <dir>` で親ディレクトリを変えられる（既定 `~/libra-run/ls`）。
`run --config path.toml` は初回だけ有効。

Windows 側: `%USERPROFILE%\libra\` に `libra-run.bat`（本体 ls）/ `libra-run-lx.bat`（搾取者 lx）と、**ls と lx の両方に効く** `libra-stop.bat` / `libra-status.bat`。デスクトップに status / stop の写し（`install.sh` は前に写した libra-pause.bat / libra-resume.bat を消す）。**bat / vbs は `tools/windows/*.in` から `install.sh` が生成する**（ディストロ名と WSL 内の `bin/libra` の場所を埋める）。リポジトリを別の場所に置いたときや、ディストロを入れ替えたときは `install.sh` を回し直す。

## 3. Windows Update で再起動しても続くようにする

1. タスク スケジューラに「LibraShogi run」（本体 ls、ログオン 1 分後）と「LibraShogi run lx」（搾取者 lx、ログオン 2 分後）を登録済み（`%USERPROFILE%\libra\LibraShogi-run.xml` / `LibraShogi-run-lx.xml`）。それぞれ `wscript.exe libra-run-hidden.vbs` / `libra-run-lx-hidden.vbs` → `wsl.exe -d <ディストロ> -- <repo>/bin/libra [--run lx] run` を非表示で起動する（vbs は `install.sh` が生成する。ファイル名は変わらないので、タスクの登録はそのままでよい）。実行時間の上限なし。タスクの「失敗時に 1 分後に再起動（999 回まで）」は起動後の異常終了には効かない（9/13 に lx が CUDA の abort で落ちたまま 4.7 h 止まった）。
   **`libra run` は監視役**で、ランナー本体（子の `run --no-supervise`）が異常終了したら 60 秒後に起動し直す。起動し直さないのは、停止（STOP フラグ、終了コード 0）、二重起動（終了コード 3）、Ctrl+C・kill・`wsl --shutdown`（SIGINT/SIGTERM/SIGHUP）、待機中に STOP が置かれたとき。15 分未満で落ちるのが 5 回続いたら諦めて止まる。記録は log.txt の `supervisor:` 行と `stdout.log`。待機中（最大 60 秒）は status が `not running` と出る。
2. **自動ログオンは利用者が設定する**（`netplwiz`）。設定しないと再起動後にログオンするまで止まる。
3. 再起動後の損失はチェックポイント間隔（10 分）＋進行中の対局分。
4. Windows Update の「アクティブ時間」を広めに設定する。
5. `checkpoints/` の最新 2 世代を 1 日 1 回 Windows 側へ写す（未実装。cron か Task Scheduler で `cp`）。

再登録するとき:

```powershell
schtasks /Create /TN "LibraShogi run" /XML "%USERPROFILE%\libra\LibraShogi-run.xml" /F
schtasks /Create /TN "LibraShogi run lx" /XML "%USERPROFILE%\libra\LibraShogi-run-lx.xml" /F
```

Windows 側のファイルの正は `tools/windows/`（`install.sh` で `C:\Users\<user>\libra` とデスクトップへ写す）。

**管理コンソール（GUI）**: デスクトップの `libra-console.bat`（`tools/windows/libra-console.ps1`、PowerShell 5.1 + WinForms、ビルド不要）。本体 ls と搾取者 lx の状態（稼働中 / 停止処理中 / 停止、step・世代・総局数、局/日の 1 時間平均と実測、終局内訳、loss、GPU メモリ、最終チェックポイント、搾取者の対本体勝率、log.txt の末尾）を 15 秒ごとに `wsl.exe -d <ディストロ> -- <repo>/bin/libra --run <run> status --json --tail 40` で取り、局/日の推移を折れ線で出す。**縦長のウィンドウが前提**で、上のタブで **ls / lx / クラウド / クラウド履歴** を切り替える。タブの見出しに稼働状態（● 稼働中 / 停止処理中 / 停止、クラウドは段階、履歴は今月の費用）が色付きで出るので、裏のタブの run が止まっても分かる。run のタブは上から状態の表・ボタン・グラフ（局/日、Elo、対外対局、学習、学習目標、較正、処理時間、終局内訳、手数、ログ。局/日・Elo・対外対局以外はそのタブの run）。**処理時間** は学習器のループの実時間（5 分の窓）を「評価 GPU（自己対局のネットの評価の待ち。同じ GPU の別 run の待ちを含む）・探索 CPU（収集・根の証明探索・反映）・リーグ・学習・保存ほか」の割合で描き、注記に最新の窓の 1 ラウンドの段ごとの ms と学習の ms/step を出す（`status.json`・`metrics.jsonl` の `timing`、`libra-league/libra_league/looptime.py`。細かい段は `libra status` の `timing:` 行）。Elo と対外対局は 1 日 1 回の計測なので「グラフの期間」にかかわらず全期間を描き、**横軸は総局数**（「Elo の横軸」で時間にも切り替えられる。既定は総局数）。伸びを決めるのは時間ではなく局数で、止めている間・GPU を分け合う間・クラウドのワーカーを足した間は同じ時間でも進みが違うため（2026-09-18 の実測で 1 時間平均の局/日は 349,411〜3,792,184 の 10.9 倍の開き。時間軸だと最初の 1.4 万局が横幅の 10.4%、局数軸なら 1.8%）。点の局数は `status --history` の `games_at`（step から `metrics.jsonl` を引き直したもの。行の `games` は打ち終わった時刻の総局数で archive より約 5 万局あとなので使わない）。`games_at` の無い古い `libra` では時間の軸に落ちる。総局数の軸は最後に測った点で右端を打ち切る（時間の軸は右端が「今」なので、次の計測までの空白が出る）。**Elo のグラフは既定で「強さの目盛り」の 1 本だけ**（太い実線と 95% 区間の縦線。`status --json --history` の `rating`＝`libra rating` の点で、全部の対局をまとめて 1 本に当てはめた Elo。0 はいちばん古い重み）。2026-09-19 に、相手ごとの線を 5 本並べていたのを畳んだ（**相手ごとの線は 1 点 200 局で 95% 区間が ±60〜90 Elo あり、練習相手の入れ替えとじゃんけんで上下するので、目盛りが上がっていても下がって見える**。旅人さんの「Elo さがってませんか？線がいっぱいあってよくわからない」。decisions.md・measurements.md 同日）。内訳が要るときは「Elo の内訳を出す」を入れると、基準比（実線）・鎖（薄い点線。同じ step を別の方法で測るので同じ時刻に 2 つの値が並ぶ。鎖が上でも基準比が下がったわけではない）・固定の参照との差（参照ごとに別の色）が足される。**基準比と「対 <参照>」は 0 の意味が違う**（基準比の 0 は系列の最初の重み、「対 …」の 0 はその参照と互角）ので、上下をそのまま比べない。目盛りが出せない古い `libra` では今までどおり相手ごとの線に落ちる。**点線は「目安の線」**（`rating.curve_fit_recent`＝最近の伸びを log2(総局数) の直線に当てはめ、そのまま延ばしたもの）で、**点がこの線に乗っているかぎり頭打ちではない**。伸びは局数の対数にほぼ比例するので、ふつうの横軸では一定の伸びでも必ず右で寝て見え、形だけでは頭打ちか区別が付かない（2026-09-19 の旅人さんの「初期に比べると伸びが緩やかなので頭打ちなのかグラフからわかりにくい」）。**「Elo の横軸」に「総局数（対数）」を足した**ので、そちらにすると目安の線が直線になり、ずれが見やすい（右へ 1 目盛りで局数が 2 倍）。目安の線は**最近の伸び**（最後の点から 4 回の倍化ぶん、最低 4 点。`rating.RECENT_DOUBLINGS`）に当てはめる: 学習の初めは基本を覚えるぶん伸び方が違い、全部の点だと実データで残差 88・最新の点が +62 上に出て「加速している」と誤読させるため（同じ点で最近の 8 点なら傾き +203／2 倍・残差 21）。状態の表の「強さ（Elo）」は目盛り（値・95% 区間・前の点からの差）を先頭に、測った step・基準比・最強比を続けて、「強さ（固定の参照 Elo）」は参照ごとの最新の 1 行を出し、どちらも得点が 15%〜85% の外なら「・天井」と付ける（差が開きすぎて Elo の値も区間も当てにならない）。文言のテストは `tools/windows/tests/console-format.tests.ps1`（CI の console ジョブ）。位置・大きさ・選んだタブは `%LOCALAPPDATA%\LibraShogi\console-layout.json` に残り、次に開いたときに戻る。ボタンは「起動」「停止」（run ごとと「全部 起動」「全部 停止」）と自動計測の前倒しだけ。稼働中は「起動」、止まっているときは「停止」を押せなくする。停止処理中は「起動」を押してよい（止まってから起動する）。「起動」はタスク スケジューラの「LibraShogi run [lx]」を `schtasks /Run` で起動し（無ければ wsl.exe を直接起動）、5 分以内に稼働を確かめられなければステータスバーに赤で出す。操作の結果はステータスバーに 1〜15 分残る。局/日の履歴は `%LOCALAPPDATA%\LibraShogi\console-history.csv` に追記（7 日分を表示）。**クラウド** タブは vast.ai の自己対局ワーカーの起動 / 停止 / 候補 / 後始末と、段階・費用・回収局数・残高を出す（`bin/libra-vast`。§「自己対局ワーカー」の 4）。起動の確認には同じ GPU の過去の実績（借りた 1 時間あたりの有効局、100 万局あたりの費用、見込みの局数）も出る。**クラウド** タブには「打ち切り（これまで）」の行があり、入札で止められた回数・平均で何時間打つと 1 回止められるか・打ち切りが無ければ 100 万局あたりいくらだったかを、借りる前に読める（余計に払っている割合が 20% 以上なら橙色）。**クラウド履歴** タブは `bin/libra-vast history --json` でセッションごとの GPU・$/h・借りた時間・費用（借りた時間 × $/h ＋ 転送料。転送料は送った・取ってきたバイト数 × ホストの単価で、記録の無い古いセッションは見積もり）・有効局（回収局 − 学習側が古すぎて捨てた局。捨てた局は学習側の log.txt の `workers: dropped` の行をセッションの期間で数える）・局/日（ブリッジの時間で換算）・100 万局あたりの費用を一覧と棒グラフで出し、合計と今月の費用（1 ドル 150 円で円に直し、月 1 万円の上限との比）を出す。**打ち切りの損**（`libra_cloud/interrupts.py`。要約の 3〜4 行目、一覧の「打ち切りの損」の列、選んだ回の明細）は 2 つを足したもの: (1) **余分な準備代** ＝ 借り直したセッションが借りてからブリッジが動き出すまで（イメージの取得とビルドで 5〜15 分）の課金。局が 1 つも出ないのに払う。(2) **打てなかった時間** ＝ 最後に回収できた時刻から次のセッションが打ち始めるまで（ホストの上で打っていた途中の局も回収できずに消える）。借り直せなかったとき（残りが 0.25 時間未満、締め切り超過、候補なし）は予定の残り時間も丸ごと数える。この 2 つを引くと「打ち切りが無ければ 100 万局あたりいくらだったか」が出る。**お金の損と時間の損は別に出す**: 止まっている間は課金されないので費用にはほとんど出ないが、局/日には効く（2026-09-20 の実測では、お金は +2.3% なのに入札の経過時間の約 11% は 1 局も進んでいなかった。measurements.md 同日）。「同じ予算でいくつ買えるか」を気にするなら前者、「今日いくつ稼げるか」なら後者を見る。**借り方（入札 / on-demand）ごとの実効費用も並べる**ので、入札の値引き（同じホストで 16〜32%。decisions.md 2026-09-15）が打ち切りの損に負けていないかがその場で分かる（どちらも 3 回以上あって 1 割以上違うときだけ助言の 1 行が出る）。使う記録はすべて既にあるもの（session.json の `continues`・`hours`、instance.json の `t_rent`・`t_bridge`、result.json の `lost`、bridge.json の `last_pull`）で、新しく取るものは無い。**コンソールを直したら、手元で `git pull` してから `tools/windows/install.sh` で写し直す**（コンソールはリポジトリのスクリプトを Windows 側にコピーして動くので、main に入れただけでは変わらない。2026-09-19 に伝え漏れて旅人さんが自分で気付いた）。ディストロ名・`bin/libra`・run の置き場・desktop の exe は、`install.sh` が同じフォルダに書く `libra-paths.json` から読む（コマンド行の `-Distro` / `-Libra` などを渡せばそちらが優先。どちらも無ければ起動時に案内を出して終わる）。

再起動の手順（bat 版）: PC を再起動するときは `libra-stop.bat` で両 run を止めて（`libra-status.bat` で `not running` を確認）から再起動し、ログオン後に両タスクが自動で再開する。搾取者を止めたままにしたいときは WSL で `bin/libra --run lx stop` だけ実行する（本体は openings を読むだけなので影響しない）。

## 4. 別作業でリソースを空けるとき

- GPU を使う作業: 管理コンソールで「停止」→ 終わったら「起動」（一時停止は廃止）
- CPU だけ使う作業: そのままで良い（ワーカーは `nice 10`）
- 数日止めても再開時のコストはゼロ（損失は最後のチェックポイント以降の進行中の対局だけ）
- 推論は CUDA Graphs で捕獲している（`[selfplay] compile`、decisions.md 2026-09-14）。起動直後の最初のラウンドで捕獲するので、起動時は torch.compile の autotune に 5〜25 秒ほど掛かる（WSL の再起動で `/tmp` のキャッシュが消えた後は長め）。`log.txt` に `selfplay: inference model=compile(max-autotune)+cudagraph` が出れば有効、`eager` なら捕獲に失敗していて理由は `stdout.log`
- 学習も torch.compile している（`[train] compile`、既定 max-autotune、decisions.md 2026-09-16）。起動後の最初の学習で compile と autotune に十数秒掛かり、その回の `train.sec` が長く出る。`log.txt` に `train: model=compile(max-autotune)` が出れば有効、`eager` なら失敗していて理由は `stdout.log`

## 5. 設定を変えるとき

`~/libra-run/ls/config.toml` を編集し、`libra stop` → `libra run`。ネットの形（`[net]`）はチェックポイントと合わなくなるので変えない（変えるなら新しい run-id）。

## 6. 監視と自動計測

`libra status` の `games/day(1h)` を docs/measurements.md に週 1 回記録する（Windows では管理コンソール `libra-console.bat` で常時見える。§3）。`libra status --json [--tail N] [--history N]` は機械可読（`process`、`flags`、`state`、`status`、`log_tail`、`--history` で `metrics`・`evals`・`anchor`・`best`・`reference`・`matches`・`calib`・`archives`・`auto`・`auto_cfg`。計測の行には `games_at`（その重みを保存した時点の総局数。`scaling.games_of_step` で step から引き直す）が付く）。`status.json` の `timing` に処理時間の内訳（5 分の窓の段ごとの秒。`libra status` では `timing:` 行に割合と 1 ラウンド・1 ステップの ms）、`engine` に終局理由の内訳（ruling41、mate、sennichite、perpetual、max_ply）がある。

**進捗の時系列**: ランナーは `metrics.jsonl` に 5 分ごと（`run.metrics_minutes`）に 1 行追記する（step、局数、局/日、loss 系、終局内訳の累積カウンタ、搾取者成績、GPU、**メモリ**）。管理コンソールの「学習」「終局内訳」「手数」タブはこれを差分で割合にして描く。「局/日」タブは隣り合う行の `games_total` の差から出した 5 分平均（`status --history` の `gpd_5m`。間が 15 分を超えた行は出さない）を描く。

**メモリ（2026-09-20 から）**: `status.json` の `mem` とその行（`libra status` の `memory:` 行）に、ランナー自身の常駐メモリ `rss_mb`（`/proc/self/status` の `VmRSS`）とスワップ `swap_mb`（`VmSwap`）が入る。`metrics.jsonl` と `progress` ブランチの `<run-id>.json` にも `rss_mb`・`swap_mb`・`gpu_mb` として毎行入るので、**クラウドからも推移を読める**。見方は 3 つ: (1) `rss_mb` はリプレイの窓にほぼ比例する（実測 1 局 約 11.3 KB、2026-09-20）ので、窓が同じなのに節目ごとに階段状に増えていたら漏れ、(2) `swap_mb` が 0 でなければ窓が RAM に載っていない（学習のバッチ作りが遅くなり局/日が 1 割落ちる。2026-09-20 の measurements.md）、(3) `gpu_mb` は torch が確保したまま持っている GPU メモリ（自動計測は別プロセスなので、ここには出ない）。**自動計測の前後の値は `auto.log` と `log.txt` の `auto: … finished` の行にも出る**（`runner rss 15.2->15.3 GB (job peak 2.1 GB)`）。計測ジョブは別プロセスなので、終われば OS がそのメモリを丸ごと返す。前後でランナーの `rss` が増え続けていないかだけを見ればよい。

**探索値の較正（`calib.jsonl`）**: ランナーは 60 分ごと（`run.calib_minutes`、0 で無効）に、窓の最新 20,000 局（`run.calib_games`）で、全読みの手の探索値（`root_q` を手番側の得点の予測 (q+1)/2 とみなす）と実際の結果（勝 1・分 0.5・負 0）を 10 区間で比べ、`calib.jsonl` に 1 行足す。3 手目（両玉の直後）・布石／本将棋 × 先手／後手の番に分け、ECE（区間ごとの |予測 − 実際| を局面数で平均）・MCE（最大）・偏り（予測の平均 − 実際の平均）・Brier（得点の二乗誤差）・基準（実際の平均を言い続けたときの Brier）を出す。証明済みの手（|q| = 1）は予測ではないので別に数え（結果との一致数も出す）、リーグの対局（本体と過去の搾取者）は局ごと除く。管理コンソールの「較正」タブは ECE の推移と最新の全体の ECE・偏りを描く。区間ごとの曲線は `bin/libra [--run lx] calib [--games N] [--json]`（書き出し済みのチャンクから読むので、止めずにいつでも回せる。1 秒前後。まだチャンクに書かれていない最新の 100 局未満は入らないので、ランナーの値とわずかに母集団が違う）。`libra eval` の較正は別の世代との対局なので実力差で崩れる（docs/method-evidence.md §4.4）。こちらは同じネットの自己対局で、GUI が先後の選択に使う勝率の当たり方を見る。搾取者 lx ではランナーは書かない（記録が全部本体との対局で、自己対局の較正と意味が違う）。`bin/libra --run lx calib` で手で見ると「本体に対する搾取者の勝率の予測」の当たり方になる。

**投了を入れたときの効き目（`libra resign`）**: `bin/libra [--run lx] resign [--games N] [--thresholds …] [--runs …] [--json]`。**投了はまだ実装していない**（decisions.md 2026-09-19）。このコマンドは、最後まで打った棋譜の上で「投了を入れていたらどこで打ち切れたか」を数え直し、しきい値 T と連続の手数 K の組ごとに、打ち切った局の割合・手数の節約率・ネットの評価の節約率（GPU が律速なので局/日の見込みの倍率 1/(1−節約)）・誤投了の割合を出す。読み取りだけで GPU も要らないので、**ランを止めずにいつでも回せる**（2 万局で数秒）。読み方: 「評価の節約」がそのまま局/日の伸びの見込み、「実は勝ち」が誤投了で、AlphaGo Zero はここを 5% 未満に保つ。

**一般化の物差し（`gen`、held-out）**: `[run] heldout_every_chunks` > 0 の run では、チャンク番号がその倍数のチャンク（100 局）の局を学習に使わず held-out に持つ（新しい `heldout_games` 局まで。再開時も同じ振り分けで読み直す）。ランナーは `gen_games`（局数の倍数を越えるごと。0 なら `gen_minutes` の時間ごと）に、窓の中と held-out から `gen_positions`（4,000）局面ずつ取り、価値 V＝W−L と学習目標の相関・二乗誤差、方策の交差エントロピー・一致率を布石／本将棋で測って status・metrics.jsonl の `gen` と log.txt に出す（libra_league/genprof.py）。**窓の中だけ良くて held-out が悪ければ、ネットは窓の局の結果を記憶していて一般化していない**（2026-09-17 の診断: 旧 ls は本将棋の価値の相関が窓の中 0.94・外 0.43。docs/status-2026-09-17.md §10）。見る目安は docs/restart-plan.md §3 M1。**この物差しで見るのは丸暗記（窓の中との差）と下がっていないかだけで、上がり続けることは期待しない**（2026-09-19。本将棋の目標はその対局の結果 z そのもので相関に上限があり、held-out も 20,000 局で入れ替わる。強さが +364 Elo 伸びた区間で +0.008 ± 0.005 しか動かない。docs/gen-metric-2026-09-19.md）。保存済みの重みを任意のチャンクで測るのは `bin/libra [--run <id>] genprof [--ckpt ckpt.pt] [--chunks 100,2000] [--n-chunks 10] [--positions 2000] [--json]`（既定は最新から 1,000 チャンクごとに 6 点。run をまたいで比べられる）。
**合否の関数（`bin/libra [--run <id>] review [--json] [--set name=value]`）**: 保存済みの値（metrics.jsonl の `gen`、`eval/best.jsonl`・`anchor.jsonl`・`reference.jsonl`、status の局/日）を読み、M1〜M4 と局/日のそれぞれに「続ける／注意／見直し／まだ無い」を付け、全体の判定を出す（docs/restart-plan.md §3 M6・§7 P4。合否の目安は docs/ls2-settings.md §5）。閾値は `--set gen_max_gap=0.1` のように上書きでき、再計測は要らない。週 1 の見直しはこの出力を docs/weekly/ に写して行う。
**リプレイの窓の拡大**: `[train] window_frac` > 0 なら窓は max(`window_games`, 総局数 × window_frac)（上限 `window_games_max`、0 で無制限）。0 なら固定（旧 ls・lx）。1 局 約 5 KB なので RAM の空きで上限を決める（100 万局で約 5 GB）。status の `window` は「窓の中の局数/今の窓の大きさ」。

**自動計測（`[auto]`、本体 ls のみ有効）**: `every_games`（総局数がその倍数（40 万・80 万…）を越えるごと。越えたら 10 分を待たずにチェックポイントを取る。0 なら節目なしで、`eval-now` / `match-now` のときだけ測る。**時間区切り `every_hours` は 2026-09-19 に廃止した**。局/日が PC の利用状況で変わるので時間では測る間隔が定まらず、2026-09-17 から局数だけで動いていたため）にチェックポイントを `checkpoints/archive/` に残し、`libra eval`（`eval_sims`=96）で 2 通りの対局をする。

- **基準比（`anchor_games`=100、これが主）**: 固定の基準ネット（`state.json` の `auto.anchor`）と対局する。結果は `eval/anchor-*.json` と 1 行ずつの `eval/anchor.jsonl`（`elo` は基準の `offset` を足した値）。基準に `anchor_rebaseline`（0.85）以上勝ったら基準を新しい世代に置き換え、そこまでの差を `offset` に足す。計測が積み上がって結果が出る前に基準が替わっても、各結果は実際に打った相手（積んだときの基準）の `offset` で数え、置き換えは今の基準と打った結果のときだけ行う。ランナーは起動時に `eval/anchor-*.json` から `anchor.jsonl` と今の基準を数え直し、違っていれば古いファイルを `anchor.jsonl.bak-<時刻>` に残して書き直す（2026-09-18 の誤りの修正）。最強と基準が同じ重みで `best_games` と `anchor_games` が同じなら、同じ組の対局になるので打たずに最強比の結果を写す（履歴に `reused`）。連続世代どうしの差（1 日で +30 Elo 程度）は 100 局の測定幅 ±70 Elo に埋もれ、鎖にすると誤差が回数の平方根で積み上がるため、基準との大きな差で測る。
- **鎖（`chain_eval`、補助）**: 直前の archive との差を足した累積。基準が直前の archive と同じときは同じ対局になるので 1 回で兼ねる。

- **最強比（`best_games`、docs/restart-plan.md §3 M2）**: これまでで最強の保存済み（`state.json` の `auto.best`）と対局する。結果は `eval/best-*.json` と `eval/best.jsonl`。新しい世代の 95% 区間の下限が 0 を超えたら最強を置き換え、そうでなければ足踏みを数え、`best_stall_alert`（3）回続いたら log.txt に WARNING を出す（見直しの合図）。
- **固定の参照（`reference_ckpts`・`reference_games`、同 M4）**: 設定に書いた重みのファイル（例: 旧 ls の 646,699 と実験の win1m.pt）と毎回対局し、`eval/reference.jsonl` に残す。run をまたいで同じ相手なので、絶対の物差しになる。コンソールの Elo タブに「対 <参照>」の線（参照ごとに別の色）で、状態の表に「強さ（固定の参照 Elo）」の行で出る。**入れ替えは自動**（`reference_rotate = 0.8`、2026-09-18 に追加）: 参照に 80% 以上勝ったら、その参照を外し、**そのときの archive** を代わりの参照にする。入れ替えは `eval/references.jsonl` に 1 行残り、今の参照は `bin/libra status --json` の `auto.references` で読める（今の参照は「設定の `reference_ckpts` ＋ 入れ替えで足したもの − 自動で外したもの」。**設定が正**なので、`config/<run-id>.toml` から参照を消せば停止 → 起動でその場で打たなくなり、自動で外した参照は設定に残っていても戻らない）。入れ替えを見送ったときは理由が `log.txt` に出る（2026-09-20 に、`~` 始まりのパスが一致せず280 万局まで黙って見送られていたため。同日の measurements.md）。手で入れ替えるときも同じ規則にする: **参照に 80% 以上勝つようになったら、そのときの archive を参照に足し、80% を超えた参照は設定から外す**（勝率が 1 に寄ると Elo が縮み、伸びていても曲線が寝て見える。docs/scaling-2026-09-18.md §6.4）。外しても曲線は短くならない: 過去の行は `eval/reference.jsonl` に残り、`libra scaling` は全期間を読む。**足す参照はこの run 自身の archive にする**。自分の archive なら「その局数で自分対自分 ＝ 0 Elo」が測らなくても分かるので、古い参照が既に天井に着いていて帯の中で重ならなくても、新しい参照が古い目盛りに必ずつながる。

**全部の対局から一度に Elo を出す（`libra rating`）**: 鎖（基準比の累積）や「1 つの参照との差」は、相手が替わるたびに誤差と**非推移性**（じゃんけん）が積み上がる。`bin/libra rating [--curve] [--json] [--anchor <点>]` は、記録した対局（最強比・基準比・固定の参照・外部計測）を**全部まとめて 1 本の Elo の目盛りに当てはめる**（Bradley-Terry の最尤推定、Hunter 2004 の MM。docs/method-evidence.md §2.9）。参照を入れ替えても鎖を継ぎ足さないので、物差しの入れ替えで目盛りが動かない。出力の χ²/自由度が**じゃんけん度**で、1 なら Elo の 1 本の目盛りで説明でき、大きいほど「相手によって強さが入れ替わる」。`--curve` で総局数に対する傾き（2 倍あたりの Elo）も出る。同じ値は `progress/<run-id>.json` の `rating` と `.md` の表にも入る。**この run 自身の archive を参照にした行は、同じ step の点にまとめる**ので、参照として打った対局もそのまま目盛りに効く。1 局も勝っていない点は Elo が発散する（Hunter の存在条件）ので `dropped` に出して外す。

**伸びの曲線（`libra scaling`）**: 「局を何倍にすると何 Elo 伸びるか」を、固定の参照の保存済みの記録から出す（対局はし直さない）。`bin/libra scaling`（`--json`、`--cost-per-1m`、`--band`）。横軸は総局数、縦軸は固定の参照との Elo で、得点が 0.2〜0.8 に収まる点だけを log2(総局数) の直線に当てはめ、傾き（2 倍あたりの Elo）・100 万局あたりの Elo・1 Elo あたりの費用を出す。参照が複数あれば、両方が帯の中にある点での差の平均でつないで 1 本にする（直に重ならない参照も、間の参照をたどってつなぐ）。どこにもつながらない参照は曲線から落ち、注意に出る。同じ値は `progress/<run-id>.json` の `scaling` と `.md` の表にも入る。読み方と 2026-09-18 の結果は docs/scaling-2026-09-18.md §6。

同じ周期で `libra match`（`match_games`=10 局、`match_go`="movetime 1000"、相手は fuseki_usi_server.py に `match_opponent_opt`="Threads=2,Fuseki_Rules=2"）も回す。**Libra 側の重みは、その節目に固定したもので打つ**（ジョブを積むときに `--ckpt` へ `checkpoints/archive/ckpt_*.pt`（`match-now` で節目以外のときはその時点の `checkpoints/ckpt_*.pt`）を渡し、同時にその時点の `latest.onnx`（同じチェックポイントの `latest.pt` から書き出した同じ重み）を隣に `ckpt_*.onnx` として写す。1 回あたり約 40 MB。書き出しが古いままで写せなかったときは match が `.pt` から書き出し、`.pt` も回転で消えていたときは最新の重みで打ってログに warning を出す）。`latest.onnx` は中身が動く別名で、計測待ちの間に世代が変わって時系列の比較にならないため（docs/restart-plan.md §7 P2、2026-09-18）。結果の行の step は ONNX の名前から読む。どちらも別プロセス（`auto.log`）で GPU を共有し、結果は `eval/auto-*.json` と `matches/auto-*.summary.json`。前倒しは `libra eval-now` / `libra match-now`（フラグ EVAL_NOW / MATCH_NOW。次のチェックポイントで実行。コンソールの「今すぐ自己評価」「今すぐ対外対局」）。相手側は 41 手目以降を必ずやねうら王（水匠5 の評価関数）に中継するので「方策ネットだけの相手」は無い。ランナーを stop すると実行中のジョブは止め、再開後に積み直す。ランナーが異常終了した場合は、起動し直したランナーが残ったジョブ（孫の相手エンジンを含むプロセス グループ）を止めて積み直す。途中まで書いた出力は `<out>.interrupted` に改名する。

## 設定の管理（2026-09-18 から）

**稼働中のランの設定の正は、リポジトリの `config/<run-id>.toml`。** `~/libra-run/<run>/config.toml` は起動のたびにそこから作り直される写しで、手で直しても次の起動で上書きされる（2026-09-18 のユーザーの依頼。手で直すのが面倒で、PC を入れ替えると設定が失われるため）。

- **直すのは Claude、反映はユーザー**。`config/<run-id>.toml` を直して main に入れ、ユーザーがコンソールの停止 → 起動を押すと効く。**手元のチェックアウトが古くても効く**（ランナーは `git fetch` して `origin/main:config/<run-id>.toml` の中身だけを読む。作業ツリー・HEAD・ローカルのブランチには触れない）。
- **ただし効くのは「今のプログラムが知っている鍵」だけ。** 設定は `origin/main` から読むのに**プログラムは手元の作業ツリー**なので、コードも要る変更（新しい鍵）では `git pull` を忘れるとその鍵だけが黙って無視される。2026-09-19 に `match_go_opp` がこれで効かず、相手まで 1 手 400 回になった 40 局を「勝率 100%」として記録した（measurements.md 同日）。**知らない鍵があれば起動時に log.txt と `bin/libra config` に WARNING を出す**（`config.unknown_keys`）。**設定の変更を頼むときは、既定で「`cd ~/LibraShogi && git pull` → コンソールの停止 → 起動」を頼む**（コードが要る変更かどうかを毎回見分けるより確実）。
- 読む順（後が勝つ）: 既定値 → `config/<run-id>.toml`（`origin/main` → 取れなければ作業ツリー）→ `~/libra-run/<run>/config.local.toml`（**その PC だけの上書き**。置き場所など。リポジトリには入れない）。`libra run --config <path>` を付けたときはこれまで通りそのファイルだけを読む（1 回きりの起動）。
- 作り直すとき、前の `config.toml` は `config.toml.bak-<時刻>` に残し、変わったキーを log.txt に並べる。ブリッジ（libra-cloud）・ワーカーの束・`libra status` などはこれまで通り `<run>/config.toml` を読むので、影響しない。
- **見るとき**: `bin/libra [--run lx] config`（設定の正・その PC の上書き・**反映待ちの違い**を出す。書き換えない）。`--json` もある。
- **リポジトリにまだ無いラン**: そのランの `config.toml` をそのまま写して作る（中身を推測して書くと学習の設定を黙って変えるため）。`bin/libra [--run lx] config --adopt`、または `progress` ブランチの `progress/<run-id>-config.toml`（効いている設定そのもの。ホームは `~` に直してある）から。ランナーは**リポジトリの作業ツリーには書かない**（あとで `git pull` とぶつかるため）。
- 環境変数 `LIBRA_CONFIG_REF` で読むところを変えられる（既定 `origin/main`、`none` で作業ツリーのファイル）。
- 公開リポジトリなので、`config/` には絶対パスを書かず `~` を使い（読むときに展開される）、秘密情報を置かない。

**進捗の書き出し（`[progress]`、既定は無効）**: `~/libra-run` は手元の PC にしか無いので、外（クラウドのセッション、別の端末）からは数値が読めない。`enabled = true` にすると、ランナーは自動計測が動いた節目と `heartbeat_minutes`（180 分）ごとに `libra progress --publish` を別プロセスで起動し、要約を GitHub の `progress` ブランチ（`branch`、`dir` の下）へ push する。

- 置くのは `progress/<run-id>-config.toml`（効いている設定そのもの。ホームは `~`）と、`progress/<run-id>.json`（機械可読。`libra status --json --history` と `libra review --json` を絞ったもの: step・総局数・局/日・窓、`auto` の基準と最強、`best`・`anchor`・`reference` の全行、外部計測の得点、`metrics` の直近 `metrics_points` 点、物差しの判定）と `progress/<run-id>.md`（同じ中身の短い表。GitHub でそのまま読める）。**手で直さない**（次の書き出しで上書きされる）。
- **絶対パスはファイル名に直し、ホームは `~` にする**（`libra_league/progress.py` の `scrub`）。棋譜・重み・秘密情報は入れない。公開リポジトリなので、足す項目は必ずこの処理を通す。
- **main と作業ツリーには触れない**。git の下位コマンド（`hash-object` → `update-index` → `write-tree` → `commit-tree`）で作ったコミットを `<commit>:refs/heads/progress` に push するだけなので、稼働中に別の作業をしていても邪魔せず、HEAD もローカルのブランチも動かず、CI も走らない。中身が前回と同じ回は積まない。ほかから同じブランチに push があって弾かれたら、読み直して積み直す（3 回まで、2・4 秒待ち）。
- push には git の認証が要る（`gh auth setup-git` 済みの credential helper）。手で試すのは `bin/libra progress --out /tmp/x`（書き出すだけ）と `bin/libra progress --publish`。`--no-push` でコミットまで、`--repo` でリポジトリ、`--points` で残す点の数を変えられる。
- 設定（`~/libra-run/ls/config.toml`）:

```toml
[progress]
enabled = true
branch = "progress"       # main には入れない
heartbeat_minutes = 180   # 節目が来なくてもこの間隔で書き出す
metrics_points = 120      # metrics.jsonl から残す点の数
```

## 7. 計測（外部エンジンとの対局）

```bash
~/LibraShogi/bin/libra match --games 20 --go "movetime 3000"      # Libra（latest.onnx）対 fuseki_usi_server.py（やねうら王＋水匠5 中継）
# 節目の重みで手で打つ（自動計測と同じ条件。`--ckpt` の隣に .onnx が無ければ書き出す。相手のルールの版は明示する）
~/LibraShogi/bin/libra match --games 10 --go "movetime 1000" --ckpt ~/libra-run/ls/checkpoints/archive/ckpt_000028908.pt \
  --opponent-opt Threads=2 --opponent-opt Fuseki_Rules=2
~/LibraShogi/bin/libra eval --a ckpt1.pt --b ckpt2.pt --games 200  # 世代間 Elo と較正
```

`match` は `~/libra-run/ls/matches/<時刻>.jsonl`（1 局 1 行）と `.summary.json`、USI ログ `.log` を書く。裁定は libra-sim（docs/rules.md）。相手のバージョンとハッシュは docs/protocol.md §5。
GPU を L-S と共有するので、計測中は ls・lx を停止するか、局/日が落ちることを承知で回す。

### 7.0 外部計測（布石は Libra 同士、41 手目から やねうら王／水匠5）

**2026-09-20 に相手を総取り替えした**（ユーザーの決定「方策ネット＋やねうら王/水匠5 での対外対局には
限界があるのでやめる。40 手目までは Elo 測定で測った最強 Libra 同士で行い、41 手目から 最強 Libra vs
やねうら王/水匠5 で対局する」）。それまでの相手（fuseki_usi_server.py ＝ 相手の方策ネットが布石を打ち、
41 手目から やねうら王へ中継）は、布石が弱いぶん Libra が勝ちすぎて物差しにならなくなっていた
（読む量のハンデを 1 手 3,000 回 → 400 回 → 8 回と下げても得点 0.875 のまま。docs/decisions.md 2026-09-19・20）。

新しい形:

1. **布石（1〜40 手目）は計測する重みが両陣とも作る**（`match_fuseki = "self"`）。置く側・選ぶ側は無く、
   同じ Libra が 40 手すべてを打つ。根の Gumbel ノイズで手が散るので、局ごとに違う布石になる。
   41 手目の裁定（docs/rules.md §3.4）で終わった布石は捨てて打ち直す。
   自己対局の棋譜から 41 手目の局面を借りる形（`selfplay`）と、SFEN の一覧を渡す形（ファイル名）もある。
2. **同じ局面を先後入れ替えて 2 局ずつ打つ。** 布石の有利不利が打ち消し合い、どちらの陣を持っても測れる。
3. **41 手目から 相手はふつうの将棋エンジン**（やねうら王＋水匠5 の評価、`Threads=1`）。布石を指せなくてよい。
4. **段は相手の読む節点数**。Libra 側は動かさない（下の「目盛りに乗せる」）。

```bash
# 手で 1 回打つ（自動計測と同じ条件。布石は Libra が 10 個作り、それぞれ先後入れ替えて 20 局）
~/LibraShogi/bin/libra match --games 20 --fuseki self \
  --go "nodes 96" --go-opp "nodes 10000" --libra-opt Mate_Nodes=200 --libra-standard \
  --ckpt ~/libra-run/ls/checkpoints/archive/ckpt_000246009.pt \
  --opponent ~/fuseki-shogi-ai/vendor/YaneuraOu/source/YaneuraOu-by-gcc --opponent-cwd ~/fuseki-shogi-ai \
  --opponent-opt EvalDir=vendor/yaneuraou_eval --opponent-opt Threads=1
```

自動計測では `config/<run-id>.toml` の `[auto]` に書く（反映は `cd ~/LibraShogi && git pull` →
コンソールの停止 → 起動）。

| 鍵 | 意味 |
|---|---|
| `match_fuseki` | `self`（Libra が両陣とも作る。今の形）／`engine`（両エンジンに布石を打たせる。昔の形）／`selfplay`（run の自己対局の 41 手目の局面）／ファイル名 |
| `match_opponent`・`match_opponent_cwd` | 相手の起動コマンドと作業フォルダ（空なら布石つきの fuseki_usi_server.py） |
| `match_go` | Libra 側の `go`。`nodes N` なら読む回数が固定で、GPU の混み具合で変わらない |
| `match_go_opp` | 相手の `go`。**コンマで区切ると段に分かれ**、局数を等分して段ごとに 1 ジョブ走る |
| `match_libra_opt`・`match_opponent_opt` | それぞれへの `setoption`（コンマ区切り） |
| `match_libra_standard` | Libra 側が自己評価と同じ読みだという申告（下） |
| `match_use_best` | 打つ重みを「最強比が決めた最強」にする（既定 true。`best_games = 0` の run では効かない） |

**目盛りに乗せる（大事）。** `libra rating` は条件の違う相手を別の点として扱う。Libra 側も同じで、
**Libra の読む量を自己評価（`eval_sims`）と変えると、「その Libra と相手」だけで閉じた塊になり、
強連結の条件で目盛りから丸ごと落ちる**。2026-09-19〜20 のハンデ付きの計測が実際にそうなっていて、
得点は残っても `libra rating` には 1 つも効いていなかった。そこで新しい形では

- Libra 側を `match_go = "nodes 96"`（＝ `[auto] eval_sims`）と `match_libra_opt = "Mate_Nodes=200"`
  （＝ `[search] mate_nodes_root`）にして、**自己評価とまったく同じ読み**にする。
- `match_libra_standard = true` を立てる。`libra rating` はこの申告がある結果の Libra 側を素の `step N`
  の点として扱う（相手と `go` が違っても分けない）。**違う読みにしたらこの申告を外す。**

点の名前は `YaneuraOu ... [EvalDir=…, Threads=1, nodes 10000]` の形。ルールの合わせ込み（`EnteringKingRule`）や
定跡の停止（`BookFile`・`USI_OwnBook`）、`USI_Hash` は名前に入れない（段を決めるものではない）。

**この形では布石の強さは測らない**（両陣とも Libra が作るので打ち消し合う）。測るのは 41 手目以降の将棋で、
布石を含めた強さは `libra rating` の内部の物差し（最強比・基準比・固定の参照）が見ている。

**どの重みで打つか。** `match_use_best`（既定 true）なら**最強比が決めた最強**で打つ。節目のジョブは
最強比 → 基準比 → 参照 → 外部計測 の順に待ち行列へ入り、待ち行列は先入れ先出しなので、外部計測が
始まるときには同じ節目の最強比（1,000 局）が終わっている（2026-09-20 のユーザーの決定「Elo 測定が
40 万局ごとに自動起動するので、その結果を待ってから最強の Libra を決めたい」）。
**新しい世代が有意に勝てなかった節目では前の最強で打ち直す**ので、外部計測の点が動かない節目がある
（同じ点に局数が足されて区間が狭まる）。最強の重みが消えていたら節目の重みで打つ（計測を落とさない）。

**段の決め方。** 得点が 0.4〜0.6 に入る段を使う（1 局あたりの情報が最大で、区間がいちばん狭い）。
釣り合う段が分からないうちは `match_go_opp` に複数書いて挟む。段をまたいで布石は同じなので（`--fuseki-seed`
が揃う）、段どうしの比較が対になる。年末の 100 局の基準値マッチは持ち時間をそろえたまま別に行う。

## 7.1 設定値のオフライン比較（`libra abtest`）

「仮」のまま残っている学習の設定値（docs/ls2-settings.md の区分「記録なし（仮）」）を決めるための手順（docs/restart-plan.md §0、同 §6 の段の条件）。
**保存済みの重みと、その重みが持っていた窓**から、設定だけを変えた「腕」を同じ step だけ学習し、腕どうしと元の重みを対局させる。

```bash
# 布石の価値目標 λ（0.5 = z と V̂41 の平均 / 1.0 = 実際の勝敗 z だけ）を、最新の archive から比べる
~/LibraShogi/bin/libra abtest --ckpt ~/libra-run/ls/checkpoints/archive/ckpt_000062426.pt \
  --arm lam05:train.lambda_z=0.5 --arm lam10:train.lambda_z=1.0 --steps 5000 --games 1000
```

- 稼働中の run には**書かない**（リプレイと重みを読むだけ）。出力は `~/libra-run/experiments/<時刻>-abtest/`（`abtest.json` と腕ごとの `.pt`）。
- 腕の間で**学習する局面と鏡映は 1 バッチずつ同じ**（同じ seed のサンプラ）。違うのは設定だけになる。
- 窓の位置（`chunk_index`）と総局数はチェックポイントに保存された値を使う（＝その節目の窓）。`--chunk-index` / `--games-total` で変えられる。
- 一般化の物差しは 2 通り出る。`gen_z` は目標を λ = 1.0（実際の勝敗）にして測った値で**腕の間で比べられる**もの、`gen_own` は腕自身の λ で測った値（学習の損失と同じ物差し）。
- 対局は評価ハーネス（`libra eval` と同じ条件、読み 96・根のノイズあり）。`--games 0` で学習と物差しだけ。
- **C++ の部分を直した版に上げたときは、回す前に `PATH=$PWD/.venv/bin:$PATH cmake --build build` でビルドし直す**（`git pull` だけでは `librasearch` の `.so` が古いまま。2026-09-18 に σ の比較が `set_side_config` が無いと言って止まった）。動いているランは読み込み済みの `.so` を使い続けるので、ビルドし直しても止まらない。
- `--publish` を付けると、結果（`experiments/<名前>.json` と `.md`）を `progress` ブランチへ push する（`~/libra-run` は手元の PC にしか無いので、外から結果を読めるようにするため。main と作業ツリーには触れない。`libra eval` にも同じ `--publish` がある）。
- 目安の時間（RTX 5070 Ti を専有、窓 47 万局、5,000 step、1,000 局 × 3）: 窓の読み込み 1〜2 分、腕 1 つの学習 約 11 分、対局 1 本 約 8 分で**合わせて 1 時間前後**。GPU を使うので、回す間は ls・lx を停止する（CLAUDE.md「稼働中のランの扱い」）。
- 扱うのは学習側（`[train]`）の設定。探索（`[search]`）の設定は窓の中の棋譜と方策の目標を作り直さないと比べられないので、この命令では変えても意味がない。

**探索の σ の形（`gumbel_rescale`）を比べるとき**は、同じ重みで側ごとに σ を変えて打つ（`libra eval --b-set`。B 側＝奇数枠の先手だけ別の設定で読む）。

```bash
~/LibraShogi/bin/libra eval --a ~/libra-run/ls/checkpoints/archive/ckpt_000062426.pt \
  --b ~/libra-run/ls/checkpoints/archive/ckpt_000062426.pt --games 1000 --sims 96 \
  --b-set gumbel_rescale=true --b-set c_scale=0.1
```

`--b-set` を付けた対局の結果は `~/libra-run/experiments/<時刻>-sigma.json`（`--out` で変えられる）。run の `eval/` には置かない（コンソールの Elo の一覧に世代間の計測として並んでしまうため）。

側ごとに変えられるのは読む手の選び方だけ（`full_sims`・`fast_sims`・`gumbel_m_full`・`gumbel_m_fast`・`c_visit`・`c_scale`・`gumbel_rescale`・`gumbel_noise`・`cpuct`）。ほかの鍵を渡すと例外になる（枠ごとに棋譜や記録の形が変わってしまうため）。
**これは「読む手の選び方」の比較で、σ が学習データ（方策の目標）の形を変える分は測れない**（窓の中の目標は今の σ で作った棋譜のもの）。そこまで見るなら σ を変えた自己対局を別に回すことになる。

## 8. desktop で Libra と指す（自分で体感する）

desktop（天秤将棋GUI 0.10.3、`%LOCALAPPDATA%\天秤将棋GUI\tenbin-shogi-gui.exe`）には Windows 版 `libra.exe` を「LibraShogi 0.0.2」として登録済み（**v0.2 から名乗りは `LibraShogi 0.2.0` に変わる。GUI の一覧の表示名が変わるので、配布物を入れ替えたら登録し直す**）（`%APPDATA%\com.fusekishogi.tenbin\engines\libra\engine\`。モデルは同じフォルダの `libra.onnx`）。2026-09-14 から DirectML 版の DLL（`onnxruntime.dll` 1.24.4・`DirectML.dll`）に差し替え、GPU で読む（`isready` で `info string … provider dml`）。以前の CPU 版は同じフォルダの `*.cpu-prev`、以前の exe は `libra.exe.prev`。学習中の ls・lx と GPU を共有するので、desktop で読ませている間は局/日が少し落ちる。 desktop 0.10.0 から、布石に対応したエンジンは本将棋（41 手目以降）の席にも選べるので、**1 回の登録で 1 手目から終局まで指せる**（布石と本将棋の両方に「LibraShogi」を選ぶ。同じ id なので 1 本のプロセスが続けて指す）。GUI が終局（千日手・入玉宣言・手数上限）を裁き、宣言できるエンジンには毎手 `Declare_Win=true` を送る（docs/protocol.md §1）。天秤将棋の両玉と先後の選択も Libra の `scale.json` と `winrate` で決まる（同 §2）。

最新のネットで指すには管理コンソールの「desktop で対局」を押す。WSL の `~/libra-run/ls/checkpoints/latest.onnx` をそのフォルダの `libra.onnx` に写し（`libra.onnx.json` に step と時刻を残す）、desktop を起動する。desktop が既に起動しているときはモデルだけ更新するので、エンジンを立て直す（desktop を開き直す）と新しいネットになる。対局画面でエンジンに「LibraShogi」を選ぶ。無人で行うには `powershell -File libra-console.ps1 -UpdateDesktopModel`。

GPU（CUDA）で読ませたいときは「エンジン」→「実行ファイルを選んで追加」で `C:\Windows\System32\wsl.exe` を選び、引数に `-d <ディストロ> -- <repo>/bin/libra-usi` を入れる（例: `-d Ubuntu-24.04 -- /home/<user>/LibraShogi/bin/libra-usi`。モデルは常に latest.onnx、学習中の GPU と共有）。

## 玉配置表（libra-scale）

`bin/libra-scale build --sims 1600` で `~/libra-run/ls/scale/scale.json` を作り、`bin/libra-scale verify --top 48 --games 100` で
釣り合い集合を検証対局の信頼区間で決め直す。エンジンには `setoption name Scale_Table value <path>` で渡す（`bin/libra match` は
`--libra-opt Scale_Table=<path>`）。表は世代ごとに作り直す（探索値は数分、検証対局は最終世代だけ本格的に）。
verify の最後のログ `v_hat - verify winrate`（scale.json の `verify.v_hat_minus_w`・`v_hat_minus_w_se`。検証した組の V̂ と実際の勝率の差の平均）を measurements.md に書く。
平均が標準誤差の 2 倍を超えて 0 から離れたら、学習目標の偏りを疑う（docs/method-evidence.md §4.4。v0.1 は −0.008）。
**全組の作り直し**（v0.1 の表、2026-09-15〜。decisions.md 同日）は `bin/libra-scale seq run --dir ~/libra-run/ls/scale/seq-v0.1 --table ~/libra-run/ls/scale/scale-v0.1.json --notify windows`（手順は libra-scale/README.md の seq）。この作り直しの間は ls・lx を止める（ユーザーの決定）。GPU を使う Windows のアプリも閉じる（共有すると専有の 0.30 倍、measurements.md 2026-09-15 21:55）。最初の約 5.5 時間は対称な置き方の 27 組だけを打つ。後手に傾いたら `~/libra-run/ls/scale/seq-v0.1/ALERT.txt` と log.txt の `ALERT` 行に出るので、`bin/libra-scale seq status --dir ~/libra-run/ls/scale/seq-v0.1` で見る。止めるときは同じディレクトリに `STOP` を置き、再開は同じ `seq run --dir`（打ち切りと局数は続きから）。
それ以外のリリースごとの build・verify は、**ls・lx は止めずに、GPU を共有したまま回す**（decisions.md 2026-09-15、ユーザーの決定）。検証対局の勝率は共有しても変わらず、変わるのは所要時間と、その間の ls の局/日（2026-09-11 の v0 では約半分）だけ。GPU メモリは ls 約 9.3 GB ＋ lx 約 1.7 GB ＋ verify 約 2.9 GB（16 GB 中）。回した時間と局/日の低下は measurements.md に書く。

## 搾取者リーグ（Main exploiter）

搾取者 lx（`~/libra-run/lx`、1.9M、64 局同時）は凍結した本体（`lx/main.pt`）と対局し、自分の手だけを学習する。偶数枠で搾取者が先手。本体の手番の手は本体のネットで木を丸ごと読む。lx の手番の手は、価値を lx のネット、木の中の本体の手番の葉の方策を本体のネットから取る（`[exploiter] opponent_prior`、既定 true。本体の応手を本体の方策で予測する。docs/exploiter-literature.md）。

**凍結相手の作り直し**: `[exploiter]` の `refresh_hours`（24）が過ぎたとき、または `main_source`（本体の `checkpoints/latest.pt`）の step が凍結相手より `refresh_steps`（25,000。本体の約 4.9k step/h で約 5 h、vast.ai のワーカーを足すと約 3 h）以上進んだとき（`refresh_check_minutes` の 5 分ごとに step だけ読む）に、`main_source` を `main_ckpt` に写し、対本体成績を履歴（`state.json` の `exploiter.history`）へ移して 0 から数え直す。本体が強くなると古い相手への勝率が飽和し（2026-09-12 に 98.3%）、収束判定「対本体勝率が頭打ち」が意味を失うため。初回は起動直後に行う。

**布石**: `openings_minutes`（60）ごとに、作り直してからのチャンクだけから搾取者が勝った布石を `openings_out` に書く。本体 ls は `[selfplay] openings` でこれを読み、新規対局の 10%（`openings_prob`）をそこから始める。相手を作り直した時点で布石は空にする（古い相手の穴なので本体に渡さない）。手動で書き出すときは `bin/libra --run lx openings`。

**本体と過去の搾取者の対局**（`[league]`、2026-09-14 から）: 本体 ls は自己対局（512 局）とは別のエンジンで `n_games`（64）局を過去の lx と打ち、自分の手だけを方策の学習に使う（価値は結果から。lx の手は学習しない）。lx は起動時（プールが空のとき）と凍結相手を作り直すたびに、作り直す前の自分を `[exploiter] pool_out`（`~/libra-run/lx/pool/lx-<step>.pt`、新しい `pool_keep` 30 個）に保存する。ls は `[league] pool` の新しい `recent`（30。保存している全部）体から、本体が勝てていない相手ほど多く選び（PFSP、(1 − 勝率)²）、`switch_games`（256）局ごとに選び直す。成績は `status.json` の `league`（`pool` に相手ごとの本体の勝敗と勝率）と `log.txt` の `league: opponent lx step ...`。プールが空なら `pool_check_minutes`（10）ごとに見に行く。止めるときは ls の `[league] enabled = false` にして停止・起動する。

**過去の自分から始め直す**（`restart-exploiter`、2026-09-17）: lx を止めた状態で `bin/libra --run lx restart-exploiter --from ~/libra-run/lx/pool/lx-<step>.pt` を実行すると何をするかが出て、`--apply` を付けると実行する。latest.pt の重みをそのスナップショットに置き換え（step の数えと乱数は引き継ぎ、最適化の内部状態は捨てる）、今の重みをプールに `lx-<今の step>.pt` で残し、リプレイのチャンクを `replay/pre-restart-<step>/` に移して窓を空から数え直し、対本体の成績を履歴に移して布石を空にする。置き換える前の latest.pt・state.json・openings.json は `backup-<時刻>/` に写す（何も消さない）。起動後はリプレイが `min_window_games`（500 局）たまるまで学習しない。戻すときは backup の 3 つを元の場所に写し、`replay/pre-restart-<step>/` のチャンクを `replay/` に戻す。

状態は `bin/libra --run lx status`（`exploiter` に勝率、`main_step`、`refreshed_at`）。管理コンソールの「対本体 勝率」行にも出る。

## 自己対局ワーカー（既定は無効。GPU を足すときの配管）

学習側（`libra run`）と自己対局だけのプロセス（`libra worker`）を分けられる（decisions.md 2026-09-14）。**同じ GPU で分けても局/日は増えない**（推論だけで GPU が埋まっているため）。別の GPU（vast.ai など、利用開始はユーザーの判断）の計算を同じ run に足すためのもの。本番の ls・lx は使っていない。

1. 学習側の `config.toml` に `[workers]` `enabled = true` を書き、停止 → 起動。学習側は今までどおり自分でも自己対局し、加えて `weights/latest.pt` を配り、`inbox/` の局を 10 秒ごとに取り込む（取り込んだ局も新規局数に数えるので、学習量の規則 `replay_ratio` は変わらない）。搾取者の run では無効。
2. ワーカーを起動: `bin/libra --run ls worker --id w1 [--n-games 512] [--threads 12]`。重みを読み、`chunk_games`（100）局ごとに `inbox/` にファイルを置き、学習側が新しい重みを配ると 10 秒以内に読み直す。同じマシンでは学習側が動いている間だけ打ち、停止（STOP）か学習側の終了で残りを書いて抜ける（学習側が止まっている間は待つ）。ワーカーは run.lock を取らないので、コンソールの状態と 起動 / 停止 は学習側だけを見る。別マシンで run ディレクトリの写し（`config.toml` と `weights/`）を使うときは `--detached`。
3. 学習側は、局を打った重みが `max_lag_steps`（2000）より古いファイルを捨て、型・形・値域が合わないファイルを `inbox/rejected/` に移す（手の合法性や方策・価値の改ざんは確かめない）。件数は `status` の `workers:` 行と status.json の `workers`。
4. **vast.ai の GPU を足す**（decisions.md 2026-09-14）: 管理コンソールの **クラウド** タブで GPU・上限 $/h・時間・信頼度の下限・CPU GHz の下限・コア数の下限・転送料の上限を選んで「起動」（確認に費用の見積もりと残高が出る）。準備に 5〜15 分。借りたホストでワーカーが打ち、手元のブリッジ（`libra_cloud.bridge`）が重みと布石を送り、局を取ってきて手を再生して検査してから `inbox/` に置く。
   - 表示（15 秒ごと。残高とインスタンスは 5 分ごと、「残高を更新」で今すぐ）: 状態（準備 → インスタンス作成 → セットアップ → 稼働 → 停止処理 → 終了）、借りた時間と残り、費用の見積もり、回収した局数と弾いた数、ls の取り込み、残高、借りているインスタンス、launcher.log の末尾。
   - 「停止」: ワーカーを止めて残りの局を取ってからインスタンスを消す（数分）。時間が来たときも同じ。「候補を見る」: 検索したオファーを全件、値段の安い順に、落ちた条件（例「コア 6 < 16」「転送料 $0.026/GB > 上限 $0.020」）と見込み（局/日、100 万局あたりの費用）を付けて出す（借りない）。○1 から順に借りる（下の「借りる順」）。条件に合うものが無いときは「1 つ緩めれば借りられる」に、どの欄をいくつにすればどのホストが通るかが出て、そのボタンで欄の値を変えて検索し直せる。「この条件で起動…」で起動の確認へ進む（起動と同じ条件で判定する）。検索の時点で 1 GPU・verified・下り 200 Mbps 以上・イメージの CUDA 以上・ディスク 30 GB 以上に絞っているので、vast.ai のサイトより件数は少ない。借りられずに終わったセッションは launcher.log の `rejected by:` の行に落ちた条件の件数が残る。
   - **借り方**は既定で「入札」（割り込みあり。同じホストで on-demand より 16〜32% 安い。decisions.md 2026-09-15）。入札額は最低入札の 10% 増しで、上限 $/h・候補の順・費用は実効単価（dph_total − 最低入札 ＋ 入札）。高い入札が来るとホストが止められる。launcher が 60 秒ごとにインスタンスを見て、2 回続けて止まっていたら（またはブリッジが ssh の失敗で抜けて止まっていたら）状態が「打ち切り」になり、インスタンスを消して、残りの時間（ブリッジが動いた時間を引く）で次のセッションを自動で起動する（「終了（打ち切り・借り直し）」、履歴には別の行）。残りが 15 分未満・最初の開始から時間 + 30 分を過ぎた・停止を押したときは借り直さない。on-demand で落ちたときも同じ。on-demand で借りるときは「借り方」を切り替える。
   - **借りる順**（2026-09-18。それまでは実効単価の安い順だった）: 過去に借りたホストの実測から見込みの局/日を当て、**見込みの 100 万局あたりの費用が安い順**に借りる。同じ GPU でも CPU によって局/日が 1.2 倍違い、速い GPU ほど 100 万局あたりでは損をすることがあるため（measurements.md 2026-09-16。5070 Ti で 566k / 576k / 675k 局/日、5080 は局/日 1.3 倍でも $/h が 1.6 倍で負ける）。
     - 実測は `~/libra-run/cloud/<セッション>/bridge/bridge.log` の回収の傾き（最初の 10 分を除いた定常状態。15 分・500 局に満たないセッションは使わない）。`libra_cloud/hosts.py`。
     - 当てる順は **同じ機械（`machine_id`）→ 同じ GPU と CPU → 同じ GPU**（どれも実測の中央値。1 回の外れに引きずられない）。`offers` の表の印は ◎ / ○ / △。**`machine_id` を記録し始めたのは 2026-09-18 なので、それ以前のセッションは GPU と CPU でしか当たらない。**
     - 実測の無いホストには実測のあるホストの中央値を当てる（前に出しすぎて外れを引くことも、後ろに回して良いホストを試さないこともない）。実測がまったく無ければこれまで通り値段の安い順。
     - 別の置き場所で 2 台目を動かしているとき（`--root ~/libra-run/cloud2`）、実測はその置き場所のセッションだけを見る。
   - **状態が「異常終了」か、セッションが動いていないのに「借りているインスタンス」が残っている（ステータスバーが赤）ときは課金が続いている。**「後始末」で libra- のラベルのインスタンスをすべて消す。
   - 同じ操作をコマンドで: `bin/libra-vast start [--gpu RTX_5070_Ti --max-dph 0.28 --hours 3]`・`stop`・`status [--account]`・`offers`・`cleanup --yes`。セッションの記録は `~/libra-run/cloud/<run>-<時刻>/`（launcher.log、setup.log、bridge/bridge.log、bridge/rejected/、result.json）。
   - 前提: ls の `config.toml` に `[workers] enabled = true`（無いと起動を断る）、vast.ai の API キーと SSH 鍵（libra-cloud/README.md）。
   - **ls を止めている間**: ホストのワーカーは `--detached` なので最後に配られた重みで打ち続け、ブリッジも `inbox/` があるので取り込み続ける（課金も続く）。ls の step が進まないので、再開後に取り込む局は `max_lag_steps` の判定でほとんど捨てられず、同じ重みの局がまとめて学習に回る。数分〜十数分の停止はそのままでよく、数時間止めるときは先にクラウドを「停止」する。「全部 停止」はクラウドを止めない。**クラウドを動かしたまま WSL の shutdown や PC の再起動をしない**（インスタンスを消すのは手元のランチャーで、ホスト側に自分で止まる仕組みが無いため、課金が続く。そうなったら「後始末」）。
   - ブリッジは重みを前の送信から 300 秒経つまで送らない（`libra_cloud.bridge --min-push-seconds`。転送料と回線の詰まりを減らすため。布石は変わったらすぐ送る）ので、bridge.log の `age` は最大で約 300 秒＋送信時間になる。
   - ls のログに `workers: dropped ... stale games` が多いときは、bridge.log の `bridge: pushed weights/latest.pt ... in X s (age Y s)` で重みがホストに届くまでの時間を見る（bridge.json の `push_s` に最大値）。
