# LibraShogi

天秤将棋（https://fusekishogi.com/rules/）の AI「Libra」。設計は docs/libra-design.md、実行計画は docs/libra-local.md。
**新しいセッションではまず本ファイル → docs/runbook.md → docs/decisions.md の末尾 → docs/measurements.md の末尾を読み、`bin/libra status` と `bin/libra --run lx status` で稼働状態を確かめてから作業する。**
ユーザー（ルール設計者、git user は kotenbu）とのやり取りは日本語。

## 不変の制約
- ゼロから開発。既存将棋AI（やねうら王、Apery、dlshogi、Lc0、水匠、fuseki-shogi-ai など）のコード・評価関数・重み・棋譜を内部に使わない。GPL コードの流用禁止。手法・論文の参照は自由。KataGo（MIT）を参考にしたら NOTICE に記載。
- ライセンス: コード Apache-2.0 / 文書 CC BY 4.0 / 自己対局データ CC0 / 重み Apache-2.0。依存追加時は LICENSES/README.md に 1 行と全文を追加。
- 水匠5・fuseki-shogi-ai・desktop の wasm は対戦相手・黒箱テストの相手専用（別プロセス、USI）。評価値や読み筋を学習信号にしない。相手の分析・相手専用の対策・相手を固定した学習はしない。
- 秘密情報（vast.ai の API キー、SSH 鍵）を絶対にコミットしない。外部からコピーしたファイルをリポジトリに入れない（third_party/ は gitignore、取得はスクリプトでハッシュ固定）。
- v0.1 まで非公開、v0.1 で公開。
- 終局規定は docs/rules.md が唯一の正（大会規定: 千日手 4 回＝引き分け、連続王手は王手側負け、入玉宣言法 27 点、本将棋 320 手で引き分け、41 手目を 1 手目として数える）。マッチの裁定はハーネス（libra-sim）が行う。

## 構成
libra-sim（C++ シミュレータ、pybind11）/ libra-net（モデル、ONNX 書き出し）/ libra-search（MCGS、df-pn、配置詰み）/ libra-engine（USI 拡張エンジン libra / libra.exe、ONNX Runtime）/ libra-league（自己対局・学習ランナー、搾取者、評価、計測ハーネス、Python 版エンジン）/ libra-scale（玉配置表）/ libra-cloud（vast.ai、未着手）/ docs / data（マニフェストのみ）/ bin（起動スクリプト）/ tools / cmake

ビルド・起動・テスト・エンジンの駆動手順は `.claude/skills/run-librashogi/SKILL.md`（`/run-librashogi`）が正。ここに書いてある通りに動かす。

## 作業の進め方
1. **1 タスク 1 ブランチ**。`git checkout -b <task>` → 実装 → テスト全通過 → `git checkout main && git merge --ff-only <task> && git push origin main && git branch -d <task>`。PR は作らない（単独開発）。CI（.github/workflows/ci.yml、ubuntu-latest）が緑であることを push 後に `gh run list --limit 2` で確かめる。
2. **テストを先に書く**。C++ は `ctest --test-dir build/libra-sim` と `build/libra-search`、Python は SKILL.md の pytest 行。変更のたびに全部通す。パイプの `| tail` は終了コードを隠すので、コミット条件に使うときは pytest の結果行を目で確認する。
3. **コミットは小さく**、メッセージは「何を・なぜ」を日本語で。`git commit -s`（DCO の Signed-off-by）を付け、末尾に `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`。
4. **決定は docs/decisions.md に 1 行**（日付、内容、理由。ユーザーの決定はその旨を書く）。**実測値は docs/measurements.md に 1 行**（日付、条件、値、備考。予定と実測の差も書く）。文書は docs/runbook.md（運用）、docs/protocol.md（GUI との接続仕様）、docs/rules.md（ルール）。libra-design.md と libra-local.md はユーザーの計画書で、書き換えない（差異は protocol.md §4 と decisions.md に記録する）。
5. **ユーザーに聞く事項**（勝手に決めない）: 対局棋譜（水匠5 との）の公開可否、vast.ai の利用開始、公開版サイトの規定を大会規定に揃えるか、desktop への終局判定追加の Issue を出す時期（Libra 側の準備ができたら起票する決定済み）、基準値マッチの持ち時間、大きな設計変更（ネットの形、学習則、ルール解釈）。それ以外は自分で決めて decisions.md に記録する。
6. **報告は簡潔に**。結果を先に、数値は表、変更点は箇条書き。手順の説明を長々と書かない。中間報告は長い待ちの前に 1 回。
7. 秘密情報・外部コードの混入を疑う操作（clone、コピー）をする前に CONTRIBUTING.md のクリーンルーム方針を確認する。相手 AI（~/fuseki-shogi-ai）の内部は読まない（起動方法とオプションだけ）。
8. **論文の手法を実装するときは、論文の方法の節（アルゴリズム・擬似コード・付録）を読んでから作り、実装との差異を docs/decisions.md に 1 行で残す**（何をそのまま使い、何を変え、なぜ変えたか）。計画書が論文を「理由」として引いているだけのときも、「やり方」は原典で確かめる。別の目的で作った部品を流用するときは、その部品の前提が新しい用途でも成り立つかを確かめて同じ行に書く。原典を読めなかったとき（認証・PDF が読めない）はその旨を書く。2026-09-14 に、搾取者が評価ハーネスの「根の手番のネットで木を丸ごと評価する」形を流用して、Wang+ 2023 が誤りとした「相手の手を自分の方策で予測する」探索のまま 3 日動いていたため。

## 稼働中のランの扱い
**ランの操作と監視はユーザーが管理コンソール（`libra-console.bat`）で行う。Claude は行わない（2026-09-12 のユーザーの決定。Claude の停止待ちの誤りで搾取者が 11 分止まったため）。**
- **ランを止めない・起動しない**。`bin/libra stop` / `run` / `eval-now` / `match-now` を実行しない。設定ファイルの編集や実装は行い、**反映に要る停止と起動はユーザーに依頼する**（「コンソールの停止を押して、止まったら起動を押してください」と伝える）。操作は起動と停止だけ（一時停止・再開は 2026-09-14 に廃止。ユーザーの決定）。
- **監視しない**。バックグラウンドタスク・待機ループ・ポーリングで稼働状態やジョブの終了を待たない。状態が要るときは `bin/libra [--run lx] status` を 1 回読むだけにして、待たずに作業を終える。結果は次にユーザーから聞かれたときに読む。
- 本体 L-S は `~/libra-run/ls`、搾取者 lx は `~/libra-run/lx` で常時稼働。2026-09-13 から二飛香（docs/rules.md §3.2）の系列（ls は旧 ls の step 229,590 の重みから、lx はゼロから）。旧ルールの系列は `~/libra-run/ls-v0`・`lx-v0` に残し再開しない（docs/runbook.md 冒頭）。ネットの形（[net]）は変えない（変えるなら新しい run-id）。
- ls は `[auto]` で 24 時間ごとに archive → 基準比の自己評価 100 局 → 外部計測 10 局を別プロセスで回す（docs/runbook.md §6）。結果は管理コンソールの Elo / 対外対局タブと `~/libra-run/ls/eval`・`matches`。
- GPU を使う計測（速度比較など）はユーザーにコンソールから ls・lx を停止してもらってから行い、終わったら起動を依頼する。長い GPU 作業を共有のまま回したときは局/日が落ちる旨を measurements.md に書く。
- 1 週間の局/日（9/18 ごろ）、2 週間ごとの 20 局計測、10 月中旬の基準値マッチ 20 局、12 月の 100 局は docs/libra-local.md §5 の予定に従う。
- Windows 側の操作（デスクトップの bat、タスク スケジューラ「LibraShogi run」「LibraShogi run lx」）は docs/runbook.md §3。自動ログオンと GPU 電力上限は設定しない（ユーザーの決定）。

## 環境
- WSL2 Ubuntu-24.04（ディストリ名は `Ubuntu-24.04`）、RAM 32 GB、RTX 5070 Ti、Ryzen 9 9950X3D。データは WSL 内の ext4。
- **sudo が使えない**（apt 不可）。cmake / ninja / pybind11 / pytest / torch / onnxruntime は `.venv` の pip、node は `~/.nvm`。apt が要るものはユーザーに依頼する（mingw-w64 は導入済み）。
- Python パッケージは pip install しない。`PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-scale`（bin/ のスクリプトと driver は自分で通す）。
- Windows 版 libra.exe は WSL の mingw クロスビルド（公開後は GitHub Actions windows-latest）。置き場所は `C:\Users\sakis\libra\engine\`。
- 相手側の資産の所在とハッシュは docs/protocol.md §5。desktop の clone は `~/tenbin-shogi-desktop`（wasm は GPL、黒箱テストの相手としてだけ実行）。
- 一時ファイルはセッションの scratchpad に置き、リポジトリや `~/libra-run` を汚さない。

## 現在の目標
2026 年内に Libra-L（自己対局のみで学習）を USI 拡張エンジンとして desktop に組み込み、v0.1 として公開する。「fuseki-shogi-ai 方策ネット＋水匠5」との 100 局は計測であり、勝てればよい。相手の分析や相手専用の対策はしない（docs/libra-local.md §4）。
残りの主な作業: 世代が進んだら scale.json と搾取者の main.pt を作り直す / 複数葉の同時評価と fp16 でエンジンを速くする / desktop 側への Issue（終局判定、玉配置表）/ libra-cloud（vast.ai はユーザーの判断待ち）/ docs/match_report.md / v0.1 の公開準備（LICENSES、モデルカード、Releases の zip）。
