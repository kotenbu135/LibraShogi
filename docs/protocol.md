# Libra と tenbin-shogi-desktop の接続仕様（確定版）

- 根拠: tenbin-shogi-desktop `12c4a8d`（0.5.0、2026-09-10）の `docs/usi-fuseki-extension.md`（布石 USI 拡張 仕様 v0）、`src/usi/parse.ts`、`src/usi/engine.ts`、`src/usi/evalscale.ts`、`src/ui/play.ts`、`src/ui/engines.ts`、`src/ui/analysis.ts`、`src/state/game.ts`、`crates/usi-host/src/lib.rs` を読んで確定した（2026-09-11）。
- fuseki-shogi-ai は `scripts/fuseki_usi_server.py` の冒頭（起動方法・オプション既定値・`Normal_Engine` への中継の設定）だけを読んだ。相手 AI の内部は読んでいない。
- 本書は docs/libra-local.md §6.2 の表を置き換える。ライセンス: CC BY 4.0。

## 1. GUI がエンジンを扱う流れ（確認済みの事実）

| 項目 | GUI の実装 | Libra が守ること |
|---|---|---|
| 起動 | `path` を `args`（空白区切り）で起動。作業フォルダは省略時に実行ファイルのフォルダ。Windows では `CREATE_NO_WINDOW`。stdin/stdout の行往復、stderr はログに流す（`usi-host`） | 標準入出力は行バッファ。stderr に進捗を書いてよい（起動中は画面に出る） |
| 登録の下見 | `usi` を送り 15 秒以内の `usiok` を待つ。`isready` は送らない（`UsiEngine.probe`） | `usi` への応答でモデルを読まない。申告（`id name`、`option`）だけを即座に返す |
| 種別の自動判定 | `option name` が `Fuseki_` で始まる項目が 1 つでもあれば `kind='fuseki'`（布石にも対応）。無ければ `id name` の既知パターン、それも無ければ `normal`（`engines.ts probeInto`） | `Fuseki_Mode` を申告する（下表）。利用者が手で「布石にも対応」に変えなくても済む |
| GPU 判定 | `DNN_` で始まる項目があれば `gpu=true`。効果: (1) `isready` を既定 1200 秒待つ（通常 120 秒）、(2) 同時に 1 本だけ起動、(3) エンジン同士の対局では先後で同じ 1 本のプロセスを使い回す | `DNN_Model` を申告するので GPU 扱い。1 プロセスで両方の手番の `go` を受けられること（状態を手番に依存させない） |
| 起動手順 | `usi` → `usiok` → 上書き分の `setoption`（既定と同じ値は送らない） → `isready` → `readyok` → `usinewgame`（`ensureStarted` → `newGame`）。`stop` を先に送ることがある | モデルの読み込みは `isready` で行う。`usinewgame` は毎局来るとは限らない（先後で共用するときは 1 回） |
| 局面 | 布石: `position fuseki` ／ `position fuseki moves K*5i K*5a P*7g ...`。`choose:` は送らない（`game.ts positionCommand`）。本将棋: `position sfen <40 手完了時の SFEN> moves 7g7f ...`（`startpos` は来ない） | `choose:` が来ても読み飛ばす。SFEN は持ち駒なし・手番 b・手数 41（通算）で来る（wasm の `fw_to_sfen` と一致させた） |
| `go` の語（対局） | 時計あり: `go btime B wtime W byoyomi Y`。時計なし: `go movetime T`（秒/手 × 1000）。**布石中も同じ**。`go nodes` は対局では来ない（`play.ts goArgs`） | `btime/wtime/byoyomi/binc/winc/movetime/nodes/infinite` をすべて受ける |
| `go` の語（検討） | `go infinite` → `stop`。`bestmove` を 10 秒以内に返さないと以後のその探索の `bestmove` は捨てられる | `stop` から 10 秒以内に必ず `bestmove` を返す |
| 毎手送られる `setoption` | `MultiPV`（申告していれば毎 `go` 前）、`Fuseki_Mode`（申告していれば布石の毎 `go` 前に `tenbin` か `fuseki`） | 走っていないときに来る前提。受けたら次の `go` から反映 |
| `info` の読み方 | 語の位置に依存しない。知らない語は 1 語だけ読み飛ばす。`score cp` / `score mate` / `winrate` / `string` のどれかがある行だけ画面へ渡す。`pv` は次の既知キーまで | 独自語を足すなら値が既知キー名と衝突しないこと（例: `prior 0.12` は可） |
| 勝率の優先順位 | `score mate` ＞ `winrate`（cp は行にあればそのまま、無ければ登録した目盛りで勝率から作る）＞ `score cp` を登録した目盛りで換算（`analysis.ts evalOfInfo`）。すべて手番側の値として読み、先手視点に直す | 常に `winrate`（手番側、0..1）を出す。`score cp` も出す（下記の式）。値は手番側 |
| 目盛り（`Eval_Coef`） | 登録時に `option name Eval_Coef` の既定値があれば `scale=その値, offset=0`。無ければ `id name` の既知パターン（水匠5: 652/+51、やねうら王: 600/0、dlshogi 系: 756/0、`Tenbin Fuseki Engine`: 435/+34）。どれでもなければ 600/0。利用者が設定画面で変えられる（`evalscale.ts`） | `Eval_Coef` は申告しない（offset 0 の式なので 435/+34 と両立しない）。Libra は `winrate` を常に出すので目盛りは表示に効かない |
| `bestmove` の検証 | 布石: wasm の合法打ち一覧と照合。本将棋: shogiops の `isLegal`。指せない手なら**同じ局面でもう 1 回 `go`**、2 回目も駄目なら対局を**一時停止**（負けにはしない）（`play.ts` 236〜270） | 非合法手を返さない。ハーネスでは非合法手は即負け（大会規定第 27 条 1 項 3 号） |
| `bestmove resign` | 投了として扱う | 使ってよい |
| `bestmove win`（本将棋） | **検証せず、その手番のエンジンの投了として扱う**（`play.ts` 379〜382: 宣言した旨を表示して `resign` を返す） | GUI 経由では `win` を送ると負けになる。ハーネスは条件（docs/rules.md）を検証して受ける。→ §4 の未対応事項 |
| `bestmove win`（40 手完了時） | GUI は 40 手目を適用した時点で自分で 41 手目の裁定（`verifyFinalSfen`）を行い、裁定に当たれば先手の勝ちで終局する。**その局面で `go` は来ない** | ハーネスは仕様 v0 どおり `go` を送り `bestmove win` を期待する。Libra は裁定に当たる局面では探索せず `bestmove win` |
| 41 手目に先手の合法手なし | shogiops の `isEnd()`（詰み・ステイルメイト）で終局。GUI が判定 | libra-sim も同じ判定（先手の負け） |
| 千日手・手数上限・宣言法 | GUI には**無い** | ハーネスが docs/rules.md で裁定する。GUI での対局結果は比較に使わない |
| 終了 | `quit` を送り 3 秒待って kill | `quit` で速やかに終了 |

## 2. 天秤将棋の 1〜2 手目と「選ぶ」

GUI（`play.ts think`）は天秤将棋の `kings`（1〜2 手目）と `choose` を、席のエンジンが外部エンジンであっても**常に内蔵の両玉の価値表**（`king_pairs_iter1177_games.json`、`BuiltinEvaluator.choose` / `placerPick`）で処理する。外部エンジンに `position fuseki` ＋ `go`（1 手目）、`position fuseki moves K*5i` ＋ `go`（2 手目）、2 手目後の `winrate`（選択）を問い合わせる口は無い。fuseki_usi_server.py 側は `Fuseki_Mode=tenbin` で 1〜2 手目を玉打ちとして応答できる実装になっているが、GUI はそれを使っていない。

したがって:

- **GUI 上の天秤将棋では、Libra の `scale.json` は使われない**（両玉の配置と選択は GUI 同梱の表で決まり、Libra は 3 手目から打つ）。
- **計測（100 局）はハーネスで行う。** ハーネスは両エンジンに `position fuseki` → `go` → `bestmove K*xx`、`position fuseki moves K*xx` → `go` → `bestmove K*yy` で両玉を置かせ、選ぶ側には `position fuseki moves K*xx K*yy` → `go` を送って `winrate`（手番＝先手の勝率）の符号で選ばせる（0.5 以上なら先手）。
- GUI 側で「置く・選ぶをエンジンに任せる」を可能にする改修は tenbin-shogi-desktop の別作業（§4）。

## 3. Libra の申告と出力

```
id name LibraShogi <version>
id author kotenbu
option name Fuseki_Mode type combo default tenbin var tenbin var fuseki
option name MultiPV type spin default 1 min 1 max 300
option name Threads type spin default 4 min 1 max 64
option name DNN_Model type string default libra.onnx
option name DNN_Batch_Size type spin default 64 min 1 max 1024
option name Sims_Fuseki type spin default 200 min 1 max 1000000
option name Sims_Normal type spin default 400 min 1 max 1000000
option name Scale_Table type string default scale.json
option name USI_Ponder type check default false
usiok
```

- `Fuseki_Mode=tenbin`: 手数 0・1 の合法手は玉打ちだけ。`fuseki`: 布石将棋（玉もいつでも打てる）。
- `info` 行: `info depth D seldepth S multipv K score cp X winrate W nodes N nps P time T pv M ...`、末尾に `info string phase fuseki|normal ply N method mcgs|mcts|proof|scale`。
- `score cp` は `winrate` から作る: `cp = round(435 · ln(p/(1−p)) + 34)`（`p` は `[1e-6, 1−1e-6]` に丸める。`-0` は `0`）。GUI の `cpToWinrate(cp, 435, 34)` の逆関数。この換算はテストで固定する。
- 布石の `pv` は候補の 1 手だけでよい（仕様 v0）。Libra は MCGS の主変化を続けて出してよい（GUI は次の既知キーまで読む）。
- 40 手完了時に手番（先手）が後手玉を取れる局面での `go` → `bestmove win`。本将棋で入玉宣言の条件（docs/rules.md）を満たすときの `go` → `bestmove win`（GUI 経由では負けになる。§4）。

## 4. libra-local.md の想定と食い違う点（ルール設計者への報告）

1. **手数上限。** 大会の最新規定（世界コンピュータ将棋選手権 大会ルール、第 36 回用 `rule.pdf`、SHA-256 `9387db36…bed`、第 27 条 3 項）は **320 手**。「256 手」は現行規定ではない（320 手への変更は 2019 年）。libra-sim は `max_ply` を設定値にし、確認が取れるまで指示どおり 256 を既定にする。
2. **本将棋の `bestmove win`。** GUI は検証せず、宣言したエンジンの投了として扱う。Libra が GUI 経由で正当な宣言をすると負けになる。desktop 側の宣言法・千日手・手数上限の判定追加（確認事項）まで、GUI 対局では宣言を出さないオプション（`Declare_Win=false`）を持たせるか、GUI を直すかの判断が要る。
3. **置く・選ぶ。** GUI は外部エンジンに両玉の配置と選択を任せない（§2）。libra-local §6.2「選択そのものはハーネス（または GUI）が行う」のうち GUI は Libra の `winrate` を使わず自分の表で選ぶ。
4. **非合法手の扱い。** GUI は 1 回聞き直し、2 回目で一時停止（負けにしない）。ハーネスは大会規定どおり負けにする。
5. **`go` の語。** GUI は対局では `movetime` か `btime/wtime/byoyomi` だけを送る（布石中も）。`go nodes` は検討・対局とも来ない。想定どおり全語を受ければよい。
6. **`Eval_Coef`。** DNN 系エンジンの目盛りは申告の `Eval_Coef`（offset 0）→ 既知名 → 600/0 の順。Libra は `winrate` を常に出すので影響しない（想定どおり）。ただし Libra の `score cp`（435/+34）と GUI に登録される既定の目盛り（600/0）は一致しないので、cp を GUI の目盛りで読み直す場面（winrate の無い行）を作らない。
7. **リポジトリ。** `kotenbu135/LibraShogi` のリモートは空だった（LICENSE のコミットが無い）。apache.org の Apache-2.0 原文を `LICENSE` として置いた。
8. **環境。** WSL に `cmake`、`ninja`、`node`、`mingw-w64` が無く、`sudo` にパスワードが要るため apt を使えない。cmake/ninja/pybind11 は pip、node は nvm で入れる。`mingw-w64` は apt が要る（`sudo apt install mingw-w64` を依頼）。Windows 側の `node.exe` は `/mnt/c/Program Files/nodejs/node.exe` にある。

## 5. 相手側（計測用）の起動情報

fuseki-shogi-ai `d2ce104`（2026-09-11）、`scripts/fuseki_usi_server.py`:

- 起動: `~/fuseki-shogi-ai/.venv/bin/python scripts/fuseki_usi_server.py`、作業フォルダ `~/fuseki-shogi-ai`。
- `id name Tenbin Fuseki Engine 0.1`。既定: `Fuseki_Method=value`、`Fuseki_Mode=tenbin`、`Fuseki_Candidates=16`、`Fuseki_Device=cuda`、`MultiPV=5`、`Threads=8`、`USI_Hash=512`。
- `Normal_Engine` 既定 `vendor/YaneuraOu/source/YaneuraOu-by-gcc`（Linux ビルド、SHA-256 `a081b77a…8d6b`、上流 commit `33ccf1f9`）、`Normal_EvalDir` 既定 `vendor/yaneuraou_eval`（`nn.bin` SHA-256 `768068f0…d76`。水匠5 の配布物と同一かは計測前に Suisho5.7z の `nn.bin` と照合する）。41 手目からは `EvalDir`、`Threads`、`USI_Hash`、`MultiPV` を中継先へ送る。
- 方策 `fuseki_degct_b3_iter1177.onnx`（SHA-256 `5bf176c6…26a0`）、両玉の表 `king_pairs_iter1177_games.json`（`e9275eca…9618`）、価値ネット `value_mid_v3_10x128_t40.pt`（`ec9882f7…2cf76`）。これらは相手の資産で、Libra は読まない。
- 40 手を打ち終えた局面で裁定に当たれば `bestmove win`。`go infinite` は `stop` まで待つ。`go nodes N` は 41 手目以降そのまま中継。
