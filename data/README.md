# data

データセットのマニフェスト（ファイル名、SHA-256、サイズ、生成世代）と取得スクリプトだけを置く。実体は配布先（GitHub Release、将来は Hugging Face Hub）。

## 自己対局の棋譜（CC0 1.0、[LICENSES/CC0-1.0.txt](../LICENSES/CC0-1.0.txt)）

1 行 1 局の JSONL。

| 鍵 | 中身 |
|---|---|
| `tokens` | 1 手目からの指し手（布石の打ち手は `K*9f` のような USI の打ち、41 手目以降は `9f8g` のような指し手）を空白で区切ったもの |
| `result` | `sente` / `gote` / `draw`（先手勝ち・後手勝ち・引き分け） |
| `reason` | 終局の理由（`no_legal_move`・`ruling41`・`sennichite`・`perpetual`・`max_ply`・`declaration`） |
| `plies` | 総手数（1 手目から数える） |
| `sfen41` | 40 手完了時（41 手目の手番）の SFEN |
| `v41` | その局面の探索値（先手から見た −1〜+1） |

ルールの版は [docs/rules.md](../docs/rules.md)（二飛香を含む、2026-09-13 以降）。

### 公開済み

| 名前 | 版 | 局数 | 圧縮後 | SHA-256 | 置き場 |
|---|---|---|---|---|---|
| `libra-v0.1-selfplay-sample.jsonl.gz` | v0.1（step 477,636 の前後 1 時間、chunk `games_006755`〜`games_007317`） | 56,300 | 13.5 MB | `28e87a95023e984629a905f6f2af1640fe6b9e234cf2a7f8b659a83219e402ae` | [v0.1 Release](https://github.com/kotenbu135/LibraShogi/releases/tag/v0.1) |
| `libra-v0.2-selfplay-sample.jsonl.gz` | v0.2（step 317,975 の前後 1 時間、chunk `games_035115`〜`games_036840`） | 172,600 | 39.7 MB | `65dca6d740c674793cb816f375e2331691c3bc63685616b36ca24aa09cb3b762` | [v0.2 Release](https://github.com/kotenbu135/LibraShogi/releases/tag/v0.2) |

**この標本について**: どちらも本体 run `ls` の自己対局そのままで、選別していない。
両玉は 1〜2 手目に一様乱数で置いてあり、**4 局に 1 局ほどは布石の裁定だけで終わる**（41 手目を指さない）。
探索に Gumbel ノイズが入るので、指し手は最善手とは限らない。

版ごとの違い:

| | v0.1 | v0.2 |
|---|---|---|
| 搾取者 `lx` が作った布石から始まる局 | 約 10%（`openings_prob` 0.1） | **無し**（`openings` が空。lx を回していない） |
| リーグ（過去の世代どうし）の対局 | 混ざる | **無し**（`[league] enabled = false`） |
| ルールの版 | 二飛香を含む | 同じ |

v0.2 の系列は 2026-09-18 にゼロから学習し直したもので、v0.1 の重みの続きではない（[docs/model-card-v0.2.md](../docs/model-card-v0.2.md) §3）。

全期間の棋譜（2026-09-16 時点で 8,709 chunk・553 MB）は Hugging Face Hub に置く予定で、まだ出していない。

作り直しは `tools/package_release.sh v0.1 <重みの置き場> '2026-09-15 12:25:25'`
（v0.2 は `tools/package_release.sh v0.2 <重みの置き場> '2026-09-21 17:22:02'`）。gzip のヘッダの mtime を 0 に固定してあるので、同じ棋譜からは同じ SHA-256 になる。

## 外部エンジンとの対局棋譜

公開可否は未定（ルール設計者の判断。相手側の配布条件の確認が要る）。
