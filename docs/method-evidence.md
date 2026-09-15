# 手法の裏取り（2026-09-15）

採用済みの手法と設定値について、出所（計画書・決定の記録・計測）と文献での扱いを並べる。**変更の案ではない**（変えるかどうかはユーザーが決める。CLAUDE.md 作業の進め方 5・10）。

区分:
- **裏取りあり**: 原典（論文・開発元の文書）と一致を確かめた、計測で決めた、またはユーザーの決定。
- **一部**: 計画書や記録に出所はあるが、値や形の根拠が無い。または文献から変えていて理由の記録が無い。
- **記録なし**: 出所も理由も記録が無い（実装時に Claude が決めた値を含む）。
- **未実装**: 計画書にあるが実装が無く、やめた記録も無い。

---

## 1. 布石の価値目標（λ·z＋(1−λ)·V̂41）の文献での扱い

### 1.1 経緯

decisions.md 2026-09-15 の同名の行。claude.ai の設計案の 1 行「分散を下げる」が出所で、論文の引用は無い。λ 0.5 は 9/11 の実装時に理由の記録なく決めた。Gemini の回答の分離学習案（40 手目の価値を本将棋のネットから取る）を z で縛った形と読める（推測）。

### 1.2 文献

| 出典 | 価値の目標 | 探索値の使い方 | 確認 |
|---|---|---|---|
| AlphaZero [Silver18] | 結果 z（勝 +1・分 0・負 −1）、損失 (z−v)² | 使わない | 原典（arXiv 版） |
| KataGo 論文 [Wu19] | 結果 z。全読みの手だけを学習に使う（価値も） | 使わない。補助は陣地・得点（結果から作る） | 原典 |
| KataGoMethods [KGM] | 本体は結果 z | 将来の探索値の指数平均（約 6・16・50 手）を**別の補助ヘッド**で学ぶ。本体の損失の重みを少し下げて補助を足すと、学習が少し速く本体の損失も少し良い | 原典 |
| MuZero [Schrittwieser20] | 盤上ゲームは終局まで（z） | Atari だけ n 手先の探索値で補う（Reanalyze は n=5） | 原典 |
| Lc0 のブログ [Lc0-18] | — | 探索値だけでは値が自己強化してずれるおそれがあるので、z との平均を提案 | 原典 |
| Lc0 のブログ [Lc0-19] | — | q_ratio を入れた test52 は test51 とほぼ同じ強さ | 原典 |
| lczero-training [Lc0-code] | 既定は z（`q_ratio` 既定 0）。混ぜるときは `q·q_ratio + z·(1−q_ratio)` | 局面ごとに探索の q と引き分け確率 d を別々に持ち、WDL にするとき探索の d を使う | 原典（コード） |
| elmo・dlshogi [Yamaoka17][Yamaoka20][dlshogi-WCSC29] | 損失を (1−λ)H(v, z) ＋ λH(v, q) の**和**にする（勝率の 2 値の交差エントロピー） | 各局面自身の根の探索値。「経験的にブートストラップ手法は、非ブートストラップ手法より性能が良い」。2017 年の比較では価値の一致率がわずかに良い（0.4439 → 0.4444） | 原典（ブログ・アピール文書）。λ の値は未確認 |
| Oracle の Connect Four [Oracle] | 平均、または z から探索値へ世代で移すと、z・探索値の単独より学習が速い | 探索値の欠点は水平線効果と、学習初期に意味が無いこと | 原典 |
| Willemsen ほか [Willemsen22] | 探索値を使う目標（soft-Z、A0C、A0GB）が z より良い（Connect Four、Breakthrough） | — | **要旨のみ。本文は未確認**（PDF を読めなかった） |

### 1.3 論点ごとの所見

| 論点 | 文献 | Libra | 所見 |
|---|---|---|---|
| (a) 探索値を混ぜるか | 混ぜない: AlphaZero、KataGo（本体）、MuZero（盤上）、Lc0 の既定。混ぜる: elmo・dlshogi、Connect Four の例。Lc0 の試行は中立 | 布石で混ぜる | どちらにも実績があり、文献から一意には決まらない |
| (b) 混ぜ方 | dlshogi は損失の和。Lc0 は q と d を別々に混ぜる。KataGo は別ヘッド | 1 つのスカラー t に混ぜてから `soft_wdl` で (max(t,0), 1−\|t\|, max(−t,0)) に戻す | **同じ形の先例は見つからなかった**。引き分けの確率は結果にも探索にも無い 1−\|t\| になる（§2 の計測: 布石の局面でネットの D は 34〜37%、実際の引き分けは 0.7%）。探索と表示は W−L だけを使うので、推論では期待値を変えない（学習への影響は未確認） |
| (c) どの局面の探索値か | elmo・dlshogi・Lc0 は各局面自身の根の値。KataGo の補助は将来の探索値の平均。MuZero（Atari）は n 手先 | 布石の全局面に、41 手目の局面 1 か所の `root_q` を使う | **同じ形（固定の境界 1 点の値を手前の全局面に配る）の先例は見つからなかった**。計測済みの後手寄りのずれ（v41 が実際より約 0.04 低い。measurements.md 2026-09-15 18:40・19:10）は、この 1 点の値に集まる |
| (d) 係数の決め方 | Lc0 は既定 0。Connect Four は世代で移す。dlshogi は値が未確認 | λ 0.5 固定 | 根拠の記録なし |
| (e) 分散を下げる他の手段 | KataGo は補助ヘッドと本体の重みの調整 | V̂41 の補助ヘッドは実装済み（重み 0.5）。設計の「41 手目から本将棋を複数回分岐」は未実装 | — |

**まとめ**: 「混ぜるか・混ぜないか」は文献でも分かれていて、どちらかをバッドプラクティスとする資料は見つからなかった。一方、混ぜる先例（elmo・dlshogi・Lc0）に共通する次の 3 点では、Libra はいずれも先例と違う形になっている。
1. 探索値は各局面自身のもの。
2. 引き分けは実際の結果か探索の値から取り、スカラーから作らない。
3. 係数は既定 0 から、または計測で決める。

---

## 2. 洗い出し

「出所」の decisions は docs/decisions.md、measurements は docs/measurements.md。

### 2.1 学習の目標と損失

| 項目 | 今の値・形 | 出所 | 文献・計測との対応 | 区分 |
|---|---|---|---|---|
| 布石の価値目標 | 0.5·z ＋ 0.5·v41 | 設計 §4.1 の 1 行、λ は実装時 | §1 | 記録なし |
| 目標の 3 値への戻し方 `soft_wdl` | 引き分け = 1−\|t\| | 9/11 の実装 | 先例なし（§1.3 (b)）。v0.1 の D: 2・10・20・30・39 手目で 0.338・0.343・0.361・0.366・0.352、45・80 手目で 0.007。実際の引き分け 0.7%（measurements 同日） | 記録なし |
| v41 の値 | 41 手目の局面の `root_q`（根の訪問の平均） | 設計は「探索値」とだけ | Lc0 は根の平均（root_q）と最善手の値（best_q）を両方持つ。v41 は実際より約 0.04 低い（measurements 18:40・19:10） | 記録なし |
| 価値を学ぶ局面 | 全局面（速読みの局面も） | decisions 9/11（値だけ） | KataGo は全読みの手だけを学習に使う（価値も） | 記録なし |
| 方策の目標 | 改善方策の上位 32 手で切って正規化し直す（`policy_topk`） | 9/11 の実装 | Gumbel 論文・AlphaZero・KataGo は全手の分布 | 記録なし |
| 改善方策の未訪問の手の値 | 根の `root_q` で補う | 9/11 の実装 | Gumbel 論文は v_mix（ネットの値と訪問した手の値の混合） | 一部（論文から変えて理由なし） |
| 損失の重み | 方策 1.0・価値 1.0・V̂41 0.5 | 9/11 の実装 | AlphaZero は等重み、KataGo は価値 1.5。V̂41 の 0.5 は根拠なし | 一部 |
| V̂41 ヘッドの目標 | `soft_wdl(v41)` の交差エントロピー | 9/11 の実装 | 上の `soft_wdl` と同じ | 記録なし |

### 2.2 学習の設定

| 項目 | 今の値 | 出所 | 文献との対応 | 区分 |
|---|---|---|---|---|
| 学習率 | 2e-4、1,000 step の立ち上げの後は一定 | 9/11 の実装 | AlphaZero は 3 回、KataGo も途中で下げる | 記録なし |
| 最適化 | AdamW (0.9, 0.98)、重み減衰 1e-4、勾配の切り詰め 1.0 | 9/11 の実装 | AlphaZero・KataGo は SGD（慣性 0.9） | 記録なし |
| バッチ・学習量 | 1,024、replay_ratio 4、256 局ごと | decisions 9/11（値だけ） | — | 記録なし |
| リプレイの窓 | 100,000 局で固定（lx 20,000） | 9/11 の実装 | KataGo は総数に応じて広げる（25 万から約 2,200 万局面） | 記録なし |
| ワーカーの局の古さの上限 | 2,000 step | decisions 9/14（手元と同じ鮮度にする理由あり） | — | 一部 |
| 鏡映増強 | 50% | 設計 §3.1・§3.2 | — | 裏取りあり（計画書） |
| ネットの形 | d 320・8 層（約 10M） | 計画 §3.2 の 10M | 形の選び方の記録なし | 一部 |
| lx の学習設定 | バッチ 512、学習率 3e-4 | lx の config | — | 記録なし |

### 2.3 自己対局の探索

| 項目 | 今の値・形 | 出所 | 文献との対応 | 区分 |
|---|---|---|---|---|
| 読みの回数の抽選 | 全読み 96 回・速読み 24 回、全読み 25% | 計画 §3.1（平均 50〜100 回、全読み 25%） | KataGo は 600・100 回、25%（のち 1,000・200 回） | 一部 |
| Gumbel の候補数と σ | m 16（速読み 8）、c_visit 50、c_scale 1.0 | 9/11 の実装 | Gumbel 論文の Go・チェスの値と一致（m 16、c_visit 50、c_scale 1.0）。速読みの 8 は根拠なし | 裏取りあり（速読みの m を除く） |
| σ に掛ける値の範囲 | [−1, 1] のまま | 9/11 の実装 | Gumbel 論文は木の中で [0, 1] に最小・最大で正規化する（mctx も） | 一部（論文から変えて理由なし） |
| 根以外の手の選び方 | PUCT、cpuct 1.5 | 設計 §3.3「PUCT＋Gumbel ルート」 | Gumbel 論文は根以外も改善方策に合わせる決定的な選び方。1.5 は根拠なし | 一部 |
| 未訪問の手の値 | 親の平均で補う | 9/11 の実装 | — | 記録なし |
| 引き分けの値 | 0 | 9/11 の実装 | AlphaZero と同じ | 裏取りあり |
| 詰み・証明の上限 | 根の詰み 200 節点、布石の証明 1,000 節点・36 手目から | 計画 §3.1「ノード上限を小さく」。200 は 9/14 に効果を測った上で変えない決定 | — | 一部 |
| 自己対局の玉配置 | 36×36 一様 | decisions 9/11「当面」 | 設計の 3 種の混合のうち一様だけ | 一部 |
| 搾取者・リーグ・openings | opponent_prior、PFSP、recent 5、openings 10% など | docs/exploiter-literature.md、decisions 9/14 | 原典と差異を記録済み | 裏取りあり |

### 2.4 計画にあって未実装（やめた記録なし）

| 項目 | 計画書 |
|---|---|
| 41 手目の局面から本将棋を複数回分岐（同一布石の勝率推定を安定化） | 設計 §4.1 |
| 補助ヘッド (ii) k 手後の駒損得・生存、(iii) 相手の次手予測、(iv) 後手玉に当たる利きの本数・遮断の可否 | 設計 §3.2。計画 §3.1 は V̂41 と次手予測を「残す」 |
| 1↔9 筋の鏡映で正規化した局面キーによる学習データの重複除去（鍵 `Position::norm_key` は libra-sim にあるが、使っているのはテストだけで、リプレイも探索も重複除去に使っていない） | 設計 §3.1 |

「後で」「当面」と記録があるもの（ペアサンプラーの重点、検証対局の逐次棄却、League exploiter）と、計画で「捨てる」としたもの（Reanalyze）は除いた。

### 2.5 玉配置表（libra-scale）とエンジン

| 項目 | 今の値 | 出所 | 対応 | 区分 |
|---|---|---|---|---|
| build の読み | 1,600 回 | decisions 9/11（値だけ） | 設計 §5 は 10^7〜10^8 ノード | 一部 |
| verify の組数・局数 | 上位 48 組 × 100 局、同数 | decisions 9/11（v0） | 計画 §3.1 は 100 組 × 1,000〜2,000 局、設計は逐次棄却・トンプソン | 一部 |
| 検証前の釣り合いの幅 | margin 0.02 | decisions 9/11（値だけ） | — | 記録なし |
| 検証後の釣り合い | 区間が最小値と重なる組 | 設計 §5 | — | 裏取りあり |
| エンジンの読みの既定 | Sims_Fuseki 400、Sims_Normal 800、Mate_Nodes 2000 | 実装 | — | 記録なし |
| 持ち時間の配分 | (秒読み＋加算＋残り/30) × 0.9 | 実装 | — | 記録なし |
| 選ぶ側の判断 | 2 手目の後の winrate の符号 | 計画 §4 | ネットの後手寄りのずれ（measurements 18:40）がそのまま効く | 裏取りあり（計画書） |

### 2.6 文書と実装の食い違い

| 項目 | 内容 |
|---|---|
| `libra-net/libra_net/losses.py` | どこからも読まれていない。書いてある損失（価値は z の 1 点、V̂41 は期待値の二乗誤差）と、実際に学習で使う `trainer.py` の損失（`soft_wdl` の交差エントロピー）が違う |
| docs/protocol.md の Sims_Normal | 既定 400 と書いてあるが、エンジンの既定は 800 |

---

## 3. 読めなかった資料

- Willemsen ほか 2022 の本文（PDF を読めなかった）。要旨だけを使った。
- dlshogi の λ の値（ブログとアピール文書に数値が無い）。
- Gumbel 論文の σ の式は、読み取りで c_scale が指数に見えた。著者の実装 mctx は積（(c_visit ＋ max N) · c_scale · q）なので、積として扱った。

## 参考文献

- [Silver18] Silver et al., A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (arXiv:1712.01815). https://ar5iv.labs.arxiv.org/html/1712.01815
- [Wu19] Wu, Accelerating Self-Play Learning in Go (arXiv:1902.10565). https://ar5iv.labs.arxiv.org/html/1902.10565
- [KGM] KataGo Methods. https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
- [Schrittwieser20] Schrittwieser et al., Mastering Atari, Go, chess and shogi by planning with a learned model (arXiv:1911.08265). https://ar5iv.labs.arxiv.org/html/1911.08265
- [Danihelka22] Danihelka et al., Policy improvement by planning with Gumbel (ICLR 2022). https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf / mctx https://github.com/google-deepmind/mctx
- [Lc0-18] Understanding Training against Q as Knowledge Distillation. https://lczero.org/blog/2018/10/understanding-training-against-q-as/
- [Lc0-19] What's going on with training! https://lczero.org/blog/2019/06/whats-going-on-with-training/
- [Lc0-code] lczero-training tf/tfprocess.py・chunkparser.py. https://github.com/LeelaChessZero/lczero-training
- [Yamaoka17] 将棋でディープラーニングする その39（ブートストラップ）、その50（訂正）. https://tadaoyamaoka.hatenablog.com/entry/2017/06/28/080414 / https://tadaoyamaoka.hatenablog.com/entry/2017/12/05/210522
- [Yamaoka20] dlshogiの学習則. https://tadaoyamaoka.hatenablog.com/entry/2020/05/31/114435
- [dlshogi-WCSC29] 第29回世界コンピュータ将棋選手権 dlshogi アピール文章. https://www.apply.computer-shogi.org/wcsc29/appeal/dlshogi/dlshogi_appeal_wcsc29.pdf
- [Oracle] Lessons From AlphaZero (part 4): Improving the Training Target. https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628
- [Willemsen22] Willemsen, Baier, Kaisers, Value targets in off-policy AlphaZero: a new greedy backup. https://research.tue.nl/en/publications/value-targets-in-off-policy-alphazero-a-new-greedy-backup/
