# モデルカード: Libra v0.1

- 対象: `libra-v0.1.pt`（PyTorch のチェックポイント）と `libra-v0.1.onnx`（fp32、opset 17）
- 版: **v0.1**（本体 run `ls` の **step 477,636**、2026-09-15 12:25:25 に到達）
- ライセンス: **Apache-2.0**（`LICENSE`）。玉配置表 `scale-v0.1.json` は **CC0 1.0**（`LICENSES/CC0-1.0.txt`）
- 作者: kotenbu（LibraShogi contributors）。リポジトリ: https://github.com/kotenbu135/LibraShogi
- 本書のライセンス: CC BY 4.0

## 1. 何のモデルか

**天秤将棋**（https://tenbinshogi.com/rules/ 、本リポジトリの規定は [rules.md](rules.md)）を指す AI「Libra」の方策・価値ネット。
1 手目から 40 手目までの布石（駒を打って陣を作る段階）と、41 手目以降の本将棋の両方を 1 つのネットで扱う。
USI を拡張したエンジン（`libra` / `libra.exe`、[protocol.md](protocol.md)）から ONNX Runtime で読み、MCGS（Gumbel AlphaZero 系の探索）の評価に使う。

## 2. ネットの形

| 項目 | 値 |
|---|---|
| 種類 | Transformer エンコーダ（81 マスをトークンにする） |
| `d_model` / 層 / ヘッド / `d_ff` | 320 / 8 / 8 / 1280（dropout 0.0） |
| パラメータ数 | **10,233,090**（約 10.2M） |
| 入力 `sq` | `float32[batch, 81, 32]`（マスごとの特徴） |
| 入力 `glob` | `float32[batch, 32]`（持ち駒・手数・段階などの大域特徴） |
| 出力 `policy` | `float32[batch, 2268]`（指し手・打ち手のロジット） |
| 出力 `wdl` | `float32[batch, 3]`（手番側の 勝ち / 引き分け / 負け のロジット） |
| 出力 `v41` | `float32[batch, 3]`（41 手目局面の見立て。布石の価値の補助ヘッド） |
| ONNX | opset 17（既定ドメイン）、IR 8、バッチ可変、producer `pytorch 2.11.0` |
| ONNX のメタデータ | `libra_step=477636`、`libra_net={"d_model":320,...}`、`libra_source=libra-v0.1.pt`、`license=Apache-2.0` |

特徴量の作り方は `libra-sim`（`Position.features`）と `libra-net/libra_net/` が正。`.pt` には `model`・`opt`・`step`・`rng`・`config`・`state` が入っている（学習の再開用。推論には `model` だけでよい）。

## 3. どう学習したか

**自己対局だけ**で学習した（AlphaZero 型）。教師データ・定跡・人の棋譜は使っていない。

| 項目 | 値 |
|---|---|
| 学習の系列 | 2026-09-13 に二飛香（[rules.md](rules.md) §3.2）へ移行した系列。旧ルールの系列の step 229,590 の重みを種にして続けた |
| step 477,636 時点 | 世代 248、自己対局の総局数 **713,760 局** |
| 自己対局の探索 | 全読み 96 回（確率 0.25）／軽い読み 24 回、Gumbel の候補 16 / 8、`policy_topk` 32、`c_visit` 50.0、`c_scale` 1.0、fp16、同時 512 局・12 スレッド |
| 学習 | batch 1024、lr 2e-4（warmup 1,000）、weight decay 1e-4、`replay_ratio` 4.0、窓 100,000 局、256 局ごとに学習、`grad_clip` 1.0、`mirror_prob` 0.5 |
| 損失の重み | policy 1.0、value 1.0、v41 0.5 |
| 布石の価値目標 | `λ·z + (1−λ)·V̂41`（λ = 0.5）。引き分けの目標は `soft_wdl`。**採用の経緯と、この作り方が結果より約 0.02 低い目標になっていることは [decisions.md](decisions.md) 2026-09-15 と [measurements.md](measurements.md) 2026-09-15 18:40 に記録した（§6 の既知の限界）** |
| 終局規定 | [rules.md](rules.md)（千日手 4 回で引き分け、連続王手は王手側負け、入玉宣言 27 点法、本将棋 320 手、41 手目を 1 手目として数える） |
| 搾取者 | 別の run `lx` が本体の弱点を突く自己対局を回し、その布石を本体の自己対局に 10% 混ぜる（`openings_prob` 0.1） |

**クリーンルーム**: 既存の将棋 AI（やねうら王、Apery、dlshogi、Lc0、水匠、Stockfish、fuseki-shogi-ai、fuseki-shogi-web など）のコード・評価関数・重み・棋譜・評価値・読み筋を、内部にも学習信号にも使っていない（[CONTRIBUTING.md](../CONTRIBUTING.md)）。GPL のコードは一行も含まない。

## 4. 再現に使うコードの版

**`1fb8606`**（2026-09-15 12:39:09、「v0.1 の重みを固定し、サイトの AI をすべて Libra にする決定を記録する」）。

- step 477,636 を出した学習プロセスは 2026-09-15 09:02:47 の再起動で立ち上がっており、そのときの `main` は `44053a1` だった。
- `44053a1` と `1fb8606` の間で `libra-net` / `libra-engine` / `libra-sim` / `libra-search` に変更はない（`git diff --stat` が空）。
- [measurements.md](measurements.md) 2026-09-15 12:38〜14:25 の 40 局は `1fb8606` 直後のエンジンで測った。

## 5. 計測

### 5.1 自己評価（基準比 Elo、100 局、ハーネスが裁定）

基準は同じ系列の step 230,210。v0.1（477,636）**そのものは自動計測の刻みに当たっていない**。前後は:

| step | 基準比 Elo | 95% 区間 | 基準に対する得点 |
|---|---|---|---|
| 274,739 | +139.0 | +70.0〜+220.5 | 0.69 |
| 422,358 | +219.9 | +146.2〜+317.1 | 0.78 |
| **477,636（v0.1）** | **未計測** | — | — |
| 525,172 | +147.2 | +77.8〜+230.0 | 0.70 |

422,358 と 525,172 の区間は重なるので、この範囲で強さが上がったとも下がったとも言えない。

### 5.2 tenbin-shogi-web の今の AI との 40 局（2026-09-15 12:38〜14:25）

**Libra 0 勝 40 敗**（先手 0/12・後手 0/28、全局詰み、平均 75.3 手）。
相手は サイトの強さ 5（両玉の価値表＋布石の方策ネット温度 0.4＋Sunfish4 の WASM、1 手 10 秒）。
Libra は読み **96 回固定**（1 手の中央値 477 ms）で、**持ち時間を揃えていない**（時間で約 21 倍の差）。

**この 40 局は「サイトの AI を Libra に置き換えられるか」を見る計測で、Libra の強さの指標ではない**（[decisions.md](decisions.md) 2026-09-15）。
強さは、相手を fuseki-shogi-ai の方策ネット＋水匠5 とし、持ち時間を両者で揃えて別に測る。

### 5.3 玉配置表（`scale-v0.1.json`、CC0）

v0.1 のネットから作った、天秤将棋の 1〜2 手目に置く両玉の「釣り合う組」の表。
`libra-scale build`（読み 1,600 回、492 組）→ `verify`（検証対局）で作る。
2026-09-16 に全組の検証対局（`seq`）で作り直したものが今の `scale-v0.1.json`（SHA-256 `4f2d652cd57ec1cefd1ac6542dc73b337835a328afa9c212811d3f7107b34aee`）。
作り方と実測は [measurements.md](measurements.md) 2026-09-15 15:15〜17:36 と 2026-09-16。

## 6. 既知の限界

1. **布石の価値の見立てが後手寄り。** 布石の価値目標に混ぜている `V̂41`（41 手目局面の探索値）が実際の結果より約 0.04 低く、目標が結果より約 0.02 低い。その結果、両玉を置いた局面のネットの評価（平均 0.459）が自己対局の実績（0.48 台）より低く出る。原因（41 手目局面でのネットの低さと、自己対局の探索値が手番側を低く見ること）は突き止めていない（[measurements.md](measurements.md) 2026-09-15 18:40、19:10）。
2. **布石の引き分け確率が実態と合わない。** 布石のネットの D は 0.32〜0.37 だが、実際の引き分けは 0.7%。学習目標 `soft_wdl` の作り方をそのまま学んでいる。探索と表示は期待値だけを使うので推論には効かないが、学習への影響は未確認（同 20:00）。
3. **搾取者に大きく負ける。** 1.9M パラメータの搾取者が 10.2M の本体に 86〜90% 勝つ。読みを 4 倍（96→400 回）にしても縮まらないので、穴は評価（方策・価値）そのものにある（同 2026-09-14 17:00・17:46）。v0.1 も同じ系列の重みで、この穴は残っている。
4. **本将棋（41 手目以降）の読み抜けがある。** 5.2 の 40 局は、勝率 0.6〜0.8 と見ていた局面から詰まされる負け方が多かった。
5. **強さを揃えた条件で測っていない。** 5.1 は自分の過去の版との比較、5.2 は持ち時間が揃っていない。外部エンジンに対する絶対的な強さは未計測。
6. **ルールの版が固定。** 二飛香を含む今の [rules.md](rules.md) だけを扱う。`Fuseki_Rules` を名乗らないので、二飛香より前の棋譜を読ませると `info string bad position` と `bestmove resign` になる。
7. **入力は 天秤将棋 / 布石将棋 専用。** 本将棋（通常の将棋）の初期局面からは指せない。

## 7. 使い方

```
# Linux（ONNX Runtime、CUDA / CPU）
bin/libra-usi                     # DNN_Model に libra-v0.1.onnx を指す

# Windows（配布物の zip）
libra.exe                         # 隣の libra.onnx を読む。既定は DirectML
```

USI の申告と `info` の形式は [protocol.md](protocol.md) §3。天秤将棋の GUI（tenbin-shogi-desktop 0.10.0 以降）は
1 回登録すれば 1 手目から終局まで指させられる（同 §1・§2）。

## 8. 検証（SHA-256）

```
efeac0b7b1a55fe990503f394121643583ea3cf4e08faa3295f0c69d02598fd5  libra-v0.1.pt
61a7aefe05edc9f140ac0f154804700d3dabb156c2d38c6acc9c7d7aa4504a07  libra-v0.1.onnx
4f2d652cd57ec1cefd1ac6542dc73b337835a328afa9c212811d3f7107b34aee  scale-v0.1.json
9b032662b2a18bcfae02b212766139f2edeeb59e178f5a60bb9abb055c74a392  libra-v0.1-windows-x64.zip
28e87a95023e984629a905f6f2af1640fe6b9e234cf2a7f8b659a83219e402ae  libra-v0.1-selfplay-sample.jsonl.gz
```

Release の `SHA256SUMS` と同じ。zip と標本は `tools/package_release.sh` で作り直しても同じ値になる（§9）。

## 9. 配布物の作り方

```bash
# 1. Windows 版をクロスビルド（libra-engine/README.md）
cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-posix.cmake \
  -DLIBRA_BUILD_PYTHON=OFF -DLIBRA_BUILD_TESTS=OFF -DLIBRA_ORT_DIR=$PWD/third_party/onnxruntime/onnxruntime-win-x64-directml-1.24.4
cmake --build build-win
# 2. Release に載せるものを作る（zip・SHA256SUMS）
tools/package_release.sh v0.1 ~/libra-run/releases/v0.1 '2026-09-15 12:25:25'   # 末尾は自己対局の標本の中心の時刻
```

zip の中身は `libra.exe`、DirectML 版 ONNX Runtime 1.24.4 と DirectML 1.15.4 の DLL、`libra.onnx`（＝`libra-v0.1.onnx`）、
`scale.json`（＝`scale-v0.1.json`）、`README.txt`、`MODEL-CARD.md`、`LICENSES/`（Apache-2.0・CC BY 4.0・CC0 1.0・NOTICE・
依存物の全文と ThirdPartyNotices）。zip と標本の gzip は日時を固定して詰めるので、同じ入力からは同じ SHA-256 になる。**`Scale_Table` の既定は空なので、GUI に登録するときにこのオプションへ `scale.json` の場所を入れる**（[protocol.md](protocol.md) §3）。

エンジンが名乗る版は `id name LibraShogi 0.0.2`（`libra-engine/src/engine.cpp` の `VERSION`。エンジンの版で、重みの版とは別）。
