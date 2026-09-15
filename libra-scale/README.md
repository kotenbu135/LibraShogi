# libra-scale — 玉配置表（天秤）

天秤層（1〜2 手目の両玉の配置と、選ぶ側の先後選択）の解法（docs/libra-design.md §5）。出力 `scale.json` は CC0。

## 手順

1. **剪定**（`pairs.py`）: 後手玉が四段目のペアは 3 手目の桂打ちで先手の裁定勝ちなので除く。36 × 27 = 972 通り、1↔9 筋の鏡映で 492 通り。
2. **探索値**（`table.py`）: 各ペアの 3 手目局面を外部駆動の MCGS で読み、先手の勝率 V̂ を得る。自己対局棋譜（一様サンプル）のペア別実測も添える。
3. **検証対局**（`verify.py`）: |V̂ − 0.5| の小さい上位ペアに限定した自己対局で実測勝率と信頼区間を出す。
4. **運用**: `balanced`（釣り合い集合）から置く側は一様に選ぶ（`Scale_Table` オプション）。選ぶ側は 2 手目後の局面の `winrate` が 0.5 以上なら先手。

```bash
bin/libra-scale build --sims 1600                 # → ~/libra-run/ls/scale/scale.json
bin/libra-scale verify --top 48 --games 100       # balanced を信頼区間で決め直す
bin/libra-scale show
```

## 全組の逐次の検証（seq）

剪定後の 492 組すべてを検証対局で測り、組ごとに打ち切る（docs/decisions.md 2026-09-15。規則は `libra_scale/seqrule.py`、実行は `seqrun.py`）。

```bash
D=~/libra-run/ls/scale/seq-v0.1
bin/libra-scale seq run --dir $D --table ~/libra-run/ls/scale/scale-v0.1.json --notify windows   # 初回（V̂ とモデルは --table から）。再開は --dir だけ
bin/libra-scale seq run --dir $D --eps-place 0.01                                # 置く側の 2 段目を足して再開
bin/libra-scale seq status --dir $D                                              # 進み具合・打ち切りの内訳・アラート・対称な組
bin/libra-scale seq table --dir $D --out ~/libra-run/ls/scale/scale-v0.1.1.json  # 表（version 1）
touch $D/STOP                                                                    # 止める（次の run が消す）
```

- **選ぶ側**: 組ごとに 100 局打ってから 50 局ごとに見て、95% で有意（|w − 0.5| > 1.96·se）か、半幅 1.96·se ≤ 0.02 で打ち切る。
- **対称な組を先に打つ**: 線対称・点対称の 27 組（鏡映込み 51 組）を最初にまとめて打ち、有意では止めずに半幅 0.02 まで打つ。
- **アラート**: 対称な組が後手に傾いたら、log.txt の `ALERT` 行と `ALERT.txt` に出す。`--notify windows` を付けると Windows の通知領域にも出す（WSL の powershell.exe）。途中の判定（早期）は「先手の得点 + 3.0·se < 0.5」、止まった後の判定（確定）は 95% 区間の上限 < 0.5。早期は取り込みのたびに見るので誤報がある（真の得点 0.5 の組でも約 2.5%）。注意として受け取り、判断は確定で行う。
- **GPU**: Windows 側のアプリと GPU を共有すると大きく遅くなる（2026-09-15 に専有の 0.30 倍。measurements.md）。回す間は GPU を使うアプリを閉じる。
- **置く側**: `--eps-place` を付けると、精度で止まり区間が 0.5 を含んだ組だけに打ち足す。`balanced` は、区間が 0.5 を含んだまま止まった組。
- **局の打ち方**: 自己対局と同じエンジン（同時 512 局）で、打ち切っていない組の玉 2 手を布石として渡す。棋譜は 1 局 1 行で `games/*.jsonl.gz` に残る（CC0）。
- **手元以外のワーカー**: 局を `inbox/` に置けば、手元の run が一緒に数える（ファイルの形は `seqrun.py` の冒頭）。
- **表（version 1）**: build の表に、`verify`（組ごとの局数・区間・打ち切りの理由）、`choose`（選ぶ側の手番）、`forced`（後手玉が四段目の組は先手）、`symmetric`（対称な組の要約）を足す。

## scale.json

```
version, license, generated_at, model{path, step}, sims, n_pairs_pruned, n_pairs_unique, pruned, balance_rule,
pairs: [{kb, kw, mirror, v_hat, selfplay{games, sente, draw, gote, winrate, ci95}, verify{...}?}],
balanced: [[kb, kw], ...]   # 鏡映も展開済み。エンジンはここだけ読む
```

V̂ と実測は「この世代の Libra 同士の勝率」であり理論値ではない。世代ごとに作り直す。
