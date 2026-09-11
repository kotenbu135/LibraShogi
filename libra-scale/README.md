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

## scale.json

```
version, license, generated_at, model{path, step}, sims, n_pairs_pruned, n_pairs_unique, pruned, balance_rule,
pairs: [{kb, kw, mirror, v_hat, selfplay{games, sente, draw, gote, winrate, ci95}, verify{...}?}],
balanced: [[kb, kw], ...]   # 鏡映も展開済み。エンジンはここだけ読む
```

V̂ と実測は「この世代の Libra 同士の勝率」であり理論値ではない。世代ごとに作り直す。
