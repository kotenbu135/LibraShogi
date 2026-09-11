# libra-search

- `include/libra/selfplay.h`, `src/selfplay.cpp`: 自己対局エンジン。同時進行する対局ごとに 1 葉を選び、特徴をバッチにまとめて外（PyTorch）に渡し、評価を受けて逆伝播する。ルートは Gumbel Top-k ＋ 逐次半減、それ以外は PUCT。布石中は Zobrist 合流の DAG。終端は libra-sim の裁定。
- `python/bindings.cpp`: `librasearch.SelfPlay(config, n_games, seed, threads)` — `collect(sq, glob)` / `apply(logits, wdl)` / `take_finished()` / `stats()` / `set_active(n)`。
- `tests/smoke_selfplay.py`: 一様方策で終局まで回し、記録を再生して整合を確かめる。

df-pn・配置詰み・41 手目の証明探索、USI 拡張エンジンはこれから足す。
