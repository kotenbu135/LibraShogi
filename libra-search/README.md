# libra-search

探索（C++17）と Python バインディング `librasearch`。自己対局（libra-league）、USI エンジン（libra-engine、`libra_league.usi_engine`）、
評価ハーネス、玉配置表（libra-scale）がこれを使う。

| ファイル | 内容 |
|---|---|
| `include/libra/selfplay.h`, `src/selfplay.cpp` | 自己対局エンジン。同時進行する対局ごとに葉を選び、特徴をバッチにまとめて外（PyTorch / ONNX Runtime）に渡し、評価を受けて逆伝播する。ルートは Gumbel Top-k ＋ 逐次半減、それ以外は PUCT。布石中は Zobrist 合流の DAG。終端は libra-sim の裁定。根で証明探索（布石は 41 手目の裁定、本将棋は df-pn の詰み）を行う。**外部駆動モード**（`set_position` / `collect_batch` / `apply_batch` / `finish_now` / `result`）は 1 局面を指定の読みの回数で読み、複数葉の同時評価（virtual loss）に対応する |
| `include/libra/dfpn.h`, `src/dfpn.cpp` | df-pn（Nagai 2002 の手法を参考にゼロから実装）。問題を差し替えて解く: `MateProblem`（本将棋の詰み）、`Ruling41Problem`（先手が 40 手完了時に後手玉に当たっている形を強制できるか）、`Mate41Problem`（後手が 41 手目に先手の合法手なしを強制できるか） |
| `python/bindings.cpp` | `librasearch.SelfPlay(config, n_games, seed, threads)` — 自己対局: `collect` / `apply` / `take_finished` / `stats` / `set_active` / `set_openings`、外部駆動: 上記。`librasearch.solve(pos, problem, max_nodes, tt_bits)` — df-pn を局面に直接（`problem` は `mate` / `ruling41` / `mate41`） |

## テスト

| ファイル | 内容 |
|---|---|
| `tests/test_dfpn.cpp` | df-pn（`ctest --test-dir build/libra-search`） |
| `tests/test_external.py` | 外部駆動モードの境界（直後の `finish_now`、budget 0、結果の形、複数葉の同時評価）。CI で回す |
| `tests/test_selfplay_threads.py` | 並列の割り当てを変えても棋譜が変わらないこと。CI で回す |
| `tests/test_dfpn_random.py` | 自己対局の局面で証明の健全性を深さ優先で確かめる（CI では回さない） |
| `tests/smoke_selfplay.py` | 一様方策・ゼロ価値で終局まで回し、CPU 側の速度を見る煙テスト（スクリプト） |
