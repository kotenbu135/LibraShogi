# libra-sim

天秤将棋の厳密シミュレータ（C++17、ビットボード、pybind11）。ルールの唯一の正は [docs/rules.md](../docs/rules.md)。
既存の将棋 AI のコードは一切参照していない（CONTRIBUTING.md のクリーンルーム方針）。

## ビルドとテスト

```bash
python3 -m venv .venv && .venv/bin/pip install cmake ninja pybind11 pytest
export PATH=$PWD/.venv/bin:$PATH
cmake -S libra-sim -B build/libra-sim -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python -c "import pybind11;print(pybind11.get_cmake_dir())") -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build build/libra-sim
./build/libra-sim/test_rules          # ルールテスト（docs/rules.md の各節）
./build/libra-sim/perft --check       # perft（既知の値と照合）
PYTHONPATH=libra-sim/python python -m pytest -q libra-sim/tests/test_python.py
```

合法手の黒箱テスト（desktop の wasm と突き合わせる。node と tenbin-shogi-desktop の clone が要る）:

```bash
TENBIN_DESKTOP_DIR=~/tenbin-shogi-desktop PYTHONPATH=libra-sim/python python libra-sim/tests/blackbox_wasm.py 400 0
```

## 構成

| ファイル | 内容 |
|---|---|
| `include/libra/types.h` | マス・駒・手の表現、USI 変換 |
| `include/libra/bitboard.h` | 81 マスのビットボード、利きの表 |
| `include/libra/position.h` | 局面。布石（玉打ち、二歩、筋埋め禁止、二飛香（天秤将棋のみ）、40 手目の制限、41 手目の裁定）と本将棋（打ち歩詰め、千日手、連続王手、宣言法、手数上限）。着手順に依存しない Zobrist と 1↔9 筋の鏡映キー |
| `python/bindings.cpp` | pybind11（`librashogi.Position`） |
| `python/librashogi/usi.py` | 勝率↔擬似 cp の換算（GUI と同じ式） |
| `tests/` | `test_rules.cpp`、`perft.cpp`、`test_python.py`、`blackbox_wasm.py`＋`wasm_oracle.mjs` |

## Python API（抜粋）

```python
import librashogi as ls
p = ls.Position("tenbin")                 # 空盤。1〜2 手目は玉打ち
p.set_position("position fuseki moves K*5i K*5a choose:sente P*7g")
p.legal_moves()                           # ['P*1c', ...]（USI）
p.do_move("P*3c"); p.undo()
p.sfen(); p.key; p.norm_key               # 拡張 SFEN、着手順非依存の鍵、鏡映で正規化した鍵
p.outcome()                               # ('ongoing'|'sente'|'gote'|'draw', reason)
p.can_declare("sente"); p.declare("sente"); p.resign("gote")
p.set_max_ply(320, count_from_41=True)    # 手数上限（大会ルール 320 手。41 手目を 1 手目として数える）
```
