# はじめての LibraShogi（ビルド・テスト・対局・学習）

このリポジトリを初めて触る人向けの手順。目的別に読む節を選ぶ。ライセンス: CC BY 4.0。

| やりたいこと | 読む節 |
|---|---|
| Libra と指してみたい（ビルドしない） | §5（Windows の配布物を GUI に登録する） |
| ソースからビルドしてエンジンを動かしたい | §1〜§4 |
| 自己対局と学習を自分の機械で回したい（v0.2 の続きも） | §1〜§3 → §6 |
| 天秤将棋の AI を自分で作りたい（ルール・シミュレータ・ハーネスを使う） | §7 |
| Windows 版 `libra.exe` を作りたい | §8 |
| つまずいたとき | §9 |

エージェント（Claude Code）向けの駆動手順は [.claude/skills/run-librashogi/SKILL.md](../.claude/skills/run-librashogi/SKILL.md)。§1〜§3 のコマンドはそちらと同じで、変えるときは両方を直す。

## 1. 準備

確認済みの環境: Ubuntu 24.04（WSL2 を含む）、g++ 13、Python 3.12。GPU は任意（NVIDIA なら CUDA 12 で速くなる。無ければ CPU で動く）。
`sudo` が使えない環境でも組めるよう、ビルド道具（cmake、ninja、pybind11）も venv の pip で入れる。
ONNX Runtime（MIT）は公式バイナリを `third_party/`（gitignore 済み）に取得する。ハッシュは `tools/fetch_onnxruntime.sh` で固定している。

```bash
git clone https://github.com/kotenbu135/LibraShogi.git
cd LibraShogi
python3 -m venv .venv
.venv/bin/pip install cmake ninja pybind11 pytest numpy onnx onnxruntime
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128   # CPU だけなら https://download.pytorch.org/whl/cpu
tools/fetch_onnxruntime.sh linux-gpu    # CPU だけなら linux-cpu
export PATH=$PWD/.venv/bin:$PATH
```

Python のパッケージ（`librashogi`、`librasearch`、`libra_net`、`libra_league`、`libra_scale`、`libra_cloud`）は pip install しない。
`bin/` のスクリプトは自分で `PYTHONPATH` を通す。自分で `python` を呼ぶときは次を付ける。

```bash
export PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-scale:libra-cloud
```

## 2. ビルド

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python -c "import pybind11;print(pybind11.get_cmake_dir())") \
  -DLIBRA_ORT_DIR=$PWD/third_party/onnxruntime/onnxruntime-linux-x64-gpu_cuda12-1.30.0   # CPU だけなら onnxruntime-linux-x64-1.30.0
cmake --build build
```

約 2 分。生成物:

| もの | 場所 |
|---|---|
| USI エンジン `libra`（隣に `libonnxruntime.so.1` などを写す） | `build/libra-engine/libra` |
| シミュレータの Python モジュール `librashogi` | `libra-sim/python/librashogi/_sim*.so` |
| 探索の Python モジュール `librasearch` | `libra-search/python/librasearch/_search*.so` |
| C++ テスト（`test_rules`、`perft`、`test_dfpn`） | `build/libra-sim/`、`build/libra-search/` |

`-DLIBRA_ORT_DIR` を渡さないとエンジンは作られない（CMake は `libra-engine: skipped` と出して成功する）。シミュレータだけが要るなら
`cmake -S libra-sim -B build/libra-sim ...` で単体でも組める（[libra-sim/README.md](../libra-sim/README.md)）。

## 3. テスト

CI（`.github/workflows/ci.yml`）と同じ。結果行の `100% tests passed` と `N passed` で確かめる（`| tail` は終了コードを隠す）。

```bash
ctest --test-dir build/libra-sim --output-on-failure      # ルールテスト（docs/rules.md の各節）＋ perft
ctest --test-dir build/libra-search --output-on-failure   # df-pn
PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-scale:libra-cloud python -m pytest -q \
  libra-sim/tests/test_python.py libra-league/tests libra-net/tests libra-engine/tests libra-scale/tests \
  libra-search/tests/test_external.py libra-search/tests/test_selfplay_threads.py libra-cloud/tests
```

pytest は 10 秒〜3 分（ランナーの煙テストと df-pn の検証が重い）。学習済みモデルは要らない（テストは小さな乱数ネットを作って使う）。

## 4. エンジンを動かす

エンジン `libra` は USI を拡張したプロトコルで話す（[protocol.md](protocol.md)）。モデル（ONNX）が要る。

| モデルの入手 | 手順 |
|---|---|
| 公開版の重みを使う | [Releases](https://github.com/kotenbu135/LibraShogi/releases) の `libra-v0.2.onnx`（と玉配置表 `scale-v0.2.json`）を取得。中身は [model-card-v0.2.md](model-card-v0.2.md) |
| 自分で学習した重みを使う | `bin/libra export`（§6）が `~/libra-run/<run-id>/checkpoints/latest.onnx` に書く |
| モデル無しで動作確認だけ | `.venv/bin/python .claude/skills/run-librashogi/driver.py engine`（乱数の小ネットで 3 局面を指す） |

`bin/libra-usi` が C++ 版を CUDA 用の `LD_LIBRARY_PATH` 付きで起動する。モデルは環境変数 `LIBRA_MODEL`（既定 `~/libra-run/ls/checkpoints/latest.onnx`）。

```bash
export LIBRA_MODEL=~/Downloads/libra-v0.2.onnx
(printf 'usi\nisready\nposition fuseki moves K*5i K*5a\ngo nodes 100\n'; sleep 5; printf 'quit\n') | bin/libra-usi | grep -v "^option"
```

- `go` の直後に `quit` を流すと探索が止められて 1 回の評価だけで指す（`nodes 0`）ので、パイプで流すときは `sleep` を挟む。
- CUDA が使えないときは CPU に自動で落ち、`info string ... provider cpu` と出る。
- 天秤将棋の 1〜2 手目（両玉）に使う玉配置表は、既定で**実行ファイルの隣の `scale.json`**（v0.2 から。それより前は既定が空）。別の場所の表を使うときだけ `setoption name Scale_Table value <scale-v0.2.json の場所>` を送る。表が無くても探索で置く。
- 局面の書き方（`position fuseki moves ...`、`choose:`、41 手目以降の SFEN）は [rules.md](rules.md) §3.5 と [protocol.md](protocol.md) §1。

## 5. GUI（tenbin-shogi-desktop）で指す

天秤将棋の GUI [tenbin-shogi-desktop](https://github.com/kotenbu135/tenbin-shogi-desktop)（0.10.0 以降）は、Libra を 1 回登録すれば 1 手目（両玉の配置と先後の選択）から終局まで指させられる。

**Windows（ビルドしない）**: [Releases](https://github.com/kotenbu135/LibraShogi/releases) の `libra-v0.2-windows-x64.zip` を展開し、GUI の「エンジン」に `libra.exe` を登録する。
モデル（隣の `libra.onnx`）も玉配置表（隣の `scale.json`）も自動で読むので、**v0.2 では登録時に入れる設定は無い**（v0.1 では `Scale_Table` に表の場所を手で入れる必要があった）。
既定の推論は DirectML（DirectX 12 の GPU なら NVIDIA・AMD・Intel で動き、無ければ CPU）。NVIDIA の GPU で速くしたいときは CUDA 版の DLL に差し替える（[libra-engine/README.md](../libra-engine/README.md)）。

**WSL でビルドしたものを Windows の GUI から使う**: 実行ファイルに `C:\Windows\System32\wsl.exe`、引数に `-d <ディストロ> -- <repo>/bin/libra-usi` を登録する（ディストロが 1 つなら `-d <ディストロ>` は省ける）。
モデルは `LIBRA_MODEL` の既定（`~/libra-run/ls/checkpoints/latest.onnx`）なので、公開版の重みを使うなら `bin/libra-usi` を呼ぶ小さなラッパを作って `LIBRA_MODEL` を設定する。

GUI での対局は非合法手を負けにしない（1 回聞き直して 2 回目で一時停止）ので、強さの比較には使わない。比較は §7 のハーネスで行う（[protocol.md](protocol.md) §4）。

## 6. 自己対局と学習を回す

`bin/libra run` が自己対局と学習を 1 プロセスで時分割し、状態を `~/libra-run/<run-id>/` に置く。冪等で、止めても同じコマンドで前回の状態から再開する。

```bash
bin/libra run                    # 既定の run-id は ls。状態は ~/libra-run/ls
bin/libra status                 # 局/日、勝敗内訳、学習損失
bin/libra stop                   # STOP フラグを置き、チェックポイントを書いて終了
bin/libra export                 # checkpoints/latest.pt → latest.onnx（チェックポイントごとに自動でも書かれる）
```

**既定の設定は本体 L-S の本番用**（約 10M パラメータの Transformer、同時 512 局、12 スレッド、GPU メモリ約 9 GB。`libra-league/libra_league/config.py` の `DEFAULTS`）。
小さく試すには、`--run` で別の run-id を選び、初回だけ `--config` で上書きする。

```toml
# small.toml（例）
[net]
d_model = 64
n_layers = 2
n_heads = 4
d_ff = 256

[selfplay]
n_games = 32
threads = 4
```

```bash
bin/libra --run small run --config small.toml     # 2 回目からは ~/libra-run/small/config.toml が使われる
```

状態ディレクトリの中身（チェックポイント、棋譜 JSONL、フラグ）と長く回すときの運用（Windows での自動起動、管理コンソール、自動計測、搾取者、クラウドのワーカー）は [runbook.md](runbook.md)。
ネットの形（`[net]`）はチェックポイントと合わなくなるので、途中で変えない（変えるなら新しい run-id）。

**v0.2 の続きから学習する**: Releases の `libra-v0.2.pt` にはモデル・オプティマイザ・step・乱数状態・設定が入っている。
新しい run-id の `~/libra-run/<run-id>/checkpoints/latest.pt` に置いてから `bin/libra --run <run-id> run` すると、その step から続く（本体 ls もこの形で旧ルールの重みから始めた。[runbook.md](runbook.md) 冒頭）。`[net]` は v0.2 と同じにする。

まったく手軽に 1 周だけ試す（一時ディレクトリ、CPU で約 1 分）:

```bash
.venv/bin/python .claude/skills/run-librashogi/driver.py runner --device cpu --seconds 180
```

## 7. 天秤将棋の AI を自分で作る人へ

Libra とは別の AI を作るときに、このリポジトリから使えるもの。

| 使えるもの | 場所 | 使い方 |
|---|---|---|
| ルールの仕様（布石・41 手目の裁定・終局規定） | [rules.md](rules.md) | 唯一の正。§6 の perft 値で自分の合法手生成を照合できる |
| 厳密シミュレータ（合法手・終局判定・SFEN・鍵） | `libra-sim`（C++ / Python `librashogi`） | [libra-sim/README.md](../libra-sim/README.md) の Python API。Apache-2.0 |
| GUI との接続仕様（USI 拡張） | [protocol.md](protocol.md) | 布石の `position fuseki`、両玉の配置と先後の選択、`winrate`、`bestmove win` |
| 対局ハーネス（無人対局、大会規定での裁定、棋譜 JSONL） | `bin/libra match` | 相手は任意の USI エンジン。下の例 |
| 玉配置表（両玉の釣り合う組） | Releases の `scale-v0.2.json`（CC0） | 形は [libra-scale/README.md](../libra-scale/README.md) |
| 自己対局の棋譜（標本） | Releases の `libra-v0.2-selfplay-sample.jsonl.gz`（CC0） | 形は [data/README.md](../data/README.md) |

自作のエンジンを Libra と戦わせる（裁定は libra-sim、非合法手は即負け、規定は rules.md）:

```bash
bin/libra match --games 20 --go "movetime 3000" \
  --opponent "<自作エンジンの起動コマンド>" --opponent-cwd <作業フォルダ> \
  --model ~/Downloads/libra-v0.2.onnx --libra-opt Scale_Table=$HOME/Downloads/scale-v0.2.json
```

棋譜は `~/libra-run/ls/matches/<時刻>.jsonl`（`--out` で変更）と `.summary.json`。相手は布石の USI 拡張（`Fuseki_Mode` の申告、`position fuseki moves ...`、両玉の配置と先後の選択、40 手完了時の `bestmove win`）を話せる必要がある。最小の実装例は `libra-league/tests/random_usi.py`（合法手から無作為に指すテスト用エンジン、約 40 行）。

## 8. Windows 版 `libra.exe` を作る

WSL の mingw-w64（posix スレッド版。`sudo apt install mingw-w64` が要る）でクロスビルドする。依存は静的リンクし、ONNX Runtime と DirectML の DLL だけ同梱する。

```bash
tools/fetch_onnxruntime.sh win-dml
export PATH="$PWD/.venv/bin:$PATH"   # cmake と ninja は pip 版（sudo が使えないため）
cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-posix.cmake \
  -DLIBRA_BUILD_PYTHON=OFF -DLIBRA_BUILD_TESTS=OFF -DLIBRA_ORT_DIR=$PWD/third_party/onnxruntime/onnxruntime-win-x64-directml-1.24.4
cmake --build build-win                  # → build-win/libra-engine/libra.exe と DLL
```

配布物（zip、SHA256SUMS、自己対局の標本）は `tools/package_release.sh`（[model-card-v0.2.md](model-card-v0.2.md) §9）。CI の `windows-cross` ジョブが同じクロスビルドを毎回確かめる。

## 9. つまずきどころ

- **`cmake: ninja not found`**: `.venv/bin` を `PATH` の先頭に置く（ninja も cmake も pip 版）。
- **`LIBRA_ORT_DIR に ONNX Runtime の展開先を指定してください`**: `tools/fetch_onnxruntime.sh linux-gpu`（または `linux-cpu`）を先に実行し、その展開先を `-DLIBRA_ORT_DIR` に渡す。
- **`ONNX Runtime library not found`**: `build/libra-engine/` に `libonnxruntime.so.1` が写っていない。`cmake --build build` をやり直す（POST_BUILD で写す）。
- **`ModuleNotFoundError: librashogi`**: `PYTHONPATH`（§1）が無いか、ビルドしていない。
- **CUDA で動かない**: CUDA EP は cuDNN 9 / CUDA 12 の `.so` が要る。`bin/libra-usi` は PyTorch の pip 配布物（`.venv/lib/python3.12/site-packages/nvidia/*/lib`）を `LD_LIBRARY_PATH` に足す。素の `build/libra-engine/libra` を CUDA で使うなら同じことをする。無ければ CPU に落ちる。
- **エンジンが `info string bad position` と `bestmove resign` を返す**: 二飛香（rules.md §3.2）より前のルールの棋譜を読ませている。Libra は今の規定だけを扱う。
- **手数上限は 320 手・41 手目起点**（`max_ply`）。256 ではない。
- **学習の GPU メモリが足りない**: `[selfplay] n_games` と `[net]` を小さくする（§6）。
- **Windows の bat が動かない**: bat は ASCII のみ。生成元は `tools/windows/*.in`（[runbook.md](runbook.md) §3）。
