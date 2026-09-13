---
name: run-librashogi
description: Build, run, test, and drive LibraShogi (天秤将棋 AI「Libra」) — the C++ USI engine `libra`, the self-play/training runner `bin/libra run`, and the libra-sim Python bindings. Use when asked to start Libra, run the engine, run self-play, run the tests, or check a running run's status.
---

LibraShogi は 天秤将棋の AI。動かす対象は 3 つ: (1) C++ の USI エンジン `build/libra-engine/libra`（標準入出力で USI を話す）、
(2) 自己対局＋学習ランナー `bin/libra run`（状態は `~/libra-run/<run-id>/`）、(3) Python バインディング `librashogi` / `librasearch`。
エージェントは `.claude/skills/run-librashogi/driver.py` で 3 つとも駆動できる。パスはすべてリポジトリ root からの相対。

## Prerequisites

Ubuntu 24.04、g++ 13、Python 3.12、GPU は任意（RTX 5070 Ti で確認。無ければ CPU で動く）。`sudo` が使えない環境で作ったので、
apt は使わず venv の pip で揃える。ONNX Runtime（MIT）は公式バイナリを `third_party/`（gitignore 済み）に取得する。

```bash
python3 -m venv .venv
.venv/bin/pip install -q cmake ninja pybind11 pytest numpy onnx onnxruntime
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128   # CPU だけなら --index-url https://download.pytorch.org/whl/cpu
tools/fetch_onnxruntime.sh linux-gpu    # → third_party/onnxruntime/onnxruntime-linux-x64-gpu_cuda12-1.30.0（CPU だけなら linux-cpu）
```

## Build

```bash
export PATH=$PWD/.venv/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python -c "import pybind11;print(pybind11.get_cmake_dir())") -DLIBRA_ORT_DIR=$PWD/third_party/onnxruntime/onnxruntime-linux-x64-gpu_cuda12-1.30.0
cmake --build build
```

生成物: `build/libra-engine/libra`（隣に `libonnxruntime.so.1` などを写す）、`libra-sim/python/librashogi/_sim*.so`、`libra-search/python/librasearch/_search*.so`。
約 2 分。Windows 版 `libra.exe` は `libra-engine/README.md` の mingw クロスビルド（`sudo apt install mingw-w64` が要る）。

## Run (agent path)

`driver.py` はリポジトリ root から `.venv/bin/python` で実行する（sys.path を自分で通すので PYTHONPATH 不要）。終了コード 0 = OK。

```bash
.venv/bin/python .claude/skills/run-librashogi/driver.py sim
# fuseki legal moves: ply0 36 (36 king squares), ply2 245; startpos perft(3) = 25470 / sim: OK
.venv/bin/python .claude/skills/run-librashogi/driver.py engine
# 学習済みモデル不要（乱数の小ネットを ONNX にして CPU で動かす）。布石 2 局面＋本将棋 1 局面に go nodes 200、stop の確認。
# fuseki ply0  bestmove K*2i   legal=True (0.06s) ... engine: OK
.venv/bin/python .claude/skills/run-librashogi/driver.py engine --provider cuda --model ~/libra-run/ls/checkpoints/latest.onnx --go "nodes 400"
# 学習済みモデルで CUDA EP。1 局面 4〜10 秒（L-S と GPU 共有時）。標準エラーに [>] 送信 / [<] 受信の全行が出る
.venv/bin/python .claude/skills/run-librashogi/driver.py runner --device cpu --seconds 180
# 一時ディレクトリに小ネットで Runner を起こし、8 局＋学習 1 回で止める。latest.pt / latest.onnx が書かれる。CPU で約 1 分。runner: OK
```

エンジンに任意の USI を送るときは `--repl`（tmux 向け。`go` を送ると `bestmove` まで待つ）:

```bash
tmux new-session -d -s libra -x 160 -y 40
tmux send-keys -t libra 'cd ~/LibraShogi && .venv/bin/python .claude/skills/run-librashogi/driver.py engine --repl 2>&1 | grep -v "option name"' Enter
timeout 120 bash -c 'until tmux capture-pane -t libra -p | grep -q "readyok"; do sleep 0.5; done'
tmux send-keys -t libra 'position fuseki moves K*5i K*5a' Enter 'go nodes 100' Enter
timeout 60 bash -c 'until tmux capture-pane -t libra -p | grep -q "bestmove"; do sleep 0.5; done'
tmux capture-pane -t libra -p | tail -8
tmux send-keys -t libra 'quit' Enter; tmux kill-session -t libra
```

| driver コマンド | 何をするか |
|---|---|
| `sim` | librashogi の合法手・perft の最小確認 |
| `engine [--model --provider --go --repl]` | `build/libra-engine/libra` を子プロセスで駆動。bestmove の合法性を libra-sim で検証 |
| `runner [--seconds --device]` | 使い捨ての状態ディレクトリで自己対局＋学習を 1 周 |

エンジン単体（driver 無し）: `bin/libra-usi` が C++ 版を CUDA 用の `LD_LIBRARY_PATH` 付きで起動する（モデルは `~/libra-run/ls/checkpoints/latest.onnx`、無ければ Python 版 `bin/libra-usi-py` に落ちる）。

```bash
(printf 'usi\nisready\nposition fuseki moves K*5i K*5a\ngo nodes 100\n'; sleep 5; printf 'quit\n') | bin/libra-usi | grep -v "^option"
```

`go` の直後に `quit` を流すと探索が止められて 1 回の評価だけで指す（`nodes 0`）ので、パイプで流すときは `sleep` を挟む。

## Run (human path) — 本番のラン

**本番のラン（ls・lx）の操作はユーザーが管理コンソールで行う。Claude は stop / run / pause / resume / eval-now / match-now を実行せず、終了待ちの監視もしない**（CLAUDE.md 「稼働中のランの扱い」）。以下は手順の記録と、driver が使う一時的な run（`driver.py runner`）向け。

本番の状態は `~/libra-run/ls`（本体 L-S）と `~/libra-run/lx`（搾取者）。冪等で、前回の状態から再開する。詳細は `docs/runbook.md`。

```bash
bin/libra status                 # ls の状態（--run lx で搾取者）
bin/libra pause / resume / stop  # フラグファイル。stop はチェックポイントを書いて終了
bin/libra export                 # latest.pt → latest.onnx（チェックポイントごとに自動でも書かれる）
bin/libra-scale show             # 玉配置表 ~/libra-run/ls/scale/scale.json
```

起動は `nohup setsid bin/libra run >> ~/libra-run/ls/stdout.log 2>&1 &`（Windows ではタスク スケジューラ「LibraShogi run」/「LibraShogi run lx」がログオン時に起動）。

### Windows の管理コンソール（tools/windows/libra-console.ps1）

WinForms の GUI。人はデスクトップの `libra-console.bat` で開く。エージェントは WSL から `powershell.exe` で無人実行できる（`tools/windows/install.sh` で `C:\Users\sakis\libra\` に写してから。UNC パスからは実行しない）:

```bash
tools/windows/install.sh
S='C:\Users\sakis\libra\libra-console.ps1'
powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -File "$S" -Screenshot 'C:\Users\sakis\libra\console-shot.png'   # 1 回更新して PNG 保存、要約を表示して終了（約 5 秒）
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$S" -Do 'lx:pause'      # ボタンと同じ呼び出しだけ実行（pause/resume/stop/eval-now/match-now）
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$S" -Do 'lx:resume'
powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -File "$S" -Screenshot 'C:\Users\sakis\libra\console-shot.png' -Tab 'Elo'   # 下のグラフのタブを選ぶ（局/日, Elo, 対外対局, 学習, 終局内訳, 手数）
```

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$S" -UpdateDesktopModel   # desktop に登録した libra.exe の libra.onnx を latest.onnx に置き換える（desktop は起動しない）

要約行には `metrics=`（metrics.jsonl の点数）`evals=` `matches=` `archives=` も出る。自動計測は `bin/libra eval-now` / `match-now`（次のチェックポイントで実行、`~/libra-run/ls/auto.log`）。

PNG は `/mnt/c/Users/sakis/libra/console-shot.png` を Read で見る。`-Do` の出力は `bin/libra` の出力そのもの（`PAUSE set` など）。

## Test

```bash
export PATH=$PWD/.venv/bin:$PATH
ctest --test-dir build/libra-sim --output-on-failure      # ルールテスト 285 件＋perft
ctest --test-dir build/libra-search --output-on-failure   # df-pn
PYTHONPATH=libra-sim/python:libra-search/python:libra-net:libra-league:libra-scale python -m pytest -q libra-sim/tests/test_python.py libra-league/tests libra-net/tests libra-engine/tests libra-scale/tests libra-search/tests/test_external.py -p no:warnings
```

pytest は 58 件、約 10 秒〜3 分（ランナーのスモークと df-pn の乱数検証が重い。L-S 稼働中は CPU を取り合って伸びる）。CI（`.github/workflows/ci.yml`）は同じ手順を ubuntu-latest の CPU で回す。

## Gotchas

- **.ps1 は UTF-8 BOM 付きで保存する。** PowerShell 5.1 は BOM が無いと ANSI（cp932）として読み、日本語のラベルが化ける。bat は ASCII のみ（cmd は UTF-8 の日本語コメントで壊れる）。
- **PowerShell の stderr は cp932。** WSL 側で読むときは `2>&1 | iconv -f cp932 -t utf-8`。`wsl.exe` 経由の Python の出力は UTF-8 なので `StandardOutputEncoding = UTF8` を指定して読む。

- **PYTHONPATH が要る。** パッケージは pip install しない（`libra-sim/python`, `libra-search/python`, `libra-net`, `libra-league`, `libra-scale`）。`bin/libra*` と driver は自分で通す。pytest を直接呼ぶときは上の環境変数を付ける。
- **`_search` を単独で import すると libra-sim のテーブルが未初期化**になり得る（モジュール初期化で `static Position` を作って回避済み）。`librashogi` を先に import する必要はない。
- **エンジンの `go` の直後に `quit`/`stop` を送っても落ちない**が、探索前に止めると 1 回だけ評価して合法手の先頭を指す（`info string no search result`）。
- **CUDA EP は cuDNN 9 / CUDA 12 の .so が要る。** `bin/libra-usi` と driver は PyTorch の pip 配布物（`.venv/lib/python3.12/site-packages/nvidia/*/lib`）を `LD_LIBRARY_PATH` に足す。素の `build/libra-engine/libra` を CUDA で使うなら同じことをする。無ければ CPU に自動フォールバックし `info string ... provider cpu` と出る。
- **エンジンは 1 葉ずつ評価する（バッチ 1）**ので GPU でも 4 ms/評価程度。L-S が同じ GPU で動いていると 10〜20 ms。速度比較は `bin/libra pause` してから。
- **本番のランと GPU を共有する。** driver の `engine --provider cuda` や `runner --device cuda` は L-S・lx と同居できるが局/日を落とす。CPU で済む確認は `--device cpu` / `--provider cpu`（既定）。
- **手数上限は 320 手・41 手目起点**（`SearchConfig.max_ply`）。256 ではない。
- **玉配置の剪定**: 後手玉が四段目のペアは 3 手目の桂打ちで先手の裁定勝ち。搾取者 lx は `search.prune_gote_rank4 = true`、本体 L-S は 36×36 一様のまま。

## Troubleshooting

- **`Floating point exception (core dumped)` が `go` 直後に出る**: ルート評価待ち中の `finish_now` で候補 0 本のまま逐次半減を初期化していた（修正済み、`libra-search/tests/test_external.py`）。再発したら `SelfPlay::finish_now` の `root_ready` 分岐を見る。
- **USI の応答行が重複・欠落する（遅いマシン、CI）**: `std::cin` が `std::cout` に tie されていて読み取りスレッドの `getline` が flush を競合させていた。`main.cpp` の `std::cin.tie(nullptr)` を消さないこと。
- **`cmake: ninja not found`**: `.venv/bin` を `PATH` の先頭に置く（ninja も cmake も pip 版）。
- **`LIBRA_ORT_DIR に ONNX Runtime の展開先を指定してください`**: `tools/fetch_onnxruntime.sh linux-gpu`（または `linux-cpu`）を先に実行し、`-DLIBRA_ORT_DIR` にその展開先を渡す。
- **`ONNX Runtime library not found`**: `build/libra-engine/` に `libonnxruntime.so.1` が写っていない。`cmake --build build` をやり直す（POST_BUILD で写す）。
- **driver の `wait()` が毎回同じ `bestmove` を返す**: 受信行の走査位置を保持していなかった（修正済み、`self.cursor`）。
- **Windows の bat が動かない**: bat は ASCII のみにする（UTF-8 の日本語コメントは cmd の Shift-JIS 解釈で改行を壊す）。WSL から試すときは `cmd.exe /c "C:\Users\sakis\libra\libra-status.bat"` のように絶対パスで。
