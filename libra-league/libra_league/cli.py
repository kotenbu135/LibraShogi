# SPDX-License-Identifier: Apache-2.0
"""`libra` コマンド: run / stop / status（docs/libra-local.md §7.2）、自己対局だけを回す worker（libra_league/workers.py）。

同時進行局数を絞る `throttle`（decisions.md 2026-09-12）と一時停止の `pause` / `resume`（2026-09-14）は廃止した。
GPU や CPU を空けるときは stop し、終わったら run する。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .looptime import format_timing
from .state import DEFAULT_ROOT, StateDir, read_json


def default_match_model(sd: StateDir) -> Path:
    """match の既定モデル。bin/libra-usi は C++ 版（ONNX）なので latest.onnx を優先し、無ければ Python 版用の latest.pt。"""
    onnx = sd.checkpoints / "latest.onnx"
    return onnx if onnx.exists() else sd.checkpoints / "latest.pt"


def export_onnx_file(ckpt: Path, out: Path, log=None) -> float:
    """チェックポイント（.pt）を推論用 ONNX に書き出し、ORT と torch の出力の差を返す（原子的に置き換える）。"""
    import os

    from libra_net.export_onnx import check, export_checkpoint, load_checkpoint

    tmp = out.with_suffix(".onnx.tmp")
    meta = export_checkpoint(ckpt, tmp)
    model, _ = load_checkpoint(ckpt)
    diff = check(model, tmp)
    os.replace(tmp, out)
    (log or print)(f"exported {out} step {meta['libra_step']} max|ort-torch| {diff:.1e}")
    return diff


def resolve_match_model(sd: StateDir, model: str | None, ckpt: str | None, log=None) -> str:
    """match で使う重み。--model が最優先。--ckpt（.pt）を渡したときは隣の同じ名前の .onnx を使い、
    無ければそこへ書き出す（自動計測は節目の archive の重みで打つ。docs/restart-plan.md §7 P2）。"""
    if model:
        return model
    if ckpt:
        p = Path(ckpt)
        onnx = p.with_suffix(".onnx")
        if not onnx.exists():
            if not p.exists():  # 写しも .pt も無い（archive でないチェックポイントが回転で消えた）: 最新の重みで打つ
                (log or print)(f"warning: neither {p} nor {onnx} exists; falling back to the latest weights")
                return str(default_match_model(sd))
            export_onnx_file(p, onnx, log)
        return str(onnx)
    return str(default_match_model(sd))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="libra", description="Libra の自己対局・学習ランナー")
    ap.add_argument("--run", default="ls", help="run-id（状態ディレクトリ ~/libra-run/<run-id>）")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="状態ディレクトリの親")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="前回状態から再開（無ければ新規）。停止処理中なら終わるのを待ち、残った STOP は消してから起動する")
    p_run.add_argument("--resume", action="store_true", help="（既定と同じ。互換のため）")
    p_run.add_argument("--config", default=None, help="config.toml（初回だけ有効。以後は状態ディレクトリの写しを使う）")
    p_run.add_argument("--no-supervise", action="store_true", help="監視役を挟まずこのプロセスで回す（監視役が子を起動するときに使う）")
    sub.add_parser("stop", help="STOP フラグを置く。チェックポイントを書いて終了")
    p_st = sub.add_parser("status", help="状態を表示")
    p_st.add_argument("--json", action="store_true", help="機械可読な JSON を 1 行で出す（Windows の管理コンソール用）")
    p_st.add_argument("--tail", type=int, default=0, help="--json のとき log.txt の末尾 N 行も含める")
    p_st.add_argument("--history", type=int, default=0, help="--json のとき metrics.jsonl（最大 N 点）、eval と match の結果一覧も含める")
    sub.add_parser("eval-now", help="EVAL_NOW フラグ: 次のチェックポイントで archive に残し、直前の archive と自己評価する（[auto] が有効な run）")
    sub.add_parser("match-now", help="MATCH_NOW フラグ: 次のチェックポイントで外部エンジンとの計測対局を積む（[auto] が有効な run）")
    p_ev = sub.add_parser("eval", help="2 つのチェックポイントを対局させて Elo 差と較正を出す")
    p_ev.add_argument("--a", required=True, help="チェックポイント A（.pt）")
    p_ev.add_argument("--b", required=True, help="チェックポイント B（.pt）")
    p_ev.add_argument("--games", type=int, default=200)
    p_ev.add_argument("--sims", type=int, default=96)
    p_ev.add_argument("--concurrent", type=int, default=128)
    p_ev.add_argument("--threads", type=int, default=8)
    p_ev.add_argument("--seed", type=int, default=0)
    p_ev.add_argument("--no-noise", action="store_true", help="根の Gumbel ノイズを切る（手を乱数で選ばない。エンジンとしての強さに近い条件）")
    p_ev.add_argument("--b-set", action="append", default=[], help="B 側だけ別の探索設定で読む（<鍵>=<値>。例 gumbel_rescale=true と c_scale=0.1 で σ の形を比べる）。"
                      "変えられるのは full_sims・fast_sims・gumbel_m_full・gumbel_m_fast・c_visit・c_scale・gumbel_rescale・gumbel_noise・cpuct")
    p_ev.add_argument("--out", default=None, help="結果 JSON の出力先（既定: <run>/eval/<時刻>.json）")
    p_ev.add_argument("--publish", action="store_true", help="結果を progress ブランチの experiments/ へ push する（手元の PC の外から読めるようにする）")
    p_ev.add_argument("--branch", default="progress", help="--publish の書き出し先（既定 progress。main には入れない）")
    p_m = sub.add_parser("match", help="計測: Libra（USI）と外部エンジンを無人対局させ棋譜を JSONL に残す")
    p_m.add_argument("--games", type=int, default=20)
    p_m.add_argument("--go", default="movetime 3000", help="go の引数（例: 'movetime 3000' / 'btime 60000 wtime 60000 byoyomi 10000'）")
    p_m.add_argument("--opponent", default=None, help="相手エンジンの起動コマンド（既定: fuseki_usi_server.py）")
    p_m.add_argument("--opponent-cwd", default=str(Path.home() / "fuseki-shogi-ai"))
    p_m.add_argument("--opponent-opt", action="append", default=[], help="相手の setoption（name=value）")
    p_m.add_argument("--libra-opt", action="append", default=[], help="Libra の setoption（name=value）")
    p_m.add_argument("--model", default=None, help="Libra のモデル（既定: <run>/checkpoints/latest.onnx。無ければ latest.pt = Python 版エンジン用）")
    p_m.add_argument("--ckpt", default=None, help="Libra の重みをチェックポイント（.pt）で指定する。隣の同じ名前の .onnx で打ち、無ければ書き出す（--model が優先）")
    p_m.add_argument("--out", default=None, help="棋譜 JSONL（既定: <run>/matches/<時刻>.jsonl）")
    p_m.add_argument("--first-placer", default="a", choices=["a", "b"], help="第 1 局で両玉を置く側（a=Libra）")
    p_ex = sub.add_parser("export", help="チェックポイント（.pt）を推論用 ONNX に書き出す（libra / libra.exe 用）")
    p_ex.add_argument("--ckpt", default=None, help="既定: <run>/checkpoints/latest.pt")
    p_ex.add_argument("--out", default=None, help="既定: <run>/checkpoints/latest.onnx（同じ場所に一時ファイルを書いてから置き換える）")
    p_w = sub.add_parser("worker", help="自己対局だけを回し、終局した局を <run>/inbox に置く（学習は [workers] enabled の run 側）")
    p_w.add_argument("--id", required=True, help="ワーカー名（英数字と _、32 文字まで。対局ファイル名と seed に使う）")
    p_w.add_argument("--weights", default=None, help="重み（既定: <run>/weights/latest.pt）")
    p_w.add_argument("--inbox", default=None, help="対局ファイルの置き場所（既定: <run>/inbox）")
    p_w.add_argument("--n-games", type=int, default=None, help="同時進行局数（既定: [selfplay] n_games）")
    p_w.add_argument("--threads", type=int, default=None, help="既定: [selfplay] threads")
    p_w.add_argument("--device", default=None, help="既定: cuda があれば cuda")
    p_w.add_argument("--detached", action="store_true", help="学習側の run.lock と STOP を見ない（別マシンで同期した run ディレクトリを使うとき）")
    p_op = sub.add_parser("openings", help="搾取者の run から、搾取者が勝った布石を openings.json に書き出す")
    p_op.add_argument("--chunks", type=int, default=50, help="新しい側から何チャンク（100 局単位）見るか")
    p_op.add_argument("--moves", type=int, default=12, help="玉 2 手のあとに残す布石の手数")
    p_op.add_argument("--out", default=None, help="既定: <run>/openings.json")
    p_rs = sub.add_parser("restart-exploiter", help="止まっている搾取者の run を、プールの過去の自分（lx-<step>.pt）の重みから始め直す（何も消さない）")
    p_rs.add_argument("--from", dest="snapshot", required=True, help="プールのスナップショット（例: ~/libra-run/lx/pool/lx-000116230.pt）")
    p_rs.add_argument("--apply", action="store_true", help="付けないと何をするかだけ出す")
    p_ca = sub.add_parser("calib", help="同じネットの自己対局の較正（探索値と結果の信頼度曲線・ECE・Brier）を、最新の対局から出す")
    p_ca.add_argument("--games", type=int, default=20000, help="新しい側から何局使うか（書き出し済みのチャンクから）")
    p_ca.add_argument("--bins", type=int, default=10)
    p_ca.add_argument("--json", action="store_true")
    p_gp = sub.add_parser("genprof", help="一般化の物差し: 保存済みの重みを、リプレイの指定チャンクの局面で測る（価値の相関・二乗誤差、方策の交差エントロピー）")
    p_gp.add_argument("--ckpt", default=None, help="既定: <run>/checkpoints/latest.pt")
    p_gp.add_argument("--chunks", default=None, help="チャンク番号（コンマ区切り。既定: 最新から 1,000 チャンクごとに 6 点）")
    p_gp.add_argument("--n-chunks", type=int, default=10, help="各点で読むチャンク数")
    p_gp.add_argument("--positions", type=int, default=2000)
    p_gp.add_argument("--seed", type=int, default=0)
    p_gp.add_argument("--json", action="store_true")
    p_pg = sub.add_parser("progress", help="学習の進み具合の要約をリポジトリのファイルに書き出す（--publish で別ブランチへ push。docs/runbook.md §6）")
    p_pg.add_argument("--out", default=None, help="書き出し先のディレクトリ（既定: 標準出力に JSON を出すだけ）")
    p_pg.add_argument("--publish", action="store_true", help="リポジトリの --branch に 1 コミット足して push する（作業ツリーと HEAD には触れない）")
    p_pg.add_argument("--repo", default=None, help="リポジトリ（既定: このチェックアウト）")
    p_pg.add_argument("--branch", default="progress", help="書き出し先のブランチ（既定: progress。main には入れない）")
    p_pg.add_argument("--dir", dest="subdir", default="progress", help="ブランチの中のディレクトリ（既定: progress）")
    p_pg.add_argument("--points", type=int, default=120, help="metrics.jsonl から残す点の数")
    p_pg.add_argument("--remote", default="origin")
    p_pg.add_argument("--no-push", dest="push", action="store_false", help="コミットまで作って push しない")
    p_cf = sub.add_parser("config", help="効いている設定と、その出どころ（リポジトリの config/<run-id>.toml）を見る。反映待ちの違いも出す")
    p_cf.add_argument("--ref", default=None, help="読むところ（既定: origin/main。none で作業ツリーのファイル）")
    p_cf.add_argument("--repo", default=None, help="リポジトリ（既定: このチェックアウト）")
    p_cf.add_argument("--adopt", action="store_true", help="今の <run>/config.toml をリポジトリの config/<run-id>.toml に写す（git に入れると、そちらが正になる）")
    p_cf.add_argument("--force", action="store_true", help="--adopt で既にあるファイルを上書きする")
    p_cf.add_argument("--json", action="store_true")
    p_sc = sub.add_parser("scaling", help="局を何倍にすると何 Elo 伸びるかを、固定の参照の保存済みの計測から出す（docs/scaling-2026-09-18.md §4）")
    p_sc.add_argument("--cost-per-1m", type=float, default=None, help="100 万局あたりの費用（ドル。既定: measurements.md 2026-09-17 の $8）")
    p_sc.add_argument("--band", default=None, help="Elo が縮まない得点の範囲（lo,hi。既定 0.2,0.8）")
    p_sc.add_argument("--json", action="store_true")
    p_rt = sub.add_parser("rating", help="記録した対局を全部まとめて 1 本の Elo の目盛りにする（Bradley-Terry。参照を入れ替えても鎖を継ぎ足さない）")
    p_rt.add_argument("--anchor", default=None, help="0 Elo に置く点（既定: いちばん古い step）")
    p_rt.add_argument("--curve", action="store_true", help="総局数に対する伸び（2 倍あたりの Elo）も出す")
    p_rt.add_argument("--json", action="store_true")
    p_ab = sub.add_parser("abtest", help="「仮」の設定値のオフライン比較: 保存済みの重みと同じ窓から、学習の設定だけを変えた腕を同じ step 学習し、対局させる（docs/restart-plan.md §0）")
    p_ab.add_argument("--ckpt", default=None, help="元の重み（既定: <run>/checkpoints/latest.pt。ふつうは checkpoints/archive の節目）")
    p_ab.add_argument("--arm", action="append", required=True, help="腕: <名前>[:<節>.<鍵>=<値>[,...]]（例 lam05:train.lambda_z=0.5 と lam10:train.lambda_z=1.0）")
    p_ab.add_argument("--set", action="append", default=[], help="全部の腕に先に当てる上書き（例 train.compile=none）")
    p_ab.add_argument("--steps", type=int, default=5000, help="腕ごとの学習 step 数")
    p_ab.add_argument("--games", type=int, default=1000, help="腕どうし・腕対元の重みの対局数（0 で対局しない）")
    p_ab.add_argument("--sims", type=int, default=96)
    p_ab.add_argument("--concurrent", type=int, default=64)
    p_ab.add_argument("--threads", type=int, default=8)
    p_ab.add_argument("--seed", type=int, default=41, help="腕の間で同じ局面を引く種（対局の種にも使う）")
    p_ab.add_argument("--positions", type=int, default=4000, help="一般化の物差しで測る局面数")
    p_ab.add_argument("--log-every", type=int, default=500, help="学習の損失を何 step ごとに出すか")
    p_ab.add_argument("--no-vs-base", dest="vs_base", action="store_false", help="元の重みとの対局を省く")
    p_ab.add_argument("--chunk-index", type=int, default=None, help="窓の右端（既定: チェックポイントに保存された値）")
    p_ab.add_argument("--games-total", type=int, default=None, help="窓の大きさの計算に使う総局数（既定: 同上）")
    p_ab.add_argument("--config-from", default="ckpt", choices=["ckpt", "run"], help="元にする設定（既定: チェックポイントに保存された設定）")
    p_ab.add_argument("--device", default=None)
    p_ab.add_argument("--out", default=None, help="出力先（既定: <root>/experiments/<時刻>-abtest）。稼働中の run には書かない")
    p_ab.add_argument("--publish", action="store_true", help="結果を progress ブランチの experiments/ へ push する（手元の PC の外から読めるようにする）")
    p_ab.add_argument("--branch", default="progress", help="--publish の書き出し先（既定 progress。main には入れない）")
    p_rv = sub.add_parser("review", help="物差し M1〜M4（gen・最強比・基準比・参照・局/日）の保存済みの値から「続ける／注意／見直し」を出す（docs/restart-plan.md §3 M6）")
    p_rv.add_argument("--set", action="append", default=[], help="閾値の上書き（name=value。gen_games, gen_max_gap, gen_max_fall, gen_min_corr, best_stall_alert, reference_games, gpd_min）")
    p_rv.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    sd = StateDir(Path(a.root) / a.run)
    if a.cmd == "run":
        if not a.no_supervise:
            # 異常終了（CUDA の abort など）したら起動し直す。停止フラグでの終了や二重起動では起動し直さない
            from .supervise import child_argv, supervise

            return supervise(sd, child_argv(a.root, a.run, a.config))
        from .runner import main_run

        main_run(sd.root, Path(a.config) if a.config else None)
        return 0
    if a.cmd == "worker":
        import signal

        import torch

        from .config import load_config
        from .workers import Worker

        if not sd.config_toml.exists():
            print(f"no run at {sd.root} (config.toml not found)", file=sys.stderr)
            return 1
        cfg = load_config(sd.config_toml)
        inbox = Path(a.inbox) if a.inbox else sd.inbox
        inbox.mkdir(parents=True, exist_ok=True)
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        w = Worker(cfg, a.id, Path(a.weights) if a.weights else sd.weights / "latest.pt", inbox, device, n_games=a.n_games, threads=a.threads,
                   stop_root=None if a.detached else sd.root, lock=None if a.detached else sd.root / "run.lock",
                   log=lambda s: print(s, flush=True))

        def on_signal(signum, frame):
            w.stopping = True

        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, on_signal)
        return w.run()
    if a.cmd == "export":
        ckpt = Path(a.ckpt) if a.ckpt else sd.root / "checkpoints" / "latest.pt"
        out = Path(a.out) if a.out else sd.root / "checkpoints" / "latest.onnx"
        export_onnx_file(ckpt, out)
        return 0
    if a.cmd == "openings":
        from .openings import openings_from_replay, write_openings

        lines = openings_from_replay(sd.replay, a.chunks, a.moves)
        out = Path(a.out) if a.out else sd.root / "openings.json"
        write_openings(out, lines, str(sd.root))
        print(f"wrote {out}: {len(lines)} openings")
        return 0
    if a.cmd == "calib":
        from .calibrate import format_table, make_row, newest_games

        state = sd.read_state() if sd.state_json.exists() else {}
        row = make_row(newest_games(sd.replay, a.games), state.get("step"), state.get("generation"), a.bins)
        print(json.dumps(row, ensure_ascii=False) if a.json else format_table(row))
        return 0
    if a.cmd == "config":
        from .runconfig import adopt, local_path, repo_config_path, resolve

        if a.adopt:
            try:
                p, _ = adopt(sd, Path(a.repo) if a.repo else None, a.force)
            except (FileExistsError, FileNotFoundError) as e:
                print(e, file=sys.stderr)
                return 1
            print(f"写した: {p}\ngit に入れると、次の起動からそちらが正になる（作業ツリーが古くても origin/main の中身を読む）")
            return 0

        # 下見なので何も書かず、ランナー向けのログも出さない（この下でまとめて出す）
        cfg, info = resolve(sd, None, Path(a.repo) if a.repo else None, a.ref, apply=False, log=lambda m: None)
        if a.json:
            print(json.dumps({"info": info, "config": cfg}, ensure_ascii=False, default=str))
            return 0
        print(f"run: {sd.root}")
        print(f"設定の正: {info['source']}" + (f"（{info['ref']}）" if info.get("ref") else ""))
        print(f"リポジトリのファイル: {repo_config_path(sd.root.name, Path(a.repo) if a.repo else None)}")
        print(f"この PC だけの上書き: {local_path(sd)}" + ("（あり）" if info.get("local") else "（無し）"))
        if info.get("adopt_hint"):
            print(f"リポジトリにまだ設定が無い（{info['adopt_hint']}）。`--adopt` で今の設定を写せる")
        if info["changed"]:
            print(f"反映待ちの違い {len(info['changed'])} 件（コンソールの停止 → 起動で効く）:")
            for line in info["changed"]:
                print(f"  {line}")
        else:
            print("反映待ちの違いは無い")
        return 0
    if a.cmd == "scaling":
        from .scaling import BAND, COST_PER_1M_USD, render, scaling

        band = BAND if a.band is None else tuple(float(x) for x in a.band.split(","))
        r = scaling(sd, a.cost_per_1m if a.cost_per_1m is not None else COST_PER_1M_USD, band)
        print(json.dumps(r, ensure_ascii=False) if a.json else render(r))
        return 0
    if a.cmd == "rating":
        from .rating import curve as rating_curve
        from .rating import rating, render as rating_render

        r = rating(sd, a.anchor)
        if a.curve:
            r["curve"] = rating_curve(sd, r)
        if a.json:
            print(json.dumps(r, ensure_ascii=False))
        else:
            print(rating_render(r))
            c = r.get("curve")
            if c and c.get("fit"):
                print("")
                print(f"== 総局数に対する伸び（土台 {c['anchor']}）==")
                for p_ in c["points"]:
                    print(f"{p_['games']:>11,} | {p_['elo']:+8.1f} | {p_['node']}")
                for iv in c["intervals"]:
                    print(f"  {iv['games_from']:>9,} → {iv['games_to']:>9,}  {iv['d_elo']:+7.1f} Elo"
                          f"  2 倍あたり {iv['elo_per_doubling']:+7.1f}")
                f_ = c["fit"]
                print(f"  {f_['n']} 点の当てはめ: {f_['elo_per_doubling']:+.1f} Elo / 2 倍（残差 {f_['rms_resid']}）")
        return 0
    if a.cmd == "review":
        from .review import format_review, review

        th = {}
        for kv in a.set:
            k, v = kv.split("=", 1)
            th[k] = float(v)
        r = review(sd, th)
        print(json.dumps(r, ensure_ascii=False) if a.json else format_review(r))
        return 0
    if a.cmd == "genprof":
        import torch

        from libra_net.export_onnx import load_checkpoint

        from .config import load_config
        from .genprof import format_rows, profile

        cfg = load_config(sd.config_toml if sd.config_toml.exists() else None)
        ckpt = Path(a.ckpt).expanduser() if a.ckpt else sd.checkpoints / "latest.pt"
        model, _ = load_checkpoint(ckpt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if a.chunks:
            starts = [int(x) for x in a.chunks.split(",")]
        else:
            newest = max((int(p.stem.split("_")[1]) for p in sd.replay.glob("chunk_*.pkl")), default=-1)
            starts = [max(0, newest + 1 - a.n_chunks - k * 1000) for k in range(6) if newest + 1 - a.n_chunks - k * 1000 >= 0]
        rows = profile(model, sd.replay, starts, a.n_chunks, a.positions, device, cfg["train"]["lambda_z"], cfg["search"]["policy_topk"],
                       cfg["search"]["max_ply"], cfg["search"]["count_from_41"], a.seed)
        print(json.dumps({"ckpt": str(ckpt), "rows": rows}, ensure_ascii=False) if a.json else f"{ckpt}\n" + format_rows(rows))
        return 0
    if a.cmd == "abtest":
        import time

        import torch

        from .abtest import format_abtest, run_abtest
        from .config import load_config

        cfg = load_config(sd.config_toml if sd.config_toml.exists() else None)
        ckpt = Path(a.ckpt).expanduser() if a.ckpt else sd.checkpoints / "latest.pt"
        out = Path(a.out).expanduser() if a.out else Path(a.root).expanduser() / "experiments" / (time.strftime("%Y%m%d-%H%M%S") + "-abtest")
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        res = run_abtest(sd, cfg, ckpt, a.arm, a.set, a.steps, a.games, a.sims, a.concurrent, a.threads, a.seed, a.positions,
                         a.log_every, a.vs_base, out, device, a.chunk_index, a.games_total, lambda s: print(s, flush=True),
                         config_from=a.config_from)
        print(format_abtest(res))
        print("written:", out / "abtest.json")
        if a.publish:
            from .abtest import publish_result

            publish_result(res, out.name, None, a.branch, "experiments", lambda s: print(s, flush=True))
        return 0
    if a.cmd == "stop":
        sd.set_flag("STOP")
        print("STOP set")
        return 0
    if a.cmd in ("eval-now", "match-now"):
        name = "EVAL_NOW" if a.cmd == "eval-now" else "MATCH_NOW"
        sd.set_flag(name)
        print(f"{name} set")
        return 0
    if a.cmd == "restart-exploiter":
        from .config import load_config
        from .restart import restart_exploiter

        restart_exploiter(sd, load_config(sd.config_toml), Path(a.snapshot).expanduser(), a.apply)
        return 0
    if a.cmd == "eval":
        import time

        from .config import load_config
        from .evaluate import main_eval

        cfg = load_config(sd.config_toml if sd.config_toml.exists() else None)
        scfg = dict(cfg["search"])
        scfg["full_sims"] = a.sims
        if a.no_noise:
            scfg["gumbel_noise"] = False
        scfg_b = None
        if a.b_set:
            from .abtest import parse_set

            scfg_b = {}
            for kv in a.b_set:
                section, key, value = parse_set(kv if "." in kv.split("=", 1)[0] else f"search.{kv}")
                if section != "search":
                    raise SystemExit("--b-set は [search] の鍵だけ")
                scfg_b[key] = value
        if a.out:
            out = Path(a.out).expanduser()
        elif scfg_b is not None:
            # 側ごとに設定を変えた対局は run の eval/ に置かない（コンソールの Elo の一覧に世代間の計測として並んでしまう）
            out = Path(a.root).expanduser() / "experiments" / (time.strftime("%Y%m%d-%H%M%S") + "-sigma.json")
        else:
            out = sd.root / "eval" / (time.strftime("%Y%m%d-%H%M%S") + ".json")
        res = main_eval(Path(a.a), Path(a.b), scfg, a.games, a.concurrent, a.threads, a.seed, out, search_cfg_b=scfg_b)
        print(json.dumps({k: v for k, v in res.items() if not k.startswith("calibration")}, ensure_ascii=False))
        for side in ("a", "b"):
            print(f"calibration_{side}:", " ".join(f"[{c['lo']:.1f},{c['hi']:.1f}) n={c['n']} pred={c['pred']:.2f} act={c['actual']:.2f}" for c in res[f"calibration_{side}"]))
        print("written:", out)
        if a.publish:
            from .abtest import publish_result

            publish_result(res, out.stem, None, a.branch, "experiments", lambda s: print(s, flush=True))
        return 0
    if a.cmd == "match":
        import time

        from .harness import run_match
        from .usi_client import UsiEngine

        root = Path(__file__).resolve().parents[2]
        out = Path(a.out) if a.out else sd.root / "matches" / (time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
        logf = open(out.with_suffix(".log"), "a", encoding="utf-8") if out.parent.exists() or not out.parent.mkdir(parents=True, exist_ok=True) else None

        def log(s: str) -> None:
            print(s, flush=True)
            if logf:
                logf.write(s + "\n")
                logf.flush()

        lopts = {"Declare_Win": "true", "DNN_Model": resolve_match_model(sd, a.model, a.ckpt, log)}
        for kv in a.libra_opt:
            k, v = kv.split("=", 1)
            lopts[k] = v
        oopts = {}
        for kv in a.opponent_opt:
            k, v = kv.split("=", 1)
            oopts[k] = v
        if a.opponent:
            ocmd = a.opponent.split()
        else:
            ocmd = [str(Path(a.opponent_cwd) / ".venv" / "bin" / "python"), "scripts/fuseki_usi_server.py"]
        libra = UsiEngine("libra", [str(root / "bin" / "libra-usi")], cwd=str(root), options=lopts, log=lambda s: logf and logf.write(s + "\n"))
        opp = UsiEngine("opp", ocmd, cwd=a.opponent_cwd, options=oopts, log=lambda s: logf and logf.write(s + "\n"))
        log(f"starting engines: libra={libra.cmd} opp={ocmd}")
        libra.start()
        opp.start()
        log(f"libra: {libra.id_name}  opp: {opp.id_name}")
        try:
            summary = run_match(libra, opp, a.games, a.go, out, log=log, first_placer=a.first_placer)
        finally:
            libra.quit()
            opp.quit()
        summary["go"] = a.go
        summary["libra_options"] = lopts
        summary["opponent_options"] = oopts
        summary["opponent_cmd"] = ocmd
        out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "games"}, ensure_ascii=False))
        print("written:", out)
        return 0
    if a.cmd == "progress":
        from .config import load_config as _lc
        from .progress import files_for, message_for, publish, repo_root

        cfg = _lc(sd.config_toml if sd.config_toml.exists() else None)
        files, snap = files_for(sd, cfg, a.points, a.subdir)
        if a.out:
            d = Path(a.out)
            for rel, content in files.items():
                f = d / Path(rel).name
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(content, encoding="utf-8")
                print("written:", f)
        if a.publish:
            repo = Path(a.repo) if a.repo else repo_root()
            publish(repo, a.branch, files, message_for(snap), remote=a.remote, push=a.push)
        elif not a.out:
            print(json.dumps(snap, ensure_ascii=False))
        return 0
    if a.cmd == "status":
        st = read_json(sd.status_json)
        state = sd.read_state()
        from .supervise import running_pid

        running = running_pid(sd.root / "run.lock") is not None
        flags = [f for f in StateDir.FLAGS if sd.flag(f)]
        if a.json:
            out = {
                "run": a.run,
                "root": str(sd.root),
                "exists": bool(st or state),
                "process": "running" if running else "not running",
                "flags": flags,
                "state": {k: state.get(k) for k in ("step", "generation", "games_total", "last_checkpoint", "elapsed")} if state else None,
                "status": st or None,
            }
            if a.tail > 0 and sd.log.exists():
                out["log_tail"] = sd.log.read_text(encoding="utf-8", errors="replace").splitlines()[-a.tail:]
            from .config import load_config as _lc

            _ac = _lc(sd.config_toml if sd.config_toml.exists() else None)["auto"]
            out["auto_cfg"] = {"enabled": bool(_ac.get("enabled")), "every_hours": _ac.get("every_hours"), "every_games": _ac.get("every_games"),
                               "eval_games": _ac.get("eval_games"), "anchor_games": _ac.get("anchor_games"),
                               "best_games": _ac.get("best_games"), "reference_games": _ac.get("reference_games"),
                               "reference_ckpts": _ac.get("reference_ckpts"), "match_games": _ac.get("match_games")}
            if a.history > 0:
                from .auto import collect_anchor, collect_best, collect_evals, collect_matches, collect_reference, list_archives, load_metrics

                out["metrics"] = load_metrics(sd, a.history)
                out["evals"] = collect_evals(sd)
                out["anchor"] = collect_anchor(sd)
                out["best"] = collect_best(sd)
                out["reference"] = collect_reference(sd)
                out["matches"] = collect_matches(sd)
                # 計測の行に「その重みを保存した時点の総局数」を足す（管理コンソールの Elo グラフの横軸。
                # 行の games は打ち終わった時刻の総局数で archive より後なので使えない。scaling.games_of_step と同じ）
                from .scaling import games_of_step, time_of_step

                _m = load_metrics(sd, 100000)
                g_of = games_of_step(_m)
                for key, step_key in (("anchor", "step"), ("best", "step"), ("reference", "step"),
                                      ("matches", "libra_step"), ("evals", "step_b")):
                    for row in out[key]:
                        row["games_at"] = g_of(row.get(step_key))
                # 全部の対局をまとめて 1 本にした Elo の目盛り（Bradley-Terry）。管理コンソールの Elo の
                # グラフはこれを主役にする（相手ごとの線は 200 局ずつで幅が広く、入れ替えとじゃんけんで
                # 上下するため、下がっていないのに下がって見える。2026-09-19 のユーザーの指摘）
                try:
                    from .rating import curve as rating_curve
                    from .rating import rating as rating_fit

                    _rt = rating_fit(sd)
                    _cv = rating_curve(sd, _rt, g_of=g_of, t_of=time_of_step(_m))
                    out["rating"] = {"anchor": _rt.get("anchor"), "fit": _rt.get("fit"),
                                     "points": _cv.get("points"), "curve_fit": _cv.get("fit"),
                                     "curve_fit_recent": _cv.get("fit_recent"),
                                     "recent_doublings": _cv.get("recent_doublings"),
                                     "thin": _cv.get("thin")}
                except Exception as e:  # 目盛りが出せなくても status は返す
                    out["rating"] = {"error": f"{type(e).__name__}: {e}"}
                from .calibrate import load_calib

                out["calib"] = load_calib(sd, a.history)
                out["archives"] = [{"step": s_, "file": p.name} for p in list_archives(sd) if (s_ := int(p.stem.split("_")[1])) is not None]
                out["auto"] = state.get("auto") if state else None
                out["exploiter_state"] = state.get("exploiter") if state else None
            print(json.dumps(out, ensure_ascii=False))
            return 0
        if not st and not state:
            print(f"no run at {sd.root}")
            return 1
        print(f"run: {sd.root}  process: {'running' if running else 'not running'}  flags: {flags or '-'}")
        if st:
            eng = st.get("engine", {})
            g = max(1, eng.get("games", 0) or 1)
            print(f"time {st['time']}  step {st['step']}  generation {st['generation']}  games_total {st['games_total']}  "
                  f"games/day(1h) {st['games_per_day_1h']}  window {st['window_games']}  elapsed {st['elapsed_h']} h  active {st['active_games']}")
            if eng:
                print(f"session: games {eng.get('games')}  avg plies {eng.get('plies_sum', 0) / g:.1f}  "
                      f"sente/draw/gote {eng.get('sente_wins')}/{eng.get('draws')}/{eng.get('gote_wins')}  "
                      f"ruling41 {eng.get('ruling41')}  mate {eng.get('no_legal_move')}  sennichite {eng.get('sennichite')}  "
                      f"perpetual {eng.get('perpetual_check')}  max_ply {eng.get('max_ply')}  sims/move {eng.get('sims', 0) / max(1, eng.get('moves', 1)):.1f}")
            if st.get("train"):
                print("train:", json.dumps(st["train"]))
            if st.get("timing"):
                print("timing:", format_timing(st["timing"]))
            if st.get("gen"):
                from .genprof import format_gen

                print(format_gen(st["gen"]))
            if st.get("workers"):
                wk = st["workers"]
                print(f"workers: games {wk.get('games')} files {wk.get('files')} stale {wk.get('stale_games')} rejected {wk.get('rejected_files')}  "
                      + " ".join(f"{k}={v}" for k, v in (wk.get("by_worker") or {}).items()))
            if st.get("restarts"):
                print("restarts:", ", ".join(st["restarts"]))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
