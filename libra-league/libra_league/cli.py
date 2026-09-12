# SPDX-License-Identifier: Apache-2.0
"""`libra` コマンド: run / pause / resume / stop / throttle / status（docs/libra-local.md §7.2）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .state import DEFAULT_ROOT, StateDir, read_json


def default_match_model(sd: StateDir) -> Path:
    """match の既定モデル。bin/libra-usi は C++ 版（ONNX）なので latest.onnx を優先し、無ければ Python 版用の latest.pt。"""
    onnx = sd.checkpoints / "latest.onnx"
    return onnx if onnx.exists() else sd.checkpoints / "latest.pt"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="libra", description="Libra の自己対局・学習ランナー")
    ap.add_argument("--run", default="ls", help="run-id（状態ディレクトリ ~/libra-run/<run-id>）")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="状態ディレクトリの親")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="前回状態から再開（無ければ新規）")
    p_run.add_argument("--resume", action="store_true", help="（既定と同じ。互換のため）")
    p_run.add_argument("--config", default=None, help="config.toml（初回だけ有効。以後は状態ディレクトリの写しを使う）")
    sub.add_parser("pause", help="PAUSE フラグを置く。ワーカーは現在のバッチを終えて待機")
    sub.add_parser("resume", help="PAUSE フラグを消す")
    sub.add_parser("stop", help="STOP フラグを置く。チェックポイントを書いて終了")
    p_th = sub.add_parser("throttle", help="同時進行局数を絞る")
    p_th.add_argument("--games", type=int, required=True, help="同時進行局数（0 で解除）")
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
    p_ev.add_argument("--out", default=None, help="結果 JSON の出力先（既定: <run>/eval/<時刻>.json）")
    p_m = sub.add_parser("match", help="計測: Libra（USI）と外部エンジンを無人対局させ棋譜を JSONL に残す")
    p_m.add_argument("--games", type=int, default=20)
    p_m.add_argument("--go", default="movetime 3000", help="go の引数（例: 'movetime 3000' / 'btime 60000 wtime 60000 byoyomi 10000'）")
    p_m.add_argument("--opponent", default=None, help="相手エンジンの起動コマンド（既定: fuseki_usi_server.py）")
    p_m.add_argument("--opponent-cwd", default=str(Path.home() / "fuseki-shogi-ai"))
    p_m.add_argument("--opponent-opt", action="append", default=[], help="相手の setoption（name=value）")
    p_m.add_argument("--libra-opt", action="append", default=[], help="Libra の setoption（name=value）")
    p_m.add_argument("--model", default=None, help="Libra のモデル（既定: <run>/checkpoints/latest.onnx。無ければ latest.pt = Python 版エンジン用）")
    p_m.add_argument("--out", default=None, help="棋譜 JSONL（既定: <run>/matches/<時刻>.jsonl）")
    p_m.add_argument("--first-placer", default="a", choices=["a", "b"], help="第 1 局で両玉を置く側（a=Libra）")
    p_ex = sub.add_parser("export", help="チェックポイント（.pt）を推論用 ONNX に書き出す（libra / libra.exe 用）")
    p_ex.add_argument("--ckpt", default=None, help="既定: <run>/checkpoints/latest.pt")
    p_ex.add_argument("--out", default=None, help="既定: <run>/checkpoints/latest.onnx（同じ場所に一時ファイルを書いてから置き換える）")
    p_op = sub.add_parser("openings", help="搾取者の run から、搾取者が勝った布石を openings.json に書き出す")
    p_op.add_argument("--chunks", type=int, default=50, help="新しい側から何チャンク（100 局単位）見るか")
    p_op.add_argument("--moves", type=int, default=12, help="玉 2 手のあとに残す布石の手数")
    p_op.add_argument("--out", default=None, help="既定: <run>/openings.json")
    a = ap.parse_args(argv)
    sd = StateDir(Path(a.root) / a.run)
    if a.cmd == "run":
        from .runner import main_run

        main_run(sd.root, Path(a.config) if a.config else None)
        return 0
    if a.cmd == "export":
        import os

        from libra_net.export_onnx import check, export_checkpoint, load_checkpoint

        ckpt = Path(a.ckpt) if a.ckpt else sd.root / "checkpoints" / "latest.pt"
        out = Path(a.out) if a.out else sd.root / "checkpoints" / "latest.onnx"
        tmp = out.with_suffix(".onnx.tmp")
        meta = export_checkpoint(ckpt, tmp)
        model, _ = load_checkpoint(ckpt)
        diff = check(model, tmp)
        os.replace(tmp, out)
        print(f"exported {out} step {meta['libra_step']} max|ort-torch| {diff:.1e}")
        return 0
    if a.cmd == "openings":
        from .openings import openings_from_replay, write_openings

        lines = openings_from_replay(sd.replay, a.chunks, a.moves)
        out = Path(a.out) if a.out else sd.root / "openings.json"
        write_openings(out, lines, str(sd.root))
        print(f"wrote {out}: {len(lines)} openings")
        return 0
    if a.cmd == "pause":
        sd.set_flag("PAUSE")
        print("PAUSE set")
        return 0
    if a.cmd == "resume":
        sd.clear_flag("PAUSE")
        print("PAUSE cleared")
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
    if a.cmd == "throttle":
        if a.games <= 0:
            sd.clear_flag("THROTTLE")
            print("THROTTLE cleared")
        else:
            sd.set_flag("THROTTLE", str(a.games))
            print(f"THROTTLE {a.games}")
        return 0
    if a.cmd == "eval":
        import time

        from .config import load_config
        from .evaluate import main_eval

        cfg = load_config(sd.config_toml if sd.config_toml.exists() else None)
        scfg = dict(cfg["search"])
        scfg["full_sims"] = a.sims
        out = Path(a.out) if a.out else sd.root / "eval" / (time.strftime("%Y%m%d-%H%M%S") + ".json")
        res = main_eval(Path(a.a), Path(a.b), scfg, a.games, a.concurrent, a.threads, a.seed, out)
        print(json.dumps({k: v for k, v in res.items() if not k.startswith("calibration")}, ensure_ascii=False))
        for side in ("a", "b"):
            print(f"calibration_{side}:", " ".join(f"[{c['lo']:.1f},{c['hi']:.1f}) n={c['n']} pred={c['pred']:.2f} act={c['actual']:.2f}" for c in res[f"calibration_{side}"]))
        print("written:", out)
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

        lopts = {"Declare_Win": "true", "DNN_Model": a.model or str(default_match_model(sd))}
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
    if a.cmd == "status":
        st = read_json(sd.status_json)
        state = sd.read_state()
        lock = sd.root / "run.lock"
        running = False
        if lock.exists():
            pid = lock.read_text().strip()
            running = Path(f"/proc/{pid}").exists()
        flags = [f for f in StateDir.FLAGS if sd.flag(f)]
        if a.json:
            out = {
                "run": a.run,
                "root": str(sd.root),
                "exists": bool(st or state),
                "process": "running" if running else "not running",
                "flags": flags,
                "throttle": sd.throttle_value(),
                "state": {k: state.get(k) for k in ("step", "generation", "games_total", "last_checkpoint", "elapsed")} if state else None,
                "status": st or None,
            }
            if a.tail > 0 and sd.log.exists():
                out["log_tail"] = sd.log.read_text(encoding="utf-8", errors="replace").splitlines()[-a.tail:]
            if a.history > 0:
                from .auto import collect_evals, collect_matches, list_archives, load_metrics

                out["metrics"] = load_metrics(sd, a.history)
                out["evals"] = collect_evals(sd)
                out["matches"] = collect_matches(sd)
                out["archives"] = [{"step": s_, "file": p.name} for p in list_archives(sd) if (s_ := int(p.stem.split("_")[1])) is not None]
                out["auto"] = state.get("auto") if state else None
                from .config import load_config

                out["auto_cfg"] = load_config(sd.config_toml if sd.config_toml.exists() else None)["auto"]
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
            if st.get("restarts"):
                print("restarts:", ", ".join(st["restarts"]))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
