# SPDX-License-Identifier: Apache-2.0
"""実行設定。TOML（config.toml）で上書きする。既定は Libra-L の L-S 構成（docs/libra-local.md §3.2）。"""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "run_id": "ls",
    "seed": 1,
    "net": {"d_model": 320, "n_layers": 8, "n_heads": 8, "d_ff": 1280, "dropout": 0.0},  # 約 10M（L-S）
    "search": {
        "full_sims": 96,
        "fast_sims": 24,
        "full_prob": 0.25,
        "gumbel_m_full": 16,
        "gumbel_m_fast": 8,
        "c_visit": 50.0,
        "c_scale": 1.0,
        # true なら σ に入れる q を mctx の completed Q（未訪問は v_mix、根の手の間で [0,1] に正規化）にする。そのとき c_scale は mctx の既定 0.1 に
        # 合わせる（docs/method-evidence.md §2.3、docs/restart-plan.md §4）。false は 2026-09-17 までの形
        "gumbel_rescale": False,
        "cpuct": 1.5,
        "draw_value": 0.0,
        "max_ply": 320,
        "count_from_41": True,
        "policy_topk": 32,
        "max_moves_per_game": 400,
        "mate_nodes_root": 200,
        "proof_nodes": 1000,
        "proof_min_ply": 36,
        # 投了（AlphaGo Zero [Silver+ 2017] Methods「Resignation」）。0 で無効（既定）。
        # 手番側の探索後の値が −resign_threshold 以下の状態がその側の連続 resign_runs 手続いたら、その手を指した後に投了する。
        # resign_disable_prob の対局は投了させず最後まで打つ（誤投了の割合を測り続けるため。原典と同じ 10%）。
        # 棋譜からの見積もりは `bin/libra resign`。0.90・1 手で評価の節約 19%・局/日 1.24 倍・誤投了 2.61%（docs/measurements.md 2026-09-19）。
        # 入れると 1 局が約 2 割短くなるので、リプレイの窓（局数で数えている）とセットで見る
        "resign_threshold": 0.0,
        "resign_runs": 1,
        "resign_disable_prob": 0.1,
        "resign_min_ply": 40,
        "defer_root_proof": True,  # 根の証明探索を GPU の評価中に解く（棋譜は変わらない。false で apply の中で解く）
        # 後手玉を一〜三段目に限る（四段目は 3 手目の桂打ちで先手の裁定勝ち。docs/rules.md §3.4）。搾取者 lx は true、
        # 本体 ls は false のまま（選ぶ側が悪い配置も評価できるように。docs/decisions.md 2026-09-11）。
        # 読むのは C++（libra-search/python/bindings.cpp の getb）だけだが、ここに無いと config/lx.toml が
        # 起動のたびに「今のプログラムが知らない設定」として警告に出る（unknown_keys）
        "prune_gote_rank4": False,
        # 自己対局で後手玉が四段目になる確率。負なら 36 マスから一様（＝0.25。乱数の引き方も入れる前と同じ）。
        # 四段目は 41 手目の裁定で先手の勝ちが決まるので、本体 ls ではこれで減らす（丸ごとは除かない。docs/v0.3-plan.md §2・§5）。
        # 計測の対局（evaluate.play_match）はこの値を無視して一様に置く（目盛りを節目の間で揃えるため）
        "gote_rank4_prob": -1.0,
    },
    "selfplay": {
        "n_games": 512,
        "threads": 12,
        "infer_dtype": "float16",
        "compile": "max-autotune",  # 推論の捕獲: none | default | max-autotune（CUDA のときだけ効く。CUDA Graphs は常に使う）
        "openings": "",            # 搾取者が見つけた布石（openings.json）。空なら使わない
        "openings_prob": 0.1,      # 新規対局が openings から始まる確率
        "openings_reload_seconds": 600,
    },
    # 搾取者（docs/libra-design.md §4.2）: main_ckpt を凍結した相手にして、自分の手だけを学習する。
    # 本体が強くなると凍結相手が古すぎて勝率が飽和するので、refresh_hours ごと、または main_source が
    # refresh_steps 以上先へ進んだら main_source で作り直し、成績と布石をそこで区切る
    # （収束判定「対本体勝率が頭打ち」は現行の本体に対してでないと意味がないため）。
    "exploiter": {"main_ckpt": "", "main_dtype": "float16",
                  "main_source": "",         # 作り直しの元（本体ランの checkpoints/latest.pt）。空なら作り直さない
                  "refresh_hours": 0.0,      # 作り直しの間隔（0 で無効。初回は起動直後に行う）
                  "refresh_steps": 0,        # main_source の step が凍結相手より何 step 進んだら作り直すか（0 で無効）
                  "refresh_check_minutes": 5.0,  # refresh_steps のために main_source の step を読む間隔
                  # 搾取者の手の探索木の中で、本体の手番の葉の方策を凍結した本体のネットから取る（価値は常に搾取者のネット）。
                  # 本体の応手を本体の方策で予測する（Wang+ 2023 の A-MCTS に倣う。docs/exploiter-literature.md）。false で木を丸ごと根の手番のネットで評価
                  "opponent_prior": True,
                  "openings_out": "",        # 見つけた布石の書き出し先（空なら書かない）。本体はこれを読む
                  "openings_minutes": 60.0,
                  "openings_chunks": 50,
                  "openings_moves": 12,
                  # 自分のスナップショットの保存先（本体の run の [league] pool が読む）。起動時にプールが空なら今の自分を、
                  # 凍結相手を作り直すたびに作り直す前の自分を lx-<step>.pt で保存し、新しい pool_keep 個を残す。空なら保存しない
                  "pool_out": "",
                  "pool_keep": 30,
                  # 課程（docs/exploiter-literature.md §2.3・§3 の 7、docs/lx-settings.md）: 凍結した本体の読みの回数を
                  # curriculum_sims の段で弱くしておき、直近 curriculum_games 局の勝率が curriculum_threshold 以上になったら
                  # 次の段へ上げる。最後の段を越えたら本番の読み（[search] の full_sims・fast_sims）。空なら課程なし。
                  # 課程の間の局は対本体勝率（収束の物差し）にも本体へ渡す布石にも数えない
                  "curriculum_sims": [],
                  "curriculum_threshold": 0.75,
                  "curriculum_games": 2000},
    # 本体と過去の搾取者の対局（libra_league/league.py、docs/decisions.md 2026-09-14）: 自己対局とは別のエンジン（別の固定バッチ）で
    # n_games 局を pool（搾取者の [exploiter] pool_out）の新しい recent 体と打ち、本体の手だけを方策の学習に使う。本体はふだん通り自分のネットで読む。
    # 相手は PFSP（本体の勝率 x に (1 − x)²）で switch_games 局ごとに選び直す。プールが空なら pool_check_minutes ごとに見に行く。搾取者の run では無効
    "league": {"enabled": False, "pool": "", "n_games": 64, "threads": 4, "recent": 30, "switch_games": 256, "pool_check_minutes": 10.0},
    "train": {
        "batch_size": 1024,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 1000,
        "replay_ratio": 4.0,
        "window_games": 100000,    # 窓の最小（局）。window_frac > 0 なら総局数 × window_frac まで広げる（上限 window_games_max、0 で無制限）。
        "window_frac": 0.0,        # 0 で固定の窓（2026-09-17 までの ls・lx）。KataGo [Wu19] は総数に応じて広げる（docs/restart-plan.md §4）
        "window_games_max": 0,     # 1 局 約 5 KB（100 万局で約 5 GB。RAM の空きで決める）
        "min_window_games": 2000,
        "train_every_games": 256,
        "policy_weight": 1.0,
        "value_weight": 1.0,
        "v41_weight": 0.5,
        "lambda_z": 0.5,
        "mirror_prob": 0.5,
        "grad_clip": 1.0,
        "full_only": False,       # True で全読みの局面だけを学習に使う（KataGo [Wu19] §3.1）。今は `libra abtest` の腕だけ。ランは起動で断る
        "opp_weight": 0.0,        # > 0 で補助方策「相手の次の手」を学習する（KataGo [Wu19] §3.4 は 0.15。net.opp_head が要る）。今は `libra abtest` の腕だけ
        "accum_steps": 1,         # > 1 でバッチを分けて勾配を足し合わせる（大きいネットがメモリに載らないとき。`libra abtest --scratch` 用）
        "compile": "max-autotune",  # 学習の forward・逆伝播の torch.compile: none | default | max-autotune（CUDA のときだけ効く。2026-09-16）
    },
    # 自己対局ワーカー（libra worker、libra_league/workers.py）: 学習側は重みを <run>/weights/latest.pt に配り（学習のたび）、
    # <run>/inbox/ に届いた局を ingest_seconds ごとに手元の自己対局と同じようにリプレイへ足す。既定は無効（今の 1 プロセスのまま）。
    # max_lag_steps: 局を打った重みが学習側より何 step 遅れていたら捨てるか（0 で捨てない）。搾取者の run では無効。
    "workers": {"enabled": False, "ingest_seconds": 10.0, "max_lag_steps": 2000},
    "run": {"checkpoint_minutes": 10, "status_seconds": 30, "chunk_games": 100, "keep_checkpoints": 3,
            "export_onnx": True,   # チェックポイントごとに latest.onnx も書く（libra / libra.exe 用）
            "metrics_minutes": 5,   # metrics.jsonl（進捗の時系列）の追記間隔
            # calib.jsonl: 窓の最新 calib_games 局で、探索値と結果の較正（libra calib と同じ）を calib_minutes ごとに足す。0 で無効
            "calib_minutes": 60, "calib_games": 20000,
            # held-out（docs/restart-plan.md §3 M1）: チャンク番号が heldout_every_chunks の倍数の局は学習に使わず、新しい heldout_games 局を持つ。
            # gen_minutes ごとに窓の中と held-out の gen_positions 局面で一般化の物差し（genprof.py）を測り、status・metrics の "gen" に出す。0 で無効
            # gen_games > 0 なら gen_minutes ではなく局数ごとに測る（局/日が変わっても同じ局数ごと。2026-09-17 のユーザーの指示）
            "heldout_every_chunks": 0, "heldout_games": 20000, "gen_minutes": 60, "gen_games": 0, "gen_positions": 4000},
    # 自動計測（docs/runbook.md §6）: 総局数が every_games の倍数を越えるごとにチェックポイントを archive に残し、直前の archive と対局させて Elo を鎖にする。
    # match_games > 0 なら外部エンジン（fuseki_usi_server.py）とも少数局を指す。どちらも別プロセスで GPU を共有する。
    # anchor_*: 固定の基準ネットとの対局。連続世代どうしの Elo は伸びが測定幅（100 局で ±70 Elo）に埋もれ、
    # 鎖にすると誤差が回数の平方根で積み上がる。基準との差は大きいままなので信号が残り、誤差も積み上がらない。
    # 基準に対する勝率が anchor_rebaseline を超えたら基準を新しい世代に置き換え、それまでの差を offset に足す。
    # every_games: 総局数がその倍数を越えるごとに測る（判断に要る局数で区切る。2026-09-17 のユーザーの指示）。0 で無効。
    # 時間区切り（every_hours）は 2026-09-19 に廃止した。局/日が PC の利用状況で変わるので、時間では測る間隔が定まらないため
    "auto": {"enabled": False, "every_games": 0, "eval_games": 100, "eval_sims": 96, "eval_concurrent": 64, "eval_threads": 4,
             "chain_eval": True, "anchor_games": 100, "anchor_rebaseline": 0.85,
             # 最強比（docs/restart-plan.md §3 M2）: これまでで最強の保存済みと best_games 局。95% 区間の下限が 0 を超えたら最強を置き換える。
             # best_stall_alert 回続けて更新できなければログに WARNING。0 で無効
             "best_games": 0, "best_stall_alert": 3,
             # 固定の参照（同 M4）: run をまたいで同じ重み（例: 旧 ls の 646,699 と実験の win1m.pt）と reference_games 局ずつ打つ。空で無効
             "reference_ckpts": [], "reference_games": 0,
             # 参照に勝ちすぎたら自動で外し、そのときの archive を参照にする（0 で無効。docs/runbook.md §6）。
             # 勝率が 1 に寄ると Elo が縮み、伸びていても曲線が寝て見えるため
             "reference_rotate": 0.0, "reference_min": 1,
             # match_opponent_opt: 相手のルールの版は既定が変わっても今のルール（二飛香）で指させるため明示する（decisions.md 2026-09-13）
             # match_go は Libra 側の `go`。match_go_opp を書くと相手だけ別の `go` になり、読む量に差を付けて測れる
             # （勝ちすぎて勝率が 1 に寄ると 1 局あたりの情報が減り、全勝すると Elo が出ない。docs/runbook.md §6）。
             # match_libra_opt は Libra への setoption（例 Sims_Normal=400,Sims_Fuseki=200）。空なら送らない
             # match_fuseki: 布石をどうするか。engine=両エンジンに打たせる／selfplay=自己対局の 41 手目の局面を使う（＝最強 Libra 同士の布石）
             # match_opponent: 相手の起動コマンド（空なら布石つきの fuseki_usi_server.py、match_fuseki が engine 以外なら やねうら王）
             # match_go_opp にコンマを書くと段ごとに分けて測る（例 "nodes 1000, nodes 10000"）。段ごとに Elo の別の点になる
             # match_libra_standard: Libra 側が自己評価と同じ読み（eval_sims）で打つとき true。Elo の目盛りで同じ点として扱う
             "match_games": 10, "match_go": "movetime 1000", "match_go_opp": "", "match_opponent_opt": "Threads=2,Fuseki_Rules=2",
             # match_use_best: 外部計測を「最強比が決めた最強の重み」で打つ（節目の新しい重みではなく）。
             # ジョブを始めるときに決めるので、同じ節目の最強比の結果を待ってから決まる
             "match_libra_opt": "", "match_fuseki": "engine", "match_opponent": "", "match_opponent_cwd": "",
             "match_libra_standard": False, "match_use_best": True},
    # 進捗の書き出し（libra_league/progress.py、docs/runbook.md §6）: 自動計測が動いた節目と heartbeat_minutes ごとに、
    # 数値の要約を repo の branch へ push する。~/libra-run の値をクラウドのセッションからも読めるようにするため（2026-09-18）。
    # repo が空ならこのチェックアウト。push には git の認証（gh の credential helper）が要る。
    "progress": {"enabled": False, "repo": "", "branch": "progress", "dir": "progress", "push": True,
                 "heartbeat_minutes": 180, "min_seconds": 120, "metrics_points": 120},
}


_MISSING = object()


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def unknown_keys(parsed: dict[str, Any]) -> list[str]:
    """設定に、今のプログラムが知らない鍵があれば `[節].鍵` の形で返す。

    設定はリポジトリの `config/<run-id>.toml`（`origin/main`）から読むのに、プログラムは手元の作業ツリー
    なので、**`git pull` を忘れると新しい鍵が黙って無視される**。2026-09-19 に `match_go_opp`（相手だけ
    別の `go`）がこれで効かず、相手まで 1 手 400 回になった 40 局を「勝率 100%」として記録してしまった。
    """
    out: list[str] = []
    for k, v in parsed.items():
        base = DEFAULTS.get(k, _MISSING)
        if base is _MISSING:
            out.append(k)
        elif isinstance(v, dict) and isinstance(base, dict):
            out += [f"{k}.{k2}" for k2 in v if k2 not in base]
    return sorted(out)


def load_config(path: Path | None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None and path.exists():
        with open(path, "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    return cfg


def dump_toml(cfg: dict[str, Any]) -> str:
    """設定を TOML 文字列にする（state ディレクトリへ写しを残す用。値は基本型だけ）。"""
    lines: list[str] = []

    def fmt(v: Any) -> str:
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        # TOML の基本文字列: 逆斜線を先に、次に引用符を逃がす（Windows の置き場所が入っても壊れない）
        return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'

    for k, v in cfg.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {fmt(v)}")
    for k, v in cfg.items():
        if isinstance(v, dict):
            lines.append(f"\n[{k}]")
            for k2, v2 in v.items():
                lines.append(f"{k2} = {fmt(v2)}")
    return "\n".join(lines) + "\n"
