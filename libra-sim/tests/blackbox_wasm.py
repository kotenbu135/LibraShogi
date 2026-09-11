# SPDX-License-Identifier: Apache-2.0
"""合法手の黒箱テスト: ランダムな布石局面で desktop の wasm（GPL、別プロセス）と libra-sim の合法手集合を突き合わせる。

環境変数 TENBIN_DESKTOP_DIR（tenbin-shogi-desktop の clone）が無ければスキップする。
使い方: PYTHONPATH=libra-sim/python python libra-sim/tests/blackbox_wasm.py [games] [seed]
"""
import os
import random
import subprocess
import sys

import librashogi as ls

DESKTOP = os.environ.get("TENBIN_DESKTOP_DIR", os.path.expanduser("~/tenbin-shogi-desktop"))
WASM = os.path.join(DESKTOP, "public", "wasm", "fuseki.mjs")
ORACLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wasm_oracle.mjs")


class Oracle:
    def __init__(self):
        self.p = subprocess.Popen(["node", ORACLE, WASM], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def ask(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        return self.p.stdout.readline().rstrip("\n")

    def legal(self):
        s = self.ask("legal")
        return set(s.split()) if s else set()

    def close(self):
        try:
            self.ask("quit")
        except Exception:
            pass
        self.p.wait(timeout=5)


def play_game(oracle, rng, mode, aggressive, stats):
    """1 局の布石を進め、各手番で合法手集合を比較する。aggressive なら先手は後手玉に当てる手を優先する。"""
    pos = ls.Position(mode)
    oracle.ask("reset")
    for ply in range(40):
        mine = set(pos.legal_moves())
        theirs = oracle.legal()
        if mode == "tenbin" and ply < 2:
            theirs = {m for m in theirs if m.startswith("K*")}  # GUI 側は kings フェーズで玉だけに絞る
        if mine != theirs:
            raise AssertionError(
                f"ply {ply} mismatch\n{pos.sfen()}\nonly libra: {sorted(mine - theirs)}\nonly wasm: {sorted(theirs - mine)}"
            )
        if ply == 39:
            stats["ply39"] += 1
            if pos.king_attacked("gote"):
                stats["restricted"] += 1
            if pos.ruling41_pending():
                stats["unblockable"] += 1
        moves = sorted(mine)
        pick = None
        if aggressive and pos.turn == "sente" and ply >= 2:
            attacking = []
            for m in moves:
                pos.do_move(m)
                if pos.king_attacked("gote"):
                    attacking.append(m)
                pos.undo()
            if attacking and rng.random() < 0.7:
                pick = rng.choice(attacking)
        if pick is None:
            pick = rng.choice(moves)
        pos.do_move(pick)
        assert oracle.ask("drop " + pick) == "ok", pick
    assert oracle.ask("done") == "1" and pos.phase == "normal"
    sfen = pos.sfen()
    their_sfen = oracle.ask("sfen")
    if sfen != their_sfen:
        raise AssertionError(f"sfen mismatch\n{sfen}\n{their_sfen}")
    verify = oracle.ask("verify " + sfen) == "1"
    result, reason = pos.outcome()
    ours_ruling = reason == "ruling41"
    if verify == ours_ruling:
        raise AssertionError(f"ruling41 mismatch: wasm verify={verify} libra={result}/{reason}\n{sfen}")
    if ours_ruling:
        stats["ruling41"] += 1
    assert (oracle.ask("attacked 1") == "1") == pos.king_attacked("gote")
    assert (oracle.ask("attacked 0") == "1") == pos.king_attacked("sente")


def main():
    if not os.path.exists(WASM):
        print(f"skip: {WASM} not found")
        return 0
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = random.Random(seed)
    oracle = Oracle()
    stats = {"ply39": 0, "restricted": 0, "unblockable": 0, "ruling41": 0}
    n = 0
    try:
        for g in range(games):
            mode = "tenbin" if g % 4 else "fuseki"
            play_game(oracle, rng, mode, aggressive=(g % 2 == 0), stats=stats)
            n += 1
    finally:
        oracle.close()
    print(f"ok: {n} games, 40 plies each, legal sets identical. {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
