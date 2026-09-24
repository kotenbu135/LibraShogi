# SPDX-License-Identifier: Apache-2.0
"""評価ハーネス（evaluate.play_match）: 局数ちょうど・先後半分ずつで打ち切ること、止められても続きから打つこと。"""
import json
from pathlib import Path

import pytest
import torch

from libra_league import evaluate
from libra_net.model import LibraNet, NetConfig

CFG = {"full_sims": 4, "fast_sims": 4, "gumbel_m_full": 4, "gumbel_m_fast": 4, "max_ply": 40, "count_from_41": False,
       "proof_min_ply": 30, "proof_nodes": 100, "mate_nodes_root": 50}


def _net(seed: int) -> LibraNet:
    torch.manual_seed(seed)
    return LibraNet(NetConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32)).eval()


def _play(n: int, concurrent: int, **kw) -> dict:
    return evaluate.play_match(_net(1), _net(2), CFG, n, concurrent, 2, 0, torch.device("cpu"), torch.float32, **kw)


@pytest.mark.parametrize("n,concurrent", [(7, 4), (6, 16), (1, 2)])
def test_play_match_plays_exactly_n_games_half_with_each_colour(n: int, concurrent: int):
    """先に終わった n 局を取るのではなく、始める前に数える対局を決めるので、局数も先後の数もちょうどになる
    （打ちかけを捨てると短い対局に偏る）。枠が要る局数より多くても（6 局を 16 枠）同じ。"""
    res = _play(n, concurrent)
    assert res["n"] == n
    assert sum(res["a_as_sente"].values()) == (n + 1) // 2
    assert sum(res["a_as_gote"].values()) == n // 2
    assert res["concurrent"] <= max(2, concurrent) and res["resumed_games"] == 0


class _Boom(Exception):
    pass


class _Breaks(torch.nn.Module):
    """k 回呼ばれたら例外を出すネット（ランの停止で計測ジョブが止められたのの代わり）。"""

    def __init__(self, net: LibraNet, k: int):
        super().__init__()
        self.net, self.k = net, k

    def forward(self, sq, glob):
        self.k -= 1
        if self.k < 0:
            raise _Boom
        return self.net(sq, glob)


def test_play_match_resumes_from_the_partial_file(tmp_path: Path):
    """止められたら打ち終えた対局は途中経過に残り、同じ条件で呼び直すと続きから打つ（打ち直さない）。"""
    part = tmp_path / "best.json.partial.jsonl"
    tag = {"a": "a.pt", "b": "b.pt", "seed": 0}
    with pytest.raises(_Boom):
        evaluate.play_match(_net(1), _Breaks(_net(2), 250), CFG, 10, 4, 2, 0, torch.device("cpu"), torch.float32,
                            partial=part, partial_tag=tag)
    lines = part.read_text(encoding="utf-8").splitlines()
    before = [json.loads(s) for s in lines[1:]]
    assert 0 < len(before) < 10
    res = _play(10, 4, partial=part, partial_tag=tag)
    assert res["n"] == 10 and res["resumed_games"] == len(before)
    assert sum(res["a_as_sente"].values()) == 5 and sum(res["a_as_gote"].values()) == 5
    after = [json.loads(s) for s in part.read_text(encoding="utf-8").splitlines()[1:]]
    assert after[:len(before)] == before and len(after) == 10
    # 条件が違う途中経過（別の組・別の局数）は使わずに退け、始めから打つ
    res = _play(10, 4, partial=part, partial_tag={**tag, "b": "c.pt"})
    assert res["resumed_games"] == 0 and res["n"] == 10
    assert part.with_name(part.name + ".stale").exists()


def test_play_match_skips_a_torn_last_line(tmp_path: Path):
    """書きかけで止められた最後の行は捨てる。"""
    part = tmp_path / "x.partial.jsonl"
    tag = {"a": "a.pt", "b": "b.pt", "seed": 0}
    _play(4, 4, partial=part, partial_tag=tag)
    rows = part.read_text(encoding="utf-8").splitlines()
    part.write_text("\n".join(rows[:3]) + "\n" + rows[3][:10], encoding="utf-8")  # 2 局と書きかけの 1 行
    res = _play(4, 4, partial=part, partial_tag=tag)
    assert res["resumed_games"] == 2 and res["n"] == 4


def test_main_eval_removes_the_partial_file_when_done(tmp_path: Path, monkeypatch):
    for name, seed in (("a.pt", 1), ("b.pt", 2)):
        m = _net(seed)
        torch.save({"model": m.state_dict(), "config": {"net": {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32}}}, tmp_path / name)
    out = tmp_path / "eval" / "best-x.json"
    res = evaluate.main_eval(tmp_path / "a.pt", tmp_path / "b.pt", CFG, 4, 4, 2, 0, out)
    assert res["n"] == 4 and out.exists()
    assert not out.with_name(out.name + ".partial.jsonl").exists()
