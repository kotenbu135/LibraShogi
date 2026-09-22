# SPDX-License-Identifier: Apache-2.0
"""`UsiEngine` の起動まわり。相手が起動できないときに理由が残るか（2026-09-21）。"""
import sys

import pytest
from libra_league.usi_client import UsiEngine


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_an_engine_that_dies_says_what_it_printed_to_stderr():
    """相手が起動できずに落ちたら、標準エラーの末尾を理由として添える。

    2026-09-21 まで `stderr=subprocess.DEVNULL` で捨てていたので、やねうら王が評価ファイルを読めずに
    落ちても理由がどこにも残らず、外部計測が 3 回続けて 1 局も記録しなかった原因を追えなかった。"""
    eng = UsiEngine("opp", _py("import sys; sys.stderr.write('Error : can not read the file nn.bin\\n'); sys.exit(1)"))
    with pytest.raises(RuntimeError) as e:
        eng.start(ready_timeout=5)
    assert "process exited" in str(e.value)
    assert "can not read the file nn.bin" in str(e.value)
    assert eng.stderr_tail and "nn.bin" in eng.stderr_tail[-1]
    eng.quit()


def test_stderr_does_not_get_mixed_into_the_usi_lines():
    """標準エラーは USI の行ではないので、待ち行列に混ぜない（混ぜると `usiok` の待ちが壊れる）。"""
    eng = UsiEngine("opp", _py(
        "import sys\n"
        "sys.stderr.write('info: loading eval\\n')\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if line == 'usi':\n"
        "        print('id name Fake 1.0'); print('usiok')\n"
        "    elif line == 'isready':\n"
        "        print('readyok')\n"
        "    elif line == 'quit':\n"
        "        break\n"
        "    sys.stdout.flush()\n"))
    eng.start(ready_timeout=10)
    assert eng.id_name == "Fake 1.0"
    assert eng.stderr_tail == ["info: loading eval"]
    eng.quit()


def test_an_engine_that_never_answers_times_out_with_its_stderr():
    """黙ったまま生きている相手は時間切れになり、そこにも標準エラーの末尾が付く。"""
    eng = UsiEngine("opp", _py(
        "import sys, time\n"
        "sys.stderr.write('waiting for the eval file\\n'); sys.stderr.flush()\n"
        "time.sleep(20)\n"))
    with pytest.raises(TimeoutError) as e:
        eng.start(usi_timeout=0.5)
    assert "no response within" in str(e.value)
    assert "waiting for the eval file" in str(e.value)
    eng.quit()


def test_an_engine_that_complains_on_stdout_and_dies_says_what_it_printed():
    """起動の失敗を標準出力に書いて終わる相手（やねうら王がそう）も、その行が理由に残る。

    2026-09-22 の外部計測は `opp: process exited` だけで、標準エラーは空だった。"""
    eng = UsiEngine("opp", _py("print('Error! : can not open the eval file EvalDir = vendor/yaneuraou_eval')"))
    with pytest.raises(RuntimeError) as e:
        eng.start(usi_timeout=5)
    assert "process exited" in str(e.value)
    assert "can not open the eval file" in str(e.value)
    eng.quit()
