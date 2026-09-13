// SPDX-License-Identifier: Apache-2.0
// 黒箱テストの相手。tenbin-shogi-desktop の public/wasm/fuseki.mjs（dlshogi 由来、GPL-3.0）を
// 別プロセスとして動かし、合法手集合だけを標準入出力で返す。wasm のコードは読まず、実行するだけ。
// 使い方: node wasm_oracle.mjs <path/to/fuseki.mjs>
//   stdin 1 行 1 命令: reset [rules] / drop X*sq / legal / sfen / verify <sfen> / attacked 0|1 / done / ply / quit
//   rules は fw_reset の旗（0 = 布石将棋、1 = 天秤将棋の二飛香。desktop 0.8.0 の Issue #1 の API）
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';

const url = pathToFileURL(process.argv[2]).href;
const mod = await import(url);
const M = await mod.default({});
M.ccall('fw_init', null, [], []);
// 二飛香の旗が無い古い wasm（desktop 0.8.0 より前）では比べられない
if (typeof M._fw_rule_nihikyo !== 'function' || M.ccall('fw_rule_nihikyo', 'number', [], []) !== 1) {
  console.error('wasm has no fw_rule_nihikyo (desktop 0.8.0 or later is required)');
  process.exit(2);
}

function legal() {
  const n = M.ccall('fw_legal_drops', 'number', [], []);
  const p = M.ccall('fw_drops_pt_ptr', 'number', [], []) >> 2;
  const q = M.ccall('fw_drops_sq_ptr', 'number', [], []) >> 2;
  const out = [];
  for (let i = 0; i < n; i++) {
    const pt = M.HEAP32[p + i], sq = M.HEAP32[q + i];
    out.push(M.ccall('fw_move_to_usi', 'string', ['number', 'number'], [pt, sq]) + '#' + pt + '#' + sq);
  }
  return out;
}

const rl = createInterface({ input: process.stdin });
for await (const line of rl) {
  const t = line.trim().split(/\s+/);
  const cmd = t[0];
  if (cmd === 'reset') { M.ccall('fw_reset', null, ['number'], [Number(t[1] ?? 0)]); console.log('ok'); }
  else if (cmd === 'legal') { console.log(legal().map((s) => s.split('#')[0]).join(' ')); }
  else if (cmd === 'drop') {
    const found = legal().find((s) => s.split('#')[0] === t[1]);
    if (!found) { console.log('illegal'); continue; }
    const [, pt, sq] = found.split('#');
    M.ccall('fw_do_drop', null, ['number', 'number'], [Number(pt), Number(sq)]);
    console.log('ok');
  }
  else if (cmd === 'sfen') { console.log(M.ccall('fw_to_sfen', 'string', [], [])); }
  else if (cmd === 'verify') { console.log(M.ccall('fw_verify_final_sfen', 'number', ['string'], [t.slice(1).join(' ')])); }
  else if (cmd === 'attacked') { console.log(M.ccall('fw_is_king_attacked', 'number', ['number'], [Number(t[1])])); }
  else if (cmd === 'done') { console.log(M.ccall('fw_is_placement_done', 'number', [], [])); }
  else if (cmd === 'ply') { console.log(M.ccall('fw_ply', 'number', [], []) + ' ' + M.ccall('fw_turn', 'number', [], [])); }
  else if (cmd === 'quit') { process.exit(0); }
  else { console.log('?'); }
}
