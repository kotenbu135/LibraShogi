// SPDX-License-Identifier: Apache-2.0
// ネットワーク入出力の符号化。手番側（mover）を下にした座標系（後手なら 180° 回転）で表す。
#pragma once
#include "libra/position.h"

namespace libra {

// 方策: 行き先マス 81 × クラス 28。
//   クラス 0..9  : 不成の盤上の手。0..7 は行き先から見た元マスの方向（DIR8）、8,9 は桂の跳び元（左・右）
//   クラス 10..19: 同じく成る手
//   クラス 20..27: 打つ手（歩 香 桂 銀 金 角 飛 玉）
constexpr int POLICY_CLASSES = 28;
constexpr int POLICY_SIZE = SQ_NB * POLICY_CLASSES;  // 2268

// 入力: マスごとの特徴 81 × SQ_FEATS と、グローバル特徴 GLOB_FEATS
constexpr int SQ_FEATS = 32;
constexpr int GLOB_FEATS = 32;

// 手番側の座標系へ（後手なら 180° 回転）
constexpr int to_mover_frame(Color turn, int sq) { return turn == BLACK ? sq : 80 - sq; }

int move_index(const Position& pos, Move m);        // 手 → 方策の添字（手番側の座標系）
Move move_from_index(const Position& pos, int idx); // 逆変換。解決できなければ MOVE_NONE（合法性は見ない）
int mirror_index(int idx);                          // 1↔9 筋の鏡映（手番側の座標系での筋反転）
void write_features(const Position& pos, float* sq_out, float* glob_out, bool mirror = false);  // sq_out: 81*SQ_FEATS, glob_out: GLOB_FEATS。mirror で 1↔9 筋反転

}  // namespace libra
