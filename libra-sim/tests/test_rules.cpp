// SPDX-License-Identifier: Apache-2.0
// ルールテスト（docs/rules.md の各節に対応）。依存なしの最小テストハーネス。
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "libra/position.h"

using namespace libra;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond)                                                              \
  do {                                                                           \
    if (cond) ++g_pass;                                                          \
    else { ++g_fail; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
  } while (0)
#define CHECK_EQ(a, b) CHECK((a) == (b))

static std::set<std::string> legal_set(Position& p) {
  MoveList ml;
  p.legal_moves(ml);
  std::set<std::string> s;
  for (Move m : ml) s.insert(move_to_usi(m));
  return s;
}

// 駒の一覧から布石の SFEN を組む。sq は "7g" 形式、駒は "P" など（大文字＝先手、小文字＝後手）。
// hands は SFEN の持ち駒欄。
static std::string build_sfen(const std::vector<std::pair<std::string, std::string>>& pieces, const std::string& hands,
                              const std::string& turn = "b") {
  std::string grid[9][9];
  for (auto& row : grid)
    for (auto& c : row) c = "";
  for (auto& [pc, sq] : pieces) {
    int s = sq_from_usi(sq);
    grid[rank_of(s)][file_of(s)] = pc;
  }
  std::string out;
  for (int r = 0; r < 9; ++r) {
    int empty = 0;
    for (int f = 8; f >= 0; --f) {
      if (grid[r][f].empty()) { ++empty; continue; }
      if (empty) { out += char('0' + empty); empty = 0; }
      out += grid[r][f];
    }
    if (empty) out += char('0' + empty);
    if (r != 8) out += '/';
  }
  return out + " " + turn + " " + hands + " 1";
}

static void test_usi_sfen() {
  CHECK_EQ(sq_to_usi(sq_from_usi("7g")), "7g");
  CHECK_EQ(move_to_usi(move_from_usi("K*5i")), "K*5i");
  CHECK_EQ(move_to_usi(move_from_usi("7g7f")), "7g7f");
  CHECK_EQ(move_to_usi(move_from_usi("2b8h+")), "2b8h+");
  CHECK_EQ(mirror_sq(sq_from_usi("1a")), sq_from_usi("9a"));
  Position p;
  CHECK_EQ(p.sfen(), "9/9/9/9/9/9/9/9/9 b KRB2G2S2N2L9Pkrb2g2s2n2l9p 1");
  CHECK(p.set_position("position fuseki moves K*5i K*5a choose:sente P*7g P*3c"));
  CHECK_EQ(p.sfen(), "4k4/9/6p2/9/9/9/2P6/9/4K4 b RB2G2S2N2L8Prb2g2s2n2l8p 5");
  Position q;
  CHECK(q.set_sfen(p.sfen(), PHASE_FUSEKI));
  CHECK_EQ(q.key(), p.key());
  CHECK_EQ(q.ply(), 4);
  CHECK(p.set_position("position startpos"));
  CHECK_EQ(p.sfen(), "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
  CHECK(p.set_position("position startpos moves 7g7f 3c3d 8h2b+ 3a2b"));
  CHECK_EQ(p.sfen(), "lnsgkg1nl/1r5s1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/7R1/LNSGKGSNL b Bb 5");
  CHECK_EQ(p.normal_ply(), 4);
  CHECK(p.set_position("position sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 moves 7g7f"));
  CHECK_EQ(p.turn(), WHITE);
  CHECK(!p.set_position("position fuseki moves K*5e"));  // 5 段目は非合法
  CHECK(!p.set_position("position fuseki moves P*7g"));  // 天秤将棋の 1 手目は玉
}

static void test_zobrist() {
  Position a, b;
  CHECK(a.set_position("position fuseki moves K*5i K*5a P*7g P*3c S*6h G*4b"));
  CHECK(b.set_position("position fuseki moves K*5i K*5a S*6h G*4b P*7g P*3c"));
  CHECK_EQ(a.key(), b.key());       // 着手順に依存しない
  CHECK_EQ(a.sfen(), b.sfen());
  Position c;
  CHECK(c.set_position("position fuseki moves K*5i K*5a P*3g P*7c S*4h G*6b"));  // 1↔9 筋の鏡映
  CHECK(c.key() != a.key());
  CHECK_EQ(c.mirror_key(), a.key());
  CHECK_EQ(c.norm_key(), a.norm_key());
  Position d;
  CHECK(d.set_position("position fuseki moves K*5i K*5a P*7g"));
  CHECK(d.key() != a.key());
  // 手番が違えば鍵も違う（同じ盤・持ち駒でも）: 本将棋で確かめる
  Position e, f;
  CHECK(e.set_sfen("k8/9/9/9/9/9/9/9/K8 b - 1", PHASE_NORMAL));
  CHECK(f.set_sfen("k8/9/9/9/9/9/9/9/K8 w - 1", PHASE_NORMAL));
  CHECK(e.key() != f.key());
  // do/undo で鍵が戻る
  std::uint64_t k0 = a.key(), m0 = a.mirror_key();
  a.do_move(move_from_usi("N*2i"));
  a.undo_move();
  CHECK_EQ(a.key(), k0);
  CHECK_EQ(a.mirror_key(), m0);
}

static void test_fuseki_drops() {
  Position p;
  CHECK(p.set_position("position fuseki"));
  auto s = legal_set(p);
  CHECK_EQ(s.size(), 36u);  // 天秤将棋の 1 手目は先手陣 36 マスへの玉打ち
  CHECK(s.count("K*5i"));
  CHECK(!s.count("K*5a"));
  CHECK(!s.count("P*7g"));
  p.do_move(move_from_usi("K*5i"));
  s = legal_set(p);
  CHECK_EQ(s.size(), 36u);
  CHECK(s.count("K*5a"));
  p.do_move(move_from_usi("K*5a"));
  s = legal_set(p);
  CHECK_EQ(s.size(), 7u * 35u);
  CHECK(!s.count("K*4i"));
  CHECK(!s.count("P*5e"));
  CHECK(s.count("P*5h"));
  // 二歩: 7 筋に歩があれば 7 筋に歩を打てない
  p.do_move(move_from_usi("P*7g"));
  p.do_move(move_from_usi("P*3c"));
  s = legal_set(p);
  CHECK(!s.count("P*7f"));
  CHECK(!s.count("P*7h"));
  CHECK(s.count("G*7f"));
  CHECK(s.count("P*8f"));
  // 筋埋め禁止: 8 筋の 3 マスを歩以外で埋めたら残る 1 マスに歩以外は打てない
  Position q;
  CHECK(q.set_position("position fuseki moves K*5i K*5a L*8i P*1c N*8h P*2c S*8g P*3c"));
  s = legal_set(q);
  CHECK(!s.count("G*8f"));
  CHECK(!s.count("B*8f"));
  CHECK(s.count("P*8f"));
  CHECK(s.count("G*7f"));
  // 布石将棋モード: 1 手目から全種
  Position r;
  CHECK(r.set_position("position fuseki", MODE_FUSEKI));
  CHECK_EQ(legal_set(r).size(), 8u * 36u);
}

// 二飛香（天秤将棋のみ）: 自陣の同じ筋に自分の飛・香は合わせて 1 枚まで。手順は Issue #1 の確認用の手順。
static const char* NIHIKYO_40 =
    "K*8g K*6b choose:sente P*1g R*8a S*7g N*7c S*9g P*4d L*5g S*6c P*8f B*3b B*3f P*8d G*8h P*3d P*5i G*7b N*4f "
    "P*7d P*9f P*9d G*6h P*1d P*2h G*5b P*4g S*4c P*7f P*6d N*6f P*2d P*3g N*5d R*2f P*5c P*6g L*2c L*7i L*5a";

static std::vector<std::string> split_tokens(const std::string& s) {
  std::vector<std::string> v;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(' ', i);
    if (j == std::string::npos) j = s.size();
    if (j > i) v.push_back(s.substr(i, j - i));
    i = j + 1;
  }
  return v;
}

static void test_nihikyo() {
  auto toks = split_tokens(NIHIKYO_40);
  std::string pre23 = "position fuseki moves";
  for (int i = 0; i < 24; ++i) pre23 += " " + toks[i];  // choose を含めて 23 手
  // 後手の反則: 8 筋に後手の飛 8a がある
  Position p;
  CHECK(p.set_position(pre23));
  CHECK_EQ(p.ply(), 23);
  auto s = legal_set(p);
  CHECK(!s.count("L*8c"));
  CHECK(!s.count("L*8d"));
  CHECK(s.count("P*1d"));
  CHECK(s.count("G*8c"));  // 飛・香以外は同じ筋に打てる
  CHECK(!p.set_position(pre23 + " L*8c"));
  // 先手の反則: 5 筋に先手の飛 5h がある
  Position q;
  CHECK(q.set_position("position fuseki moves K*5i K*5a choose:sente R*5h P*1c"));
  s = legal_set(q);
  CHECK(!s.count("L*5g"));
  CHECK(!s.count("L*5f"));
  CHECK(s.count("L*4g"));
  CHECK(s.count("P*5g"));
  CHECK(s.count("G*5g"));
  // 香が先でも同じ（香香・飛香）
  Position l;
  CHECK(l.set_position("position fuseki moves K*5i K*5a choose:sente L*3i P*1c"));
  s = legal_set(l);
  CHECK(!s.count("L*3h"));
  CHECK(!s.count("R*3f"));
  CHECK(s.count("R*4h"));
  CHECK(s.count("L*4h"));
  // 相手の飛・香は数えない
  Position o;
  CHECK(o.set_position("position fuseki moves K*5i K*5a choose:sente R*5h"));
  s = legal_set(o);
  CHECK(s.count("L*5c"));
  CHECK(s.count("R*5c"));
  // 布石将棋モードでは同じ手が合法のまま
  Position f;
  CHECK(f.set_position("position fuseki moves K*5i K*5a R*5h P*1c", MODE_FUSEKI));
  CHECK(legal_set(f).count("L*5g"));
  std::string pre23f = "position fuseki moves";
  for (int i = 0; i < 24; ++i)
    if (toks[i].rfind("choose:", 0) != 0) pre23f += " " + toks[i];
  CHECK(f.set_position(pre23f + " L*8c", MODE_FUSEKI));
  // 二飛香に合う 40 手はすべて合法で、41 手目の局面まで進む
  Position g;
  CHECK(g.set_position("position fuseki"));
  for (const std::string& t : toks) {
    if (t.rfind("choose:", 0) == 0) continue;
    CHECK(legal_set(g).count(t));
    g.do_move(move_from_usi(t));
  }
  CHECK_EQ(g.phase(), PHASE_NORMAL);
  CHECK_EQ(g.ply(), 40);
}

// 40 手目の制限と二飛香: 飛 5f が後手玉 5c に当たり、遮るマスは 5d だけ。後手の最後の 1 枚は香で、
// 5 筋には後手の香 5b があるので L*5d は二飛香で打てない → 遮る手が無く 41 手目の裁定（二飛香は外れない）。
static void test_nihikyo_move40() {
  std::vector<std::pair<std::string, std::string>> pcs = {{"K", "5i"}, {"R", "5f"}, {"B", "1h"}};
  for (int f = 1; f <= 9; ++f) pcs.push_back({"P", std::to_string(f) + "g"});
  pcs.insert(pcs.end(), {{"L", "1i"}, {"L", "9i"}, {"N", "2i"}, {"N", "8i"}, {"S", "3i"}, {"S", "7i"}, {"G", "4i"}, {"G", "6i"}});
  pcs.insert(pcs.end(), {{"n", "2a"}, {"s", "3a"}, {"g", "4a"}, {"p", "5a"}, {"b", "6a"}, {"s", "7a"}, {"n", "8a"}, {"g", "9a"},
                         {"p", "1b"}, {"p", "2b"}, {"p", "3b"}, {"p", "4b"}, {"l", "5b"}, {"p", "6b"}, {"p", "7b"}, {"p", "8b"}, {"p", "9b"},
                         {"k", "5c"}, {"r", "9c"}});
  const std::string sfen = build_sfen(pcs, "l", "w");
  Position p;
  CHECK(p.set_sfen(sfen, PHASE_FUSEKI));
  CHECK_EQ(p.mode(), MODE_TENBIN);
  CHECK_EQ(p.ply(), 39);
  CHECK(p.king_attacked(WHITE));
  CHECK(p.ruling41_pending());
  auto s = legal_set(p);
  CHECK(!s.count("L*5d"));
  CHECK(!s.count("L*9d"));  // 9 筋には後手の飛 9c
  CHECK_EQ(s.size(), 15u);  // 後手陣の空き 17 マスのうち 5d・9d を除く
  p.do_move(move_from_usi("L*1a"));
  CHECK_EQ(p.outcome().result, BLACK_WIN);
  CHECK_EQ(p.outcome().reason, R_RULING41);
  // 布石将棋モードでは L*5d で遮れる
  Position q;
  q.reset(MODE_FUSEKI);
  CHECK(q.set_sfen(sfen, PHASE_FUSEKI));
  CHECK_EQ(q.mode(), MODE_FUSEKI);
  CHECK(!q.ruling41_pending());
  s = legal_set(q);
  CHECK_EQ(s.size(), 1u);
  CHECK(s.count("L*5d"));
}

// 39 手目直後の局面を組む。先手は 20 枚すべて、後手は 19 枚を置き、最後の 1 枚（金）が持ち駒。
// black_extra / white_layout で飛角の位置を変える。
static std::vector<std::pair<std::string, std::string>> black_pieces(const std::string& bishop_sq) {
  std::vector<std::pair<std::string, std::string>> v = {{"K", "5i"}, {"R", "5f"}, {"B", bishop_sq}};
  for (int f = 1; f <= 9; ++f) v.push_back({"P", std::to_string(f) + "g"});
  v.insert(v.end(), {{"L", "1i"}, {"L", "9i"}, {"N", "2i"}, {"N", "8i"}, {"S", "3i"}, {"S", "7i"}, {"G", "4i"}, {"G", "6i"}});
  return v;
}
static std::vector<std::pair<std::string, std::string>> white_pieces() {
  std::vector<std::pair<std::string, std::string>> v = {
      {"l", "1a"}, {"n", "2a"}, {"s", "3a"}, {"g", "4a"}, {"p", "5a"}, {"b", "6a"}, {"s", "7a"}, {"n", "8a"}, {"l", "9a"},
      {"p", "1b"}, {"p", "2b"}, {"p", "3b"}, {"p", "4b"}, {"k", "5b"}, {"p", "6b"}, {"p", "7b"}, {"p", "8b"}, {"p", "9b"},
      {"r", "9c"}};
  return v;
}

static void test_move40_restriction() {
  // 飛が 5 筋から後手玉（5b）に当たっている。角は 1h で歩に遮られている → 40 手目は 5c か 5d への遮断だけ
  auto pcs = black_pieces("1h");
  auto w = white_pieces();
  pcs.insert(pcs.end(), w.begin(), w.end());
  Position p;
  CHECK(p.set_sfen(build_sfen(pcs, "g", "w"), PHASE_FUSEKI));
  CHECK_EQ(p.ply(), 39);
  CHECK_EQ(p.turn(), WHITE);
  CHECK(p.king_attacked(WHITE));
  CHECK(!p.ruling41_pending());
  auto s = legal_set(p);
  CHECK_EQ(s.size(), 2u);
  CHECK(s.count("G*5c"));
  CHECK(s.count("G*5d"));
  p.do_move(move_from_usi("G*5c"));
  CHECK_EQ(p.phase(), PHASE_NORMAL);
  CHECK_EQ(p.ply(), 40);
  CHECK_EQ(p.normal_ply(), 0);
  CHECK_EQ(p.turn(), BLACK);
  CHECK_EQ(p.outcome().result, ONGOING);
  CHECK(!p.king_attacked(WHITE));
  CHECK_EQ(p.sfen().substr(p.sfen().size() - 7), " b - 41");
  p.undo_move();
  CHECK_EQ(p.phase(), PHASE_FUSEKI);
  CHECK_EQ(legal_set(p).size(), 2u);
}

static void test_ruling41() {
  // 飛（5f→5b）と角（1f→2e→3d→4c→5b）の二本当て。1 枚では遮れない → 制限が外れ、40 手完了で先手勝ち
  auto pcs = black_pieces("1f");
  auto w = white_pieces();
  pcs.insert(pcs.end(), w.begin(), w.end());
  Position p;
  CHECK(p.set_sfen(build_sfen(pcs, "g", "w"), PHASE_FUSEKI));
  CHECK(p.king_attacked(WHITE));
  CHECK(p.ruling41_pending());
  auto s = legal_set(p);
  CHECK_EQ(s.size(), 17u);  // 後手陣の空き 17 マスすべて
  CHECK(s.count("G*5c"));
  CHECK(s.count("G*1d"));
  p.do_move(move_from_usi("G*5c"));
  CHECK_EQ(p.phase(), PHASE_NORMAL);
  CHECK_EQ(p.outcome().result, BLACK_WIN);
  CHECK_EQ(p.outcome().reason, R_RULING41);
  CHECK_EQ(legal_set(p).size(), 0u);
  p.undo_move();
  CHECK_EQ(p.outcome().result, ONGOING);
  // 遮断不能の局面を 40 手完了の SFEN として読んでも同じ裁定
  p.do_move(move_from_usi("G*1d"));
  Position q;
  CHECK(q.set_sfen(p.sfen(), PHASE_NORMAL));
  CHECK_EQ(q.outcome().result, BLACK_WIN);
  CHECK_EQ(q.outcome().reason, R_RULING41);
}

static void test_mate_at_41() {
  // 40 手完了時に先手玉（5f）が飛（5b）と角（1b）の両王手。逃げ場（4e 5e 6e）は桂と飛で塞がれ、先手に合法手なし
  std::vector<std::pair<std::string, std::string>> pcs = {
      {"K", "5f"}, {"P", "5g"}, {"G", "4f"}, {"G", "6f"}, {"S", "4g"}, {"S", "6g"},
      {"P", "1i"}, {"P", "2i"}, {"P", "3i"}, {"P", "4i"}, {"P", "6i"}, {"P", "7i"}, {"P", "8i"}, {"P", "9i"},
      {"L", "1h"}, {"L", "9h"}, {"N", "2h"}, {"N", "8h"}, {"R", "3h"}, {"B", "7h"},
      {"k", "9a"}, {"p", "1a"}, {"b", "1b"}, {"p", "2a"}, {"n", "3c"}, {"p", "3a"}, {"p", "4a"}, {"p", "5a"}, {"r", "5b"},
      {"p", "6a"}, {"n", "7c"}, {"p", "7a"}, {"p", "8a"}, {"p", "9b"}, {"l", "8b"}, {"l", "4b"}, {"s", "6b"}, {"s", "7b"},
      {"g", "8c"}, {"g", "9c"}};
  Position p;
  CHECK(p.set_sfen(build_sfen(pcs, "-", "b"), PHASE_FUSEKI));
  CHECK_EQ(p.phase(), PHASE_NORMAL);
  CHECK(!p.king_attacked(WHITE));
  CHECK(p.in_check());
  CHECK_EQ(legal_set(p).size(), 0u);
  CHECK_EQ(p.outcome().result, WHITE_WIN);
  CHECK_EQ(p.outcome().reason, R_NO_LEGAL_MOVE);
  // 同じ局面を標準 SFEN で読んでも同じ
  Position q;
  CHECK(q.set_sfen(build_sfen(pcs, "-", "b"), PHASE_NORMAL));
  CHECK_EQ(q.outcome().result, WHITE_WIN);
}

static void test_normal_rules() {
  Position p;
  // 打ち歩詰め: 5b への歩打ちが詰みになる → 非合法。金（5c）を外せば玉が取れるので合法
  CHECK(p.set_sfen("4k4/9/4G4/9/9/9/9/9/3L1L2K b P 1", PHASE_NORMAL));
  auto s = legal_set(p);
  CHECK(!s.count("P*5b"));
  CHECK(s.count("P*5d"));
  CHECK(!s.count("P*5a"));  // 一段目に歩は打てない
  CHECK(s.count("P*3b"));
  CHECK(p.set_sfen("4k4/9/9/9/9/9/9/9/3L1L2K b P 1", PHASE_NORMAL));
  CHECK(legal_set(p).count("P*5b"));
  // 成り: 敵陣へ入る手は成れる。歩・香の一段目、桂の二段目以内は成る手だけ
  CHECK(p.set_sfen("k8/9/4P4/9/9/9/9/9/K8 b - 1", PHASE_NORMAL));
  s = legal_set(p);
  CHECK(s.count("5c5b+"));
  CHECK(s.count("5c5b"));
  CHECK(p.set_sfen("k8/4P4/9/9/9/9/9/9/K8 b - 1", PHASE_NORMAL));
  s = legal_set(p);
  CHECK(s.count("5b5a+"));
  CHECK(!s.count("5b5a"));
  CHECK(p.set_sfen("k8/9/9/4N4/9/9/9/9/K8 b - 1", PHASE_NORMAL));
  s = legal_set(p);
  CHECK(s.count("5d4b+"));
  CHECK(!s.count("5d4b"));
  // 王手放置・自殺手は非合法
  CHECK(p.set_sfen("k8/9/9/9/9/9/9/r8/K8 b - 1", PHASE_NORMAL));
  s = legal_set(p);
  CHECK(s.count("9i9h"));   // 飛を取る
  CHECK(s.count("9i8i"));   // 8i は飛の利きに無い
  CHECK(!s.count("9i8h"));  // 8h は飛の利き
  // 二歩
  CHECK(p.set_sfen("k8/9/9/9/9/9/4P4/9/K8 b P 1", PHASE_NORMAL));
  s = legal_set(p);
  CHECK(!s.count("P*5d"));
  CHECK(s.count("P*4d"));
  // 桂・香の打てない段
  CHECK(p.set_sfen("k8/9/9/9/9/9/9/9/K8 b NL 1", PHASE_NORMAL));
  s = legal_set(p);
  CHECK(!s.count("N*5b"));
  CHECK(s.count("N*5c"));
  CHECK(!s.count("L*5a"));
  CHECK(s.count("L*5b"));
}

static void test_repetition() {
  Position p;
  CHECK(p.set_sfen("k8/9/9/9/9/9/9/9/K8 b - 1", PHASE_NORMAL));
  const char* cyc[] = {"9i8i", "9a8a", "8i9i", "8a9a"};
  for (int rep = 0; rep < 3; ++rep)
    for (const char* m : cyc) {
      CHECK_EQ(p.outcome().result, ONGOING);
      p.do_move(move_from_usi(m));
    }
  CHECK_EQ(p.repetition_count(), 4);
  CHECK_EQ(p.outcome().result, DRAW);
  CHECK_EQ(p.outcome().reason, R_SENNICHITE);
  p.undo_move();
  CHECK_EQ(p.outcome().result, ONGOING);
  // 連続王手の千日手: 先手の飛が王手を続ける → 先手の負け
  Position q;
  CHECK(q.set_sfen("k8/8R/9/9/9/9/9/9/8K b - 1", PHASE_NORMAL));
  const char* chk[] = {"1b1a", "9a9b", "1a1b", "9b9a"};
  for (int rep = 0; rep < 3; ++rep)
    for (const char* m : chk) {
      CHECK_EQ(q.outcome().result, ONGOING);
      CHECK(q.is_legal(move_from_usi(m)));
      q.do_move(move_from_usi(m));
    }
  CHECK_EQ(q.outcome().result, WHITE_WIN);
  CHECK_EQ(q.outcome().reason, R_PERPETUAL_CHECK);
  // 後手が王手を続ける版
  Position r;
  CHECK(r.set_sfen("8k/9/9/9/9/9/9/8r/K8 w - 1", PHASE_NORMAL));
  const char* chk2[] = {"1h1i", "9i9h", "1i1h", "9h9i"};
  for (int rep = 0; rep < 3; ++rep)
    for (const char* m : chk2) {
      CHECK(r.is_legal(move_from_usi(m)));
      r.do_move(move_from_usi(m));
    }
  CHECK_EQ(r.outcome().result, BLACK_WIN);
  CHECK_EQ(r.outcome().reason, R_PERPETUAL_CHECK);
}

static void test_declaration() {
  // 先手: 玉 5b、敵陣に飛角金金銀銀桂桂香香歩（11 枚、18 点）、持ち駒 歩 9 → 28 点
  Position p;
  CHECK(p.set_sfen("BNSGKGSNR/L7L/4P4/9/9/9/9/9/4k4 b 9P 1", PHASE_NORMAL));
  CHECK_EQ(p.declaration_pieces(BLACK), 11);
  CHECK_EQ(p.declaration_points(BLACK), 28);
  CHECK(p.can_declare(BLACK));
  CHECK(!p.can_declare(WHITE));
  CHECK(p.set_sfen("BNSGKGSNR/L7L/4P4/9/9/9/9/9/4k4 b 8P 1", PHASE_NORMAL));
  CHECK_EQ(p.declaration_points(BLACK), 27);
  CHECK(!p.can_declare(BLACK));
  p.declare(BLACK);
  CHECK_EQ(p.outcome().result, WHITE_WIN);
  CHECK_EQ(p.outcome().reason, R_ILLEGAL_DECLARATION);
  // 手番でなければ宣言できない
  CHECK(p.set_sfen("BNSGKGSNR/L7L/4P4/9/9/9/9/9/4k4 w 9P 1", PHASE_NORMAL));
  CHECK(!p.can_declare(BLACK));
  // 王手がかかっていれば宣言できない（後手の飛が 5 筋から）
  CHECK(p.set_sfen("BNSGKGSNR/L3r3L/4P4/9/9/9/9/9/4k4 b 9P 1", PHASE_NORMAL));
  CHECK(!p.can_declare(BLACK));
  // 玉が敵陣に無ければ不可
  CHECK(p.set_sfen("BNSG1GSNR/L7L/4P4/4K4/9/9/9/9/4k4 b 9P 1", PHASE_NORMAL));
  CHECK(!p.can_declare(BLACK));
  // 後手は 27 点でよい。敵陣（g〜i 段）に 10 枚、持ち駒 歩 8 → 10 + 8 + 大駒 8 = 26… 飛角 10 + 8 小駒 = 18、持ち駒 9 歩 = 27
  CHECK(p.set_sfen("4K4/9/9/9/9/9/4p4/l7l/bnsgkgsnr w 9p 1", PHASE_NORMAL));
  CHECK_EQ(p.declaration_points(WHITE), 28);
  CHECK(p.can_declare(WHITE));
  CHECK(p.set_sfen("4K4/9/9/9/9/9/4p4/l7l/bnsgkgsnr w 8p 1", PHASE_NORMAL));
  CHECK_EQ(p.declaration_points(WHITE), 27);
  CHECK(p.can_declare(WHITE));
  CHECK(p.set_sfen("4K4/9/9/9/9/9/4p4/l7l/bnsgkgsnr w 7p 1", PHASE_NORMAL));
  CHECK(!p.can_declare(WHITE));
  // 成駒の点数: 竜・馬 5 点
  CHECK(p.set_sfen("+BNSGKGSN+R/L7L/4P4/9/9/9/9/9/4k4 b 9P 1", PHASE_NORMAL));
  CHECK_EQ(p.declaration_points(BLACK), 28);
  p.declare(BLACK);
  CHECK_EQ(p.outcome().result, BLACK_WIN);
  CHECK_EQ(p.outcome().reason, R_DECLARATION);
}

static void test_max_ply() {
  Position p;
  CHECK(p.set_sfen("k8/9/9/9/9/9/9/9/K8 b - 1", PHASE_NORMAL));
  p.set_max_ply(4, true);
  const char* mv[] = {"9i8i", "9a8a", "8i7i", "8a7a"};
  for (const char* m : mv) {
    CHECK_EQ(p.outcome().result, ONGOING);
    p.do_move(move_from_usi(m));
  }
  CHECK_EQ(p.outcome().result, DRAW);
  CHECK_EQ(p.outcome().reason, R_MAX_PLY);
  CHECK_EQ(legal_set(p).size(), 0u);
  p.undo_move();
  CHECK_EQ(p.outcome().result, ONGOING);
  // 通算で数える設定: 41 手目を 41 として、44 で上限
  Position q;
  CHECK(q.set_sfen("k8/9/9/9/9/9/9/9/K8 b - 41", PHASE_NORMAL));  // 40 手完了の局面（通算 41 手目）
  CHECK_EQ(q.ply(), 40);
  CHECK_EQ(q.normal_ply(), 0);
  q.set_max_ply(44, false);
  for (const char* m : mv) q.do_move(move_from_usi(m));
  CHECK_EQ(q.outcome().reason, R_MAX_PLY);
  // 上限の手で詰めば詰みが優先
  Position r;
  CHECK(r.set_sfen("k8/9/9/9/9/9/9/9/K8 b RRG 1", PHASE_NORMAL));
  r.set_max_ply(3, true);
  r.do_move(move_from_usi("R*8c"));
  r.do_move(move_from_usi("9a9b"));
  r.do_move(move_from_usi("R*9c"));  // 9b の玉: 8c の飛と 9c の飛 → 詰み（8a 8b 8c は飛の利き、9a は縦）
  CHECK_EQ(r.outcome().result, BLACK_WIN);
  CHECK_EQ(r.outcome().reason, R_NO_LEGAL_MOVE);
}

static void test_harness_results() {
  Position p;
  CHECK(p.set_position("position fuseki moves K*5i K*5a"));
  p.resign(BLACK);
  CHECK_EQ(p.outcome().result, WHITE_WIN);
  CHECK_EQ(p.outcome().reason, R_RESIGN);
  CHECK_EQ(legal_set(p).size(), 0u);
  Position q;
  CHECK(q.set_position("position startpos"));
  q.illegal_move(WHITE);
  CHECK_EQ(q.outcome().result, BLACK_WIN);
  CHECK_EQ(q.outcome().reason, R_ILLEGAL_MOVE);
  Position r;
  CHECK(r.set_position("position startpos"));
  r.timeout(BLACK);
  CHECK_EQ(r.outcome().reason, R_TIMEOUT);
}

static void test_full_game() {
  // 布石 40 手を実際に打ち、本将棋へ移る
  Position p;
  CHECK(p.set_position("position fuseki"));
  for (int i = 0; i < 40; ++i) {
    MoveList ml;
    p.legal_moves(ml);
    CHECK(ml.n > 0);
    p.do_move(ml.m[i % ml.n]);
  }
  CHECK_EQ(p.ply(), 40);
  CHECK_EQ(p.phase(), PHASE_NORMAL);
  CHECK_EQ(p.hand(BLACK, PAWN), 0);
  CHECK_EQ(p.hand(WHITE, KING), 0);
  Outcome o = p.outcome();
  CHECK(o.result == ONGOING || o.reason == R_RULING41 || o.reason == R_NO_LEGAL_MOVE);
  for (int i = 0; i < 40; ++i) p.undo_move();
  CHECK_EQ(p.ply(), 0);
  CHECK_EQ(p.sfen(), "9/9/9/9/9/9/9/9/9 b KRB2G2S2N2L9Pkrb2g2s2n2l9p 1");
}

int main() {
  test_usi_sfen();
  test_zobrist();
  test_fuseki_drops();
  test_nihikyo();
  test_nihikyo_move40();
  test_move40_restriction();
  test_ruling41();
  test_mate_at_41();
  test_normal_rules();
  test_repetition();
  test_declaration();
  test_max_ply();
  test_harness_results();
  test_full_game();
  std::printf("%d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
