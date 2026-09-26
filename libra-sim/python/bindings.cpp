// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "libra/encoding.h"
#include "libra/position.h"

namespace py = pybind11;
using namespace libra;

namespace {

Color color_arg(const std::string& s) {
  if (s == "sente" || s == "b" || s == "black") return BLACK;
  if (s == "gote" || s == "w" || s == "white") return WHITE;
  throw py::value_error("color must be 'sente' or 'gote'");
}
std::string color_name(Color c) { return c == BLACK ? "sente" : "gote"; }
Mode mode_arg(const std::string& s) {
  if (s == "tenbin") return MODE_TENBIN;
  if (s == "fuseki") return MODE_FUSEKI;
  throw py::value_error("mode must be 'tenbin' or 'fuseki'");
}
Phase phase_arg(const std::string& s) {
  if (s == "fuseki") return PHASE_FUSEKI;
  if (s == "normal") return PHASE_NORMAL;
  throw py::value_error("phase must be 'fuseki' or 'normal'");
}

}  // namespace

PYBIND11_MODULE(_sim, m) {
  m.doc() = "libra-sim: 天秤将棋の厳密シミュレータ";
  py::class_<Position>(m, "Position")
      .def(py::init([](const std::string& mode) {
             auto p = std::make_unique<Position>();
             p->reset(mode_arg(mode));
             return p;
           }),
           py::arg("mode") = "tenbin")
      .def("reset", [](Position& p, const std::string& mode) { p.reset(mode_arg(mode)); }, py::arg("mode") = "tenbin")
      .def("set_sfen",
           [](Position& p, const std::string& sfen, const std::string& phase) {
             if (!p.set_sfen(sfen, phase_arg(phase))) throw py::value_error("bad sfen: " + sfen);
           },
           py::arg("sfen"), py::arg("phase"))
      .def("set_position",
           [](Position& p, const std::string& line, const std::string& mode) {
             if (!p.set_position(line, mode_arg(mode))) throw py::value_error("bad position: " + line);
           },
           py::arg("line"), py::arg("mode") = "tenbin")
      .def("sfen", &Position::sfen)
      .def("pretty", &Position::pretty)
      .def_property_readonly("mode", [](const Position& p) { return p.mode() == MODE_TENBIN ? "tenbin" : "fuseki"; })
      .def_property_readonly("phase", [](const Position& p) { return p.phase() == PHASE_FUSEKI ? "fuseki" : "normal"; })
      .def_property_readonly("turn", [](const Position& p) { return color_name(p.turn()); })
      .def_property_readonly("ply", &Position::ply)
      .def_property_readonly("normal_ply", &Position::normal_ply)
      .def_property_readonly("key", &Position::key)
      .def_property_readonly("mirror_key", &Position::mirror_key)
      .def_property_readonly("norm_key", &Position::norm_key)
      .def("piece_on", [](const Position& p, int sq) { return int(p.piece_on(sq)); }, py::arg("sq"))
      .def("hand", [](const Position& p, const std::string& c, const std::string& pt) {
             PieceType t = pt_from_char(pt.empty() ? '?' : pt[0]);
             if (t == NO_PT) throw py::value_error("bad piece type");
             return p.hand(color_arg(c), t);
           })
      .def("king_sq", [](const Position& p, const std::string& c) { int k = p.king_sq(color_arg(c)); return k == SQ_NONE ? -1 : k; })
      .def("legal_moves",
           [](Position& p) {
             MoveList ml;
             p.legal_moves(ml);
             std::vector<std::string> out;
             out.reserve(ml.n);
             for (Move x : ml) out.push_back(move_to_usi(x));
             return out;
           })
      .def("legal_move_codes",
           [](Position& p) {
             MoveList ml;
             p.legal_moves(ml);
             return std::vector<std::uint32_t>(ml.begin(), ml.end());
           })
      .def("is_legal", [](Position& p, const std::string& usi) { Move m = move_from_usi(usi); return m != MOVE_NONE && p.is_legal(m); })
      .def("do_move",
           [](Position& p, const std::string& usi) {
             Move m = move_from_usi(usi);
             if (m == MOVE_NONE || !p.is_legal(m)) throw py::value_error("illegal move: " + usi);
             p.do_move(m);
           },
           py::arg("usi"))
      .def("do_move_code", [](Position& p, std::uint32_t m) { p.do_move(Move(m)); })
      .def("undo", &Position::undo_move)
      .def("last_move", [](const Position& p) { return move_to_usi(p.last_move()); })
      .def("in_check", &Position::in_check)
      .def("king_attacked", [](const Position& p, const std::string& c) { return p.king_attacked(color_arg(c)); })
      .def("ruling41_pending", &Position::ruling41_pending)
      .def("outcome",
           [](Position& p) {
             Outcome o = p.outcome();
             return py::make_tuple(std::string(result_name(o.result)), std::string(reason_name(o.reason)));
           })
      .def("is_over", [](Position& p) { return p.outcome().result != ONGOING; })
      .def("can_declare", [](const Position& p, const std::string& c) { return p.can_declare(color_arg(c)); })
      .def("declaration_points", [](const Position& p, const std::string& c) { return p.declaration_points(color_arg(c)); })
      .def("declaration_pieces", [](const Position& p, const std::string& c) { return p.declaration_pieces(color_arg(c)); })
      .def("declare", [](Position& p, const std::string& c) { p.declare(color_arg(c)); })
      .def("resign", [](Position& p, const std::string& c) { p.resign(color_arg(c)); })
      .def("timeout", [](Position& p, const std::string& c) { p.timeout(color_arg(c)); })
      .def("illegal_move", [](Position& p, const std::string& c) { p.illegal_move(color_arg(c)); })
      .def("repetition_count", &Position::repetition_count)
      .def("set_max_ply", &Position::set_max_ply, py::arg("n"), py::arg("count_from_41") = true)
      .def_property_readonly("max_ply", &Position::max_ply)
      .def("perft", &Position::perft, py::arg("depth"))
      .def("move_index", [](const Position& p, const std::string& usi) {
             Move m = move_from_usi(usi);
             if (m == MOVE_NONE) throw py::value_error("bad move: " + usi);
             return move_index(p, m);
           })
      .def("move_from_index", [](const Position& p, int idx) { return move_to_usi(move_from_index(p, idx)); })
      .def("features", [](const Position& p, bool mirror) {
             py::array_t<float> sq({SQ_NB, SQ_FEATS});
             py::array_t<float> glob({GLOB_FEATS});
             write_features(p, sq.mutable_data(), glob.mutable_data(), mirror);
             return py::make_tuple(sq, glob);
           }, py::arg("mirror") = false);
  // 学習用: 玉配置と手順から ply 手目の局面を再生し、特徴をバッチの行に書く。
  // kb/kw: 玉のマス、moves: uint32 の手、plies: 取り出す手数（玉 2 手を含む通算）、mirror: 行ごとの鏡映。
  // sq_out [N,81,SQ_FEATS]、glob_out [N,GLOB_FEATS]、side_out [N]（手番が先手なら 1）、fuseki_out [N]
  m.def("replay_features",
        [](py::array_t<std::int32_t> kb, py::array_t<std::int32_t> kw, py::list moves_list, py::array_t<std::int32_t> plies,
           py::array_t<std::uint8_t> mirror, py::array_t<float, py::array::c_style> sq_out,
           py::array_t<float, py::array::c_style> glob_out, py::array_t<std::uint8_t> side_out,
           py::array_t<std::uint8_t> fuseki_out, int max_ply, bool count_from_41) {
          int n = int(plies.shape(0));
          std::vector<py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>> mv;
          mv.reserve(n);
          for (int i = 0; i < n; ++i) mv.push_back(moves_list[i].cast<py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>>());
          auto kb_ = kb.unchecked<1>();
          auto kw_ = kw.unchecked<1>();
          auto pl_ = plies.unchecked<1>();
          auto mi_ = mirror.unchecked<1>();
          auto side_ = side_out.mutable_unchecked<1>();
          auto fu_ = fuseki_out.mutable_unchecked<1>();
          float* sq = sq_out.mutable_data();
          float* gl = glob_out.mutable_data();
          std::vector<const std::uint32_t*> ptrs(n);
          std::vector<int> lens(n);
          for (int i = 0; i < n; ++i) {
            ptrs[i] = mv[i].data();
            lens[i] = int(mv[i].shape(0));
          }
          py::gil_scoped_release nogil;
          Position pos;
          for (int i = 0; i < n; ++i) {
            pos.reset(MODE_TENBIN);
            pos.set_max_ply(max_ply, count_from_41);
            pos.do_move(make_drop(KING, kb_(i)));
            pos.do_move(make_drop(KING, kw_(i)));
            int target = pl_(i);
            for (int k = 0; k + 2 < target && k < lens[i]; ++k) pos.do_move(Move(ptrs[i][k]));
            write_features(pos, sq + size_t(i) * SQ_NB * SQ_FEATS, gl + size_t(i) * GLOB_FEATS, mi_(i) != 0);
            side_(i) = pos.turn() == BLACK ? 1 : 0;
            fu_(i) = pos.phase() == PHASE_FUSEKI ? 1 : 0;
          }
        });
  // 学習用（補助の「駒が最後まで残るか」、KataGo [Wu19] §4.1 の陣地の予測の将棋版）: ply 手目の局面の盤上の駒（玉を除く）が、
  // 記録の最後の手まで取られずに盤に残るか。surv_out [N,81] を特徴と同じマスの並び（手番側から見た向き・鏡映）で書く:
  // 1 = 残る、0 = 途中で取られる、-1 = 空きか玉（学習に使わない）。成っても同じ駒として数える。
  m.def("replay_survival",
        [](py::array_t<std::int32_t> kb, py::array_t<std::int32_t> kw, py::list moves_list, py::array_t<std::int32_t> plies,
           py::array_t<std::uint8_t> mirror, py::array_t<float, py::array::c_style> surv_out, int max_ply, bool count_from_41) {
          int n = int(plies.shape(0));
          std::vector<py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>> mv;
          mv.reserve(n);
          for (int i = 0; i < n; ++i) mv.push_back(moves_list[i].cast<py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>>());
          auto kb_ = kb.unchecked<1>();
          auto kw_ = kw.unchecked<1>();
          auto pl_ = plies.unchecked<1>();
          auto mi_ = mirror.unchecked<1>();
          float* out = surv_out.mutable_data();
          std::vector<const std::uint32_t*> ptrs(n);
          std::vector<int> lens(n);
          for (int i = 0; i < n; ++i) {
            ptrs[i] = mv[i].data();
            lens[i] = int(mv[i].shape(0));
          }
          py::gil_scoped_release nogil;
          Position pos;
          for (int i = 0; i < n; ++i) {
            pos.reset(MODE_TENBIN);
            pos.set_max_ply(max_ply, count_from_41);
            pos.do_move(make_drop(KING, kb_(i)));
            pos.do_move(make_drop(KING, kw_(i)));
            int target = pl_(i);
            int k = 0;
            for (; k + 2 < target && k < lens[i]; ++k) pos.do_move(Move(ptrs[i][k]));
            float* row = out + size_t(i) * SQ_NB;
            Color us = pos.turn();
            bool mirror = mi_(i) != 0;
            int tag[SQ_NB];  // マス → いま置いてある、追いかけている駒の元のマス（無ければ -1）
            for (int sq = 0; sq < SQ_NB; ++sq) {
              Piece p = pos.piece_on(sq);
              bool track = p != NO_PIECE && type_of(p) != KING;
              tag[sq] = track ? sq : -1;
              int t = to_mover_frame(us, sq);
              row[mirror ? mirror_sq(t) : t] = track ? 1.0f : -1.0f;
            }
            for (; k < lens[i]; ++k) {  // 手を盤の上だけで追う（取る手・動かす手・打つ手）。局面は進めなくてよい
              Move m = Move(ptrs[i][k]);
              int to = to_sq(m);
              if (tag[to] >= 0) {
                int t = to_mover_frame(us, tag[to]);
                row[mirror ? mirror_sq(t) : t] = 0.0f;
              }
              if (is_drop(m)) {
                tag[to] = -1;
              } else {
                tag[to] = tag[from_sq(m)];
                tag[from_sq(m)] = -1;
              }
            }
          }
        });
  m.def("mirror_index", &mirror_index);
  m.attr("POLICY_SIZE") = POLICY_SIZE;
  m.attr("POLICY_CLASSES") = POLICY_CLASSES;
  m.attr("SQ_FEATS") = SQ_FEATS;
  m.attr("GLOB_FEATS") = GLOB_FEATS;
  m.def("move_to_usi", [](std::uint32_t m) { return move_to_usi(Move(m)); });
  m.def("move_from_usi", [](const std::string& s) { return std::uint32_t(move_from_usi(s)); });
  m.def("sq_from_usi", &sq_from_usi);
  m.def("sq_to_usi", &sq_to_usi);
  m.def("mirror_sq", &mirror_sq);
  m.attr("__version__") = "0.0.1";
}
