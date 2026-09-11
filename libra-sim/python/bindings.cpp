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
      .def("features", [](const Position& p) {
             py::array_t<float> sq({SQ_NB, SQ_FEATS});
             py::array_t<float> glob({GLOB_FEATS});
             write_features(p, sq.mutable_data(), glob.mutable_data());
             return py::make_tuple(sq, glob);
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
