// SPDX-License-Identifier: Apache-2.0
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "libra/dfpn.h"
#include "libra/selfplay.h"

namespace py = pybind11;
using namespace libra;

namespace {

SearchConfig config_from_dict(const py::dict& d) {
  SearchConfig c;
  auto geti = [&](const char* k, int& v) { if (d.contains(k)) v = d[k].cast<int>(); };
  auto getf = [&](const char* k, float& v) { if (d.contains(k)) v = d[k].cast<float>(); };
  auto getb = [&](const char* k, bool& v) { if (d.contains(k)) v = d[k].cast<bool>(); };
  geti("full_sims", c.full_sims);
  geti("fast_sims", c.fast_sims);
  getf("full_prob", c.full_prob);
  geti("gumbel_m_full", c.gumbel_m_full);
  geti("gumbel_m_fast", c.gumbel_m_fast);
  getf("c_visit", c.c_visit);
  getf("c_scale", c.c_scale);
  getf("cpuct", c.cpuct);
  getf("draw_value", c.draw_value);
  geti("max_ply", c.max_ply);
  getb("count_from_41", c.count_from_41);
  geti("policy_topk", c.policy_topk);
  geti("max_moves_per_game", c.max_moves_per_game);
  if (d.contains("king_pairs")) {
    c.king_pairs.clear();
    for (auto item : d["king_pairs"].cast<py::list>()) {
      auto pr = item.cast<py::sequence>();
      c.king_pairs.push_back(pr[0].cast<int>());
      c.king_pairs.push_back(pr[1].cast<int>());
    }
  }
  geti("mate_nodes_root", c.mate_nodes_root);
  geti("proof_nodes", c.proof_nodes);
  geti("proof_min_ply", c.proof_min_ply);
  getb("external", c.external);
  return c;
}

py::dict record_to_dict(const GameRecord& r) {
  py::dict d;
  d["slot"] = r.slot;
  d["kb"] = r.kb;
  d["kw"] = r.kw;
  d["result"] = r.result;
  d["reason"] = r.reason;
  d["v41"] = r.v41;
  d["plies"] = r.plies;
  d["sfen41"] = r.sfen41;
  size_t n = r.moves.size();
  py::array_t<std::uint32_t> moves(n);
  py::array_t<std::uint8_t> full(n);
  py::array_t<float> root_q(n);
  std::vector<std::int32_t> off(n + 1, 0);
  size_t total = 0;
  for (size_t i = 0; i < n; ++i) {
    moves.mutable_at(i) = r.moves[i].move;
    full.mutable_at(i) = r.moves[i].full ? 1 : 0;
    root_q.mutable_at(i) = r.moves[i].root_q;
    total += r.moves[i].policy.size();
    off[i + 1] = std::int32_t(total);
  }
  py::array_t<std::int16_t> pidx(total);
  py::array_t<float> pp(total);
  size_t k = 0;
  for (const MoveRecord& mr : r.moves)
    for (auto& [idx, p] : mr.policy) {
      pidx.mutable_at(k) = std::int16_t(idx);
      pp.mutable_at(k) = p;
      ++k;
    }
  py::array_t<std::int32_t> poff(n + 1);
  for (size_t i = 0; i <= n; ++i) poff.mutable_at(i) = off[i];
  d["moves"] = moves;
  d["full"] = full;
  d["root_q"] = root_q;
  d["policy_idx"] = pidx;
  d["policy_p"] = pp;
  d["policy_off"] = poff;
  return d;
}

}  // namespace

PYBIND11_MODULE(_search, m) {
  m.doc() = "libra-search: 自己対局エンジン（MCGS/MCTS、Gumbel ルート、バッチ評価）と df-pn";
  // libra_sim は _sim と _search の両方に静的リンクされ、利きと Zobrist の表がモジュールごとにある。
  // _sim 側で作った Position をこちらの関数に渡しても正しく動くよう、こちらの表をここで初期化する
  static Position init_tables;
  (void)init_tables;
  // df-pn を局面に対して直接呼ぶ（テスト・USI エンジン用）。problem: "mate" | "ruling41" | "mate41"。
  // 戻り値: (result: "proven"|"disproven"|"unknown", best_move_usi, nodes)
  m.def("solve",
        [](Position& pos, const std::string& problem, std::uint64_t max_nodes, int tt_bits) {
          DfPn d(tt_bits);
          Move best = MOVE_NONE;
          ProofResult r;
          bool or_node;
          if (problem == "mate") {
            MateProblem p;
            r = d.solve(pos, p, true, max_nodes, &best);
          } else if (problem == "ruling41") {
            Ruling41Problem p;
            or_node = pos.turn() == BLACK;
            r = d.solve(pos, p, or_node, max_nodes, &best);
          } else if (problem == "mate41") {
            Mate41Problem p;
            or_node = pos.turn() == WHITE;
            r = d.solve(pos, p, or_node, max_nodes, &best);
          } else {
            throw py::value_error("problem must be mate | ruling41 | mate41");
          }
          const char* name = r == PROOF_PROVEN ? "proven" : r == PROOF_DISPROVEN ? "disproven" : "unknown";
          return py::make_tuple(std::string(name), move_to_usi(best), d.nodes());
        },
        py::arg("pos"), py::arg("problem"), py::arg("max_nodes") = 10000, py::arg("tt_bits") = 18);
  py::class_<SelfPlay>(m, "SelfPlay")
      .def(py::init([](const py::dict& cfg, int n_games, std::uint64_t seed, int threads) {
             return std::make_unique<SelfPlay>(config_from_dict(cfg), n_games, seed, threads);
           }),
           py::arg("config"), py::arg("n_games"), py::arg("seed") = 0, py::arg("threads") = 1)
      .def_property_readonly("n_games", &SelfPlay::n_games)
      .def("collect",
           [](SelfPlay& s, py::array_t<float, py::array::c_style> sq, py::array_t<float, py::array::c_style> glob) {
             if (sq.ndim() != 3 || sq.shape(0) != s.n_games() || sq.shape(1) != SQ_NB || sq.shape(2) != SQ_FEATS)
               throw py::value_error("sq must be [n_games, 81, SQ_FEATS] float32");
             if (glob.ndim() != 2 || glob.shape(0) != s.n_games() || glob.shape(1) != GLOB_FEATS)
               throw py::value_error("glob must be [n_games, GLOB_FEATS] float32");
             int n;
             {
               py::gil_scoped_release nogil;
               n = s.collect(sq.mutable_data(), glob.mutable_data());
             }
             return n;
           },
           py::arg("sq"), py::arg("glob"))
      .def("apply",
           [](SelfPlay& s, py::array_t<float, py::array::c_style | py::array::forcecast> logits,
              py::array_t<float, py::array::c_style | py::array::forcecast> wdl) {
             if (logits.ndim() != 2 || logits.shape(0) != s.n_games() || logits.shape(1) != POLICY_SIZE)
               throw py::value_error("logits must be [n_games, POLICY_SIZE]");
             if (wdl.ndim() != 2 || wdl.shape(0) != s.n_games() || wdl.shape(1) != 3)
               throw py::value_error("wdl must be [n_games, 3]");
             py::gil_scoped_release nogil;
             s.apply(logits.data(), wdl.data());
           },
           py::arg("logits"), py::arg("wdl"))
      .def("take_finished",
           [](SelfPlay& s) {
             py::list out;
             for (const GameRecord& r : s.take_finished()) out.append(record_to_dict(r));
             return out;
           })
      .def("set_active", &SelfPlay::set_active)
      .def("set_position", [](SelfPlay& s, int slot, const std::string& line, int sims, bool full, const std::string& mode) { return s.set_position(slot, line, sims, full, mode == "fuseki" ? MODE_FUSEKI : MODE_TENBIN); }, py::arg("slot"), py::arg("usi_line"), py::arg("sims"), py::arg("full") = true, py::arg("mode") = "tenbin")
      .def("idle", &SelfPlay::idle, py::arg("slot"))
      .def("finish_now", &SelfPlay::finish_now, py::arg("slot"))
      .def("result",
           [](const SelfPlay& s, int slot) {
             const SearchResult& r = s.result(slot);
             py::dict d;
             d["ready"] = r.ready;
             d["best"] = move_to_usi(Move(r.best));
             d["root_q"] = r.root_q;
             d["sims"] = r.sims;
             py::list cands;
             for (const Candidate& c : r.cands) {
               py::dict cd;
               cd["move"] = move_to_usi(Move(c.move));
               cd["visits"] = c.visits;
               cd["q"] = c.q;
               cd["prior"] = c.prior;
               cands.append(cd);
             }
             d["cands"] = cands;
             py::list pv;
             for (std::uint32_t m : r.pv) pv.append(move_to_usi(Move(m)));
             d["pv"] = pv;
             return d;
           },
           py::arg("slot"))
      .def("root_turns",
           [](const SelfPlay& s) {
             py::array_t<std::int8_t> out(s.n_games());
             s.root_turns(out.mutable_data());
             return out;
           })
      .def_property_readonly("active", &SelfPlay::active)
      .def("stats", [](const SelfPlay& s) {
        SelfPlayStats st = s.stats();
        py::dict d;
        d["games"] = st.games;
        d["moves"] = st.moves;
        d["sims"] = st.sims;
        d["evals"] = st.evals;
        d["sente_wins"] = st.results[0];
        d["draws"] = st.results[1];
        d["gote_wins"] = st.results[2];
        d["ruling41"] = st.ruling41;
        d["no_legal_move"] = st.no_legal;
        d["sennichite"] = st.sennichite;
        d["perpetual_check"] = st.perpetual;
        d["max_ply"] = st.max_ply;
        d["timeout"] = st.timeout;
        d["plies_sum"] = st.plies_sum;
        d["mate_found"] = st.mate_found;
        d["proof_found"] = st.proof_found;
        d["proof_calls"] = st.proof_calls;
        d["proof_nodes"] = st.proof_nodes;
        return d;
      });
}
