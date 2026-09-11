// SPDX-License-Identifier: Apache-2.0
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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
  return c;
}

py::dict record_to_dict(const GameRecord& r) {
  py::dict d;
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
  m.doc() = "libra-search: 自己対局エンジン（MCGS/MCTS、Gumbel ルート、バッチ評価）";
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
        return d;
      });
}
