// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection {
void checked_chart_from_physical(const gsl::not_null<PotentialVector*> u,
                                 const gsl::not_null<VelocityVector*> w,
                                 const gsl::not_null<LogMetricChart*> chart,
                                 const PhysicalPointState& state,
                                 const char* const context) {
  const std::array<double, 2> traces = chart_from_physical(u, w, chart, state);
  double a_norm = 0.0;
  for (size_t c = 0; c < 5; ++c) {
    a_norm += square(gsl::at(*w, VelocityIndex::a + c));
  }
  if (std::abs(traces[0]) > chart_trace_tolerance or
      std::abs(traces[1]) > chart_trace_tolerance * (1.0 + std::sqrt(a_norm))) {
    ERROR("The physical CCZ4 state ("
          << context
          << ") is off the algebraic-constraint surface: "
             "log(det(conformal metric))/2 = "
          << traces[0] << ", trace of ATilde = " << traces[1] << " (tolerance "
          << chart_trace_tolerance
          << "). The state must be projected before constructing the chart.");
  }
}

void local_chart_fields(
    const gsl::not_null<LocalChartFields*> chart,
    const gsl::not_null<std::vector<LogMetricChart>*> charts,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, 3>& a_tilde, const Scalar<DataVector>& lapse) {
  const size_t number_of_points = get(lapse).size();
  chart->initialize(number_of_points);
  charts->resize(number_of_points);
  auto& log_metric = get<Tags::LogConformalMetric>(*chart);
  auto& a = get<Tags::TransformedATilde>(*chart);
  for (size_t s = 0; s < number_of_points; ++s) {
    PhysicalPointState state{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        state.conformal_metric(i, j) = conformal_metric.get(i, j)[s];
        state.a_tilde(i, j) = a_tilde.get(i, j)[s];
      }
    }
    state.conformal_factor = get(conformal_factor)[s];
    state.lapse = get(lapse)[s];
    PotentialVector u_s{};
    VelocityVector w_s{};
    checked_chart_from_physical(make_not_null(&u_s), make_not_null(&w_s),
                                make_not_null(&(*charts)[s]), state, "volume");
    for (size_t c = 0; c < 5; ++c) {
      log_metric.get(c)[s] = gsl::at(u_s, PotentialIndex::log_metric + c);
      a.get(c)[s] = gsl::at(w_s, VelocityIndex::a + c);
    }
  }
}

std::vector<LogMetricChart> log_metric_charts(const PotentialFields& u) {
  const size_t number_of_points = get(u.lapse).size();
  std::vector<LogMetricChart> charts{};
  charts.reserve(number_of_points);
  for (size_t s = 0; s < number_of_points; ++s) {
    const PotentialVector u_s = potentials_at(u, s);
    charts.emplace_back(stf_decode(&u_s[PotentialIndex::log_metric]));
  }
  return charts;
}

void assemble_potential_gradients(
    const gsl::not_null<Tags::PotentialGradients::type*> gradients,
    const TensorMetafunctions::prepend_spatial_index<
        Tags::LogConformalMetric::type, 3, UpLo::Lo, Frame::Inertial>&
        d_log_conformal_metric,
    const tnsr::i<DataVector, 3>& d_conformal_factor,
    const tnsr::i<DataVector, 3>& d_lapse,
    const tnsr::iJ<DataVector, 3>& d_shift) {
  for (size_t i = 0; i < 3; ++i) {
    for (size_t c = 0; c < 5; ++c) {
      gradients->get(i, PotentialIndex::log_metric + c) =
          d_log_conformal_metric.get(i, c);
    }
    gradients->get(i, PotentialIndex::conformal_factor) =
        d_conformal_factor.get(i);
    gradients->get(i, PotentialIndex::lapse) = d_lapse.get(i);
    for (size_t k = 0; k < 3; ++k) {
      gradients->get(i, PotentialIndex::shift + k) = d_shift.get(i, k);
    }
  }
}

void potential_gradients(
    const gsl::not_null<Tags::PotentialGradients::type*> gradients,
    const Tags::PotentialRateTarget::type& potentials, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian) {
  using variables_tags = tmpl::list<Tags::PotentialRateTarget>;
  Variables<variables_tags> variables{mesh.number_of_grid_points()};
  get<Tags::PotentialRateTarget>(variables) = potentials;
  const auto derivatives =
      partial_derivatives<variables_tags>(variables, mesh, inverse_jacobian);
  *gradients = get<::Tags::deriv<Tags::PotentialRateTarget, tmpl::size_t<3>,
                                 Frame::Inertial>>(derivatives);
}
}  // namespace Ccz4::GradientProjection
