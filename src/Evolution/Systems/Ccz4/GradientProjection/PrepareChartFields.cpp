// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection {
void potential_gradients(
    const gsl::not_null<Tags::ChartQ::type*> gradients,
    const Tags::ChartU::type& potentials, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian) {
  using variables_tags = tmpl::list<Tags::ChartU>;
  Variables<variables_tags> variables{mesh.number_of_grid_points()};
  get<Tags::ChartU>(variables) = potentials;
  const auto derivatives =
      partial_derivatives<variables_tags>(variables, mesh, inverse_jacobian);
  *gradients =
      get<::Tags::deriv<Tags::ChartU, tmpl::size_t<3>, Frame::Inertial>>(
          derivatives);
}

void PrepareChartFields::apply(
    const gsl::not_null<Tags::ChartU::type*> u,
    const gsl::not_null<Tags::ChartW::type*> w,
    const gsl::not_null<Tags::ChartQ::type*> q,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, 3>& a_tilde,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& theta, const tnsr::I<DataVector, 3>& gamma_hat,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& b, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian) {
  const size_t number_of_points = mesh.number_of_grid_points();
  for (size_t s = 0; s < number_of_points; ++s) {
    PhysicalPointState state{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        state.conformal_metric(i, j) = conformal_metric.get(i, j)[s];
        state.a_tilde(i, j) = a_tilde.get(i, j)[s];
      }
      gsl::at(state.gamma_hat, i) = gamma_hat.get(i)[s];
      gsl::at(state.shift, i) = shift.get(i)[s];
      gsl::at(state.b, i) = b.get(i)[s];
    }
    state.conformal_factor = get(conformal_factor)[s];
    state.trace_extrinsic_curvature = get(trace_extrinsic_curvature)[s];
    state.theta = get(theta)[s];
    state.lapse = get(lapse)[s];
    PotentialVector u_s{};
    VelocityVector w_s{};
    const std::array<double, 2> traces =
        chart_from_physical(make_not_null(&u_s), make_not_null(&w_s), state);
    double a_norm = 0.0;
    for (size_t c = 0; c < 5; ++c) {
      a_norm += square(gsl::at(w_s, VelocityIndex::a + c));
    }
    if (std::abs(traces[0]) > chart_trace_tolerance or
        std::abs(traces[1]) >
            chart_trace_tolerance * (1.0 + std::sqrt(a_norm))) {
      ERROR(
          "The physical CCZ4 state is off the algebraic-constraint surface: "
          "log(det(conformal metric))/2 = "
          << traces[0] << ", trace of ATilde = " << traces[1] << " (tolerance "
          << chart_trace_tolerance
          << "). The state must be projected before preparing the chart.");
    }
    set_potentials_at(u, s, u_s);
    set_velocities_at(w, s, w_s);
  }
  potential_gradients(q, *u, mesh, inverse_jacobian);
}
}  // namespace Ccz4::GradientProjection
