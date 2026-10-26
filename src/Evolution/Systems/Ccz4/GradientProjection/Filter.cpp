// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/Filter.hpp"

#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceConstrainedEvolution.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::GradientProjection {
void EnforceAlgebraicConstraints::apply(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde) {
  ::Ccz4::fd::enforce_algebraic_constraints(conformal_metric, a_tilde);
}

void filter_velocities(
    const gsl::not_null<Tags::TransformedATilde::type*> a,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde,
    const gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
    const gsl::not_null<Scalar<DataVector>*> theta,
    const gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat,
    const gsl::not_null<tnsr::I<DataVector, 3>*> b, const PotentialFields& u,
    const std::vector<LogMetricChart>& charts,
    const Tags::PotentialGradients::type& q, const Mesh<3>& mesh,
    const LocalOperators& local_operators, const double gain, const double f) {
  // On a shell the filter is never the identity: at unit gain it still
  // removes the angular content outside the retained space.
  if (gain == 1.0 and not local_operators.is_spherical_shell()) {
    return;
  }
  local_operators.check_mesh(mesh);
  const size_t number_of_points = mesh.number_of_grid_points();
  ASSERT(charts.size() == number_of_points,
         "Expected one chart per grid point, got " << charts.size() << " for "
                                                   << number_of_points);
  const VelocityFields w{*a, *trace_extrinsic_curvature, *theta, *gamma_hat,
                         *b};
  // 1. The potential rates R(u,w,q) and GammaHat - b (filtered below), and
  // the gradient part R(u,0,q) (kept for the recovery).
  Tags::PotentialRateTarget::type rates{number_of_points};
  Tags::PotentialRateTarget::type gradient_rates{number_of_points};
  tnsr::I<DataVector, 3> z{number_of_points};
  for (size_t s = 0; s < number_of_points; ++s) {
    const PotentialVector u_s = potentials_at(u, s);
    const VelocityVector w_s = velocities_at(w, s);
    const GradientVector q_s = gradients_at(q, s);
    const ShiftGradientTerms shift_terms{charts[s], q_s};
    set_potentials_at(make_not_null(&rates), s,
                      potential_rhs(charts[s], shift_terms, u_s, w_s, q_s, f));
    set_potentials_at(
        make_not_null(&gradient_rates), s,
        potential_rhs(charts[s], shift_terms, u_s, VelocityVector{}, q_s, f));
    for (size_t i = 0; i < 3; ++i) {
      z.get(i)[s] = gsl::at(w_s, VelocityIndex::gamma_hat + i) -
                    gsl::at(w_s, VelocityIndex::b + i);
    }
  }
  if (local_operators.is_spherical_shell()) {
    local_operators.apply_shell_filter(
        {{&rates.get(0), &rates.get(1), &rates.get(2), &rates.get(3),
          &rates.get(4), &rates.get(5), &rates.get(6), &rates.get(7),
          &rates.get(8), &rates.get(9), &z.get(0), &z.get(1), &z.get(2)}},
        gain);
  } else {
    std::vector<gsl::not_null<DataVector*>> fields{};
    fields.reserve(number_of_potentials + 3);
    for (auto& component : rates) {
      fields.emplace_back(&component);
    }
    for (auto& component : z) {
      fields.emplace_back(&component);
    }
    local_operators.apply_filter(fields, gain);
  }
  // 2. The recovery of w+ and the physical ATilde at fixed H.
  for (size_t s = 0; s < number_of_points; ++s) {
    PotentialVector filtered_rate_minus_gradient_rate{};
    for (size_t c = 0; c < number_of_potentials; ++c) {
      gsl::at(filtered_rate_minus_gradient_rate, c) =
          rates.get(c)[s] - gradient_rates.get(c)[s];
    }
    VelocityVector w_s =
        apply_rhs_inverse(get(u.lapse)[s], get(u.conformal_factor)[s],
                          filtered_rate_minus_gradient_rate, f);
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(w_s, VelocityIndex::gamma_hat + i) += z.get(i)[s];
    }
    for (size_t c = 0; c < 5; ++c) {
      a->get(c)[s] = gsl::at(w_s, VelocityIndex::a + c);
    }
    get(*trace_extrinsic_curvature)[s] =
        w_s[VelocityIndex::trace_extrinsic_curvature];
    get(*theta)[s] = w_s[VelocityIndex::theta];
    for (size_t i = 0; i < 3; ++i) {
      gamma_hat->get(i)[s] = gsl::at(w_s, VelocityIndex::gamma_hat + i);
      b->get(i)[s] = gsl::at(w_s, VelocityIndex::b + i);
    }
    const Matrix3 physical_a_tilde =
        0.5 * charts[s].tangent(stf_decode(&w_s[VelocityIndex::a]));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        a_tilde->get(i, j)[s] = physical_a_tilde(i, j);
      }
    }
  }
}
}  // namespace Ccz4::GradientProjection
