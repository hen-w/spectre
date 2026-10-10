// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/FinalizeTimeDerivative.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::GradientProjection {
void finalized_chart_rates(
    const gsl::not_null<Tags::ChartU::type*> u_rate,
    const gsl::not_null<Tags::ChartW::type*> w_rate,
    const gsl::not_null<Tags::ChartQ::type*> fitted_gradient,
    const Tags::ChartU::type& u, const Tags::ChartW::type& w,
    const Tags::ChartQ::type& q, const Tags::ChartU::type& dt_u_target,
    const Tags::ChartW::type& dt_w_target,
    const Tags::ChartQ::type& dt_q_target, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const LocalOperators& local_operators, const Scalar<DataVector>& eta,
    const double f) {
  local_operators.check_mesh(mesh);
  const size_t number_of_points = mesh.number_of_grid_points();

  // 1. The constrained least-squares fit of the potential rates.
  std::array<double, number_of_potentials> means{};
  std::array<std::array<const DataVector*, number_of_potentials>, 3> targets{};
  for (size_t a = 0; a < number_of_potentials; ++a) {
    gsl::at(means, a) = local_operators.mean(dt_u_target.get(a));
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(gsl::at(targets, i), a) = &dt_q_target.get(i, a);
    }
  }
  std::array<DataVector, number_of_potentials> fitted{};
  local_operators.fit(make_not_null(&fitted), targets, means);
  for (size_t a = 0; a < number_of_potentials; ++a) {
    u_rate->get(a) = gsl::at(fitted, a);
  }
  potential_gradients(fitted_gradient, *u_rate, mesh, inverse_jacobian);

  // 2. The compatibility correction.
  for (size_t s = 0; s < number_of_points; ++s) {
    const PotentialVector u_s = potentials_at(u, s);
    const VelocityVector w_s = velocities_at(w, s);
    const GradientVector q_s = gradients_at(q, s);
    const PotentialVector r_s = potentials_at(*u_rate, s);
    const GradientVector target_s = gradients_at(dt_q_target, s);
    const GradientVector fitted_gradient_s = gradients_at(*fitted_gradient, s);
    const LogMetricChart chart{stf_decode(&u_s[PotentialIndex::log_metric])};
    const PotentialVector rhs = potential_rhs(chart, u_s, w_s, q_s, f);
    GradientVector defect{};
    PotentialVector rhs_minus_fit{};
    for (size_t a = 0; a < number_of_potentials; ++a) {
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(gsl::at(defect, i), a) =
            gsl::at(gsl::at(target_s, i), a) -
            gsl::at(gsl::at(fitted_gradient_s, i), a);
      }
      gsl::at(rhs_minus_fit, a) = gsl::at(rhs, a) - gsl::at(r_s, a);
    }
    const PotentialVector gradient_defect_term =
        potential_rhs(chart, u_s, VelocityVector{}, defect, f);
    const PotentialVector nonlinear_term =
        potential_rhs_potential_derivative(chart, u_s, w_s, q_s, rhs_minus_fit);
    PotentialVector correction_input{};
    for (size_t a = 0; a < number_of_potentials; ++a) {
      gsl::at(correction_input, a) =
          gsl::at(gradient_defect_term, a) + gsl::at(nonlinear_term, a);
    }
    const VelocityVector correction =
        apply_rhs_inverse(u_s[PotentialIndex::lapse], correction_input, f);
    for (size_t a = 0; a < number_of_velocities; ++a) {
      w_rate->get(a)[s] = dt_w_target.get(a)[s] + gsl::at(correction, a);
    }
    // Replace the volume source -eta b by -eta r_beta / f. The complete
    // fitted rate includes the mean correction from internal value jumps.
    for (size_t i = 0; i < 3; ++i) {
      w_rate->get(VelocityIndex::b + i)[s] -=
          get(eta)[s] * (gsl::at(r_s, PotentialIndex::shift + i) / f -
                        gsl::at(w_s, VelocityIndex::b + i));
    }
  }
}

void FinalizeTimeDerivative::apply(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> dt_conformal_metric,
    const gsl::not_null<Scalar<DataVector>*> dt_conformal_factor,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> dt_a_tilde,
    const gsl::not_null<Scalar<DataVector>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Scalar<DataVector>*> dt_theta,
    const gsl::not_null<tnsr::I<DataVector, 3>*> dt_gamma_hat,
    const gsl::not_null<Scalar<DataVector>*> dt_lapse,
    const gsl::not_null<tnsr::I<DataVector, 3>*> dt_shift,
    const gsl::not_null<tnsr::I<DataVector, 3>*> dt_b,
    const Tags::ChartU::type& u, const Tags::ChartW::type& w,
    const Tags::ChartQ::type& q, const Tags::ChartU::type& dt_u_target,
    const Tags::ChartW::type& dt_w_target,
    const Tags::ChartQ::type& dt_q_target, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const LocalOperators& local_operators, const Scalar<DataVector>& eta) {
  const size_t number_of_points = mesh.number_of_grid_points();
  Tags::ChartU::type u_rate{number_of_points};
  Tags::ChartW::type w_rate{number_of_points};
  Tags::ChartQ::type fitted_gradient{number_of_points};
  finalized_chart_rates(make_not_null(&u_rate), make_not_null(&w_rate),
                        make_not_null(&fitted_gradient), u, w, q, dt_u_target,
                        dt_w_target, dt_q_target, mesh, inverse_jacobian,
                        local_operators, eta, ::Ccz4::fd::System::f);

  // 4. Conversion to the physical rates.
  for (size_t s = 0; s < number_of_points; ++s) {
    const PotentialVector u_s = potentials_at(u, s);
    const VelocityVector w_s = velocities_at(w, s);
    const PotentialVector r_s = potentials_at(u_rate, s);
    const VelocityVector w_rate_s = velocities_at(w_rate, s);
    const LogMetricChart chart{stf_decode(&u_s[PotentialIndex::log_metric])};
    const Matrix3 log_metric_rate =
        stf_decode(&r_s[PotentialIndex::log_metric]);
    const Matrix3 metric_rate = chart.tangent(log_metric_rate);
    const Matrix3 a_tilde_rate =
        0.5 * (chart.tangent(stf_decode(&w_rate_s[VelocityIndex::a])) +
               chart.second_derivative(log_metric_rate,
                                       stf_decode(&w_s[VelocityIndex::a])));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        dt_conformal_metric->get(i, j)[s] = metric_rate(i, j);
        dt_a_tilde->get(i, j)[s] = a_tilde_rate(i, j);
      }
    }
    get(*dt_conformal_factor)[s] =
        std::exp(u_s[PotentialIndex::log_conformal_factor]) *
        r_s[PotentialIndex::log_conformal_factor];
    get(*dt_lapse)[s] = r_s[PotentialIndex::lapse];
    get(*dt_trace_extrinsic_curvature)[s] =
        w_rate_s[VelocityIndex::trace_extrinsic_curvature];
    get(*dt_theta)[s] = w_rate_s[VelocityIndex::theta];
    for (size_t i = 0; i < 3; ++i) {
      dt_shift->get(i)[s] = gsl::at(r_s, PotentialIndex::shift + i);
      dt_gamma_hat->get(i)[s] = gsl::at(w_rate_s, VelocityIndex::gamma_hat + i);
      dt_b->get(i)[s] = gsl::at(w_rate_s, VelocityIndex::b + i);
    }
  }
}
}  // namespace Ccz4::GradientProjection
