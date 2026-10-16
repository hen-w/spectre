// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryCorrections/UpwindGradientProjection.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/NormalOperator.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::GradientProjection::BoundaryCorrections {

UpwindGradientProjection::UpwindGradientProjection(
    const double external_dissipation_factor)
    : external_dissipation_factor_(external_dissipation_factor) {}

UpwindGradientProjection::UpwindGradientProjection(CkMigrateMessage* msg)
    : BoundaryCorrection(msg) {}

std::unique_ptr<evolution::BoundaryCorrection>
UpwindGradientProjection::get_clone() const {
  return std::make_unique<UpwindGradientProjection>(*this);
}

void UpwindGradientProjection::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | external_dissipation_factor_;
}

double UpwindGradientProjection::dg_package_data(
    const gsl::not_null<Tags::PackagedVelocities::type*> packaged_w,
    const gsl::not_null<Tags::PackagedGradients::type*> packaged_q,
    const gsl::not_null<Tags::PackagedPotentials::type*> packaged_u,
    const gsl::not_null<tnsr::i<DataVector, 3>*> packaged_normal_covector,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, 3>& a_tilde,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& theta, const tnsr::I<DataVector, 3>& gamma_hat,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& b, const Tags::PotentialGradients::type& q,
    const tnsr::i<DataVector, 3>& normal_covector,
    const std::optional<tnsr::I<DataVector, 3>>& /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
    const Direction<3>& /*face_direction*/) const {
  *packaged_q = q;
  *packaged_normal_covector = normal_covector;
  double largest_speed = 0.0;
  for (size_t s = 0; s < get(lapse).size(); ++s) {
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
    LogMetricChart chart{};
    checked_chart_from_physical(make_not_null(&u_s), make_not_null(&w_s),
                                make_not_null(&chart), state, "face");
    set_potentials_at(packaged_u, s, u_s);
    set_velocities_at(packaged_w, s, w_s);
    largest_speed =
        std::max(largest_speed,
                 largest_characteristic_speed(
                     chart, u_s,
                     {{get<0>(normal_covector)[s], get<1>(normal_covector)[s],
                       get<2>(normal_covector)[s]}},
                     ::Ccz4::fd::System::f));
  }
  return largest_speed;
}

template <bool ForExternalBoundary>
void UpwindGradientProjection::dg_boundary_terms(
    const gsl::not_null<Tags::VelocityTarget::type*> w_correction,
    const gsl::not_null<Tags::GradientTarget::type*> q_correction,
    const gsl::not_null<Tags::PotentialRateTarget::type*> u_correction,
    const Tags::PackagedVelocities::type& w_int,
    const Tags::PackagedGradients::type& q_int,
    const Tags::PackagedPotentials::type& u_int,
    const tnsr::i<DataVector, 3>& normal_covector_int,
    const Tags::PackagedVelocities::type& w_ext,
    const Tags::PackagedGradients::type& q_ext,
    const Tags::PackagedPotentials::type& u_ext,
    const tnsr::i<DataVector, 3>& /*normal_covector_ext*/,
    const dg::Formulation dg_formulation) const {
  if (dg_formulation != dg::Formulation::StrongInertial) {
    ERROR("UpwindGradientProjection supports only the strong formulation.");
  }
  const double dissipation_factor =
      ForExternalBoundary ? external_dissipation_factor_ : 1.0;
  // An analytic exterior state need not match outgoing fields. Penalizing
  // their jumps would impose extra boundary data beyond ordinary upwinding.
  constexpr double scalar_penalty = ForExternalBoundary ? 0.0 : 1.0;
  const size_t number_of_points = get<0>(u_int).size();
  for (size_t s = 0; s < number_of_points; ++s) {
    const PotentialVector u_int_s = potentials_at(u_int, s);
    const PotentialVector u_ext_s = potentials_at(u_ext, s);
    const VelocityVector w_int_s = velocities_at(w_int, s);
    const VelocityVector w_ext_s = velocities_at(w_ext, s);
    const GradientVector q_int_s = gradients_at(q_int, s);
    const GradientVector q_ext_s = gradients_at(q_ext, s);
    const std::array<double, 3> normal{{get<0>(normal_covector_int)[s],
                                        get<1>(normal_covector_int)[s],
                                        get<2>(normal_covector_int)[s]}};
    PotentialVector u_average{};
    for (size_t a = 0; a < number_of_potentials; ++a) {
      gsl::at(u_average, a) = 0.5 * (gsl::at(u_int_s, a) + gsl::at(u_ext_s, a));
    }
    VelocityVector jump_w{};
    for (size_t a = 0; a < number_of_velocities; ++a) {
      gsl::at(jump_w, a) = gsl::at(w_ext_s, a) - gsl::at(w_int_s, a);
    }
    GradientVector jump_q{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < number_of_potentials; ++a) {
        gsl::at(gsl::at(jump_q, i), a) =
            gsl::at(gsl::at(q_ext_s, i), a) - gsl::at(gsl::at(q_int_s, i), a);
      }
    }
    const LogMetricChart chart{
        stf_decode(&u_average[PotentialIndex::log_metric])};
    const NormalSubspaceVector central = normal_flux(
        chart, u_average, normal, jump_w, jump_q, ::Ccz4::fd::System::f);
    NormalSubspaceVector absolute{};
    if (dissipation_factor != 0.0) {
      const NormalCharacteristics characteristics{chart, u_average, normal,
                                                  ::Ccz4::fd::System::f};
      absolute = characteristics.apply_sign(central);
    }
    VelocityVector correction_w{};
    for (size_t a = 0; a < number_of_velocities; ++a) {
      gsl::at(correction_w, a) =
          -0.5 * (gsl::at(central, a) +
                  dissipation_factor * (gsl::at(absolute, a) +
                                        scalar_penalty * gsl::at(jump_w, a)));
    }
    GradientVector correction_q{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < number_of_potentials; ++a) {
        gsl::at(gsl::at(correction_q, i), a) =
            -0.5 *
            (gsl::at(normal, i) * gsl::at(central, number_of_velocities + a) +
             dissipation_factor *
                 (gsl::at(normal, i) *
                      gsl::at(absolute, number_of_velocities + a) +
                  scalar_penalty * gsl::at(gsl::at(jump_q, i), a)));
      }
    }
    PotentialVector correction_u{};
    if constexpr (not ForExternalBoundary) {
      for (size_t a = 0; a < number_of_potentials; ++a) {
        gsl::at(correction_u, a) =
            -0.5 * (gsl::at(u_ext_s, a) - gsl::at(u_int_s, a));
      }
    }
    set_velocities_at(w_correction, s, correction_w);
    set_gradients_at(q_correction, s, correction_q);
    set_potentials_at(u_correction, s, correction_u);
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID UpwindGradientProjection::my_PUP_ID = 0;

#define INSTANTIATE(EXTERNAL)                                               \
  template void UpwindGradientProjection::dg_boundary_terms<EXTERNAL>(      \
      gsl::not_null<Tags::VelocityTarget::type*>,                           \
      gsl::not_null<Tags::GradientTarget::type*>,                           \
      gsl::not_null<Tags::PotentialRateTarget::type*>,                      \
      const Tags::PackagedVelocities::type&,                                \
      const Tags::PackagedGradients::type&,                                 \
      const Tags::PackagedPotentials::type&, const tnsr::i<DataVector, 3>&, \
      const Tags::PackagedVelocities::type&,                                \
      const Tags::PackagedGradients::type&,                                 \
      const Tags::PackagedPotentials::type&, const tnsr::i<DataVector, 3>&, \
      dg::Formulation) const;
INSTANTIATE(false)
INSTANTIATE(true)
#undef INSTANTIATE
}  // namespace Ccz4::GradientProjection::BoundaryCorrections
