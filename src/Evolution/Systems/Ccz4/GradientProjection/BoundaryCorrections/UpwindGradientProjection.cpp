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
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::GradientProjection::BoundaryCorrections {
namespace {
// The scalar jump penalty in the Euclidean-normal convention.
constexpr double scalar_penalty = 1.0;
}  // namespace

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
    const gsl::not_null<Tags::ChartW::type*> packaged_w,
    const gsl::not_null<Tags::ChartQ::type*> packaged_q,
    const gsl::not_null<Tags::ChartU::type*> packaged_u,
    const gsl::not_null<tnsr::i<DataVector, 3>*> packaged_normal_covector,
    const Tags::ChartW::type& w, const Tags::ChartQ::type& q,
    const Tags::ChartU::type& u,
    const tnsr::ii<DataVector, 3>& /*conformal_metric*/,
    const Scalar<DataVector>& /*conformal_factor*/,
    const tnsr::ii<DataVector, 3>& /*a_tilde*/,
    const Scalar<DataVector>& /*trace_extrinsic_curvature*/,
    const Scalar<DataVector>& /*theta*/,
    const tnsr::I<DataVector, 3>& /*gamma_hat*/,
    const Scalar<DataVector>& /*lapse*/,
    const tnsr::I<DataVector, 3>& /*shift*/,
    const tnsr::I<DataVector, 3>& /*b*/,
    const tnsr::i<DataVector, 3>& normal_covector,
    const std::optional<tnsr::I<DataVector, 3>>& /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
    const Direction<3>& /*face_direction*/) const {
  *packaged_w = w;
  *packaged_q = q;
  *packaged_u = u;
  *packaged_normal_covector = normal_covector;
  double largest_speed = 0.0;
  for (size_t s = 0; s < get<0>(u).size(); ++s) {
    const PotentialVector u_s = potentials_at(u, s);
    const LogMetricChart chart{stf_decode(&u_s[PotentialIndex::log_metric])};
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
    const gsl::not_null<Tags::ChartW::type*> w_correction,
    const gsl::not_null<Tags::ChartQ::type*> q_correction,
    const gsl::not_null<Tags::ChartU::type*> u_correction,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric_correction,
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde_correction,
    const gsl::not_null<Scalar<DataVector>*>
        trace_extrinsic_curvature_correction,
    const gsl::not_null<Scalar<DataVector>*> theta_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat_correction,
    const gsl::not_null<Scalar<DataVector>*> lapse_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> b_correction,
    const Tags::ChartW::type& w_int, const Tags::ChartQ::type& q_int,
    const Tags::ChartU::type& u_int,
    const tnsr::i<DataVector, 3>& normal_covector_int,
    const Tags::ChartW::type& w_ext, const Tags::ChartQ::type& q_ext,
    const Tags::ChartU::type& u_ext,
    const tnsr::i<DataVector, 3>& /*normal_covector_ext*/,
    const dg::Formulation dg_formulation) const {
  if (dg_formulation != dg::Formulation::StrongInertial) {
    ERROR("UpwindGradientProjection supports only the strong formulation.");
  }
  const double dissipation_factor =
      ForExternalBoundary ? external_dissipation_factor_ : 1.0;
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
  // The physical fields receive no boundary correction.
  for (auto& component : *conformal_metric_correction) {
    component = 0.0;
  }
  get(*conformal_factor_correction) = 0.0;
  for (auto& component : *a_tilde_correction) {
    component = 0.0;
  }
  get(*trace_extrinsic_curvature_correction) = 0.0;
  get(*theta_correction) = 0.0;
  for (auto& component : *gamma_hat_correction) {
    component = 0.0;
  }
  get(*lapse_correction) = 0.0;
  for (auto& component : *shift_correction) {
    component = 0.0;
  }
  for (auto& component : *b_correction) {
    component = 0.0;
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID UpwindGradientProjection::my_PUP_ID = 0;

#define INSTANTIATE(EXTERNAL)                                                 \
  template void UpwindGradientProjection::dg_boundary_terms<EXTERNAL>(        \
      gsl::not_null<Tags::ChartW::type*>, gsl::not_null<Tags::ChartQ::type*>, \
      gsl::not_null<Tags::ChartU::type*>,                                     \
      gsl::not_null<tnsr::ii<DataVector, 3>*>,                                \
      gsl::not_null<Scalar<DataVector>*>,                                     \
      gsl::not_null<tnsr::ii<DataVector, 3>*>,                                \
      gsl::not_null<Scalar<DataVector>*>, gsl::not_null<Scalar<DataVector>*>, \
      gsl::not_null<tnsr::I<DataVector, 3>*>,                                 \
      gsl::not_null<Scalar<DataVector>*>,                                     \
      gsl::not_null<tnsr::I<DataVector, 3>*>,                                 \
      gsl::not_null<tnsr::I<DataVector, 3>*>, const Tags::ChartW::type&,      \
      const Tags::ChartQ::type&, const Tags::ChartU::type&,                   \
      const tnsr::i<DataVector, 3>&, const Tags::ChartW::type&,               \
      const Tags::ChartQ::type&, const Tags::ChartU::type&,                   \
      const tnsr::i<DataVector, 3>&, dg::Formulation) const;
INSTANTIATE(false)
INSTANTIATE(true)
#undef INSTANTIATE
}  // namespace Ccz4::GradientProjection::BoundaryCorrections
