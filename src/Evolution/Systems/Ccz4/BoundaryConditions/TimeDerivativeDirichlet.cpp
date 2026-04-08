// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/BoundaryConditions/TimeDerivativeDirichlet.hpp"

#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace Ccz4::BoundaryConditions {

// LCOV_EXCL_START
TimeDerivativeDirichlet::TimeDerivativeDirichlet(
    CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}
// LCOV_EXCL_STOP

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
TimeDerivativeDirichlet::get_clone() const {
  return std::make_unique<TimeDerivativeDirichlet>(*this);
}

void TimeDerivativeDirichlet::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
}
// NOLINTNEXTLINE
PUP::able::PUP_ID TimeDerivativeDirichlet::my_PUP_ID = 0;

std::optional<std::string> TimeDerivativeDirichlet::dg_time_derivative(
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        dt_conformal_metric_correction,
    const gsl::not_null<Scalar<DataVector>*>
        dt_conformal_factor_correction,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        dt_a_tilde_correction,
    const gsl::not_null<Scalar<DataVector>*>
        dt_trace_extrinsic_curvature_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_theta_correction,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        dt_gamma_hat_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_lapse_correction,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        dt_shift_correction,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        dt_auxiliary_shift_b_correction,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        dt_field_a_correction,
    const gsl::not_null<tnsr::iJ<DataVector, 3, Frame::Inertial>*>
        dt_field_b_correction,
    const gsl::not_null<tnsr::ijj<DataVector, 3, Frame::Inertial>*>
        dt_field_d_correction,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        dt_field_p_correction,
    const gsl::not_null<Scalar<DataVector>*>
        dt_u_scalar3_minus_correction,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        dt_u_vector2_minus_correction,
    const gsl::not_null<Scalar<DataVector>*>
        dt_u_scalar2_minus_correction,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        dt_u_tensor_minus_correction,
    const std::optional<
        tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/) const {
  // Zero corrections: the infrastructure does NOT pre-initialize them.
  // The actual BC logic is applied later by OverwriteExternalBoundaryDtDirichlet,
  // which directly overwrites dt_vars at external boundary face nodes.
  for (auto& component : *dt_conformal_metric_correction) {
    component = 0.0;
  }
  get(*dt_conformal_factor_correction) = 0.0;
  for (auto& component : *dt_a_tilde_correction) {
    component = 0.0;
  }
  get(*dt_trace_extrinsic_curvature_correction) = 0.0;
  get(*dt_theta_correction) = 0.0;
  for (auto& component : *dt_gamma_hat_correction) {
    component = 0.0;
  }
  get(*dt_lapse_correction) = 0.0;
  for (auto& component : *dt_shift_correction) {
    component = 0.0;
  }
  for (auto& component : *dt_auxiliary_shift_b_correction) {
    component = 0.0;
  }
  for (auto& component : *dt_field_a_correction) {
    component = 0.0;
  }
  for (auto& component : *dt_field_b_correction) {
    component = 0.0;
  }
  for (auto& component : *dt_field_d_correction) {
    component = 0.0;
  }
  for (auto& component : *dt_field_p_correction) {
    component = 0.0;
  }
  get(*dt_u_scalar3_minus_correction) = 0.0;
  for (auto& component : *dt_u_vector2_minus_correction) {
    component = 0.0;
  }
  get(*dt_u_scalar2_minus_correction) = 0.0;
  for (auto& component : *dt_u_tensor_minus_correction) {
    component = 0.0;
  }
  return {};
}

void TimeDerivativeDirichlet::fd_ghost(
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>,
    const gsl::not_null<Scalar<DataVector>*>,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>,
    const gsl::not_null<Scalar<DataVector>*>,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>,
    const gsl::not_null<Scalar<DataVector>*>,
    const gsl::not_null<Scalar<DataVector>*>,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>,
    const Direction<3>& /*direction*/) const {
  ERROR(
      "TimeDerivativeDirichlet fd_ghost is not implemented. "
      "This BC is only available for the DG (LDG) path.");
}
}  // namespace Ccz4::BoundaryConditions
