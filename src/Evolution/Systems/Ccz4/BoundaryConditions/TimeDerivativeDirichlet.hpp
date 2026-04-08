// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::BoundaryConditions {
/*!
 * \brief TimeDerivative boundary condition that freezes incoming characteristic
 * modes.
 *
 * \details This is a `TimeDerivative`-type boundary condition marker.
 * The `dg_time_derivative` method is a no-op (returns zero corrections).
 * The actual boundary condition logic is implemented by the companion
 * `OverwriteExternalBoundaryDtDirichlet` MutateApply action, which detects
 * faces with this BC via `dynamic_cast` and modifies the volume `dt_vars`
 * directly.
 *
 * At each external boundary face, the action:
 * 1. Decomposes the time derivative into characteristic modes
 * 2. Zeros all incoming dt characteristic modes
 * 3. Inverse-transforms back to evolved variables
 *
 * This is a simpler alternative to `ConstraintsRadiationPreserving`:
 * no constraint or radiation preservation logic, just clean freezing
 * of incoming modes.
 */
class TimeDerivativeDirichlet final : public BoundaryCondition {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "TimeDerivative boundary condition that freezes incoming "
      "characteristic modes to zero. The actual logic is in the "
      "OverwriteExternalBoundaryDtDirichlet MutateApply action."};

  TimeDerivativeDirichlet() = default;
  TimeDerivativeDirichlet(TimeDerivativeDirichlet&&) = default;
  TimeDerivativeDirichlet& operator=(TimeDerivativeDirichlet&&) = default;
  TimeDerivativeDirichlet(const TimeDerivativeDirichlet&) = default;
  TimeDerivativeDirichlet& operator=(const TimeDerivativeDirichlet&) = default;
  ~TimeDerivativeDirichlet() override = default;

  explicit TimeDerivativeDirichlet(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition,
      TimeDerivativeDirichlet);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::TimeDerivative;

  void pup(PUP::er& p) override;

  // The dg_time_derivative is a no-op: zero corrections for all 13 evolved
  // variables. The real boundary condition logic is in
  // OverwriteExternalBoundaryDtDirichlet.
  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<>;
  using dg_interior_dt_vars_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_time_derivative(
      // 13 not_null correction outputs (variables_tag_list order):
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
          dt_conformal_metric_correction,
      gsl::not_null<Scalar<DataVector>*> dt_conformal_factor_correction,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
          dt_a_tilde_correction,
      gsl::not_null<Scalar<DataVector>*>
          dt_trace_extrinsic_curvature_correction,
      gsl::not_null<Scalar<DataVector>*> dt_theta_correction,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          dt_gamma_hat_correction,
      gsl::not_null<Scalar<DataVector>*> dt_lapse_correction,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          dt_shift_correction,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          dt_auxiliary_shift_b_correction,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          dt_field_a_correction,
      gsl::not_null<tnsr::iJ<DataVector, 3, Frame::Inertial>*>
          dt_field_b_correction,
      gsl::not_null<tnsr::ijj<DataVector, 3, Frame::Inertial>*>
          dt_field_d_correction,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          dt_field_p_correction,
      gsl::not_null<Scalar<DataVector>*> dt_u_scalar3_minus_correction,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          dt_u_vector2_minus_correction,
      gsl::not_null<Scalar<DataVector>*> dt_u_scalar2_minus_correction,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
          dt_u_tensor_minus_correction,
      // Standard DG time derivative args:
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector) const;

  // FD interface: not implemented
  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags = tmpl::list<>;
  using fd_interior_primitive_variables_tags = tmpl::list<>;
  using fd_gridless_tags = tmpl::list<>;
  void fd_ghost(
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>,
      gsl::not_null<Scalar<DataVector>*>,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>,
      gsl::not_null<Scalar<DataVector>*>,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>,
      gsl::not_null<Scalar<DataVector>*>,
      gsl::not_null<Scalar<DataVector>*>,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>,
      const Direction<3>&) const;

 private:
};
}  // namespace Ccz4::BoundaryConditions
