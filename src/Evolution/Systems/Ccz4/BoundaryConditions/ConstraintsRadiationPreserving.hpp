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
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace Ccz4::BoundaryConditions {
/*!
 * \brief Constraints and radiation preserving boundary conditions via
 * characteristic decomposition.
 *
 * This boundary condition is identical to DirichletCharacteristics, except that
 * the analytic solution is evaluated at a fixed "initial time" rather than the
 * current simulation time. This freezes the boundary data at the analytic
 * values corresponding to the initial time while still using the characteristic
 * mode-mixing machinery to replace incoming modes.
 */
class ConstraintsRadiationPreserving final : public BoundaryCondition {
 public:
  /// \brief What analytic solution/data to prescribe.
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution/data to prescribe.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };
  /// \brief Debug kill switch: prescribe analytic data on outgoing modes
  /// instead of incoming ones.
  struct PrescribeOutgoing {
    static constexpr Options::String help =
        "If true, prescribe analytic values on OUTGOING modes (and leave "
        "incoming modes at interior values). For debugging only.";
    using type = bool;
  };
  /// \brief The time at which to evaluate the analytic solution for the
  /// boundary data.
  struct InitialTime {
    static constexpr Options::String help =
        "Time at which to evaluate the analytic solution for boundary data. "
        "Set to 0.0 to freeze the boundary at initial data.";
    using type = double;
  };
  using options =
      tmpl::list<AnalyticPrescription, PrescribeOutgoing, InitialTime>;
  static constexpr Options::String help{
      "Constraints and radiation preserving boundary conditions. "
      "Like DirichletCharacteristics but evaluates the analytic solution at "
      "a fixed initial time rather than the current simulation time."};

  ConstraintsRadiationPreserving() = default;
  ConstraintsRadiationPreserving(ConstraintsRadiationPreserving&&) = default;
  ConstraintsRadiationPreserving& operator=(ConstraintsRadiationPreserving&&) =
      default;
  ConstraintsRadiationPreserving(const ConstraintsRadiationPreserving&);
  ConstraintsRadiationPreserving& operator=(
      const ConstraintsRadiationPreserving&);
  ~ConstraintsRadiationPreserving() override = default;

  explicit ConstraintsRadiationPreserving(CkMigrateMessage* msg);

  explicit ConstraintsRadiationPreserving(
      std::unique_ptr<evolution::initial_data::InitialData>
          analytic_prescription,
      bool prescribe_outgoing, double initial_time);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition,
      ConstraintsRadiationPreserving);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::GhostAndTimeDerivative;

  void pup(PUP::er& p) override;

  // DG interface: Ghost BC providing external state for LDG boundary
  // corrections. Incoming characteristic modes are replaced with analytic
  // target values evaluated at the initial time.
  using dg_interior_evolved_variables_tags =
      ::Ccz4::fd::System::variables_tag_list;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>>;
  using dg_interior_dt_vars_tags = tmpl::list<>;
  using dg_gridless_tags =
      tmpl::list<::Tags::Time, ::Ccz4::fd::Tags::EvolveLapseAndShift>;

  std::optional<std::string> dg_ghost(
      // not_null exterior outputs (variables_tag_list order):
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> conformal_metric,
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> a_tilde,
      gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> theta,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> gamma_hat,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> auxiliary_shift_b,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> field_a,
      gsl::not_null<tnsr::iJ<DataVector, 3, Frame::Inertial>*> field_b,
      gsl::not_null<tnsr::ijj<DataVector, 3, Frame::Inertial>*> field_d,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> field_p,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> u_tensor_minus,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
          boundary_conformal_metric,
      gsl::not_null<Scalar<DataVector>*> boundary_conformal_factor,
      gsl::not_null<Scalar<DataVector>*> boundary_lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> boundary_shift,
      gsl::not_null<Scalar<DataVector>*> boundary_theta,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> boundary_z,
      // Standard DG ghost args:
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      // dg_interior_evolved_variables_tags (all interior vars):
      const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_conformal_metric,
      const Scalar<DataVector>& interior_conformal_factor,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_a_tilde,
      const Scalar<DataVector>& interior_trace_extrinsic_curvature,
      const Scalar<DataVector>& interior_theta,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_gamma_hat,
      const Scalar<DataVector>& interior_lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_auxiliary_shift_b,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interior_field_a,
      const tnsr::iJ<DataVector, 3, Frame::Inertial>& interior_field_b,
      const tnsr::ijj<DataVector, 3, Frame::Inertial>& interior_field_d,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interior_field_p,
      const tnsr::ii<DataVector, 3, Frame::Inertial>&
          interior_boundary_u_tensor_minus,
      const tnsr::ii<DataVector, 3, Frame::Inertial>&
          interior_boundary_conformal_metric,
      const Scalar<DataVector>& interior_boundary_conformal_factor,
      const Scalar<DataVector>& interior_boundary_lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_boundary_shift,
      const Scalar<DataVector>& interior_boundary_theta,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interior_boundary_z,
      // dg_interior_temporary_tags:
      const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
      // dg_gridless_tags:
      double time, bool evolve_lapse_and_shift) const;

  std::optional<std::string> dg_time_derivative(
      // dt correction outputs (variables_tag_list order):
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
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
          dt_u_tensor_minus_correction,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
          dt_boundary_conformal_metric_correction,
      gsl::not_null<Scalar<DataVector>*>
          dt_boundary_conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*> dt_boundary_lapse_correction,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          dt_boundary_shift_correction,
      gsl::not_null<Scalar<DataVector>*> dt_boundary_theta_correction,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          dt_boundary_z_correction,
      // Standard DG time derivative args:
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      // dg_interior_evolved_variables_tags:
      const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_conformal_metric,
      const Scalar<DataVector>& interior_conformal_factor,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_a_tilde,
      const Scalar<DataVector>& interior_trace_extrinsic_curvature,
      const Scalar<DataVector>& interior_theta,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_gamma_hat,
      const Scalar<DataVector>& interior_lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_auxiliary_shift_b,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interior_field_a,
      const tnsr::iJ<DataVector, 3, Frame::Inertial>& interior_field_b,
      const tnsr::ijj<DataVector, 3, Frame::Inertial>& interior_field_d,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interior_field_p,
      const tnsr::ii<DataVector, 3, Frame::Inertial>&
          interior_boundary_u_tensor_minus,
      const tnsr::ii<DataVector, 3, Frame::Inertial>&
          interior_boundary_conformal_metric,
      const Scalar<DataVector>& interior_boundary_conformal_factor,
      const Scalar<DataVector>& interior_boundary_lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_boundary_shift,
      const Scalar<DataVector>& interior_boundary_theta,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interior_boundary_z,
      // dg_interior_temporary_tags:
      const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
      // dg_gridless_tags:
      double time, bool evolve_lapse_and_shift) const;

  // FD interface: not implemented
  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags = tmpl::list<>;
  using fd_interior_primitive_variables_tags = tmpl::list<>;
  using fd_gridless_tags = tmpl::list<>;
  void fd_ghost(gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>,
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
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
  bool prescribe_outgoing_{false};
  double initial_time_{0.0};
};
}  // namespace Ccz4::BoundaryConditions
