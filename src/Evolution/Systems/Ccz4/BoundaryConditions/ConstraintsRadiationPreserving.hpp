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
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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
 * \brief Ghost boundary condition that applies constraints and
 * radiation preserving boundary conditions (CRPBC) for SoCcz4.
 *
 * \details This is a `Ghost`-type boundary condition. The incoming
 * characteristic modes (UScalar3Minus, UVector2Minus, UScalar2Minus,
 * UTensorMinus) are evolved as independent variables whose dt is
 * computed algebraically from the CRPBC equations (in LdgTimeDerivative).
 * Their time-integrated values are used in `dg_ghost` to construct the
 * exterior state via inverse characteristic transform.
 *
 * Gauge modes (UVector3Minus, UScalar4Minus, UScalar5Minus) and
 * zero-speed modes (UVector1Zero, UScalar1Zero) are set from an
 * analytic prescription.
 */
class ConstraintsRadiationPreserving final : public BoundaryCondition {
 public:
  /// \brief What analytic solution/data to prescribe for gauge modes.
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution/data to prescribe for gauge modes.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };
  /// \brief Debug flag: if true, prescribe ALL incoming modes from analytic
  /// solution (same behavior as DirichletCharacteristics). The evolved
  /// boundary-mode variables are ignored.
  struct UseAnalyticForAll {
    static constexpr Options::String help =
        "If true, prescribe ALL incoming characteristic modes from the "
        "analytic solution (like DirichletCharacteristics). For debugging.";
    using type = bool;
    static type default_value() { return false; }
  };
  struct PenaltyMultiplier {
    static constexpr Options::String help =
        "Multiplier for the theta-constraint penalty term at the boundary. "
        "The effective penalty coefficient is PenaltyMultiplier / h, where h "
        "is the grid spacing in the outward normal direction.";
    using type = double;
    static type default_value() { return 1.0; }
  };
  using options =
      tmpl::list<AnalyticPrescription, UseAnalyticForAll, PenaltyMultiplier>;
  static constexpr Options::String help{
      "Constraints and radiation preserving boundary conditions. "
      "Uses Ghost BC with time-integrated incoming characteristic modes."};

  ConstraintsRadiationPreserving() = default;
  ConstraintsRadiationPreserving(ConstraintsRadiationPreserving&&) = default;
  ConstraintsRadiationPreserving& operator=(ConstraintsRadiationPreserving&&) =
      default;
  ConstraintsRadiationPreserving(const ConstraintsRadiationPreserving&);
  ConstraintsRadiationPreserving& operator=(
      const ConstraintsRadiationPreserving&);
  ~ConstraintsRadiationPreserving() override = default;

  explicit ConstraintsRadiationPreserving(
      std::unique_ptr<evolution::initial_data::InitialData>
          analytic_prescription,
      bool use_analytic_for_all = false, double penalty_multiplier = 1.0);

  explicit ConstraintsRadiationPreserving(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition,
      ConstraintsRadiationPreserving);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  double penalty_multiplier() const { return penalty_multiplier_; }

  // DG interface: Ghost BC providing external state
  using dg_interior_evolved_variables_tags =
      ::Ccz4::fd::System::variables_tag_list;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>>;
  using dg_interior_dt_vars_tags = tmpl::list<>;
  using dg_gridless_tags =
      tmpl::list<::Tags::Time, ::Ccz4::fd::Tags::EvolveLapseAndShift>;

  std::optional<std::string> dg_ghost(
      // 17 not_null exterior outputs (variables_tag_list order):
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
      gsl::not_null<Scalar<DataVector>*> u_scalar3_minus,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> u_vector2_minus,
      gsl::not_null<Scalar<DataVector>*> u_scalar2_minus,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> u_tensor_minus,
      // Standard DG ghost args:
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      // dg_interior_evolved_variables_tags (all 17 interior vars):
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
      const Scalar<DataVector>& interior_u_scalar3_minus,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interior_u_vector2_minus,
      const Scalar<DataVector>& interior_u_scalar2_minus,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_u_tensor_minus,
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
  bool use_analytic_for_all_{false};
  double penalty_multiplier_{1.0};
};
}  // namespace Ccz4::BoundaryConditions
