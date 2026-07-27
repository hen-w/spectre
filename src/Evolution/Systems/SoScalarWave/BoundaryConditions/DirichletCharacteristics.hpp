// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedFields/Tags.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
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

namespace SoScalarWave::BoundaryConditions {
/*!
 * \brief Sets boundary conditions using characteristic decomposition:
 * incoming modes from analytic data, outgoing modes from interior.
 *
 * The characteristic speeds (including mesh velocity \f$v_g\f$) are:
 * - \f$v^+\f$: speed \f$+1 - v_g \cdot n\f$
 * - \f$v^-\f$: speed \f$-1 - v_g \cdot n\f$
 * - \f$v^{\hat\psi}\f$: speed \f$-v_g \cdot n\f$
 * - \f$v^{\hat 0}_i\f$: speed \f$-v_g \cdot n\f$
 *
 * Each mode is taken from the interior if its speed is positive (outgoing)
 * or from analytic data if its speed is negative (incoming). For modes with
 * exactly zero speed, the `PrescribeZeroSpeedModes` option determines whether
 * to use the analytic data (true) or the interior (false).
 */
template <size_t Dim>
class DirichletCharacteristics final : public BoundaryCondition<Dim> {
 public:
  /// \brief What analytic solution/data to prescribe.
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution/data to prescribe.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };

  /// \brief Whether to prescribe zero-speed characteristic modes from the
  /// analytic solution (true) or leave them as interior values (false).
  struct PrescribeZeroSpeedModes {
    static constexpr Options::String help =
        "If true, zero-speed modes (VPsi, VZero) are set from the analytic "
        "data. If false, they are taken from the interior.";
    using type = bool;
  };

  /// \brief If true, ghost Psi is copied directly from the interior evolved
  /// Psi (original behavior before BoundaryPsi was introduced). Cannot be
  /// combined with PrescribeZeroSpeedModes=true.
  struct CopyPsiFromInterior {
    static constexpr Options::String help =
        "If true, ghost Psi is copied from the interior Psi, ignoring "
        "BoundaryPsi. Cannot be true when PrescribeZeroSpeedModes is also "
        "true.";
    using type = bool;
  };

  /// \brief If true, incoming characteristic modes (speed < 0) are set to
  /// zero instead of analytic data.
  struct ZeroIncomingMode {
    static constexpr Options::String help =
        "If true, incoming characteristic modes are set to zero instead of "
        "analytic data.";
    using type = bool;
  };

  using options = tmpl::list<AnalyticPrescription, PrescribeZeroSpeedModes,
                             CopyPsiFromInterior, ZeroIncomingMode>;

  static constexpr Options::String help{
      "Boundary condition using characteristic decomposition. Incoming modes "
      "are set from analytic data, outgoing modes from the interior."};

  DirichletCharacteristics() = default;
  DirichletCharacteristics(DirichletCharacteristics&&) = default;
  DirichletCharacteristics& operator=(DirichletCharacteristics&&) = default;
  DirichletCharacteristics(const DirichletCharacteristics&);
  DirichletCharacteristics& operator=(const DirichletCharacteristics&);
  ~DirichletCharacteristics() override = default;

  DirichletCharacteristics(std::unique_ptr<evolution::initial_data::InitialData>
                               analytic_prescription,
                           bool prescribe_zero_speed_modes,
                           bool copy_psi_from_interior,
                           bool zero_incoming_mode);

  explicit DirichletCharacteristics(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DirichletCharacteristics);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using dg_interior_dt_vars_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<::Tags::Time>;

  // The boundary-evolved twin of Psi, stored and time-integrated per external
  // face by the boundary-evolved-fields facility. Its current value is fed into
  // `dg_ghost` as an extra argument and its time derivative is produced by
  // `boundary_field_time_derivatives`.
  using boundary_evolved_variables =
      tmpl::list<evolution::dg::Tags::BoundaryValue<Tags::Psi>>;

  // The projected interior inputs to `boundary_field_time_derivatives`. These
  // are a subset of the fields projected for `dg_ghost` (they equal the
  // `dg_interior_*` tags), so no extra projection is needed.
  using boundary_field_time_derivatives_evolved_variables_tags =
      tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>;
  using boundary_field_time_derivatives_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;

  // The exterior fields are filled positionally in the framework's ghost-fill
  // order: the evolved variables (Psi, Pi) followed by the auxiliary variable
  // (Phi). The current per-face boundary value `boundary_psi_value` is supplied
  // by the facility immediately after the normal covector.
  std::optional<std::string> dg_ghost(
      gsl::not_null<Scalar<DataVector>*> psi,
      gsl::not_null<Scalar<DataVector>*> pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const Scalar<DataVector>& boundary_psi_value,
      const Scalar<DataVector>& interior_psi,
      const Scalar<DataVector>& interior_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      double time) const;

  // Produces the per-face time derivative of the boundary-evolved field
  // `BoundaryValue<Psi>`: dt = -0.5 (v^+ + v^-) = -Pi_boundary. The current
  // boundary value `boundary_psi_value` is unused (dt does not depend on the
  // integrated value here).
  std::optional<std::string> boundary_field_time_derivatives(
      gsl::not_null<Scalar<DataVector>*> dt_boundary_psi,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const Scalar<DataVector>& boundary_psi_value,
      const Scalar<DataVector>& interior_psi,
      const Scalar<DataVector>& interior_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      double time) const;

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
  bool prescribe_zero_speed_modes_;
  bool copy_psi_from_interior_;
  bool zero_incoming_mode_;
};
}  // namespace SoScalarWave::BoundaryConditions
