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

  using options = tmpl::list<AnalyticPrescription, PrescribeZeroSpeedModes,
                             CopyPsiFromInterior>;

  static constexpr Options::String help{
      "Boundary condition using characteristic decomposition. Incoming modes "
      "are set from analytic data, outgoing modes from the interior."};

  DirichletCharacteristics() = default;
  DirichletCharacteristics(DirichletCharacteristics&&) = default;
  DirichletCharacteristics& operator=(DirichletCharacteristics&&) = default;
  DirichletCharacteristics(const DirichletCharacteristics&);
  DirichletCharacteristics& operator=(const DirichletCharacteristics&);
  ~DirichletCharacteristics() override = default;

  DirichletCharacteristics(
      std::unique_ptr<evolution::initial_data::InitialData>
          analytic_prescription,
      bool prescribe_zero_speed_modes, bool copy_psi_from_interior);

  explicit DirichletCharacteristics(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DirichletCharacteristics);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::GhostAndTimeDerivative;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>, Tags::BoundaryPsi>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using dg_interior_dt_vars_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<::Tags::Time>;

  std::optional<std::string> dg_ghost(
      gsl::not_null<Scalar<DataVector>*> psi,
      gsl::not_null<Scalar<DataVector>*> pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
      gsl::not_null<Scalar<DataVector>*> boundary_psi,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const Scalar<DataVector>& interior_psi,
      const Scalar<DataVector>& interior_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
      const Scalar<DataVector>& interior_boundary_psi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      double time) const;

  std::optional<std::string> dg_time_derivative(
      gsl::not_null<Scalar<DataVector>*> dt_psi_correction,
      gsl::not_null<Scalar<DataVector>*> dt_pi_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          dt_phi_correction,
      gsl::not_null<Scalar<DataVector>*> dt_boundary_psi_correction,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const Scalar<DataVector>& interior_psi,
      const Scalar<DataVector>& interior_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
      const Scalar<DataVector>& interior_boundary_psi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      double time) const;

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
  bool prescribe_zero_speed_modes_;
  bool copy_psi_from_interior_;
};
}  // namespace SoScalarWave::BoundaryConditions
