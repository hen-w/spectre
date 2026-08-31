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
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Options/String.hpp"
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

namespace SecondOrderScalarWave::BoundaryConditions {
/*!
 * \brief Sets boundary conditions using the characteristic decomposition:
 * incoming modes from analytic data, outgoing modes from the interior.
 *
 * The characteristic fields and their speeds are (see
 * `SecondOrderScalarWave::characteristic_fields`):
 * - \f$v^+ = \Pi + n^i\Phi_i\f$: speed \f$+1\f$ (always outgoing, taken from
 *   the interior)
 * - \f$v^- = \Pi - n^i\Phi_i\f$: speed \f$-1\f$ (always incoming, taken from
 *   the analytic data, or set to zero if `ZeroIncomingMode` is true)
 * - \f$v^0_i\f$: speed \f$0\f$ (taken from the analytic data if
 *   `PrescribeZeroSpeedModes` is true, from the interior otherwise)
 *
 * \f$\Psi\f$ has no characteristic field in the second-order system. The
 * ghost \f$\Psi\f$ is, by default, the boundary-evolved
 * `evolution::dg::Tags::BoundaryValue<Tags::Psi>`, integrated per face from
 * the time derivative this class produces in
 * `boundary_field_time_derivatives`:
 * \f$\partial_t\Psi_b = -\Pi_b = -\tfrac{1}{2}(v^+ + v^-)\f$ evaluated from
 * the mixed characteristic modes. `CopyPsiFromInterior` instead copies the
 * ghost \f$\Psi\f$ from the interior (and the boundary-evolved value is
 * unused, so its time derivative is set to zero);
 * `PrescribeZeroSpeedModes` instead sets the ghost \f$\Psi\f$ from the
 * analytic data (its zero-speed content is prescribed like \f$v^0_i\f$).
 *
 * Moving meshes are not supported: the characteristic speeds are defined
 * without a mesh velocity, so both member functions error if a face mesh
 * velocity is supplied.
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
        "If true, the zero-speed content (v^0 and the ghost Psi) is set from "
        "the analytic data. If false, v^0 is taken from the interior and the "
        "ghost Psi from the boundary-evolved value.";
    using type = bool;
  };

  /// \brief If true, the ghost Psi is copied directly from the interior
  /// evolved Psi, ignoring the boundary-evolved value. Cannot be combined
  /// with `PrescribeZeroSpeedModes=true`.
  struct CopyPsiFromInterior {
    static constexpr Options::String help =
        "If true, ghost Psi is copied from the interior Psi, ignoring the "
        "boundary-evolved value. Cannot be true when PrescribeZeroSpeedModes "
        "is also true.";
    using type = bool;
  };

  /// \brief If true, the incoming characteristic mode \f$v^-\f$ is set to
  /// zero instead of analytic data.
  struct ZeroIncomingMode {
    static constexpr Options::String help =
        "If true, the incoming characteristic mode is set to zero instead of "
        "analytic data.";
    using type = bool;
  };

  using options = tmpl::list<AnalyticPrescription, PrescribeZeroSpeedModes,
                             CopyPsiFromInterior, ZeroIncomingMode>;

  static constexpr Options::String help{
      "Boundary condition using the characteristic decomposition. Incoming "
      "modes are set from analytic data, outgoing modes from the interior; "
      "the ghost Psi is the boundary-evolved value by default."};

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

  // Opt into evolving the system's boundary variables (BoundaryValue(Psi)):
  // this class produces their per-face time derivatives in
  // `boundary_field_time_derivatives` and consumes their current values in
  // `dg_ghost`.
  static constexpr bool evolves_boundary_variables = true;

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using dg_interior_dt_vars_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<::Tags::Time>;

  // The projected interior inputs to `boundary_field_time_derivatives`.
  // These are a subset of the fields projected for `dg_ghost` (a subset of
  // the `dg_interior_*` tags), so no extra projection is needed.
  using boundary_field_time_derivatives_evolved_variables_tags =
      tmpl::list<Tags::Pi, Tags::Phi<Dim>>;
  using boundary_field_time_derivatives_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;

  // The exterior fields are filled positionally in the framework's ghost-fill
  // order: the evolved variables (Psi, Pi) followed by the auxiliary variable
  // (Phi). The current per-face boundary value `boundary_psi_value` is
  // supplied by the boundary-evolved-variables plumbing immediately after the
  // normal covector.
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

  // Produces the per-face time derivative of the boundary-evolved
  // `BoundaryValue<Psi>`: dt = -Pi_boundary = -0.5 (v^+ + v^-) from the mixed
  // characteristic modes. The current boundary value `boundary_psi_value` is
  // unused (the time derivative does not depend on the integrated value).
  std::optional<std::string> boundary_field_time_derivatives(
      gsl::not_null<Scalar<DataVector>*> dt_boundary_psi,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const Scalar<DataVector>& boundary_psi_value,
      const Scalar<DataVector>& interior_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      double time) const;

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
  bool prescribe_zero_speed_modes_{false};
  bool copy_psi_from_interior_{false};
  bool zero_incoming_mode_{false};
};
}  // namespace SecondOrderScalarWave::BoundaryConditions
