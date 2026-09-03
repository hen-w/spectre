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
 * The characteristic fields are \f$v^0_i\f$, \f$v^+ = \Pi + n^i\Phi_i\f$, and
 * \f$v^- = \Pi - n^i\Phi_i\f$ (see
 * `SecondOrderScalarWave::characteristic_fields`). Their speeds are the
 * grid-frame ones returned by
 * `SecondOrderScalarWave::characteristic_speeds(normal, mesh_velocity)`, i.e.
 * \f$\lambda^0 = -n_i v^i\f$ and \f$\lambda^\pm = \pm 1 - n_i v^i\f$ (on a
 * static mesh these reduce to \f$0, +1, -1\f$).
 *
 * The mode selection is pointwise on the face: a mode with negative speed is
 * incoming and is prescribed from the analytic data (or set to zero if
 * `ZeroIncomingMode` is true), while a mode with non-negative speed is taken
 * from the interior. On a static mesh only \f$v^-\f$ is ever incoming, so
 * this reduces to \f$v^+\f$ and \f$v^0_i\f$ from the interior and \f$v^-\f$
 * from the data.
 *
 * \f$\Psi\f$ has no characteristic field in the second-order system. The
 * ghost \f$\Psi\f$ is the boundary-evolved
 * `evolution::dg::Tags::BoundaryValue<Tags::Psi>`, integrated per face from
 * the time derivative this class produces in
 * `boundary_field_time_derivatives`:
 * \f$\partial_t\Psi_b = -\Pi_b + v^i(\Phi_b)_i\f$, with both \f$\Pi_b\f$ and
 * \f$(\Phi_b)_i\f$ taken from the mixed-mode ghost state (on a static mesh
 * the mesh-velocity term drops and this is \f$-\Pi_b\f$).
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

  /// \brief If true, every characteristic mode is set to zero wherever it is
  /// incoming (negative speed), instead of taking the analytic data.
  struct ZeroIncomingMode {
    static constexpr Options::String help =
        "If true, each characteristic mode is set to zero wherever it is "
        "incoming, instead of analytic data. On a static mesh only v^- is "
        "incoming.";
    using type = bool;
  };

  using options = tmpl::list<AnalyticPrescription, ZeroIncomingMode>;

  static constexpr Options::String help{
      "Boundary condition using the characteristic decomposition. Incoming "
      "modes are set from analytic data, outgoing modes from the interior; "
      "the ghost Psi is the boundary-evolved value."};

  DirichletCharacteristics() = default;
  DirichletCharacteristics(DirichletCharacteristics&&) = default;
  DirichletCharacteristics& operator=(DirichletCharacteristics&&) = default;
  DirichletCharacteristics(const DirichletCharacteristics&);
  DirichletCharacteristics& operator=(const DirichletCharacteristics&);
  ~DirichletCharacteristics() override = default;

  DirichletCharacteristics(std::unique_ptr<evolution::initial_data::InitialData>
                               analytic_prescription,
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
      tmpl::list<Tags::Pi, Tags::Phi<Dim>>;
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
      const Scalar<DataVector>& interior_pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      double time) const;

  // Produces the per-face time derivative of the boundary-evolved
  // `BoundaryValue<Psi>`: dt = -Pi_boundary + v^i (Phi_boundary)_i from the
  // mixed-mode ghost state. The current boundary value `boundary_psi_value`
  // is unused (the time derivative does not depend on the integrated value).
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
  bool zero_incoming_mode_{false};
};
}  // namespace SecondOrderScalarWave::BoundaryConditions
