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
 * \brief Imposes Dirichlet boundary conditions on the time derivatives of
 * Psi and Pi by replacing them with values from the analytic solution.
 *
 * \details This is a `TimeDerivative`-type boundary condition. At external
 * boundary faces, the volume time derivatives of Psi and Pi are replaced by
 * the analytic time derivatives. The auxiliary variable Phi receives no
 * correction.
 *
 * The correction added to the volume time derivative is:
 *
 * \f{align*}
 *   \delta(\partial_t \Psi) &= -(\partial_t \Psi)_{\mathrm{volume}}
 *                              + (\partial_t \Psi)_{\mathrm{analytic}} \\
 *   \delta(\partial_t \Pi)  &= -(\partial_t \Pi)_{\mathrm{volume}}
 *                              + (\partial_t \Pi)_{\mathrm{analytic}} \\
 *   \delta(\partial_t \Phi_i) &= 0
 * \f}
 *
 * so that after the infrastructure adds the correction, the result at the
 * boundary face is exactly the analytic time derivative.
 */
template <size_t Dim>
class TimeDerivativeDirichlet final : public BoundaryCondition<Dim> {
 public:
  /// \brief What analytic solution/data to prescribe.
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution/data to prescribe.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };

  using options = tmpl::list<AnalyticPrescription>;

  static constexpr Options::String help{
      "TimeDerivativeDirichlet boundary conditions replacing dt Psi and"
      " dt Pi at external boundaries with analytic time derivative values."};

  TimeDerivativeDirichlet() = default;
  TimeDerivativeDirichlet(TimeDerivativeDirichlet&&) = default;
  TimeDerivativeDirichlet& operator=(TimeDerivativeDirichlet&&) = default;
  TimeDerivativeDirichlet(const TimeDerivativeDirichlet&);
  TimeDerivativeDirichlet& operator=(const TimeDerivativeDirichlet&);
  ~TimeDerivativeDirichlet() override = default;

  explicit TimeDerivativeDirichlet(
      std::unique_ptr<evolution::initial_data::InitialData>
          analytic_prescription);

  explicit TimeDerivativeDirichlet(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, TimeDerivativeDirichlet);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::TimeDerivative;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using dg_interior_dt_vars_tags =
      tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>>;
  using dg_gridless_tags = tmpl::list<::Tags::Time>;

  std::optional<std::string> dg_time_derivative(
      gsl::not_null<Scalar<DataVector>*> dt_psi_correction,
      gsl::not_null<Scalar<DataVector>*> dt_pi_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          dt_phi_correction,

      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const Scalar<DataVector>& volume_dt_psi,
      const Scalar<DataVector>& volume_dt_pi,
      double time) const;

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
};
}  // namespace SoScalarWave::BoundaryConditions
