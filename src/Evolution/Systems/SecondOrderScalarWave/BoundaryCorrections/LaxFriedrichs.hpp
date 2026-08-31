// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace SecondOrderScalarWave::BoundaryCorrections {
/*!
 * \brief A Lax-Friedrichs-type boundary correction for the second-order
 * scalar wave system, providing the numerical fluxes for the local
 * discontinuous Galerkin (LDG) scheme.
 *
 * Each evaluation of the system's time derivative proceeds in two passes: the
 * auxiliary pass computes \f$\Phi_i = \partial_i \Psi\f$, then the physical
 * pass evolves \f$\partial_t \Pi = -\partial_i \Phi^i\f$ and
 * \f$\partial_t \Psi = -\Pi\f$. This class supplies the numerical flux for
 * each pass: the `dg_auxiliary_*` interface couples neighboring elements in
 * the auxiliary pass, the standard `dg_*` interface in the physical pass.
 *
 * Below, \f$n^i\f$ denotes the interior element's outward-directed unit face
 * normal. \f$\{\{\Psi\}\}\f$ denotes the central average of \f$\Psi\f$ across
 * the interface, and
 * \f$[[\Psi]]=n^\text{int}\Psi^\text{int}+n^\text{ext}\Psi^\text{ext}\f$
 * denotes the jump of \f$\Psi\f$ across the interface.
 *
 * ### Auxiliary pass
 *
 * Integrating \f$\Phi_i = \partial_i \Psi\f$ against a test
 * function \f$l\f$ by parts twice over an element \f$K\f$ gives
 *
 * \f{align*}{
 * \int_K \Phi_i l = \int_K (\partial_i \Psi)\, l
 *   + \oint_{\partial K} l n_i \left(\Psi^* - \Psi\right),
 * \f}
 *
 * where
 *
 * \f{align*}{
 * \Psi^* = \{\{ \Psi \}\}
 *   - \frac{\tau_2}{2} [[ \Psi ]],
 * \f}
 *
 * \f$\Psi^*\f$ is generalized to more than 1D naively by interpreting
 * \f$\Psi^*n_i\f$ as the sum of a scalar-vector product from the central
 * average term and a dot product from the jump term.
 *
 * ### Physical pass
 *
 * Integrating \f$\partial_t \Pi = -\partial_i \Phi^i\f$ by parts twice gives
 *
 * \f{align*}{
 * \int_K (\partial_t \Pi)\, l = -\int_K (\partial_i f_\Phi^i)\, l
 *   - \oint_{\partial K}
 *   \left[(f_\Phi^i)^* - f_\Phi^i\right] l n_i ,
 * \f}
 *
 * where
 *
 * \f{align*}{
 * f_\Phi^i = \Phi^i,
 * \f}
 *
 * \f{align*}{
 * (f_\Phi^i)^* = \{\{ f_\Phi^i \}\} + \frac{\tau_1}{2} [[ \Pi ]]
 * \f}
 *
 * \f$\partial_t \Psi = -\Pi\f$ contains no spatial derivative and receives
 * no boundary correction.
 */
template <size_t Dim>
class LaxFriedrichs final : public evolution::BoundaryCorrection {
 public:
  struct Tau1 {
    using type = double;
    static constexpr Options::String help = {
        "The penalty parameter tau1 for the physical numerical flux."};
  };
  struct Tau2 {
    using type = double;
    static constexpr Options::String help = {
        "The penalty parameter tau2 for the auxiliary numerical flux"};
  };

  using options = tmpl::list<Tau1, Tau2>;
  static constexpr Options::String help = {
      "Boundary correction to the LDG method using the LaxFriedrichs numerical "
      "flux."};

  LaxFriedrichs() = default;
  explicit LaxFriedrichs(double tau1, double tau2);
  LaxFriedrichs(const LaxFriedrichs&) = default;
  LaxFriedrichs& operator=(const LaxFriedrichs&) = default;
  LaxFriedrichs(LaxFriedrichs&&) = default;
  LaxFriedrichs& operator=(LaxFriedrichs&&) = default;
  ~LaxFriedrichs() override = default;

  /// \cond
  explicit LaxFriedrichs(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(LaxFriedrichs);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  using dg_package_field_tags = tmpl::list<Tags::Pi, Tags::NormalDotPhi>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;

  double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_pi,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_phi,

      const Scalar<DataVector>& /*psi*/, const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/)
      const;

  void dg_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,

      const Scalar<DataVector>& pi_int,
      const Scalar<DataVector>& normal_dot_phi_int,

      const Scalar<DataVector>& pi_ext,
      const Scalar<DataVector>& normal_dot_phi_ext,

      dg::Formulation /*dg_formulation*/) const;

  using dg_auxiliary_package_field_tags =
      tmpl::list<Tags::Psi, Tags::PsiTimesNormal<Dim>>;
  using dg_auxiliary_package_data_temporary_tags = tmpl::list<>;
  using dg_auxiliary_package_data_volume_tags = tmpl::list<>;
  using dg_auxiliary_boundary_terms_volume_tags = tmpl::list<>;

  double dg_auxiliary_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_psi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          psi_times_normal,

      const Scalar<DataVector>& psi, const Scalar<DataVector>& /*pi*/,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/)
      const;

  void dg_auxiliary_boundary_terms(
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          phi_boundary_correction,

      const Scalar<DataVector>& psi_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& psi_times_normal_int,

      const Scalar<DataVector>& psi_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& psi_times_normal_ext,

      dg::Formulation /*dg_formulation*/) const;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const LaxFriedrichs<LocalDim>& lhs,
                         const LaxFriedrichs<LocalDim>& rhs);

  double tau1_ = std::numeric_limits<double>::signaling_NaN();
  double tau2_ = std::numeric_limits<double>::signaling_NaN();
};

template <size_t Dim>
bool operator!=(const LaxFriedrichs<Dim>& lhs, const LaxFriedrichs<Dim>& rhs);
}  // namespace SecondOrderScalarWave::BoundaryCorrections
