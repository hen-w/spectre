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
 * \f$[[\Psi]]^i=n^i\left(\Psi^\text{int}-\Psi^\text{ext}\right)\f$
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
 * \f}
 *
 * is the central flux.
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
 * (f_\Phi^i)^* = \{\{ f_\Phi^i \}\}
 *   + \frac{\tau \lambda_\text{max}}{2} [[ \Pi ]]^i .
 * \f}
 *
 * Here \f$\lambda_\text{max}\f$ is the largest characteristic-speed magnitude
 * over the modes and both sides of the interface, and the input-file option
 * \f$\tau\f$ is a factor multiplying it (\f$\tau = 1\f$ gives the standard
 * Lax-Friedrichs coefficient). On a static mesh all speeds are \f$0, \pm 1\f$,
 * so \f$\lambda_\text{max} = 1\f$.
 *
 * ### Moving meshes
 *
 * On a moving mesh the underlying equations are the grid-frame ones,
 *
 * \f{align*}{
 * \partial_t \Psi =& -\Pi + v^i \Phi_i, \\
 * \partial_t \Pi =& -\partial_i \Phi^i + v^i \partial_i \Pi,
 * \f}
 *
 * where \f$v^i\f$ is the mesh velocity and the LDG prescription has already
 * replaced \f$\partial_i\Psi \to \Phi_i\f$ in the \f$\Psi\f$ equation. The
 * LDG construction applies to these equations as given; it involves no
 * notion of the mesh motion itself.
 *
 * For \f$\Pi\f$, the interface coupling is the numerical flux above applied
 * to the grid-frame flux \f$f_\Phi^i = \Phi^i - v^i \Pi\f$:
 *
 * \f{align*}{
 * (f_\Phi^i)^* = \{\{ \Phi^i - v^i \Pi \}\}
 *   + \frac{\tau \lambda_\text{max}}{2} [[ \Pi ]]^i .
 * \f}
 *
 * The grid-frame characteristic speeds along a normal \f$n^i\f$ are
 * \f$\{-n_iv^i,\, 1 - n_iv^i,\, -1 - n_iv^i\}\f$ (evaluating along
 * \f$-n^i\f$ flips all signs), so the largest magnitude over the modes and
 * both sides of the interface is
 * \f$\lambda_\text{max} = 1 + |n_iv^i|\f$; since the mesh velocity is
 * continuous across the interface, the interior \f$n_iv^i\f$ covers both
 * sides.
 *
 * Each side packages \f$n_iv^i\f$ with its own outward normal, so the
 * \f$-v^i\Pi\f$ part of the central average appears in the \f$\Pi\f$
 * boundary correction as
 * \f$+\tfrac12[(n_iv^i)^\text{int}\Pi^\text{int}
 * + (n_iv^i)^\text{ext}\Pi^\text{ext}]\f$.
 *
 * For \f$\Psi\f$, the term \f$v^i\Phi_i\f$ is implemented as the volume
 * term \f$v^i\partial_i\Psi\f$ (added with the raw spectral derivative by
 * the generic moving-mesh volume machinery) plus the lifted face term
 *
 * \f{align*}{
 * G_\Psi = \tfrac{1}{2}\left[(n_iv^i)^\text{int}\,\Psi^\text{int}
 *   + (n_iv^i)^\text{ext}\,\Psi^\text{ext}\right],
 * \f}
 *
 * which is \f$v^i\f$ contracted with the auxiliary pass's boundary
 * correction, so volume term plus lift equal \f$v^i\Phi_i\f$ (exactly at
 * the Gauss-Lobatto boundary nodes carrying the lift). Both face terms
 * vanish identically on a static mesh and on continuous data.
 */
template <size_t Dim>
class LaxFriedrichs final : public evolution::BoundaryCorrection {
 public:
  struct Tau {
    using type = double;
    static constexpr Options::String help = {
        "Factor multiplying the largest characteristic-speed magnitude at the "
        "interface to form the penalty coefficient of the physical numerical "
        "flux. A value of 1.0 gives the standard Lax-Friedrichs coefficient."};
  };

  using options = tmpl::list<Tau>;
  static constexpr Options::String help = {
      "Boundary correction to the LDG method using the LaxFriedrichs numerical "
      "flux."};

  LaxFriedrichs() = default;
  explicit LaxFriedrichs(double tau);
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

  using dg_package_field_tags =
      tmpl::list<Tags::Pi, Tags::NormalDotPhi, Tags::Psi,
                 Tags::NormalDotMeshVelocity>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;

  double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_pi,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_psi,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_mesh_velocity,

      const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) const;

  void dg_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,

      const Scalar<DataVector>& pi_int,
      const Scalar<DataVector>& normal_dot_phi_int,
      const Scalar<DataVector>& psi_int,
      const Scalar<DataVector>& normal_dot_mesh_velocity_int,

      const Scalar<DataVector>& pi_ext,
      const Scalar<DataVector>& normal_dot_phi_ext,
      const Scalar<DataVector>& psi_ext,
      const Scalar<DataVector>& normal_dot_mesh_velocity_ext,

      dg::Formulation /*dg_formulation*/) const;

  using dg_auxiliary_package_field_tags = tmpl::list<Tags::PsiTimesNormal<Dim>>;
  using dg_auxiliary_package_data_temporary_tags = tmpl::list<>;
  using dg_auxiliary_package_data_volume_tags = tmpl::list<>;
  using dg_auxiliary_boundary_terms_volume_tags = tmpl::list<>;

  double dg_auxiliary_package_data(
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

      const tnsr::i<DataVector, Dim, Frame::Inertial>& psi_times_normal_int,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& psi_times_normal_ext,

      dg::Formulation /*dg_formulation*/) const;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const LaxFriedrichs<LocalDim>& lhs,
                         const LaxFriedrichs<LocalDim>& rhs);

  double tau_ = std::numeric_limits<double>::signaling_NaN();
};

template <size_t Dim>
bool operator!=(const LaxFriedrichs<Dim>& lhs, const LaxFriedrichs<Dim>& rhs);
}  // namespace SecondOrderScalarWave::BoundaryCorrections
