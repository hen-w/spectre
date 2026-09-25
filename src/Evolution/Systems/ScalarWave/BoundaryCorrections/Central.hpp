// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Direction;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ScalarWave::BoundaryCorrections {
/*!
 * \brief Computes the pure central-flux boundary correction for the
 * first-order scalar wave system (experimental, no constraint damping).
 *
 * \details This is a vanilla central numerical flux for the plain first-order
 * scalar wave system
 *
 * \f{align*}{
 *   \partial_t \Psi &= -\Pi, \\
 *   \partial_t \Pi &= -\partial_i \Phi^i, \\
 *   \partial_t \Phi_i &= -\partial_i \Pi,
 * \f}
 *
 * with **no** constraint-damping (\f$\gamma_2\f$) terms. It exists to isolate,
 * as a discriminating experiment, whether interface instabilities seen with
 * central auxiliary fluxes are masked by upwind dissipation.
 *
 * Each side packages, using its own outward unit normal \f$n_i\f$,
 *
 * \f{align*}{
 *   \Pi, \quad n^i \Phi_i, \quad n_i \Pi .
 * \f}
 *
 * The strong-form central flux \f$F^* = (F^{\mathrm{int}} +
 * F^{\mathrm{ext}})/2\f$ gives the boundary corrections
 *
 * \f{align*}{
 *   D_{\Psi} &= 0, \\
 *   D_{\Pi} &= -\frac{1}{2}\left(
 *     (n^i\Phi_i)^{\mathrm{int}} + (n^i\Phi_i)^{\mathrm{ext}}\right), \\
 *   D_{\Phi_i} &= -\frac{1}{2}\left(
 *     (n_i\Pi)^{\mathrm{int}} + (n_i\Pi)^{\mathrm{ext}}\right),
 * \f}
 *
 * where the external quantities were computed with \f$n^{\mathrm{ext}}_i =
 * -n^{\mathrm{int}}_i\f$. The overall minus sign accounts for the minus sign in
 * `LiftFlux.hpp`. These are the \f$\tau_1=\tau_2=0\f$ limit of
 * `SoScalarWave::BoundaryCorrections::LaxFriedrichs`, whose sign structure is
 * mirrored here.
 */
template <size_t Dim>
class Central final : public evolution::BoundaryCorrection {
 private:
  struct Pi : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct NormalDotPhi : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct NormalTimesPi : db::SimpleTag {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  };

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the pure central-flux boundary correction for the first-order "
      "scalar wave system, without any constraint-damping (gamma2) terms. This "
      "is an experimental correction."};

  Central() = default;
  Central(const Central&) = default;
  Central& operator=(const Central&) = default;
  Central(Central&&) = default;
  Central& operator=(Central&&) = default;
  ~Central() override = default;

  /// \cond
  explicit Central(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Central);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  using dg_package_field_tags = tmpl::list<Pi, NormalDotPhi, NormalTimesPi>;
  // ScalarWave's shared boundary conditions (e.g. DirichletAnalytic,
  // SphericalRadiation) always produce a ConstraintGamma2 field for the ghost
  // state. We accept it as a temporary to satisfy that interface but ignore it
  // entirely: this is the vanilla central flux with no constraint damping.
  using dg_package_data_temporary_tags = tmpl::list<Tags::ConstraintGamma2>;
  using dg_package_data_volume_tags = tmpl::list<>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;
  // ScalarWave is not an LDG system and has no auxiliary pass. These empty
  // lists satisfy the type-level requirements of the LDG-aware
  // ComputeTimeDerivative and ApplyBoundaryCorrections actions, which
  // transform over every registered boundary correction's auxiliary tag
  // aliases; the auxiliary code paths are never instantiated for this system.
  using dg_auxiliary_package_field_tags = tmpl::list<>;
  using dg_auxiliary_package_data_temporary_tags = tmpl::list<>;
  using dg_auxiliary_package_data_volume_tags = tmpl::list<>;
  using dg_auxiliary_boundary_terms_volume_tags = tmpl::list<>;

  double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_pi,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_phi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_normal_times_pi,

      const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,

      const Scalar<DataVector>& constraint_gamma2,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const Direction<Dim>& face_direction) const;

  /// The LDG-aware actions invoke `dg_boundary_terms` as a member template
  /// with a `ForExternalBoundary` flag. ScalarWave has no auxiliary pass and
  /// uses the same boundary terms on internal and external faces, so both
  /// instantiations forward to the untemplated implementation.
  template <bool ForExternalBoundary = false, typename... Args>
  void dg_boundary_terms(Args&&... args) const {
    dg_boundary_terms_impl(std::forward<Args>(args)...);
  }

  void dg_boundary_terms_impl(
      gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          phi_boundary_correction,

      const Scalar<DataVector>& pi_int,
      const Scalar<DataVector>& normal_dot_phi_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_times_pi_int,

      const Scalar<DataVector>& pi_ext,
      const Scalar<DataVector>& normal_dot_phi_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_times_pi_ext,

      dg::Formulation dg_formulation) const;
};
}  // namespace ScalarWave::BoundaryCorrections
