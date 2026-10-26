// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

/// Boundary corrections of the CCZ4 gradient-projection scheme.
namespace Ccz4::GradientProjection::BoundaryCorrections {
/// \brief The treatment of the dissipative part \f$|A_n|\f$ on external
/// boundaries: drop it, apply it once, or apply it twice. The central part
/// \f$A_n\f$ is always applied.
enum class ExternalCorrection : uint8_t { Central, Upwind, DoubledDissipation };

/// The factor multiplying \f$|A_n|\f$ on external boundaries for the given
/// `ExternalCorrection`: 0, 1, or 2.
double external_upwind_factor(ExternalCorrection external_correction);

std::ostream& operator<<(std::ostream& os,
                         ExternalCorrection external_correction);
/*!
 * \brief The full upwind correction of the temporary targets plus a scalar
 * jump penalty and the internal value-jump input of the potential mean.
 *
 * \warning Upwinding here uses the characteristics of the 43-component
 * first-order temporary-target operator on `(w, q_i)`, with `q_i = D_i u`.
 * These generally differ from the 23 second-order characteristics in
 * `Evolution/Systems/Ccz4/FiniteDifference/Characteristics.hpp`: the full
 * first-order propagating amplitudes can also contain tangential derivatives
 * of `u`. The normal-only decomposition in `NormalCharacteristics` is a
 * computational reduction of the full operator, not a replacement by the
 * second-order characteristics. The gradients are still recomputed from `u`,
 * not independently evolved.
 *
 * \details Each element packages the chart fields \f$(w,q_1,q_2,q_3,u)\f$
 * (53 scalars per point) of its face values and its outward Euclidean unit
 * normal covector \f$n\f$: \f$H\f$ and \f$a\f$ are constructed from the
 * physical face values (an `ERROR` off the algebraic-constraint surface, see
 * `Ccz4::GradientProjection::checked_chart_from_physical`), \f$q\f$ is the
 * stored gradient and the other components are the physical fields. At each
 * face point the coefficient state is the chart average
 * \f$u_*=(u^-+u^+)/2\f$ (the arithmetic mean of \f$H\f$ and of the physical
 * \f$\phi,\alpha,\beta\f$) and \f$[Y]=Y^+-Y^-\f$ with \f$Y=(w,q)\f$. With the
 * normal principal matrix \f$A_n(u_*)\f$ (see
 * `Ccz4::GradientProjection::normal_flux`) and its spectral absolute value
 * (see `Ccz4::GradientProjection::NormalCharacteristics`) the target rates
 * receive
 * \f[
 *  Y_t\mathrel{+}=\frac12\mathcal L_f\{A_n+|A_n|
 *  +\sigma_pP_p+\sigma_tP_t+\sigma_gP_g\}[Y],
 * \f]
 * where \f$P_p\f$, \f$P_t\f$ and \f$P_g\f$ are the spectral projectors of
 * \f$A_n\f$ onto its twenty propagating modes, its twenty stationary modes
 * carrying tangential-gradient mismatch and its three stationary
 * \f$\hat\Gamma-b\f$ modes (see
 * `Ccz4::GradientProjection::NormalCharacteristics::
 * apply_upwind_and_penalties`),
 * \f$\sigma_p,\sigma_t,\sigma_g\ge0\f$ are the scalar jump penalties
 * `PropagatingScalarPenalty`, `StationaryGradientScalarPenalty` and
 * `GammaHatMinusBScalarPenalty`, \f$\mathcal L_f\f$ is the strong-form lift
 * with the Euclidean normal magnitude and the normal matrix uses the
 * Gamma-driver parameter `Ccz4::fd::System::f`. Equal penalties reduce the
 * penalty term to \f$\sigma_pI_{43}\f$, the previous componentwise scalar
 * penalty. The corrections are written to the targets of
 * `Ccz4::GradientProjection::Tags::RhsTargets`. On internal faces the
 * potential slots receive \f$\frac12\mathcal L_f(u^+-u^-)\f$, so that after
 * lifting they hold \f$R-S_uu\f$; on external faces they receive nothing.
 * Only the local normal is used; the packaged neighbor normal is ignored.
 *
 * On external faces the exterior state comes from the boundary condition and
 * the scalar jump penalties are omitted. The dissipative part \f$|A_n|\f$ is
 * multiplied by the factor of `ExternalCorrection` (`Central`: 0, `Upwind`:
 * 1, `DoubledDissipation`: 2); the central part \f$A_n\f$ is unchanged.
 *
 * Only the strong formulation is supported. The physical fields receive no
 * correction: their rates are computed from the corrected targets by
 * `FinalizeTimeDerivative`.
 *
 * On the spherical-harmonic side of an internal face whose neighbor
 * connection is not conforming in the block topology (a wedge-shell
 * interface), `dg_filter_boundary_correction` filters the assembled
 * correction before it is lifted, when `ShellBoundaryModesToKill` has a
 * value \f$k\f$. It applies the native hard tensor spherical-harmonic filter
 * \f$I+\Delta\f$, with \f$\Delta\f$ from `ylm::TensorYlm::fill_filter` with
 * Spherepack normalization, \f$k\f$ removed bands and no exponential roll-off,
 * to the Spherepack coefficients of the Cartesian components of each tensor
 * of the velocity and gradient corrections, with no mass or radial weight:
 *
 * - \f$a_t\f$: symmetric rank two (`tnsr::ii` storage), through `stf_decode`
 *   and `stf_encode`, which preserve the trace-free subspace;
 * - \f$K_t\f$, \f$\Theta_t\f$: scalars;
 * - \f$\hat\Gamma_t\f$, \f$b_t\f$, \f$(D_i\phi)_t\f$, \f$(D_i\alpha)_t\f$:
 *   vectors;
 * - \f$(D_iH)_t\f$: rank three, symmetric in the last two indices
 *   (`tnsr::ijj` storage), the trace-free pair through `stf_decode` and
 *   `stf_encode`;
 * - \f$(D_i\beta^j)_t\f$: general rank two (`tnsr::ij` storage), not
 *   symmetrized.
 *
 * The derivative index of the gradients is a spatial tensor index, the
 * chart index is not. The potential-rate correction is never filtered. With
 * zero removed bands the filter still removes the tensor content of total
 * degree above \f$L\f$; without a value the hook does nothing.
 */
class UpwindGradientProjection final : public evolution::BoundaryCorrection {
 public:
  struct ExternalCorrectionOption {
    static std::string name() { return "ExternalCorrection"; }
    using type = ExternalCorrection;
    static constexpr Options::String help = {
        "The treatment of the dissipative part |A_n| on external boundaries: "
        "Central drops it, Upwind applies it once, DoubledDissipation applies "
        "it twice. The central part A_n is always applied."};
  };
  struct PropagatingScalarPenalty {
    using type = double;
    static constexpr Options::String help = {
        "The scalar jump penalty sigma_p on the twenty propagating "
        "first-order characteristic modes, added to |A_n| on internal faces. "
        "External boundaries have no scalar jump penalties."};
    static type lower_bound() { return 0.0; }
  };
  struct StationaryGradientScalarPenalty {
    using type = double;
    static constexpr Options::String help = {
        "The scalar jump penalty sigma_t on the twenty stationary "
        "first-order characteristic modes carrying tangential-gradient "
        "mismatch, on internal faces. The accepted candidate uses the "
        "PropagatingScalarPenalty value."};
    static type lower_bound() { return 0.0; }
  };
  struct GammaHatMinusBScalarPenalty {
    using type = double;
    static constexpr Options::String help = {
        "The scalar jump penalty sigma_g on the three stationary GammaHat-b "
        "first-order characteristic modes, on internal faces. The accepted "
        "candidate removes it (0); setting all three penalties equal "
        "recovers the previous componentwise scalar penalty."};
    static type lower_bound() { return 0.0; }
  };
  struct ShellBoundaryModesToKill {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "The number of top angular bands removed by the native tensor "
        "spherical-harmonic filter applied to the velocity and gradient "
        "corrections on the spherical-harmonic side of a wedge-shell "
        "interface before they are lifted. 0 is not a bypass: it still "
        "removes the tensor content of total degree above L. 'None' disables "
        "this filter. It must not exceed L of a filtered face."};
  };

  using options =
      tmpl::list<ExternalCorrectionOption, PropagatingScalarPenalty,
                 StationaryGradientScalarPenalty, GammaHatMinusBScalarPenalty,
                 ShellBoundaryModesToKill>;
  static constexpr Options::String help = {
      "Full first-order characteristic upwind correction of the CCZ4 "
      "gradient-projection targets with group scalar jump penalties on "
      "internal faces only."};

  UpwindGradientProjection() = default;
  UpwindGradientProjection(ExternalCorrection external_correction,
                           double propagating_scalar_penalty,
                           double stationary_gradient_scalar_penalty,
                           double gamma_hat_minus_b_scalar_penalty,
                           std::optional<size_t> shell_boundary_modes_to_kill);
  UpwindGradientProjection(const UpwindGradientProjection&) = default;
  UpwindGradientProjection& operator=(const UpwindGradientProjection&) =
      default;
  UpwindGradientProjection(UpwindGradientProjection&&) = default;
  UpwindGradientProjection& operator=(UpwindGradientProjection&&) = default;
  ~UpwindGradientProjection() override = default;

  /// \cond
  explicit UpwindGradientProjection(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(UpwindGradientProjection);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  ExternalCorrection external_correction() const {
    return external_correction_;
  }
  double propagating_scalar_penalty() const {
    return propagating_scalar_penalty_;
  }
  double stationary_gradient_scalar_penalty() const {
    return stationary_gradient_scalar_penalty_;
  }
  double gamma_hat_minus_b_scalar_penalty() const {
    return gamma_hat_minus_b_scalar_penalty_;
  }
  const std::optional<size_t>& shell_boundary_modes_to_kill() const {
    return shell_boundary_modes_to_kill_;
  }

  using dg_package_field_tags =
      tmpl::list<Tags::PackagedVelocities, Tags::PackagedGradients,
                 Tags::PackagedPotentials,
                 ::Ccz4::Tags::NormalCovector<DataVector, 3>>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;
  using dg_auxiliary_package_field_tags = tmpl::list<>;
  using dg_auxiliary_package_data_temporary_tags = tmpl::list<>;
  using dg_auxiliary_package_data_volume_tags = tmpl::list<>;
  using dg_auxiliary_boundary_terms_volume_tags = tmpl::list<>;

  /// Packages the chart fields of the face values and the normal. Returns
  /// the largest characteristic speed magnitude at the face state.
  double dg_package_data(
      gsl::not_null<Tags::PackagedVelocities::type*> packaged_w,
      gsl::not_null<Tags::PackagedGradients::type*> packaged_q,
      gsl::not_null<Tags::PackagedPotentials::type*> packaged_u,
      gsl::not_null<tnsr::i<DataVector, 3>*> packaged_normal_covector,
      const tnsr::ii<DataVector, 3>& conformal_metric,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::ii<DataVector, 3>& a_tilde,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const Scalar<DataVector>& theta, const tnsr::I<DataVector, 3>& gamma_hat,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
      const tnsr::I<DataVector, 3>& b, const Tags::PotentialGradients::type& q,
      const tnsr::i<DataVector, 3>& normal_covector,
      const std::optional<tnsr::I<DataVector, 3>>& /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
      const Direction<3>& /*face_direction*/) const;

  template <bool ForExternalBoundary>
  void dg_boundary_terms(
      gsl::not_null<Tags::VelocityTarget::type*> w_correction,
      gsl::not_null<Tags::GradientTarget::type*> q_correction,
      gsl::not_null<Tags::PotentialRateTarget::type*> u_correction,
      const Tags::PackagedVelocities::type& w_int,
      const Tags::PackagedGradients::type& q_int,
      const Tags::PackagedPotentials::type& u_int,
      const tnsr::i<DataVector, 3>& normal_covector_int,
      const Tags::PackagedVelocities::type& w_ext,
      const Tags::PackagedGradients::type& q_ext,
      const Tags::PackagedPotentials::type& u_ext,
      const tnsr::i<DataVector, 3>& /*normal_covector_ext*/,
      dg::Formulation dg_formulation) const;

  /*!
   * \brief Filter the assembled correction of an internal face before it is
   * lifted, see the class documentation.
   *
   * \details Does nothing unless `ShellBoundaryModesToKill` has a value,
   * `has_nonconforming_block_connection` is true and both bases of
   * `face_mesh` are spherical harmonics. It is an `ERROR` if it would act
   * with more removed bands than \f$L\f$ of the face, or on a face mesh that
   * is not \f$(L+1,2L+1)\f$ with Gauss and equiangular quadratures or that
   * does not match the number of points of `corrections`.
   */
  void dg_filter_boundary_correction(
      gsl::not_null<Tags::RhsTargets::type*> corrections,
      const Mesh<2>& face_mesh, bool has_nonconforming_block_connection) const;

 private:
  ExternalCorrection external_correction_ = ExternalCorrection::Upwind;
  double propagating_scalar_penalty_ =
      std::numeric_limits<double>::signaling_NaN();
  double stationary_gradient_scalar_penalty_ =
      std::numeric_limits<double>::signaling_NaN();
  double gamma_hat_minus_b_scalar_penalty_ =
      std::numeric_limits<double>::signaling_NaN();
  std::optional<size_t> shell_boundary_modes_to_kill_{};
};
}  // namespace Ccz4::GradientProjection::BoundaryCorrections

template <>
struct Options::create_from_yaml<
    Ccz4::GradientProjection::BoundaryCorrections::ExternalCorrection> {
  template <typename Metavariables>
  static Ccz4::GradientProjection::BoundaryCorrections::ExternalCorrection
  create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
Ccz4::GradientProjection::BoundaryCorrections::ExternalCorrection
Options::create_from_yaml<
    Ccz4::GradientProjection::BoundaryCorrections::ExternalCorrection>::
    create<void>(const Options::Option& options);
