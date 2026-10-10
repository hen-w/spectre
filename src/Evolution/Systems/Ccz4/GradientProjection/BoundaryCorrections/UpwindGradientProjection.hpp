// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
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

/// Boundary corrections of the CCZ4 gradient-projection scheme.
namespace Ccz4::GradientProjection::BoundaryCorrections {
/*!
 * \brief The full upwind correction of the temporary targets plus a unit
 * scalar jump penalty and the internal value-jump input of the potential
 * mean.
 *
 * \details Each element packages its chart fields \f$(w,q_1,q_2,q_3,u)\f$
 * (53 scalars per point) and its outward Euclidean unit normal covector
 * \f$n\f$. At each face point the coefficient state is the chart average
 * \f$u_*=(u^-+u^+)/2\f$ and \f$[Y]=Y^+-Y^-\f$ with \f$Y=(w,q)\f$. With the
 * normal principal matrix \f$A_n(u_*)\f$ (see
 * `Ccz4::GradientProjection::normal_flux`) and its spectral absolute value
 * (see `Ccz4::GradientProjection::NormalCharacteristics`) the target rates
 * receive
 * \f[
 *  Y_t\mathrel{+}=\frac12\mathcal L_f\{A_n+|A_n|+\sigma I_{43}\}[Y],\qquad
 *  \sigma=1,
 * \f]
 * where \f$\mathcal L_f\f$ is the strong-form lift with the Euclidean normal
 * magnitude and the normal matrix uses the Gamma-driver parameter
 * `Ccz4::fd::System::f`. On internal faces the potential slots receive
 * \f$\frac12\mathcal L_f(u^+-u^-)\f$, so that after lifting they hold
 * \f$R-S_uu\f$; on external faces they receive nothing.
 *
 * On external faces the exterior state comes from the boundary condition and
 * the dissipative part \f$|A_n|+\sigma I\f$ is multiplied by the option
 * `ExternalDissipationFactor` (0: central, 1: standard, 2: doubled radial
 * dissipation); the central part \f$A_n\f$ is unchanged.
 *
 * Only the strong formulation is supported. The physical fields receive no
 * correction: their rates are computed from the corrected targets by
 * `FinalizeTimeDerivative`.
 */
class UpwindGradientProjection final : public evolution::BoundaryCorrection {
 public:
  struct ExternalDissipationFactor {
    using type = double;
    static constexpr Options::String help = {
        "The factor multiplying |A_n| + sigma I on external boundaries: 0 is "
        "central, 1 standard, 2 doubled dissipation."};
    static type lower_bound() { return 0.0; }
  };

  using options = tmpl::list<ExternalDissipationFactor>;
  static constexpr Options::String help = {
      "Full upwind correction of the CCZ4 gradient-projection targets with a "
      "unit scalar jump penalty."};

  UpwindGradientProjection() = default;
  explicit UpwindGradientProjection(double external_dissipation_factor);
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

  double external_dissipation_factor() const {
    return external_dissipation_factor_;
  }

  using dg_package_field_tags =
      tmpl::list<Tags::ChartW, Tags::ChartQ, Tags::ChartU,
                 ::Ccz4::Tags::NormalCovector<DataVector, 3>>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;
  using dg_auxiliary_package_field_tags = tmpl::list<>;
  using dg_auxiliary_package_data_temporary_tags = tmpl::list<>;
  using dg_auxiliary_package_data_volume_tags = tmpl::list<>;
  using dg_auxiliary_boundary_terms_volume_tags = tmpl::list<>;

  /// Packages the chart fields and the normal. Returns the largest
  /// characteristic speed magnitude at the interior face state.
  double dg_package_data(
      gsl::not_null<Tags::ChartW::type*> packaged_w,
      gsl::not_null<Tags::ChartQ::type*> packaged_q,
      gsl::not_null<Tags::ChartU::type*> packaged_u,
      gsl::not_null<tnsr::i<DataVector, 3>*> packaged_normal_covector,
      const Tags::ChartW::type& w, const Tags::ChartQ::type& q,
      const Tags::ChartU::type& u,
      const tnsr::ii<DataVector, 3>& /*conformal_metric*/,
      const Scalar<DataVector>& /*conformal_factor*/,
      const tnsr::ii<DataVector, 3>& /*a_tilde*/,
      const Scalar<DataVector>& /*trace_extrinsic_curvature*/,
      const Scalar<DataVector>& /*theta*/,
      const tnsr::I<DataVector, 3>& /*gamma_hat*/,
      const Scalar<DataVector>& /*lapse*/,
      const tnsr::I<DataVector, 3>& /*shift*/,
      const tnsr::I<DataVector, 3>& /*b*/,
      const tnsr::i<DataVector, 3>& normal_covector,
      const std::optional<tnsr::I<DataVector, 3>>& /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
      const Direction<3>& /*face_direction*/) const;

  template <bool ForExternalBoundary>
  void dg_boundary_terms(
      gsl::not_null<Tags::ChartW::type*> w_correction,
      gsl::not_null<Tags::ChartQ::type*> q_correction,
      gsl::not_null<Tags::ChartU::type*> u_correction,
      gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric_correction,
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde_correction,
      gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature_correction,
      gsl::not_null<Scalar<DataVector>*> theta_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat_correction,
      gsl::not_null<Scalar<DataVector>*> lapse_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*> b_correction,
      const Tags::ChartW::type& w_int, const Tags::ChartQ::type& q_int,
      const Tags::ChartU::type& u_int,
      const tnsr::i<DataVector, 3>& normal_covector_int,
      const Tags::ChartW::type& w_ext, const Tags::ChartQ::type& q_ext,
      const Tags::ChartU::type& u_ext,
      const tnsr::i<DataVector, 3>& /*normal_covector_ext*/,
      dg::Formulation dg_formulation) const;

 private:
  double external_dissipation_factor_ =
      std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Ccz4::GradientProjection::BoundaryCorrections
