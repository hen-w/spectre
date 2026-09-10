// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
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

namespace Ccz4::BoundaryCorrections {
/*!
 * \brief An experimental LDG boundary correction with shift-gradient-modified
 * Lax-Friedrichs dissipative penalties.
 *
 * \warning This boundary correction is experimental. It is identical to
 * `Ccz4::BoundaryCorrections::LaxFriedrichs` in every respect (volume
 * equations, central flux expressions, packaging, auxiliary reconstruction,
 * and the auxiliary boundary corrections are all unchanged), except that the
 * dissipative Lax-Friedrichs jumps of the physical variables
 * \f$\tilde{A}_{ij}\f$, \f$K\f$, and \f$\Theta\f$ are shifted by combinations
 * of the auxiliary variable \f$B_i{}^k = \partial_i \beta^k\f$
 * (`Ccz4::Tags::FieldB`, not `Ccz4::Tags::AuxiliaryShiftB`).
 *
 * On each side of the interface separately define the trace
 *
 * \f{align}
 *   T &= B_k{}^k,
 * \f}
 *
 * where \f$T\f$ is the trace of `Ccz4::Tags::FieldB`, i.e. a mixed spatial
 * derivative of the shift, and the trace-free symmetric combination
 *
 * \f{align}
 *   Q_{ij} &= \frac{1}{2}\left(\bar{\gamma}_{jk} B_i{}^k
 *             + \bar{\gamma}_{ik} B_j{}^k\right)
 *             - \frac{1}{3}\bar{\gamma}_{ij} T,
 * \f}
 *
 * where the upper index of \f$B_i{}^k\f$ is lowered with each side's own
 * conformal spatial metric \f$\bar{\gamma}_{ij}\f$
 * (`Ccz4::Tags::ConformalMetric`). The dissipative jumps used in the
 * Lax-Friedrichs penalty are then replaced by
 *
 * \f{align}
 *   [\tilde{A}_{ij}] &\to [\tilde{A}_{ij} - Q_{ij}/\alpha], \\
 *   [K] &\to [K - T/\alpha], \\
 *   [\Theta] &\to [\Theta - T/(2\alpha)],
 * \f}
 *
 * where the jump \f$[\cdot]\f$ is the exterior minus interior value, and the
 * overall sign, the \f$1/2\f$ factor, and the `UseCentralFluxAtBoundary`
 * override of the effective \f$\tau_1\f$ are all identical to
 * `Ccz4::BoundaryCorrections::LaxFriedrichs`. Equivalently, relative to the
 * Lax-Friedrichs boundary corrections, the physical corrections for
 * \f$\tilde{A}_{ij}\f$, \f$K\f$, and \f$\Theta\f$ pick up the extra terms
 * \f$+\tfrac{1}{2}\tau_1 [Q_{ij}/\alpha]\f$,
 * \f$+\tfrac{1}{2}\tau_1 [T/\alpha]\f$, and
 * \f$+\tfrac{1}{4}\tau_1 [T/\alpha]\f$, respectively. The lapse is positive
 * and evaluated on each side separately, not averaged across the interface.
 * The full (both normal and tangential) components of \f$B_i{}^k\f$ enter;
 * it is not projected onto the interface normal. No other physical penalty
 * and none of the central physical
 * fluxes are modified: \f$\tilde{A}_{ij}\f$, \f$K\f$, and \f$\Theta\f$
 * appearing inside the flux-dot-normal expressions are untouched.
 *
 * The lapse factors follow from the advective time derivatives of the
 * conformal metric, conformal factor, and lapse. They reduce to the original
 * shift-gradient modification at unit lapse. This motivation does not
 * establish stability on arbitrary backgrounds.
 *
 * The motivating semidiscrete spectral result excludes exponentially growing
 * eigenmodes for the frozen unit-lapse Minkowski system with the analyzed
 * gamma-driver gauge (driver coefficient one, `shifting_shift=false`),
 * algebraic trace constraints enforced, a static periodic Cartesian
 * tensor-product LGL mesh, \f$\tau_1 \geq 0\f$, and \f$\tau_2 = 1\f$. Damping
 * terms and filters are excluded from that analysis. This is not a
 * resolution-uniform energy bound or a fully discrete stability theorem. The
 * side-local metric lowering above extends the linearized prescription
 * algebraically; nonlinear, curvilinear, moving-mesh, external-boundary, and
 * DG-subcell-interface stability are not established. Arbitrary `Tau2` remains
 * available, as in LaxFriedrichs.
 */
template <size_t Dim>
class ProposedFlux final : public evolution::BoundaryCorrection {
 public:
  struct Tau1 {
    using type = double;
    static constexpr Options::String help = {
        "The penalty parameter tau1 for the Lax-Friedrichs numerical flux."};
  };
  struct Tau2 {
    using type = double;
    static constexpr Options::String help = {
        "The penalty parameter tau2 for the auxiliary numerical flux"};
  };
  struct UseCentralFluxAtBoundary {
    using type = bool;
    static constexpr Options::String help = {
        "If true, use central flux (tau1=0, tau2=1) at external boundaries. "
        "If false, use the same tau1/tau2 as interior faces."};
    static constexpr type default_value = true;
  };

  using options = tmpl::list<Tau1, Tau2, UseCentralFluxAtBoundary>;
  static constexpr Options::String help = {
      "An experimental LDG boundary correction with shift-gradient-modified "
      "Lax-Friedrichs penalties."};

  ProposedFlux() = default;
  explicit ProposedFlux(double tau1, double tau2,
                        bool use_central_flux_at_boundary = true);
  ProposedFlux(const ProposedFlux&) = default;
  ProposedFlux& operator=(const ProposedFlux&) = default;
  ProposedFlux(ProposedFlux&&) = default;
  ProposedFlux& operator=(ProposedFlux&&) = default;
  ~ProposedFlux() override = default;

  /// \cond
  explicit ProposedFlux(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ProposedFlux);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  using dg_package_field_tags = tmpl::list<
      // evolved variables
      ::Ccz4::Tags::ConformalMetric<DataVector, 3>,
      ::Ccz4::Tags::ConformalFactor<DataVector>,
      ::Ccz4::Tags::ATilde<DataVector, 3>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Ccz4::Tags::Theta<DataVector>, ::Ccz4::Tags::GammaHat<DataVector, 3>,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>,
      // auxiliary reduction variables
      ::Ccz4::Tags::FieldA<DataVector, 3>, ::Ccz4::Tags::FieldB<DataVector, 3>,
      ::Ccz4::Tags::FieldD<DataVector, 3>, ::Ccz4::Tags::FieldP<DataVector, 3>,
      // normal covector
      ::Ccz4::Tags::NormalCovector<DataVector, 3>>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;

  double dg_package_data(
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          packaged_conformal_metric,
      gsl::not_null<Scalar<DataVector>*> packaged_conformal_factor,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          packaged_a_tilde,
      gsl::not_null<Scalar<DataVector>*> packaged_trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> packaged_theta,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          packaged_gamma_hat,
      gsl::not_null<Scalar<DataVector>*> packaged_lapse,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> packaged_shift,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          packaged_auxiliary_shift_b,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_field_a,
      gsl::not_null<tnsr::iJ<DataVector, Dim, Frame::Inertial>*>
          packaged_field_b,
      gsl::not_null<tnsr::ijj<DataVector, Dim, Frame::Inertial>*>
          packaged_field_d,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_field_p,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_normal_covector,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& a_tilde,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const Scalar<DataVector>& theta,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& gamma_hat,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_shift_b,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b,
      const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>&
      /*boundary_conformal_metric*/,
      const Scalar<DataVector>& /*boundary_conformal_factor*/,
      const Scalar<DataVector>& /*boundary_lapse*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*boundary_shift*/,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
      const Direction<Dim>& /*face_direction*/) const;

  template <bool ForExternalBoundary = false>
  void dg_boundary_terms(
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          conformal_metric_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> conformal_factor_boundary_correction,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          a_tilde_boundary_correction,
      gsl::not_null<Scalar<DataVector>*>
          trace_extrinsic_curvature_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> theta_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          gamma_hat_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> lapse_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          shift_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          auxiliary_shift_b_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          field_a_boundary_correction,
      gsl::not_null<tnsr::iJ<DataVector, Dim, Frame::Inertial>*>
          field_b_boundary_correction,
      gsl::not_null<tnsr::ijj<DataVector, Dim, Frame::Inertial>*>
          field_d_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          field_p_boundary_correction,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          boundary_conformal_metric_boundary_correction,
      gsl::not_null<Scalar<DataVector>*>
          boundary_conformal_factor_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> boundary_lapse_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          boundary_shift_boundary_correction,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric_int,
      const Scalar<DataVector>& conformal_factor_int,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& a_tilde_int,
      const Scalar<DataVector>& trace_extrinsic_curvature_int,
      const Scalar<DataVector>& theta_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& gamma_hat_int,
      const Scalar<DataVector>& lapse_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_shift_b_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a_int,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b_int,
      const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector_int,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric_ext,
      const Scalar<DataVector>& conformal_factor_ext,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& a_tilde_ext,
      const Scalar<DataVector>& trace_extrinsic_curvature_ext,
      const Scalar<DataVector>& theta_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& gamma_hat_ext,
      const Scalar<DataVector>& lapse_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_shift_b_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a_ext,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b_ext,
      const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector_ext,

      dg::Formulation /*dg_formulation*/) const;

  using dg_auxiliary_package_field_tags =
      tmpl::list<Ccz4::Tags::ConformalMetric<DataVector, 3>,
                 Ccz4::Tags::ConformalFactor<DataVector>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 Ccz4::Tags::NormalCovector<DataVector, 3>>;
  using dg_auxiliary_package_data_temporary_tags = tmpl::list<>;
  using dg_auxiliary_package_data_volume_tags = tmpl::list<>;
  using dg_auxiliary_boundary_terms_volume_tags = tmpl::list<>;

  double dg_auxiliary_package_data(
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          packaged_conformal_metric,
      gsl::not_null<Scalar<DataVector>*> packaged_conformal_factor,
      gsl::not_null<Scalar<DataVector>*> packaged_lapse,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> packaged_shift,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_normal_covector,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& /*a_tilde*/,
      const Scalar<DataVector>& /*trace_extrinsic_curvature*/,
      const Scalar<DataVector>& /*theta*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*gamma_hat*/,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*auxiliary_shift_b*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*field_a*/,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& /*field_b*/,
      const tnsr::ijj<DataVector, Dim, Frame::Inertial>& /*field_d*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*field_p*/,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>&
      /*boundary_conformal_metric*/,
      const Scalar<DataVector>& /*boundary_conformal_factor*/,
      const Scalar<DataVector>& /*boundary_lapse*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*boundary_shift*/,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
      const Direction<Dim>& /*face_direction*/) const;

  void dg_auxiliary_boundary_terms(
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          conformal_metric_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> conformal_factor_boundary_correction,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          a_tilde_boundary_correction,
      gsl::not_null<Scalar<DataVector>*>
          trace_extrinsic_curvature_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> theta_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          gamma_hat_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> lapse_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          shift_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          auxiliary_shift_b_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          field_a_boundary_correction,
      gsl::not_null<tnsr::iJ<DataVector, Dim, Frame::Inertial>*>
          field_b_boundary_correction,
      gsl::not_null<tnsr::ijj<DataVector, Dim, Frame::Inertial>*>
          field_d_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          field_p_boundary_correction,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          boundary_conformal_metric_boundary_correction,
      gsl::not_null<Scalar<DataVector>*>
          boundary_conformal_factor_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> boundary_lapse_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          boundary_shift_boundary_correction,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric_int,
      const Scalar<DataVector>& conformal_factor_int,
      const Scalar<DataVector>& lapse_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector_int,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric_ext,
      const Scalar<DataVector>& conformal_factor_ext,
      const Scalar<DataVector>& lapse_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector_ext,

      dg::Formulation /*dg_formulation*/) const;

 private:
  double tau1_ = std::numeric_limits<double>::signaling_NaN();
  double tau2_ = std::numeric_limits<double>::signaling_NaN();
  bool use_central_flux_at_boundary_ = true;
};
}  // namespace Ccz4::BoundaryCorrections
