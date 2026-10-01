// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/DiscontinuousGalerkin/InterfaceOrientation.hpp"
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
 * \brief The corrected parity-paired LDG boundary correction for the SoCcz4
 * system.
 *
 * \warning This boundary correction is experimental. The external-boundary
 * treatment is unqualified (see below).
 *
 * This is a nonlinear extension to general faces of the corrected parity
 * candidate. Unlike `Ccz4::BoundaryCorrections::LaxFriedrichs` (central flux
 * plus a symmetric penalty) it builds the physical and auxiliary numerical
 * traces by splitting every field into components that are even or odd under
 * reflection across the face and taking each parity from a fixed side of the
 * interface. The authoritative Cartesian specification, including every flux
 * expression, is the nonlinear completion recorded in
 * `corrected_soccz4_flux.tex`; the generalization to arbitrary face normals is
 * fixed by the conventions documented here.
 *
 * ### Reflection decomposition and the flat-delta decision
 *
 * From the interior element's packaged normal covector \f$n_i\f$ build the
 * flat-normalized pair \f$\hat n_i = n_i/\sqrt{\delta^{jk}n_jn_k}\f$,
 * \f$\hat n^i = \delta^{ij}\hat n_j\f$, the reflection
 * \f$R^i{}_j = \delta^i{}_j - 2\hat n^i\hat n_j\f$, and the tangential
 * projector \f$q^i{}_j = \delta^i{}_j - \hat n^i\hat n_j\f$. A field is split
 * into an even part (reflection eigenvalue \f$+1\f$) and an odd part
 * (eigenvalue \f$-1\f$):
 * - a scalar is even;
 * - a vector \f$v^k\f$ has odd part \f$\hat n^k(\hat n_j v^j)\f$;
 * - a symmetric tensor \f$T_{ij}\f$ reflects as
 *   \f$R^k{}_i R^l{}_j T_{kl}\f$.
 *
 * The normal pair, the reflection, and the flat raising used by the feedback
 * term below are all built with the Euclidean metric \f$\delta_{ij}\f$, not
 * with the dynamical conformal metric. This is a deliberate choice: the two
 * abutting elements package anti-parallel normals, so a flat \f$R\f$ and
 * \f$q\f$ are single valued across the interface, whereas a metric-dependent
 * normal would be interface-discontinuous. On the archived trumpet tests the
 * conformal metric is the identity at every face, so the flat and metric
 * constructions coincide there.
 *
 * ### Side convention
 *
 * The two elements of an interior mortar are labeled Upper and Lower by the
 * per-mortar `evolution::dg::InterfaceOrientation` (see that class). The Lower
 * element is the "minus" side and the Upper element is the "plus" side; the
 * oriented jump is \f$[w]=w^{\mathrm{Upper}}-w^{\mathrm{Lower}}\f$. In terms of
 * the element-local jump \f$\Delta_K w = w^{\mathrm{ext}}-w^{\mathrm{int}}\f$,
 * \f$[w] = s\,\Delta_K w\f$ with the element-local sign \f$s=+1\f$ for the
 * Lower element and \f$s=-1\f$ for the Upper element (the TeX's \f$n_m\f$). The
 * prefixed normal \f$\hat n_{\mathrm{pref}} = s\,\hat n\f$ points from the
 * Lower element into the Upper element.
 *
 * The auxiliary-pass trace takes each field's even part from the Lower side and
 * its odd part from the Upper side,
 * \f$w^\ast = \{w\} - \tfrac{\tau_2}{2}\,s\,R[\Delta_K w]\f$ with
 * \f$\{w\}=\tfrac12(w^{\mathrm{int}}+w^{\mathrm{ext}})\f$; the physical-pass
 * nonadvective combination takes the OPPOSITE assignment (even part from the
 * Upper side), while the advective term reuses the auxiliary-style trace of its
 * differentiated field.
 *
 * ### Shared coefficient state
 *
 * Every nonlinear coefficient product in the numerical physical flux is formed
 * from a single face state built by averaging the primitive coefficients,
 * \f$\alpha_F=\{\alpha\}\f$, \f$\phi_F=\{\phi\}\f$,
 * \f$\beta_F^i=\{\beta^i\}\f$,
 * \f$(\tilde\gamma_F)_{ij}=\{\tilde\gamma_{ij}\}\f$, and inverting the averaged
 * metric once (never averaging inverse metrics). The side-local interior flux
 * subtracted in the correction \f$n_m(f_u^m-f_u^{*m})\f$ retains its side-local
 * coefficients, as does the reconstructed connection \f$\widetilde\Gamma_h^i\f$
 * (packaged per side from the side's own inverse conformal metric, keeping the
 * trace term).
 *
 * ### Penalties and feedback
 *
 * With penalty strength \f$\tau_1\f$ the dissipative penalties are plain value
 * jumps: \f$-\tfrac{\tau_1}{2}\Delta_K\f$ applied to
 * \f$\tilde A_{ij}\f$, \f$K\f$, \f$\Theta\f$,
 * \f$\hat\Gamma^i-\widetilde\Gamma_h^i\f$, and
 * \f$b^i-\widetilde\Gamma_h^i\f$, and
 * \f$-\tfrac{\text{ShiftPenalty}\,\tau_1}{2}\Delta_K\beta^i\f$ for the shift.
 * The
 * \f$\hat\Gamma^i\f$ and \f$b^i\f$ numerical fluxes carry the additional
 * feedback term
 * \f$-2(\beta_F\cdot\hat n_{\mathrm{pref}})\,R^i{}_j\,\delta^{jl}\Delta_K
 * J_l\f$ with \f$J_l=\hat n_{\mathrm{pref}}^k\hat n_{\mathrm{pref}}^m
 * D_{klm}\f$ per side (from the packaged, reconstructed \f$D\f$). The quantity
 * \f$\beta_F\cdot\hat n_{\mathrm{pref}}\f$ is the same number on both elements,
 * and \f$J_l\f$ is independent of the prefix sign, so the element-local jump
 * \f$\Delta_K J_l\f$ alone flips the feedback between the two outward fluxes.
 * The feedback vanishes identically at zero shift.
 *
 * ### Options
 *
 * - `Tau1`: the physical penalty strength \f$\tau_1\f$.
 * - `Tau2`: the coefficient of the one-sided jump part of the auxiliary-style
 *   traces (the model's bias). \f$\tau_2=1\f$ is the exact one-sided trace and
 *   the qualified value; \f$\tau_2\f$ scales only the jump part of those
 *   traces, not the exact parity selection of the nonadvective physical flux.
 * - `ShiftPenalty`: multiplies the shift value-jump penalty only.
 * - `UseCentralFluxAtBoundary`: if true (default), external boundaries use the
 *   plain central flux with no penalty, exactly like
 *   `Ccz4::BoundaryCorrections::LaxFriedrichs`; if false, the full parity
 *   scheme is applied against the ghost data with the interior element treated
 *   as the Lower side.
 *
 * ### External boundaries
 *
 * The external-boundary treatment is unqualified: neither the central-flux
 * branch nor the full-scheme branch has been analyzed for stability against
 * boundary-condition ghost data. Use at external boundaries at your own risk.
 */
template <size_t Dim>
class ParityFlux final : public evolution::BoundaryCorrection {
 public:
  struct GammaTildeH : db::SimpleTag {
    using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
  };

  struct Tau1 {
    using type = double;
    static constexpr Options::String help = {
        "The physical penalty strength tau1 for the value-jump penalties."};
  };
  struct Tau2 {
    using type = double;
    static constexpr Options::String help = {
        "The bias scaling the one-sided jump part of the auxiliary-style "
        "traces. tau2=1 is the exact one-sided (alternating) trace."};
  };
  struct ShiftPenalty {
    using type = double;
    static constexpr Options::String help = {
        "Multiplies the shift value-jump penalty only."};
    static constexpr type default_value = 1.0;
  };
  struct UseCentralFluxAtBoundary {
    using type = bool;
    static constexpr Options::String help = {
        "If true, use the plain central flux (no penalty) at external "
        "boundaries. If false, apply the full parity scheme against the ghost "
        "data with the interior element treated as the Lower side."};
    static constexpr type default_value = true;
  };

  using options =
      tmpl::list<Tau1, Tau2, ShiftPenalty, UseCentralFluxAtBoundary>;
  static constexpr Options::String help = {
      "The corrected parity-paired LDG boundary correction for SoCcz4."};

  ParityFlux() = default;
  explicit ParityFlux(double tau1, double tau2, double shift_penalty = 1.0,
                      bool use_central_flux_at_boundary = true);
  ParityFlux(const ParityFlux&) = default;
  ParityFlux& operator=(const ParityFlux&) = default;
  ParityFlux(ParityFlux&&) = default;
  ParityFlux& operator=(ParityFlux&&) = default;
  ~ParityFlux() override = default;

  /// \cond
  explicit ParityFlux(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ParityFlux);  // NOLINT
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
      // side-local reconstructed connection
      GammaTildeH,
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
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          packaged_gamma_tilde_h,
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
      const tnsr::I<DataVector, Dim, Frame::Inertial>& gamma_tilde_h_int,
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
      const tnsr::I<DataVector, Dim, Frame::Inertial>& gamma_tilde_h_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector_ext,

      evolution::dg::InterfaceOrientation orientation,
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

      evolution::dg::InterfaceOrientation orientation,
      dg::Formulation /*dg_formulation*/) const;

 private:
  double tau1_ = std::numeric_limits<double>::signaling_NaN();
  double tau2_ = std::numeric_limits<double>::signaling_NaN();
  double shift_penalty_ = 1.0;
  bool use_central_flux_at_boundary_ = true;
};
}  // namespace Ccz4::BoundaryCorrections
