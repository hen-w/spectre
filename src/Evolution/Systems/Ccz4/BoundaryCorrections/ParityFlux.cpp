// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/BoundaryCorrections/ParityFlux.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DiscontinuousGalerkin/InterfaceOrientation.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Ccz4::BoundaryCorrections {

namespace {
// The element-local sign s: +1 if the interior element is the Lower ("minus")
// side of the interface, -1 if it is the Upper ("plus") side. An external
// boundary that runs the full scheme treats the interior element as the Lower
// side, so s = +1 there as well.
double element_local_sign(
    const evolution::dg::InterfaceOrientation orientation) {
  switch (orientation) {
    case evolution::dg::InterfaceOrientation::InteriorIsUpper:
      return -1.0;
    case evolution::dg::InterfaceOrientation::InteriorIsLower:
      return 1.0;
    case evolution::dg::InterfaceOrientation::ExternalBoundary:
      return 1.0;
    default:
      ERROR("Unknown InterfaceOrientation.");
  }
}
}  // namespace

template <size_t Dim>
ParityFlux<Dim>::ParityFlux(CkMigrateMessage* msg) : BoundaryCorrection(msg) {}

template <size_t Dim>
ParityFlux<Dim>::ParityFlux(const double tau1, const double tau2,
                            const double shift_penalty,
                            const bool use_central_flux_at_boundary)
    : tau1_(tau1),
      tau2_(tau2),
      shift_penalty_(shift_penalty),
      use_central_flux_at_boundary_(use_central_flux_at_boundary) {}

template <size_t Dim>
std::unique_ptr<evolution::BoundaryCorrection> ParityFlux<Dim>::get_clone()
    const {
  return std::make_unique<ParityFlux>(*this);
}

template <size_t Dim>
void ParityFlux<Dim>::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | tau1_;
  p | tau2_;
  p | shift_penalty_;
  p | use_central_flux_at_boundary_;
}

template <size_t Dim>
double ParityFlux<Dim>::dg_package_data(
    gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
        packaged_conformal_metric,
    gsl::not_null<Scalar<DataVector>*> packaged_conformal_factor,
    gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*> packaged_a_tilde,
    gsl::not_null<Scalar<DataVector>*> packaged_trace_extrinsic_curvature,
    gsl::not_null<Scalar<DataVector>*> packaged_theta,
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        packaged_gamma_hat,
    gsl::not_null<Scalar<DataVector>*> packaged_lapse,
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> packaged_shift,
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        packaged_auxiliary_shift_b,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> packaged_field_a,
    gsl::not_null<tnsr::iJ<DataVector, Dim, Frame::Inertial>*> packaged_field_b,
    gsl::not_null<tnsr::ijj<DataVector, Dim, Frame::Inertial>*>
        packaged_field_d,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> packaged_field_p,
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
    const Direction<Dim>& /*face_direction*/) const {
  *packaged_conformal_metric = conformal_metric;
  *packaged_conformal_factor = conformal_factor;
  *packaged_a_tilde = a_tilde;
  *packaged_trace_extrinsic_curvature = trace_extrinsic_curvature;
  *packaged_theta = theta;
  *packaged_gamma_hat = gamma_hat;
  *packaged_lapse = lapse;
  *packaged_shift = shift;
  *packaged_auxiliary_shift_b = auxiliary_shift_b;
  *packaged_field_a = field_a;
  *packaged_field_b = field_b;
  *packaged_field_d = field_d;
  *packaged_field_p = field_p;
  *packaged_normal_covector = normal_covector;

  // Side-local reconstructed connection, using this side's own inverse
  // conformal metric (never the shared face metric), keeping the trace term:
  //   GammaTilde_h^i = gtilde^{jk} gtilde^{il} (D_jkl + D_kjl - D_ljk).
  const auto inverse_conformal_metric =
      determinant_and_inverse(conformal_metric).second;
  ::tenex::evaluate<ti::I>(
      packaged_gamma_tilde_h,
      inverse_conformal_metric(ti::J, ti::K) *
          inverse_conformal_metric(ti::I, ti::L) *
          (field_d(ti::j, ti::k, ti::l) + field_d(ti::k, ti::j, ti::l) -
           field_d(ti::l, ti::j, ti::k)));

  return 0.0;
}

template <size_t Dim>
template <bool ForExternalBoundary>
void ParityFlux<Dim>::dg_boundary_terms(
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

    const evolution::dg::InterfaceOrientation orientation,
    dg::Formulation /*dg_formulation*/) const {
  // Boundary second-order corrections are always zero.
  *boundary_conformal_metric_boundary_correction =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
          conformal_factor_int, 0.0);
  *boundary_conformal_factor_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *boundary_lapse_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *boundary_shift_boundary_correction =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          conformal_factor_int, 0.0);

  // The conformal metric, conformal factor, and lapse carry no numerical flux
  // and no penalty, exactly as in LaxFriedrichs.
  *conformal_metric_boundary_correction =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
          conformal_metric_int, 0.0);
  *conformal_factor_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *lapse_boundary_correction =
      make_with_value<Scalar<DataVector>>(lapse_int, 0.0);

  const size_t num_points = get(conformal_factor_int).size();

  const bool external =
      orientation == evolution::dg::InterfaceOrientation::ExternalBoundary;
  const bool central_external = external and use_central_flux_at_boundary_;
  const double sign = element_local_sign(orientation);

  // --- common per-side precomputations (side-local) ---
  const Scalar<DataVector> normal_dot_shift_int =
      dot_product(shift_int, normal_covector_int);
  const Scalar<DataVector> normal_dot_shift_ext =
      dot_product(shift_ext, normal_covector_ext);
  const auto inverse_conformal_metric_int =
      determinant_and_inverse(conformal_metric_int).second;
  const auto inverse_conformal_metric_ext =
      determinant_and_inverse(conformal_metric_ext).second;
  const tnsr::I<DataVector, Dim, Frame::Inertial>
      inverse_conformal_metric_dot_normal_int =
          ::tenex::evaluate<ti::I>(inverse_conformal_metric_int(ti::I, ti::J) *
                                   normal_covector_int(ti::j));
  const tnsr::I<DataVector, Dim, Frame::Inertial>
      inverse_conformal_metric_dot_normal_ext =
          ::tenex::evaluate<ti::I>(inverse_conformal_metric_ext(ti::I, ti::J) *
                                   normal_covector_ext(ti::j));
  Scalar<DataVector> conformal_factor_squared_int;
  get(conformal_factor_squared_int) =
      get(conformal_factor_int) * get(conformal_factor_int);
  Scalar<DataVector> conformal_factor_squared_ext;
  get(conformal_factor_squared_ext) =
      get(conformal_factor_ext) * get(conformal_factor_ext);
  const Scalar<DataVector> gamma_hat_dot_normal_int =
      dot_product(gamma_hat_int, normal_covector_int);
  const Scalar<DataVector> gamma_hat_dot_normal_ext =
      dot_product(gamma_hat_ext, normal_covector_ext);

  // The nonadvective physical flux dot (interior) normal for each physical
  // variable. These are the LaxFriedrichs flux-dot-normal expressions with the
  // advective -(shift.n) w term omitted (shift_dot_normal passed as zero gives
  // the nonadvective part); they are evaluated once with side-local
  // coefficients (for the interior flux f_u^m) and once with the shared face
  // coefficients (for g_u^{m,+-}). The advective term is added separately.
  const auto k_flux_dot_normal =
      [](const Scalar<DataVector>& shift_dot_normal,
         const tnsr::I<DataVector, Dim, Frame::Inertial>&
             inverse_conformal_metric_dot_normal,
         const Scalar<DataVector>& trace_extrinsic_curvature,
         const Scalar<DataVector>& lapse,
         const Scalar<DataVector>& conformal_factor_squared,
         const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a,
         const tnsr::II<DataVector, Dim, Frame::Inertial>&
             inverse_conformal_metric,
         const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d,
         const Scalar<DataVector>& gamma_hat_dot_normal,
         const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p) {
        Scalar<DataVector> result;
        ::tenex::evaluate(
            make_not_null(&result),
            -1.0 * shift_dot_normal() * trace_extrinsic_curvature() +
                lapse() * conformal_factor_squared() *
                    inverse_conformal_metric_dot_normal(ti::I) *
                    field_a(ti::i) +
                lapse() * conformal_factor_squared() *
                    inverse_conformal_metric(ti::I, ti::J) *
                    field_d(ti::k, ti::i, ti::j) *
                    inverse_conformal_metric_dot_normal(ti::K) -
                lapse() * conformal_factor_squared() * gamma_hat_dot_normal() -
                4.0 * lapse() * conformal_factor_squared() *
                    inverse_conformal_metric_dot_normal(ti::I) *
                    field_p(ti::i));
        return result;
      };

  const auto a_tilde_flux_dot_normal =
      [](const Scalar<DataVector>& shift_dot_normal,
         const tnsr::ii<DataVector, Dim, Frame::Inertial>& a_tilde,
         const Scalar<DataVector>& lapse,
         const Scalar<DataVector>& conformal_factor_squared,
         const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
         const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a,
         const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric,
         const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d,
         const tnsr::I<DataVector, Dim, Frame::Inertial>& gamma_hat,
         const Scalar<DataVector>& gamma_hat_dot_normal,
         const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p,
         const tnsr::I<DataVector, Dim, Frame::Inertial>&
             inverse_conformal_metric_dot_normal,
         const tnsr::II<DataVector, Dim, Frame::Inertial>&
             inverse_conformal_metric) {
        tnsr::ii<DataVector, Dim, Frame::Inertial> result;
        ::tenex::evaluate<ti::i, ti::j>(
            make_not_null(&result),
            -1.0 * shift_dot_normal() * a_tilde(ti::i, ti::j) +
                lapse() * conformal_factor_squared() *
                    (0.5 * normal_covector(ti::i) * field_a(ti::j) +
                     0.5 * normal_covector(ti::j) * field_a(ti::i) -
                     conformal_metric(ti::i, ti::j) *
                         inverse_conformal_metric_dot_normal(ti::K) *
                         field_a(ti::k) / 3.0 +
                     inverse_conformal_metric_dot_normal(ti::K) *
                         field_d(ti::k, ti::i, ti::j) -
                     conformal_metric(ti::i, ti::j) *
                         inverse_conformal_metric(ti::M, ti::N) *
                         inverse_conformal_metric_dot_normal(ti::K) *
                         field_d(ti::k, ti::m, ti::n) / 3.0 -
                     0.5 * normal_covector(ti::i) *
                         conformal_metric(ti::j, ti::k) * gamma_hat(ti::K) -
                     0.5 * normal_covector(ti::j) *
                         conformal_metric(ti::i, ti::k) * gamma_hat(ti::K) +
                     conformal_metric(ti::i, ti::j) * gamma_hat_dot_normal() /
                         3.0 -
                     0.5 * normal_covector(ti::i) * field_p(ti::j) -
                     0.5 * normal_covector(ti::j) * field_p(ti::i) +
                     conformal_metric(ti::i, ti::j) *
                         inverse_conformal_metric_dot_normal(ti::K) *
                         field_p(ti::k) / 3.0));
        return result;
      };

  const auto theta_flux_dot_normal =
      [](const Scalar<DataVector>& shift_dot_normal,
         const Scalar<DataVector>& theta, const Scalar<DataVector>& lapse,
         const Scalar<DataVector>& conformal_factor_squared,
         const tnsr::I<DataVector, Dim, Frame::Inertial>&
             inverse_conformal_metric_dot_normal,
         const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d,
         const Scalar<DataVector>& gamma_hat_dot_normal,
         const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p,
         const tnsr::II<DataVector, Dim, Frame::Inertial>&
             inverse_conformal_metric) {
        Scalar<DataVector> result;
        ::tenex::evaluate(
            make_not_null(&result),
            -1.0 * shift_dot_normal() * theta() +
                0.5 * lapse() * conformal_factor_squared() *
                    (inverse_conformal_metric(ti::I, ti::J) *
                         inverse_conformal_metric_dot_normal(ti::K) *
                         field_d(ti::k, ti::i, ti::j) -
                     gamma_hat_dot_normal() -
                     4.0 * inverse_conformal_metric_dot_normal(ti::I) *
                         field_p(ti::i)));
        return result;
      };

  const auto gamma_hat_flux_dot_normal =
      [](const Scalar<DataVector>& shift_dot_normal,
         const tnsr::I<DataVector, Dim, Frame::Inertial>& gamma_hat,
         const Scalar<DataVector>& lapse,
         const tnsr::I<DataVector, Dim, Frame::Inertial>&
             inverse_conformal_metric_dot_normal,
         const Scalar<DataVector>& trace_extrinsic_curvature,
         const Scalar<DataVector>& theta,
         const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b,
         const tnsr::II<DataVector, Dim, Frame::Inertial>&
             inverse_conformal_metric,
         const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector) {
        tnsr::I<DataVector, Dim, Frame::Inertial> result;
        ::tenex::evaluate<ti::I>(
            make_not_null(&result),
            -1.0 * shift_dot_normal() * gamma_hat(ti::I) +
                (4.0 / 3.0) * lapse() *
                    inverse_conformal_metric_dot_normal(ti::I) *
                    trace_extrinsic_curvature() -
                2.0 * lapse() * inverse_conformal_metric_dot_normal(ti::I) *
                    theta() -
                inverse_conformal_metric_dot_normal(ti::J) *
                    field_b(ti::j, ti::I) -
                inverse_conformal_metric_dot_normal(ti::I) *
                    field_b(ti::j, ti::J) / 6.0 -
                inverse_conformal_metric(ti::I, ti::K) * field_b(ti::k, ti::J) *
                    normal_covector(ti::j) / 6.0);
        return result;
      };

  const auto b_flux_dot_normal =
      [&gamma_hat_flux_dot_normal](
          const Scalar<DataVector>& shift_dot_normal,
          const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_shift_b,
          const tnsr::I<DataVector, Dim, Frame::Inertial>& gamma_hat,
          const Scalar<DataVector>& lapse,
          const tnsr::I<DataVector, Dim, Frame::Inertial>&
              inverse_conformal_metric_dot_normal,
          const Scalar<DataVector>& trace_extrinsic_curvature,
          const Scalar<DataVector>& theta,
          const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b,
          const tnsr::II<DataVector, Dim, Frame::Inertial>&
              inverse_conformal_metric,
          const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector) {
        tnsr::I<DataVector, Dim, Frame::Inertial> result =
            gamma_hat_flux_dot_normal(
                shift_dot_normal, gamma_hat, lapse,
                inverse_conformal_metric_dot_normal, trace_extrinsic_curvature,
                theta, field_b, inverse_conformal_metric, normal_covector);
        if constexpr (::Ccz4::fd::System::shifting_shift) {
          ::tenex::update<ti::I>(
              make_not_null(&result),
              result(ti::I) + shift_dot_normal() * gamma_hat(ti::I) -
                  shift_dot_normal() * auxiliary_shift_b(ti::I));
        }
        return result;
      };

  // The physical-pass corrections of the auxiliary reduction variables
  // field_a/b/d/p are never consumed (the auxiliaries are reconstructed, not
  // evolved; their dt does not enter the operator), so they are set to zero.
  *field_a_boundary_correction =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          conformal_factor_int, 0.0);
  *field_b_boundary_correction =
      make_with_value<tnsr::iJ<DataVector, Dim, Frame::Inertial>>(
          conformal_factor_int, 0.0);
  *field_d_boundary_correction =
      make_with_value<tnsr::ijj<DataVector, Dim, Frame::Inertial>>(
          conformal_factor_int, 0.0);
  *field_p_boundary_correction =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          conformal_factor_int, 0.0);
  const auto zero_scalar =
      make_with_value<Scalar<DataVector>>(DataVector(num_points), 0.0);

  if (central_external) {
    // Plain central flux with no penalty, exactly like the LaxFriedrichs
    // external-boundary branch. The shift carries no correction here.
    *shift_boundary_correction =
        make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(shift_int,
                                                                   0.0);
    const auto k_int = k_flux_dot_normal(
        normal_dot_shift_int, inverse_conformal_metric_dot_normal_int,
        trace_extrinsic_curvature_int, lapse_int, conformal_factor_squared_int,
        field_a_int, inverse_conformal_metric_int, field_d_int,
        gamma_hat_dot_normal_int, field_p_int);
    const auto k_ext = k_flux_dot_normal(
        normal_dot_shift_ext, inverse_conformal_metric_dot_normal_ext,
        trace_extrinsic_curvature_ext, lapse_ext, conformal_factor_squared_ext,
        field_a_ext, inverse_conformal_metric_ext, field_d_ext,
        gamma_hat_dot_normal_ext, field_p_ext);
    get(*trace_extrinsic_curvature_boundary_correction) =
        -0.5 * (get(k_int) + get(k_ext));

    const auto at_int = a_tilde_flux_dot_normal(
        normal_dot_shift_int, a_tilde_int, lapse_int,
        conformal_factor_squared_int, normal_covector_int, field_a_int,
        conformal_metric_int, field_d_int, gamma_hat_int,
        gamma_hat_dot_normal_int, field_p_int,
        inverse_conformal_metric_dot_normal_int, inverse_conformal_metric_int);
    const auto at_ext = a_tilde_flux_dot_normal(
        normal_dot_shift_ext, a_tilde_ext, lapse_ext,
        conformal_factor_squared_ext, normal_covector_ext, field_a_ext,
        conformal_metric_ext, field_d_ext, gamma_hat_ext,
        gamma_hat_dot_normal_ext, field_p_ext,
        inverse_conformal_metric_dot_normal_ext, inverse_conformal_metric_ext);
    ::tenex::evaluate<ti::i, ti::j>(
        a_tilde_boundary_correction,
        -0.5 * (at_int(ti::i, ti::j) + at_ext(ti::i, ti::j)));

    const auto th_int = theta_flux_dot_normal(
        normal_dot_shift_int, theta_int, lapse_int,
        conformal_factor_squared_int, inverse_conformal_metric_dot_normal_int,
        field_d_int, gamma_hat_dot_normal_int, field_p_int,
        inverse_conformal_metric_int);
    const auto th_ext = theta_flux_dot_normal(
        normal_dot_shift_ext, theta_ext, lapse_ext,
        conformal_factor_squared_ext, inverse_conformal_metric_dot_normal_ext,
        field_d_ext, gamma_hat_dot_normal_ext, field_p_ext,
        inverse_conformal_metric_ext);
    get(*theta_boundary_correction) = -0.5 * (get(th_int) + get(th_ext));

    const auto gh_int = gamma_hat_flux_dot_normal(
        normal_dot_shift_int, gamma_hat_int, lapse_int,
        inverse_conformal_metric_dot_normal_int, trace_extrinsic_curvature_int,
        theta_int, field_b_int, inverse_conformal_metric_int,
        normal_covector_int);
    const auto gh_ext = gamma_hat_flux_dot_normal(
        normal_dot_shift_ext, gamma_hat_ext, lapse_ext,
        inverse_conformal_metric_dot_normal_ext, trace_extrinsic_curvature_ext,
        theta_ext, field_b_ext, inverse_conformal_metric_ext,
        normal_covector_ext);
    ::tenex::evaluate<ti::I>(gamma_hat_boundary_correction,
                             -0.5 * (gh_int(ti::I) + gh_ext(ti::I)));
    const auto b_int = b_flux_dot_normal(
        normal_dot_shift_int, auxiliary_shift_b_int, gamma_hat_int, lapse_int,
        inverse_conformal_metric_dot_normal_int, trace_extrinsic_curvature_int,
        theta_int, field_b_int, inverse_conformal_metric_int,
        normal_covector_int);
    const auto b_ext = b_flux_dot_normal(
        normal_dot_shift_ext, auxiliary_shift_b_ext, gamma_hat_ext, lapse_ext,
        inverse_conformal_metric_dot_normal_ext, trace_extrinsic_curvature_ext,
        theta_ext, field_b_ext, inverse_conformal_metric_ext,
        normal_covector_ext);
    ::tenex::evaluate<ti::I>(auxiliary_shift_b_boundary_correction,
                             -0.5 * (b_int(ti::I) + b_ext(ti::I)));
    return;
  }

  // --- parity scheme for the primary physical variables ---

  // Flat-normalized interior normal and reflection R^i_j = delta^i_j -
  // 2 nhat^i nhat_j (flat pair, single valued across the interface).
  DataVector normal_magnitude(num_points, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    normal_magnitude += normal_covector_int.get(i) * normal_covector_int.get(i);
  }
  normal_magnitude = sqrt(normal_magnitude);
  auto nhat = make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
      DataVector(num_points), 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    nhat.get(i) = normal_covector_int.get(i) / normal_magnitude;
  }
  // R^i_j; symmetric with flat raising/lowering.
  std::array<std::array<DataVector, Dim>, Dim> reflection{};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      gsl::at(gsl::at(reflection, i), j) = -2.0 * nhat.get(i) * nhat.get(j);
      if (i == j) {
        gsl::at(gsl::at(reflection, i), j) += 1.0;
      }
    }
  }
  const auto reflect_vector =
      [&reflection](const tnsr::I<DataVector, Dim, Frame::Inertial>& v) {
        auto result =
            make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(v, 0.0);
        for (size_t i = 0; i < Dim; ++i) {
          for (size_t j = 0; j < Dim; ++j) {
            result.get(i) += gsl::at(gsl::at(reflection, i), j) * v.get(j);
          }
        }
        return result;
      };
  const auto reflect_sym =
      [&reflection](const tnsr::ii<DataVector, Dim, Frame::Inertial>& t) {
        auto result =
            make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(t, 0.0);
        for (size_t i = 0; i < Dim; ++i) {
          for (size_t j = i; j < Dim; ++j) {
            for (size_t k = 0; k < Dim; ++k) {
              for (size_t l = 0; l < Dim; ++l) {
                result.get(i, j) += gsl::at(gsl::at(reflection, k), i) *
                                    gsl::at(gsl::at(reflection, l), j) *
                                    t.get(k, l);
              }
            }
          }
        }
        return result;
      };

  // Shared face coefficient state: average the primitive coefficients, invert
  // the averaged metric once.
  auto conformal_metric_face =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
          conformal_metric_int, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      conformal_metric_face.get(i, j) = 0.5 * (conformal_metric_int.get(i, j) +
                                               conformal_metric_ext.get(i, j));
    }
  }
  const auto inverse_conformal_metric_face =
      determinant_and_inverse(conformal_metric_face).second;
  const tnsr::I<DataVector, Dim, Frame::Inertial>
      inverse_conformal_metric_face_dot_normal =
          ::tenex::evaluate<ti::I>(inverse_conformal_metric_face(ti::I, ti::J) *
                                   normal_covector_int(ti::j));
  Scalar<DataVector> lapse_face;
  get(lapse_face) = 0.5 * (get(lapse_int) + get(lapse_ext));
  Scalar<DataVector> conformal_factor_face;
  get(conformal_factor_face) =
      0.5 * (get(conformal_factor_int) + get(conformal_factor_ext));
  Scalar<DataVector> conformal_factor_squared_face;
  get(conformal_factor_squared_face) =
      get(conformal_factor_face) * get(conformal_factor_face);
  auto shift_face = make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
      shift_int, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    shift_face.get(i) = 0.5 * (shift_int.get(i) + shift_ext.get(i));
  }
  // beta_F . n_int (raw normal; advective, outward-normal convention).
  const Scalar<DataVector> beta_face_dot_normal =
      dot_product(shift_face, normal_covector_int);
  // beta_F . nhat_pref = sign * (beta_F . nhat): same number on both elements.
  DataVector beta_face_dot_nhat(num_points, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    beta_face_dot_nhat += shift_face.get(i) * nhat.get(i);
  }
  const DataVector beta_face_dot_nhat_pref = sign * beta_face_dot_nhat;

  // gamma_hat . n_int for the shared-coefficient flux arguments (z-side).
  const Scalar<DataVector> gamma_hat_int_dot_normal_int =
      dot_product(gamma_hat_int, normal_covector_int);
  const Scalar<DataVector> gamma_hat_ext_dot_normal_int =
      dot_product(gamma_hat_ext, normal_covector_int);

  // --- K ---
  {
    const auto interior = k_flux_dot_normal(
        normal_dot_shift_int, inverse_conformal_metric_dot_normal_int,
        trace_extrinsic_curvature_int, lapse_int, conformal_factor_squared_int,
        field_a_int, inverse_conformal_metric_int, field_d_int,
        gamma_hat_dot_normal_int, field_p_int);
    const auto g_int =
        k_flux_dot_normal(zero_scalar, inverse_conformal_metric_face_dot_normal,
                          trace_extrinsic_curvature_int, lapse_face,
                          conformal_factor_squared_face, field_a_int,
                          inverse_conformal_metric_face, field_d_int,
                          gamma_hat_int_dot_normal_int, field_p_int);
    const auto g_ext =
        k_flux_dot_normal(zero_scalar, inverse_conformal_metric_face_dot_normal,
                          trace_extrinsic_curvature_ext, lapse_face,
                          conformal_factor_squared_face, field_a_ext,
                          inverse_conformal_metric_face, field_d_ext,
                          gamma_hat_ext_dot_normal_int, field_p_ext);
    // scalar: even (reflection eigenvalue +1).
    DataVector nonadvective = 0.5 * (get(g_int) + get(g_ext)) +
                              0.5 * sign * (get(g_ext) - get(g_int));
    DataVector trace = 0.5 * (get(trace_extrinsic_curvature_int) +
                              get(trace_extrinsic_curvature_ext)) -
                       0.5 * tau2_ * sign *
                           (get(trace_extrinsic_curvature_ext) -
                            get(trace_extrinsic_curvature_int));
    get(*trace_extrinsic_curvature_boundary_correction) =
        nonadvective - get(beta_face_dot_normal) * trace +
        (-0.5 * tau1_ *
         (get(trace_extrinsic_curvature_ext) -
          get(trace_extrinsic_curvature_int))) -
        get(interior);
  }

  // --- Theta ---
  {
    const auto interior = theta_flux_dot_normal(
        normal_dot_shift_int, theta_int, lapse_int,
        conformal_factor_squared_int, inverse_conformal_metric_dot_normal_int,
        field_d_int, gamma_hat_dot_normal_int, field_p_int,
        inverse_conformal_metric_int);
    const auto g_int = theta_flux_dot_normal(
        zero_scalar, theta_int, lapse_face, conformal_factor_squared_face,
        inverse_conformal_metric_face_dot_normal, field_d_int,
        gamma_hat_int_dot_normal_int, field_p_int,
        inverse_conformal_metric_face);
    const auto g_ext = theta_flux_dot_normal(
        zero_scalar, theta_ext, lapse_face, conformal_factor_squared_face,
        inverse_conformal_metric_face_dot_normal, field_d_ext,
        gamma_hat_ext_dot_normal_int, field_p_ext,
        inverse_conformal_metric_face);
    DataVector nonadvective = 0.5 * (get(g_int) + get(g_ext)) +
                              0.5 * sign * (get(g_ext) - get(g_int));
    DataVector trace = 0.5 * (get(theta_int) + get(theta_ext)) -
                       0.5 * tau2_ * sign * (get(theta_ext) - get(theta_int));
    get(*theta_boundary_correction) =
        nonadvective - get(beta_face_dot_normal) * trace +
        (-0.5 * tau1_ * (get(theta_ext) - get(theta_int))) - get(interior);
  }

  // --- ATilde ---
  {
    const auto interior = a_tilde_flux_dot_normal(
        normal_dot_shift_int, a_tilde_int, lapse_int,
        conformal_factor_squared_int, normal_covector_int, field_a_int,
        conformal_metric_int, field_d_int, gamma_hat_int,
        gamma_hat_dot_normal_int, field_p_int,
        inverse_conformal_metric_dot_normal_int, inverse_conformal_metric_int);
    const auto g_int = a_tilde_flux_dot_normal(
        zero_scalar, a_tilde_int, lapse_face, conformal_factor_squared_face,
        normal_covector_int, field_a_int, conformal_metric_face, field_d_int,
        gamma_hat_int, gamma_hat_int_dot_normal_int, field_p_int,
        inverse_conformal_metric_face_dot_normal,
        inverse_conformal_metric_face);
    const auto g_ext = a_tilde_flux_dot_normal(
        zero_scalar, a_tilde_ext, lapse_face, conformal_factor_squared_face,
        normal_covector_int, field_a_ext, conformal_metric_face, field_d_ext,
        gamma_hat_ext, gamma_hat_ext_dot_normal_int, field_p_ext,
        inverse_conformal_metric_face_dot_normal,
        inverse_conformal_metric_face);
    auto delta_g = make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
        DataVector(num_points), 0.0);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        delta_g.get(i, j) = g_ext.get(i, j) - g_int.get(i, j);
      }
    }
    const auto reflected_delta_g = reflect_sym(delta_g);
    auto delta_a_tilde =
        make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
            DataVector(num_points), 0.0);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        delta_a_tilde.get(i, j) = a_tilde_ext.get(i, j) - a_tilde_int.get(i, j);
      }
    }
    const auto reflected_delta_a_tilde = reflect_sym(delta_a_tilde);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        const DataVector nonadvective =
            0.5 * (g_int.get(i, j) + g_ext.get(i, j)) +
            0.5 * sign * reflected_delta_g.get(i, j);
        const DataVector trace =
            0.5 * (a_tilde_int.get(i, j) + a_tilde_ext.get(i, j)) -
            0.5 * tau2_ * sign * reflected_delta_a_tilde.get(i, j);
        a_tilde_boundary_correction->get(i, j) =
            nonadvective - get(beta_face_dot_normal) * trace +
            (-0.5 * tau1_ * (a_tilde_ext.get(i, j) - a_tilde_int.get(i, j))) -
            interior.get(i, j);
      }
    }
  }

  // --- GammaHat and b (identical flux; different penalty argument) ---
  {
    const auto interior = gamma_hat_flux_dot_normal(
        normal_dot_shift_int, gamma_hat_int, lapse_int,
        inverse_conformal_metric_dot_normal_int, trace_extrinsic_curvature_int,
        theta_int, field_b_int, inverse_conformal_metric_int,
        normal_covector_int);
    const auto g_int = gamma_hat_flux_dot_normal(
        zero_scalar, gamma_hat_int, lapse_face,
        inverse_conformal_metric_face_dot_normal, trace_extrinsic_curvature_int,
        theta_int, field_b_int, inverse_conformal_metric_face,
        normal_covector_int);
    const auto g_ext = gamma_hat_flux_dot_normal(
        zero_scalar, gamma_hat_ext, lapse_face,
        inverse_conformal_metric_face_dot_normal, trace_extrinsic_curvature_ext,
        theta_ext, field_b_ext, inverse_conformal_metric_face,
        normal_covector_int);
    auto delta_g = make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
        DataVector(num_points), 0.0);
    for (size_t i = 0; i < Dim; ++i) {
      delta_g.get(i) = g_ext.get(i) - g_int.get(i);
    }
    const auto reflected_delta_g = reflect_vector(delta_g);
    // w_u = GammaHat for both equations.
    auto delta_gamma_hat =
        make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
            DataVector(num_points), 0.0);
    for (size_t i = 0; i < Dim; ++i) {
      delta_gamma_hat.get(i) = gamma_hat_ext.get(i) - gamma_hat_int.get(i);
    }
    const auto reflected_delta_gamma_hat = reflect_vector(delta_gamma_hat);

    // Feedback: J_l = nhat^k nhat^m D_klm per side; term =
    // -2 (beta_F . nhat_pref) R^i_j delta^{jl} Delta_K J_l.
    auto j_int = make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
        DataVector(num_points), 0.0);
    auto j_ext = make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
        DataVector(num_points), 0.0);
    for (size_t l = 0; l < Dim; ++l) {
      for (size_t k = 0; k < Dim; ++k) {
        for (size_t m = 0; m < Dim; ++m) {
          j_int.get(l) += nhat.get(k) * nhat.get(m) * field_d_int.get(k, l, m);
          j_ext.get(l) += nhat.get(k) * nhat.get(m) * field_d_ext.get(k, l, m);
        }
      }
    }
    auto delta_j = make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
        DataVector(num_points), 0.0);
    for (size_t i = 0; i < Dim; ++i) {
      delta_j.get(i) = j_ext.get(i) - j_int.get(i);
    }
    const auto reflected_delta_j = reflect_vector(delta_j);

    for (size_t i = 0; i < Dim; ++i) {
      const DataVector nonadvective = 0.5 * (g_int.get(i) + g_ext.get(i)) +
                                      0.5 * sign * reflected_delta_g.get(i);
      const DataVector trace =
          0.5 * (gamma_hat_int.get(i) + gamma_hat_ext.get(i)) -
          0.5 * tau2_ * sign * reflected_delta_gamma_hat.get(i);
      const DataVector feedback =
          -2.0 * beta_face_dot_nhat_pref * reflected_delta_j.get(i);
      const DataVector common = nonadvective -
                                get(beta_face_dot_normal) * trace + feedback -
                                interior.get(i);
      gamma_hat_boundary_correction->get(i) =
          common - 0.5 * tau1_ *
                       ((gamma_hat_ext.get(i) - gamma_tilde_h_ext.get(i)) -
                        (gamma_hat_int.get(i) - gamma_tilde_h_int.get(i)));
      auxiliary_shift_b_boundary_correction->get(i) =
          common -
          0.5 * tau1_ *
              ((auxiliary_shift_b_ext.get(i) - gamma_tilde_h_ext.get(i)) -
               (auxiliary_shift_b_int.get(i) - gamma_tilde_h_int.get(i)));
    }
  }

  // --- shift (beta): no physical flux, shift-penalty only ---
  for (size_t i = 0; i < Dim; ++i) {
    shift_boundary_correction->get(i) =
        -0.5 * shift_penalty_ * tau1_ * (shift_ext.get(i) - shift_int.get(i));
  }
}

template <size_t Dim>
double ParityFlux<Dim>::dg_auxiliary_package_data(
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
    const Direction<Dim>& /*face_direction*/) const {
  *packaged_conformal_metric = conformal_metric;
  *packaged_conformal_factor = conformal_factor;
  *packaged_lapse = lapse;
  *packaged_shift = shift;
  *packaged_normal_covector = normal_covector;

  return 0.0;
}

template <size_t Dim>
void ParityFlux<Dim>::dg_auxiliary_boundary_terms(
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
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
    /*normal_covector_ext*/,

    const evolution::dg::InterfaceOrientation orientation,
    dg::Formulation /*dg_formulation*/) const {
  // The exterior normal is anti-parallel to the interior normal; the
  // auxiliary correction n_i (w_int - w^*) is expressed through the interior
  // normal, so the exterior normal is not used here.
  // only auxiliary reduction variables have nonzero boundary corrections
  *conformal_metric_boundary_correction =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
          conformal_metric_int, 0.0);
  *conformal_factor_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *a_tilde_boundary_correction =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
          conformal_metric_int, 0.0);
  *trace_extrinsic_curvature_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *theta_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *gamma_hat_boundary_correction =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(shift_int,
                                                                 0.0);
  *lapse_boundary_correction =
      make_with_value<Scalar<DataVector>>(lapse_int, 0.0);
  *shift_boundary_correction =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(shift_int,
                                                                 0.0);
  *auxiliary_shift_b_boundary_correction =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(shift_int,
                                                                 0.0);

  // Boundary second-order corrections are always zero
  *boundary_conformal_metric_boundary_correction =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
          conformal_factor_int, 0.0);
  *boundary_conformal_factor_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *boundary_lapse_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *boundary_shift_boundary_correction =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          conformal_factor_int, 0.0);

  const size_t num_points = get(conformal_factor_int).size();
  const bool external =
      orientation == evolution::dg::InterfaceOrientation::ExternalBoundary;
  const bool central_external = external and use_central_flux_at_boundary_;
  const double sign = element_local_sign(orientation);
  // Scale on the parity-jump part of the auxiliary trace. The central-flux
  // external branch takes the plain average ({w}); the jump part is dropped.
  const double jump_coefficient = central_external ? 0.0 : (tau2_ * sign);

  // Flat-normalized interior normal and reflection.
  DataVector normal_magnitude(num_points, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    normal_magnitude += normal_covector_int.get(i) * normal_covector_int.get(i);
  }
  normal_magnitude = sqrt(normal_magnitude);
  auto nhat = make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
      DataVector(num_points), 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    nhat.get(i) = normal_covector_int.get(i) / normal_magnitude;
  }
  std::array<std::array<DataVector, Dim>, Dim> reflection{};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      gsl::at(gsl::at(reflection, i), j) = -2.0 * nhat.get(i) * nhat.get(j);
      if (i == j) {
        gsl::at(gsl::at(reflection, i), j) += 1.0;
      }
    }
  }

  // log(lapse), log(conformal_factor): scalars (even). w^* = {w} - (c/2)[w],
  // [w] = w_ext - w_int. Correction = n_int ( w_int - w^* ).
  Scalar<DataVector> log_lapse_int;
  Scalar<DataVector> log_lapse_ext;
  get(log_lapse_int) = log(get(lapse_int));
  get(log_lapse_ext) = log(get(lapse_ext));
  const DataVector log_lapse_star =
      0.5 * (get(log_lapse_int) + get(log_lapse_ext)) -
      0.5 * jump_coefficient * (get(log_lapse_ext) - get(log_lapse_int));
  for (size_t i = 0; i < Dim; ++i) {
    field_a_boundary_correction->get(i) =
        normal_covector_int.get(i) * (get(log_lapse_int) - log_lapse_star);
  }

  Scalar<DataVector> log_conformal_factor_int;
  Scalar<DataVector> log_conformal_factor_ext;
  get(log_conformal_factor_int) = log(get(conformal_factor_int));
  get(log_conformal_factor_ext) = log(get(conformal_factor_ext));
  const DataVector log_conformal_factor_star =
      0.5 * (get(log_conformal_factor_int) + get(log_conformal_factor_ext)) -
      0.5 * jump_coefficient *
          (get(log_conformal_factor_ext) - get(log_conformal_factor_int));
  for (size_t i = 0; i < Dim; ++i) {
    field_p_boundary_correction->get(i) =
        normal_covector_int.get(i) *
        (get(log_conformal_factor_int) - log_conformal_factor_star);
  }

  // shift beta^j: vector. w^*_j = {w}_j - (c/2) R[ [w] ]_j.
  auto shift_jump = make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
      DataVector(num_points), 0.0);
  for (size_t j = 0; j < Dim; ++j) {
    shift_jump.get(j) = shift_ext.get(j) - shift_int.get(j);
  }
  auto reflected_shift_jump =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          DataVector(num_points), 0.0);
  for (size_t j = 0; j < Dim; ++j) {
    for (size_t k = 0; k < Dim; ++k) {
      reflected_shift_jump.get(j) +=
          gsl::at(gsl::at(reflection, j), k) * shift_jump.get(k);
    }
  }
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      const DataVector shift_star =
          0.5 * (shift_int.get(j) + shift_ext.get(j)) -
          0.5 * jump_coefficient * reflected_shift_jump.get(j);
      field_b_boundary_correction->get(i, j) =
          normal_covector_int.get(i) * (shift_int.get(j) - shift_star);
    }
  }

  // conformal metric gamma_tilde_jk: symmetric tensor. D reconstruction
  // carries an additional factor 1/2.
  auto metric_jump =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
          DataVector(num_points), 0.0);
  for (size_t j = 0; j < Dim; ++j) {
    for (size_t k = j; k < Dim; ++k) {
      metric_jump.get(j, k) =
          conformal_metric_ext.get(j, k) - conformal_metric_int.get(j, k);
    }
  }
  auto reflected_metric_jump =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
          DataVector(num_points), 0.0);
  for (size_t j = 0; j < Dim; ++j) {
    for (size_t k = j; k < Dim; ++k) {
      for (size_t a = 0; a < Dim; ++a) {
        for (size_t b = 0; b < Dim; ++b) {
          reflected_metric_jump.get(j, k) +=
              gsl::at(gsl::at(reflection, a), j) *
              gsl::at(gsl::at(reflection, b), k) * metric_jump.get(a, b);
        }
      }
    }
  }
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      for (size_t k = j; k < Dim; ++k) {
        const DataVector metric_star =
            0.5 * (conformal_metric_int.get(j, k) +
                   conformal_metric_ext.get(j, k)) -
            0.5 * jump_coefficient * reflected_metric_jump.get(j, k);
        field_d_boundary_correction->get(i, j, k) =
            0.5 * normal_covector_int.get(i) *
            (conformal_metric_int.get(j, k) - metric_star);
      }
    }
  }
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID ParityFlux<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                \
  template class ParityFlux<DIM(data)>;                                       \
  template void ParityFlux<DIM(data)>::dg_boundary_terms<false>(              \
      gsl::not_null<tnsr::ii<DataVector, DIM(data), Frame::Inertial>*>,       \
      gsl::not_null<Scalar<DataVector>*>,                                     \
      gsl::not_null<tnsr::ii<DataVector, DIM(data), Frame::Inertial>*>,       \
      gsl::not_null<Scalar<DataVector>*>, gsl::not_null<Scalar<DataVector>*>, \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,        \
      gsl::not_null<Scalar<DataVector>*>,                                     \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,        \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,        \
      gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>,        \
      gsl::not_null<tnsr::iJ<DataVector, DIM(data), Frame::Inertial>*>,       \
      gsl::not_null<tnsr::ijj<DataVector, DIM(data), Frame::Inertial>*>,      \
      gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>,        \
      gsl::not_null<tnsr::ii<DataVector, DIM(data), Frame::Inertial>*>,       \
      gsl::not_null<Scalar<DataVector>*>, gsl::not_null<Scalar<DataVector>*>, \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,        \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&,                                              \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&, const Scalar<DataVector>&,                   \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const Scalar<DataVector>&,                                              \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::iJ<DataVector, DIM(data), Frame::Inertial>&,                \
      const tnsr::ijj<DataVector, DIM(data), Frame::Inertial>&,               \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&,                                              \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&, const Scalar<DataVector>&,                   \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const Scalar<DataVector>&,                                              \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::iJ<DataVector, DIM(data), Frame::Inertial>&,                \
      const tnsr::ijj<DataVector, DIM(data), Frame::Inertial>&,               \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      evolution::dg::InterfaceOrientation, dg::Formulation) const;            \
  template void ParityFlux<DIM(data)>::dg_boundary_terms<true>(               \
      gsl::not_null<tnsr::ii<DataVector, DIM(data), Frame::Inertial>*>,       \
      gsl::not_null<Scalar<DataVector>*>,                                     \
      gsl::not_null<tnsr::ii<DataVector, DIM(data), Frame::Inertial>*>,       \
      gsl::not_null<Scalar<DataVector>*>, gsl::not_null<Scalar<DataVector>*>, \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,        \
      gsl::not_null<Scalar<DataVector>*>,                                     \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,        \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,        \
      gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>,        \
      gsl::not_null<tnsr::iJ<DataVector, DIM(data), Frame::Inertial>*>,       \
      gsl::not_null<tnsr::ijj<DataVector, DIM(data), Frame::Inertial>*>,      \
      gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>,        \
      gsl::not_null<tnsr::ii<DataVector, DIM(data), Frame::Inertial>*>,       \
      gsl::not_null<Scalar<DataVector>*>, gsl::not_null<Scalar<DataVector>*>, \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>,        \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&,                                              \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&, const Scalar<DataVector>&,                   \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const Scalar<DataVector>&,                                              \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::iJ<DataVector, DIM(data), Frame::Inertial>&,                \
      const tnsr::ijj<DataVector, DIM(data), Frame::Inertial>&,               \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&,                                              \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&, const Scalar<DataVector>&,                   \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const Scalar<DataVector>&,                                              \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::iJ<DataVector, DIM(data), Frame::Inertial>&,                \
      const tnsr::ijj<DataVector, DIM(data), Frame::Inertial>&,               \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      evolution::dg::InterfaceOrientation, dg::Formulation) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (3))

#undef INSTANTIATION
#undef DIM

}  // namespace Ccz4::BoundaryCorrections
