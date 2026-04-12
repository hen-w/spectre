// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/BoundaryCorrections/LaxFriedrichs.hpp"

#include <cmath>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::BoundaryCorrections {

template <size_t Dim>
LaxFriedrichs<Dim>::LaxFriedrichs(CkMigrateMessage* msg)
    : BoundaryCorrection(msg) {}

template <size_t Dim>
LaxFriedrichs<Dim>::LaxFriedrichs(const double tau1, const double tau2)
    : tau1_(tau1), tau2_(tau2) {}

template <size_t Dim>
std::unique_ptr<evolution::BoundaryCorrection> LaxFriedrichs<Dim>::get_clone()
    const {
  return std::make_unique<LaxFriedrichs>(*this);
}

template <size_t Dim>
void LaxFriedrichs<Dim>::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | tau1_;
  p | tau2_;
}

template <size_t Dim>
double LaxFriedrichs<Dim>::dg_package_data(
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
    gsl::not_null<Scalar<DataVector>*> /*packaged_u_scalar3_minus*/,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
    /*packaged_u_vector2_minus*/,
    gsl::not_null<Scalar<DataVector>*> /*packaged_u_scalar2_minus*/,
    gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
    /*packaged_u_tensor_minus*/,
    gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
    /*packaged_boundary_conformal_metric*/,
    gsl::not_null<Scalar<DataVector>*> /*packaged_boundary_conformal_factor*/,
    gsl::not_null<Scalar<DataVector>*> /*packaged_boundary_lapse*/,
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
    /*packaged_boundary_shift*/,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_normal_covector,
    gsl::not_null<Scalar<DataVector>*> packaged_inverse_grid_spacing,

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

    const Scalar<DataVector>& /*u_scalar3_minus*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*u_vector2_minus*/,
    const Scalar<DataVector>& /*u_scalar2_minus*/,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& /*u_tensor_minus*/,

    const tnsr::ii<DataVector, Dim, Frame::Inertial>&
    /*boundary_conformal_metric*/,
    const Scalar<DataVector>& /*boundary_conformal_factor*/,
    const Scalar<DataVector>& /*boundary_lapse*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /*boundary_shift*/,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
    const Direction<Dim>& face_direction, const Mesh<Dim>& volume_mesh,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& volume_coords) const {
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

  // Compute inverse grid spacing: 1/d where d = n̂ · (x_face - x_interior)
  const size_t dir = face_direction.dimension();
  const bool upper = (face_direction.side() == Side::Upper);
  const size_t N = volume_mesh.extents(dir);
  const size_t face_idx = upper ? N - 1 : 0;
  const size_t interior_idx = upper ? N - 2 : 1;
  const auto face_x =
      data_on_slice(volume_coords, volume_mesh.extents(), dir, face_idx);
  const auto interior_x =
      data_on_slice(volume_coords, volume_mesh.extents(), dir, interior_idx);
  DataVector d(face_x.get(0).size(), 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    d += normal_covector.get(i) * (face_x.get(i) - interior_x.get(i));
  }
  get(*packaged_inverse_grid_spacing) = abs(1.0 / d);

  return 0.0;
}

template <size_t Dim>
void LaxFriedrichs<Dim>::dg_boundary_terms(
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
    gsl::not_null<Scalar<DataVector>*> u_scalar3_minus_boundary_correction,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        u_vector2_minus_boundary_correction,
    gsl::not_null<Scalar<DataVector>*> u_scalar2_minus_boundary_correction,
    gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
        u_tensor_minus_boundary_correction,
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
    const Scalar<DataVector>& /*u_scalar3_minus_int*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*u_vector2_minus_int*/,
    const Scalar<DataVector>& /*u_scalar2_minus_int*/,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& /*u_tensor_minus_int*/,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>&
    /*boundary_conformal_metric_int*/,
    const Scalar<DataVector>& /*boundary_conformal_factor_int*/,
    const Scalar<DataVector>& /*boundary_lapse_int*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /*boundary_shift_int*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector_int,
    const Scalar<DataVector>& inverse_grid_spacing_int,

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
    const Scalar<DataVector>& /*u_scalar3_minus_ext*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*u_vector2_minus_ext*/,
    const Scalar<DataVector>& /*u_scalar2_minus_ext*/,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& /*u_tensor_minus_ext*/,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>&
    /*boundary_conformal_metric_ext*/,
    const Scalar<DataVector>& /*boundary_conformal_factor_ext*/,
    const Scalar<DataVector>& /*boundary_lapse_ext*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /*boundary_shift_ext*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector_ext,
    const Scalar<DataVector>& inverse_grid_spacing_ext,

    dg::Formulation /*dg_formulation*/) const {
  // Boundary mode corrections are always zero
  *u_scalar3_minus_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *u_vector2_minus_boundary_correction =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          conformal_factor_int, 0.0);
  *u_scalar2_minus_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *u_tensor_minus_boundary_correction =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
          conformal_factor_int, 0.0);
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

  constexpr double f_param = ::Ccz4::fd::System::f;
  Scalar<DataVector> tau1_eff;
  get(tau1_eff) =
      tau1_ * max(get(inverse_grid_spacing_int), get(inverse_grid_spacing_ext));
  // boundary corrections for conformal metric, conformal factor, lapse,
  // and shift are identically zero.
  *conformal_metric_boundary_correction =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
          conformal_metric_int, 0.0);
  *conformal_factor_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *lapse_boundary_correction =
      make_with_value<Scalar<DataVector>>(lapse_int, 0.0);
  *shift_boundary_correction =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(shift_int,
                                                                 0.0);

  // precompute common terms in the remaining boundary corrections
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

  // define lambdas to compute flux dot normal
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

  const auto field_a_flux_dot_normal =
      [](const Scalar<DataVector>& shift_dot_normal,
         const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a,
         const Scalar<DataVector>& trace_extrinsic_curvature,
         const Scalar<DataVector>& theta,
         const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector) {
        tnsr::i<DataVector, Dim, Frame::Inertial> result;
        ::tenex::evaluate<ti::k>(
            make_not_null(&result),
            -shift_dot_normal() * field_a(ti::k) +
                2.0 * normal_covector(ti::k) * trace_extrinsic_curvature() -
                4.0 * normal_covector(ti::k) * theta());
        return result;
      };

  const auto field_b_flux_dot_normal =
      [&f_param](
          const Scalar<DataVector>& shift_dot_normal,
          const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b,
          const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_shift_b,
          const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector) {
        tnsr::iJ<DataVector, Dim, Frame::Inertial> result;
        if constexpr (::Ccz4::fd::System::shifting_shift) {
          ::tenex::evaluate<ti::k, ti::I>(
              make_not_null(&result),
              -shift_dot_normal() * field_b(ti::k, ti::I) -
                  f_param * normal_covector(ti::k) * auxiliary_shift_b(ti::I));
        } else {
          ::tenex::evaluate<ti::k, ti::I>(
              make_not_null(&result),
              -f_param * normal_covector(ti::k) * auxiliary_shift_b(ti::I));
        }
        return result;
      };

  const auto field_d_flux_dot_normal =
      [](const Scalar<DataVector>& shift_dot_normal,
         const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d,
         const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric,
         const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b,
         const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
         const Scalar<DataVector>& lapse,
         const tnsr::ii<DataVector, Dim, Frame::Inertial>& a_tilde) {
        tnsr::ijj<DataVector, Dim, Frame::Inertial> result;
        ::tenex::evaluate<ti::k, ti::i, ti::j>(
            make_not_null(&result),
            -shift_dot_normal() * field_d(ti::k, ti::i, ti::j) -
                0.25 * conformal_metric(ti::l, ti::i) *
                    (normal_covector(ti::k) * field_b(ti::j, ti::L) +
                     normal_covector(ti::j) * field_b(ti::k, ti::L)) -
                0.25 * conformal_metric(ti::l, ti::j) *
                    (normal_covector(ti::k) * field_b(ti::i, ti::L) +
                     normal_covector(ti::i) * field_b(ti::k, ti::L)) +
                (1.0 / 6.0) * conformal_metric(ti::i, ti::j) *
                    (normal_covector(ti::k) * field_b(ti::l, ti::L) +
                     normal_covector(ti::l) * field_b(ti::k, ti::L)) +
                lapse() * normal_covector(ti::k) * a_tilde(ti::i, ti::j));
        return result;
      };

  const auto field_p_flux_dot_normal =
      [](const Scalar<DataVector>& shift_dot_normal,
         const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p,
         const Scalar<DataVector>& lapse,
         const Scalar<DataVector>& trace_extrinsic_curvature,
         const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b,
         const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector) {
        tnsr::i<DataVector, Dim, Frame::Inertial> result;
        ::tenex::evaluate<ti::k>(
            make_not_null(&result),
            -shift_dot_normal() * field_p(ti::k) -
                (lapse() / 3.0) * normal_covector(ti::k) *
                    trace_extrinsic_curvature() +
                (1.0 / 6.0) * normal_covector(ti::k) * field_b(ti::l, ti::L) +
                (1.0 / 6.0) * normal_covector(ti::l) * field_b(ti::k, ti::L));
        return result;
      };

  // compute boundary correction for trace_extrinsic_curvature
  const auto trace_extrinsic_curvature_flux_dot_normal_int = k_flux_dot_normal(
      normal_dot_shift_int, inverse_conformal_metric_dot_normal_int,
      trace_extrinsic_curvature_int, lapse_int, conformal_factor_squared_int,
      field_a_int, inverse_conformal_metric_int, field_d_int,
      gamma_hat_dot_normal_int, field_p_int);
  const auto trace_extrinsic_curvature_flux_dot_normal_ext = k_flux_dot_normal(
      normal_dot_shift_ext, inverse_conformal_metric_dot_normal_ext,
      trace_extrinsic_curvature_ext, lapse_ext, conformal_factor_squared_ext,
      field_a_ext, inverse_conformal_metric_ext, field_d_ext,
      gamma_hat_dot_normal_ext, field_p_ext);
  ::tenex::evaluate(trace_extrinsic_curvature_boundary_correction,
                    -0.5 * (trace_extrinsic_curvature_flux_dot_normal_ext() +
                            trace_extrinsic_curvature_flux_dot_normal_int()) -
                        0.5 * tau1_eff() *
                            (trace_extrinsic_curvature_ext() -
                             trace_extrinsic_curvature_int()));

  // compute boundary correction for a_tilde
  const auto a_tilde_flux_dot_normal_int = a_tilde_flux_dot_normal(
      normal_dot_shift_int, a_tilde_int, lapse_int,
      conformal_factor_squared_int, normal_covector_int, field_a_int,
      conformal_metric_int, field_d_int, gamma_hat_int,
      gamma_hat_dot_normal_int, field_p_int,
      inverse_conformal_metric_dot_normal_int, inverse_conformal_metric_int);
  const auto a_tilde_flux_dot_normal_ext = a_tilde_flux_dot_normal(
      normal_dot_shift_ext, a_tilde_ext, lapse_ext,
      conformal_factor_squared_ext, normal_covector_ext, field_a_ext,
      conformal_metric_ext, field_d_ext, gamma_hat_ext,
      gamma_hat_dot_normal_ext, field_p_ext,
      inverse_conformal_metric_dot_normal_ext, inverse_conformal_metric_ext);
  ::tenex::evaluate<ti::i, ti::j>(
      a_tilde_boundary_correction,
      -0.5 * (a_tilde_flux_dot_normal_ext(ti::i, ti::j) +
              a_tilde_flux_dot_normal_int(ti::i, ti::j)) -
          0.5 * tau1_eff() *
              (a_tilde_ext(ti::i, ti::j) - a_tilde_int(ti::i, ti::j)));

  // compute boundary correction for theta
  const auto theta_flux_dot_normal_int = theta_flux_dot_normal(
      normal_dot_shift_int, theta_int, lapse_int, conformal_factor_squared_int,
      inverse_conformal_metric_dot_normal_int, field_d_int,
      gamma_hat_dot_normal_int, field_p_int, inverse_conformal_metric_int);
  const auto theta_flux_dot_normal_ext = theta_flux_dot_normal(
      normal_dot_shift_ext, theta_ext, lapse_ext, conformal_factor_squared_ext,
      inverse_conformal_metric_dot_normal_ext, field_d_ext,
      gamma_hat_dot_normal_ext, field_p_ext, inverse_conformal_metric_ext);
  ::tenex::evaluate(
      theta_boundary_correction,
      -0.5 * (theta_flux_dot_normal_ext() + theta_flux_dot_normal_int()) -
          0.5 * tau1_eff() * (theta_ext() - theta_int()));

  // compute boundary correction for gamma_hat
  const auto gamma_hat_flux_dot_normal_int = gamma_hat_flux_dot_normal(
      normal_dot_shift_int, gamma_hat_int, lapse_int,
      inverse_conformal_metric_dot_normal_int, trace_extrinsic_curvature_int,
      theta_int, field_b_int, inverse_conformal_metric_int,
      normal_covector_int);
  const auto gamma_hat_flux_dot_normal_ext = gamma_hat_flux_dot_normal(
      normal_dot_shift_ext, gamma_hat_ext, lapse_ext,
      inverse_conformal_metric_dot_normal_ext, trace_extrinsic_curvature_ext,
      theta_ext, field_b_ext, inverse_conformal_metric_ext,
      normal_covector_ext);
  ::tenex::evaluate<ti::I>(
      gamma_hat_boundary_correction,
      -0.5 * (gamma_hat_flux_dot_normal_ext(ti::I) +
              gamma_hat_flux_dot_normal_int(ti::I)) -
          0.5 * tau1_eff() * (gamma_hat_ext(ti::I) - gamma_hat_int(ti::I)));

  // compute boundary correction for auxiliary_shift_b
  const auto b_flux_dot_normal_int = b_flux_dot_normal(
      normal_dot_shift_int, auxiliary_shift_b_int, gamma_hat_int, lapse_int,
      inverse_conformal_metric_dot_normal_int, trace_extrinsic_curvature_int,
      theta_int, field_b_int, inverse_conformal_metric_int,
      normal_covector_int);
  const auto b_flux_dot_normal_ext = b_flux_dot_normal(
      normal_dot_shift_ext, auxiliary_shift_b_ext, gamma_hat_ext, lapse_ext,
      inverse_conformal_metric_dot_normal_ext, trace_extrinsic_curvature_ext,
      theta_ext, field_b_ext, inverse_conformal_metric_ext,
      normal_covector_ext);
  ::tenex::evaluate<ti::I>(
      auxiliary_shift_b_boundary_correction,
      -0.5 * (b_flux_dot_normal_ext(ti::I) + b_flux_dot_normal_int(ti::I)) -
          0.5 * tau1_eff() *
              (auxiliary_shift_b_ext(ti::I) - auxiliary_shift_b_int(ti::I)));

  // compute boundary correction for field_a
  const auto field_a_flux_dot_normal_int = field_a_flux_dot_normal(
      normal_dot_shift_int, field_a_int, trace_extrinsic_curvature_int,
      theta_int, normal_covector_int);
  const auto field_a_flux_dot_normal_ext = field_a_flux_dot_normal(
      normal_dot_shift_ext, field_a_ext, trace_extrinsic_curvature_ext,
      theta_ext, normal_covector_ext);
  ::tenex::evaluate<ti::k>(
      field_a_boundary_correction,
      -0.5 * (field_a_flux_dot_normal_int(ti::k) +
              field_a_flux_dot_normal_ext(ti::k)) -
          0.5 * tau1_eff() * (field_a_ext(ti::k) - field_a_int(ti::k)));

  // compute boundary correction for field_b
  const auto field_b_flux_dot_normal_int =
      field_b_flux_dot_normal(normal_dot_shift_int, field_b_int,
                              auxiliary_shift_b_int, normal_covector_int);
  const auto field_b_flux_dot_normal_ext =
      field_b_flux_dot_normal(normal_dot_shift_ext, field_b_ext,
                              auxiliary_shift_b_ext, normal_covector_ext);
  ::tenex::evaluate<ti::k, ti::I>(
      field_b_boundary_correction,
      -0.5 * (field_b_flux_dot_normal_int(ti::k, ti::I) +
              field_b_flux_dot_normal_ext(ti::k, ti::I)) -
          0.5 * tau1_eff() *
              (field_b_ext(ti::k, ti::I) - field_b_int(ti::k, ti::I)));

  // compute boundary correction for field_d
  const auto field_d_flux_dot_normal_int = field_d_flux_dot_normal(
      normal_dot_shift_int, field_d_int, conformal_metric_int, field_b_int,
      normal_covector_int, lapse_int, a_tilde_int);
  const auto field_d_flux_dot_normal_ext = field_d_flux_dot_normal(
      normal_dot_shift_ext, field_d_ext, conformal_metric_ext, field_b_ext,
      normal_covector_ext, lapse_ext, a_tilde_ext);
  ::tenex::evaluate<ti::k, ti::i, ti::j>(
      field_d_boundary_correction,
      -0.5 * (field_d_flux_dot_normal_int(ti::k, ti::i, ti::j) +
              field_d_flux_dot_normal_ext(ti::k, ti::i, ti::j)) -
          0.5 * tau1_eff() *
              (field_d_ext(ti::k, ti::i, ti::j) -
               field_d_int(ti::k, ti::i, ti::j)));

  // compute boundary correction for field_p
  const auto field_p_flux_dot_normal_int = field_p_flux_dot_normal(
      normal_dot_shift_int, field_p_int, lapse_int,
      trace_extrinsic_curvature_int, field_b_int, normal_covector_int);
  const auto field_p_flux_dot_normal_ext = field_p_flux_dot_normal(
      normal_dot_shift_ext, field_p_ext, lapse_ext,
      trace_extrinsic_curvature_ext, field_b_ext, normal_covector_ext);
  ::tenex::evaluate<ti::k>(
      field_p_boundary_correction,
      -0.5 * (field_p_flux_dot_normal_int(ti::k) +
              field_p_flux_dot_normal_ext(ti::k)) -
          0.5 * tau1_eff() * (field_p_ext(ti::k) - field_p_int(ti::k)));
}

template <size_t Dim>
double LaxFriedrichs<Dim>::dg_auxiliary_package_data(
    gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
        packaged_conformal_metric,
    gsl::not_null<Scalar<DataVector>*> packaged_conformal_factor,
    gsl::not_null<Scalar<DataVector>*> packaged_lapse,
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> packaged_shift,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_normal_covector,
    gsl::not_null<Scalar<DataVector>*> packaged_inverse_grid_spacing,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> packaged_field_a,
    gsl::not_null<tnsr::iJ<DataVector, Dim, Frame::Inertial>*> packaged_field_b,
    gsl::not_null<tnsr::ijj<DataVector, Dim, Frame::Inertial>*>
        packaged_field_d,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> packaged_field_p,

    const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& /*a_tilde*/,
    const Scalar<DataVector>& /*trace_extrinsic_curvature*/,
    const Scalar<DataVector>& /*theta*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /*gamma_hat*/,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /*auxiliary_shift_b*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a,
    const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b,
    const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p,

    const Scalar<DataVector>& /*u_scalar3_minus*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*u_vector2_minus*/,
    const Scalar<DataVector>& /*u_scalar2_minus*/,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& /*u_tensor_minus*/,

    const tnsr::ii<DataVector, Dim, Frame::Inertial>&
    /*boundary_conformal_metric*/,
    const Scalar<DataVector>& /*boundary_conformal_factor*/,
    const Scalar<DataVector>& /*boundary_lapse*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /*boundary_shift*/,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
    const Direction<Dim>& face_direction, const Mesh<Dim>& volume_mesh,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& volume_coords) const {
  *packaged_conformal_metric = conformal_metric;
  *packaged_conformal_factor = conformal_factor;
  *packaged_lapse = lapse;
  *packaged_shift = shift;
  *packaged_normal_covector = normal_covector;
  *packaged_field_a = field_a;
  *packaged_field_b = field_b;
  *packaged_field_d = field_d;
  *packaged_field_p = field_p;

  // Compute inverse grid spacing: same as dg_package_data
  const size_t dir = face_direction.dimension();
  const bool upper = (face_direction.side() == Side::Upper);
  const size_t N = volume_mesh.extents(dir);
  const size_t face_idx = upper ? N - 1 : 0;
  const size_t interior_idx = upper ? N - 2 : 1;
  const auto face_x =
      data_on_slice(volume_coords, volume_mesh.extents(), dir, face_idx);
  const auto interior_x =
      data_on_slice(volume_coords, volume_mesh.extents(), dir, interior_idx);
  DataVector d(face_x.get(0).size(), 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    d += normal_covector.get(i) * (face_x.get(i) - interior_x.get(i));
  }
  get(*packaged_inverse_grid_spacing) = abs(1.0 / d);

  return 0.0;
}

template <size_t Dim>
void LaxFriedrichs<Dim>::dg_auxiliary_boundary_terms(
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
    gsl::not_null<Scalar<DataVector>*> u_scalar3_minus_boundary_correction,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        u_vector2_minus_boundary_correction,
    gsl::not_null<Scalar<DataVector>*> u_scalar2_minus_boundary_correction,
    gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
        u_tensor_minus_boundary_correction,
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
    const Scalar<DataVector>& inverse_grid_spacing_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a_int,
    const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b_int,
    const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p_int,

    const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric_ext,
    const Scalar<DataVector>& conformal_factor_ext,
    const Scalar<DataVector>& lapse_ext,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector_ext,
    const Scalar<DataVector>& inverse_grid_spacing_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a_ext,
    const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b_ext,
    const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p_ext,

    dg::Formulation /*dg_formulation*/) const {
  const DataVector tau2_eff =
      make_with_value<DataVector>(get(inverse_grid_spacing_int).size(), tau2_);
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

  // Boundary mode corrections are always zero
  *u_scalar3_minus_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *u_vector2_minus_boundary_correction =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          conformal_factor_int, 0.0);
  *u_scalar2_minus_boundary_correction =
      make_with_value<Scalar<DataVector>>(conformal_factor_int, 0.0);
  *u_tensor_minus_boundary_correction =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(
          conformal_factor_int, 0.0);
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

  // compute auxiliary boundary correction for field_a = d_log_lapse
  Scalar<DataVector> log_lapse_int;
  Scalar<DataVector> log_lapse_ext;
  get(log_lapse_int) = log(get(lapse_int));
  get(log_lapse_ext) = log(get(lapse_ext));
  for (size_t i = 0; i < Dim; ++i) {
    field_a_boundary_correction->get(i) =
        0.5 * (get(log_lapse_int) * normal_covector_int.get(i) +
               get(log_lapse_ext) * normal_covector_ext.get(i)) -
        0.5 * tau2_eff * (field_a_ext.get(i) - field_a_int.get(i));
  }

  // compute auxiliary boundary correction for field_b = d_shift
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      field_b_boundary_correction->get(i, j) =
          0.5 * (shift_int.get(j) * normal_covector_int.get(i) +
                 shift_ext.get(j) * normal_covector_ext.get(i)) -
          0.5 * tau2_eff * (field_b_ext.get(i, j) - field_b_int.get(i, j));
    }
  }

  // compute auxiliary boundary correction for field_d = 0.5 *
  // d_conformal_metric
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      for (size_t k = j; k < Dim; ++k) {
        field_d_boundary_correction->get(i, j, k) =
            0.5 * (0.5 * conformal_metric_int.get(j, k) *
                       normal_covector_int.get(i) +
                   0.5 * conformal_metric_ext.get(j, k) *
                       normal_covector_ext.get(i)) -
            0.5 * tau2_eff *
                (field_d_ext.get(i, j, k) - field_d_int.get(i, j, k));
      }
    }
  }

  // compute auxiliary boundary correction for field_p = d_log_conformal_factor
  Scalar<DataVector> log_conformal_factor_int;
  Scalar<DataVector> log_conformal_factor_ext;
  get(log_conformal_factor_int) = log(get(conformal_factor_int));
  get(log_conformal_factor_ext) = log(get(conformal_factor_ext));
  for (size_t i = 0; i < Dim; ++i) {
    field_p_boundary_correction->get(i) =
        0.5 * (get(log_conformal_factor_int) * normal_covector_int.get(i) +
               get(log_conformal_factor_ext) * normal_covector_ext.get(i)) -
        0.5 * tau2_eff * (field_p_ext.get(i) - field_p_int.get(i));
  }
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID LaxFriedrichs<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) template class LaxFriedrichs<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (3))

#undef INSTANTIATION
#undef DIM

}  // namespace Ccz4::BoundaryCorrections
