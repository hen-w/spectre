// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/Ccz4/BoundaryCorrections/LaxFriedrichs.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

constexpr size_t face_size = 25;  // 5x5 face

// Normal covector for upper_xi face: (1, 0, 0)
tnsr::i<DataVector, 3, Frame::Inertial> make_unit_normal(
    const size_t num_points) {
  auto n = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(num_points), 0.0);
  get<0>(n) = 1.0;
  return n;
}

// Helper to compute inverse of a 3x3 symmetric matrix at a single point
void invert_3x3_symmetric(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& metric,
    tnsr::II<DataVector, 3, Frame::Inertial>* inv_metric, const size_t q) {
  // Extract 3x3 matrix at point q
  std::array<std::array<double, 3>, 3> m{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      m[i][j] = metric.get(i, j)[q];
    }
  }
  // Compute determinant
  const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                     m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                     m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const double inv_det = 1.0 / det;
  // Cofactor matrix / det
  inv_metric->get(0, 0)[q] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
  inv_metric->get(0, 1)[q] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
  inv_metric->get(0, 2)[q] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
  inv_metric->get(1, 1)[q] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
  inv_metric->get(1, 2)[q] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
  inv_metric->get(2, 2)[q] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
  // Fill symmetric parts
  inv_metric->get(1, 0)[q] = inv_metric->get(0, 1)[q];
  inv_metric->get(2, 0)[q] = inv_metric->get(0, 2)[q];
  inv_metric->get(2, 1)[q] = inv_metric->get(1, 2)[q];
}

// Generate a well-conditioned conformal metric: delta_{ij} + eps * random_{ij}
// This keeps the metric near identity so its inverse stays O(1).
template <typename Generator>
tnsr::ii<DataVector, 3, Frame::Inertial> make_random_conformal_metric(
    const gsl::not_null<Generator*> gen, const size_t num_points) {
  std::uniform_real_distribution<> small_dist(-0.1, 0.1);
  auto perturbation =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          gen, small_dist, DataVector(num_points));
  auto result = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(num_points), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result.get(i, j) = perturbation.get(i, j);
      if (i == j) {
        result.get(i, j) += 1.0;
      }
    }
  }
  return result;
}

void test_dg_package_data() {
  const auto direction = Direction<3>::upper_xi();

  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> correction(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 2.0);

  // Create random face data (25 points)
  const auto conformal_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto conformal_factor = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto a_tilde =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto trace_extrinsic_curvature =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&gen), dist,
                                                  DataVector(face_size));
  const auto theta = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto gamma_hat =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto auxiliary_shift_b =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_a =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_d =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_p =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  // Boundary second-order fields (unused by LF)
  const auto boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto normal_covector = make_unit_normal(face_size);

  // Output packaged data
  auto pkg_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto pkg_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_a_tilde = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_gamma_hat = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_auxiliary_shift_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto pkg_field_a = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_field_b = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_field_d = make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_field_p = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_normal_covector =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);

  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;
  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
      std::nullopt;

  const double result = correction.dg_package_data(
      make_not_null(&pkg_conformal_metric),
      make_not_null(&pkg_conformal_factor), make_not_null(&pkg_a_tilde),
      make_not_null(&pkg_trace_extrinsic_curvature), make_not_null(&pkg_theta),
      make_not_null(&pkg_gamma_hat), make_not_null(&pkg_lapse),
      make_not_null(&pkg_shift), make_not_null(&pkg_auxiliary_shift_b),
      make_not_null(&pkg_field_a), make_not_null(&pkg_field_b),
      make_not_null(&pkg_field_d), make_not_null(&pkg_field_p),
      make_not_null(&pkg_normal_covector), conformal_metric,
      conformal_factor, a_tilde, trace_extrinsic_curvature, theta, gamma_hat,
      lapse, shift, auxiliary_shift_b, field_a, field_b, field_d, field_p,
      boundary_conformal_metric, boundary_conformal_factor, boundary_lapse,
      boundary_shift,
      normal_covector, mesh_velocity, normal_dot_mesh_velocity,
      direction);

  CHECK(result == 0.0);

  // All 13 fields are direct copies
  CHECK_ITERABLE_APPROX(pkg_conformal_metric, conformal_metric);
  CHECK_ITERABLE_APPROX(pkg_conformal_factor, conformal_factor);
  CHECK_ITERABLE_APPROX(pkg_a_tilde, a_tilde);
  CHECK_ITERABLE_APPROX(pkg_trace_extrinsic_curvature,
                        trace_extrinsic_curvature);
  CHECK_ITERABLE_APPROX(pkg_theta, theta);
  CHECK_ITERABLE_APPROX(pkg_gamma_hat, gamma_hat);
  CHECK_ITERABLE_APPROX(pkg_lapse, lapse);
  CHECK_ITERABLE_APPROX(pkg_shift, shift);
  CHECK_ITERABLE_APPROX(pkg_auxiliary_shift_b, auxiliary_shift_b);
  CHECK_ITERABLE_APPROX(pkg_field_a, field_a);
  CHECK_ITERABLE_APPROX(pkg_field_b, field_b);
  CHECK_ITERABLE_APPROX(pkg_field_d, field_d);
  CHECK_ITERABLE_APPROX(pkg_field_p, field_p);

  // Normal covector is copied
  CHECK_ITERABLE_APPROX(pkg_normal_covector, normal_covector);
}

void test_dg_auxiliary_package_data() {
  const auto direction = Direction<3>::upper_xi();

  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> correction(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 2.0);

  // Create all 13 evolved+aux fields on the face (only some are used)
  const auto conformal_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto conformal_factor = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto a_tilde =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto trace_extrinsic_curvature =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&gen), dist,
                                                  DataVector(face_size));
  const auto theta = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto gamma_hat =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto auxiliary_shift_b =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_a =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_d =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_p =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  // Boundary second-order fields (unused by LF aux)
  const auto boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto normal_covector = make_unit_normal(face_size);

  // Output packaged data (9 fields for auxiliary)
  auto pkg_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto pkg_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_normal_covector =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);

  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;
  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
      std::nullopt;

  const double result = correction.dg_auxiliary_package_data(
      make_not_null(&pkg_conformal_metric),
      make_not_null(&pkg_conformal_factor), make_not_null(&pkg_lapse),
      make_not_null(&pkg_shift), make_not_null(&pkg_normal_covector),
      conformal_metric, conformal_factor, a_tilde,
      trace_extrinsic_curvature, theta, gamma_hat, lapse, shift,
      auxiliary_shift_b, field_a, field_b, field_d, field_p,
      boundary_conformal_metric, boundary_conformal_factor, boundary_lapse,
      boundary_shift,
      normal_covector, mesh_velocity, normal_dot_mesh_velocity,
      direction);

  CHECK(result == 0.0);

  // Only conformal_metric, conformal_factor, lapse, shift, normal are packaged
  CHECK_ITERABLE_APPROX(pkg_conformal_metric, conformal_metric);
  CHECK_ITERABLE_APPROX(pkg_conformal_factor, conformal_factor);
  CHECK_ITERABLE_APPROX(pkg_lapse, lapse);
  CHECK_ITERABLE_APPROX(pkg_shift, shift);
  CHECK_ITERABLE_APPROX(pkg_normal_covector, normal_covector);
}

void test_dg_boundary_terms() {
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> correction(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  // Interior and exterior packaged data.
  // Use well-conditioned conformal metrics (near identity) so that inverse
  // values stay O(1), keeping floating-point round-off within default Approx.
  const auto conformal_metric_int =
      make_random_conformal_metric(make_not_null(&gen), face_size);
  const auto conformal_factor_int = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto a_tilde_int =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto trace_extrinsic_curvature_int =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&gen), dist,
                                                  DataVector(face_size));
  const auto theta_int = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto gamma_hat_int =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse_int = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift_int =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto auxiliary_shift_b_int =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_a_int =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b_int =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_d_int =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_p_int =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto normal_covector_int = make_unit_normal(face_size);

  const auto conformal_metric_ext =
      make_random_conformal_metric(make_not_null(&gen), face_size);
  const auto conformal_factor_ext = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto a_tilde_ext =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto trace_extrinsic_curvature_ext =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&gen), dist,
                                                  DataVector(face_size));
  const auto theta_ext = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto gamma_hat_ext =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse_ext = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift_ext =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto auxiliary_shift_b_ext =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_a_ext =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b_ext =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_d_ext =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_p_ext =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  // Exterior normal points opposite
  auto normal_covector_ext =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  get<0>(normal_covector_ext) = -1.0;

  // Output corrections
  auto corr_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_a_tilde = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto corr_trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_gamma_hat =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto corr_auxiliary_shift_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_field_a = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto corr_field_b = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto corr_field_d =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_field_p = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto corr_boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_boundary_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_boundary_shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);

  correction.dg_boundary_terms(
      make_not_null(&corr_conformal_metric),
      make_not_null(&corr_conformal_factor), make_not_null(&corr_a_tilde),
      make_not_null(&corr_trace_extrinsic_curvature),
      make_not_null(&corr_theta), make_not_null(&corr_gamma_hat),
      make_not_null(&corr_lapse), make_not_null(&corr_shift),
      make_not_null(&corr_auxiliary_shift_b), make_not_null(&corr_field_a),
      make_not_null(&corr_field_b), make_not_null(&corr_field_d),
      make_not_null(&corr_field_p),
      make_not_null(&corr_boundary_conformal_metric),
      make_not_null(&corr_boundary_conformal_factor),
      make_not_null(&corr_boundary_lapse), make_not_null(&corr_boundary_shift),
      conformal_metric_int, conformal_factor_int, a_tilde_int,
      trace_extrinsic_curvature_int, theta_int, gamma_hat_int, lapse_int,
      shift_int, auxiliary_shift_b_int, field_a_int, field_b_int, field_d_int,
      field_p_int, normal_covector_int, conformal_metric_ext,
      conformal_factor_ext, a_tilde_ext, trace_extrinsic_curvature_ext,
      theta_ext, gamma_hat_ext, lapse_ext, shift_ext, auxiliary_shift_b_ext,
      field_a_ext, field_b_ext, field_d_ext, field_p_ext,
      normal_covector_ext,
      dg::Formulation::StrongInertial);

  // --- Check zero corrections ---
  const auto zero_ii =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto zero_scalar =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto zero_I = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto zero_i = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto zero_iJ =
      make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto zero_ijj =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);

  CHECK_ITERABLE_APPROX(corr_conformal_metric, zero_ii);
  CHECK_ITERABLE_APPROX(corr_conformal_factor, zero_scalar);
  CHECK_ITERABLE_APPROX(corr_lapse, zero_scalar);
  CHECK_ITERABLE_APPROX(corr_shift, zero_I);

  // --- Check nonzero corrections using direct loop-based computation ---

  // Precompute inverse conformal metrics and derived quantities per point
  auto inv_cm_int = make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto inv_cm_ext = make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  for (size_t q = 0; q < face_size; ++q) {
    invert_3x3_symmetric(conformal_metric_int, &inv_cm_int, q);
    invert_3x3_symmetric(conformal_metric_ext, &inv_cm_ext, q);
  }

  // n_i dot shift
  DataVector n_dot_shift_int(face_size, 0.0);
  DataVector n_dot_shift_ext(face_size, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    n_dot_shift_int += normal_covector_int.get(i) * shift_int.get(i);
    n_dot_shift_ext += normal_covector_ext.get(i) * shift_ext.get(i);
  }

  // inverse_conformal_metric_dot_normal: g^{Ij} n_j
  auto icm_dot_n_int = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto icm_dot_n_ext = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  for (size_t I = 0; I < 3; ++I) {
    for (size_t j = 0; j < 3; ++j) {
      icm_dot_n_int.get(I) += inv_cm_int.get(I, j) * normal_covector_int.get(j);
      icm_dot_n_ext.get(I) += inv_cm_ext.get(I, j) * normal_covector_ext.get(j);
    }
  }

  // conformal_factor_squared
  const DataVector cf_sq_int =
      get(conformal_factor_int) * get(conformal_factor_int);
  const DataVector cf_sq_ext =
      get(conformal_factor_ext) * get(conformal_factor_ext);

  // gamma_hat dot normal
  DataVector gh_dot_n_int(face_size, 0.0);
  DataVector gh_dot_n_ext(face_size, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    gh_dot_n_int += gamma_hat_int.get(i) * normal_covector_int.get(i);
    gh_dot_n_ext += gamma_hat_ext.get(i) * normal_covector_ext.get(i);
  }

  // --- K boundary correction ---
  // k_flux_dot_normal = -n.beta * K + alpha * phi^2 * [
  //   g^{Ij}n_j * A_I + g^{IJ} D_{kij} g^{Kk}n_K - gamma_hat_dot_n
  //   - 4 * g^{Ij}n_j * P_i ]
  auto compute_k_flux_dot_normal =
      [&](const DataVector& n_dot_beta,
          const tnsr::I<DataVector, 3, Frame::Inertial>& icm_n,
          const Scalar<DataVector>& K, const Scalar<DataVector>& alpha,
          const DataVector& phi_sq,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fa,
          const tnsr::II<DataVector, 3, Frame::Inertial>& inv_g,
          const tnsr::ijj<DataVector, 3, Frame::Inertial>& fd,
          const DataVector& gh_n,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fp) {
        DataVector result(face_size, 0.0);
        // -n.beta * K
        result -= n_dot_beta * get(K);
        // + alpha * phi^2 * g^{Ij}n_j * A_I
        for (size_t I = 0; I < 3; ++I) {
          result += get(alpha) * phi_sq * icm_n.get(I) * fa.get(I);
        }
        // + alpha * phi^2 * g^{IJ} * D_{kij} * g^{Kk}n_K
        // = alpha * phi^2 * sum_{I,J,k} g^{IJ} * D_{k,I,J} * icm_n(k)
        // Note: the contraction is g^{IJ} * D_{k,i,j} * g^{Kk}n_K
        //  = sum_k icm_n(k) * sum_{I,J} g^{IJ} * D_{k,I,J}
        for (size_t k = 0; k < 3; ++k) {
          for (size_t I = 0; I < 3; ++I) {
            for (size_t J = 0; J < 3; ++J) {
              result += get(alpha) * phi_sq * inv_g.get(I, J) *
                        fd.get(k, I, J) * icm_n.get(k);
            }
          }
        }
        // - alpha * phi^2 * gamma_hat_dot_n
        result -= get(alpha) * phi_sq * gh_n;
        // - 4 * alpha * phi^2 * g^{Ij}n_j * P_I
        for (size_t I = 0; I < 3; ++I) {
          result -= 4.0 * get(alpha) * phi_sq * icm_n.get(I) * fp.get(I);
        }
        return result;
      };

  const DataVector k_flux_int = compute_k_flux_dot_normal(
      n_dot_shift_int, icm_dot_n_int, trace_extrinsic_curvature_int, lapse_int,
      cf_sq_int, field_a_int, inv_cm_int, field_d_int, gh_dot_n_int,
      field_p_int);
  const DataVector k_flux_ext = compute_k_flux_dot_normal(
      n_dot_shift_ext, icm_dot_n_ext, trace_extrinsic_curvature_ext, lapse_ext,
      cf_sq_ext, field_a_ext, inv_cm_ext, field_d_ext, gh_dot_n_ext,
      field_p_ext);

  auto expected_corr_K =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  get(expected_corr_K) = -0.5 * tau2 * (k_flux_ext + k_flux_int) -
                         0.5 * tau1 *
                             (get(trace_extrinsic_curvature_ext) -
                              get(trace_extrinsic_curvature_int));
  CHECK_ITERABLE_APPROX(corr_trace_extrinsic_curvature, expected_corr_K);

  // --- theta boundary correction ---
  auto compute_theta_flux_dot_normal =
      [&](const DataVector& n_dot_beta, const Scalar<DataVector>& th,
          const Scalar<DataVector>& alpha, const DataVector& phi_sq,
          const tnsr::I<DataVector, 3, Frame::Inertial>& icm_n,
          const tnsr::ijj<DataVector, 3, Frame::Inertial>& fd,
          const DataVector& gh_n,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fp,
          const tnsr::II<DataVector, 3, Frame::Inertial>& inv_g) {
        DataVector result(face_size, 0.0);
        // -n.beta * theta
        result -= n_dot_beta * get(th);
        // + 0.5 * alpha * phi^2 * g^{IJ} * D_{k,I,J} * icm_n(k)
        for (size_t k = 0; k < 3; ++k) {
          for (size_t I = 0; I < 3; ++I) {
            for (size_t J = 0; J < 3; ++J) {
              result += 0.5 * get(alpha) * phi_sq * inv_g.get(I, J) *
                        fd.get(k, I, J) * icm_n.get(k);
            }
          }
        }
        // - 0.5 * alpha * phi^2 * gamma_hat_dot_n
        result -= 0.5 * get(alpha) * phi_sq * gh_n;
        // - 0.5 * alpha * phi^2 * 4 * icm_n(I) * P_I
        for (size_t I = 0; I < 3; ++I) {
          result -= 0.5 * 4.0 * get(alpha) * phi_sq * icm_n.get(I) * fp.get(I);
        }
        return result;
      };

  const DataVector theta_flux_int = compute_theta_flux_dot_normal(
      n_dot_shift_int, theta_int, lapse_int, cf_sq_int, icm_dot_n_int,
      field_d_int, gh_dot_n_int, field_p_int, inv_cm_int);
  const DataVector theta_flux_ext = compute_theta_flux_dot_normal(
      n_dot_shift_ext, theta_ext, lapse_ext, cf_sq_ext, icm_dot_n_ext,
      field_d_ext, gh_dot_n_ext, field_p_ext, inv_cm_ext);

  auto expected_corr_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  get(expected_corr_theta) = -0.5 * tau2 * (theta_flux_ext + theta_flux_int) -
                             0.5 * tau1 * (get(theta_ext) - get(theta_int));
  CHECK_ITERABLE_APPROX(corr_theta, expected_corr_theta);

  // --- a_tilde boundary correction ---
  // F^{a_tilde}_{ij} . n = -n.beta * A~_{ij} + alpha * phi^2 * (...)
  auto compute_a_tilde_flux =
      [&](const DataVector& n_dot_beta,
          const tnsr::ii<DataVector, 3, Frame::Inertial>& at,
          const Scalar<DataVector>& alpha, const DataVector& phi_sq,
          const tnsr::i<DataVector, 3, Frame::Inertial>& n_cov,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fa,
          const tnsr::ii<DataVector, 3, Frame::Inertial>& cm,
          const tnsr::ijj<DataVector, 3, Frame::Inertial>& fd,
          const tnsr::I<DataVector, 3, Frame::Inertial>& gh,
          const DataVector& gh_n,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fp,
          const tnsr::I<DataVector, 3, Frame::Inertial>& icm_n,
          const tnsr::II<DataVector, 3, Frame::Inertial>& inv_g) {
        tnsr::ii<DataVector, 3, Frame::Inertial> result(face_size, 0.0);
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = i; j < 3; ++j) {
            auto& r = result.get(i, j);
            // -n.beta * A~_{ij}
            r -= n_dot_beta * at.get(i, j);

            const DataVector alp_phi2 = get(alpha) * phi_sq;

            // + 0.5 * n_i * A_j + 0.5 * n_j * A_i
            r += alp_phi2 * (0.5 * n_cov.get(i) * fa.get(j) +
                             0.5 * n_cov.get(j) * fa.get(i));

            // - g_{ij} * icm_n^K * A_K / 3
            DataVector icm_n_dot_fa(face_size, 0.0);
            for (size_t K = 0; K < 3; ++K) {
              icm_n_dot_fa += icm_n.get(K) * fa.get(K);
            }
            r -= alp_phi2 * cm.get(i, j) * icm_n_dot_fa / 3.0;

            // + icm_n^K * D_{K,i,j}
            for (size_t K = 0; K < 3; ++K) {
              r += alp_phi2 * icm_n.get(K) * fd.get(K, i, j);
            }

            // - g_{ij} * g^{MN} * icm_n^K * D_{K,M,N} / 3
            DataVector trace_term(face_size, 0.0);
            for (size_t K = 0; K < 3; ++K) {
              for (size_t M = 0; M < 3; ++M) {
                for (size_t N = 0; N < 3; ++N) {
                  trace_term +=
                      inv_g.get(M, N) * icm_n.get(K) * fd.get(K, M, N);
                }
              }
            }
            r -= alp_phi2 * cm.get(i, j) * trace_term / 3.0;

            // - 0.5 * n_i * g_{jk} * gamma_hat^K - 0.5 * n_j * g_{ik} *
            // gamma_hat^K
            for (size_t K = 0; K < 3; ++K) {
              r -= alp_phi2 * (0.5 * n_cov.get(i) * cm.get(j, K) * gh.get(K) +
                               0.5 * n_cov.get(j) * cm.get(i, K) * gh.get(K));
            }

            // + g_{ij} * gamma_hat_dot_normal / 3
            r += alp_phi2 * cm.get(i, j) * gh_n / 3.0;

            // - 0.5 * n_i * P_j - 0.5 * n_j * P_i
            r -= alp_phi2 * (0.5 * n_cov.get(i) * fp.get(j) +
                             0.5 * n_cov.get(j) * fp.get(i));

            // + g_{ij} * icm_n^K * P_K / 3
            DataVector icm_n_dot_fp(face_size, 0.0);
            for (size_t K = 0; K < 3; ++K) {
              icm_n_dot_fp += icm_n.get(K) * fp.get(K);
            }
            r += alp_phi2 * cm.get(i, j) * icm_n_dot_fp / 3.0;
          }
        }
        return result;
      };

  const auto at_flux_int = compute_a_tilde_flux(
      n_dot_shift_int, a_tilde_int, lapse_int, cf_sq_int, normal_covector_int,
      field_a_int, conformal_metric_int, field_d_int, gamma_hat_int,
      gh_dot_n_int, field_p_int, icm_dot_n_int, inv_cm_int);
  const auto at_flux_ext = compute_a_tilde_flux(
      n_dot_shift_ext, a_tilde_ext, lapse_ext, cf_sq_ext, normal_covector_ext,
      field_a_ext, conformal_metric_ext, field_d_ext, gamma_hat_ext,
      gh_dot_n_ext, field_p_ext, icm_dot_n_ext, inv_cm_ext);

  auto expected_corr_a_tilde =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      expected_corr_a_tilde.get(i, j) =
          -0.5 * tau2 * (at_flux_ext.get(i, j) + at_flux_int.get(i, j)) -
          0.5 * tau1 * (a_tilde_ext.get(i, j) - a_tilde_int.get(i, j));
    }
  }
  CHECK_ITERABLE_APPROX(corr_a_tilde, expected_corr_a_tilde);

  // --- gamma_hat boundary correction ---
  auto compute_gamma_hat_flux =
      [&](const DataVector& n_dot_beta,
          const tnsr::I<DataVector, 3, Frame::Inertial>& gh,
          const Scalar<DataVector>& alpha,
          const tnsr::I<DataVector, 3, Frame::Inertial>& icm_n,
          const Scalar<DataVector>& trace_K, const Scalar<DataVector>& th,
          const tnsr::iJ<DataVector, 3, Frame::Inertial>& fb,
          const tnsr::II<DataVector, 3, Frame::Inertial>& inv_g,
          const tnsr::i<DataVector, 3, Frame::Inertial>& n_cov) {
        tnsr::I<DataVector, 3, Frame::Inertial> result(face_size, 0.0);
        for (size_t I = 0; I < 3; ++I) {
          auto& r = result.get(I);
          // -n.beta * gamma_hat^I
          r -= n_dot_beta * gh.get(I);
          // + (4/3) * alpha * icm_n^I * K
          r += (4.0 / 3.0) * get(alpha) * icm_n.get(I) * get(trace_K);
          // - 2 * alpha * icm_n^I * theta
          r -= 2.0 * get(alpha) * icm_n.get(I) * get(th);
          // - icm_n^J * B_{j}^I
          for (size_t J = 0; J < 3; ++J) {
            r -= icm_n.get(J) * fb.get(J, I);
          }
          // - icm_n^I * B_{j}^J / 6
          for (size_t j = 0; j < 3; ++j) {
            r -= icm_n.get(I) * fb.get(j, j) / 6.0;
          }
          // - g^{IK} * B_{k}^J * n_j / 6
          for (size_t kk = 0; kk < 3; ++kk) {
            for (size_t j = 0; j < 3; ++j) {
              r -= inv_g.get(I, kk) * fb.get(kk, j) * n_cov.get(j) / 6.0;
            }
          }
        }
        return result;
      };

  const auto gh_flux_int = compute_gamma_hat_flux(
      n_dot_shift_int, gamma_hat_int, lapse_int, icm_dot_n_int,
      trace_extrinsic_curvature_int, theta_int, field_b_int, inv_cm_int,
      normal_covector_int);
  const auto gh_flux_ext = compute_gamma_hat_flux(
      n_dot_shift_ext, gamma_hat_ext, lapse_ext, icm_dot_n_ext,
      trace_extrinsic_curvature_ext, theta_ext, field_b_ext, inv_cm_ext,
      normal_covector_ext);

  auto expected_corr_gamma_hat =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t I = 0; I < 3; ++I) {
    expected_corr_gamma_hat.get(I) =
        -0.5 * tau2 * (gh_flux_ext.get(I) + gh_flux_int.get(I)) -
        0.5 * tau1 * (gamma_hat_ext.get(I) - gamma_hat_int.get(I));
  }
  CHECK_ITERABLE_APPROX(corr_gamma_hat, expected_corr_gamma_hat);

  // --- auxiliary_shift_b boundary correction ---
  // b_flux = gamma_hat_flux (since shifting_shift = false)
  auto expected_corr_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t I = 0; I < 3; ++I) {
    expected_corr_b.get(I) =
        -0.5 * tau2 * (gh_flux_ext.get(I) + gh_flux_int.get(I)) -
        0.5 * tau1 *
            (auxiliary_shift_b_ext.get(I) - auxiliary_shift_b_int.get(I));
  }
  CHECK_ITERABLE_APPROX(corr_auxiliary_shift_b, expected_corr_b);

  // --- field_a boundary correction ---
  // F^{field_a}_k . n = -n.beta * A_k + 2 * n_k * K - 4 * n_k * theta
  auto compute_field_a_flux =
      [&](const DataVector& n_dot_beta,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fa,
          const Scalar<DataVector>& K, const Scalar<DataVector>& th,
          const tnsr::i<DataVector, 3, Frame::Inertial>& n_cov) {
        tnsr::i<DataVector, 3, Frame::Inertial> result(face_size, 0.0);
        for (size_t k = 0; k < 3; ++k) {
          result.get(k) = -n_dot_beta * fa.get(k) +
                          2.0 * n_cov.get(k) * get(K) -
                          4.0 * n_cov.get(k) * get(th);
        }
        return result;
      };
  const auto fa_flux_int = compute_field_a_flux(n_dot_shift_int, field_a_int,
                                                trace_extrinsic_curvature_int,
                                                theta_int, normal_covector_int);
  const auto fa_flux_ext = compute_field_a_flux(n_dot_shift_ext, field_a_ext,
                                                trace_extrinsic_curvature_ext,
                                                theta_ext, normal_covector_ext);
  auto expected_corr_field_a =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t k = 0; k < 3; ++k) {
    expected_corr_field_a.get(k) =
        -0.5 * tau2 * (fa_flux_ext.get(k) + fa_flux_int.get(k)) -
        0.5 * tau1 * (field_a_ext.get(k) - field_a_int.get(k));
  }
  CHECK_ITERABLE_APPROX(corr_field_a, expected_corr_field_a);

  // --- field_b boundary correction ---
  // F^{field_b}_{k,I} . n = -n.beta * B_{k}^I - f * n_k * b^I
  // (shifting_shift branch; non-shifting omits the shift_dot_normal term)
  constexpr double f_param = Ccz4::fd::System::f;
  auto compute_field_b_flux =
      [&](const DataVector& n_dot_beta,
          const tnsr::iJ<DataVector, 3, Frame::Inertial>& fb,
          const tnsr::I<DataVector, 3, Frame::Inertial>& aux_b,
          const tnsr::i<DataVector, 3, Frame::Inertial>& n_cov) {
        tnsr::iJ<DataVector, 3, Frame::Inertial> result(face_size, 0.0);
        for (size_t k = 0; k < 3; ++k) {
          for (size_t I = 0; I < 3; ++I) {
            if constexpr (Ccz4::fd::System::shifting_shift) {
              result.get(k, I) = -n_dot_beta * fb.get(k, I) -
                                 f_param * n_cov.get(k) * aux_b.get(I);
            } else {
              result.get(k, I) = -f_param * n_cov.get(k) * aux_b.get(I);
            }
          }
        }
        return result;
      };
  const auto fb_flux_int = compute_field_b_flux(
      n_dot_shift_int, field_b_int, auxiliary_shift_b_int, normal_covector_int);
  const auto fb_flux_ext = compute_field_b_flux(
      n_dot_shift_ext, field_b_ext, auxiliary_shift_b_ext, normal_covector_ext);
  auto expected_corr_field_b =
      make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t k = 0; k < 3; ++k) {
    for (size_t I = 0; I < 3; ++I) {
      expected_corr_field_b.get(k, I) =
          -0.5 * tau2 * (fb_flux_ext.get(k, I) + fb_flux_int.get(k, I)) -
          0.5 * tau1 * (field_b_ext.get(k, I) - field_b_int.get(k, I));
    }
  }
  CHECK_ITERABLE_APPROX(corr_field_b, expected_corr_field_b);

  // --- field_d boundary correction ---
  // Complex flux involving field_d, conformal_metric, field_b, lapse, a_tilde
  auto compute_field_d_flux =
      [&](const DataVector& n_dot_beta,
          const tnsr::ijj<DataVector, 3, Frame::Inertial>& fd,
          const tnsr::ii<DataVector, 3, Frame::Inertial>& cm,
          const tnsr::iJ<DataVector, 3, Frame::Inertial>& fb,
          const tnsr::i<DataVector, 3, Frame::Inertial>& n_cov,
          const Scalar<DataVector>& alpha,
          const tnsr::ii<DataVector, 3, Frame::Inertial>& at) {
        tnsr::ijj<DataVector, 3, Frame::Inertial> result(face_size, 0.0);
        // contracted_field_b = B_l^L (trace)
        DataVector contracted_fb(face_size, 0.0);
        for (size_t l = 0; l < 3; ++l) {
          contracted_fb += fb.get(l, l);
        }
        for (size_t k = 0; k < 3; ++k) {
          for (size_t i = 0; i < 3; ++i) {
            for (size_t j = i; j < 3; ++j) {
              auto& r = result.get(k, i, j);
              r = -n_dot_beta * fd.get(k, i, j);
              // -0.25 * g_{l,i} * (n_k * B_j^l + n_j * B_k^l)
              for (size_t l = 0; l < 3; ++l) {
                r -=
                    0.25 * cm.get(l, i) *
                    (n_cov.get(k) * fb.get(j, l) + n_cov.get(j) * fb.get(k, l));
                // -0.25 * g_{l,j} * (n_k * B_i^l + n_i * B_k^l)
                r -=
                    0.25 * cm.get(l, j) *
                    (n_cov.get(k) * fb.get(i, l) + n_cov.get(i) * fb.get(k, l));
              }
              // + (1/6) * g_{ij} * (n_k * B_l^L + n_l * B_k^L)
              r += (1.0 / 6.0) * cm.get(i, j) * (n_cov.get(k) * contracted_fb);
              for (size_t l = 0; l < 3; ++l) {
                r += (1.0 / 6.0) * cm.get(i, j) * n_cov.get(l) * fb.get(k, l);
              }
              // + alpha * n_k * A~_{ij}
              r += get(alpha) * n_cov.get(k) * at.get(i, j);
            }
          }
        }
        return result;
      };
  const auto fd_flux_int = compute_field_d_flux(
      n_dot_shift_int, field_d_int, conformal_metric_int, field_b_int,
      normal_covector_int, lapse_int, a_tilde_int);
  const auto fd_flux_ext = compute_field_d_flux(
      n_dot_shift_ext, field_d_ext, conformal_metric_ext, field_b_ext,
      normal_covector_ext, lapse_ext, a_tilde_ext);
  auto expected_corr_field_d =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        expected_corr_field_d.get(k, i, j) =
            -0.5 * tau2 *
                (fd_flux_ext.get(k, i, j) + fd_flux_int.get(k, i, j)) -
            0.5 * tau1 *
                (field_d_ext.get(k, i, j) - field_d_int.get(k, i, j));
      }
    }
  }
  CHECK_ITERABLE_APPROX(corr_field_d, expected_corr_field_d);

  // --- field_p boundary correction ---
  // F^{field_p}_k . n = -n.beta * P_k - (alpha/3)*n_k*K
  //   + (1/6)*n_k*B_l^L + (1/6)*n_l*B_k^L
  auto compute_field_p_flux =
      [&](const DataVector& n_dot_beta,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fp,
          const Scalar<DataVector>& alpha, const Scalar<DataVector>& K,
          const tnsr::iJ<DataVector, 3, Frame::Inertial>& fb,
          const tnsr::i<DataVector, 3, Frame::Inertial>& n_cov) {
        tnsr::i<DataVector, 3, Frame::Inertial> result(face_size, 0.0);
        DataVector contracted_fb(face_size, 0.0);
        for (size_t l = 0; l < 3; ++l) {
          contracted_fb += fb.get(l, l);
        }
        for (size_t k = 0; k < 3; ++k) {
          result.get(k) = -n_dot_beta * fp.get(k) -
                          (get(alpha) / 3.0) * n_cov.get(k) * get(K) +
                          (1.0 / 6.0) * n_cov.get(k) * contracted_fb;
          for (size_t l = 0; l < 3; ++l) {
            result.get(k) += (1.0 / 6.0) * n_cov.get(l) * fb.get(k, l);
          }
        }
        return result;
      };
  const auto fp_flux_int = compute_field_p_flux(
      n_dot_shift_int, field_p_int, lapse_int, trace_extrinsic_curvature_int,
      field_b_int, normal_covector_int);
  const auto fp_flux_ext = compute_field_p_flux(
      n_dot_shift_ext, field_p_ext, lapse_ext, trace_extrinsic_curvature_ext,
      field_b_ext, normal_covector_ext);
  auto expected_corr_field_p =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t k = 0; k < 3; ++k) {
    expected_corr_field_p.get(k) =
        -0.5 * tau2 * (fp_flux_ext.get(k) + fp_flux_int.get(k)) -
        0.5 * tau1 * (field_p_ext.get(k) - field_p_int.get(k));
  }
  CHECK_ITERABLE_APPROX(corr_field_p, expected_corr_field_p);
}

void test_dg_auxiliary_boundary_terms() {
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> correction(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 2.0);

  // Interior packaged aux data
  const auto conformal_metric_int =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto conformal_factor_int = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse_int = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift_int =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto normal_covector_int = make_unit_normal(face_size);

  // Exterior packaged aux data
  const auto conformal_metric_ext =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto conformal_factor_ext = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse_ext = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift_ext =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  auto normal_covector_ext =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  get<0>(normal_covector_ext) = -1.0;

  // Output corrections (all 17 evolved+aux+boundary fields)
  auto corr_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_a_tilde = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto corr_trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_gamma_hat =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto corr_auxiliary_shift_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_field_a = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto corr_field_b = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto corr_field_d =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_field_p = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto corr_boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_boundary_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_boundary_shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);

  correction.dg_auxiliary_boundary_terms(
      make_not_null(&corr_conformal_metric),
      make_not_null(&corr_conformal_factor), make_not_null(&corr_a_tilde),
      make_not_null(&corr_trace_extrinsic_curvature),
      make_not_null(&corr_theta), make_not_null(&corr_gamma_hat),
      make_not_null(&corr_lapse), make_not_null(&corr_shift),
      make_not_null(&corr_auxiliary_shift_b), make_not_null(&corr_field_a),
      make_not_null(&corr_field_b), make_not_null(&corr_field_d),
      make_not_null(&corr_field_p),
      make_not_null(&corr_boundary_conformal_metric),
      make_not_null(&corr_boundary_conformal_factor),
      make_not_null(&corr_boundary_lapse), make_not_null(&corr_boundary_shift),
      conformal_metric_int, conformal_factor_int, lapse_int, shift_int,
      normal_covector_int, conformal_metric_ext, conformal_factor_ext,
      lapse_ext, shift_ext, normal_covector_ext,
      dg::Formulation::StrongInertial);

  // --- Check zero corrections ---
  const auto zero_ii =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto zero_scalar =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto zero_I = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);

  CHECK_ITERABLE_APPROX(corr_conformal_metric, zero_ii);
  CHECK_ITERABLE_APPROX(corr_conformal_factor, zero_scalar);
  CHECK_ITERABLE_APPROX(corr_a_tilde, zero_ii);
  CHECK_ITERABLE_APPROX(corr_trace_extrinsic_curvature, zero_scalar);
  CHECK_ITERABLE_APPROX(corr_theta, zero_scalar);
  CHECK_ITERABLE_APPROX(corr_gamma_hat, zero_I);
  CHECK_ITERABLE_APPROX(corr_lapse, zero_scalar);
  CHECK_ITERABLE_APPROX(corr_shift, zero_I);
  CHECK_ITERABLE_APPROX(corr_auxiliary_shift_b, zero_I);

  // --- Check nonzero auxiliary corrections ---
  // The auxiliary pass uses pure central flux (no tau1/tau2 penalties).

  const DataVector log_lapse_int = log(get(lapse_int));
  const DataVector log_lapse_ext = log(get(lapse_ext));
  const DataVector log_cf_int = log(get(conformal_factor_int));
  const DataVector log_cf_ext = log(get(conformal_factor_ext));

  // field_a correction: central flux uses log(lapse), penalty uses field_a
  auto expected_corr_field_a =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    expected_corr_field_a.get(i) =
        0.5 * (log_lapse_int * normal_covector_int.get(i) +
               log_lapse_ext * normal_covector_ext.get(i));
  }
  CHECK_ITERABLE_APPROX(corr_field_a, expected_corr_field_a);

  // field_b correction: central flux uses shift, penalty uses field_b
  auto expected_corr_field_b =
      make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t J = 0; J < 3; ++J) {
      expected_corr_field_b.get(i, J) =
          0.5 * (shift_int.get(J) * normal_covector_int.get(i) +
                 shift_ext.get(J) * normal_covector_ext.get(i));
    }
  }
  CHECK_ITERABLE_APPROX(corr_field_b, expected_corr_field_b);

  // field_d correction: central flux uses 0.5*conformal_metric, penalty uses
  // field_d
  auto expected_corr_field_d =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = j; k < 3; ++k) {
        expected_corr_field_d.get(i, j, k) =
            0.5 * (0.5 * conformal_metric_int.get(j, k) *
                       normal_covector_int.get(i) +
                   0.5 * conformal_metric_ext.get(j, k) *
                       normal_covector_ext.get(i));
      }
    }
  }
  CHECK_ITERABLE_APPROX(corr_field_d, expected_corr_field_d);

  // field_p correction: central flux uses log(conformal_factor), penalty uses
  // field_p
  auto expected_corr_field_p =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    expected_corr_field_p.get(i) =
        0.5 * (log_cf_int * normal_covector_int.get(i) +
               log_cf_ext * normal_covector_ext.get(i));
  }
  CHECK_ITERABLE_APPROX(corr_field_p, expected_corr_field_p);
}

void test_use_central_flux_at_boundary_false() {
  // When use_central_flux_at_boundary is false, the ForExternalBoundary=true
  // code path should use the user-specified tau1/tau2 instead of 0.0/1.0.
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> correction_central(
      tau1, tau2, true);
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> correction_no_central(
      tau1, tau2, false);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  const auto conformal_metric_int =
      make_random_conformal_metric(make_not_null(&gen), face_size);
  const auto conformal_factor_int = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto a_tilde_int =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto trace_extrinsic_curvature_int =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&gen), dist,
                                                  DataVector(face_size));
  const auto theta_int = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto gamma_hat_int =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse_int = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift_int =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto auxiliary_shift_b_int =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_a_int =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b_int =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_d_int =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_p_int =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto normal_covector_int = make_unit_normal(face_size);

  const auto conformal_metric_ext =
      make_random_conformal_metric(make_not_null(&gen), face_size);
  const auto conformal_factor_ext = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto a_tilde_ext =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto trace_extrinsic_curvature_ext =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&gen), dist,
                                                  DataVector(face_size));
  const auto theta_ext = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto gamma_hat_ext =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse_ext = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift_ext =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto auxiliary_shift_b_ext =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_a_ext =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b_ext =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_d_ext =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_p_ext =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  auto normal_covector_ext =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  get<0>(normal_covector_ext) = -1.0;

  // Call ForExternalBoundary=true with use_central_flux_at_boundary=true
  auto corr_K_central =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_dummy_ii =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_dummy_scalar =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_dummy_I =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_dummy_i =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_dummy_iJ =
      make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto corr_dummy_ijj =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);

  correction_central.dg_boundary_terms<true>(
      make_not_null(&corr_dummy_ii), make_not_null(&corr_dummy_scalar),
      make_not_null(&corr_dummy_ii), make_not_null(&corr_K_central),
      make_not_null(&corr_dummy_scalar), make_not_null(&corr_dummy_I),
      make_not_null(&corr_dummy_scalar), make_not_null(&corr_dummy_I),
      make_not_null(&corr_dummy_I), make_not_null(&corr_dummy_i),
      make_not_null(&corr_dummy_iJ), make_not_null(&corr_dummy_ijj),
      make_not_null(&corr_dummy_i), make_not_null(&corr_dummy_ii),
      make_not_null(&corr_dummy_scalar), make_not_null(&corr_dummy_scalar),
      make_not_null(&corr_dummy_I),
      conformal_metric_int, conformal_factor_int, a_tilde_int,
      trace_extrinsic_curvature_int, theta_int, gamma_hat_int, lapse_int,
      shift_int, auxiliary_shift_b_int, field_a_int, field_b_int, field_d_int,
      field_p_int, normal_covector_int, conformal_metric_ext,
      conformal_factor_ext, a_tilde_ext, trace_extrinsic_curvature_ext,
      theta_ext, gamma_hat_ext, lapse_ext, shift_ext, auxiliary_shift_b_ext,
      field_a_ext, field_b_ext, field_d_ext, field_p_ext,
      normal_covector_ext, dg::Formulation::StrongInertial);

  // Call ForExternalBoundary=true with use_central_flux_at_boundary=false
  auto corr_K_no_central =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  // Reset dummies
  corr_dummy_ii = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  corr_dummy_scalar =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  corr_dummy_I = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  corr_dummy_i = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  corr_dummy_iJ = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  corr_dummy_ijj = make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);

  correction_no_central.dg_boundary_terms<true>(
      make_not_null(&corr_dummy_ii), make_not_null(&corr_dummy_scalar),
      make_not_null(&corr_dummy_ii), make_not_null(&corr_K_no_central),
      make_not_null(&corr_dummy_scalar), make_not_null(&corr_dummy_I),
      make_not_null(&corr_dummy_scalar), make_not_null(&corr_dummy_I),
      make_not_null(&corr_dummy_I), make_not_null(&corr_dummy_i),
      make_not_null(&corr_dummy_iJ), make_not_null(&corr_dummy_ijj),
      make_not_null(&corr_dummy_i), make_not_null(&corr_dummy_ii),
      make_not_null(&corr_dummy_scalar), make_not_null(&corr_dummy_scalar),
      make_not_null(&corr_dummy_I),
      conformal_metric_int, conformal_factor_int, a_tilde_int,
      trace_extrinsic_curvature_int, theta_int, gamma_hat_int, lapse_int,
      shift_int, auxiliary_shift_b_int, field_a_int, field_b_int, field_d_int,
      field_p_int, normal_covector_int, conformal_metric_ext,
      conformal_factor_ext, a_tilde_ext, trace_extrinsic_curvature_ext,
      theta_ext, gamma_hat_ext, lapse_ext, shift_ext, auxiliary_shift_b_ext,
      field_a_ext, field_b_ext, field_d_ext, field_p_ext,
      normal_covector_ext, dg::Formulation::StrongInertial);

  // Call ForExternalBoundary=false (interior) with no_central — should match
  // no_central at external boundary since both use tau1/tau2
  auto corr_K_interior =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  corr_dummy_ii = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  corr_dummy_scalar =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  corr_dummy_I = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  corr_dummy_i = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  corr_dummy_iJ = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  corr_dummy_ijj = make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);

  correction_no_central.dg_boundary_terms(
      make_not_null(&corr_dummy_ii), make_not_null(&corr_dummy_scalar),
      make_not_null(&corr_dummy_ii), make_not_null(&corr_K_interior),
      make_not_null(&corr_dummy_scalar), make_not_null(&corr_dummy_I),
      make_not_null(&corr_dummy_scalar), make_not_null(&corr_dummy_I),
      make_not_null(&corr_dummy_I), make_not_null(&corr_dummy_i),
      make_not_null(&corr_dummy_iJ), make_not_null(&corr_dummy_ijj),
      make_not_null(&corr_dummy_i), make_not_null(&corr_dummy_ii),
      make_not_null(&corr_dummy_scalar), make_not_null(&corr_dummy_scalar),
      make_not_null(&corr_dummy_I),
      conformal_metric_int, conformal_factor_int, a_tilde_int,
      trace_extrinsic_curvature_int, theta_int, gamma_hat_int, lapse_int,
      shift_int, auxiliary_shift_b_int, field_a_int, field_b_int, field_d_int,
      field_p_int, normal_covector_int, conformal_metric_ext,
      conformal_factor_ext, a_tilde_ext, trace_extrinsic_curvature_ext,
      theta_ext, gamma_hat_ext, lapse_ext, shift_ext, auxiliary_shift_b_ext,
      field_a_ext, field_b_ext, field_d_ext, field_p_ext,
      normal_covector_ext, dg::Formulation::StrongInertial);

  // With use_central_flux_at_boundary=false, external boundary should match
  // interior (both use tau1/tau2)
  CHECK_ITERABLE_APPROX(corr_K_no_central, corr_K_interior);

  // The central and no_central results should differ (since tau1!=0, tau2!=1)
  // Just check they are NOT equal at at least one point
  bool differs = false;
  for (size_t q = 0; q < face_size; ++q) {
    if (get(corr_K_central)[q] != get(corr_K_no_central)[q]) {
      differs = true;
      break;
    }
  }
  CHECK(differs);
}

void test_serialization() {
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> correction(1.5, 2.3,
                                                                false);
  const auto deserialized = serialize_and_deserialize(correction);
  // Verify it works by running a simple package_data call
  // (if pup restored tau1_ and tau2_ correctly, the computation will match)
  const auto direction = Direction<3>::upper_xi();
  const auto normal = make_unit_normal(face_size);

  auto pkg_cm = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_cf = make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_at = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_K = make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_gh = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_b = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_fa = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_fb = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_fd = make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_fp = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_n = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);

  // Use simple constant inputs
  const auto cm = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 1.0);
  const auto cf =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 1.0);
  const auto at = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.5);
  const auto K =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.1);
  const auto theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.2);
  const auto gh = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.3);
  const auto lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 1.0);
  const auto shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto b = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto fa = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto fb = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto fd = make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto fp = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  // Boundary SO fields (zero)
  const auto bcm = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto bcf =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto blapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto bshift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);

  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;
  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
      std::nullopt;

  // Call on original
  const double result_orig = correction.dg_package_data(
      make_not_null(&pkg_cm), make_not_null(&pkg_cf), make_not_null(&pkg_at),
      make_not_null(&pkg_K), make_not_null(&pkg_theta), make_not_null(&pkg_gh),
      make_not_null(&pkg_lapse), make_not_null(&pkg_shift),
      make_not_null(&pkg_b), make_not_null(&pkg_fa), make_not_null(&pkg_fb),
      make_not_null(&pkg_fd), make_not_null(&pkg_fp),
      make_not_null(&pkg_n),
      cm, cf, at, K, theta, gh, lapse, shift, b, fa,
      fb, fd, fp, bcm, bcf, blapse, bshift, normal,
      mesh_velocity, normal_dot_mesh_velocity, direction);
  const auto pkg_n_orig = pkg_n;

  // Reset and call on deserialized
  for (auto& component : pkg_n) {
    component = 0.0;
  }
  const double result_deser = deserialized.dg_package_data(
      make_not_null(&pkg_cm), make_not_null(&pkg_cf), make_not_null(&pkg_at),
      make_not_null(&pkg_K), make_not_null(&pkg_theta), make_not_null(&pkg_gh),
      make_not_null(&pkg_lapse), make_not_null(&pkg_shift),
      make_not_null(&pkg_b), make_not_null(&pkg_fa), make_not_null(&pkg_fb),
      make_not_null(&pkg_fd), make_not_null(&pkg_fp),
      make_not_null(&pkg_n),
      cm, cf, at, K, theta, gh, lapse, shift, b, fa,
      fb, fd, fp, bcm, bcf, blapse, bshift, normal,
      mesh_velocity, normal_dot_mesh_velocity, direction);

  CHECK(result_orig == result_deser);
  CHECK_ITERABLE_APPROX(pkg_n, pkg_n_orig);
}

// Regression test: verify dg_package_data writes all components of the
// packaged Variables buffer (no leftover sNaN from uninitialized memory).
void test_dg_package_data_all_components_written() {
  using package_tags =
      Ccz4::BoundaryCorrections::LaxFriedrichs<3>::dg_package_field_tags;
  Variables<package_tags> packaged_vars(face_size);

  const auto direction = Direction<3>::upper_xi();
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> correction(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 2.0);

  const auto conformal_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto conformal_factor = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto a_tilde =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto trace_extrinsic_curvature =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&gen), dist,
                                                  DataVector(face_size));
  const auto theta = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto gamma_hat =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto auxiliary_shift_b =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_a =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_d =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_p =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto normal_covector = make_unit_normal(face_size);
  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;
  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
      std::nullopt;

  correction.dg_package_data(
      make_not_null(
          &get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::ConformalFactor<DataVector>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::ATilde<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(packaged_vars)),
      make_not_null(&get<::Ccz4::Tags::Theta<DataVector>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::GammaHat<DataVector, 3>>(packaged_vars)),
      make_not_null(&get<gr::Tags::Lapse<DataVector>>(packaged_vars)),
      make_not_null(&get<gr::Tags::Shift<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::FieldA<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::FieldB<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::FieldD<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::FieldP<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::NormalCovector<DataVector, 3>>(packaged_vars)),
      conformal_metric, conformal_factor, a_tilde, trace_extrinsic_curvature,
      theta, gamma_hat, lapse, shift, auxiliary_shift_b, field_a, field_b,
      field_d, field_p, boundary_conformal_metric, boundary_conformal_factor,
      boundary_lapse, boundary_shift, normal_covector, mesh_velocity,
      normal_dot_mesh_velocity, direction);

  for (size_t i = 0; i < packaged_vars.size(); ++i) {
    CAPTURE(i);
    CHECK(not std::isnan(packaged_vars.data()[i]));
  }
}

// Regression test: verify dg_auxiliary_package_data writes all components of
// the packaged Variables buffer (no leftover sNaN from uninitialized memory).
void test_dg_auxiliary_package_data_all_components_written() {
  using aux_package_tags =
      Ccz4::BoundaryCorrections::LaxFriedrichs<3>::dg_auxiliary_package_field_tags;
  Variables<aux_package_tags> packaged_vars(face_size);

  const auto direction = Direction<3>::upper_xi();
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> correction(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 2.0);

  const auto conformal_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto conformal_factor = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto a_tilde =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto trace_extrinsic_curvature =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&gen), dist,
                                                  DataVector(face_size));
  const auto theta = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto gamma_hat =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto auxiliary_shift_b =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_a =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_d =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_p =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto normal_covector = make_unit_normal(face_size);
  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;
  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
      std::nullopt;

  correction.dg_auxiliary_package_data(
      make_not_null(
          &get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::ConformalFactor<DataVector>>(packaged_vars)),
      make_not_null(&get<gr::Tags::Lapse<DataVector>>(packaged_vars)),
      make_not_null(&get<gr::Tags::Shift<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::NormalCovector<DataVector, 3>>(packaged_vars)),
      conformal_metric, conformal_factor, a_tilde, trace_extrinsic_curvature,
      theta, gamma_hat, lapse, shift, auxiliary_shift_b, field_a, field_b,
      field_d, field_p, boundary_conformal_metric, boundary_conformal_factor,
      boundary_lapse, boundary_shift, normal_covector, mesh_velocity,
      normal_dot_mesh_velocity, direction);

  for (size_t i = 0; i < packaged_vars.size(); ++i) {
    CAPTURE(i);
    CHECK(not std::isnan(packaged_vars.data()[i]));
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.BoundaryCorrections.LaxFriedrichs",
    "[Unit][Evolution]") {
  SECTION("dg_package_data") { test_dg_package_data(); }
  SECTION("dg_auxiliary_package_data") { test_dg_auxiliary_package_data(); }
  SECTION("dg_boundary_terms") { test_dg_boundary_terms(); }
  SECTION("dg_auxiliary_boundary_terms") { test_dg_auxiliary_boundary_terms(); }
  SECTION("use_central_flux_at_boundary_false") {
    test_use_central_flux_at_boundary_false();
  }
  SECTION("serialization") { test_serialization(); }
  SECTION("dg_package_data_all_components_written") {
    test_dg_package_data_all_components_written();
  }
  SECTION("dg_auxiliary_package_data_all_components_written") {
    test_dg_auxiliary_package_data_all_components_written();
  }
}
