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
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/Ccz4/BoundaryCorrections/LaxFriedrichs.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

// Analytic 5-point GLL nodes on [-1,1]
const double gll_node_0 = -1.0;
const double gll_node_1 = -std::sqrt(3.0 / 7.0);
const double gll_node_2 = 0.0;
const double gll_node_3 = std::sqrt(3.0 / 7.0);
const double gll_node_4 = 1.0;

// For upper_xi face: face at x=1, interior at x=sqrt(3/7)
// d = 1 - sqrt(3/7), inverse_grid_spacing = 1/d
const double analytic_inv_grid_spacing = 1.0 / (gll_node_4 - gll_node_3);

constexpr size_t face_size = 25;  // 5x5 face

// Build volume coordinates for a 5^3 GLL grid on [-1,1]^3 with identity map
tnsr::I<DataVector, 3, Frame::Inertial> make_volume_coords() {
  const std::array<double, 5> nodes = {gll_node_0, gll_node_1, gll_node_2,
                                       gll_node_3, gll_node_4};
  tnsr::I<DataVector, 3, Frame::Inertial> coords(125_st);
  // SpECTRE ordering: fastest index = first dimension (xi)
  for (size_t kz = 0; kz < 5; ++kz) {
    for (size_t jy = 0; jy < 5; ++jy) {
      for (size_t ix = 0; ix < 5; ++ix) {
        const size_t idx = ix + 5 * (jy + 5 * kz);
        get<0>(coords)[idx] = nodes[ix];
        get<1>(coords)[idx] = nodes[jy];
        get<2>(coords)[idx] = nodes[kz];
      }
    }
  }
  return coords;
}

Mesh<3> make_volume_mesh() {
  return Mesh<3>{5, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}

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
  const auto volume_mesh = make_volume_mesh();
  const auto volume_coords = make_volume_coords();
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
  // Boundary modes (unused by LF, but must be passed)
  const auto u_tensor_minus =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
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
  const auto boundary_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_z =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
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
  auto pkg_u_tensor_minus =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto pkg_boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto pkg_boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_boundary_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_boundary_shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto pkg_boundary_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_boundary_z =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto pkg_normal_covector =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto pkg_inverse_grid_spacing =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);

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
      make_not_null(&pkg_u_tensor_minus),
      make_not_null(&pkg_boundary_conformal_metric),
      make_not_null(&pkg_boundary_conformal_factor),
      make_not_null(&pkg_boundary_lapse), make_not_null(&pkg_boundary_shift),
      make_not_null(&pkg_boundary_theta), make_not_null(&pkg_boundary_z),
      make_not_null(&pkg_normal_covector),
      make_not_null(&pkg_inverse_grid_spacing), conformal_metric,
      conformal_factor, a_tilde, trace_extrinsic_curvature, theta, gamma_hat,
      lapse, shift, auxiliary_shift_b, field_a, field_b, field_d, field_p,
      u_tensor_minus,
      boundary_conformal_metric, boundary_conformal_factor, boundary_lapse,
      boundary_shift, boundary_theta, boundary_z,
      normal_covector, mesh_velocity, normal_dot_mesh_velocity,
      direction, volume_mesh, volume_coords);

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

  // Inverse grid spacing matches analytic value
  const auto expected_inv_grid_spacing = make_with_value<Scalar<DataVector>>(
      DataVector(face_size), analytic_inv_grid_spacing);
  CHECK_ITERABLE_APPROX(pkg_inverse_grid_spacing, expected_inv_grid_spacing);
}

void test_dg_auxiliary_package_data() {
  const auto volume_mesh = make_volume_mesh();
  const auto volume_coords = make_volume_coords();
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
  // Boundary modes (unused by LF aux)
  const auto u_tensor_minus =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
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
  const auto boundary_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_z =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto normal_covector = make_unit_normal(face_size);

  // Output packaged data (10 fields for auxiliary)
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
  auto pkg_inverse_grid_spacing =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_field_a = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_field_b = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_field_d = make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_field_p = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);

  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;
  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
      std::nullopt;

  const double result = correction.dg_auxiliary_package_data(
      make_not_null(&pkg_conformal_metric),
      make_not_null(&pkg_conformal_factor), make_not_null(&pkg_lapse),
      make_not_null(&pkg_shift), make_not_null(&pkg_normal_covector),
      make_not_null(&pkg_inverse_grid_spacing), make_not_null(&pkg_field_a),
      make_not_null(&pkg_field_b), make_not_null(&pkg_field_d),
      make_not_null(&pkg_field_p), conformal_metric, conformal_factor, a_tilde,
      trace_extrinsic_curvature, theta, gamma_hat, lapse, shift,
      auxiliary_shift_b, field_a, field_b, field_d, field_p, u_tensor_minus,
      boundary_conformal_metric, boundary_conformal_factor, boundary_lapse,
      boundary_shift, boundary_theta, boundary_z,
      normal_covector, mesh_velocity, normal_dot_mesh_velocity,
      direction, volume_mesh, volume_coords);

  CHECK(result == 0.0);

  // Only conformal_metric, conformal_factor, lapse, shift, normal, inv grid
  // spacing are packaged
  CHECK_ITERABLE_APPROX(pkg_conformal_metric, conformal_metric);
  CHECK_ITERABLE_APPROX(pkg_conformal_factor, conformal_factor);
  CHECK_ITERABLE_APPROX(pkg_lapse, lapse);
  CHECK_ITERABLE_APPROX(pkg_shift, shift);
  CHECK_ITERABLE_APPROX(pkg_normal_covector, normal_covector);

  const auto expected_inv_grid_spacing = make_with_value<Scalar<DataVector>>(
      DataVector(face_size), analytic_inv_grid_spacing);
  CHECK_ITERABLE_APPROX(pkg_inverse_grid_spacing, expected_inv_grid_spacing);
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
  // Boundary modes and SO fields (zero, unused by LF)
  const auto u_tensor_minus_int =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_conformal_metric_int =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_conformal_factor_int =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_lapse_int =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_shift_int =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_theta_int =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_z_int =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto normal_covector_int = make_unit_normal(face_size);
  const auto inverse_grid_spacing_int = make_with_value<Scalar<DataVector>>(
      DataVector(face_size), analytic_inv_grid_spacing);

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
  const auto u_tensor_minus_ext =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_conformal_metric_ext =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_conformal_factor_ext =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_lapse_ext =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_shift_ext =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_theta_ext =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_z_ext =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  // Exterior normal points opposite
  auto normal_covector_ext =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  get<0>(normal_covector_ext) = -1.0;
  // Use a different inverse grid spacing for exterior to test max()
  const double ext_inv_grid_spacing = analytic_inv_grid_spacing * 1.1;
  const auto inverse_grid_spacing_ext = make_with_value<Scalar<DataVector>>(
      DataVector(face_size), ext_inv_grid_spacing);

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
  auto corr_u_tensor_minus =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
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
  auto corr_boundary_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_boundary_z =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);

  correction.dg_boundary_terms(
      make_not_null(&corr_conformal_metric),
      make_not_null(&corr_conformal_factor), make_not_null(&corr_a_tilde),
      make_not_null(&corr_trace_extrinsic_curvature),
      make_not_null(&corr_theta), make_not_null(&corr_gamma_hat),
      make_not_null(&corr_lapse), make_not_null(&corr_shift),
      make_not_null(&corr_auxiliary_shift_b), make_not_null(&corr_field_a),
      make_not_null(&corr_field_b), make_not_null(&corr_field_d),
      make_not_null(&corr_field_p), make_not_null(&corr_u_tensor_minus),
      make_not_null(&corr_boundary_conformal_metric),
      make_not_null(&corr_boundary_conformal_factor),
      make_not_null(&corr_boundary_lapse), make_not_null(&corr_boundary_shift),
      make_not_null(&corr_boundary_theta), make_not_null(&corr_boundary_z),
      conformal_metric_int, conformal_factor_int, a_tilde_int,
      trace_extrinsic_curvature_int, theta_int, gamma_hat_int, lapse_int,
      shift_int, auxiliary_shift_b_int, field_a_int, field_b_int, field_d_int,
      field_p_int, u_tensor_minus_int, boundary_conformal_metric_int,
      boundary_conformal_factor_int, boundary_lapse_int, boundary_shift_int,
      boundary_theta_int, boundary_z_int,
      normal_covector_int, inverse_grid_spacing_int, conformal_metric_ext,
      conformal_factor_ext, a_tilde_ext, trace_extrinsic_curvature_ext,
      theta_ext, gamma_hat_ext, lapse_ext, shift_ext, auxiliary_shift_b_ext,
      field_a_ext, field_b_ext, field_d_ext, field_p_ext, u_tensor_minus_ext,
      boundary_conformal_metric_ext, boundary_conformal_factor_ext,
      boundary_lapse_ext, boundary_shift_ext, boundary_theta_ext,
      boundary_z_ext, normal_covector_ext,
      inverse_grid_spacing_ext, dg::Formulation::StrongInertial);

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
  // tau1_eff = tau1 * max(inv_grid_spacing_int, inv_grid_spacing_ext)
  const double tau1_eff = tau1 * ext_inv_grid_spacing;  // ext > int

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
  get(expected_corr_K) = -0.5 * (k_flux_ext + k_flux_int) -
                         0.5 * tau1_eff *
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
  get(expected_corr_theta) = -0.5 * (theta_flux_ext + theta_flux_int) -
                             0.5 * tau1_eff * (get(theta_ext) - get(theta_int));
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
          -0.5 * (at_flux_ext.get(i, j) + at_flux_int.get(i, j)) -
          0.5 * tau1_eff * (a_tilde_ext.get(i, j) - a_tilde_int.get(i, j));
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
        -0.5 * (gh_flux_ext.get(I) + gh_flux_int.get(I)) -
        0.5 * tau1_eff * (gamma_hat_ext.get(I) - gamma_hat_int.get(I));
  }
  CHECK_ITERABLE_APPROX(corr_gamma_hat, expected_corr_gamma_hat);

  // --- auxiliary_shift_b boundary correction ---
  // b_flux = gamma_hat_flux (since shifting_shift = false)
  auto expected_corr_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t I = 0; I < 3; ++I) {
    expected_corr_b.get(I) =
        -0.5 * (gh_flux_ext.get(I) + gh_flux_int.get(I)) -
        0.5 * tau1_eff *
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
        -0.5 * (fa_flux_ext.get(k) + fa_flux_int.get(k)) -
        0.5 * tau1_eff * (field_a_ext.get(k) - field_a_int.get(k));
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
          -0.5 * (fb_flux_ext.get(k, I) + fb_flux_int.get(k, I)) -
          0.5 * tau1_eff * (field_b_ext.get(k, I) - field_b_int.get(k, I));
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
            -0.5 * (fd_flux_ext.get(k, i, j) + fd_flux_int.get(k, i, j)) -
            0.5 * tau1_eff *
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
        -0.5 * (fp_flux_ext.get(k) + fp_flux_int.get(k)) -
        0.5 * tau1_eff * (field_p_ext.get(k) - field_p_int.get(k));
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
  const auto inverse_grid_spacing_int = make_with_value<Scalar<DataVector>>(
      DataVector(face_size), analytic_inv_grid_spacing);
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
  const double ext_inv_grid_spacing = analytic_inv_grid_spacing * 1.1;
  const auto inverse_grid_spacing_ext = make_with_value<Scalar<DataVector>>(
      DataVector(face_size), ext_inv_grid_spacing);
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

  // Output corrections (all 21 evolved+aux+boundary fields)
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
  auto corr_u_tensor_minus =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
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
  auto corr_boundary_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto corr_boundary_z =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);

  correction.dg_auxiliary_boundary_terms(
      make_not_null(&corr_conformal_metric),
      make_not_null(&corr_conformal_factor), make_not_null(&corr_a_tilde),
      make_not_null(&corr_trace_extrinsic_curvature),
      make_not_null(&corr_theta), make_not_null(&corr_gamma_hat),
      make_not_null(&corr_lapse), make_not_null(&corr_shift),
      make_not_null(&corr_auxiliary_shift_b), make_not_null(&corr_field_a),
      make_not_null(&corr_field_b), make_not_null(&corr_field_d),
      make_not_null(&corr_field_p), make_not_null(&corr_u_tensor_minus),
      make_not_null(&corr_boundary_conformal_metric),
      make_not_null(&corr_boundary_conformal_factor),
      make_not_null(&corr_boundary_lapse), make_not_null(&corr_boundary_shift),
      make_not_null(&corr_boundary_theta), make_not_null(&corr_boundary_z),
      conformal_metric_int, conformal_factor_int, lapse_int, shift_int,
      normal_covector_int, inverse_grid_spacing_int, field_a_int, field_b_int,
      field_d_int, field_p_int, conformal_metric_ext, conformal_factor_ext,
      lapse_ext, shift_ext, normal_covector_ext, inverse_grid_spacing_ext,
      field_a_ext, field_b_ext, field_d_ext, field_p_ext,
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
  // tau2_eff is just tau2 (no grid spacing multiplication in production code)
  const double tau2_eff = tau2;

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
               log_lapse_ext * normal_covector_ext.get(i)) -
        0.5 * tau2_eff * (field_a_ext.get(i) - field_a_int.get(i));
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
                 shift_ext.get(J) * normal_covector_ext.get(i)) -
          0.5 * tau2_eff * (field_b_ext.get(i, J) - field_b_int.get(i, J));
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
                       normal_covector_ext.get(i)) -
            0.5 * tau2_eff *
                (field_d_ext.get(i, j, k) - field_d_int.get(i, j, k));
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
               log_cf_ext * normal_covector_ext.get(i)) -
        0.5 * tau2_eff * (field_p_ext.get(i) - field_p_int.get(i));
  }
  CHECK_ITERABLE_APPROX(corr_field_p, expected_corr_field_p);
}

void test_serialization() {
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> correction(1.5, 2.3);
  const auto deserialized = serialize_and_deserialize(correction);
  // Verify it works by running a simple package_data call
  // (if pup restored tau1_ and tau2_ correctly, the computation will match)
  const auto volume_mesh = make_volume_mesh();
  const auto volume_coords = make_volume_coords();
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
  auto pkg_u_tm = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_bcm = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_bcf =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_blapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_bshift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_btheta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_bz = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_n = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_inv_gs =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);

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
  // Boundary modes and SO fields (zero)
  const auto u_tm = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto bcm = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto bcf =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto blapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto bshift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto btheta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto bz = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
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
      make_not_null(&pkg_u_tm), make_not_null(&pkg_bcm),
      make_not_null(&pkg_bcf), make_not_null(&pkg_blapse),
      make_not_null(&pkg_bshift), make_not_null(&pkg_btheta),
      make_not_null(&pkg_bz), make_not_null(&pkg_n),
      make_not_null(&pkg_inv_gs), cm, cf, at, K, theta, gh, lapse, shift, b, fa,
      fb, fd, fp, u_tm, bcm, bcf, blapse, bshift, btheta, bz, normal,
      mesh_velocity, normal_dot_mesh_velocity, direction, volume_mesh,
      volume_coords);
  const auto inv_gs_orig = pkg_inv_gs;

  // Reset and call on deserialized
  get(pkg_inv_gs) = 0.0;
  const double result_deser = deserialized.dg_package_data(
      make_not_null(&pkg_cm), make_not_null(&pkg_cf), make_not_null(&pkg_at),
      make_not_null(&pkg_K), make_not_null(&pkg_theta), make_not_null(&pkg_gh),
      make_not_null(&pkg_lapse), make_not_null(&pkg_shift),
      make_not_null(&pkg_b), make_not_null(&pkg_fa), make_not_null(&pkg_fb),
      make_not_null(&pkg_fd), make_not_null(&pkg_fp),
      make_not_null(&pkg_u_tm), make_not_null(&pkg_bcm),
      make_not_null(&pkg_bcf), make_not_null(&pkg_blapse),
      make_not_null(&pkg_bshift), make_not_null(&pkg_btheta),
      make_not_null(&pkg_bz), make_not_null(&pkg_n),
      make_not_null(&pkg_inv_gs), cm, cf, at, K, theta, gh, lapse, shift, b, fa,
      fb, fd, fp, u_tm, bcm, bcf, blapse, bshift, btheta, bz, normal,
      mesh_velocity, normal_dot_mesh_velocity, direction, volume_mesh,
      volume_coords);

  CHECK(result_orig == result_deser);
  CHECK_ITERABLE_APPROX(pkg_inv_gs, inv_gs_orig);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.BoundaryCorrections.LaxFriedrichs",
    "[Unit][Evolution]") {
  SECTION("dg_package_data") { test_dg_package_data(); }
  SECTION("dg_auxiliary_package_data") { test_dg_auxiliary_package_data(); }
  SECTION("dg_boundary_terms") { test_dg_boundary_terms(); }
  SECTION("dg_auxiliary_boundary_terms") { test_dg_auxiliary_boundary_terms(); }
  SECTION("serialization") { test_serialization(); }
}
