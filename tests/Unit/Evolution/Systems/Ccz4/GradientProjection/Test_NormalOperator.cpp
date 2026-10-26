// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Characteristics.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/NormalOperator.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;
using helpers::to_data_vector;
constexpr const char* module = "GradientProjectionReference";
constexpr size_t n43 = gp::number_of_target_components;
using Square = std::vector<double>;  // row-major 43x43

gp::VelocityVector unit_w(const size_t index) {
  gp::VelocityVector w{};
  if (index < gp::number_of_velocities) {
    gsl::at(w, index) = 1.0;
  }
  return w;
}

gp::GradientVector unit_q(const size_t index) {
  gp::GradientVector q{};
  if (index >= gp::number_of_velocities) {
    const size_t k = index - gp::number_of_velocities;
    gsl::at(gsl::at(q, k / 10), k % 10) = 1.0;
  }
  return q;
}

std::array<double, n43> flatten_target(const gp::VelocityVector& w,
                                       const gp::GradientVector& q) {
  std::array<double, n43> result{};
  for (size_t a = 0; a < gp::number_of_velocities; ++a) {
    gsl::at(result, a) = gsl::at(w, a);
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < 10; ++a) {
      gsl::at(result, 13 + 10 * i + a) = gsl::at(gsl::at(q, i), a);
    }
  }
  return result;
}

// Assemble a C++ operator column by column.
template <typename Operator>
Square assemble(const Operator& op) {
  Square result(n43 * n43, 0.0);
  for (size_t column = 0; column < n43; ++column) {
    gp::VelocityVector w{};
    gp::GradientVector q{};
    op(make_not_null(&w), make_not_null(&q), unit_w(column), unit_q(column));
    const auto image = flatten_target(w, q);
    for (size_t row = 0; row < n43; ++row) {
      result[row * n43 + column] = gsl::at(image, row);
    }
  }
  return result;
}

Square to_square(const DataVector& flat) {
  return Square(flat.begin(), flat.end());
}

double max_abs(const Square& a) {
  double result = 0.0;
  for (const double x : a) {
    result = std::max(result, std::abs(x));
  }
  return result;
}

double max_abs_difference(const Square& a, const Square& b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result = std::max(result, std::abs(a[i] - b[i]));
  }
  return result;
}

std::array<double, n43> multiply(const Square& a,
                                 const std::array<double, n43>& y) {
  std::array<double, n43> result{};
  for (size_t row = 0; row < n43; ++row) {
    for (size_t column = 0; column < n43; ++column) {
      gsl::at(result, row) += a[row * n43 + column] * gsl::at(y, column);
    }
  }
  return result;
}

// The infinity-norm condition number of the analytic characteristic
// transform, which bounds the rounding error of everything computed through
// it. (The eigenvector matrix returned by a numerical eigensolver is not a
// usable measure: within the repeated speeds its basis is arbitrary and can be
// nearly singular.)
double transform_condition(const gp::NormalCharacteristics& characteristics) {
  constexpr size_t size = gp::number_of_normal_components;
  std::array<double, size> forward_rows{};
  std::array<double, size> inverse_rows{};
  for (size_t column = 0; column < size; ++column) {
    gp::NormalSubspaceVector unit{};
    gsl::at(unit, column) = 1.0;
    const auto forward = characteristics.characteristic_fields(unit);
    const auto inverse =
        characteristics.normal_fields_from_characteristic_fields(unit);
    for (size_t row = 0; row < size; ++row) {
      gsl::at(forward_rows, row) += std::abs(gsl::at(forward, row));
      gsl::at(inverse_rows, row) += std::abs(gsl::at(inverse, row));
    }
  }
  return *alg::max_element(forward_rows) * *alg::max_element(inverse_rows);
}

DataVector to_data_vector(const std::array<double, 3>& v) {
  return DataVector{v[0], v[1], v[2]};
}

struct Case {
  helpers::RandomState state;
  std::array<double, 3> normal;
};

// Admissible states covering Minkowski, non-unit lapse and conformal factor,
// shifts, anisotropic metrics, coordinate-axis, oblique and non-unit normals.
std::vector<Case> cases(const gsl::not_null<std::mt19937*> gen,
                        const double f) {
  std::vector<Case> result{};
  helpers::RandomState minkowski{};
  minkowski.u[gp::PotentialIndex::lapse] = 1.0;
  minkowski.u[gp::PotentialIndex::conformal_factor] = 1.0;
  for (size_t axis = 0; axis < 3; ++axis) {
    std::array<double, 3> normal{};
    gsl::at(normal, axis) = 1.0;
    result.push_back({minkowski, normal});
  }
  result.push_back({minkowski, helpers::random_unit_vector(gen)});
  for (size_t sample = 0; sample < 24; ++sample) {
    const double h_scale = std::array{0.0, 0.1, 0.4}[sample % 3];
    const double shift_scale = std::array{0.0, 0.15, 0.4}[(sample / 3) % 3];
    helpers::RandomState state{};
    std::array<double, 3> normal{};
    // Random states away from the gauge-speed collisions, near which every
    // eigendecomposition is ill-conditioned (see test_near_collisions).
    do {
      state = helpers::random_state(gen, h_scale, shift_scale, 0.2);
      normal = helpers::random_unit_vector(gen);
      if (sample % 4 == 3) {
        // A coordinate-axis normal on a curved state.
        normal = {{0.0, 0.0, -1.0}};
      }
    } while (transform_condition(gp::NormalCharacteristics{
                 gp::LogMetricChart{
                     gp::stf_decode(&state.u[gp::PotentialIndex::log_metric])},
                 state.u, normal, f}) > 1.0e5);
    if (sample % 5 == 4) {
      // The map is linear in the covector; check a non-unit one.
      for (auto& component : normal) {
        component *= 2.5;
      }
    }
    result.push_back({state, normal});
  }
  return result;
}

// The C++ normal matrix against the independent column assembly of the
// archived model, for all 43 columns (all tangential gradient couplings).
void test_normal_matrix(const std::vector<Case>& all_cases, const double f) {
  for (const auto& [state, normal] : all_cases) {
    CAPTURE(normal);
    const gp::LogMetricChart chart{
        gp::stf_decode(&state.u[gp::PotentialIndex::log_metric])};
    const Square actual = assemble([&](const auto result_w, const auto result_q,
                                       const auto& w, const auto& q) {
      gp::apply_normal_matrix(result_w, result_q, chart, state.u, normal, w, q,
                              f);
    });
    const Square expected = to_square(
        pypp::call<DataVector>(module, "normal_matrix", to_data_vector(state.u),
                               to_data_vector(normal), f));
    CHECK(max_abs_difference(actual, expected) < 1e-12 * max_abs(expected));
  }
}

// The analytic absolute value against two numerical oracles: an eigenvector
// decomposition and the ordered three-block real Schur route.
void test_absolute_matrix(const std::vector<Case>& all_cases, const double f) {
  size_t eig_comparisons = 0;
  for (const auto& [state, normal] : all_cases) {
    CAPTURE(normal);
    CAPTURE(state.u);
    const gp::LogMetricChart chart{
        gp::stf_decode(&state.u[gp::PotentialIndex::log_metric])};
    const Square actual = assemble([&](const auto result_w, const auto result_q,
                                       const auto& w, const auto& q) {
      gp::apply_absolute_normal_matrix(result_w, result_q, chart, state.u,
                                       normal, w, q, f);
    });
    const gp::NormalCharacteristics characteristics{chart, state.u, normal, f};
    const double condition = transform_condition(characteristics);
    CAPTURE(condition);
    REQUIRE(condition < 1.0e6);
    // Near a sonic point the numerical oracles must separate the structural
    // zero speeds from a nearly vanishing one, and their accuracy degrades in
    // inverse proportion to that speed relative to the largest.
    double smallest_speed = std::numeric_limits<double>::infinity();
    double largest_speed = 0.0;
    for (size_t k = 0; k < gp::number_of_normal_components; ++k) {
      const double speed = std::abs(gsl::at(characteristics.speeds(), k));
      largest_speed = std::max(largest_speed, speed);
      if (not gp::NormalCharacteristics::is_kernel_field(k)) {
        smallest_speed = std::min(smallest_speed, speed);
      }
    }
    const double oracle_condition =
        std::max(condition, 10.0 * largest_speed / smallest_speed);
    CAPTURE(oracle_condition);
    // The ordered Schur route needs no eigenvectors.
    const Square schur_reference = to_square(pypp::call<DataVector>(
        module, "absolute_normal_matrix_schur", to_data_vector(state.u),
        to_data_vector(normal), f));
    CHECK(max_abs_difference(actual, schur_reference) <
          1.0e-13 * oracle_condition * max_abs(actual));
    // The eigenvector route is only usable when the numerical eigenvector
    // basis is.
    const double eig_condition =
        pypp::call<double>(module, "eigenvector_condition",
                           to_data_vector(state.u), to_data_vector(normal), f);
    CAPTURE(eig_condition);
    if (eig_condition < 1.0e8) {
      ++eig_comparisons;
      const Square eig_reference = to_square(pypp::call<DataVector>(
          module, "absolute_normal_matrix_eig", to_data_vector(state.u),
          to_data_vector(normal), f));
      CHECK(max_abs_difference(actual, eig_reference) <
            1.0e-13 * std::max(oracle_condition, eig_condition) *
                max_abs(actual));
    }

    // |A_{-n}| = |A_n| and |A_{cn}| = c |A_n| for c > 0.
    const std::array<double, 3> reversed{{-normal[0], -normal[1], -normal[2]}};
    const Square reversed_absolute =
        assemble([&](const auto result_w, const auto result_q, const auto& w,
                     const auto& q) {
          gp::apply_absolute_normal_matrix(result_w, result_q, chart, state.u,
                                           reversed, w, q, f);
        });
    CHECK(max_abs_difference(reversed_absolute, actual) <
          1e-12 * max_abs(actual));
  }
  CHECK(2 * eig_comparisons >= all_cases.size());
}

// Eigenvalues, eigenvector residuals and the characteristic transforms.
void test_characteristics(const std::vector<Case>& all_cases, const double f,
                          const gsl::not_null<std::mt19937*> gen) {
  std::normal_distribution<double> dist(0.0, 1.0);
  for (const auto& [state, normal] : all_cases) {
    CAPTURE(normal);
    const gp::LogMetricChart chart{
        gp::stf_decode(&state.u[gp::PotentialIndex::log_metric])};
    const gp::NormalCharacteristics characteristics{chart, state.u, normal, f};
    const auto& speeds = characteristics.speeds();
    double speed_scale = 0.0;
    for (const double speed : speeds) {
      speed_scale = std::max(speed_scale, std::abs(speed));
    }
    CHECK(gp::largest_characteristic_speed(chart, state.u, normal, f) ==
          approx(speed_scale).epsilon(1e-12));
    const double condition = transform_condition(characteristics);
    CAPTURE(condition);
    const double speed_tolerance = std::max(1e-11, 1e-13 * condition);

    // All 43 eigenvalues of A_n: the 23 of M plus the 20 structural zeros.
    std::vector<double> all_speeds(speeds.begin(), speeds.end());
    all_speeds.resize(n43, 0.0);
    std::sort(all_speeds.begin(), all_speeds.end());
    const DataVector numerical =
        pypp::call<DataVector>(module, "sorted_speeds", to_data_vector(state.u),
                               to_data_vector(normal), f);
    CHECK(pypp::call<double>(module, "max_imaginary_speed",
                             to_data_vector(state.u), to_data_vector(normal),
                             f) < speed_tolerance * speed_scale);
    for (size_t k = 0; k < n43; ++k) {
      CHECK(all_speeds[k] ==
            approx(numerical[k]).scale(speed_scale).epsilon(speed_tolerance));
    }

    // Right eigenvectors of A_n: J R e_k, residual against the independently
    // assembled matrix.
    const Square a = to_square(
        pypp::call<DataVector>(module, "normal_matrix", to_data_vector(state.u),
                               to_data_vector(normal), f));
    for (size_t k = 0; k < gp::number_of_normal_components; ++k) {
      gp::NormalSubspaceVector unit{};
      gsl::at(unit, k) = 1.0;
      const gp::NormalSubspaceVector z =
          characteristics.normal_fields_from_characteristic_fields(unit);
      gp::VelocityVector w{};
      gp::GradientVector q{};
      gp::embed_normal_subspace_vector(make_not_null(&w), make_not_null(&q),
                                       normal, z);
      const auto y = flatten_target(w, q);
      const auto image = multiply(a, y);
      double residual = 0.0;
      double norm = 0.0;
      for (size_t row = 0; row < n43; ++row) {
        residual =
            std::max(residual, std::abs(gsl::at(image, row) -
                                        gsl::at(speeds, k) * gsl::at(y, row)));
        norm = std::max(norm, std::abs(gsl::at(y, row)));
      }
      CAPTURE(k);
      CHECK(norm > 0.0);
      CHECK(residual < 1e-12 * speed_scale * norm);
      if (gp::NormalCharacteristics::is_kernel_field(k)) {
        CHECK(gsl::at(speeds, k) == 0.0);
      } else {
        CHECK(gsl::at(speeds, k) != 0.0);
      }
    }

    // Characteristic transform and its inverse.
    gp::NormalSubspaceVector z{};
    for (auto& component : z) {
      component = dist(*gen);
    }
    const auto fields = characteristics.characteristic_fields(z);
    const auto round_trip =
        characteristics.normal_fields_from_characteristic_fields(fields);
    gp::NormalSubspaceVector c{};
    for (auto& component : c) {
      component = dist(*gen);
    }
    const auto c_round_trip = characteristics.characteristic_fields(
        characteristics.normal_fields_from_characteristic_fields(c));
    double z_scale = 0.0;
    double c_scale = 0.0;
    for (size_t k = 0; k < gp::number_of_normal_components; ++k) {
      z_scale = std::max(z_scale, std::abs(gsl::at(z, k)));
      c_scale = std::max(c_scale, std::abs(gsl::at(c, k)));
    }
    for (size_t k = 0; k < gp::number_of_normal_components; ++k) {
      CHECK(std::abs(gsl::at(round_trip, k) - gsl::at(z, k)) <
            1e-14 * condition * z_scale);
      CHECK(std::abs(gsl::at(c_round_trip, k) - gsl::at(c, k)) <
            1e-14 * condition * c_scale);
    }
  }
}

// The speeds of the existing physical-unit-normal formulas, converted to the
// Euclidean-unit covector: speed(n) = -s * speed_old(n / s) with
// s = phi sqrt(g^{ij} n_i n_j) (the old formulas use the propagation
// convention u_t + lambda u_n = 0, the target operator Y_t = A^i d_i Y).
void test_normalization_conversion(const std::vector<Case>& all_cases,
                                   const double f) {
  for (const auto& [state, normal] : all_cases) {
    const gp::LogMetricChart chart{
        gp::stf_decode(&state.u[gp::PotentialIndex::log_metric])};
    const gp::NormalCharacteristics characteristics{chart, state.u, normal, f};
    const double phi = state.u[gp::PotentialIndex::conformal_factor];
    const gp::Matrix3 inverse_metric = chart.inverse_metric();
    double norm_squared = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        norm_squared +=
            inverse_metric(i, j) * gsl::at(normal, i) * gsl::at(normal, j);
      }
    }
    const double s = phi * std::sqrt(norm_squared);
    tnsr::i<DataVector, 3> physical_normal{1_st};
    tnsr::I<DataVector, 3> shift{1_st};
    for (size_t i = 0; i < 3; ++i) {
      physical_normal.get(i) = gsl::at(normal, i) / s;
      shift.get(i) = gsl::at(state.u, gp::PotentialIndex::shift + i);
    }
    const auto old_speeds = Ccz4::fd::characteristic_speeds(
        Scalar<DataVector>{DataVector{state.u[gp::PotentialIndex::lapse]}},
        shift, Scalar<DataVector>{DataVector{phi}}, f, physical_normal);
    std::vector<double> converted_old{};
    for (const auto& speed : old_speeds) {
      converted_old.push_back(-s * speed[0]);
    }
    for (const double speed : characteristics.speeds()) {
      const bool found =
          alg::any_of(converted_old, [&speed, &s](const double other) {
            return std::abs(other - speed) < 1e-12 * (1.0 + s);
          });
      CAPTURE(speed);
      CHECK(found);
    }
    for (const double speed : converted_old) {
      const bool found = alg::any_of(
          characteristics.speeds(), [&speed, &s](const double other) {
            return std::abs(other - speed) < 1e-12 * (1.0 + s);
          });
      CAPTURE(speed);
      CHECK(found);
    }

    // |A(n)| = s |A(n / s)| for the absolute matrix itself.
    const std::array<double, 3> scaled{
        {normal[0] / s, normal[1] / s, normal[2] / s}};
    for (size_t column = 0; column < n43; column += 7) {
      gp::VelocityVector w_unit{};
      gp::GradientVector q_unit{};
      gp::VelocityVector w_scaled{};
      gp::GradientVector q_scaled{};
      gp::apply_absolute_normal_matrix(
          make_not_null(&w_unit), make_not_null(&q_unit), chart, state.u,
          normal, unit_w(column), unit_q(column), f);
      gp::apply_absolute_normal_matrix(
          make_not_null(&w_scaled), make_not_null(&q_scaled), chart, state.u,
          scaled, unit_w(column), unit_q(column), f);
      const auto unit_image = flatten_target(w_unit, q_unit);
      const auto scaled_image = flatten_target(w_scaled, q_scaled);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(gsl::at(unit_image, row) ==
              approx(s * gsl::at(scaled_image, row)).epsilon(1e-12));
      }
    }
  }
}

// Approaching a collision of two speeds of the same sign, |A_n| stays smooth
// but the characteristic transform becomes singular: the analytic |A_n| loses
// accuracy like (machine epsilon) / delta, delta the relative speed gap.
void test_near_collisions(const double f) {
  const std::array<double, 3> oblique{{0.3, -0.5, 0.8}};
  for (const bool flat : {true, false}) {
    CAPTURE(flat);
    gp::PotentialVector u{};
    const double phi = flat ? 1.0 : 0.9;
    if (not flat) {
      const std::array<double, 5> h{{0.1, -0.2, 0.05, 0.15, -0.1}};
      for (size_t a = 0; a < 5; ++a) {
        gsl::at(u, gp::PotentialIndex::log_metric + a) = gsl::at(h, a);
      }
    }
    u[gp::PotentialIndex::conformal_factor] = phi;
    const std::array<double, 3> normal =
        flat ? std::array<double, 3>{{1.0, 0.0, 0.0}} : oblique;
    const gp::LogMetricChart chart{
        gp::stf_decode(&u[gp::PotentialIndex::log_metric])};
    for (const double delta : {1.0e-4, 1.0e-6, 1.0e-8}) {
      CAPTURE(delta);
      // alpha phi = sqrt(4 f/3): longitudinal shift and light speeds;
      // phi sqrt(2 alpha) = sqrt(4 f/3): longitudinal shift and 1+log speeds.
      const double longitudinal_speed = std::sqrt(4.0 * f / 3.0);
      for (const double lapse :
           {longitudinal_speed * (1.0 + delta) / phi,
            0.5 * square(longitudinal_speed * (1.0 + delta) / phi)}) {
        CAPTURE(lapse);
        u[gp::PotentialIndex::lapse] = lapse;
        const Square actual =
            assemble([&](const auto result_w, const auto result_q,
                         const auto& w, const auto& q) {
              gp::apply_absolute_normal_matrix(result_w, result_q, chart, u,
                                               normal, w, q, f);
            });
        const Square schur_reference = to_square(pypp::call<DataVector>(
            module, "absolute_normal_matrix_schur", to_data_vector(u),
            to_data_vector(normal), f));
        CHECK(max_abs_difference(actual, schur_reference) <
              std::max(1.0e-12, 1.0e-14 / delta) * max_abs(actual));
      }
    }
  }
}

// Sonic points and gauge-speed collisions are reported, never regularized.
void test_degeneracies(const double f) {
  const auto state_with = [](const double lapse, const double phi,
                             const double shift_x) {
    gp::PotentialVector u{};
    u[gp::PotentialIndex::lapse] = lapse;
    u[gp::PotentialIndex::conformal_factor] = phi;
    u[gp::PotentialIndex::shift] = shift_x;
    return u;
  };
  const gp::LogMetricChart flat{gp::Matrix3{}};
  const std::array<double, 3> normal{{1.0, 0.0, 0.0}};
  // beta.n = alpha phi m
  CHECK_THROWS_WITH(
      (gp::NormalCharacteristics{flat, state_with(0.5, 0.8, 0.4), normal, f}),
      Catch::Matchers::ContainsSubstring("light-cone characteristic speed"));
  // beta.n = phi m sqrt(2 alpha)
  CHECK_THROWS_WITH(
      (gp::NormalCharacteristics{flat, state_with(0.5, 0.8, 0.8), normal, f}),
      Catch::Matchers::ContainsSubstring("A 1+log characteristic speed"));
  // alpha phi = sqrt(4 f/3): longitudinal shift speed equals the light speed.
  CHECK_THROWS_WITH(
      (gp::NormalCharacteristics{
          flat, state_with(std::sqrt(4.0 * f / 3.0), 1.0, 0.0), normal, f}),
      Catch::Matchers::ContainsSubstring("coincides with a light-cone speed"));
  // phi sqrt(2 alpha) = sqrt(4 f/3): longitudinal shift speed equals 1+log.
  CHECK_THROWS_WITH(
      (gp::NormalCharacteristics{flat, state_with(2.0 * f / 3.0, 1.0, 0.0),
                                 normal, f}),
      Catch::Matchers::ContainsSubstring("coincides with a 1+log speed"));
  // Away from exact degeneracy the decomposition is used.
  CHECK_NOTHROW((gp::NormalCharacteristics{
      flat, state_with(1.005 * 2.0 * f / 3.0, 1.0, 0.0), normal, f}));
}

// In unit Minkowski space with zero shift the longitudinal shift speed
// sqrt(4 f/3) collides with the light speed at f = 3/4 and with the 1+log
// speed at f = 3/2, where the continuum symbol is defective; a non-positive f
// is rejected.
void test_gamma_driver_parameter() {
  const gp::LogMetricChart flat{gp::Matrix3{}};
  gp::PotentialVector minkowski{};
  minkowski[gp::PotentialIndex::lapse] = 1.0;
  minkowski[gp::PotentialIndex::conformal_factor] = 1.0;
  const std::array<double, 3> normal{{0.0, 1.0, 0.0}};
  CHECK_THROWS_WITH(
      (gp::NormalCharacteristics{flat, minkowski, normal, 0.75}),
      Catch::Matchers::ContainsSubstring("coincides with a light-cone speed"));
  CHECK_THROWS_WITH(
      (gp::NormalCharacteristics{flat, minkowski, normal, 1.5}),
      Catch::Matchers::ContainsSubstring("coincides with a 1+log speed"));
  for (const double f : {0.0, -0.5}) {
    CAPTURE(f);
    CHECK_THROWS_WITH((gp::NormalCharacteristics{flat, minkowski, normal, f}),
                      Catch::Matchers::ContainsSubstring(
                          "The Gamma-driver parameter f must be positive"));
    CHECK_THROWS_WITH(
        gp::largest_characteristic_speed(flat, minkowski, normal, f),
        Catch::Matchers::ContainsSubstring(
            "The Gamma-driver parameter f must be positive"));
  }
}

// The physical-phi operator against the frozen log(phi) operator at the same
// state: A_n = S A_n^log S^-1 and |A_n| = S |A_n^log| S^-1 with S multiplying
// the three gradients of phi by phi, for random anisotropic metrics, shifts
// and conformal factors including small ones.
void test_phi_similarity(const gsl::not_null<std::mt19937*> gen,
                         const double f) {
  for (size_t sample = 0; sample < 12; ++sample) {
    const std::array<double, 2> phi_range =
        sample < 4 ? std::array<double, 2>{{0.05, 0.2}}
                   : std::array<double, 2>{{0.5, 2.0}};
    helpers::RandomState state{};
    std::array<double, 3> normal{};
    do {
      state =
          helpers::random_state(gen, 0.4, 0.2, 0.2, {{0.4, 1.5}}, phi_range);
      normal = helpers::random_unit_vector(gen);
    } while (transform_condition(gp::NormalCharacteristics{
                 gp::LogMetricChart{
                     gp::stf_decode(&state.u[gp::PotentialIndex::log_metric])},
                 state.u, normal, f}) > 1.0e5);
    const double phi = state.u[gp::PotentialIndex::conformal_factor];
    CAPTURE(phi);
    gp::PotentialVector u_log = state.u;
    u_log[gp::PotentialIndex::conformal_factor] = std::log(phi);
    const gp::LogMetricChart chart{
        gp::stf_decode(&state.u[gp::PotentialIndex::log_metric])};
    const auto similar = [phi](const Square& log_matrix) {
      Square result = log_matrix;
      for (size_t row = 0; row < n43; ++row) {
        for (size_t column = 0; column < n43; ++column) {
          const auto scale = [phi](const size_t index) {
            return index >= gp::number_of_velocities and
                           (index - gp::number_of_velocities) % 10 ==
                               gp::PotentialIndex::conformal_factor
                       ? phi
                       : 1.0;
          };
          result[row * n43 + column] *= scale(row) / scale(column);
        }
      }
      return result;
    };
    const Square actual = assemble([&](const auto result_w, const auto result_q,
                                       const auto& w, const auto& q) {
      gp::apply_normal_matrix(result_w, result_q, chart, state.u, normal, w, q,
                              f);
    });
    const Square expected = similar(to_square(pypp::call<DataVector>(
        module, "normal_matrix_log", to_data_vector(u_log),
        to_data_vector(normal), f)));
    CHECK(max_abs_difference(actual, expected) < 1e-12 * max_abs(expected));
    const Square actual_absolute =
        assemble([&](const auto result_w, const auto result_q, const auto& w,
                     const auto& q) {
          gp::apply_absolute_normal_matrix(result_w, result_q, chart, state.u,
                                           normal, w, q, f);
        });
    const gp::NormalCharacteristics characteristics{chart, state.u, normal, f};
    const Square expected_absolute = similar(to_square(pypp::call<DataVector>(
        module, "absolute_normal_matrix_log_schur", to_data_vector(u_log),
        to_data_vector(normal), f)));
    CHECK(max_abs_difference(actual_absolute, expected_absolute) <
          1.0e-12 * transform_condition(characteristics) *
              max_abs(expected_absolute));
  }
}

Square square_multiply(const Square& a, const Square& b) {
  Square result(n43 * n43, 0.0);
  for (size_t row = 0; row < n43; ++row) {
    for (size_t inner = 0; inner < n43; ++inner) {
      const double left = a[row * n43 + inner];
      if (left == 0.0) {
        continue;
      }
      for (size_t column = 0; column < n43; ++column) {
        result[row * n43 + column] += left * b[inner * n43 + column];
      }
    }
  }
  return result;
}

void unflatten_target(const gsl::not_null<gp::VelocityVector*> w,
                      const gsl::not_null<gp::GradientVector*> q,
                      const std::array<double, n43>& flat) {
  for (size_t a = 0; a < gp::number_of_velocities; ++a) {
    gsl::at(*w, a) = gsl::at(flat, a);
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      gsl::at(gsl::at(*q, i), a) = gsl::at(flat, 13 + 10 * i + a);
    }
  }
}

gp::NormalSubspaceVector unit_normal_subspace(const size_t index) {
  gp::NormalSubspaceVector unit{};
  gsl::at(unit, index) = 1.0;
  return unit;
}

// The GammaHat-b projector P_g of the full operator: the kernel entries of
// the characteristic transform depend only on the velocity components, so the
// projection of a full jump needs only its velocity jump.
std::array<double, n43> kernel_projection(
    const gp::NormalCharacteristics& characteristics,
    const std::array<double, 3>& normal, const gp::VelocityVector& jump_w) {
  gp::NormalSubspaceVector z{};
  for (size_t a = 0; a < gp::number_of_velocities; ++a) {
    gsl::at(z, a) = gsl::at(jump_w, a);
  }
  auto fields = characteristics.characteristic_fields(z);
  for (size_t k = 0; k < gp::number_of_normal_components; ++k) {
    if (not gp::NormalCharacteristics::is_kernel_field(k)) {
      gsl::at(fields, k) = 0.0;
    }
  }
  gp::VelocityVector w{};
  gp::GradientVector q{};
  gp::embed_normal_subspace_vector(
      make_not_null(&w), make_not_null(&q), normal,
      characteristics.normal_fields_from_characteristic_fields(fields));
  return flatten_target(w, q);
}

// A random gradient jump tangential to the Euclidean normal.
gp::GradientVector random_tangential(const gsl::not_null<std::mt19937*> gen,
                                     const std::array<double, 3>& normal) {
  std::normal_distribution<double> dist(0.0, 1.0);
  const double normal_squared =
      normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
  gp::GradientVector tangential{};
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    const std::array<double, 3> vector{{dist(*gen), dist(*gen), dist(*gen)}};
    const double projection = (vector[0] * normal[0] + vector[1] * normal[1] +
                               vector[2] * normal[2]) /
                              normal_squared;
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(gsl::at(tangential, i), a) =
          gsl::at(vector, i) - gsl::at(normal, i) * projection;
    }
  }
  return tangential;
}

// Equivalences and eigenvector actions of apply_upwind_and_penalties: pure
// upwinding and uniform penalties reproduce the previous operator, the
// (1,1,0) candidate removes exactly the GammaHat-b projector, and the three
// penalty groups act on their full eigenvectors, including the stationary
// modes carrying tangential-gradient mismatch.
void test_upwind_and_penalties(const std::vector<Case>& all_cases,
                               const double f,
                               const gsl::not_null<std::mt19937*> gen) {
  std::normal_distribution<double> dist(0.0, 1.0);
  constexpr size_t n23 = gp::number_of_normal_components;
  for (const auto& [state, normal] : all_cases) {
    CAPTURE(normal);
    CAPTURE(state.u);
    const gp::LogMetricChart chart{
        gp::stf_decode(&state.u[gp::PotentialIndex::log_metric])};
    const gp::NormalCharacteristics characteristics{chart, state.u, normal, f};
    const double condition = transform_condition(characteristics);
    CAPTURE(condition);
    double largest_speed = 0.0;
    double smallest_speed = std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < n23; ++k) {
      const double speed = std::abs(gsl::at(characteristics.speeds(), k));
      largest_speed = std::max(largest_speed, speed);
      if (not gp::NormalCharacteristics::is_kernel_field(k)) {
        smallest_speed = std::min(smallest_speed, speed);
      }
    }
    // Dividing by the speeds (the unequal sigma_p, sigma_t branch and the
    // reference solve below) amplifies errors by their spread.
    const double oracle_condition =
        std::max(condition, 10.0 * largest_speed / smallest_speed);
    CAPTURE(oracle_condition);

    const auto apply_full = [&](const gp::VelocityVector& jw,
                                const gp::GradientVector& jq,
                                const double upwind, const double sigma_p,
                                const double sigma_t, const double sigma_g) {
      const auto flux = gp::normal_flux(chart, state.u, normal, jw, jq, f);
      gp::VelocityVector w{};
      gp::GradientVector q{};
      characteristics.apply_upwind_and_penalties(
          make_not_null(&w), make_not_null(&q), normal, jw, jq, flux, upwind,
          sigma_p, sigma_t, sigma_g);
      return flatten_target(w, q);
    };

    // A random full jump, including tangential gradient content.
    gp::VelocityVector jump_w{};
    gp::GradientVector jump_q{};
    for (auto& component : jump_w) {
      component = dist(*gen);
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        gsl::at(gsl::at(jump_q, i), a) = dist(*gen);
      }
    }
    const auto jump_flat = flatten_target(jump_w, jump_q);
    const auto flux =
        gp::normal_flux(chart, state.u, normal, jump_w, jump_q, f);

    // The GammaHat and b rows of the flux are equal, so the structural-kernel
    // entries of the transformed flux vanish identically, not just to
    // rounding.
    const auto flux_fields = characteristics.characteristic_fields(flux);
    for (size_t k = 0; k < n23; ++k) {
      if (gp::NormalCharacteristics::is_kernel_field(k)) {
        CHECK(gsl::at(flux_fields, k) == 0.0);
      }
    }

    // The previous upwind path from the unchanged primitives.
    gp::VelocityVector old_w{};
    gp::GradientVector old_q{};
    gp::embed_normal_subspace_vector(make_not_null(&old_w),
                                     make_not_null(&old_q), normal,
                                     characteristics.apply_sign(flux));
    const auto old_flat = flatten_target(old_w, old_q);
    double scale = 1.0;
    for (size_t row = 0; row < n43; ++row) {
      scale = std::max(scale, std::abs(gsl::at(old_flat, row)));
      scale = std::max(scale, std::abs(gsl::at(jump_flat, row)));
    }
    const double tolerance = 1.0e-13 * condition * scale;

    // Pure upwinding reproduces the previous operator; a zero factor with
    // zero penalties gives exact zeros.
    for (const double factor : {0.0, 1.0, 2.0}) {
      const auto actual = apply_full(jump_w, jump_q, factor, 0.0, 0.0, 0.0);
      for (size_t row = 0; row < n43; ++row) {
        if (factor == 0.0) {
          CHECK(gsl::at(actual, row) == 0.0);
        } else {
          CHECK(std::abs(gsl::at(actual, row) -
                         factor * gsl::at(old_flat, row)) < factor * tolerance);
        }
      }
    }

    // Uniform penalties are the previous operator plus sigma times the jump.
    for (const double sigma : {0.5, 1.0}) {
      const auto actual = apply_full(jump_w, jump_q, 1.0, sigma, sigma, sigma);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(std::abs(gsl::at(actual, row) - gsl::at(old_flat, row) -
                       sigma * gsl::at(jump_flat, row)) < tolerance);
      }
    }

    // A rescaled covector: the upwind part is linear in the normal while the
    // full-operator eigenvectors, and so the penalties, are unchanged.
    {
      const double rescale = 1.5 + std::abs(dist(*gen));
      const std::array<double, 3> scaled_normal{
          {rescale * normal[0], rescale * normal[1], rescale * normal[2]}};
      const gp::NormalCharacteristics scaled_characteristics{chart, state.u,
                                                             scaled_normal, f};
      const auto scaled_flux =
          gp::normal_flux(chart, state.u, scaled_normal, jump_w, jump_q, f);
      gp::VelocityVector w{};
      gp::GradientVector q{};
      scaled_characteristics.apply_upwind_and_penalties(
          make_not_null(&w), make_not_null(&q), scaled_normal, jump_w, jump_q,
          scaled_flux, 1.0, 0.8, 0.3, 0.1);
      const auto scaled_actual = flatten_target(w, q);
      const auto unscaled_upwind =
          apply_full(jump_w, jump_q, 1.0, 0.0, 0.0, 0.0);
      const auto penalties = apply_full(jump_w, jump_q, 0.0, 0.8, 0.3, 0.1);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(std::abs(gsl::at(scaled_actual, row) -
                       rescale * gsl::at(unscaled_upwind, row) -
                       gsl::at(penalties, row)) <
              1.0e-12 * oracle_condition * (1.0 + rescale) * scale);
      }
    }

    // The (1,1,0) candidate removes exactly the GammaHat-b projector.
    const auto kernel_part = kernel_projection(characteristics, normal, jump_w);
    {
      const auto actual = apply_full(jump_w, jump_q, 1.0, 1.0, 1.0, 0.0);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(std::abs(gsl::at(actual, row) - gsl::at(old_flat, row) -
                       gsl::at(jump_flat, row) + gsl::at(kernel_part, row)) <
              tolerance);
      }
    }

    // The projection depends only on the difference of the GammaHat and b
    // jumps, and a jump with equal GammaHat and b jumps has no kernel content:
    // the candidate then returns the uniform-penalty correction.
    {
      gp::VelocityVector jump_w_equal = jump_w;
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(jump_w_equal, gp::VelocityIndex::b + i) =
            gsl::at(jump_w, gp::VelocityIndex::gamma_hat + i);
      }
      const auto kernel_equal =
          kernel_projection(characteristics, normal, jump_w_equal);
      gp::VelocityVector jump_w_shifted = jump_w;
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(jump_w_shifted, gp::VelocityIndex::gamma_hat + i) += 0.3;
        gsl::at(jump_w_shifted, gp::VelocityIndex::b + i) += 0.3;
      }
      const auto kernel_shifted =
          kernel_projection(characteristics, normal, jump_w_shifted);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(std::abs(gsl::at(kernel_equal, row)) < tolerance);
        CHECK(std::abs(gsl::at(kernel_shifted, row) -
                       gsl::at(kernel_part, row)) < tolerance);
      }
      const auto candidate =
          apply_full(jump_w_equal, jump_q, 1.0, 1.0, 1.0, 0.0);
      const auto uniform = apply_full(jump_w_equal, jump_q, 1.0, 1.0, 1.0, 1.0);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(std::abs(gsl::at(candidate, row) - gsl::at(uniform, row)) <
              tolerance);
      }
    }

    // Linearity in the jump at fixed distinct coefficients.
    {
      gp::VelocityVector jump_w_2{};
      gp::GradientVector jump_q_2{};
      for (auto& component : jump_w_2) {
        component = dist(*gen);
      }
      for (size_t i = 0; i < 3; ++i) {
        for (size_t a = 0; a < gp::number_of_potentials; ++a) {
          gsl::at(gsl::at(jump_q_2, i), a) = dist(*gen);
        }
      }
      gp::VelocityVector jump_w_sum{};
      gp::GradientVector jump_q_sum{};
      for (size_t a = 0; a < gp::number_of_velocities; ++a) {
        gsl::at(jump_w_sum, a) =
            gsl::at(jump_w, a) + 2.0 * gsl::at(jump_w_2, a);
      }
      for (size_t i = 0; i < 3; ++i) {
        for (size_t a = 0; a < gp::number_of_potentials; ++a) {
          gsl::at(gsl::at(jump_q_sum, i), a) =
              gsl::at(gsl::at(jump_q, i), a) +
              2.0 * gsl::at(gsl::at(jump_q_2, i), a);
        }
      }
      const auto first = apply_full(jump_w, jump_q, 1.0, 0.8, 0.3, 0.1);
      const auto second = apply_full(jump_w_2, jump_q_2, 1.0, 0.8, 0.3, 0.1);
      const auto combined =
          apply_full(jump_w_sum, jump_q_sum, 1.0, 0.8, 0.3, 0.1);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(std::abs(gsl::at(combined, row) - gsl::at(first, row) -
                       2.0 * gsl::at(second, row)) <
              3.0 * oracle_condition * 1.0e-13 * scale);
      }
    }

    // Dense identities of the GammaHat-b projector: a projector of rank
    // three annihilated by the full operator on both sides.
    const Square a_n = assemble([&](const auto result_w, const auto result_q,
                                    const auto& w, const auto& q) {
      gp::apply_normal_matrix(result_w, result_q, chart, state.u, normal, w, q,
                              f);
    });
    const Square p_g = assemble([&](const auto result_w, const auto result_q,
                                    const auto& w, const auto& q) {
      (void)q;
      unflatten_target(result_w, result_q,
                       kernel_projection(characteristics, normal, w));
    });
    const double p_scale = std::max(1.0, max_abs(p_g));
    const double a_scale = std::max(1.0, max_abs(a_n));
    CHECK(max_abs_difference(square_multiply(p_g, p_g), p_g) <
          1.0e-12 * condition * p_scale);
    CHECK(max_abs(square_multiply(a_n, p_g)) <
          1.0e-12 * condition * a_scale * p_scale);
    CHECK(max_abs(square_multiply(p_g, a_n)) <
          1.0e-12 * condition * a_scale * p_scale);
    double trace = 0.0;
    for (size_t d = 0; d < n43; ++d) {
      trace += p_g[d * n43 + d];
    }
    CHECK(trace == approx(3.0).epsilon(1.0e-10 * condition));

    // Dense forward and inverse transforms and the dense normal block M.
    blaze::DynamicMatrix<double> t_matrix(n23, n23);
    blaze::DynamicMatrix<double> t_inverse(n23, n23);
    blaze::DynamicMatrix<double> m(n23, n23);
    for (size_t column = 0; column < n23; ++column) {
      const auto unit = unit_normal_subspace(column);
      const auto forward = characteristics.characteristic_fields(unit);
      const auto backward =
          characteristics.normal_fields_from_characteristic_fields(unit);
      gp::VelocityVector w{};
      gp::GradientVector q{};
      gp::embed_normal_subspace_vector(make_not_null(&w), make_not_null(&q),
                                       normal, unit);
      const auto image = gp::normal_flux(chart, state.u, normal, w, q, f);
      for (size_t row = 0; row < n23; ++row) {
        t_matrix(row, column) = gsl::at(forward, row);
        t_inverse(row, column) = gsl::at(backward, row);
        m(row, column) = gsl::at(image, row);
      }
    }
    // M completed by the kernel projector R_k L_k is invertible; for a
    // right-hand side with no left-kernel content its inverse solves
    // M x = b with no kernel characteristic content. This is the reference
    // construction of the K map of the stationary eigenvectors.
    blaze::DynamicMatrix<double> m_completed = m;
    for (size_t k = 0; k < n23; ++k) {
      if (not gp::NormalCharacteristics::is_kernel_field(k)) {
        continue;
      }
      for (size_t row = 0; row < n23; ++row) {
        for (size_t column = 0; column < n23; ++column) {
          m_completed(row, column) += t_inverse(row, k) * t_matrix(k, column);
        }
      }
    }
    blaze::invert(m_completed);

    // A stationary eigenvector carrying a random tangential-gradient jump.
    const gp::GradientVector tangential = random_tangential(gen, normal);
    const auto rhs = gp::normal_flux(chart, state.u, normal,
                                     gp::VelocityVector{}, tangential, f);
    blaze::DynamicVector<double> rhs_vector(n23);
    for (size_t row = 0; row < n23; ++row) {
      rhs_vector[row] = gsl::at(rhs, row);
    }
    const blaze::DynamicVector<double> k_t = m_completed * rhs_vector;
    double rhs_scale = 1.0;
    double k_t_scale = 1.0;
    for (size_t row = 0; row < n23; ++row) {
      rhs_scale = std::max(rhs_scale, std::abs(rhs_vector[row]));
      k_t_scale = std::max(k_t_scale, std::abs(k_t[row]));
    }
    const blaze::DynamicVector<double> solve_residual = m * k_t - rhs_vector;
    gp::NormalSubspaceVector k_t_subspace{};
    for (size_t row = 0; row < n23; ++row) {
      CHECK(std::abs(solve_residual[row]) <
            1.0e-11 * oracle_condition * rhs_scale);
      gsl::at(k_t_subspace, row) = k_t[row];
    }
    const auto k_t_fields = characteristics.characteristic_fields(k_t_subspace);
    for (size_t k = 0; k < n23; ++k) {
      if (gp::NormalCharacteristics::is_kernel_field(k)) {
        CHECK(std::abs(gsl::at(k_t_fields, k)) <
              1.0e-11 * oracle_condition * k_t_scale);
      }
    }
    gp::VelocityVector stationary_w{};
    gp::GradientVector stationary_q{};
    for (size_t a = 0; a < gp::number_of_velocities; ++a) {
      gsl::at(stationary_w, a) = -k_t[a];
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        gsl::at(gsl::at(stationary_q, i), a) =
            -gsl::at(normal, i) * k_t[gp::number_of_velocities + a] +
            gsl::at(gsl::at(tangential, i), a);
      }
    }
    const auto stationary_flat = flatten_target(stationary_w, stationary_q);
    double stationary_scale = 1.0;
    for (size_t row = 0; row < n43; ++row) {
      stationary_scale =
          std::max(stationary_scale, std::abs(gsl::at(stationary_flat, row)));
    }
    // The full operator annihilates it (independent check through the dense
    // column assembly).
    {
      const auto image = multiply(a_n, stationary_flat);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(std::abs(gsl::at(image, row)) <
              1.0e-10 * oracle_condition * largest_speed * stationary_scale);
      }
    }
    // The new operator weights it with sigma_t alone.
    {
      const auto actual =
          apply_full(stationary_w, stationary_q, 0.0, 0.0, 0.7, 0.0);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(std::abs(gsl::at(actual, row) -
                       0.7 * gsl::at(stationary_flat, row)) <
              1.0e-10 * oracle_condition * stationary_scale);
      }
    }

    // The production stationary coupling solves the same defining equations:
    // the inverse transform of zero characteristic amplitudes with this
    // tangential part is the dense stationary eigenvector.
    {
      gp::VelocityVector production_w{};
      gp::GradientVector production_q{};
      characteristics.full_fields_from_characteristic_fields(
          make_not_null(&production_w), make_not_null(&production_q), normal,
          gp::NormalSubspaceVector{}, tangential);
      const auto production_flat = flatten_target(production_w, production_q);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(std::abs(gsl::at(production_flat, row) -
                       gsl::at(stationary_flat, row)) <
              1.0e-10 * oracle_condition * stationary_scale);
      }
    }

    // Round trips of the full transform and the amplitude action of the full
    // operator: A_n maps the amplitudes (c, t) to (lambda c, 0).
    {
      gp::NormalSubspaceVector forward_fields{};
      gp::GradientVector forward_tangential{};
      characteristics.full_characteristic_fields(
          make_not_null(&forward_fields), make_not_null(&forward_tangential),
          normal, jump_w, jump_q);
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        CHECK(std::abs(normal[0] * gsl::at(forward_tangential[0], a) +
                       normal[1] * gsl::at(forward_tangential[1], a) +
                       normal[2] * gsl::at(forward_tangential[2], a)) <
              1.0e-13 * oracle_condition * scale);
      }
      gp::VelocityVector back_w{};
      gp::GradientVector back_q{};
      characteristics.full_fields_from_characteristic_fields(
          make_not_null(&back_w), make_not_null(&back_q), normal,
          forward_fields, forward_tangential);
      const auto back_flat = flatten_target(back_w, back_q);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(std::abs(gsl::at(back_flat, row) - gsl::at(jump_flat, row)) <
              1.0e-12 * oracle_condition * scale);
      }
      // forward of inverse on random amplitudes.
      gp::NormalSubspaceVector amplitudes{};
      for (auto& amplitude : amplitudes) {
        amplitude = dist(*gen);
      }
      const gp::GradientVector amplitude_tangential =
          random_tangential(gen, normal);
      gp::VelocityVector mode_w{};
      gp::GradientVector mode_q{};
      characteristics.full_fields_from_characteristic_fields(
          make_not_null(&mode_w), make_not_null(&mode_q), normal, amplitudes,
          amplitude_tangential);
      gp::NormalSubspaceVector recovered_fields{};
      gp::GradientVector recovered_tangential{};
      characteristics.full_characteristic_fields(
          make_not_null(&recovered_fields),
          make_not_null(&recovered_tangential), normal, mode_w, mode_q);
      double amplitude_scale = 1.0;
      for (size_t k = 0; k < n23; ++k) {
        amplitude_scale =
            std::max(amplitude_scale, std::abs(gsl::at(amplitudes, k)));
      }
      for (size_t k = 0; k < n23; ++k) {
        CHECK(std::abs(gsl::at(recovered_fields, k) - gsl::at(amplitudes, k)) <
              1.0e-11 * oracle_condition * amplitude_scale);
      }
      for (size_t i = 0; i < 3; ++i) {
        for (size_t a = 0; a < gp::number_of_potentials; ++a) {
          CHECK(std::abs(gsl::at(gsl::at(recovered_tangential, i), a) -
                         gsl::at(gsl::at(amplitude_tangential, i), a)) <
                1.0e-11 * oracle_condition * amplitude_scale);
        }
      }
      // The dense operator maps the mode to the speed-weighted amplitudes.
      const auto image = multiply(a_n, flatten_target(mode_w, mode_q));
      gp::VelocityVector image_w{};
      gp::GradientVector image_q{};
      unflatten_target(make_not_null(&image_w), make_not_null(&image_q), image);
      gp::NormalSubspaceVector image_fields{};
      gp::GradientVector image_tangential{};
      characteristics.full_characteristic_fields(
          make_not_null(&image_fields), make_not_null(&image_tangential),
          normal, image_w, image_q);
      for (size_t k = 0; k < n23; ++k) {
        CHECK(std::abs(gsl::at(image_fields, k) -
                       gsl::at(characteristics.speeds(), k) *
                           gsl::at(amplitudes, k)) <
              1.0e-9 * oracle_condition * largest_speed * amplitude_scale);
      }
      for (size_t i = 0; i < 3; ++i) {
        for (size_t a = 0; a < gp::number_of_potentials; ++a) {
          CHECK(std::abs(gsl::at(gsl::at(image_tangential, i), a)) <
                1.0e-9 * oracle_condition * largest_speed * amplitude_scale);
        }
      }
    }

    // Path equivalences: the fused dissipation agrees with the general path
    // composed from the public transforms, at the uniform and candidate
    // settings it shortcuts.
    const auto general_path = [&](const gp::VelocityVector& jw,
                                  const gp::GradientVector& jq,
                                  const double upwind, const double sigma_p,
                                  const double sigma_t, const double sigma_g) {
      gp::NormalSubspaceVector fields{};
      gp::GradientVector forward_tangential{};
      characteristics.full_characteristic_fields(
          make_not_null(&fields), make_not_null(&forward_tangential), normal,
          jw, jq);
      for (size_t k = 0; k < n23; ++k) {
        if (gp::NormalCharacteristics::is_kernel_field(k)) {
          gsl::at(fields, k) *= sigma_g;
        } else {
          gsl::at(fields, k) *=
              upwind * std::abs(gsl::at(characteristics.speeds(), k)) + sigma_p;
        }
      }
      gp::GradientVector scaled_tangential{};
      for (size_t i = 0; i < 3; ++i) {
        for (size_t a = 0; a < gp::number_of_potentials; ++a) {
          gsl::at(gsl::at(scaled_tangential, i), a) =
              sigma_t * gsl::at(gsl::at(forward_tangential, i), a);
        }
      }
      gp::VelocityVector w{};
      gp::GradientVector q{};
      characteristics.full_fields_from_characteristic_fields(
          make_not_null(&w), make_not_null(&q), normal, fields,
          scaled_tangential);
      return flatten_target(w, q);
    };
    for (const auto& [sigma_p, sigma_t, sigma_g] :
         std::array<std::array<double, 3>, 3>{
             {{{1.0, 1.0, 1.0}}, {{1.0, 1.0, 0.0}}, {{0.5, 0.5, 2.0}}}}) {
      const auto fused =
          apply_full(jump_w, jump_q, 1.0, sigma_p, sigma_t, sigma_g);
      const auto general =
          general_path(jump_w, jump_q, 1.0, sigma_p, sigma_t, sigma_g);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(std::abs(gsl::at(fused, row) - gsl::at(general, row)) <
              1.0e-11 * oracle_condition * (largest_speed + 1.0) * scale);
      }
    }

    // Independent oracle for the general path: the flux-division evaluation,
    // valid away from degeneracies. The propagating amplitudes are the
    // transformed flux over the speeds; the kernel amplitudes come from the
    // normal-subspace part of the jump.
    {
      const double sigma_p = 0.8;
      const double sigma_t = 0.3;
      const double sigma_g = 0.1;
      gp::NormalSubspaceVector z{};
      gp::GradientVector oracle_tangential{};
      const double normal_squared =
          normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
      for (size_t a = 0; a < gp::number_of_velocities; ++a) {
        gsl::at(z, a) = gsl::at(jump_w, a);
      }
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        const double r = (normal[0] * gsl::at(jump_q[0], a) +
                          normal[1] * gsl::at(jump_q[1], a) +
                          normal[2] * gsl::at(jump_q[2], a)) /
                         normal_squared;
        gsl::at(z, gp::number_of_velocities + a) = r;
        for (size_t i = 0; i < 3; ++i) {
          gsl::at(gsl::at(oracle_tangential, i), a) =
              gsl::at(gsl::at(jump_q, i), a) - gsl::at(normal, i) * r;
        }
      }
      const auto transformed_flux = characteristics.characteristic_fields(flux);
      const auto zeta = characteristics.characteristic_fields(z);
      gp::NormalSubspaceVector weighted{};
      for (size_t k = 0; k < n23; ++k) {
        if (gp::NormalCharacteristics::is_kernel_field(k)) {
          gsl::at(weighted, k) = sigma_g * gsl::at(zeta, k);
        } else {
          const double speed = gsl::at(characteristics.speeds(), k);
          gsl::at(weighted, k) =
              (speed < 0.0 ? -1.0 : 1.0) * gsl::at(transformed_flux, k) +
              sigma_t * gsl::at(zeta, k) +
              (sigma_p - sigma_t) * gsl::at(transformed_flux, k) / speed;
        }
      }
      gp::VelocityVector oracle_w{};
      gp::GradientVector oracle_q{};
      gp::embed_normal_subspace_vector(
          make_not_null(&oracle_w), make_not_null(&oracle_q), normal,
          characteristics.normal_fields_from_characteristic_fields(weighted));
      for (size_t i = 0; i < 3; ++i) {
        for (size_t a = 0; a < gp::number_of_potentials; ++a) {
          gsl::at(gsl::at(oracle_q, i), a) +=
              sigma_t * gsl::at(gsl::at(oracle_tangential, i), a);
        }
      }
      const auto oracle_flat = flatten_target(oracle_w, oracle_q);
      const auto actual =
          apply_full(jump_w, jump_q, 1.0, sigma_p, sigma_t, sigma_g);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(std::abs(gsl::at(actual, row) - gsl::at(oracle_flat, row)) <
              1.0e-10 * oracle_condition * (largest_speed + 1.0) * scale);
      }
    }

    // Kernel eigenvectors get sigma_g, propagating eigenvectors get
    // abs(speed) + sigma_p (through the general path).
    for (size_t k = 0; k < n23; ++k) {
      const auto unit = unit_normal_subspace(k);
      gp::VelocityVector mode_w{};
      gp::GradientVector mode_q{};
      gp::embed_normal_subspace_vector(
          make_not_null(&mode_w), make_not_null(&mode_q), normal,
          characteristics.normal_fields_from_characteristic_fields(unit));
      const auto mode_flat = flatten_target(mode_w, mode_q);
      double mode_scale = 1.0;
      for (size_t row = 0; row < n43; ++row) {
        mode_scale = std::max(mode_scale, std::abs(gsl::at(mode_flat, row)));
      }
      CAPTURE(k);
      if (gp::NormalCharacteristics::is_kernel_field(k)) {
        const auto actual = apply_full(mode_w, mode_q, 0.0, 0.0, 0.0, 0.9);
        for (size_t row = 0; row < n43; ++row) {
          CHECK(std::abs(gsl::at(actual, row) - 0.9 * gsl::at(mode_flat, row)) <
                1.0e-11 * oracle_condition * mode_scale);
        }
      } else {
        const auto actual = apply_full(mode_w, mode_q, 1.0, 0.4, 0.0, 0.0);
        const double expected_factor =
            std::abs(gsl::at(characteristics.speeds(), k)) + 0.4;
        for (size_t row = 0; row < n43; ++row) {
          CHECK(std::abs(gsl::at(actual, row) -
                         expected_factor * gsl::at(mode_flat, row)) <
                1.0e-11 * oracle_condition * (largest_speed + 1.0) *
                    mode_scale);
        }
      }
    }
  }
}

// Cost of the dissipation per face point, including the construction of the
// characteristic data, for the previous componentwise-penalty path and the
// new grouped penalties.
void test_timing(const gsl::not_null<std::mt19937*> gen) {
  constexpr double f = 1.0;
  struct TimingPoint {
    gp::LogMetricChart chart;
    gp::PotentialVector u;
    std::array<double, 3> normal;
    gp::VelocityVector jump_w;
    gp::GradientVector jump_q;
    gp::NormalSubspaceVector flux;
  };
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<TimingPoint> points{};
  while (points.size() < 100) {
    const auto state = helpers::random_state(gen, 0.2, 0.2, 0.3);
    const auto normal = helpers::random_unit_vector(gen);
    const gp::LogMetricChart chart{
        gp::stf_decode(&state.u[gp::PotentialIndex::log_metric])};
    if (transform_condition(
            gp::NormalCharacteristics{chart, state.u, normal, f}) > 1.0e5) {
      continue;
    }
    TimingPoint point{chart, state.u, normal, {}, {}, {}};
    for (auto& component : point.jump_w) {
      component = dist(*gen);
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        gsl::at(gsl::at(point.jump_q, i), a) = dist(*gen);
      }
    }
    point.flux =
        gp::normal_flux(chart, state.u, normal, point.jump_w, point.jump_q, f);
    points.push_back(point);
  }
  constexpr size_t passes = 50;
  double checksum = 0.0;
  const auto time_ns_per_point = [&points, &checksum](const auto& body) {
    const auto start = std::chrono::steady_clock::now();
    for (size_t pass = 0; pass < passes; ++pass) {
      for (const auto& point : points) {
        checksum += body(point);
      }
    }
    const auto stop = std::chrono::steady_clock::now();
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(stop -
                                                                    start)
                   .count()) /
           static_cast<double>(passes * points.size());
  };
  const double old_time = time_ns_per_point([](const TimingPoint& point) {
    const gp::NormalCharacteristics characteristics{point.chart, point.u,
                                                    point.normal, f};
    gp::VelocityVector w{};
    gp::GradientVector q{};
    gp::embed_normal_subspace_vector(make_not_null(&w), make_not_null(&q),
                                     point.normal,
                                     characteristics.apply_sign(point.flux));
    for (size_t a = 0; a < gp::number_of_velocities; ++a) {
      gsl::at(w, a) += gsl::at(point.jump_w, a);
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        gsl::at(gsl::at(q, i), a) += gsl::at(gsl::at(point.jump_q, i), a);
      }
    }
    return gsl::at(w, 0) + gsl::at(gsl::at(q, 0), 0);
  });
  const auto time_new = [&time_ns_per_point](const double sigma_p,
                                             const double sigma_t,
                                             const double sigma_g) {
    return time_ns_per_point(
        [sigma_p, sigma_t, sigma_g](const TimingPoint& point) {
          const gp::NormalCharacteristics characteristics{point.chart, point.u,
                                                          point.normal, f};
          gp::VelocityVector w{};
          gp::GradientVector q{};
          characteristics.apply_upwind_and_penalties(
              make_not_null(&w), make_not_null(&q), point.normal, point.jump_w,
              point.jump_q, point.flux, 1.0, sigma_p, sigma_t, sigma_g);
          return gsl::at(w, 0) + gsl::at(gsl::at(q, 0), 0);
        });
  };
  const double uniform_time = time_new(1.0, 1.0, 1.0);
  const double candidate_time = time_new(1.0, 1.0, 0.0);
  const double split_time = time_new(1.0, 0.5, 0.0);
  // The general path forced at the candidate coefficients, through the public
  // full transform.
  const double forced_general_time =
      time_ns_per_point([](const TimingPoint& point) {
        const gp::NormalCharacteristics characteristics{point.chart, point.u,
                                                        point.normal, f};
        gp::NormalSubspaceVector fields{};
        gp::GradientVector tangential{};
        characteristics.full_characteristic_fields(
            make_not_null(&fields), make_not_null(&tangential), point.normal,
            point.jump_w, point.jump_q);
        for (size_t k = 0; k < gp::number_of_normal_components; ++k) {
          if (not gp::NormalCharacteristics::is_kernel_field(k)) {
            gsl::at(fields, k) *=
                std::abs(gsl::at(characteristics.speeds(), k)) + 1.0;
          } else {
            gsl::at(fields, k) = 0.0;
          }
        }
        gp::VelocityVector w{};
        gp::GradientVector q{};
        characteristics.full_fields_from_characteristic_fields(
            make_not_null(&w), make_not_null(&q), point.normal, fields,
            tangential);
        return gsl::at(w, 0) + gsl::at(gsl::at(q, 0), 0);
      });
  Parallel::printf(
      "UpwindGradientProjection dissipation (ns per point, including the "
      "characteristic construction): previous componentwise %.0f, "
      "uniform (1,1,1) %.0f, candidate (1,1,0) %.0f, general split "
      "(1,0.5,0) %.0f, general forced at the candidate %.0f "
      "(checksum %.6e)\n",
      old_time, uniform_time, candidate_time, split_time, forced_general_time,
      checksum);
}

// The conformal factor is no longer positive by construction: the normal
// operator, its characteristic decomposition and speeds reject it.
void test_conformal_factor_positivity(const double f) {
  const gp::LogMetricChart flat{gp::Matrix3{}};
  const std::array<double, 3> normal{{0.0, 1.0, 0.0}};
  for (const double phi : {0.0, -0.3}) {
    CAPTURE(phi);
    gp::PotentialVector u{};
    u[gp::PotentialIndex::lapse] = 1.0;
    u[gp::PotentialIndex::conformal_factor] = phi;
    CHECK_THROWS_WITH(
        (gp::NormalCharacteristics{flat, u, normal, f}),
        Catch::Matchers::ContainsSubstring("The conformal factor must be "
                                           "positive"));
    CHECK_THROWS_WITH(
        gp::largest_characteristic_speed(flat, u, normal, f),
        Catch::Matchers::ContainsSubstring("The conformal factor must be "
                                           "positive"));
    CHECK_THROWS_WITH(gp::normal_flux(flat, u, normal, gp::VelocityVector{},
                                      gp::GradientVector{}, f),
                      Catch::Matchers::ContainsSubstring(
                          "The conformal factor must be positive"));
  }
}
}  // namespace

// [[TimeOut, 60]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection.NormalOperator",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Evolution/Systems/Ccz4/GradientProjection/");
  MAKE_GENERATOR(gen);
  for (const double f : {1.0, 0.5, 1.25}) {
    CAPTURE(f);
    const auto all_cases = cases(make_not_null(&gen), f);
    test_normal_matrix(all_cases, f);
    test_absolute_matrix(all_cases, f);
    test_characteristics(all_cases, f, make_not_null(&gen));
    test_normalization_conversion(all_cases, f);
    test_upwind_and_penalties(all_cases, f, make_not_null(&gen));
    test_near_collisions(f);
    test_degeneracies(f);
    test_phi_similarity(make_not_null(&gen), f);
    test_conformal_factor_positivity(f);
  }
  test_gamma_driver_parameter();
  test_timing(make_not_null(&gen));
}
