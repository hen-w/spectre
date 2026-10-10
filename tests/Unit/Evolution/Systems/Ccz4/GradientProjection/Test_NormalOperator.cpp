// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
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
    const double phi =
        std::exp(state.u[gp::PotentialIndex::log_conformal_factor]);
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
    u[gp::PotentialIndex::log_conformal_factor] = std::log(phi);
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
    u[gp::PotentialIndex::log_conformal_factor] = std::log(phi);
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
    test_near_collisions(f);
    test_degeneracies(f);
  }
  test_gamma_driver_parameter();
}
