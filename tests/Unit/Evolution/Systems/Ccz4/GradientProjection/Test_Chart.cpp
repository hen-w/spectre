// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;
using helpers::flatten;
using helpers::to_data_vector;
constexpr const char* module = "GradientProjectionReference";

gp::Matrix3 identity() {
  gp::Matrix3 result{};
  for (size_t i = 0; i < 3; ++i) {
    result(i, i) = 1.0;
  }
  return result;
}

double trace(const gp::Matrix3& m) { return m(0, 0) + m(1, 1) + m(2, 2); }

// The fixed STF basis is orthonormal, matches the archived model ordering and
// encodes the trace-free part only.
void test_stf_basis(const gsl::not_null<std::mt19937*> gen) {
  for (size_t a = 0; a < 5; ++a) {
    std::array<double, 5> unit{};
    gsl::at(unit, a) = 1.0;
    const gp::Matrix3 basis = gp::stf_decode(unit.data());
    CHECK(helpers::max_abs_difference(
              basis, pypp::call<DataVector>(module, "stf_basis_matrix",
                                            static_cast<double>(a))) < 1e-15);
    CHECK(std::abs(trace(basis)) < 1e-15);
    CHECK(helpers::max_abs(basis - blaze::trans(basis)) == 0.0);
    for (size_t b = 0; b < 5; ++b) {
      std::array<double, 5> other{};
      gsl::at(other, b) = 1.0;
      const gp::Matrix3 basis_b = gp::stf_decode(other.data());
      double frobenius = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          frobenius += basis(i, j) * basis_b(i, j);
        }
      }
      CHECK(frobenius == approx(a == b ? 1.0 : 0.0));
    }
  }
  const gp::Matrix3 x = helpers::random_symmetric(gen, 1.0);
  const std::array<double, 5> encoded = gp::stf_encode(x);
  const gp::Matrix3 traceless = x - (trace(x) / 3.0) * identity();
  CHECK(helpers::max_abs(gp::stf_decode(encoded.data()) - traceless) < 1e-14);
  const std::array<double, 5> round_trip =
      gp::stf_encode(gp::stf_decode(encoded.data()));
  for (size_t a = 0; a < 5; ++a) {
    CHECK(gsl::at(round_trip, a) == approx(gsl::at(encoded, a)));
  }
}

// LAPACK dsyev through blaze: ascending eigenvalues, eigenvectors in the
// columns, for rotated anisotropic and repeated spectra.
void test_eigensystem(const gsl::not_null<std::mt19937*> gen) {
  for (const auto& spectrum :
       std::vector<std::array<double, 3>>{{{0.3, 2.0, 7.0}},
                                          {{1.0, 1.0, 4.0}},
                                          {{2.0, 2.0, 2.0}},
                                          {{1.0e-4, 1.0, 1.0e4}}}) {
    const gp::Matrix3 rotation = helpers::random_rotation(gen);
    gp::Matrix3 diagonal{};
    for (size_t i = 0; i < 3; ++i) {
      diagonal(i, i) = gsl::at(spectrum, i);
    }
    const gp::Matrix3 matrix = rotation * diagonal * blaze::trans(rotation);
    std::array<double, 3> values{};
    gp::Matrix3 vectors{};
    gp::symmetric_eigensystem(make_not_null(&values), make_not_null(&vectors),
                              matrix);
    for (size_t i = 0; i < 3; ++i) {
      CHECK(gsl::at(values, i) ==
            approx(gsl::at(spectrum, i)).scale(spectrum[2]));
    }
    CHECK(helpers::max_abs(blaze::trans(vectors) * vectors - identity()) <
          1e-14);
    gp::Matrix3 values_diagonal{};
    for (size_t i = 0; i < 3; ++i) {
      values_diagonal(i, i) = gsl::at(values, i);
    }
    CHECK(helpers::max_abs(matrix * vectors - vectors * values_diagonal) <
          1e-13 * spectrum[2]);
  }
}

// The chart maps against SciPy's matrix exponential, logarithm and Frechet
// derivatives, including repeated and nearly repeated spectra and
// ill-conditioned metrics.
void test_chart_maps(const gsl::not_null<std::mt19937*> gen) {
  std::vector<gp::Matrix3> log_metrics{gp::Matrix3{}};
  for (const auto& spectrum : std::vector<std::array<double, 3>>{
           {{0.0, 0.0, 0.0}},
           {{0.1, 0.1, -0.2}},
           {{0.1, 0.1 + 1.0e-12, -0.2}},
           {{0.1, 0.1 + 1.0e-9, -0.2}},
           {{0.1, 0.1 + 1.0e-6, -0.2}},
           {{0.1, 0.1 + 1.0e-3, -0.2}},
           {{0.1 + 1.0e-10, 0.1, 0.1 - 1.0e-10}},
           {{0.4, -0.1, -0.3}},
           {{1.2, -0.3, -0.9}},
           // lambda_max / lambda_min = exp(2 * 4.6) ~ 1e4
           {{2.3, 0.0, -2.3}}}) {
    log_metrics.push_back(helpers::log_metric_with_spectrum(gen, spectrum));
  }
  for (size_t sample = 0; sample < 4; ++sample) {
    log_metrics.push_back(helpers::random_symmetric(gen, 0.3));
  }
  for (gp::Matrix3 h : log_metrics) {
    h -= (trace(h) / 3.0) * identity();
    CAPTURE(h);
    const gp::LogMetricChart chart{h};
    const gp::Matrix3 metric = chart.metric();
    const double scale = helpers::max_abs(metric);
    // Condition number lambda_max / lambda_min of the metric.
    const double condition = std::exp(
        2.0 * (chart.log_eigenvalues()[2] - chart.log_eigenvalues()[0]));
    CAPTURE(condition);
    CHECK(helpers::max_abs_difference(
              metric, pypp::call<DataVector>(module, "metric", flatten(h))) <
          1e-13 * scale);
    CHECK(helpers::max_abs(chart.inverse_metric() * metric - identity()) <
          1e-14 * (10.0 + condition));
    CHECK(helpers::max_abs(chart.sqrt_metric() * chart.sqrt_metric() - metric) <
          1e-14 * scale);
    CHECK(helpers::max_abs(chart.inverse_sqrt_metric() * chart.sqrt_metric() -
                           identity()) < 1e-13);
    CHECK(helpers::max_abs(chart.log_metric() - h) < 1e-14);

    // Round trip through the metric, and against SciPy's logm.
    const gp::LogMetricChart from_metric =
        gp::LogMetricChart::from_metric(metric);
    // |delta H| <= |delta g| / (2 lambda_min), so the round trip loses
    // accuracy in proportion to the metric's condition number.
    CHECK(helpers::max_abs(from_metric.log_metric() - h) <
          1e-15 * (100.0 + condition));
    CHECK(helpers::max_abs_difference(
              from_metric.log_metric(),
              pypp::call<DataVector>(module, "log_metric", flatten(metric))) <
          1e-15 * (1000.0 + condition));

    for (size_t sample = 0; sample < 3; ++sample) {
      const gp::Matrix3 x = helpers::random_symmetric(gen, 1.0);
      const gp::Matrix3 y = helpers::random_symmetric(gen, 1.0);
      const gp::Matrix3 tangent = chart.tangent(x);
      const double tangent_scale = helpers::max_abs(tangent) + scale;
      CHECK(helpers::max_abs_difference(
                tangent, pypp::call<DataVector>(module, "tangent", flatten(h),
                                                flatten(x))) <
            1e-13 * tangent_scale);
      // The tangent map's condition number is that of the metric.
      CHECK(helpers::max_abs(chart.inverse_tangent(tangent) - x) <
            1e-15 * (100.0 + condition));
      const gp::Matrix3 inverse_tangent = chart.inverse_tangent(x);
      CHECK(
          helpers::max_abs_difference(
              inverse_tangent, pypp::call<DataVector>(module, "inverse_tangent",
                                                      flatten(h), flatten(x))) <
          1e-11 * (helpers::max_abs(inverse_tangent) + 1.0));
      const gp::Matrix3 second = chart.second_derivative(x, y);
      CHECK(helpers::max_abs_difference(
                second,
                pypp::call<DataVector>(module, "second_derivative", flatten(h),
                                       flatten(x), flatten(y))) <
            1e-12 * (helpers::max_abs(second) + scale));
      CHECK(helpers::max_abs(second - chart.second_derivative(y, x)) <
            1e-13 * (helpers::max_abs(second) + scale));
      // tr(g^{-1} T_H X) = 2 tr X
      CHECK(trace(chart.inverse_metric() * tangent) ==
            approx(2.0 * trace(x)).epsilon(1e-12).scale(tangent_scale));
    }
  }

  // At H = 0: T X = 2 X and S[X, Y] = 2 (XY + YX).
  const gp::LogMetricChart flat{gp::Matrix3{}};
  const gp::Matrix3 x = helpers::random_symmetric(gen, 1.0);
  const gp::Matrix3 y = helpers::random_symmetric(gen, 1.0);
  CHECK(helpers::max_abs(flat.tangent(x) - 2.0 * x) < 1e-15);
  CHECK(helpers::max_abs(flat.second_derivative(x, y) - 2.0 * (x * y + y * x)) <
        1e-14);

  // Non-positive-definite metrics are rejected, not clipped.
  gp::Matrix3 indefinite = identity();
  indefinite(2, 2) = -0.5;
  CHECK_THROWS_WITH(gp::LogMetricChart::from_metric(indefinite),
                    Catch::Matchers::ContainsSubstring(
                        "The conformal metric is not positive definite"));
}

// Divided differences of exp against high-precision Python Decimal values.
void test_divided_differences(const gsl::not_null<std::mt19937*> gen) {
  std::uniform_real_distribution<double> center(-3.0, 3.0);
  for (const double gap : {0.0, 1.0e-15, 1.0e-12, 1.0e-8, 1.0e-4, 0.1, 0.5,
                           0.99, 1.01, 2.0, 6.0}) {
    for (size_t sample = 0; sample < 3; ++sample) {
      const double x = center(*gen);
      const double y = x + gap;
      const double z = x + 0.37 * gap;
      CAPTURE(x);
      CAPTURE(gap);
      const double first = gp::exp_divided_difference(x, y);
      CHECK(first ==
            approx(pypp::call<double>(module, "exp_divided_difference", x, y))
                .epsilon(1e-14));
      for (const auto& nodes : std::vector<std::array<double, 3>>{
               {{x, y, z}}, {{x, x, y}}, {{y, x, y}}, {{z, y, x}}}) {
        const double second =
            gp::exp_second_divided_difference(nodes[0], nodes[1], nodes[2]);
        CHECK(second ==
              approx(pypp::call<double>(module, "exp_second_divided_difference",
                                        nodes[0], nodes[1], nodes[2]))
                  .epsilon(2e-14));
      }
    }
  }
  CHECK(gp::exp_second_divided_difference(0.3, 0.3, 0.3) ==
        approx(0.5 * std::exp(0.3)));
  CHECK(gp::sinhc(0.0) == 1.0);
}

gp::PotentialVector add(const gp::PotentialVector& a,
                        const gp::PotentialVector& b, const double factor) {
  gp::PotentialVector result{};
  for (size_t i = 0; i < result.size(); ++i) {
    gsl::at(result, i) = gsl::at(a, i) + factor * gsl::at(b, i);
  }
  return result;
}

// R, its inverse P and the potential derivative R_u against the archived
// model functions (complex-step Jacobian for R_u), for the Gamma-driver
// parameter f and physical conformal factors in `phi_range`. Independently of
// the physical-phi reference: R against the frozen log(phi) right-hand side
// (R_phi = phi R_log(phi) with q_log(phi) = q_phi / phi), R_u against central
// differences of R itself, L P = I with the production L, and the explicit
// conformal-factor rows of L, E and R_u.
void test_potential_rhs(const gsl::not_null<std::mt19937*> gen, const double f,
                        const std::array<double, 2>& phi_range) {
  for (size_t sample = 0; sample < 12; ++sample) {
    const auto state = helpers::random_state(gen, sample < 4 ? 0.0 : 0.3, 0.3,
                                             0.2, {{0.4, 1.5}}, phi_range);
    const gp::LogMetricChart chart{
        gp::stf_decode(&state.u[gp::PotentialIndex::log_metric])};
    const double phi = state.u[gp::PotentialIndex::conformal_factor];
    const double lapse = state.u[gp::PotentialIndex::lapse];
    const gp::PotentialVector rhs =
        gp::potential_rhs(chart, state.u, state.w, state.q, f);
    const DataVector expected = pypp::call<DataVector>(
        module, "original_rhs", to_data_vector(state.u),
        to_data_vector(state.w), to_data_vector(state.q), f);
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      CHECK(gsl::at(rhs, a) == approx(expected[a]).epsilon(1e-13));
    }
    // The frozen log(phi) right-hand side at the same state.
    gp::PotentialVector u_log = state.u;
    u_log[gp::PotentialIndex::conformal_factor] = std::log(phi);
    gp::GradientVector q_log = state.q;
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(q_log, i)[gp::PotentialIndex::conformal_factor] /= phi;
    }
    const DataVector log_rhs = pypp::call<DataVector>(
        module, "original_rhs_log", to_data_vector(u_log),
        to_data_vector(state.w), to_data_vector(q_log), f);
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      const double scaled_log_rhs = a == gp::PotentialIndex::conformal_factor
                                        ? phi * log_rhs[a]
                                        : log_rhs[a];
      CHECK(gsl::at(rhs, a) == approx(scaled_log_rhs).epsilon(1e-13));
    }

    // L P = I and Z P = 0, where L y = R(u, y, 0).
    std::normal_distribution<double> normal(0.0, 1.0);
    gp::PotentialVector delta{};
    for (auto& component : delta) {
      component = normal(*gen);
    }
    const gp::VelocityVector inverse =
        gp::apply_rhs_inverse(lapse, phi, delta, f);
    const gp::PotentialVector l_p =
        gp::potential_rhs(chart, state.u, inverse, gp::GradientVector{}, f);
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      CHECK(gsl::at(l_p, a) == approx(gsl::at(delta, a)).epsilon(1e-13));
    }
    for (size_t i = 0; i < 3; ++i) {
      CHECK(gsl::at(inverse, gp::VelocityIndex::gamma_hat + i) ==
            gsl::at(inverse, gp::VelocityIndex::b + i));
    }
    // The model's recovery with q = 0 and Gamma - b = 0 is P.
    const DataVector recovered =
        pypp::call<DataVector>(module, "recover", to_data_vector(state.u),
                               to_data_vector(gp::GradientVector{}),
                               to_data_vector(delta), DataVector{3, 0.0}, f);
    for (size_t a = 0; a < gp::number_of_velocities; ++a) {
      CHECK(gsl::at(inverse, a) == approx(recovered[a]).epsilon(1e-13));
    }

    // The conformal-factor rows: L maps delta K to alpha phi delta K / 3, E
    // has the advection coefficient beta^i and the shift-divergence
    // coefficient -phi / 3.
    gp::VelocityVector unit_k{};
    unit_k[gp::VelocityIndex::trace_extrinsic_curvature] = 1.0;
    CHECK(gp::potential_rhs(chart, state.u, unit_k, gp::GradientVector{},
                            f)[gp::PotentialIndex::conformal_factor] ==
          approx(lapse * phi / 3.0));
    for (size_t i = 0; i < 3; ++i) {
      gp::GradientVector unit_q_phi{};
      gsl::at(unit_q_phi, i)[gp::PotentialIndex::conformal_factor] = 1.0;
      CHECK(gp::potential_rhs(chart, state.u, gp::VelocityVector{}, unit_q_phi,
                              f)[gp::PotentialIndex::conformal_factor] ==
            approx(gsl::at(state.u, gp::PotentialIndex::shift + i)));
      gp::GradientVector unit_divergence{};
      gsl::at(gsl::at(unit_divergence, i), gp::PotentialIndex::shift + i) = 1.0;
      CHECK(gp::potential_rhs(
                chart, state.u, gp::VelocityVector{}, unit_divergence,
                f)[gp::PotentialIndex::conformal_factor] == approx(-phi / 3.0));
    }

    // R_u against a complex-step Jacobian and central differences of R.
    const DataVector jacobian = pypp::call<DataVector>(
        module, "rhs_potential_jacobian", to_data_vector(state.u),
        to_data_vector(state.w), to_data_vector(state.q), f);
    const double step = 1.0e-6;
    for (size_t direction = 0; direction < gp::number_of_potentials;
         ++direction) {
      gp::PotentialVector delta_u{};
      gsl::at(delta_u, direction) = 1.0;
      const gp::PotentialVector derivative =
          gp::potential_rhs_potential_derivative(chart, state.u, state.w,
                                                 state.q, delta_u);
      const gp::PotentialVector u_plus = add(state.u, delta_u, step);
      const gp::PotentialVector u_minus = add(state.u, delta_u, -step);
      const gp::PotentialVector difference =
          add(gp::potential_rhs(gp::LogMetricChart{gp::stf_decode(
                                    &u_plus[gp::PotentialIndex::log_metric])},
                                u_plus, state.w, state.q, f),
              gp::potential_rhs(gp::LogMetricChart{gp::stf_decode(
                                    &u_minus[gp::PotentialIndex::log_metric])},
                                u_minus, state.w, state.q, f),
              -1.0);
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        CHECK(gsl::at(derivative, a) ==
              approx(jacobian[10 * a + direction]).epsilon(1e-12));
        CHECK(gsl::at(derivative, a) ==
              approx(gsl::at(difference, a) / (2.0 * step))
                  .epsilon(1e-8)
                  .scale(1.0));
      }
    }
    // The conformal-factor row of R_u: delta phi (alpha K - d) / 3 +
    // phi K delta alpha / 3 + delta beta^i q_{phi,i}.
    const double trace_k =
        state.w[gp::VelocityIndex::trace_extrinsic_curvature];
    double divergence = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      divergence += gsl::at(gsl::at(state.q, i), gp::PotentialIndex::shift + i);
    }
    gp::PotentialVector unit_phi{};
    unit_phi[gp::PotentialIndex::conformal_factor] = 1.0;
    CHECK(gp::potential_rhs_potential_derivative(
              chart, state.u, state.w, state.q,
              unit_phi)[gp::PotentialIndex::conformal_factor] ==
          approx((lapse * trace_k - divergence) / 3.0));
    gp::PotentialVector unit_lapse{};
    unit_lapse[gp::PotentialIndex::lapse] = 1.0;
    CHECK(gp::potential_rhs_potential_derivative(
              chart, state.u, state.w, state.q,
              unit_lapse)[gp::PotentialIndex::conformal_factor] ==
          approx(phi * trace_k / 3.0));
  }
}

void test_chart_from_physical(const gsl::not_null<std::mt19937*> gen) {
  const auto state = helpers::random_state(gen, 0.3, 0.3, 0.2);
  const gp::LogMetricChart chart{
      gp::stf_decode(&state.u[gp::PotentialIndex::log_metric])};
  gp::PhysicalPointState physical{};
  physical.conformal_metric = chart.metric();
  physical.conformal_factor = state.u[gp::PotentialIndex::conformal_factor];
  physical.a_tilde =
      0.5 * chart.tangent(gp::stf_decode(&state.w[gp::VelocityIndex::a]));
  physical.trace_extrinsic_curvature =
      state.w[gp::VelocityIndex::trace_extrinsic_curvature];
  physical.theta = state.w[gp::VelocityIndex::theta];
  physical.lapse = state.u[gp::PotentialIndex::lapse];
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(physical.shift, i) =
        gsl::at(state.u, gp::PotentialIndex::shift + i);
    gsl::at(physical.gamma_hat, i) =
        gsl::at(state.w, gp::VelocityIndex::gamma_hat + i);
    gsl::at(physical.b, i) = gsl::at(state.w, gp::VelocityIndex::b + i);
  }
  gp::PotentialVector u{};
  gp::VelocityVector w{};
  const auto traces =
      gp::chart_from_physical(make_not_null(&u), make_not_null(&w), physical);
  CHECK(std::abs(traces[0]) < 1e-14);
  CHECK(std::abs(traces[1]) < 1e-13);
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    CHECK(gsl::at(u, a) == approx(gsl::at(state.u, a)));
  }
  // The conformal factor itself is the potential.
  CHECK(u[gp::PotentialIndex::conformal_factor] == physical.conformal_factor);
  for (size_t a = 0; a < gp::number_of_velocities; ++a) {
    CHECK(gsl::at(w, a) == approx(gsl::at(state.w, a)));
  }
  // Off-constraint data is reported through the traces.
  gp::PhysicalPointState off_constraint = physical;
  off_constraint.conformal_metric *= 1.1;
  off_constraint.a_tilde += 0.2 * physical.conformal_metric;
  const auto off_traces = gp::chart_from_physical(
      make_not_null(&u), make_not_null(&w), off_constraint);
  CHECK(off_traces[0] == approx(1.5 * std::log(1.1)));
  CHECK(off_traces[1] == approx(0.6 / 1.1));

  gp::PhysicalPointState bad_lapse = physical;
  bad_lapse.lapse = 0.0;
  CHECK_THROWS_WITH(
      gp::chart_from_physical(make_not_null(&u), make_not_null(&w), bad_lapse),
      Catch::Matchers::ContainsSubstring("The lapse must be positive"));
  gp::PhysicalPointState bad_phi = physical;
  bad_phi.conformal_factor = -1.0;
  CHECK_THROWS_WITH(
      gp::chart_from_physical(make_not_null(&u), make_not_null(&w), bad_phi),
      Catch::Matchers::ContainsSubstring(
          "The conformal factor must be positive"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.GradientProjection.Chart",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Evolution/Systems/Ccz4/GradientProjection/");
  MAKE_GENERATOR(gen);
  test_stf_basis(make_not_null(&gen));
  test_eigensystem(make_not_null(&gen));
  test_chart_maps(make_not_null(&gen));
  test_divided_differences(make_not_null(&gen));
  for (const double f : {1.0, 0.5, 1.25}) {
    CAPTURE(f);
    test_potential_rhs(make_not_null(&gen), f, {{0.6, 1.3}});
  }
  // Small conformal factors exercise the 1 / phi of P.
  test_potential_rhs(make_not_null(&gen), 1.0, {{0.02, 0.1}});
  test_chart_from_physical(make_not_null(&gen));
}
