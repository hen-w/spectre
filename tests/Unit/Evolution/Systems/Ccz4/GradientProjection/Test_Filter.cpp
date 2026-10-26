// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Filter.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;

struct PhysicalState {
  tnsr::ii<DataVector, 3> conformal_metric;
  tnsr::ii<DataVector, 3> a_tilde;
  Scalar<DataVector> trace_extrinsic_curvature;
  Scalar<DataVector> theta;
  tnsr::I<DataVector, 3> gamma_hat;
  tnsr::I<DataVector, 3> b;
};

// An on-constraint physical state with a spatially varying metric (or the
// unit metric).
PhysicalState random_state(const gsl::not_null<std::mt19937*> gen,
                           const size_t n, const bool unit_metric) {
  PhysicalState state{tnsr::ii<DataVector, 3>{n}, tnsr::ii<DataVector, 3>{n},
                      Scalar<DataVector>{n},      Scalar<DataVector>{n},
                      tnsr::I<DataVector, 3>{n},  tnsr::I<DataVector, 3>{n}};
  std::normal_distribution<double> normal(0.0, 1.0);
  for (size_t s = 0; s < n; ++s) {
    const auto chart_state =
        helpers::random_state(gen, unit_metric ? 0.0 : 0.4, 0.0, 0.5);
    const gp::LogMetricChart chart{
        gp::stf_decode(&chart_state.u[gp::PotentialIndex::log_metric])};
    const gp::Matrix3 metric = chart.metric();
    const gp::Matrix3 a_tilde =
        0.5 *
        chart.tangent(gp::stf_decode(&chart_state.w[gp::VelocityIndex::a]));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        state.conformal_metric.get(i, j)[s] = metric(i, j);
        state.a_tilde.get(i, j)[s] = a_tilde(i, j);
      }
      state.gamma_hat.get(i)[s] = normal(*gen);
      state.b.get(i)[s] = normal(*gen);
    }
    get(state.trace_extrinsic_curvature)[s] = normal(*gen);
    get(state.theta)[s] = normal(*gen);
  }
  return state;
}

DataVector trace_with_metric(const tnsr::ii<DataVector, 3>& metric,
                             const tnsr::ii<DataVector, 3>& a_tilde) {
  const auto inverse = determinant_and_inverse(metric).second;
  DataVector result(get<0, 0>(metric).size(), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result += inverse.get(i, j) * a_tilde.get(i, j);
    }
  }
  return result;
}

// The velocity filter on a curved element (non-separable mass) and a random
// on-constraint state with random stored gradients: it realizes the defining
// relations R(u,w+,q) = F R(u,w,q) and GammaHat+ - b+ = F (GammaHat - b)
// (checked with the output and the unchanged inputs only, never through the
// inverse), keeps ATilde consistent with the local a and trace-free, is the
// identity at gain 1, reduces to the componentwise filter on a constant unit
// background with vanishing gradients, and differs from it otherwise.
void test_filter_velocities(const gsl::not_null<std::mt19937*> gen,
                            const double gain, const bool unit_background) {
  CAPTURE(gain);
  CAPTURE(unit_background);
  const auto geometry = helpers::make_geometry({{4, 5, 4}}, true);
  const gp::LocalOperators operators{geometry.mesh, geometry.inverse_jacobian,
                                     geometry.det_inverse_jacobian, 4};
  const size_t n = geometry.mesh.number_of_grid_points();
  const double f = Ccz4::fd::System::f;
  helpers::ChartData chart{n};
  gp::Tags::PotentialGradients::type q{n, 0.0};
  gp::PotentialVector constant_u{};
  constant_u[gp::PotentialIndex::conformal_factor] = 1.0;
  constant_u[gp::PotentialIndex::lapse] = 1.0;
  for (size_t s = 0; s < n; ++s) {
    const auto state = helpers::random_state(gen, 0.4, 0.3, 0.5);
    chart.set(s, unit_background ? constant_u : state.u, state.w);
    if (not unit_background) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t c = 0; c < gp::number_of_potentials; ++c) {
          q.get(i, c)[s] = gsl::at(gsl::at(state.q, i), c);
        }
      }
    }
  }
  const std::vector<gp::LogMetricChart> charts =
      gp::log_metric_charts(chart.potentials());
  const helpers::ChartData original = chart;
  const auto rates_and_z = [&](const helpers::ChartData& data) {
    gp::Tags::PotentialRateTarget::type rates{n};
    tnsr::I<DataVector, 3> z{n};
    for (size_t s = 0; s < n; ++s) {
      const gp::VelocityVector w_s = gp::velocities_at(data.velocities(), s);
      gp::set_potentials_at(
          make_not_null(&rates), s,
          gp::potential_rhs(charts[s], gp::potentials_at(data.potentials(), s),
                            w_s, gp::gradients_at(q, s), f));
      for (size_t i = 0; i < 3; ++i) {
        z.get(i)[s] = gsl::at(w_s, gp::VelocityIndex::gamma_hat + i) -
                      gsl::at(w_s, gp::VelocityIndex::b + i);
      }
    }
    return std::make_pair(rates, z);
  };
  const auto [original_rates, original_z] = rates_and_z(original);
  gp::filter_velocities(make_not_null(&chart.a), make_not_null(&chart.a_tilde),
                        make_not_null(&chart.trace_extrinsic_curvature),
                        make_not_null(&chart.theta),
                        make_not_null(&chart.gamma_hat),
                        make_not_null(&chart.b), chart.potentials(), charts, q,
                        geometry.mesh, operators, gain, f);
  // The potentials are untouched (they are not even writable above).
  CHECK(chart.log_conformal_metric == original.log_conformal_metric);
  CHECK(chart.conformal_metric == original.conformal_metric);
  const auto filtered = [&operators, &gain](DataVector field) {
    operators.apply_filter(make_not_null(&field), gain);
    return field;
  };
  const auto custom = Approx::custom().epsilon(1.0e-12).scale(1.0);
  if (gain == 1.0) {
    CHECK(chart.a == original.a);
    CHECK(chart.a_tilde == original.a_tilde);
    CHECK(chart.trace_extrinsic_curvature ==
          original.trace_extrinsic_curvature);
    CHECK(chart.theta == original.theta);
    CHECK(chart.gamma_hat == original.gamma_hat);
    CHECK(chart.b == original.b);
    return;
  }
  // The defining relations, from the output and the inputs.
  const auto [new_rates, new_z] = rates_and_z(chart);
  for (size_t c = 0; c < gp::number_of_potentials; ++c) {
    CHECK_ITERABLE_CUSTOM_APPROX(new_rates.get(c),
                                 filtered(original_rates.get(c)), custom);
  }
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_CUSTOM_APPROX(new_z.get(i), filtered(original_z.get(i)),
                                 custom);
  }
  // The physical ATilde is the chart's a at the unchanged H, and so remains
  // trace-free: decoding it with the production chart conversion returns a.
  for (size_t s = 0; s < n; ++s) {
    gp::PhysicalPointState state{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        state.conformal_metric(i, j) = chart.conformal_metric.get(i, j)[s];
        state.a_tilde(i, j) = chart.a_tilde.get(i, j)[s];
      }
    }
    state.conformal_factor = get(chart.conformal_factor)[s];
    state.lapse = get(chart.lapse)[s];
    gp::PotentialVector u_s{};
    gp::VelocityVector w_s{};
    gp::LogMetricChart decoded_chart{};
    const auto traces =
        gp::chart_from_physical(make_not_null(&u_s), make_not_null(&w_s),
                                make_not_null(&decoded_chart), state);
    CHECK(std::abs(traces[1]) < 1.0e-12);
    for (size_t c = 0; c < 5; ++c) {
      CHECK(gsl::at(w_s, gp::VelocityIndex::a + c) ==
            custom(chart.a.get(c)[s]));
    }
  }
  CHECK(max(abs(trace_with_metric(chart.conformal_metric, chart.a_tilde))) <
        1.0e-12);
  // On the constant unit background with q = 0 the rates are a constant
  // linear map of w, so the filter is the componentwise filter of each
  // velocity; on the varying background it is not.
  double largest_difference = 0.0;
  for (size_t c = 0; c < 5; ++c) {
    largest_difference =
        std::max(largest_difference,
                 max(abs(chart.a.get(c) - filtered(original.a.get(c)))));
  }
  largest_difference =
      std::max(largest_difference,
               max(abs(get(chart.trace_extrinsic_curvature) -
                       filtered(get(original.trace_extrinsic_curvature)))));
  largest_difference =
      std::max(largest_difference,
               max(abs(get(chart.theta) - filtered(get(original.theta)))));
  for (size_t i = 0; i < 3; ++i) {
    largest_difference = std::max(
        largest_difference,
        max(abs(chart.gamma_hat.get(i) - filtered(original.gamma_hat.get(i)))));
    largest_difference =
        std::max(largest_difference,
                 max(abs(chart.b.get(i) - filtered(original.b.get(i)))));
  }
  CAPTURE(largest_difference);
  if (unit_background) {
    CHECK(largest_difference < 1.0e-12);
  } else {
    CHECK(largest_difference > 1.0e-3);
  }
}

// The gain option: values in [0, 1] are accepted and passed through
// unchanged; values outside are rejected.
void test_gain_option() {
  using option_tag = gp::OptionTags::TopShellFilterGain;
  CHECK(TestHelpers::test_option_tag<option_tag>("0.0") == 0.0);
  CHECK(TestHelpers::test_option_tag<option_tag>("0.25") == 0.25);
  CHECK(TestHelpers::test_option_tag<option_tag>("1.0") == 1.0);
  CHECK(gp::Tags::TopShellFilterGain::create_from_options(0.25) == 0.25);
  CHECK_THROWS_WITH(TestHelpers::test_option_tag<option_tag>("-0.1"),
                    Catch::Matchers::ContainsSubstring(
                        "Value -0.1 is below the lower bound of 0"));
  CHECK_THROWS_WITH(TestHelpers::test_option_tag<option_tag>("1.5"),
                    Catch::Matchers::ContainsSubstring(
                        "Value 1.5 is above the upper bound of 1"));
}

void test_enforce_algebraic_constraints(
    const gsl::not_null<std::mt19937*> gen) {
  const size_t n = 7;
  PhysicalState state = random_state(gen, n, false);
  for (auto& component : state.conformal_metric) {
    component *= 1.3;
  }
  for (size_t i = 0; i < 3; ++i) {
    state.a_tilde.get(i, i) += 0.2;
  }
  gp::EnforceAlgebraicConstraints::apply(make_not_null(&state.conformal_metric),
                                         make_not_null(&state.a_tilde));
  CHECK_ITERABLE_APPROX(get(determinant(state.conformal_metric)),
                        DataVector(n, 1.0));
  CHECK(max(abs(trace_with_metric(state.conformal_metric, state.a_tilde))) <
        1e-12);
  const PhysicalState projected = state;
  gp::EnforceAlgebraicConstraints::apply(make_not_null(&state.conformal_metric),
                                         make_not_null(&state.a_tilde));
  CHECK_ITERABLE_APPROX(state.a_tilde, projected.a_tilde);
  CHECK_ITERABLE_APPROX(state.conformal_metric, projected.conformal_metric);

  tnsr::ii<DataVector, 3> degenerate{1_st, 0.0};
  tnsr::ii<DataVector, 3> a_tilde{1_st, 0.0};
  CHECK_THROWS_WITH(gp::EnforceAlgebraicConstraints::apply(
                        make_not_null(&degenerate), make_not_null(&a_tilde)),
                    Catch::Matchers::ContainsSubstring(
                        "determinant of the conformal spatial metric is "
                        "non-positive"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.GradientProjection.Filter",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  for (const double gain : {0.0, 0.1, 0.37, 1.0}) {
    test_filter_velocities(make_not_null(&gen), gain, false);
    test_filter_velocities(make_not_null(&gen), gain, true);
  }
  test_gain_option();
  test_enforce_algebraic_constraints(make_not_null(&gen));
}
