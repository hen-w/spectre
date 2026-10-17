// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Filter.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
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

void apply_filter(const gsl::not_null<PhysicalState*> state,
                  const helpers::Geometry& geometry,
                  const gp::LocalOperators& operators, const double gain,
                  const TimeStepId& time_step_id,
                  const TimeStepId& next_time_step_id) {
  gp::FilterAndEnforceConstraints::apply(
      make_not_null(&state->conformal_metric), make_not_null(&state->a_tilde),
      make_not_null(&state->trace_extrinsic_curvature),
      make_not_null(&state->theta), make_not_null(&state->gamma_hat),
      make_not_null(&state->b), geometry.mesh, operators, gain, time_step_id,
      next_time_step_id);
}

void test_filter_and_projection(const gsl::not_null<std::mt19937*> gen,
                                const bool unit_metric, const double gain) {
  CAPTURE(unit_metric);
  CAPTURE(gain);
  const auto geometry = helpers::make_geometry({{4, 4, 4}}, true);
  const gp::LocalOperators operators{geometry.mesh, geometry.inverse_jacobian,
                                     geometry.det_inverse_jacobian};
  const size_t n = geometry.mesh.number_of_grid_points();
  const PhysicalState original = random_state(gen, n, unit_metric);
  PhysicalState state = original;
  const Slab slab{0.0, 1.0};
  apply_filter(make_not_null(&state), geometry, operators, gain,
               TimeStepId{true, 0, slab.start()},
               TimeStepId{true, 0, slab.end()});

  const auto filtered = [&operators, &gain](DataVector field) {
    operators.apply_filter(make_not_null(&field), gain);
    return field;
  };
  // K, Theta, GammaHat and b are filtered componentwise with the given gain.
  CHECK_ITERABLE_APPROX(get(state.trace_extrinsic_curvature),
                        filtered(get(original.trace_extrinsic_curvature)));
  CHECK_ITERABLE_APPROX(get(state.theta), filtered(get(original.theta)));
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_APPROX(state.gamma_hat.get(i),
                          filtered(original.gamma_hat.get(i)));
    CHECK_ITERABLE_APPROX(state.b.get(i), filtered(original.b.get(i)));
  }
  // The conformal metric is not filtered; with unit determinant the
  // projection leaves it unchanged.
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      CHECK_ITERABLE_APPROX(state.conformal_metric.get(i, j),
                            original.conformal_metric.get(i, j));
    }
  }
  // Each ATilde component is filtered, then the trace is projected out with
  // the metric. On a varying metric the componentwise filter alone breaks the
  // trace, unless the gain is one; at unit metric it preserves the trace-free
  // subspace.
  tnsr::ii<DataVector, 3> filtered_a_tilde{n};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      filtered_a_tilde.get(i, j) = filtered(original.a_tilde.get(i, j));
    }
  }
  const DataVector filtered_trace =
      trace_with_metric(original.conformal_metric, filtered_a_tilde);
  // Rounding in the projection (determinant normalization and the inverse of
  // a metric with condition number up to ~e^1.6) is ~1e-14.
  const auto custom = Approx::custom().epsilon(1e-12).scale(1.0);
  if (unit_metric or gain == 1.0) {
    CHECK(max(abs(filtered_trace)) < 1e-12);
  } else {
    CHECK(max(abs(filtered_trace)) > 1e-4);
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      CHECK_ITERABLE_CUSTOM_APPROX(
          state.a_tilde.get(i, j),
          DataVector{filtered_a_tilde.get(i, j) -
                     filtered_trace * original.conformal_metric.get(i, j) /
                         3.0},
          custom);
    }
  }
  CHECK(max(abs(trace_with_metric(state.conformal_metric, state.a_tilde))) <
        1e-12);

  // Unused self-start substeps do not advance the state, so nothing is
  // filtered.
  PhysicalState unused = original;
  apply_filter(make_not_null(&unused), geometry, operators, gain,
               TimeStepId{true, -1, slab.start()},
               TimeStepId{true, 0, slab.start()});
  CHECK(get(unused.theta) == get(original.theta));
  CHECK(unused.a_tilde == original.a_tilde);
  CHECK(unused.gamma_hat == original.gamma_hat);
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
    test_filter_and_projection(make_not_null(&gen), false, gain);
    test_filter_and_projection(make_not_null(&gen), true, gain);
  }
  test_gain_option();
  test_enforce_algebraic_constraints(make_not_null(&gen));
}
