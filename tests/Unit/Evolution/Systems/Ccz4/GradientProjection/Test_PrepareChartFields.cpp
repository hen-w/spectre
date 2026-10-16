// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;
using Vars = gp::ComputeVolumeTargets::variables_tag::type;
using DtVars = gp::ComputeVolumeTargets::dt_variables_tag::type;
using Targets = gp::Tags::RhsTargets::type;
using physical_tags = Ccz4::fd::System::original_evolved_variables_tags;

// The evolved variables (physical fields and stored gradients) of the chart
// data; the gradient slots are filled with `gradient_fill`.
Vars evolved_from_chart(const helpers::ChartData& chart,
                        const double gradient_fill) {
  const size_t n = get(chart.lapse).size();
  Vars vars{n};
  get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(vars) =
      chart.conformal_metric;
  get<Ccz4::Tags::ConformalFactor<DataVector>>(vars) = chart.conformal_factor;
  get<Ccz4::Tags::ATilde<DataVector, 3>>(vars) = chart.a_tilde;
  get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars) =
      chart.trace_extrinsic_curvature;
  get<Ccz4::Tags::Theta<DataVector>>(vars) = chart.theta;
  get<Ccz4::Tags::GammaHat<DataVector, 3>>(vars) = chart.gamma_hat;
  get<gr::Tags::Lapse<DataVector>>(vars) = chart.lapse;
  get<gr::Tags::Shift<DataVector, 3>>(vars) = chart.shift;
  get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(vars) = chart.b;
  get<gp::Tags::PotentialGradients>(vars) =
      gp::Tags::PotentialGradients::type{n, gradient_fill};
  return vars;
}

// The ordinary element derivative of a tensor (of any index type).
template <typename Tag>
auto derivative(const typename Tag::type& tensor, const Mesh<3>& mesh,
                const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                                      Frame::Inertial>& inverse_jacobian) {
  Variables<tmpl::list<Tag>> variables{mesh.number_of_grid_points()};
  get<Tag>(variables) = tensor;
  return get<::Tags::deriv<Tag, tmpl::size_t<3>, Frame::Inertial>>(
      partial_derivatives<tmpl::list<Tag>>(variables, mesh, inverse_jacobian));
}

template <typename TagsList>
bool all_zero(const Variables<TagsList>& vars) {
  for (size_t i = 0; i < vars.size(); ++i) {
    if (vars.data()[i] != 0.0) {
      return false;
    }
  }
  return true;
}

template <typename Tags>
Variables<Tags> part(const Vars& vars) {
  Variables<Tags> result{vars.number_of_grid_points()};
  tmpl::for_each<Tags>(
      [&result, &vars]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        get<Tag>(result) = get<Tag>(vars);
      });
  return result;
}

// The local H and a recovered from on-constraint physical fields, and the
// loud rejection of off-constraint data (the encoding would silently drop
// the traces).
void test_local_chart_fields(const gsl::not_null<std::mt19937*> gen) {
  const size_t n = 7;
  helpers::ChartData chart{n};
  for (size_t s = 0; s < n; ++s) {
    const auto state = helpers::random_state(gen, 0.4, 0.3, 0.3);
    chart.set(s, state.u, state.w);
  }
  gp::LocalChartFields local{n};
  gp::local_chart_fields(make_not_null(&local), chart.conformal_metric,
                         chart.conformal_factor, chart.a_tilde, chart.lapse);
  const auto custom = Approx::custom().epsilon(1e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(get<gp::Tags::LogConformalMetric>(local),
                               chart.log_conformal_metric, custom);
  CHECK_ITERABLE_CUSTOM_APPROX(get<gp::Tags::TransformedATilde>(local), chart.a,
                               custom);

  // The charts of the log metric at every point, exactly as built pointwise.
  const gp::PotentialFields u = chart.potentials();
  const std::vector<gp::LogMetricChart> charts = gp::log_metric_charts(u);
  REQUIRE(charts.size() == n);
  for (size_t s = 0; s < n; ++s) {
    const gp::PotentialVector u_s = gp::potentials_at(u, s);
    CHECK(charts[s] == gp::LogMetricChart{gp::stf_decode(
                           &u_s[gp::PotentialIndex::log_metric])});
  }

  tnsr::ii<DataVector, 3> off_constraint = chart.conformal_metric;
  for (auto& component : off_constraint) {
    component *= 1.0 + 1.0e-6;
  }
  CHECK_THROWS_WITH(gp::local_chart_fields(
                        make_not_null(&local), off_constraint,
                        chart.conformal_factor, chart.a_tilde, chart.lapse),
                    Catch::Matchers::ContainsSubstring(
                        "off the algebraic-constraint surface"));
  tnsr::ii<DataVector, 3> traceful = chart.a_tilde;
  for (size_t i = 0; i < 3; ++i) {
    traceful.get(i, i) += 1.0e-6 * chart.conformal_metric.get(i, i);
  }
  CHECK_THROWS_WITH(
      gp::local_chart_fields(make_not_null(&local), chart.conformal_metric,
                             chart.conformal_factor, traceful, chart.lapse),
      Catch::Matchers::ContainsSubstring(
          "off the algebraic-constraint surface"));
  Scalar<DataVector> negative_phi = chart.conformal_factor;
  get(negative_phi)[3] = -0.1;
  CHECK_THROWS_WITH(
      gp::local_chart_fields(make_not_null(&local), chart.conformal_metric,
                             negative_phi, chart.a_tilde, chart.lapse),
      Catch::Matchers::ContainsSubstring(
          "The conformal factor must be positive"));
}

// The volume mutator on chart potentials that are quadratic polynomials in
// the logical coordinates of an affine element (with random velocities):
// the stored gradients are overwritten by the exact gradients, the physical
// fields are not modified, the poisoned targets are replaced by the volume
// targets, all time derivatives are zeroed, and repeated evaluation is
// deterministic.
void test_compute_volume_targets(const gsl::not_null<std::mt19937*> gen) {
  const auto geometry = helpers::make_geometry({{4, 4, 4}}, false);
  const size_t n = geometry.mesh.number_of_grid_points();
  std::normal_distribution<double> normal(0.0, 0.1);
  // u_A = c_A + sum_i l_{A,i} xi^i + sum_{i<=j} m_{A,ij} xi^i xi^j
  std::array<double, gp::number_of_potentials> constant{};
  std::array<std::array<double, 3>, gp::number_of_potentials> linear{};
  std::array<std::array<double, 6>, gp::number_of_potentials> quadratic{};
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    gsl::at(constant, a) = normal(*gen);
    for (auto& value : gsl::at(linear, a)) {
      value = normal(*gen);
    }
    for (auto& value : gsl::at(quadratic, a)) {
      value = normal(*gen);
    }
  }
  // Smaller variations of the lapse and conformal factor keep them positive.
  for (const size_t a :
       {gp::PotentialIndex::lapse, gp::PotentialIndex::conformal_factor}) {
    for (auto& value : gsl::at(linear, a)) {
      value *= 0.3;
    }
    for (auto& value : gsl::at(quadratic, a)) {
      value *= 0.3;
    }
  }
  constant[gp::PotentialIndex::lapse] = 1.0;
  constant[gp::PotentialIndex::conformal_factor] = 0.9;
  const std::array<std::array<size_t, 2>, 6> pairs{
      {{{0, 0}}, {{0, 1}}, {{0, 2}}, {{1, 1}}, {{1, 2}}, {{2, 2}}}};
  const auto& xi = geometry.logical;
  gp::Tags::PotentialRateTarget::type expected_u{n};
  gp::Tags::PotentialGradients::type expected_q{n, 0.0};
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    DataVector value(n, gsl::at(constant, a));
    for (size_t i = 0; i < 3; ++i) {
      value += gsl::at(gsl::at(linear, a), i) * xi.get(i);
      // d/dx^i = (2 / width_i) d/dxi^i on the affine element
      expected_q.get(i, a) +=
          gsl::at(gsl::at(linear, a), i) * geometry.inverse_jacobian.get(i, i);
    }
    for (size_t p = 0; p < pairs.size(); ++p) {
      const auto [i, j] = gsl::at(pairs, p);
      const double c = gsl::at(gsl::at(quadratic, a), p);
      value += c * xi.get(i) * xi.get(j);
      expected_q.get(i, a) +=
          c * xi.get(j) * geometry.inverse_jacobian.get(i, i);
      expected_q.get(j, a) +=
          c * xi.get(i) * geometry.inverse_jacobian.get(j, j);
    }
    expected_u.get(a) = value;
  }
  REQUIRE(min(expected_u.get(gp::PotentialIndex::conformal_factor)) > 0.2);
  REQUIRE(min(expected_u.get(gp::PotentialIndex::lapse)) > 0.2);
  helpers::ChartData chart{n};
  for (size_t s = 0; s < n; ++s) {
    gp::VelocityVector w{};
    for (auto& value : w) {
      value = normal(*gen);
    }
    chart.set(s, gp::potentials_at(expected_u, s), w);
  }
  const Vars initial = evolved_from_chart(chart, 7.0e3);
  Vars vars = initial;
  DtVars dt_vars{n, -3.0e3};
  Targets targets{n, 5.0e3};
  const Scalar<DataVector> kappa_1{n, 0.3};
  const Scalar<DataVector> kappa_2{n, 0.0};
  const double kappa_3 = 0.5;
  const Scalar<DataVector> eta{n, 0.4};
  const Scalar<DataVector> k_0{n, 0.0};
  const auto apply = [&]() {
    gp::ComputeVolumeTargets::apply(
        make_not_null(&vars), make_not_null(&dt_vars), make_not_null(&targets),
        geometry.mesh, geometry.inverse_jacobian, kappa_1, kappa_2, kappa_3,
        eta, k_0);
  };
  apply();

  const auto custom = Approx::custom().epsilon(1e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(get<gp::Tags::PotentialGradients>(vars),
                               expected_q, custom);
  CHECK(part<physical_tags>(vars) == part<physical_tags>(initial));
  CHECK(all_zero(dt_vars));

  // The targets from independently formed derivatives of the local chart
  // fields and of the physical fields.
  gp::LocalChartFields local{n};
  gp::local_chart_fields(make_not_null(&local), chart.conformal_metric,
                         chart.conformal_factor, chart.a_tilde, chart.lapse);
  const auto& mesh = geometry.mesh;
  const auto& inverse_jacobian = geometry.inverse_jacobian;
  const auto d_a = derivative<gp::Tags::TransformedATilde>(
      get<gp::Tags::TransformedATilde>(local), mesh, inverse_jacobian);
  const auto d_k = derivative<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
      chart.trace_extrinsic_curvature, mesh, inverse_jacobian);
  const auto d_theta = derivative<Ccz4::Tags::Theta<DataVector>>(
      chart.theta, mesh, inverse_jacobian);
  const auto d_gamma_hat = derivative<Ccz4::Tags::GammaHat<DataVector, 3>>(
      chart.gamma_hat, mesh, inverse_jacobian);
  const auto d_b = derivative<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(
      chart.b, mesh, inverse_jacobian);
  const auto d_q = derivative<gp::Tags::PotentialGradients>(expected_q, mesh,
                                                            inverse_jacobian);
  Targets expected{n};
  gp::volume_targets(
      make_not_null(&get<gp::Tags::VelocityTarget>(expected)),
      make_not_null(&get<gp::Tags::GradientTarget>(expected)),
      make_not_null(&get<gp::Tags::PotentialRateTarget>(expected)),
      chart.conformal_metric, chart.a_tilde,
      gp::PotentialFields{get<gp::Tags::LogConformalMetric>(local),
                          chart.conformal_factor, chart.lapse, chart.shift},
      gp::VelocityFields{get<gp::Tags::TransformedATilde>(local),
                         chart.trace_extrinsic_curvature, chart.theta,
                         chart.gamma_hat, chart.b},
      expected_q, gp::VelocityDerivatives{d_a, d_k, d_theta, d_gamma_hat, d_b},
      d_q, kappa_1, kappa_2, kappa_3, eta, k_0, Ccz4::fd::System::f);
  CHECK_VARIABLES_CUSTOM_APPROX(targets, expected,
                                Approx::custom().epsilon(1e-11).scale(1.0));
  // The potential slot is the original right-hand side R of the reference.
  for (size_t s = 0; s < n; s += 7) {
    const DataVector rhs = pypp::call<DataVector>(
        "GradientProjectionReference", "original_rhs",
        helpers::to_data_vector(gp::potentials_at(expected_u, s)),
        helpers::to_data_vector(gp::velocities_at(chart.velocities(), s)),
        helpers::to_data_vector(gp::gradients_at(expected_q, s)),
        Ccz4::fd::System::f);
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      CHECK(get<gp::Tags::PotentialRateTarget>(targets).get(a)[s] ==
            approx(rhs[a]).epsilon(1e-11).scale(1.0));
    }
  }

  // Repeated evaluation from poisoned scratch reproduces the result bitwise.
  const Vars first_vars = vars;
  const Targets first_targets = targets;
  get<gp::Tags::PotentialGradients>(vars) =
      gp::Tags::PotentialGradients::type{n, -9.0e3};
  targets = Targets{n, 4.0e3};
  dt_vars = DtVars{n, 2.0e3};
  apply();
  CHECK(vars == first_vars);
  CHECK(targets == first_targets);
  CHECK(all_zero(dt_vars));

  // The scheme supports only K0 = 0 (R_alpha omits it); anything else is an
  // error before any output is touched.
  Scalar<DataVector> nonzero_k_0{n, 0.0};
  get(nonzero_k_0)[n / 2] = 1.0e-3;
  targets = Targets{n, 4.0e3};
  const Targets targets_before = targets;
  CHECK_THROWS_WITH(
      gp::ComputeVolumeTargets::apply(
          make_not_null(&vars), make_not_null(&dt_vars),
          make_not_null(&targets), geometry.mesh, geometry.inverse_jacobian,
          kappa_1, kappa_2, kappa_3, eta, nonzero_k_0),
      Catch::Matchers::ContainsSubstring(
          "The CCZ4 gradient-projection scheme requires K0 = 0"));
  CHECK(vars == first_vars);
  CHECK(targets == targets_before);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection.PrepareChartFields",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Evolution/Systems/Ccz4/GradientProjection/");
  MAKE_GENERATOR(gen);
  test_local_chart_fields(make_not_null(&gen));
  test_compute_volume_targets(make_not_null(&gen));
}
