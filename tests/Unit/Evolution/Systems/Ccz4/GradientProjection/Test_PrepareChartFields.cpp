// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;

struct Physical {
  tnsr::ii<DataVector, 3> g;
  Scalar<DataVector> phi;
  tnsr::ii<DataVector, 3> a_tilde;
  Scalar<DataVector> k;
  Scalar<DataVector> theta;
  tnsr::I<DataVector, 3> gamma_hat;
  Scalar<DataVector> lapse;
  tnsr::I<DataVector, 3> shift;
  tnsr::I<DataVector, 3> b;
};

void apply(const gsl::not_null<gp::Tags::ChartU::type*> u,
           const gsl::not_null<gp::Tags::ChartW::type*> w,
           const gsl::not_null<gp::Tags::ChartQ::type*> q,
           const Physical& physical, const helpers::Geometry& geometry) {
  gp::PrepareChartFields::apply(
      u, w, q, physical.g, physical.phi, physical.a_tilde, physical.k,
      physical.theta, physical.gamma_hat, physical.lapse, physical.shift,
      physical.b, geometry.mesh, geometry.inverse_jacobian);
}

// Chart fields that are quadratic polynomials in the logical coordinates on an
// affine element, mapped to the physical state; the reconstruction recovers
// them and their exact gradients, overwriting whatever the scratch fields held.
void test_reconstruction(const gsl::not_null<std::mt19937*> gen) {
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
  constant[gp::PotentialIndex::lapse] = 1.0;
  const std::array<std::array<size_t, 2>, 6> pairs{
      {{{0, 0}}, {{0, 1}}, {{0, 2}}, {{1, 1}}, {{1, 2}}, {{2, 2}}}};
  const auto& xi = geometry.logical;
  gp::Tags::ChartU::type expected_u{n};
  gp::Tags::ChartQ::type expected_q{n, 0.0};
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
  gp::Tags::ChartW::type expected_w{n};
  for (auto& component : expected_w) {
    for (auto& value : component) {
      value = normal(*gen);
    }
  }
  Physical physical{tnsr::ii<DataVector, 3>{n}, Scalar<DataVector>{n},
                    tnsr::ii<DataVector, 3>{n}, Scalar<DataVector>{n},
                    Scalar<DataVector>{n},      tnsr::I<DataVector, 3>{n},
                    Scalar<DataVector>{n},      tnsr::I<DataVector, 3>{n},
                    tnsr::I<DataVector, 3>{n}};
  for (size_t s = 0; s < n; ++s) {
    const auto u = gp::potentials_at(expected_u, s);
    const auto w = gp::velocities_at(expected_w, s);
    const gp::LogMetricChart chart{
        gp::stf_decode(&u[gp::PotentialIndex::log_metric])};
    const gp::Matrix3 g = chart.metric();
    const gp::Matrix3 a_tilde =
        0.5 * chart.tangent(gp::stf_decode(&w[gp::VelocityIndex::a]));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        physical.g.get(i, j)[s] = g(i, j);
        physical.a_tilde.get(i, j)[s] = a_tilde(i, j);
      }
      physical.shift.get(i)[s] = gsl::at(u, gp::PotentialIndex::shift + i);
      physical.gamma_hat.get(i)[s] =
          gsl::at(w, gp::VelocityIndex::gamma_hat + i);
      physical.b.get(i)[s] = gsl::at(w, gp::VelocityIndex::b + i);
    }
    get(physical.phi)[s] =
        std::exp(u[gp::PotentialIndex::log_conformal_factor]);
    get(physical.lapse)[s] = u[gp::PotentialIndex::lapse];
    get(physical.k)[s] = w[gp::VelocityIndex::trace_extrinsic_curvature];
    get(physical.theta)[s] = w[gp::VelocityIndex::theta];
  }

  gp::Tags::ChartU::type u{n, 1.0e3};
  gp::Tags::ChartW::type w{n, -1.0e3};
  gp::Tags::ChartQ::type q{n, 7.0e3};
  apply(make_not_null(&u), make_not_null(&w), make_not_null(&q), physical,
        geometry);
  const auto custom = Approx::custom().epsilon(1e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(u, expected_u, custom);
  CHECK_ITERABLE_CUSTOM_APPROX(w, expected_w, custom);
  CHECK_ITERABLE_CUSTOM_APPROX(q, expected_q, custom);

  // Off-constraint data is never silently projected by the chart encoding.
  Physical off_constraint = physical;
  for (auto& component : off_constraint.g) {
    component *= 1.0 + 1.0e-6;
  }
  CHECK_THROWS_WITH(apply(make_not_null(&u), make_not_null(&w),
                          make_not_null(&q), off_constraint, geometry),
                    Catch::Matchers::ContainsSubstring(
                        "off the algebraic-constraint surface"));
  Physical traceful = physical;
  for (size_t i = 0; i < 3; ++i) {
    traceful.a_tilde.get(i, i) += 1.0e-6 * physical.g.get(i, i);
  }
  CHECK_THROWS_WITH(apply(make_not_null(&u), make_not_null(&w),
                          make_not_null(&q), traceful, geometry),
                    Catch::Matchers::ContainsSubstring(
                        "off the algebraic-constraint surface"));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection.PrepareChartFields",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  test_reconstruction(make_not_null(&gen));
}
