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
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;
using helpers::to_data_vector;
constexpr const char* module = "GradientProjectionReference";
constexpr size_t n43 = gp::number_of_target_components;

// Pointwise chart data: values (the physical fields consistent with H and a)
// and (raw, possibly non-commuting) derivatives.
struct PointData {
  helpers::ChartData chart{1_st};
  gp::Tags::PotentialGradients::type q{1_st};
  gp::DerivTransformedATilde::type d_a{1_st};
  tnsr::i<DataVector, 3> d_k{1_st};
  tnsr::i<DataVector, 3> d_theta{1_st};
  tnsr::iJ<DataVector, 3> d_gamma_hat{1_st};
  tnsr::iJ<DataVector, 3> d_b{1_st};
  gp::DerivPotentialGradients::type d_q{1_st};

  gp::VelocityDerivatives d_w() const {
    return {d_a, d_k, d_theta, d_gamma_hat, d_b};
  }
  // The component D_i w_c in the layout of VelocityIndex.
  double& d_w_component(const size_t i, const size_t c) {
    if (c < gp::VelocityIndex::trace_extrinsic_curvature) {
      return d_a.get(i, c)[0];
    } else if (c == gp::VelocityIndex::trace_extrinsic_curvature) {
      return d_k.get(i)[0];
    } else if (c == gp::VelocityIndex::theta) {
      return d_theta.get(i)[0];
    } else if (c < gp::VelocityIndex::b) {
      return d_gamma_hat.get(i, c - gp::VelocityIndex::gamma_hat)[0];
    }
    return d_b.get(i, c - gp::VelocityIndex::b)[0];
  }
};

PointData random_point(const gsl::not_null<std::mt19937*> gen,
                       const double h_scale, const double shift_scale,
                       const double field_scale) {
  const auto state =
      helpers::random_state(gen, h_scale, shift_scale, field_scale);
  std::normal_distribution<double> normal(0.0, field_scale);
  PointData result{};
  result.chart.set(0, state.u, state.w);
  gp::set_gradients_at(make_not_null(&result.q), 0, state.q);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t c = 0; c < gp::number_of_velocities; ++c) {
      result.d_w_component(i, c) = normal(*gen);
    }
  }
  for (auto& component : result.d_q) {
    component = normal(*gen);
  }
  return result;
}

struct Targets {
  gp::Tags::VelocityTarget::type dt_w{1_st};
  gp::Tags::GradientTarget::type dt_q{1_st};
  gp::Tags::PotentialRateTarget::type dt_u{1_st};
};

struct Gauge {
  Scalar<DataVector> kappa_1{DataVector{0.37}};
  Scalar<DataVector> kappa_2{DataVector{0.0}};
  double kappa_3 = 1.0;
  Scalar<DataVector> eta{DataVector{0.0}};
  Scalar<DataVector> k_0{DataVector{0.0}};
};

Targets targets(const PointData& point, const double f,
                const Gauge& gauge = {}) {
  Targets result{};
  gp::volume_targets(make_not_null(&result.dt_w), make_not_null(&result.dt_q),
                     make_not_null(&result.dt_u), point.chart.conformal_metric,
                     point.chart.a_tilde, point.chart.potentials(),
                     point.chart.velocities(), point.q, point.d_w(), point.d_q,
                     gauge.kappa_1, gauge.kappa_2, gauge.kappa_3, gauge.eta,
                     gauge.k_0, f);
  return result;
}

std::array<double, n43> flatten_targets(const Targets& t) {
  std::array<double, n43> result{};
  for (size_t a = 0; a < gp::number_of_velocities; ++a) {
    gsl::at(result, a) = t.dt_w.get(a)[0];
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      gsl::at(result, 13 + 10 * i + a) = t.dt_q.get(i, a)[0];
    }
  }
  return result;
}

// The chain-rule inputs to the production kernel against SciPy matrix
// exponentials and their Frechet derivatives, for exact (commuting) and
// non-commuting second derivatives.
void test_kernel_inputs(const gsl::not_null<std::mt19937*> gen) {
  for (size_t sample = 0; sample < 6; ++sample) {
    PointData point = random_point(gen, 0.4, 0.3, 0.3);
    const bool symmetric_second_derivatives = sample % 2 == 0;
    if (symmetric_second_derivatives) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < i; ++j) {
          for (size_t a = 0; a < gp::number_of_potentials; ++a) {
            point.d_q.get(i, j, a) = point.d_q.get(j, i, a);
          }
        }
      }
    }
    const auto inputs = gp::kernel_inputs(
        point.chart.potentials(), point.chart.a, point.q, point.d_a, point.d_q);
    const gp::PotentialVector u =
        gp::potentials_at(point.chart.potentials(), 0);
    const gp::VelocityVector w = gp::velocities_at(point.chart.velocities(), 0);
    const gp::GradientVector q = gp::gradients_at(point.q, 0);
    DataVector h5{5};
    DataVector dh5{15};
    DataVector ddh5{45};
    DataVector a5{5};
    DataVector da5{15};
    for (size_t c = 0; c < 5; ++c) {
      h5[c] = u[c];
      a5[c] = w[c];
      for (size_t i = 0; i < 3; ++i) {
        dh5[5 * i + c] = gsl::at(q, i)[c];
        da5[5 * i + c] = point.d_a.get(i, c)[0];
        for (size_t j = 0; j < 3; ++j) {
          // The kernel uses the symmetrized second derivatives.
          ddh5[5 * (3 * i + j) + c] =
              0.5 * (point.d_q.get(i, j, c)[0] + point.d_q.get(j, i, c)[0]);
        }
      }
    }
    const DataVector metric_data =
        pypp::call<DataVector>(module, "metric_and_derivatives", h5, dh5, ddh5);
    const DataVector a_tilde_data = pypp::call<DataVector>(
        module, "a_tilde_and_derivative", h5, dh5, a5, da5);
    const double tolerance = 1e-12;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        // The physical fields passed to the kernel are e^{2H} and
        // T_H a / 2.
        CHECK(point.chart.conformal_metric.get(i, j)[0] ==
              approx(metric_data[3 * i + j]).epsilon(tolerance));
        CHECK(point.chart.a_tilde.get(i, j)[0] ==
              approx(a_tilde_data[3 * i + j]).epsilon(tolerance));
        for (size_t k = 0; k < 3; ++k) {
          CHECK(inputs.field_d.get(i, j, k)[0] ==
                approx(0.5 * metric_data[9 + 9 * i + 3 * j + k])
                    .epsilon(tolerance));
          CHECK(inputs.d_a_tilde.get(i, j, k)[0] ==
                approx(a_tilde_data[9 + 9 * i + 3 * j + k]).epsilon(tolerance));
          for (size_t l = 0; l < 3; ++l) {
            CHECK(inputs.d_field_d.get(i, j, k, l)[0] ==
                  approx(0.5 * metric_data[36 + 27 * i + 9 * j + 3 * k + l])
                      .epsilon(tolerance));
          }
        }
      }
    }
    const double lapse = u[gp::PotentialIndex::lapse];
    const double phi = u[gp::PotentialIndex::conformal_factor];
    for (size_t i = 0; i < 3; ++i) {
      CHECK(inputs.field_a.get(i)[0] ==
            approx(gsl::at(q, i)[gp::PotentialIndex::lapse] / lapse));
      CHECK(inputs.field_p.get(i)[0] ==
            approx(gsl::at(q, i)[gp::PotentialIndex::conformal_factor] / phi));
      for (size_t j = 0; j < 3; ++j) {
        const auto sym = [&point, i, j](const size_t component) {
          return 0.5 * (point.d_q.get(i, j, component)[0] +
                        point.d_q.get(j, i, component)[0]);
        };
        CHECK(inputs.field_b.get(i, j)[0] ==
              gsl::at(gsl::at(q, i), gp::PotentialIndex::shift + j));
        CHECK(inputs.d_field_a.get(i, j)[0] ==
              approx(sym(gp::PotentialIndex::lapse) / lapse -
                     gsl::at(q, i)[gp::PotentialIndex::lapse] *
                         gsl::at(q, j)[gp::PotentialIndex::lapse] /
                         square(lapse)));
        CHECK(inputs.d_field_p.get(i, j)[0] ==
              approx(sym(gp::PotentialIndex::conformal_factor) / phi -
                     gsl::at(q, i)[gp::PotentialIndex::conformal_factor] *
                         gsl::at(q, j)[gp::PotentialIndex::conformal_factor] /
                         square(phi)));
        for (size_t k = 0; k < 3; ++k) {
          CHECK(inputs.d_field_b.get(i, j, k)[0] ==
                approx(sym(gp::PotentialIndex::shift + k)));
        }
      }
    }
    // FieldB references the stored shift gradient.
    for (size_t i = 0; i < 3; ++i) {
      for (size_t k = 0; k < 3; ++k) {
        CHECK(inputs.field_b.get(i, k).data() ==
              point.q.get(i, gp::PotentialIndex::shift + k).data());
      }
    }
  }
}

// FieldP and its symmetrized derivative for a smooth spatially varying
// conformal factor, a quadratic polynomial (differentiated exactly by the
// element derivative), against the analytic d_i log(phi) and
// d_i d_j log(phi).
void test_variable_phi_inputs() {
  const auto geometry = helpers::make_geometry({{4, 4, 4}}, false);
  const size_t n = geometry.mesh.number_of_grid_points();
  const auto& xi = geometry.logical;
  // phi = 1 + c_i xi^i + m_ij xi^i xi^j in the logical coordinates.
  const std::array<double, 3> linear{{0.2, -0.15, 0.1}};
  const std::array<std::array<double, 3>, 3> quadratic{
      {{{0.05, 0.02, -0.01}}, {{0.02, -0.04, 0.03}}, {{-0.01, 0.03, 0.06}}}};
  Scalar<DataVector> phi{DataVector(n, 1.0)};
  // Analytic derivatives with respect to the inertial coordinates of the
  // affine map, d xi^a / d x^i = inverse_jacobian(a, i) (diagonal).
  tnsr::i<DataVector, 3> d_phi{n, 0.0};
  tnsr::ii<DataVector, 3> dd_phi{n, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    get(phi) += gsl::at(linear, i) * xi.get(i);
    d_phi.get(i) += gsl::at(linear, i) * geometry.inverse_jacobian.get(i, i);
    for (size_t j = 0; j < 3; ++j) {
      get(phi) += gsl::at(gsl::at(quadratic, i), j) * xi.get(i) * xi.get(j);
      d_phi.get(i) += 2.0 * gsl::at(gsl::at(quadratic, i), j) * xi.get(j) *
                      geometry.inverse_jacobian.get(i, i);
      if (j >= i) {
        dd_phi.get(i, j) = 2.0 * gsl::at(gsl::at(quadratic, i), j) *
                           geometry.inverse_jacobian.get(i, i) *
                           geometry.inverse_jacobian.get(j, j);
      }
    }
  }
  REQUIRE(min(get(phi)) > 0.4);
  helpers::ChartData chart{n};
  for (size_t s = 0; s < n; ++s) {
    gp::PotentialVector u{};
    u[gp::PotentialIndex::conformal_factor] = get(phi)[s];
    u[gp::PotentialIndex::lapse] = 1.0;
    chart.set(s, u, gp::VelocityVector{});
  }
  gp::Tags::PotentialGradients::type q{n, 0.0};
  const auto element_d_phi =
      partial_derivative(phi, geometry.mesh, geometry.inverse_jacobian);
  for (size_t i = 0; i < 3; ++i) {
    q.get(i, gp::PotentialIndex::conformal_factor) = element_d_phi.get(i);
  }
  gp::DerivPotentialGradients::type d_q{n, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    const auto dd = partial_derivative(
        Scalar<DataVector>{q.get(i, gp::PotentialIndex::conformal_factor)},
        geometry.mesh, geometry.inverse_jacobian);
    for (size_t k = 0; k < 3; ++k) {
      d_q.get(k, i, gp::PotentialIndex::conformal_factor) = dd.get(k);
    }
  }
  const gp::DerivTransformedATilde::type d_a{n, 0.0};
  const auto inputs =
      gp::kernel_inputs(chart.potentials(), chart.a, q, d_a, d_q);
  const auto custom = Approx::custom().epsilon(1e-12).scale(1.0);
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_CUSTOM_APPROX(inputs.field_p.get(i),
                                 DataVector{d_phi.get(i) / get(phi)}, custom);
    for (size_t j = i; j < 3; ++j) {
      CHECK_ITERABLE_CUSTOM_APPROX(
          inputs.d_field_p.get(i, j),
          DataVector{dd_phi.get(i, j) / get(phi) -
                     d_phi.get(i) * d_phi.get(j) / square(get(phi))},
          custom);
    }
  }
}

// The volume targets are affine in the derivatives (D_i w, D_i q_j) with
// coefficients that depend only on u: their principal part must equal the
// independently assembled normal matrix A^i for each coordinate direction,
// including every tangential gradient coupling and the curl ordering.
void test_principal_part(const gsl::not_null<std::mt19937*> gen,
                         const double f) {
  for (size_t sample = 0; sample < 5; ++sample) {
    const PointData base = random_point(gen, sample == 0 ? 0.0 : 0.4,
                                        sample == 0 ? 0.0 : 0.3, 0.2);
    const auto base_targets = flatten_targets(targets(base, f));
    const gp::PotentialVector base_u =
        gp::potentials_at(base.chart.potentials(), 0);
    const DataVector u = to_data_vector(base_u);
    const double phi = base_u[gp::PotentialIndex::conformal_factor];
    gp::PotentialVector base_u_log = base_u;
    base_u_log[gp::PotentialIndex::conformal_factor] = std::log(phi);
    const DataVector u_log = to_data_vector(base_u_log);
    std::array<std::array<double, n43 * n43>, 3> actual{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t column = 0; column < n43; ++column) {
        PointData perturbed = base;
        if (column < gp::number_of_velocities) {
          perturbed.d_w_component(i, column) += 1.0;
        } else {
          const size_t k = column - gp::number_of_velocities;
          perturbed.d_q.get(i, k / 10, k % 10)[0] += 1.0;
        }
        const auto perturbed_targets = flatten_targets(targets(perturbed, f));
        for (size_t row = 0; row < n43; ++row) {
          gsl::at(actual, i)[row * n43 + column] =
              gsl::at(perturbed_targets, row) - gsl::at(base_targets, row);
        }
      }
    }
    for (size_t i = 0; i < 3; ++i) {
      std::array<double, 3> axis{};
      gsl::at(axis, i) = 1.0;
      const DataVector expected = pypp::call<DataVector>(
          module, "normal_matrix", u, DataVector{axis[0], axis[1], axis[2]}, f);
      double scale = 0.0;
      for (size_t k = 0; k < n43 * n43; ++k) {
        scale = std::max(scale, std::abs(expected[k]));
      }
      // The frozen log(phi) symbol conjugated by the scaling of the
      // gradients of phi.
      const DataVector expected_log =
          pypp::call<DataVector>(module, "normal_matrix_log", u_log,
                                 DataVector{axis[0], axis[1], axis[2]}, f);
      const auto phi_scale = [phi](const size_t index) {
        return index >= gp::number_of_velocities and
                       (index - gp::number_of_velocities) % 10 ==
                           gp::PotentialIndex::conformal_factor
                   ? phi
                   : 1.0;
      };
      for (size_t k = 0; k < n43 * n43; ++k) {
        CAPTURE(i);
        CAPTURE(k / n43);
        CAPTURE(k % n43);
        CHECK(gsl::at(gsl::at(actual, i), k) ==
              approx(expected[k]).scale(scale).epsilon(1e-10));
        CHECK(gsl::at(gsl::at(actual, i), k) ==
              approx(expected_log[k] * phi_scale(k / n43) / phi_scale(k % n43))
                  .scale(scale)
                  .epsilon(1e-10));
      }
    }
    // An oblique direction: sum_i n_i A^i = A_n.
    const auto normal = helpers::random_unit_vector(gen);
    const DataVector expected =
        pypp::call<DataVector>(module, "normal_matrix", u,
                               DataVector{normal[0], normal[1], normal[2]}, f);
    for (size_t k = 0; k < n43 * n43; k += 7) {
      const double combined = normal[0] * actual[0][k] +
                              normal[1] * actual[1][k] +
                              normal[2] * actual[2][k];
      CHECK(combined == approx(expected[k]).scale(1.0).epsilon(1e-10));
    }
  }
}

// Spec section 5.1: damping terms retained in the linearized physical rows
// about a flat state with constant conformal factor phi_0 and lapse alpha_0
// (each damping term carries one lapse factor). Central differences remove
// quadratic terms. The b rate carries -eta b for every f.
void test_minkowski_sources(const gsl::not_null<std::mt19937*> gen,
                            const double f, const double phi_0,
                            const double alpha_0) {
  for (const auto& [kappa_1, eta] : std::array<std::array<double, 2>, 4>{
           {{{0.0, 0.0}}, {{0.001, 0.0}}, {{0.37, 0.0}}, {{0.37, 0.8}}}}) {
    CAPTURE(kappa_1);
    CAPTURE(eta);
    Gauge gauge{};
    get(gauge.kappa_1) = kappa_1;
    get(gauge.eta) = eta;
    PointData minkowski{};
    gp::PotentialVector background{};
    background[gp::PotentialIndex::conformal_factor] = phi_0;
    background[gp::PotentialIndex::lapse] = alpha_0;
    minkowski.chart.set(0, background, gp::VelocityVector{});
    for (auto& component : minkowski.q) {
      component = 0.0;
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t c = 0; c < gp::number_of_velocities; ++c) {
        minkowski.d_w_component(i, c) = 0.0;
      }
    }
    for (auto& component : minkowski.d_q) {
      component = 0.0;
    }
    std::normal_distribution<double> normal(0.0, 1.0);
    const double theta = normal(*gen);
    std::array<double, 3> gamma{{normal(*gen), normal(*gen), normal(*gen)}};
    gp::GradientVector q_h{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t c = 0; c < 5; ++c) {
        gsl::at(gsl::at(q_h, i), c) = normal(*gen);
      }
    }
    const double epsilon = 1.0e-5;
    const auto response = [&](const double sign) {
      PointData point = minkowski;
      gp::VelocityVector w{};
      w[gp::VelocityIndex::theta] = sign * epsilon * theta;
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(w, gp::VelocityIndex::gamma_hat + i) =
            sign * epsilon * gsl::at(gamma, i);
        gsl::at(w, gp::VelocityIndex::b + i) =
            sign * epsilon * gsl::at(gamma, i);
        for (size_t c = 0; c < 5; ++c) {
          point.q.get(i, c) = sign * epsilon * gsl::at(gsl::at(q_h, i), c);
        }
      }
      point.chart.set(0, background, w);
      return targets(point, f, gauge);
    };
    const Targets plus = response(1.0);
    const Targets minus = response(-1.0);
    const auto derivative = [&plus, &minus, epsilon](const size_t component) {
      return (plus.dt_w.get(component)[0] - minus.dt_w.get(component)[0]) /
             (2.0 * epsilon);
    };
    CHECK(derivative(gp::VelocityIndex::trace_extrinsic_curvature) ==
          approx(-3.0 * alpha_0 * kappa_1 * theta).epsilon(1e-8));
    CHECK(derivative(gp::VelocityIndex::theta) ==
          approx(-2.0 * alpha_0 * kappa_1 * theta).epsilon(1e-8));
    for (size_t i = 0; i < 3; ++i) {
      double gamma_tilde = 0.0;
      for (size_t j = 0; j < 3; ++j) {
        gamma_tilde += 2.0 * gp::stf_decode(gsl::at(q_h, j).data())(i, j);
      }
      const double expected =
          -alpha_0 * kappa_1 * (gsl::at(gamma, i) - gamma_tilde);
      CHECK(derivative(gp::VelocityIndex::gamma_hat + i) ==
            approx(expected).epsilon(1e-8));
      CHECK(derivative(gp::VelocityIndex::b + i) ==
            approx(expected - eta * gsl::at(gamma, i)).epsilon(1e-8));
    }
  }
}

// The remaining components at curved states: chart rate of a against an
// independent complex-step derivative of (1/2) D(exp(2H))[a], the production
// K, Theta, GammaHat, b rates plus the curl ordering, R and the gradient
// targets against the model.
void test_components(const gsl::not_null<std::mt19937*> gen, const double f) {
  const Gauge gauge{};
  for (size_t sample = 0; sample < 6; ++sample) {
    const PointData point = random_point(gen, 0.4, 0.3, 0.2);
    const Targets result = targets(point, f, gauge);
    const auto rates = gp::kernel_rates(
        gp::kernel_inputs(point.chart.potentials(), point.chart.a, point.q,
                          point.d_a, point.d_q),
        point.chart.conformal_metric, point.chart.a_tilde,
        point.chart.potentials(), point.chart.velocities(), point.d_w(),
        gauge.kappa_1, gauge.kappa_2, gauge.kappa_3, gauge.eta, gauge.k_0, f);
    const gp::PotentialVector u =
        gp::potentials_at(point.chart.potentials(), 0);
    const gp::VelocityVector w = gp::velocities_at(point.chart.velocities(), 0);
    const gp::GradientVector q = gp::gradients_at(point.q, 0);
    const DataVector rhs =
        pypp::call<DataVector>(module, "original_rhs", to_data_vector(u),
                               to_data_vector(w), to_data_vector(q), f);
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      CHECK(result.dt_u.get(a)[0] == approx(rhs[a]).epsilon(1e-12));
    }
    // The potential rates of the production kernel (with K_0 = 0) are R:
    // T_H^-1 g_t = R_H, phi_t = R_phi, alpha_t = R_alpha, beta_t = R_beta.
    {
      const gp::LogMetricChart chart{
          gp::stf_decode(&u[gp::PotentialIndex::log_metric])};
      gp::Matrix3 metric_rate{};
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          metric_rate(i, j) = rates.conformal_metric.get(i, j)[0];
        }
      }
      const std::array<double, 5> h_rate =
          gp::stf_encode(chart.inverse_tangent(metric_rate));
      for (size_t c = 0; c < 5; ++c) {
        CHECK(gsl::at(h_rate, c) ==
              approx(rhs[gp::PotentialIndex::log_metric + c])
                  .epsilon(1e-11)
                  .scale(1.0));
      }
      CHECK(get(rates.conformal_factor)[0] ==
            approx(rhs[gp::PotentialIndex::conformal_factor]).epsilon(1e-12));
      CHECK(get(rates.lapse)[0] ==
            approx(rhs[gp::PotentialIndex::lapse]).epsilon(1e-12));
      for (size_t i = 0; i < 3; ++i) {
        CHECK(rates.shift.get(i)[0] ==
              approx(rhs[gp::PotentialIndex::shift + i]).epsilon(1e-12));
      }
    }

    DataVector h5{5};
    DataVector rh5{5};
    DataVector a5{5};
    DataVector at5{5};
    for (size_t c = 0; c < 5; ++c) {
      h5[c] = u[c];
      rh5[c] = rhs[c];
      a5[c] = w[c];
      at5[c] = result.dt_w.get(c)[0];
    }
    const DataVector a_tilde_rate = pypp::call<DataVector>(
        module, "a_tilde_time_derivative", h5, rh5, a5, at5);
    double scale = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        scale = std::max(scale, std::abs(rates.a_tilde.get(i, j)[0]));
      }
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        CHECK(a_tilde_rate[3 * i + j] ==
              approx(rates.a_tilde.get(i, j)[0]).scale(scale).epsilon(1e-11));
      }
    }
    CHECK(result.dt_w.get(gp::VelocityIndex::trace_extrinsic_curvature)[0] ==
          get(rates.trace_extrinsic_curvature)[0]);
    CHECK(result.dt_w.get(gp::VelocityIndex::theta)[0] == get(rates.theta)[0]);
    const DataVector metric_flat = pypp::call<DataVector>(
        module, "metric", helpers::flatten(gp::stf_decode(h5.data())));
    const gp::Matrix3 inverse_metric =
        blaze::inv(helpers::unflatten(metric_flat));
    for (size_t k = 0; k < 3; ++k) {
      double ordering = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          ordering += inverse_metric(k, i) *
                      (point.d_q.get(i, j, gp::PotentialIndex::shift + j)[0] -
                       point.d_q.get(j, i, gp::PotentialIndex::shift + j)[0]) /
                      6.0;
        }
      }
      // The ordering term uses the SciPy inverse of the metric, which agrees
      // with the production eigendecomposition inverse to ~1e-13 relative on
      // the anisotropic metrics sampled here.
      CHECK(result.dt_w.get(gp::VelocityIndex::gamma_hat + k)[0] ==
            approx(rates.gamma_hat.get(k)[0] + ordering)
                .epsilon(1e-12)
                .scale(1.0));
      CHECK(result.dt_w.get(gp::VelocityIndex::b + k)[0] ==
            approx(rates.b.get(k)[0] + ordering).epsilon(1e-12).scale(1.0));
    }

    // q_{i,t} = R_u q_i + R(u, D_i w, D_i q).
    const DataVector jacobian = pypp::call<DataVector>(
        module, "rhs_potential_jacobian", to_data_vector(u), to_data_vector(w),
        to_data_vector(q), f);
    for (size_t i = 0; i < 3; ++i) {
      const gp::VelocityVector d_i_w =
          gp::velocity_derivatives_at(point.d_w(), i, 0);
      gp::GradientVector d_i_q{};
      for (size_t j = 0; j < 3; ++j) {
        for (size_t a = 0; a < gp::number_of_potentials; ++a) {
          gsl::at(gsl::at(d_i_q, j), a) = point.d_q.get(i, j, a)[0];
        }
      }
      const DataVector principal = pypp::call<DataVector>(
          module, "original_rhs", to_data_vector(u), to_data_vector(d_i_w),
          to_data_vector(d_i_q), f);
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        double lower_order = 0.0;
        for (size_t b = 0; b < gp::number_of_potentials; ++b) {
          lower_order += jacobian[10 * a + b] * gsl::at(gsl::at(q, i), b);
        }
        CHECK(result.dt_q.get(i, a)[0] ==
              approx(principal[a] + lower_order).epsilon(1e-11));
      }
    }
  }
}
}  // namespace

// [[TimeOut, 30]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection.TimeDerivative",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Evolution/Systems/Ccz4/GradientProjection/");
  MAKE_GENERATOR(gen);
  test_kernel_inputs(make_not_null(&gen));
  test_variable_phi_inputs();
  for (const double f : {1.0, 0.5, 1.25}) {
    CAPTURE(f);
    test_principal_part(make_not_null(&gen), f);
    test_minkowski_sources(make_not_null(&gen), f, 1.0, 1.0);
    test_minkowski_sources(make_not_null(&gen), f, 0.7, 0.6);
    test_components(make_not_null(&gen), f);
  }
}
