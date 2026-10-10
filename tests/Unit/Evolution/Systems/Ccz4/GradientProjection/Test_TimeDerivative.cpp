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
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;
using helpers::to_data_vector;
constexpr const char* module = "GradientProjectionReference";
constexpr size_t n43 = gp::number_of_target_components;

// Pointwise chart data: values and (raw, possibly non-commuting) derivatives.
struct PointData {
  gp::Tags::ChartU::type u{1_st};
  gp::Tags::ChartW::type w{1_st};
  gp::Tags::ChartQ::type q{1_st};
  gp::DerivChartW::type d_w{1_st};
  gp::DerivChartQ::type d_q{1_st};
};

PointData random_point(const gsl::not_null<std::mt19937*> gen,
                       const double h_scale, const double shift_scale,
                       const double field_scale) {
  const auto state =
      helpers::random_state(gen, h_scale, shift_scale, field_scale);
  std::normal_distribution<double> normal(0.0, field_scale);
  PointData result{};
  gp::set_potentials_at(make_not_null(&result.u), 0, state.u);
  gp::set_velocities_at(make_not_null(&result.w), 0, state.w);
  gp::set_gradients_at(make_not_null(&result.q), 0, state.q);
  for (auto& component : result.d_w) {
    component = normal(*gen);
  }
  for (auto& component : result.d_q) {
    component = normal(*gen);
  }
  return result;
}

struct Targets {
  gp::Tags::ChartW::type dt_w{1_st};
  gp::Tags::ChartQ::type dt_q{1_st};
  gp::Tags::ChartU::type dt_u{1_st};
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
                     make_not_null(&result.dt_u), point.u, point.w, point.q,
                     point.d_w, point.d_q, gauge.kappa_1, gauge.kappa_2,
                     gauge.kappa_3, gauge.eta, gauge.k_0, f);
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
    const auto inputs =
        gp::kernel_inputs(point.u, point.w, point.q, point.d_w, point.d_q);
    const gp::PotentialVector u = gp::potentials_at(point.u, 0);
    const gp::VelocityVector w = gp::velocities_at(point.w, 0);
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
        da5[5 * i + c] = point.d_w.get(i, c)[0];
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
        CHECK(inputs.conformal_metric.get(i, j)[0] ==
              approx(metric_data[3 * i + j]).epsilon(tolerance));
        CHECK(inputs.a_tilde.get(i, j)[0] ==
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
    CHECK(get(inputs.lapse)[0] == lapse);
    CHECK(get(inputs.conformal_factor)[0] ==
          approx(std::exp(u[gp::PotentialIndex::log_conformal_factor])));
    for (size_t i = 0; i < 3; ++i) {
      CHECK(inputs.field_a.get(i)[0] ==
            approx(gsl::at(q, i)[gp::PotentialIndex::lapse] / lapse));
      CHECK(inputs.field_p.get(i)[0] ==
            gsl::at(q, i)[gp::PotentialIndex::log_conformal_factor]);
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
              approx(sym(gp::PotentialIndex::log_conformal_factor)));
        for (size_t k = 0; k < 3; ++k) {
          CHECK(inputs.d_field_b.get(i, j, k)[0] ==
                approx(sym(gp::PotentialIndex::shift + k)));
        }
        CHECK(inputs.d_gamma_hat.get(i, j)[0] ==
              point.d_w.get(i, gp::VelocityIndex::gamma_hat + j)[0]);
        CHECK(inputs.d_b.get(i, j)[0] ==
              point.d_w.get(i, gp::VelocityIndex::b + j)[0]);
      }
      CHECK(inputs.d_trace_extrinsic_curvature.get(i)[0] ==
            point.d_w.get(i, gp::VelocityIndex::trace_extrinsic_curvature)[0]);
      CHECK(inputs.d_theta.get(i)[0] ==
            point.d_w.get(i, gp::VelocityIndex::theta)[0]);
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
    const DataVector u = to_data_vector(gp::potentials_at(base.u, 0));
    std::array<std::array<double, n43 * n43>, 3> actual{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t column = 0; column < n43; ++column) {
        PointData perturbed = base;
        if (column < gp::number_of_velocities) {
          perturbed.d_w.get(i, column)[0] += 1.0;
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
      for (size_t k = 0; k < n43 * n43; ++k) {
        CAPTURE(i);
        CAPTURE(k / n43);
        CAPTURE(k % n43);
        CHECK(gsl::at(gsl::at(actual, i), k) ==
              approx(expected[k]).scale(scale).epsilon(1e-10));
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

// Spec section 5.1: damping terms retained in the linearized physical rows at
// unit Minkowski. Central differences remove quadratic terms. The b rate
// carries -eta b for every f.
void test_minkowski_sources(const gsl::not_null<std::mt19937*> gen,
                            const double f) {
  for (const auto& [kappa_1, eta] : std::array<std::array<double, 2>, 4>{
           {{{0.0, 0.0}}, {{0.001, 0.0}}, {{0.37, 0.0}}, {{0.37, 0.8}}}}) {
    CAPTURE(kappa_1);
    CAPTURE(eta);
    Gauge gauge{};
    get(gauge.kappa_1) = kappa_1;
    get(gauge.eta) = eta;
    PointData minkowski{};
    for (auto& component : minkowski.u) {
      component = 0.0;
    }
    minkowski.u.get(gp::PotentialIndex::lapse) = 1.0;
    for (auto& component : minkowski.w) {
      component = 0.0;
    }
    for (auto& component : minkowski.q) {
      component = 0.0;
    }
    for (auto& component : minkowski.d_w) {
      component = 0.0;
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
      point.w.get(gp::VelocityIndex::theta) = sign * epsilon * theta;
      for (size_t i = 0; i < 3; ++i) {
        point.w.get(gp::VelocityIndex::gamma_hat + i) =
            sign * epsilon * gsl::at(gamma, i);
        point.w.get(gp::VelocityIndex::b + i) =
            sign * epsilon * gsl::at(gamma, i);
        for (size_t c = 0; c < 5; ++c) {
          point.q.get(i, c) = sign * epsilon * gsl::at(gsl::at(q_h, i), c);
        }
      }
      return targets(point, f, gauge);
    };
    const Targets plus = response(1.0);
    const Targets minus = response(-1.0);
    const auto derivative = [&plus, &minus, epsilon](const size_t component) {
      return (plus.dt_w.get(component)[0] - minus.dt_w.get(component)[0]) /
             (2.0 * epsilon);
    };
    CHECK(derivative(gp::VelocityIndex::trace_extrinsic_curvature) ==
          approx(-3.0 * kappa_1 * theta).epsilon(1e-8));
    CHECK(derivative(gp::VelocityIndex::theta) ==
          approx(-2.0 * kappa_1 * theta).epsilon(1e-8));
    for (size_t i = 0; i < 3; ++i) {
      double gamma_tilde = 0.0;
      for (size_t j = 0; j < 3; ++j) {
        gamma_tilde += 2.0 * gp::stf_decode(gsl::at(q_h, j).data())(i, j);
      }
      const double expected = -kappa_1 * (gsl::at(gamma, i) - gamma_tilde);
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
        gp::kernel_inputs(point.u, point.w, point.q, point.d_w, point.d_q),
        gauge.kappa_1, gauge.kappa_2, gauge.kappa_3, gauge.eta, gauge.k_0, f);
    const gp::PotentialVector u = gp::potentials_at(point.u, 0);
    const gp::VelocityVector w = gp::velocities_at(point.w, 0);
    const gp::GradientVector q = gp::gradients_at(point.q, 0);
    const DataVector rhs =
        pypp::call<DataVector>(module, "original_rhs", to_data_vector(u),
                               to_data_vector(w), to_data_vector(q), f);
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      CHECK(result.dt_u.get(a)[0] == approx(rhs[a]).epsilon(1e-12));
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
      CHECK(result.dt_w.get(gp::VelocityIndex::gamma_hat + k)[0] ==
            approx(rates.gamma_hat.get(k)[0] + ordering));
      CHECK(result.dt_w.get(gp::VelocityIndex::b + k)[0] ==
            approx(rates.b.get(k)[0] + ordering));
    }

    // q_{i,t} = R_u q_i + R(u, D_i w, D_i q).
    const DataVector jacobian = pypp::call<DataVector>(
        module, "rhs_potential_jacobian", to_data_vector(u), to_data_vector(w),
        to_data_vector(q), f);
    for (size_t i = 0; i < 3; ++i) {
      gp::VelocityVector d_i_w{};
      gp::GradientVector d_i_q{};
      for (size_t a = 0; a < gp::number_of_velocities; ++a) {
        gsl::at(d_i_w, a) = point.d_w.get(i, a)[0];
      }
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
  for (const double f : {1.0, 0.5, 1.25}) {
    CAPTURE(f);
    test_principal_part(make_not_null(&gen), f);
    test_minkowski_sources(make_not_null(&gen), f);
    test_components(make_not_null(&gen), f);
  }
}
