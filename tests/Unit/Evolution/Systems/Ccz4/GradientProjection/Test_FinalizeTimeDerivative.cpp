// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/FinalizeTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;
constexpr const char* module = "GradientProjectionReference";

struct ElementData {
  helpers::Geometry geometry;
  gp::LocalOperators operators;
  gp::Tags::ChartU::type u;
  gp::Tags::ChartW::type w;
  gp::Tags::ChartQ::type q;
  gp::Tags::ChartU::type dt_u;
  gp::Tags::ChartW::type dt_w;
  gp::Tags::ChartQ::type dt_q;
};

// Random admissible chart fields with q = D u, and random targets.
ElementData make_element(const gsl::not_null<std::mt19937*> gen,
                         const bool curved) {
  ElementData result{};
  result.geometry = helpers::make_geometry({{4, 4, 4}}, curved);
  result.operators =
      gp::LocalOperators{result.geometry.mesh, result.geometry.inverse_jacobian,
                         result.geometry.det_inverse_jacobian};
  const size_t n = result.geometry.mesh.number_of_grid_points();
  result.u = gp::Tags::ChartU::type{n};
  result.w = gp::Tags::ChartW::type{n};
  result.q = gp::Tags::ChartQ::type{n};
  result.dt_u = gp::Tags::ChartU::type{n};
  result.dt_w = gp::Tags::ChartW::type{n};
  result.dt_q = gp::Tags::ChartQ::type{n};
  for (size_t s = 0; s < n; ++s) {
    const auto state = helpers::random_state(gen, 0.3, 0.3, 0.3);
    gp::set_potentials_at(make_not_null(&result.u), s, state.u);
    gp::set_velocities_at(make_not_null(&result.w), s, state.w);
    const auto targets = helpers::random_state(gen, 1.0, 1.0, 1.0);
    gp::set_potentials_at(make_not_null(&result.dt_u), s, targets.u);
    gp::set_velocities_at(make_not_null(&result.dt_w), s, targets.w);
    gp::set_gradients_at(make_not_null(&result.dt_q), s, targets.q);
  }
  gp::potential_gradients(make_not_null(&result.q), result.u,
                          result.geometry.mesh,
                          result.geometry.inverse_jacobian);
  return result;
}

gp::PotentialVector add(const gp::PotentialVector& a,
                        const gp::PotentialVector& b, const double factor) {
  gp::PotentialVector result{};
  for (size_t i = 0; i < result.size(); ++i) {
    gsl::at(result, i) = gsl::at(a, i) + factor * gsl::at(b, i);
  }
  return result;
}

// A smooth, spatially varying damping parameter eta on the element.
Scalar<DataVector> varying_eta(const helpers::Geometry& geometry) {
  return Scalar<DataVector>{
      0.6 + 0.3 * get<0>(geometry.logical) * get<1>(geometry.logical) -
      0.2 * get<2>(geometry.logical)};
}

struct ChartRates {
  gp::Tags::ChartU::type u_rate;
  gp::Tags::ChartW::type w_rate;
  gp::Tags::ChartQ::type fitted_gradient;
};

ChartRates chart_rates(const ElementData& data,
                       const gp::Tags::ChartW::type& dt_w_target,
                       const Scalar<DataVector>& eta, const double f) {
  const size_t n = data.geometry.mesh.number_of_grid_points();
  ChartRates result{gp::Tags::ChartU::type{n}, gp::Tags::ChartW::type{n},
                    gp::Tags::ChartQ::type{n}};
  gp::finalized_chart_rates(
      make_not_null(&result.u_rate), make_not_null(&result.w_rate),
      make_not_null(&result.fitted_gradient), data.u, data.w, data.q, data.dt_u,
      dt_w_target, data.dt_q, data.geometry.mesh,
      data.geometry.inverse_jacobian, data.operators, eta, f);
  return result;
}

// The velocity target with the volume source -eta b of the b rate.
gp::Tags::ChartW::type with_volume_damping(const ElementData& data,
                                           const Scalar<DataVector>& eta) {
  gp::Tags::ChartW::type result = data.dt_w;
  for (size_t i = 0; i < 3; ++i) {
    result.get(gp::VelocityIndex::b + i) -=
        get(eta) * data.w.get(gp::VelocityIndex::b + i);
  }
  return result;
}

template <typename VectorExpression>
double max_abs(const VectorExpression& x) {
  return max(abs(DataVector{x}));
}

// dR/dt along the numerical rates, R_u r + L w_t + sum_i E_i D_i r, must
// equal the target rate R_u R + L w_t^* + sum_i E_i q_{i,t}^*: this needs both
// compatibility terms. Also checks Z P = 0 and the mean of the fit. R, L, E_i
// and P are those of the Gamma-driver parameter f.
void test_compatibility(const gsl::not_null<std::mt19937*> gen,
                        const bool curved, const double f) {
  const ElementData data = make_element(gen, curved);
  const size_t n = data.geometry.mesh.number_of_grid_points();
  gp::Tags::ChartU::type u_rate{n};
  gp::Tags::ChartW::type w_rate{n};
  gp::Tags::ChartQ::type fitted_gradient{n};
  // The identity holds without the fitted shift damping (eta = 0).
  gp::finalized_chart_rates(make_not_null(&u_rate), make_not_null(&w_rate),
                            make_not_null(&fitted_gradient), data.u, data.w,
                            data.q, data.dt_u, data.dt_w, data.dt_q,
                            data.geometry.mesh, data.geometry.inverse_jacobian,
                            data.operators, Scalar<DataVector>{n, 0.0}, f);
  // The fit fixes the mass-weighted mean of the potential rate.
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    CHECK(data.operators.mean(u_rate.get(a)) ==
          approx(data.operators.mean(data.dt_u.get(a))));
  }
  // The returned gradient of the fit is the ordinary element derivative.
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    const auto derivative =
        partial_derivative(Scalar<DataVector>{u_rate.get(a)},
                           data.geometry.mesh, data.geometry.inverse_jacobian);
    for (size_t i = 0; i < 3; ++i) {
      CHECK_ITERABLE_APPROX(fitted_gradient.get(i, a), derivative.get(i));
    }
  }

  double largest_control_defect = 0.0;
  for (size_t s = 0; s < n; ++s) {
    const auto u = gp::potentials_at(data.u, s);
    const auto w = gp::velocities_at(data.w, s);
    const auto q = gp::gradients_at(data.q, s);
    const auto r = gp::potentials_at(u_rate, s);
    const auto w_t = gp::velocities_at(w_rate, s);
    const auto d_r = gp::gradients_at(fitted_gradient, s);
    const auto w_target = gp::velocities_at(data.dt_w, s);
    const auto q_target = gp::gradients_at(data.dt_q, s);
    const gp::LogMetricChart chart{
        gp::stf_decode(&u[gp::PotentialIndex::log_metric])};
    const auto rhs = gp::potential_rhs(chart, u, w, q, f);
    const auto numerical_rate =
        add(add(gp::potential_rhs_potential_derivative(chart, u, w, q, r),
                gp::potential_rhs(chart, u, w_t, gp::GradientVector{}, f), 1.0),
            gp::potential_rhs(chart, u, gp::VelocityVector{}, d_r, f), 1.0);
    const auto target_rate = add(
        add(gp::potential_rhs_potential_derivative(chart, u, w, q, rhs),
            gp::potential_rhs(chart, u, w_target, gp::GradientVector{}, f),
            1.0),
        gp::potential_rhs(chart, u, gp::VelocityVector{}, q_target, f), 1.0);
    double scale = 1.0;
    for (const double x : target_rate) {
      scale = std::max(scale, std::abs(x));
    }
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      CHECK(gsl::at(numerical_rate, a) ==
            approx(gsl::at(target_rate, a)).scale(scale).epsilon(1e-11));
    }
    // Z P = 0: the target rate of GammaHat - b is preserved.
    for (size_t i = 0; i < 3; ++i) {
      CHECK(gsl::at(w_t, gp::VelocityIndex::gamma_hat + i) -
                gsl::at(w_t, gp::VelocityIndex::b + i) ==
            approx(gsl::at(w_target, gp::VelocityIndex::gamma_hat + i) -
                   gsl::at(w_target, gp::VelocityIndex::b + i)));
    }
    // Control: without R_u (R - r) the identity fails.
    const auto missing =
        gp::apply_rhs_inverse(u[gp::PotentialIndex::lapse],
                              gp::potential_rhs_potential_derivative(
                                  chart, u, w, q, add(rhs, r, -1.0)),
                              f);
    gp::VelocityVector w_t_control = w_t;
    for (size_t a = 0; a < gp::number_of_velocities; ++a) {
      gsl::at(w_t_control, a) -= gsl::at(missing, a);
    }
    const auto control_rate = add(
        add(gp::potential_rhs_potential_derivative(chart, u, w, q, r),
            gp::potential_rhs(chart, u, w_t_control, gp::GradientVector{}, f),
            1.0),
        gp::potential_rhs(chart, u, gp::VelocityVector{}, d_r, f), 1.0);
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      largest_control_defect = std::max(
          largest_control_defect,
          std::abs(gsl::at(control_rate, a) - gsl::at(target_rate, a)) / scale);
    }
  }
  CHECK(largest_control_defect > 1e-4);
}

// When the gradient targets are the element gradients of a field with the
// prescribed mean, the fit reproduces that field and the defect vanishes.
void test_representable_targets(const gsl::not_null<std::mt19937*> gen,
                                const double f) {
  ElementData data = make_element(gen, true);
  const size_t n = data.geometry.mesh.number_of_grid_points();
  gp::Tags::ChartU::type exact{n};
  for (size_t s = 0; s < n; ++s) {
    gp::set_potentials_at(make_not_null(&exact), s,
                          helpers::random_state(gen, 1.0, 1.0, 1.0).u);
  }
  gp::potential_gradients(make_not_null(&data.dt_q), exact, data.geometry.mesh,
                          data.geometry.inverse_jacobian);
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    data.dt_u.get(a) = exact.get(a);
  }
  gp::Tags::ChartU::type u_rate{n};
  gp::Tags::ChartW::type w_rate{n};
  gp::Tags::ChartQ::type fitted_gradient{n};
  gp::finalized_chart_rates(make_not_null(&u_rate), make_not_null(&w_rate),
                            make_not_null(&fitted_gradient), data.u, data.w,
                            data.q, data.dt_u, data.dt_w, data.dt_q,
                            data.geometry.mesh, data.geometry.inverse_jacobian,
                            data.operators, Scalar<DataVector>{n, 0.0}, f);
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    CHECK_ITERABLE_CUSTOM_APPROX(u_rate.get(a), exact.get(a),
                                 Approx::custom().epsilon(1e-10).scale(1.0));
  }
  for (size_t s = 0; s < n; ++s) {
    const auto u = gp::potentials_at(data.u, s);
    const auto w = gp::velocities_at(data.w, s);
    const auto q = gp::gradients_at(data.q, s);
    const gp::LogMetricChart chart{
        gp::stf_decode(&u[gp::PotentialIndex::log_metric])};
    const auto expected =
        gp::apply_rhs_inverse(u[gp::PotentialIndex::lapse],
                              gp::potential_rhs_potential_derivative(
                                  chart, u, w, q,
                                  add(gp::potential_rhs(chart, u, w, q, f),
                                      gp::potentials_at(exact, s), -1.0)),
                              f);
    for (size_t a = 0; a < gp::number_of_velocities; ++a) {
      CHECK(w_rate.get(a)[s] ==
            approx(data.dt_w.get(a)[s] + gsl::at(expected, a))
                .epsilon(1e-9)
                .scale(1.0));
    }
  }
}

// The physical rates of the production mutator, with the configured
// Gamma-driver parameter Ccz4::fd::System::f and its eta argument: g_t =
// T_H r_H (SciPy Frechet derivative), the algebraic-constraint tangency, A_t
// from an independent complex step, and the direct copies.
void test_physical_rates(const gsl::not_null<std::mt19937*> gen,
                         const bool damped) {
  const ElementData data = make_element(gen, true);
  const size_t n = data.geometry.mesh.number_of_grid_points();
  const Scalar<DataVector> eta =
      damped ? varying_eta(data.geometry) : Scalar<DataVector>{n, 0.0};
  tnsr::ii<DataVector, 3> dt_g{n};
  Scalar<DataVector> dt_phi{n};
  tnsr::ii<DataVector, 3> dt_a_tilde{n};
  Scalar<DataVector> dt_k{n};
  Scalar<DataVector> dt_theta{n};
  tnsr::I<DataVector, 3> dt_gamma_hat{n};
  Scalar<DataVector> dt_lapse{n};
  tnsr::I<DataVector, 3> dt_shift{n};
  tnsr::I<DataVector, 3> dt_b{n};
  gp::FinalizeTimeDerivative::apply(
      make_not_null(&dt_g), make_not_null(&dt_phi), make_not_null(&dt_a_tilde),
      make_not_null(&dt_k), make_not_null(&dt_theta),
      make_not_null(&dt_gamma_hat), make_not_null(&dt_lapse),
      make_not_null(&dt_shift), make_not_null(&dt_b), data.u, data.w, data.q,
      data.dt_u, data.dt_w, data.dt_q, data.geometry.mesh,
      data.geometry.inverse_jacobian, data.operators, eta);
  const auto [u_rate, w_rate, fitted_gradient] =
      chart_rates(data, data.dt_w, eta, ::Ccz4::fd::System::f);
  // The b rates of the mutator carry the fitted shift damping of eta: they
  // match the chart rates of eta at every point and, when damped, differ from
  // those of eta = 0.
  for (size_t i = 0; i < 3; ++i) {
    CHECK(dt_b.get(i) == w_rate.get(gp::VelocityIndex::b + i));
    CHECK(dt_gamma_hat.get(i) == w_rate.get(gp::VelocityIndex::gamma_hat + i));
  }
  if (damped) {
    const ChartRates without_correction = chart_rates(
        data, data.dt_w, Scalar<DataVector>{n, 0.0}, ::Ccz4::fd::System::f);
    CHECK(max_abs(dt_b.get(0) -
                  without_correction.w_rate.get(gp::VelocityIndex::b)) >
          0.1);
  }
  for (size_t s = 0; s < n; s += 5) {
    const auto u = gp::potentials_at(data.u, s);
    const auto w = gp::velocities_at(data.w, s);
    const auto r = gp::potentials_at(u_rate, s);
    const auto w_t = gp::velocities_at(w_rate, s);
    const gp::Matrix3 h = gp::stf_decode(&u[gp::PotentialIndex::log_metric]);
    const gp::Matrix3 r_h = gp::stf_decode(&r[gp::PotentialIndex::log_metric]);
    const DataVector expected_dt_g = pypp::call<DataVector>(
        module, "tangent", helpers::flatten(h), helpers::flatten(r_h));
    DataVector h5{5};
    DataVector rh5{5};
    DataVector a5{5};
    DataVector at5{5};
    for (size_t c = 0; c < 5; ++c) {
      h5[c] = u[c];
      rh5[c] = r[c];
      a5[c] = w[c];
      at5[c] = w_t[c];
    }
    const DataVector expected_dt_a_tilde = pypp::call<DataVector>(
        module, "a_tilde_time_derivative", h5, rh5, a5, at5);
    gp::Matrix3 metric_rate{};
    gp::Matrix3 a_tilde_rate{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        metric_rate(i, j) = dt_g.get(i, j)[s];
        a_tilde_rate(i, j) = dt_a_tilde.get(i, j)[s];
        CHECK(dt_g.get(i, j)[s] ==
              approx(expected_dt_g[3 * i + j]).epsilon(1e-11).scale(1.0));
        CHECK(dt_a_tilde.get(i, j)[s] ==
              approx(expected_dt_a_tilde[3 * i + j]).epsilon(1e-11).scale(1.0));
      }
    }
    const gp::LogMetricChart chart{h};
    const gp::Matrix3 inverse_metric = chart.inverse_metric();
    const gp::Matrix3 a_tilde =
        0.5 * chart.tangent(gp::stf_decode(&w[gp::VelocityIndex::a]));
    // tr(g^{-1} g_t) = 0 and tr(g^{-1} A_t) = tr(g^{-1} g_t g^{-1} A).
    const gp::Matrix3 g_inv_g_t = inverse_metric * metric_rate;
    CHECK(std::abs(g_inv_g_t(0, 0) + g_inv_g_t(1, 1) + g_inv_g_t(2, 2)) <
          1e-11 * (1.0 + helpers::max_abs(g_inv_g_t)));
    const gp::Matrix3 lhs = inverse_metric * a_tilde_rate;
    const gp::Matrix3 rhs = g_inv_g_t * inverse_metric * a_tilde;
    CHECK(lhs(0, 0) + lhs(1, 1) + lhs(2, 2) ==
          approx(rhs(0, 0) + rhs(1, 1) + rhs(2, 2))
              .epsilon(1e-11)
              .scale(1.0 + helpers::max_abs(lhs)));
    CHECK(get(dt_phi)[s] ==
          approx(std::exp(u[gp::PotentialIndex::log_conformal_factor]) *
                 r[gp::PotentialIndex::log_conformal_factor]));
    CHECK(get(dt_lapse)[s] == r[gp::PotentialIndex::lapse]);
    CHECK(get(dt_k)[s] == w_t[gp::VelocityIndex::trace_extrinsic_curvature]);
    CHECK(get(dt_theta)[s] == w_t[gp::VelocityIndex::theta]);
    for (size_t i = 0; i < 3; ++i) {
      CHECK(dt_shift.get(i)[s] == gsl::at(r, gp::PotentialIndex::shift + i));
      CHECK(dt_gamma_hat.get(i)[s] ==
            gsl::at(w_t, gp::VelocityIndex::gamma_hat + i));
      CHECK(dt_b.get(i)[s] == gsl::at(w_t, gp::VelocityIndex::b + i));
    }
  }
}

// The fitted shift damping. With the volume source -eta b in the b rows of
// the velocity target, the finalized chart rates differ from the undamped
// rates (eta = 0, no volume source) only in the b rows, by -eta r_beta / f
// with r_beta the complete fitted shift rate including its prescribed mean;
// the potential rates, fitted gradients and all other velocity rates are
// unchanged. The data distinguish r_beta from f b, from its mean-free part and
// from the pointwise shift target, and b from GammaHat.
void test_fitted_shift_damping(const gsl::not_null<std::mt19937*> gen,
                               const bool curved, const double f) {
  ElementData data = make_element(gen, curved);
  const size_t n = data.geometry.mesh.number_of_grid_points();
  // A prescribed mean of the shift rate.
  for (size_t i = 0; i < 3; ++i) {
    data.dt_u.get(gp::PotentialIndex::shift + i) +=
        1.5 + 0.3 * static_cast<double>(i);
  }
  const ChartRates undamped =
      chart_rates(data, data.dt_w, Scalar<DataVector>{n, 0.0}, f);
  for (size_t i = 0; i < 3; ++i) {
    const DataVector& r_beta =
        undamped.u_rate.get(gp::PotentialIndex::shift + i);
    const DataVector& b = data.w.get(gp::VelocityIndex::b + i);
    CHECK(max_abs(b - data.w.get(gp::VelocityIndex::gamma_hat + i)) > 0.1);
    CHECK(max_abs(r_beta - f * b) > 0.1);
    CHECK(std::abs(data.operators.mean(r_beta)) > 0.5);
  }

  for (const size_t eta_case : {0_st, 1_st, 2_st}) {
    CAPTURE(eta_case);
    const Scalar<DataVector> eta =
        eta_case == 0 ? Scalar<DataVector>{n, 0.0}
                      : (eta_case == 1 ? Scalar<DataVector>{n, 0.7}
                                       : varying_eta(data.geometry));
    const ChartRates damped =
        chart_rates(data, with_volume_damping(data, eta), eta, f);
    CHECK(damped.u_rate == undamped.u_rate);
    CHECK(damped.fitted_gradient == undamped.fitted_gradient);
    for (size_t a = 0; a < gp::VelocityIndex::b; ++a) {
      CHECK(damped.w_rate.get(a) == undamped.w_rate.get(a));
    }
    for (size_t i = 0; i < 3; ++i) {
      const DataVector& r_beta =
          undamped.u_rate.get(gp::PotentialIndex::shift + i);
      const DataVector difference =
          damped.w_rate.get(gp::VelocityIndex::b + i) -
          undamped.w_rate.get(gp::VelocityIndex::b + i);
      const DataVector expected = -get(eta) * r_beta / f;
      CHECK_ITERABLE_CUSTOM_APPROX(difference, expected,
                                   Approx::custom().epsilon(1e-13).scale(1.0));
      if (eta_case != 0) {
        // Controls: the mean-free part of the fit, the pointwise shift target
        // and the unreplaced volume source -eta b all give different rates.
        const DataVector mean_free = r_beta - data.operators.mean(r_beta);
        CHECK(max_abs(difference + get(eta) * mean_free / f) > 0.1);
        CHECK(max_abs(difference +
                      get(eta) * data.dt_u.get(gp::PotentialIndex::shift + i) /
                          f) > 0.1);
        CHECK(max_abs(difference +
                      get(eta) * data.w.get(gp::VelocityIndex::b + i)) > 0.1);
      }
    }
  }
}

// Two limits of the fitted shift damping with nonzero b and a spatially
// varying eta: a stationary fitted shift rate (r_beta = 0) cancels the volume
// source -eta b exactly, and the continuum-compatible fitted rate
// r_beta = f b leaves the volume source unchanged.
void test_shift_damping_limits(const gsl::not_null<std::mt19937*> gen,
                               const bool curved, const double f) {
  ElementData data = make_element(gen, curved);
  const size_t n = data.geometry.mesh.number_of_grid_points();
  const Scalar<DataVector> eta = varying_eta(data.geometry);
  const Scalar<DataVector> zero_eta{n, 0.0};

  // Stationary: zero shift-rate targets and mean.
  for (size_t i = 0; i < 3; ++i) {
    data.dt_u.get(gp::PotentialIndex::shift + i) = 0.0;
    for (size_t j = 0; j < 3; ++j) {
      data.dt_q.get(j, gp::PotentialIndex::shift + i) = 0.0;
    }
  }
  {
    const ChartRates undamped = chart_rates(data, data.dt_w, zero_eta, f);
    const ChartRates damped =
        chart_rates(data, with_volume_damping(data, eta), eta, f);
    for (size_t i = 0; i < 3; ++i) {
      CHECK(max_abs(damped.u_rate.get(gp::PotentialIndex::shift + i)) == 0.0);
      CHECK(max_abs(data.w.get(gp::VelocityIndex::b + i)) > 0.1);
      CHECK_ITERABLE_CUSTOM_APPROX(
          damped.w_rate.get(gp::VelocityIndex::b + i),
          undamped.w_rate.get(gp::VelocityIndex::b + i),
          Approx::custom().epsilon(1e-13).scale(1.0));
    }
  }

  // Continuum compatible: the shift-rate targets are the element gradients
  // and mean of f b, which the fit reproduces.
  gp::Tags::ChartU::type compatible{n, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    compatible.get(gp::PotentialIndex::shift + i) =
        f * data.w.get(gp::VelocityIndex::b + i);
  }
  gp::Tags::ChartQ::type compatible_gradient{n};
  gp::potential_gradients(make_not_null(&compatible_gradient), compatible,
                          data.geometry.mesh, data.geometry.inverse_jacobian);
  for (size_t i = 0; i < 3; ++i) {
    data.dt_u.get(gp::PotentialIndex::shift + i) =
        compatible.get(gp::PotentialIndex::shift + i);
    for (size_t j = 0; j < 3; ++j) {
      data.dt_q.get(j, gp::PotentialIndex::shift + i) =
          compatible_gradient.get(j, gp::PotentialIndex::shift + i);
    }
  }
  {
    const ChartRates undamped = chart_rates(data, data.dt_w, zero_eta, f);
    const ChartRates damped =
        chart_rates(data, with_volume_damping(data, eta), eta, f);
    for (size_t i = 0; i < 3; ++i) {
      CHECK_ITERABLE_CUSTOM_APPROX(
          damped.u_rate.get(gp::PotentialIndex::shift + i),
          compatible.get(gp::PotentialIndex::shift + i),
          Approx::custom().epsilon(1e-10).scale(1.0));
      const DataVector unchanged_source =
          undamped.w_rate.get(gp::VelocityIndex::b + i) -
          get(eta) * data.w.get(gp::VelocityIndex::b + i);
      CHECK_ITERABLE_CUSTOM_APPROX(damped.w_rate.get(gp::VelocityIndex::b + i),
                                   unchanged_source,
                                   Approx::custom().epsilon(1e-9).scale(1.0));
    }
  }
}
}  // namespace

// [[TimeOut, 20]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection.FinalizeTimeDerivative",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Evolution/Systems/Ccz4/GradientProjection/");
  MAKE_GENERATOR(gen);
  for (const double f : {1.0, 0.5, 1.25}) {
    CAPTURE(f);
    test_compatibility(make_not_null(&gen), false, f);
    test_compatibility(make_not_null(&gen), true, f);
    test_representable_targets(make_not_null(&gen), f);
    for (const bool curved : {false, true}) {
      CAPTURE(curved);
      test_fitted_shift_damping(make_not_null(&gen), curved, f);
      test_shift_damping_limits(make_not_null(&gen), curved, f);
    }
  }
  test_physical_rates(make_not_null(&gen), false);
  test_physical_rates(make_not_null(&gen), true);
}
