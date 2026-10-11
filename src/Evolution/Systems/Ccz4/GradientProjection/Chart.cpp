// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"

#include <algorithm>
#include <array>
#include <blaze/math/lapack/clapack/syev.h>
#include <cmath>
#include <cstddef>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::GradientProjection {
namespace {
const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
const double inv_sqrt6 = 1.0 / std::sqrt(6.0);
const double sqrt2 = std::sqrt(2.0);

// W_ij = g_jk B_i^k + g_ik B_j^k - 2/3 g_ij d for a symmetric g.
Matrix3 shift_strain(const Matrix3& metric, const Matrix3& shift_gradient) {
  const double divergence =
      shift_gradient(0, 0) + shift_gradient(1, 1) + shift_gradient(2, 2);
  const Matrix3 b_times_g = shift_gradient * metric;
  return b_times_g + blaze::trans(b_times_g) -
         (2.0 / 3.0) * divergence * metric;
}

// B_i^k = q_{beta^k,i}
Matrix3 shift_gradient_matrix(const GradientVector& q) {
  Matrix3 result{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      result(i, k) = gsl::at(gsl::at(q, i), PotentialIndex::shift + k);
    }
  }
  return result;
}
}  // namespace

std::array<double, 5> stf_encode(const Matrix3& x) {
  return {{(x(0, 0) - x(1, 1)) * inv_sqrt2,
           (x(0, 0) + x(1, 1) - 2.0 * x(2, 2)) * inv_sqrt6,
           (x(0, 1) + x(1, 0)) * inv_sqrt2, (x(0, 2) + x(2, 0)) * inv_sqrt2,
           (x(1, 2) + x(2, 1)) * inv_sqrt2}};
}

Matrix3 stf_decode(const double* const components) {
  Matrix3 result{};
  result(0, 0) = components[0] * inv_sqrt2 + components[1] * inv_sqrt6;
  result(1, 1) = -components[0] * inv_sqrt2 + components[1] * inv_sqrt6;
  result(2, 2) = -2.0 * components[1] * inv_sqrt6;
  result(0, 1) = result(1, 0) = components[2] * inv_sqrt2;
  result(0, 2) = result(2, 0) = components[3] * inv_sqrt2;
  result(1, 2) = result(2, 1) = components[4] * inv_sqrt2;
  return result;
}

double sinhc(const double x) {
  if (std::abs(x) < 1.0e-4) {
    const double x_squared = x * x;
    return 1.0 + x_squared / 6.0 * (1.0 + x_squared / 20.0);
  }
  return std::sinh(x) / x;
}

double exp_divided_difference(const double x, const double y) {
  return std::exp(0.5 * (x + y)) * sinhc(0.5 * (x - y));
}

double exp_second_divided_difference(const double x, const double y,
                                     const double z) {
  std::array<double, 3> nodes{{x, y, z}};
  std::sort(nodes.begin(), nodes.end());
  const double span = nodes[2] - nodes[0];
  if (span > 1.0) {
    return (exp_divided_difference(nodes[1], nodes[2]) -
            exp_divided_difference(nodes[0], nodes[1])) /
           span;
  }
  const double mean = (nodes[0] + nodes[1] + nodes[2]) / 3.0;
  const double d0 = nodes[0] - mean;
  const double d1 = nodes[1] - mean;
  const double d2 = nodes[2] - mean;
  // Complete homogeneous symmetric polynomials h_j of (d0, d1, d2) by the
  // recurrences h_j(d2) = d2 h_{j-1}(d2), h_j(d1,d2) = d1 h_{j-1}(d1,d2) +
  // h_j(d2), h_j(d0,d1,d2) = d0 h_{j-1}(d0,d1,d2) + h_j(d1,d2).
  double h_2 = 1.0;
  double h_12 = 1.0;
  double h_012 = 1.0;
  double factorial = 2.0;
  double sum = 0.5;
  // With |d| <= 2/3 the j-th term is bounded by (2/3)^j / (2 j!), so the
  // terms beyond j = 24 are below 1e-27 while the sum exceeds 0.25.
  for (size_t j = 1; j <= 24; ++j) {
    h_2 *= d2;
    h_12 = d1 * h_12 + h_2;
    h_012 = d0 * h_012 + h_12;
    factorial *= static_cast<double>(j + 2);
    sum += h_012 / factorial;
  }
  return std::exp(mean) * sum;
}

void symmetric_eigensystem(
    const gsl::not_null<std::array<double, 3>*> eigenvalues,
    const gsl::not_null<Matrix3*> eigenvectors, const Matrix3& matrix) {
  // Copy into a contiguous column-major buffer for LAPACK.
  std::array<double, 9> buffer{};
  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(buffer, i + 3 * j) = 0.5 * (matrix(i, j) + matrix(j, i));
    }
  }
  std::array<double, 32> work{};
  blaze::blas_int_t info = 0;
  blaze::syev('V', 'L', 3, buffer.data(), 3, eigenvalues->data(), work.data(),
              static_cast<blaze::blas_int_t>(work.size()), &info);
  if (info != 0) {
    ERROR("LAPACK dsyev failed with info = " << info << " for the matrix\n"
                                             << matrix);
  }
  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < 3; ++i) {
      (*eigenvectors)(i, j) = gsl::at(buffer, i + 3 * j);
    }
  }
}

LogMetricChart::LogMetricChart(const Matrix3& log_metric) {
  symmetric_eigensystem(make_not_null(&h_), make_not_null(&q_), log_metric);
  compute_coefficients();
}

LogMetricChart::LogMetricChart(const std::array<double, 3>& log_eigenvalues,
                               const Matrix3& eigenvectors)
    : h_(log_eigenvalues), q_(eigenvectors) {
  compute_coefficients();
}

LogMetricChart LogMetricChart::from_metric(const Matrix3& metric) {
  std::array<double, 3> lambda{};
  Matrix3 q{};
  symmetric_eigensystem(make_not_null(&lambda), make_not_null(&q), metric);
  if (not(lambda[0] > 0.0)) {
    ERROR("The conformal metric is not positive definite. Eigenvalues: "
          << lambda[0] << ", " << lambda[1] << ", " << lambda[2]
          << "\nConformal metric:\n"
          << metric);
  }
  return LogMetricChart{{{0.5 * std::log(lambda[0]), 0.5 * std::log(lambda[1]),
                          0.5 * std::log(lambda[2])}},
                        q};
}

void LogMetricChart::compute_coefficients() {
  for (size_t r = 0; r < 3; ++r) {
    for (size_t s = 0; s < 3; ++s) {
      gsl::at(tangent_coefficients_, r + 3 * s) =
          2.0 * std::exp(gsl::at(h_, r) + gsl::at(h_, s)) *
          sinhc(gsl::at(h_, r) - gsl::at(h_, s));
    }
  }
  for (size_t r = 0; r < 3; ++r) {
    for (size_t t = r; t < 3; ++t) {
      for (size_t s = t; s < 3; ++s) {
        const double value =
            4.0 * exp_second_divided_difference(2.0 * gsl::at(h_, r),
                                                2.0 * gsl::at(h_, t),
                                                2.0 * gsl::at(h_, s));
        // The divided difference is symmetric in its arguments.
        for (const auto& [i, j, k] :
             std::array<std::array<size_t, 3>, 6>{{{{r, t, s}},
                                                   {{r, s, t}},
                                                   {{t, r, s}},
                                                   {{t, s, r}},
                                                   {{s, r, t}},
                                                   {{s, t, r}}}}) {
          gsl::at(second_coefficients_, i + 3 * j + 9 * k) = value;
        }
      }
    }
  }
}

Matrix3 LogMetricChart::to_eigenbasis(const Matrix3& x) const {
  return blaze::trans(q_) * x * q_;
}

Matrix3 LogMetricChart::from_eigenbasis(const Matrix3& x) const {
  return q_ * x * blaze::trans(q_);
}

Matrix3 LogMetricChart::eigen_function(
    const std::array<double, 3>& values) const {
  Matrix3 diagonal{};
  for (size_t r = 0; r < 3; ++r) {
    diagonal(r, r) = gsl::at(values, r);
  }
  return from_eigenbasis(diagonal);
}

Matrix3 LogMetricChart::log_metric() const { return eigen_function(h_); }

Matrix3 LogMetricChart::metric() const {
  return eigen_function(
      {{std::exp(2.0 * h_[0]), std::exp(2.0 * h_[1]), std::exp(2.0 * h_[2])}});
}

Matrix3 LogMetricChart::inverse_metric() const {
  return eigen_function({{std::exp(-2.0 * h_[0]), std::exp(-2.0 * h_[1]),
                          std::exp(-2.0 * h_[2])}});
}

Matrix3 LogMetricChart::sqrt_metric() const {
  return eigen_function({{std::exp(h_[0]), std::exp(h_[1]), std::exp(h_[2])}});
}

Matrix3 LogMetricChart::inverse_sqrt_metric() const {
  return eigen_function(
      {{std::exp(-h_[0]), std::exp(-h_[1]), std::exp(-h_[2])}});
}

Matrix3 LogMetricChart::tangent(const Matrix3& x) const {
  Matrix3 rotated = to_eigenbasis(x);
  for (size_t s = 0; s < 3; ++s) {
    for (size_t r = 0; r < 3; ++r) {
      rotated(r, s) *= gsl::at(tangent_coefficients_, r + 3 * s);
    }
  }
  return from_eigenbasis(rotated);
}

Matrix3 LogMetricChart::inverse_tangent(const Matrix3& x) const {
  Matrix3 rotated = to_eigenbasis(x);
  for (size_t s = 0; s < 3; ++s) {
    for (size_t r = 0; r < 3; ++r) {
      rotated(r, s) /= gsl::at(tangent_coefficients_, r + 3 * s);
    }
  }
  return from_eigenbasis(rotated);
}

Matrix3 LogMetricChart::second_derivative(const Matrix3& x,
                                          const Matrix3& y) const {
  const Matrix3 x_rotated = to_eigenbasis(x);
  const Matrix3 y_rotated = to_eigenbasis(y);
  Matrix3 result{};
  for (size_t s = 0; s < 3; ++s) {
    for (size_t r = 0; r < 3; ++r) {
      double value = 0.0;
      for (size_t t = 0; t < 3; ++t) {
        value += gsl::at(second_coefficients_, r + 3 * t + 9 * s) *
                 (x_rotated(r, t) * y_rotated(t, s) +
                  y_rotated(r, t) * x_rotated(t, s));
      }
      result(r, s) = value;
    }
  }
  return from_eigenbasis(result);
}

std::array<double, 2> chart_from_physical(
    const gsl::not_null<PotentialVector*> u,
    const gsl::not_null<VelocityVector*> w, const PhysicalPointState& state) {
  if (not(state.conformal_factor > 0.0)) {
    ERROR("The conformal factor must be positive, but is "
          << state.conformal_factor);
  }
  if (not(state.lapse > 0.0)) {
    ERROR("The lapse must be positive, but is " << state.lapse);
  }
  const LogMetricChart chart =
      LogMetricChart::from_metric(state.conformal_metric);
  const Matrix3 log_metric = chart.log_metric();
  const Matrix3 a = 2.0 * chart.inverse_tangent(state.a_tilde);
  const std::array<double, 5> encoded_h = stf_encode(log_metric);
  const std::array<double, 5> encoded_a = stf_encode(a);
  for (size_t c = 0; c < 5; ++c) {
    gsl::at(*u, PotentialIndex::log_metric + c) = gsl::at(encoded_h, c);
    gsl::at(*w, VelocityIndex::a + c) = gsl::at(encoded_a, c);
  }
  (*u)[PotentialIndex::conformal_factor] = state.conformal_factor;
  (*u)[PotentialIndex::lapse] = state.lapse;
  (*w)[VelocityIndex::trace_extrinsic_curvature] =
      state.trace_extrinsic_curvature;
  (*w)[VelocityIndex::theta] = state.theta;
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(*u, PotentialIndex::shift + i) = gsl::at(state.shift, i);
    gsl::at(*w, VelocityIndex::gamma_hat + i) = gsl::at(state.gamma_hat, i);
    gsl::at(*w, VelocityIndex::b + i) = gsl::at(state.b, i);
  }
  const auto& h = chart.log_eigenvalues();
  return {{h[0] + h[1] + h[2], a(0, 0) + a(1, 1) + a(2, 2)}};
}

PotentialVector potential_rhs(const LogMetricChart& chart,
                              const PotentialVector& u, const VelocityVector& w,
                              const GradientVector& q, const double f) {
  const double lapse = u[PotentialIndex::lapse];
  const Matrix3 strain = shift_strain(chart.metric(), shift_gradient_matrix(q));
  const double divergence = q[0][PotentialIndex::shift] +
                            q[1][PotentialIndex::shift + 1] +
                            q[2][PotentialIndex::shift + 2];
  const std::array<double, 5> shift_term =
      stf_encode(chart.inverse_tangent(strain));
  PotentialVector result{};
  for (size_t a = 0; a < 5; ++a) {
    double advection = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      advection += gsl::at(u, PotentialIndex::shift + i) *
                   gsl::at(gsl::at(q, i), PotentialIndex::log_metric + a);
    }
    gsl::at(result, PotentialIndex::log_metric + a) =
        -lapse * gsl::at(w, VelocityIndex::a + a) + advection +
        gsl::at(shift_term, a);
  }
  double conformal_factor_advection = 0.0;
  double lapse_advection = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    conformal_factor_advection +=
        gsl::at(u, PotentialIndex::shift + i) *
        gsl::at(gsl::at(q, i), PotentialIndex::conformal_factor);
    lapse_advection += gsl::at(u, PotentialIndex::shift + i) *
                       gsl::at(gsl::at(q, i), PotentialIndex::lapse);
  }
  const double trace_k = w[VelocityIndex::trace_extrinsic_curvature];
  result[PotentialIndex::conformal_factor] =
      conformal_factor_advection + u[PotentialIndex::conformal_factor] *
                                       (lapse * trace_k - divergence) / 3.0;
  result[PotentialIndex::lapse] =
      lapse_advection - 2.0 * lapse * (trace_k - 2.0 * w[VelocityIndex::theta]);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, PotentialIndex::shift + i) =
        f * gsl::at(w, VelocityIndex::b + i);
  }
  return result;
}

PotentialVector potential_rhs_potential_derivative(
    const LogMetricChart& chart, const PotentialVector& u,
    const VelocityVector& w, const GradientVector& q,
    const PotentialVector& delta_u) {
  const Matrix3 metric = chart.metric();
  const Matrix3 shift_gradient = shift_gradient_matrix(q);
  const Matrix3 delta_log_metric =
      stf_decode(&delta_u[PotentialIndex::log_metric]);
  const Matrix3 strain_over_tangent =
      chart.inverse_tangent(shift_strain(metric, shift_gradient));
  const Matrix3 delta_strain = chart.inverse_tangent(
      shift_strain(chart.tangent(delta_log_metric), shift_gradient) -
      chart.second_derivative(delta_log_metric, strain_over_tangent));
  const std::array<double, 5> delta_shift_term = stf_encode(delta_strain);
  const double delta_lapse = delta_u[PotentialIndex::lapse];
  PotentialVector result{};
  for (size_t a = 0; a < 5; ++a) {
    double advection = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      advection += gsl::at(delta_u, PotentialIndex::shift + i) *
                   gsl::at(gsl::at(q, i), PotentialIndex::log_metric + a);
    }
    gsl::at(result, PotentialIndex::log_metric + a) =
        -delta_lapse * gsl::at(w, VelocityIndex::a + a) + advection +
        gsl::at(delta_shift_term, a);
  }
  double conformal_factor_advection = 0.0;
  double lapse_advection = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    conformal_factor_advection +=
        gsl::at(delta_u, PotentialIndex::shift + i) *
        gsl::at(gsl::at(q, i), PotentialIndex::conformal_factor);
    lapse_advection += gsl::at(delta_u, PotentialIndex::shift + i) *
                       gsl::at(gsl::at(q, i), PotentialIndex::lapse);
  }
  const double trace_k = w[VelocityIndex::trace_extrinsic_curvature];
  const double divergence = q[0][PotentialIndex::shift] +
                            q[1][PotentialIndex::shift + 1] +
                            q[2][PotentialIndex::shift + 2];
  result[PotentialIndex::conformal_factor] =
      conformal_factor_advection +
      delta_u[PotentialIndex::conformal_factor] *
          (u[PotentialIndex::lapse] * trace_k - divergence) / 3.0 +
      u[PotentialIndex::conformal_factor] * trace_k * delta_lapse / 3.0;
  result[PotentialIndex::lapse] =
      lapse_advection -
      2.0 * delta_lapse * (trace_k - 2.0 * w[VelocityIndex::theta]);
  // R_beta = f b does not depend on u.
  return result;
}

VelocityVector apply_rhs_inverse(const double lapse,
                                 const double conformal_factor,
                                 const PotentialVector& delta_rhs,
                                 const double f) {
  VelocityVector result{};
  for (size_t a = 0; a < 5; ++a) {
    gsl::at(result, VelocityIndex::a + a) =
        -gsl::at(delta_rhs, PotentialIndex::log_metric + a) / lapse;
  }
  const double delta_k = 3.0 * delta_rhs[PotentialIndex::conformal_factor] /
                         (lapse * conformal_factor);
  result[VelocityIndex::trace_extrinsic_curvature] = delta_k;
  result[VelocityIndex::theta] =
      delta_rhs[PotentialIndex::lapse] / (4.0 * lapse) + 0.5 * delta_k;
  for (size_t i = 0; i < 3; ++i) {
    const double delta_b = gsl::at(delta_rhs, PotentialIndex::shift + i) / f;
    gsl::at(result, VelocityIndex::gamma_hat + i) = delta_b;
    gsl::at(result, VelocityIndex::b + i) = delta_b;
  }
  return result;
}
}  // namespace Ccz4::GradientProjection
