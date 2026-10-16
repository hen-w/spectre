// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <pup.h>
#include <utility>

#include "DataStructures/StaticMatrix.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace Ccz4::GradientProjection {
namespace {
const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
const double inv_sqrt6 = 1.0 / std::sqrt(6.0);
const double sqrt2 = std::sqrt(2.0);

// 1/(j+2)! for j = 0, ..., 16: the coefficients of the series in
// exp_second_divided_difference.
constexpr std::array<double, 17> series_inverse_factorials = []() {
  std::array<double, 17> result{};
  double factorial = 2.0;
  result[0] = 1.0 / factorial;
  for (size_t j = 1; j < result.size(); ++j) {
    factorial *= static_cast<double>(j + 2);
    result[j] = 1.0 / factorial;
  }
  return result;
}();

// The position of t_rs = t_sr among the 6 distinct tangent coefficients,
// indexed by r + 3 s.
constexpr std::array<size_t, 9> tangent_index{{0, 1, 2, 1, 3, 4, 2, 4, 5}};

// The position of 4 f[2h_r, 2h_t, 2h_s] among the 10 distinct second
// coefficients, indexed by r + 3 t + 9 s; the divided difference is symmetric
// in its arguments.
constexpr std::array<size_t, 27> second_index = []() {
  std::array<size_t, 27> result{};
  size_t position = 0;
  for (size_t a = 0; a < 3; ++a) {
    for (size_t b = a; b < 3; ++b) {
      for (size_t c = b; c < 3; ++c) {
        result[a + 3 * b + 9 * c] = result[a + 3 * c + 9 * b] =
            result[b + 3 * a + 9 * c] = result[b + 3 * c + 9 * a] =
                result[c + 3 * a + 9 * b] = result[c + 3 * b + 9 * a] =
                    position;
        ++position;
      }
    }
  }
  return result;
}();

// W_ij = g_jk B_i^k + g_ik B_j^k - 2/3 g_ij d for a symmetric g, with g and B
// in the same orthonormal basis.
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
  double sum = series_inverse_factorials[0];
  // With |d| <= 2/3 the j-th term is bounded by (2/3)^j / (2 j!), so the
  // terms beyond j = 16 add up to less than 1.5e-18, while the sum exceeds
  // 0.25: the truncation error is below 6e-18 relative, under the
  // double-precision unit roundoff 1.1e-16.
  for (size_t j = 1; j < series_inverse_factorials.size(); ++j) {
    h_2 *= d2;
    h_12 = d1 * h_12 + h_2;
    h_012 = d0 * h_012 + h_12;
    sum += h_012 * gsl::at(series_inverse_factorials, j);
  }
  return std::exp(mean) * sum;
}

void symmetric_eigensystem(
    const gsl::not_null<std::array<double, 3>*> eigenvalues,
    const gsl::not_null<Matrix3*> eigenvectors, const Matrix3& matrix) {
  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < 3; ++i) {
      if (not std::isfinite(matrix(i, j))) {
        ERROR("The matrix has a non-finite entry:\n" << matrix);
      }
    }
  }
  // Cyclic Jacobi iteration on the symmetric part of the matrix: the rotation
  // in the (p, q) plane zeroes the entry A_pq, the off-diagonal entries
  // converge quadratically to zero and the diagonal to the eigenvalues. The
  // eigenvectors are the product of the rotations.
  std::array<double, 3>& diagonal = *eigenvalues;
  diagonal = {{matrix(0, 0), matrix(1, 1), matrix(2, 2)}};
  // off[r] is the entry A_pq of the plane (p, q) that excludes the index r.
  std::array<double, 3> off{{0.5 * (matrix(1, 2) + matrix(2, 1)),
                             0.5 * (matrix(0, 2) + matrix(2, 0)),
                             0.5 * (matrix(0, 1) + matrix(1, 0))}};
  Matrix3& vectors = *eigenvectors;
  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < 3; ++i) {
      vectors(i, j) = i == j ? 1.0 : 0.0;
    }
  }
  // The planes (p, q) and the index r they exclude.
  constexpr std::array<std::array<size_t, 3>, 3> planes{
      {{{0, 1, 2}}, {{0, 2, 1}}, {{1, 2, 0}}}};
  constexpr size_t maximum_sweeps = 50;
  for (size_t sweep = 0; sweep < maximum_sweeps; ++sweep) {
    if (off[0] == 0.0 and off[1] == 0.0 and off[2] == 0.0) {
      // Ascending eigenvalues, with the eigenvectors permuted alike.
      const auto swap = [&diagonal, &vectors](const size_t i, const size_t j) {
        std::swap(gsl::at(diagonal, i), gsl::at(diagonal, j));
        for (size_t k = 0; k < 3; ++k) {
          std::swap(vectors(k, i), vectors(k, j));
        }
      };
      if (diagonal[1] < diagonal[0]) {
        swap(0, 1);
      }
      if (diagonal[2] < diagonal[1]) {
        swap(1, 2);
      }
      if (diagonal[1] < diagonal[0]) {
        swap(0, 1);
      }
      return;
    }
    for (const auto& [p, q, r] : planes) {
      double& a_pq = gsl::at(off, r);
      if (a_pq == 0.0) {
        continue;
      }
      double& a_pp = gsl::at(diagonal, p);
      double& a_qq = gsl::at(diagonal, q);
      // An entry below the rounding of both diagonal entries it couples is
      // dropped instead of rotated.
      const double hundred_a_pq = 100.0 * std::abs(a_pq);
      if (std::abs(a_pp) + hundred_a_pq == std::abs(a_pp) and
          std::abs(a_qq) + hundred_a_pq == std::abs(a_qq)) {
        a_pq = 0.0;
        continue;
      }
      // t = tan(phi) of the rotation angle |phi| <= pi / 4 that zeroes A_pq,
      // the smaller root of t^2 + 2 theta t - 1 = 0. For A_pq below the
      // rounding of A_qq - A_pp the root is A_pq / (A_qq - A_pp) to rounding
      // (and theta^2 could overflow).
      const double difference = a_qq - a_pp;
      double t = 0.0;
      if (std::abs(difference) + hundred_a_pq == std::abs(difference)) {
        t = a_pq / difference;
      } else {
        const double theta = 0.5 * difference / a_pq;
        t = std::copysign(
            1.0 / (std::abs(theta) + std::sqrt(1.0 + theta * theta)), theta);
      }
      const double c = 1.0 / std::sqrt(1.0 + t * t);
      const double s = t * c;
      const double tau = s / (1.0 + c);
      a_pp -= t * a_pq;
      a_qq += t * a_pq;
      a_pq = 0.0;
      // The other entries of rows p and q: A_rp = off[q] and A_rq = off[p].
      double& a_rp = gsl::at(off, q);
      double& a_rq = gsl::at(off, p);
      const double old_a_rp = a_rp;
      a_rp -= s * (a_rq + tau * old_a_rp);
      a_rq += s * (old_a_rp - tau * a_rq);
      for (size_t k = 0; k < 3; ++k) {
        const double v_p = vectors(k, p);
        const double v_q = vectors(k, q);
        vectors(k, p) = v_p - s * (v_q + tau * v_p);
        vectors(k, q) = v_q + s * (v_p - tau * v_q);
      }
    }
  }
  ERROR("The Jacobi eigenvalue iteration did not converge in "
        << maximum_sweeps << " sweeps for the matrix\n"
        << matrix);
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

LogMetricChart LogMetricChart::trace_free() const {
  // Every coefficient is a divided difference of exp(2 h) at the eigenvalues,
  // so shifting them by -mean scales it by exp(-2 mean).
  const double mean = (h_[0] + h_[1] + h_[2]) / 3.0;
  const double scale = std::exp(-2.0 * mean);
  LogMetricChart result{};
  result.h_ = {{h_[0] - mean, h_[1] - mean, h_[2] - mean}};
  result.q_ = q_;
  for (size_t i = 0; i < tangent_coefficients_.size(); ++i) {
    gsl::at(result.tangent_coefficients_, i) =
        scale * gsl::at(tangent_coefficients_, i);
  }
  for (size_t i = 0; i < second_coefficients_.size(); ++i) {
    gsl::at(result.second_coefficients_, i) =
        scale * gsl::at(second_coefficients_, i);
  }
  return result;
}

void LogMetricChart::pup(PUP::er& p) {
  p | h_;
  p | q_;
  p | tangent_coefficients_;
  p | second_coefficients_;
}

bool operator==(const LogMetricChart& lhs, const LogMetricChart& rhs) {
  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < 3; ++i) {
      if (lhs.q_(i, j) != rhs.q_(i, j)) {
        return false;
      }
    }
  }
  return lhs.h_ == rhs.h_ and
         lhs.tangent_coefficients_ == rhs.tangent_coefficients_ and
         lhs.second_coefficients_ == rhs.second_coefficients_;
}

bool operator!=(const LogMetricChart& lhs, const LogMetricChart& rhs) {
  return not(lhs == rhs);
}

void LogMetricChart::compute_coefficients() {
  // t_rs = 2 e^{h_r} e^{h_s} sinhc(h_r - h_s) = t_sr.
  const std::array<double, 3> exp_h{
      {std::exp(h_[0]), std::exp(h_[1]), std::exp(h_[2])}};
  for (size_t r = 0; r < 3; ++r) {
    for (size_t s = r; s < 3; ++s) {
      gsl::at(tangent_coefficients_, gsl::at(tangent_index, r + 3 * s)) =
          2.0 * gsl::at(exp_h, r) * gsl::at(exp_h, s) *
          sinhc(gsl::at(h_, r) - gsl::at(h_, s));
    }
  }
  for (size_t r = 0; r < 3; ++r) {
    for (size_t t = r; t < 3; ++t) {
      for (size_t s = t; s < 3; ++s) {
        gsl::at(second_coefficients_,
                gsl::at(second_index, r + 3 * t + 9 * s)) =
            4.0 * exp_second_divided_difference(2.0 * gsl::at(h_, r),
                                                2.0 * gsl::at(h_, t),
                                                2.0 * gsl::at(h_, s));
      }
    }
  }
}

Matrix3 LogMetricChart::to_eigenbasis(const Matrix3& x) const {
  return blaze::trans(q_) * x * q_;
}

Matrix3 LogMetricChart::from_eigenbasis(const Matrix3& x_hat) const {
  return q_ * x_hat * blaze::trans(q_);
}

Matrix3 LogMetricChart::eigenbasis_metric() const {
  // t_rr = 2 e^{2 h_r}
  Matrix3 result{};
  for (size_t r = 0; r < 3; ++r) {
    result(r, r) =
        0.5 * gsl::at(tangent_coefficients_, gsl::at(tangent_index, r + 3 * r));
  }
  return result;
}

Matrix3 LogMetricChart::eigenbasis_tangent(const Matrix3& x_hat) const {
  Matrix3 result{};
  for (size_t s = 0; s < 3; ++s) {
    for (size_t r = 0; r < 3; ++r) {
      result(r, s) = x_hat(r, s) * gsl::at(tangent_coefficients_,
                                           gsl::at(tangent_index, r + 3 * s));
    }
  }
  return result;
}

Matrix3 LogMetricChart::eigenbasis_inverse_tangent(const Matrix3& x_hat) const {
  Matrix3 result{};
  for (size_t s = 0; s < 3; ++s) {
    for (size_t r = 0; r < 3; ++r) {
      result(r, s) = x_hat(r, s) / gsl::at(tangent_coefficients_,
                                           gsl::at(tangent_index, r + 3 * s));
    }
  }
  return result;
}

Matrix3 LogMetricChart::eigenbasis_second_derivative(
    const Matrix3& x_hat, const Matrix3& y_hat) const {
  Matrix3 result{};
  for (size_t s = 0; s < 3; ++s) {
    for (size_t r = 0; r < 3; ++r) {
      double value = 0.0;
      for (size_t t = 0; t < 3; ++t) {
        value += gsl::at(second_coefficients_,
                         gsl::at(second_index, r + 3 * t + 9 * s)) *
                 (x_hat(r, t) * y_hat(t, s) + y_hat(r, t) * x_hat(t, s));
      }
      result(r, s) = value;
    }
  }
  return result;
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
  return from_eigenbasis(eigenbasis_tangent(to_eigenbasis(x)));
}

Matrix3 LogMetricChart::inverse_tangent(const Matrix3& x) const {
  return from_eigenbasis(eigenbasis_inverse_tangent(to_eigenbasis(x)));
}

Matrix3 LogMetricChart::second_derivative(const Matrix3& x,
                                          const Matrix3& y) const {
  return from_eigenbasis(
      eigenbasis_second_derivative(to_eigenbasis(x), to_eigenbasis(y)));
}

std::array<double, 2> chart_from_physical(
    const gsl::not_null<PotentialVector*> u,
    const gsl::not_null<VelocityVector*> w,
    const gsl::not_null<LogMetricChart*> chart,
    const PhysicalPointState& state) {
  if (not(state.conformal_factor > 0.0)) {
    ERROR("The conformal factor must be positive, but is "
          << state.conformal_factor);
  }
  if (not(state.lapse > 0.0)) {
    ERROR("The lapse must be positive, but is " << state.lapse);
  }
  const LogMetricChart full_chart =
      LogMetricChart::from_metric(state.conformal_metric);
  const Matrix3 log_metric = full_chart.log_metric();
  const Matrix3 a = 2.0 * full_chart.inverse_tangent(state.a_tilde);
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
  *chart = full_chart.trace_free();
  const auto& h = full_chart.log_eigenvalues();
  return {{h[0] + h[1] + h[2], a(0, 0) + a(1, 1) + a(2, 2)}};
}

ShiftGradientTerms::ShiftGradientTerms(const LogMetricChart& chart,
                                       const GradientVector& q)
    : eigenbasis_shift_gradient(chart.to_eigenbasis(shift_gradient_matrix(q))),
      eigenbasis_strain_over_tangent(chart.eigenbasis_inverse_tangent(
          shift_strain(chart.eigenbasis_metric(), eigenbasis_shift_gradient))) {
}

PotentialVector potential_rhs(const LogMetricChart& chart,
                              const PotentialVector& u, const VelocityVector& w,
                              const GradientVector& q, const double f) {
  return potential_rhs(chart, ShiftGradientTerms{chart, q}, u, w, q, f);
}

PotentialVector potential_rhs(const LogMetricChart& chart,
                              const ShiftGradientTerms& shift_terms,
                              const PotentialVector& u, const VelocityVector& w,
                              const GradientVector& q, const double f) {
  const double lapse = u[PotentialIndex::lapse];
  const double divergence = q[0][PotentialIndex::shift] +
                            q[1][PotentialIndex::shift + 1] +
                            q[2][PotentialIndex::shift + 2];
  const std::array<double, 5> shift_term = stf_encode(
      chart.from_eigenbasis(shift_terms.eigenbasis_strain_over_tangent));
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
  return potential_rhs_potential_derivative(chart, ShiftGradientTerms{chart, q},
                                            u, w, q, delta_u);
}

PotentialVector potential_rhs_potential_derivative(
    const LogMetricChart& chart, const ShiftGradientTerms& shift_terms,
    const PotentialVector& u, const VelocityVector& w, const GradientVector& q,
    const PotentialVector& delta_u) {
  // In the eigenbasis of H.
  const Matrix3 delta_log_metric_hat =
      chart.to_eigenbasis(stf_decode(&delta_u[PotentialIndex::log_metric]));
  const Matrix3 delta_strain_hat = chart.eigenbasis_inverse_tangent(
      shift_strain(chart.eigenbasis_tangent(delta_log_metric_hat),
                   shift_terms.eigenbasis_shift_gradient) -
      chart.eigenbasis_second_derivative(
          delta_log_metric_hat, shift_terms.eigenbasis_strain_over_tangent));
  const std::array<double, 5> delta_shift_term =
      stf_encode(chart.from_eigenbasis(delta_strain_hat));
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
