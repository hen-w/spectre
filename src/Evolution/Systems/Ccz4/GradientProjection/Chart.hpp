// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <blaze/math/StaticMatrix.h>
#include <cstddef>

#include "Utilities/Gsl.hpp"

/// Pointwise algebra of the CCZ4 gradient-projection scheme.
namespace Ccz4::GradientProjection {
/// A full \f$3\times3\f$ matrix used for symmetric tensors in the chart.
using Matrix3 = blaze::StaticMatrix<double, 3, 3, blaze::columnMajor>;

/// Number of chart potentials \f$u\f$.
constexpr size_t number_of_potentials = 10;
/// Number of chart velocities \f$w\f$.
constexpr size_t number_of_velocities = 13;
/// Number of components of the target vector \f$Y=(w,q_1,q_2,q_3)\f$.
constexpr size_t number_of_target_components =
    number_of_velocities + 3 * number_of_potentials;
/// Dimension of the normal-only subspace \f$(w,r)\f$ on which the normal
/// principal matrix acts after factorization.
constexpr size_t number_of_normal_components =
    number_of_velocities + number_of_potentials;

/*!
 * \brief Component offsets of the chart potentials
 * \f$u=(H_0,\ldots,H_4,\phi,\alpha,\beta^1,\beta^2,\beta^3)\f$.
 *
 * \f$H=\frac12\log\tilde\gamma\f$ is stored in the symmetric trace-free basis
 * of `stf_encode`; \f$\phi\f$, \f$\alpha\f$ and \f$\beta\f$ are the
 * physical conformal factor, lapse and shift.
 */
namespace PotentialIndex {
constexpr size_t log_metric = 0;
constexpr size_t conformal_factor = 5;
constexpr size_t lapse = 6;
constexpr size_t shift = 7;
}  // namespace PotentialIndex

/*!
 * \brief Component offsets of the chart velocities
 * \f$w=(a_0,\ldots,a_4,K,\Theta,\hat\Gamma^1,\hat\Gamma^2,\hat\Gamma^3,
 * b^1,b^2,b^3)\f$, where \f$a=2\mathcal{T}_H^{-1}\tilde A\f$ is stored in the
 * basis of `stf_encode`.
 */
namespace VelocityIndex {
constexpr size_t a = 0;
constexpr size_t trace_extrinsic_curvature = 5;
constexpr size_t theta = 6;
constexpr size_t gamma_hat = 7;
constexpr size_t b = 10;
}  // namespace VelocityIndex

/// Chart potentials \f$u\f$ at a point.
using PotentialVector = std::array<double, number_of_potentials>;
/// Chart velocities \f$w\f$ at a point.
using VelocityVector = std::array<double, number_of_velocities>;
/// Gradients \f$q_i\f$, `[i][A]`, of the potentials at a point.
using GradientVector = std::array<PotentialVector, 3>;

/*!
 * \brief Encode the trace-free part of a symmetric matrix in the fixed
 * Euclidean-orthonormal symmetric trace-free basis.
 *
 * \details The basis is \f$E_0=\mathrm{diag}(1,-1,0)/\sqrt2\f$,
 * \f$E_1=\mathrm{diag}(1,1,-2)/\sqrt6\f$, and \f$E_2,E_3,E_4\f$ the symmetric
 * \f$xy\f$, \f$xz\f$, \f$yz\f$ matrices with off-diagonal entries
 * \f$1/\sqrt2\f$. The encoding is the full Frobenius contraction
 * \f$X_A=E_A:X\f$, so the Euclidean trace of \f$X\f$ is discarded.
 */
std::array<double, 5> stf_encode(const Matrix3& x);

/// Decode \f$X=\sum_A X_A E_A\f$, see `stf_encode`.
Matrix3 stf_decode(const double* components);

/// The function \f$\sinh(x)/x\f$, continuous at \f$x=0\f$.
double sinhc(double x);

/*!
 * \brief First divided difference \f$f[x,y]\f$ of \f$f=\exp\f$, evaluated
 * as \f$e^{(x+y)/2}\,\mathrm{sinhc}((x-y)/2)\f$, which is accurate for nearly
 * equal and repeated arguments.
 */
double exp_divided_difference(double x, double y);

/*!
 * \brief Second divided difference \f$f[x,y,z]\f$ of \f$f=\exp\f$, with the
 * continuous limits at repeated arguments (\f$f[x,x,x]=e^x/2\f$).
 *
 * \details For a node span \f$\max-\min\le1\f$ the value is summed from the
 * Taylor series \f$e^m\sum_j h_j(\delta)/(j+2)!\f$ about the node mean
 * \f$m\f$, where \f$h_j\f$ are the complete homogeneous symmetric polynomials
 * of the node offsets \f$\delta\f$. Larger spans use the recurrence
 * \f$(f[y,z]-f[x,y])/(z-x)\f$ with sorted nodes, whose cancellation is then
 * bounded.
 */
double exp_second_divided_difference(double x, double y, double z);

/*!
 * \brief The logarithmic conformal-metric chart at a point.
 *
 * \details The conformal metric is \f$g=e^{2H}\f$ with symmetric \f$H\f$. The
 * chart stores the library (LAPACK `dsyev`) eigendecomposition
 * \f$H=Q\,\mathrm{diag}(h_r)Q^T\f$ and evaluates, without differentiating
 * eigenvectors,
 *
 * - the tangent map \f$\mathcal T_HX=D(e^{2H})[X]\f$, which acts entrywise in
 *   the eigenbasis with \f$t_{rs}=(\lambda_r-\lambda_s)/(h_r-h_s)
 *   =2e^{h_r+h_s}\mathrm{sinhc}(h_r-h_s)\f$, \f$\lambda_r=e^{2h_r}\f$,
 * - its inverse \f$\mathcal T_H^{-1}\f$, dividing by \f$t_{rs}\f$, and
 * - the second derivative
 *   \f$\mathcal S_H[X,Y]=(D\mathcal T_H[X])Y\f$ with eigenbasis entries
 *   \f$4\sum_t f[2h_r,2h_t,2h_s](X_{rt}Y_{ts}+Y_{rt}X_{ts})\f$, \f$f=\exp\f$.
 *
 * At \f$H=0\f$, \f$\mathcal T_HX=2X\f$ and \f$\mathcal S_H[X,Y]=2(XY+YX)\f$.
 */
class LogMetricChart {
 public:
  /// Decompose the logarithm \f$H\f$ (symmetric) directly.
  explicit LogMetricChart(const Matrix3& log_metric);

  /*!
   * \brief Build the chart from a conformal metric \f$g\f$.
   *
   * It is an `ERROR` if \f$g\f$ is not positive definite; eigenvalues are
   * never clipped.
   */
  static LogMetricChart from_metric(const Matrix3& metric);

  /// \f$H\f$.
  Matrix3 log_metric() const;
  /// \f$g=e^{2H}\f$.
  Matrix3 metric() const;
  /// \f$g^{-1}=e^{-2H}\f$.
  Matrix3 inverse_metric() const;
  /// The symmetric square root \f$X=e^{H}\f$ of the metric.
  Matrix3 sqrt_metric() const;
  /// \f$X^{-1}=e^{-H}\f$.
  Matrix3 inverse_sqrt_metric() const;

  /// \f$\mathcal T_HX\f$.
  Matrix3 tangent(const Matrix3& x) const;
  /// \f$\mathcal T_H^{-1}X\f$.
  Matrix3 inverse_tangent(const Matrix3& x) const;
  /// \f$\mathcal S_H[X,Y]\f$.
  Matrix3 second_derivative(const Matrix3& x, const Matrix3& y) const;

  /// Eigenvalues \f$h_r\f$ of \f$H\f$ in ascending order.
  const std::array<double, 3>& log_eigenvalues() const { return h_; }
  /// Eigenvectors of \f$H\f$ (columns).
  const Matrix3& eigenvectors() const { return q_; }

 private:
  LogMetricChart(const std::array<double, 3>& log_eigenvalues,
                 const Matrix3& eigenvectors);
  void compute_coefficients();
  Matrix3 to_eigenbasis(const Matrix3& x) const;
  Matrix3 from_eigenbasis(const Matrix3& x) const;
  Matrix3 eigen_function(const std::array<double, 3>& values) const;

  std::array<double, 3> h_{};
  Matrix3 q_{};
  // t_rs
  std::array<double, 9> tangent_coefficients_{};
  // 4 f[2h_r, 2h_t, 2h_s] indexed by r + 3 t + 9 s
  std::array<double, 27> second_coefficients_{};
};

/*!
 * \brief Symmetric eigendecomposition \f$A=Q\,\mathrm{diag}(\lambda)Q^T\f$ of
 * a symmetric \f$3\times3\f$ matrix using LAPACK `dsyev`. Eigenvalues are
 * ascending and eigenvectors are the columns of the returned matrix.
 */
void symmetric_eigensystem(gsl::not_null<std::array<double, 3>*> eigenvalues,
                           gsl::not_null<Matrix3*> eigenvectors,
                           const Matrix3& matrix);

/*!
 * \brief The physical CCZ4 state at a point.
 */
struct PhysicalPointState {
  Matrix3 conformal_metric{};
  double conformal_factor = 0.0;
  Matrix3 a_tilde{};
  double trace_extrinsic_curvature = 0.0;
  double theta = 0.0;
  std::array<double, 3> gamma_hat{};
  double lapse = 0.0;
  std::array<double, 3> shift{};
  std::array<double, 3> b{};
};

/*!
 * \brief Convert the physical state to the chart potentials \f$u\f$ and
 * velocities \f$w\f$.
 *
 * \details The potentials are \f$H=\frac12\log\tilde\gamma\f$ from the library
 * eigendecomposition, \f$\phi\f$, \f$\alpha\f$, \f$\beta\f$ and
 * \f$a=2\mathcal T_H^{-1}\tilde A\f$. The encodings of \f$H\f$ and \f$a\f$
 * discard their Euclidean traces
 * \f$\mathrm{tr}H=\frac12\log\det\tilde\gamma\f$ and
 * \f$\mathrm{tr}\,a=\tilde\gamma^{ij}\tilde A_{ij}\f$, which vanish on the
 * algebraic-constraint surface; they are returned (in this order) so callers
 * can verify that no off-constraint information is dropped. It is an `ERROR`
 * if \f$\tilde\gamma\f$ is not positive definite or \f$\phi\f$ or
 * \f$\alpha\f$ is not positive.
 */
std::array<double, 2> chart_from_physical(gsl::not_null<PotentialVector*> u,
                                          gsl::not_null<VelocityVector*> w,
                                          const PhysicalPointState& state);

/*!
 * \brief The original pointwise potential right-hand side \f$R(u,w,q)\f$.
 *
 * \details With \f$B_i{}^j=q_{\beta,i}{}^j\f$, \f$d=B_i{}^i\f$ and the
 * Gamma-driver parameter \f$f\f$,
 * \f{align*}{
 * W_{ij}&=g_{jk}B_i{}^k+g_{ik}B_j{}^k-\tfrac23g_{ij}d,\\
 * R_H&=-\alpha a+\beta^iq_{H,i}+\mathcal T_H^{-1}W,\\
 * R_\phi&=\beta^iq_{\phi,i}+\phi(\alpha K-d)/3,\\
 * R_\alpha&=\beta^iq_{\alpha,i}-2\alpha(K-2\Theta),\\
 * R_\beta&=fb.
 * \f}
 * The lapse rate assumes \f$K_0=0\f$. For fixed \f$u\f$ this is linear in
 * \f$(w,q)\f$ with no constant term, so \f$L\,y=R(u,y,0)\f$ and
 * \f$\sum_jE_jx_j=R(u,0,x)\f$.
 */
PotentialVector potential_rhs(const LogMetricChart& chart,
                              const PotentialVector& u, const VelocityVector& w,
                              const GradientVector& q, double f);

/*!
 * \brief The directional derivative \f$R_u\,\delta u\f$ at fixed \f$(w,q)\f$,
 * including the \f$H\f$ dependence of both \f$g\f$ and
 * \f$\mathcal T_H^{-1}\f$:
 *
 * \f{align*}{
 * \delta R_H&=-\delta\alpha\,a+\delta\beta^iq_{H,i}
 * +\mathcal T_H^{-1}\left(W(\mathcal T_H\delta H)
 * -\mathcal S_H[\delta H,\mathcal T_H^{-1}W]\right),\\
 * \delta R_\phi&=\delta\beta^iq_{\phi,i}+\delta\phi(\alpha K-d)/3
 * +\phi K\delta\alpha/3,\\
 * \delta R_\alpha&=\delta\beta^iq_{\alpha,i}-2\delta\alpha(K-2\Theta),\qquad
 * \delta R_\beta=0,
 * \f}
 * where \f$W(\delta g)\f$ is \f$W\f$ with \f$g\f$ replaced by \f$\delta g\f$.
 */
PotentialVector potential_rhs_potential_derivative(
    const LogMetricChart& chart, const PotentialVector& u,
    const VelocityVector& w, const GradientVector& q,
    const PotentialVector& delta_u);

/*!
 * \brief Apply \f$P=(L;Z)^{-1}(I_{10};0)\f$ to a potential-rate increment.
 *
 * \details Explicitly, \f$\delta a=-\delta R_H/\alpha\f$,
 * \f$\delta K=3\delta R_\phi/(\alpha\phi)\f$,
 * \f$\delta\Theta=\delta R_\alpha/(4\alpha)+\delta K/2\f$,
 * \f$\delta b=\delta\hat\Gamma=\delta R_\beta/f\f$, so that \f$LP=I\f$ (with
 * the Gamma-driver parameter \f$f\f$ of `potential_rhs`) and \f$ZP=0\f$ with
 * \f$Zw=\hat\Gamma-b\f$.
 */
VelocityVector apply_rhs_inverse(double lapse, double conformal_factor,
                                 const PotentialVector& delta_rhs, double f);
}  // namespace Ccz4::GradientProjection
