// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Ccz4::GradientProjection {
/*!
 * \brief Static element-local operators: the constrained least-squares
 * gradient fit and the mass-weighted top-shell filter.
 *
 * \details For a static tensor-product Legendre-Gauss-Lobatto element with
 * ordinary derivatives \f$D_i=\sum_a\mathrm{diag}(J^{-1}{}^a{}_i)\hat D_a\f$
 * and coordinate-map mass \f$M=\mathrm{diag}(J\,w_1w_2w_3)\f$, with
 * \f$m=M\mathbf 1\f$ and \f$V=\mathbf 1^Tm\f$:
 *
 * - `fit` solves, for each field,
 *   \f$\min_r\frac12\sum_i(D_ir-q_i^*)^TM(D_ir-q_i^*)\f$ subject to
 *   \f$m^Tr/V=\mu\f$, as
 *   \f$r=\mu\mathbf 1+Z_0(Z_0^TK_DZ_0)^{-1}Z_0^Tb_D\f$ with
 *   \f$K_D=\sum_iD_i^TMD_i\f$, \f$b_D=\sum_iD_i^TMq_i^*\f$ and \f$Z_0\f$ a
 *   basis of the mass-zero-mean subspace. The Cholesky factor of
 *   \f$Z_0^TK_DZ_0\f$ is built once; it is an `ERROR` if this matrix has a
 *   relative eigenvalue below \f$10^{-12}\f$, indicating an additional
 *   nullspace or excessive numerical conditioning of the reduced system.
 * - `apply_filter` applies \f$F=\Pi+g(I-\Pi)\f$ with the given gain \f$g\f$,
 *   where \f$\Pi\f$ is the \f$M\f$-orthogonal projector onto the
 *   tensor-product polynomials of degree at most \f$p_i-1\f$ in every logical
 *   direction. The removed shell \f$\max(l/p_0,m/p_1,n/p_2)=1\f$ contains the
 *   top mode in each direction.
 *
 * Both use one mass-weighted QR factorization
 * \f$M^{1/2}B=QR\f$ of the Legendre Vandermonde matrix \f$B\f$ with columns
 * ordered (constant, remaining retained modes, removed shell), so
 * \f$Z_0=M^{-1/2}Q_{[:,1:]}\f$ and
 * \f$\Pi=M^{-1/2}Q_{\mathrm{low}}Q_{\mathrm{low}}^TM^{1/2}\f$. Normal-equation
 * inverses are never formed. The operators are valid only for the mesh and
 * Jacobians they were built with.
 */
class LocalOperators {
 public:
  LocalOperators() = default;
  LocalOperators(const Mesh<3>& mesh,
                 const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                                       Frame::Inertial>& inverse_jacobian,
                 const Scalar<DataVector>& det_inverse_jacobian);

  const Mesh<3>& mesh() const { return mesh_; }
  /// The diagonal of the coordinate-map mass matrix \f$m=J\,w_1w_2w_3\f$.
  const DataVector& mass() const { return mass_; }
  /// \f$V=\mathbf 1^Tm\f$.
  double volume() const { return volume_; }
  /// The mass-weighted mean \f$m^Tv/V\f$.
  double mean(const DataVector& field) const;
  /// The number of retained (filtered at unit gain) modes.
  size_t number_of_retained_modes() const { return number_of_retained_modes_; }

  /*!
   * \brief Fit `fields.size()` potential rates.
   *
   * `gradient_targets[i][A]` are the target gradients \f$q^*_{i}\f$ of field
   * `A` and `means[A]` the prescribed mass-weighted means.
   */
  template <size_t NumberOfFields>
  void fit(gsl::not_null<std::array<DataVector, NumberOfFields>*> fields,
           const std::array<std::array<const DataVector*, NumberOfFields>, 3>&
               gradient_targets,
           const std::array<double, NumberOfFields>& means) const;

  /// \f$\Pi v\f$.
  DataVector project_retained(const DataVector& field) const;

  /// Apply the top-shell filter with the given `gain` in place.
  void apply_filter(gsl::not_null<DataVector*> field, double gain) const;

  /// Apply the top-shell filter with the given `gain` in place to each of
  /// `fields`: the same as `apply_filter` on each field, with the projections
  /// of all fields computed by one matrix-matrix product.
  void apply_filter(const std::vector<gsl::not_null<DataVector*>>& fields,
                    double gain) const;

  /// Error if `mesh` differs from the mesh the operators were built for.
  void check_mesh(const Mesh<3>& mesh) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  friend bool operator==(const LocalOperators& lhs, const LocalOperators& rhs);

  Mesh<3> mesh_{};
  DataVector mass_{};
  DataVector sqrt_mass_{};
  double volume_ = 0.0;
  std::array<DataVector, 9> inverse_jacobian_{};
  // Q of the mass-weighted QR, columns (constant, retained, removed shell).
  Matrix orthonormal_basis_{};
  size_t number_of_retained_modes_ = 0;
  // Lower Cholesky factor of Z0^T K_D Z0.
  Matrix fit_cholesky_factor_{};
};

bool operator!=(const LocalOperators& lhs, const LocalOperators& rhs);
}  // namespace Ccz4::GradientProjection
