// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Ccz4::GradientProjection {
/// The number of fields filtered by the shell velocity filter: the potential
/// rates \f$R\f$ in the order of `PotentialIndex`, followed by the three
/// components of \f$s=\hat\Gamma-b\f$.
constexpr size_t number_of_shell_filter_fields = number_of_potentials + 3;

/// The fields filtered by the shell velocity filter, see
/// `number_of_shell_filter_fields`.
using ShellFilterFields =
    std::array<gsl::not_null<DataVector*>, number_of_shell_filter_fields>;

/*!
 * \brief The retained angular spaces of the native tensor spherical-harmonic
 * filter, shared by all shell elements with the same \f$L\f$ and number of
 * removed bands.
 *
 * \details For the ranks \f$r=0,1,2\f$ (`Scalar`, `tnsr::i` and `tnsr::ii`
 * structures) the native filter is \f$I+\Delta_r\f$ with \f$\Delta_r\f$ from
 * `ylm::TensorYlm::fill_filter` with Spherepack normalization,
 * `angular_modes_to_kill` removed bands and no exponential roll-off, acting
 * on the Spherepack coefficients of the Cartesian components. Its retained
 * space \f$A_r\f$ is its range. On rank two it is used in the five
 * orthonormal symmetric trace-free coordinates of `stf_encode`, through
 * `stf_decode_matrix` and `stf_encode_matrix` (which account for the
 * multiplicities of the off-diagonal components), so that the component norm
 * is the Frobenius norm.
 *
 * With the Gram matrix \f$G=S^TWS\f$ of the scalar synthesis \f$S\f$ of
 * `ShellAngularOperators` in the Spherepack integration weights \f$W\f$, the
 * native filter is checked, in the coefficients of each angular degree
 * \f$l\f$ (it does not couple different degrees), to be \f$G\f$-self-adjoint
 * and idempotent: `self_adjointness_errors` holds the relative Frobenius norm
 * of the antisymmetric part of \f$C^T(I+\Delta)C^{-T}\f$, \f$G=CC^T\f$, and
 * `projector_errors` the largest distance of an eigenvalue of its symmetric
 * part from 0 or 1. The Spherepack analysis is checked to be
 * \f$G^{-1}S^TW\f$, so that the native nodal filter is the
 * \f$W\f$-orthogonal projector onto \f$A_r\f$. Each of these checks is an
 * `ERROR` beyond \f$10^{-11}\f$, as is `angular_modes_to_kill` larger than
 * \f$L\f$. Zero removed bands is not the identity: the native filter still
 * removes the tensor content of total degree above \f$L\f$.
 *
 * `retained_bases[r]` (\f$\Phi_r\f$) holds a \f$W\f$-orthonormal basis of
 * \f$A_r\f$ (the eigenvectors with eigenvalues above \f$\frac12\f$) as nodal
 * values, row \f$j+n_\Omega c\f$ for the angular point \f$j\f$ of component
 * \f$c\f$ (with 1, 3 and 5 components), one basis function per column.
 */
struct ShellFilterAngularSpaces {
  ShellFilterAngularSpaces(size_t l_max_in, size_t angular_modes_to_kill_in);

  size_t l_max;
  size_t angular_modes_to_kill;
  std::array<SimpleSparseMatrix, 3> native_filters;
  std::array<Matrix, 3> retained_bases;
  std::array<double, 3> self_adjointness_errors;
  std::array<double, 3> projector_errors;
  /// The \f$6\times5\f$ matrix of `stf_decode` into `tnsr::ii` storage.
  Matrix stf_decode_matrix;
  /// The \f$5\times6\f$ matrix of `stf_encode` from `tnsr::ii` storage.
  Matrix stf_encode_matrix;
};

/// The cached `ShellFilterAngularSpaces` for the given \f$L\ge2\f$ and
/// number of removed bands.
const ShellFilterAngularSpaces& shell_filter_angular_spaces(
    size_t l_max, size_t angular_modes_to_kill);

/*!
 * \brief The general evaluation of the shell velocity filter, valid for any
 * positive mass.
 *
 * \details The filter acts on five groups of `ShellFilterFields`: the five
 * trace-free components of \f$R_H\f$ (rank two), \f$R_\phi\f$ and
 * \f$R_\alpha\f$ (scalars), \f$R_\beta\f$ and \f$s\f$ (vectors). For a group
 * of rank \f$r\f$ let \f$S_A=\mathcal P_{N_r-1}\otimes A_r\f$ and
 * \f$S_0=\mathcal P_{N_r-2}\otimes A_r\f$, with \f$\mathcal P_k\f$ the radial
 * polynomials of degree at most \f$k\f$ and \f$A_r\f$ the retained angular
 * space of `ShellFilterAngularSpaces`, and let \f$\Pi_A\f$, \f$\Pi_0\f$ be
 * the orthogonal projectors onto them in the mass inner product
 * \f$\sum_{c,p}m_pu_c(p)v_c(p)\f$ over the group's components. The filter is
 *
 * \f{align*}{
 *   F=\Pi_0+g(\Pi_A-\Pi_0),\qquad
 *   \|Fv\|_M^2=\|\Pi_0v\|_M^2+g^2\|(\Pi_A-\Pi_0)v\|_M^2.
 * \f}
 *
 * At \f$g=1\f$ it is \f$\Pi_A\f$, which removes the angular content outside
 * \f$A_r\f$. In the radial nodal basis the mass is block diagonal: with the
 * angular coordinates \f$c_i\f$ of a function of \f$S_A\f$ at the radial
 * point \f$i\f$ (values \f$\Phi_rc_i\f$), its mass norm is
 * \f$\sum_ic_i^TA_ic_i\f$ with \f$A_i=\Phi_r^TM_i\Phi_r\f$, \f$M_i\f$ the
 * mass at the radial point \f$i\f$. So
 * \f$\Pi_Av=\sum_i\Phi_rA_i^{-1}\Phi_r^TM_iv_i\f$ exactly. \f$S_0\f$ is the
 * subspace with \f$\sum_it_ic_i=0\f$, \f$t\f$ the last row of the radial
 * nodal-to-modal matrix, and its \f$M\f$-orthogonal complement in \f$S_A\f$
 * consists of the \f$c_i=t_iA_i^{-1}y\f$. Hence
 * \f$(\Pi_A-\Pi_0)v\f$ has the coordinates \f$t_iA_i^{-1}H^{-1}
 * \sum_kt_kc_k\f$ with \f$H=\sum_it_i^2A_i^{-1}\f$. The Cholesky factors of
 * the \f$A_i\f$ and of \f$H\f$ are stored; it is an `ERROR` if one of these
 * matrices is not positive definite.
 */
class GeneralShellFilter {
 public:
  GeneralShellFilter() = default;
  GeneralShellFilter(const Mesh<3>& mesh, const DataVector& mass,
                     size_t angular_modes_to_kill);

  /// Apply the filter with the gain \f$g\f$ in place.
  void apply(const ShellFilterFields& fields, double gain) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  friend bool operator==(const GeneralShellFilter& lhs,
                         const GeneralShellFilter& rhs);

  void apply_rank(const std::vector<DataVector*>& fields, size_t rank,
                  double gain) const;

  Mesh<3> mesh_{};
  size_t angular_modes_to_kill_ = 0;
  DataVector mass_{};
  DataVector top_radial_mode_{};
  // For each rank, the lower Cholesky factors of A_0, ..., A_{N_r-1} and of
  // H, side by side.
  std::array<Matrix, 3> factors_{};
  // Not serialized: re-resolved from `shell_filter_angular_spaces` at
  // construction and when unpacking, so no application takes the cache lock.
  const ShellFilterAngularSpaces* spaces_ = nullptr;
};

bool operator!=(const GeneralShellFilter& lhs, const GeneralShellFilter& rhs);

/*!
 * \brief The separable evaluation of the shell velocity filter.
 *
 * \details With \f$\rho_i\f$ the average over the angular points \f$j\f$ of
 * \f$m(i,j)/w_\Omega(j)\f$, this applies to each group of `ShellFilterFields`
 * (see `GeneralShellFilter`) the native angular tensor filter of
 * `ShellFilterAngularSpaces` (through the Spherepack transforms, on the
 * `tnsr::ii` storage for rank two) at every radial point, and then the
 * radial filter \f$I-(1-g)\,q\,(\rho q)^T\f$ along every radial line, where
 * \f$q\propto t/\rho\f$ (\f$t\f$ as in `GeneralShellFilter`) is the unit
 * \f$\rho\f$-weighted complement of \f$\mathcal P_{N_r-2}\f$. It equals the
 * general evaluation when the mass is \f$m(i,j)=\rho_iw_\Omega(j)\f$.
 */
class SeparableShellFilter {
 public:
  SeparableShellFilter() = default;
  SeparableShellFilter(const Mesh<3>& mesh, const DataVector& mass,
                       size_t angular_modes_to_kill);

  /// Apply the filter with the gain \f$g\f$ in place.
  void apply(const ShellFilterFields& fields, double gain) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  friend bool operator==(const SeparableShellFilter& lhs,
                         const SeparableShellFilter& rhs);

  void apply_rank(const std::vector<DataVector*>& fields, size_t rank,
                  double gain) const;

  Mesh<3> mesh_{};
  size_t angular_modes_to_kill_ = 0;
  DataVector radial_weights_{};
  DataVector top_radial_mode_{};
  // Not serialized: re-resolved from `shell_filter_angular_spaces` at
  // construction and when unpacking, so no application takes the cache lock.
  const ShellFilterAngularSpaces* spaces_ = nullptr;
};

bool operator!=(const SeparableShellFilter& lhs,
                const SeparableShellFilter& rhs);

/*!
 * \brief Static element-local operators: the least-squares gradient fit of
 * the potential rates and the mass-weighted velocity filters.
 *
 * \details Two element families are supported, both with ordinary
 * derivatives \f$D_i\f$ combining the native logical derivatives with the
 * inverse Jacobian, and with the coordinate-map mass
 * \f$M=\mathrm{diag}(m)\f$, \f$m_p=w(p)/\det J^{-1}(p)\f$, and
 * \f$V=\mathbf 1^Tm\f$:
 *
 * - tensor-product Legendre-Gauss-Lobatto elements, with \f$w\f$ the product
 *   of the one-dimensional quadrature weights;
 * - spherical-harmonic shells: Legendre-Gauss-Lobatto in the first (radial)
 *   dimension with \f$N_r\ge2\f$ points, and a scalar spherical-harmonic
 *   basis with Gauss and equiangular quadratures and \f$L+1\f$ and
 *   \f$2L+1\f$ points, \f$L\ge2\f$, in the other two. Here
 *   \f$w=w_rw_\Omega\f$ with \f$w_\Omega\f$ the Spherepack integration
 *   weights, which contain the angular measure.
 *
 * On tensor-product elements:
 *
 * - `fit` solves, for each field,
 *   \f$\min_r\frac12\sum_i(D_ir-q_i^*)^TM(D_ir-q_i^*)\f$ subject to
 *   \f$m^Tr/V=\mu\f$, as
 *   \f$r=\mu\mathbf 1+Z_0(Z_0^TK_DZ_0)^{-1}Z_0^Tb_D\f$ with
 *   \f$K_D=\sum_iD_i^TMD_i\f$, \f$b_D=\sum_iD_i^TMq_i^*\f$ and \f$Z_0\f$ a
 *   basis of the mass-zero-mean subspace. The Cholesky factor of
 *   \f$Z_0^TK_DZ_0\f$ is built once; it is an `ERROR` if this matrix has a
 *   relative eigenvalue below \f$10^{-8}\f$ (the same bound as the shell
 *   fit): the eigenvalues are squared singular values of the fit operator,
 *   so this guarantees normal-equation solve errors below \f$\sim10^{-8}\f$
 *   and rejects an additional nullspace or an excessively ill-conditioned
 *   element.
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
 * \f$\Pi=M^{-1/2}Q_{\mathrm{low}}Q_{\mathrm{low}}^TM^{1/2}\f$.
 *
 * On spherical-harmonic shells the trial space of the fit is spanned by all
 * \f$P_k(\xi)Y_{lm}\f$ with \f$0\le k<N_r\f$ and \f$0\le l\le L\f$ (real
 * harmonics), of dimension \f$K=N_r(L+1)^2\f$, with nodal values
 * \f$B=V_r\otimes S\f$: \f$V_r\f$ the radial Legendre Vandermonde matrix and
 * \f$S\f$ the angular synthesis of `ShellAngularOperators`. The derivatives
 * are \f$D_i=\sum_a\mathrm{diag}(J^{-1}{}^a_i)\hat D_a\f$ with the logical
 * derivatives \f$\hat D_0=D_r\otimes I\f$ and \f$\hat D_{1,2}=I\otimes
 * G_{1,2}\f$, which are those of `partial_derivatives`. The fit minimizes
 * \f$\sum_i(D_iBc-q_i^*)^TM(D_iBc-q_i^*)\f$ over the coefficients \f$c\f$.
 * Its derivative kernel \f$\{c:D_iBc=0\}\f$ consists of the radially
 * constant functions times the angular kernel of `ShellAngularOperators`,
 * for every map. With the Gram matrix \f$B^TMB=L_GL_G^T\f$, the stiffness
 * \f$K_D=\sum_i(D_iB)^TM(D_iB)\f$, the \f$M\f$-orthonormal coordinates
 * \f$\tilde c=L_G^Tc\f$, an orthonormal basis \f$\tilde N\f$ of the kernel
 * in these coordinates and an orthonormal basis \f$W\f$ of its complement,
 * `fit_from_provisional` returns \f$r=Bc\f$ for each field with provisional
 * rate \f$r^*\f$, where
 *
 * \f{align*}{
 *   c=L_G^{-T}\left[W\left(W^T\tilde K_DW\right)^{-1}W^TL_G^{-1}
 *     \sum_i(D_iB)^TMq_i^*+\tilde N\tilde N^TL_G^{-1}B^TMr^*\right],
 *   \qquad\tilde K_D=L_G^{-1}K_DL_G^{-T}:
 * \f}
 *
 * the least-squares fit of the gradient targets over the trial space,
 * completed in the derivative kernel by the \f$M\f$-projections of
 * \f$r^*\f$ onto the kernel functions, which the fit leaves unchanged. The
 * Cholesky factor of \f$W^T\tilde K_DW\f$ is built once; as on
 * tensor-product elements, it is an `ERROR` if this matrix has a relative
 * eigenvalue below \f$10^{-12}\f$. It is also an `ERROR` if the derivatives
 * of the \f$M\f$-orthonormal kernel functions exceed \f$10^{-11}\f$ times
 * the largest singular value of the weighted derivative operator, or if the
 * logical derivatives built from `ShellAngularOperators` differ from those
 * of `partial_derivatives` by more than roundoff.
 *
 * On tensor-product elements normal-equation inverses are never formed; the
 * shell fit solves its normal equations by Cholesky in mass-orthonormal
 * coordinates, justified by the measured conditioning and guarded by the
 * relative-eigenvalue check.
 *
 * On spherical-harmonic shells the velocity filter acts on grouped tensors:
 * `apply_shell_filter` applies the `SeparableShellFilter` when the mass is
 * separable to within `shell_mass_separability_residual()` \f$\le10^{-13}\f$
 * and the `GeneralShellFilter` otherwise (the two evaluate the same operator
 * up to that residual); `apply_shell_filter_general` and
 * `apply_shell_filter_separable` apply a specific evaluation. All use
 * `shell_angular_modes_to_kill` removed angular bands, which must not exceed
 * \f$L\f$ (it is ignored on tensor-product elements). The operators are valid
 * only for the mesh and Jacobians they were built with.
 */
class LocalOperators {
 public:
  LocalOperators() = default;
  LocalOperators(const Mesh<3>& mesh,
                 const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                                       Frame::Inertial>& inverse_jacobian,
                 const Scalar<DataVector>& det_inverse_jacobian,
                 size_t shell_angular_modes_to_kill);

  const Mesh<3>& mesh() const { return mesh_; }
  /// Whether the operators were built for a spherical-harmonic shell.
  bool is_spherical_shell() const { return spherical_shell_; }
  /// The diagonal \f$m\f$ of the coordinate-map mass matrix.
  const DataVector& mass() const { return mass_; }
  /// \f$V=\mathbf 1^Tm\f$.
  double volume() const { return volume_; }
  /// The mass-weighted mean \f$m^Tv/V\f$.
  double mean(const DataVector& field) const;
  /// The number of retained (filtered at unit gain) modes. It is an `ERROR`
  /// on a spherical-harmonic shell.
  size_t number_of_retained_modes() const;

  /*!
   * \brief Fit `fields.size()` potential rates on a tensor-product element.
   *
   * `gradient_targets[i][A]` are the target gradients \f$q^*_{i}\f$ of field
   * `A` and `means[A]` the prescribed mass-weighted means. It is an `ERROR`
   * on a spherical-harmonic shell.
   */
  template <size_t NumberOfFields>
  void fit(gsl::not_null<std::array<DataVector, NumberOfFields>*> fields,
           const std::array<std::array<const DataVector*, NumberOfFields>, 3>&
               gradient_targets,
           const std::array<double, NumberOfFields>& means) const;

  /*!
   * \brief Fit `fields.size()` potential rates to the gradient targets
   * `gradient_targets[i][A]`, given the provisional rates
   * `provisional_rates[A]`.
   *
   * \details On a tensor-product element this is `fit` with the
   * mass-weighted means of the provisional rates. On a spherical-harmonic
   * shell the provisional rates complete the fit in the derivative kernel.
   */
  template <size_t NumberOfFields>
  void fit_from_provisional(
      gsl::not_null<std::array<DataVector, NumberOfFields>*> fields,
      const std::array<std::array<const DataVector*, NumberOfFields>, 3>&
          gradient_targets,
      const std::array<const DataVector*, NumberOfFields>& provisional_rates)
      const;

  /// The nodal values \f$BL_G^{-T}\tilde N\f$ of the \f$M\f$-orthonormal
  /// basis of the shell derivative kernel, one function per column. It is an
  /// `ERROR` on a tensor-product element.
  Matrix derivative_kernel() const;

  /// @{
  /*!
   * \brief The shell trial basis \f$B\f$ and its derivatives \f$D_iB\f$, as
   * applied by `fit_from_provisional`, and their transposes.
   *
   * \details Each column of `coefficients` (\f$K\f$ rows) holds the
   * coefficients of one trial function, the coefficient of
   * \f$P_k(\xi)Y_{lm}\f$ at row \f$k+N_r\,n\f$, where \f$n\f$ is the
   * position of \f$(l,m)\f$ (and of the cosine or sine part) in the
   * `ylm::SpherepackIterator` order. Each column of `values` and of
   * `fluxes[i]` holds nodal values. `trial_function_gradients_transpose`
   * returns \f$\sum_i(D_iB)^Tf_i\f$ for the nodal `fluxes` \f$f_i\f$. These
   * are `ERROR`s on a tensor-product element.
   */
  Matrix trial_functions(const Matrix& coefficients) const;
  Matrix trial_functions_transpose(const Matrix& values) const;
  std::array<Matrix, 3> trial_function_gradients(
      const Matrix& coefficients) const;
  Matrix trial_function_gradients_transpose(
      const std::array<Matrix, 3>& fluxes) const;
  /// @}

  /// \f$\Pi v\f$. It is an `ERROR` on a spherical-harmonic shell.
  DataVector project_retained(const DataVector& field) const;

  /// Apply the top-shell filter with the given `gain` in place. It is an
  /// `ERROR` on a spherical-harmonic shell.
  void apply_filter(gsl::not_null<DataVector*> field, double gain) const;

  /// Apply the top-shell filter with the given `gain` in place to each of
  /// `fields`: the same as `apply_filter` on each field, with the projections
  /// of all fields computed by one matrix-matrix product. It is an `ERROR` on
  /// a spherical-harmonic shell.
  void apply_filter(const std::vector<gsl::not_null<DataVector*>>& fields,
                    double gain) const;

  /// The number of removed angular bands of the shell velocity filter.
  size_t shell_angular_modes_to_kill() const {
    return shell_angular_modes_to_kill_;
  }

  /*!
   * \brief Apply the shell velocity filter with the given `gain` in place:
   * the `SeparableShellFilter` when `shell_mass_separability_residual()` is
   * at most \f$10^{-13}\f$ and the `GeneralShellFilter` otherwise.
   *
   * \details The two are evaluations of the same operator whose results
   * differ by the residual (relative) up to a conditioning factor of order
   * one, so the dispatch cannot change the filter beyond roundoff; a mass
   * that genuinely deviates from
   * \f$\rho_iw_\Omega(j)\f$ takes the general evaluation. It is an `ERROR`
   * on a tensor-product element.
   */
  void apply_shell_filter(const ShellFilterFields& fields, double gain) const;

  /// Apply the `GeneralShellFilter` with the given `gain` in place. It is an
  /// `ERROR` on a tensor-product element.
  void apply_shell_filter_general(const ShellFilterFields& fields,
                                  double gain) const;

  /// Apply the `SeparableShellFilter` with the given `gain` in place. It is
  /// an `ERROR` on a tensor-product element.
  void apply_shell_filter_separable(const ShellFilterFields& fields,
                                    double gain) const;

  /// The largest relative deviation of the mass from the separable form
  /// \f$\rho_iw_\Omega(j)\f$ of `SeparableShellFilter`. It is an `ERROR` on
  /// a tensor-product element.
  double shell_mass_separability_residual() const;

  /// Error if `mesh` differs from the mesh the operators were built for.
  void check_mesh(const Mesh<3>& mesh) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  friend bool operator==(const LocalOperators& lhs, const LocalOperators& rhs);

  void build_tensor_product(const Mesh<3>& mesh);
  void build_spherical_shell(
      const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian);
  template <size_t NumberOfFields>
  void fit_spherical_shell(
      gsl::not_null<std::array<DataVector, NumberOfFields>*> fields,
      const std::array<std::array<const DataVector*, NumberOfFields>, 3>&
          gradient_targets,
      const std::array<const DataVector*, NumberOfFields>& provisional_rates)
      const;
  void solve_spherical_shell(gsl::not_null<double*> coefficients,
                             size_t number_of_fields,
                             size_t leading_dimension) const;
  void require_tensor_product(const char* operation) const;
  void require_spherical_shell(const char* operation) const;

  Mesh<3> mesh_{};
  bool spherical_shell_ = false;
  DataVector mass_{};
  DataVector sqrt_mass_{};
  double volume_ = 0.0;
  std::array<DataVector, 9> inverse_jacobian_{};
  // Tensor-product elements: Q of the mass-weighted QR, columns (constant,
  // retained, removed shell).
  Matrix orthonormal_basis_{};
  size_t number_of_retained_modes_ = 0;
  // Tensor-product elements: lower Cholesky factor of Z0^T K_D Z0.
  Matrix fit_cholesky_factor_{};
  // Shells: the lower triangle (with the diagonal) holds L_G. The upper
  // triangle (with the diagonal) of the block of rows [0, K - n) and columns
  // [n, K), n the kernel dimension, holds the upper Cholesky factor of
  // W^T K~_D W; it lies strictly above the diagonal.
  Matrix shell_factors_{};
  // Shells: the Householder reflectors and their scalar factors of the QR
  // factorization [N~ W] R of the kernel in the M-orthonormal coordinates.
  Matrix shell_kernel_reflectors_{};
  DataVector shell_kernel_reflector_factors_{};
  // Shells: the coefficients L_G^{-T} N~ of the M-orthonormal kernel basis.
  Matrix shell_kernel_{};
  size_t shell_angular_modes_to_kill_ = 0;
  GeneralShellFilter general_shell_filter_{};
  SeparableShellFilter separable_shell_filter_{};
  double shell_mass_separability_residual_ = 0.0;
};

/*!
 * \brief The angular operators of the scalar spherical-harmonic basis with
 * \f$l_{\max}=m_{\max}=L\f$, shared by all shell elements with this \f$L\f$.
 *
 * \details With \f$n_\Omega=(L+1)(2L+1)\f$ angular grid points in the
 * `ylm::Spherepack` ordering and the \f$n_Y=(L+1)^2\f$ valid real harmonics
 * in the `ylm::SpherepackIterator` order:
 *
 * - `synthesis` (\f$S\f$, \f$n_\Omega\times n_Y\f$): the nodal values of the
 *   harmonics with unit Spherepack coefficients.
 * - `derivatives` (\f$G_1,G_2\f$, \f$n_\Omega\times n_\Omega\f$): the
 *   logical angular derivatives of nodal values that
 *   `ylm::Spherepack::gradient_all_offsets` applies, column \f$j\f$ being
 *   the derivative of the \f$j\f$-th unit nodal vector.
 * - `trial_derivatives` (\f$2n_\Omega\times n_Y\f$): the stacked products
 *   \f$G_1S\f$ and \f$G_2S\f$.
 * - `kernel` (\f$n_Y\times n_0\f$): an orthonormal basis of the coefficient
 *   vectors annihilated by `trial_derivatives`, the singular vectors with
 *   singular values below \f$10^{-11}\f$ times the largest. It is an
 *   `ERROR` if its dimension is not \f$1+(L+1)\f$ for even \f$L\f$ and
 *   \f$1+L\f$ for odd \f$L\f$, or if the smallest retained singular value is
 *   below 100 times the largest discarded one.
 */
struct ShellAngularOperators {
  explicit ShellAngularOperators(size_t l_max);

  size_t l_max;
  Matrix synthesis;
  std::array<Matrix, 2> derivatives;
  Matrix trial_derivatives;
  Matrix kernel;
};

/// The cached `ShellAngularOperators` for the given \f$L\f$,
/// \f$2\le L\le150\f$.
const ShellAngularOperators& shell_angular_operators(size_t l_max);

bool operator!=(const LocalOperators& lhs, const LocalOperators& rhs);
}  // namespace Ccz4::GradientProjection
