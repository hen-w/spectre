// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"

#include <algorithm>
#include <array>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/Submatrix.h>
#include <blaze/math/dense/QR.h>
#include <blaze/math/lapack/clapack/geqrf.h>
#include <blaze/math/lapack/clapack/ormqr.h>
#include <blaze/math/lapack/clapack/potrf.h>
#include <blaze/math/lapack/clapack/potrs.h>
#include <blaze/math/lapack/clapack/trtrs.h>
#include <blaze/math/lapack/gesvd.h>
#include <blaze/math/lapack/syev.h>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <pup.h>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"
#include "Utilities/StaticCache.hpp"

namespace Ccz4::GradientProjection {
namespace {
// Singular values at most this fraction of the largest belong to the
// derivative kernel.
constexpr double shell_kernel_tolerance = 1.0e-11;
// The smallest admissible ratio of the smallest retained to the largest
// discarded angular singular value.
constexpr double shell_kernel_minimum_gap = 100.0;
// The smallest admissible eigenvalue of the reduced gradient-fit stiffness
// relative to its largest, for both element families. The eigenvalues are
// squared singular values of the fit operator, and the normal-equation solve
// loses about machine epsilon over this ratio in relative accuracy, so this
// bound guarantees fit errors below ~1e-8; the measured ratio on supported
// shell meshes is ~3e-4, four orders of magnitude away.
constexpr double fit_minimum_relative_eigenvalue = 1.0e-8;
// The largest admissible difference between the logical derivatives built
// from the angular operators and those of `partial_derivatives`, relative
// to the largest derivative.
constexpr double shell_derivative_tolerance = 1.0e-12;
// The largest admissible relative deviation of the native angular filter and
// of the Spherepack transforms from the properties the shell velocity filter
// relies on.
constexpr double shell_filter_tolerance = 1.0e-11;
// The shell velocity filter uses its separable evaluation when the largest
// relative deviation of the mass from rho(i) w_Omega(j) is at most this.
// Both evaluations are the same operator up to that deviation, so the
// dispatch cannot change the filter by more than this relative amount.
constexpr double shell_mass_separability_threshold = 1.0e-13;

// A batch of F functions on a shell with N_r radial points is stored as a
// column-major (N_r F) x n array whose entry (i + N_r A, j) belongs to
// function A at the radial index i and the angular point j (nodal values,
// n = n_Omega) or harmonic j (coefficients of P_i Y_j, n = n_Y). The same
// storage is an N_r x (F n) array, on which radial matrices act.
void apply_radial(const gsl::not_null<double*> result, const Matrix& radial,
                  const char transpose, const double* const values,
                  const size_t columns) {
  const size_t radial_size = radial.rows();
  dgemm_(transpose, 'N', radial_size, columns, radial_size, 1.0, radial.data(),
         radial.spacing(), values, radial_size, 0.0, result.get(), radial_size);
}

// result = beta result + values op(angular), with `values` of size
// rows x inner.
void apply_angular(const gsl::not_null<double*> result,
                   const double* const values, const size_t rows,
                   const Matrix& angular, const char transpose,
                   const size_t result_columns, const size_t inner,
                   const double beta) {
  dgemm_('N', transpose, rows, result_columns, inner, 1.0, values, rows,
         angular.data(), angular.spacing(), beta, result.get(), rows);
}

// Nodal values B c of a batch of trial functions.
void synthesize(const gsl::not_null<double*> values,
                const double* const coefficients, const size_t number_of_fields,
                const Mesh<3>& mesh, const ShellAngularOperators& angular) {
  const size_t rows = mesh.extents(0) * number_of_fields;
  const size_t harmonics = angular.synthesis.columns();
  DataVector radial_values{rows * harmonics};
  apply_radial(make_not_null(radial_values.data()),
               Spectral::modal_to_nodal_matrix(mesh.slice_through(0)), 'N',
               coefficients, number_of_fields * harmonics);
  apply_angular(values, radial_values.data(), rows, angular.synthesis, 'T',
                angular.synthesis.rows(), harmonics, 0.0);
}

// B^T v for a batch of nodal values v.
void synthesize_transpose(const gsl::not_null<double*> coefficients,
                          const double* const values,
                          const size_t number_of_fields, const Mesh<3>& mesh,
                          const ShellAngularOperators& angular) {
  const size_t rows = mesh.extents(0) * number_of_fields;
  const size_t harmonics = angular.synthesis.columns();
  DataVector angular_coefficients{rows * harmonics};
  apply_angular(make_not_null(angular_coefficients.data()), values, rows,
                angular.synthesis, 'N', harmonics, angular.synthesis.rows(),
                0.0);
  apply_radial(coefficients,
               Spectral::modal_to_nodal_matrix(mesh.slice_through(0)), 'T',
               angular_coefficients.data(), number_of_fields * harmonics);
}

// The logical derivatives \hat D_a B c of a batch of trial functions, stored
// as three consecutive nodal batches.
void logical_gradient(const gsl::not_null<double*> logical,
                      const double* const coefficients,
                      const size_t number_of_fields, const Mesh<3>& mesh,
                      const ShellAngularOperators& angular) {
  const size_t rows = mesh.extents(0) * number_of_fields;
  const size_t angular_size = angular.synthesis.rows();
  const size_t harmonics = angular.synthesis.columns();
  DataVector radial_values{rows * harmonics};
  DataVector radial_derivatives{rows * harmonics};
  apply_radial(make_not_null(radial_values.data()),
               Spectral::modal_to_nodal_matrix(mesh.slice_through(0)), 'N',
               coefficients, number_of_fields * harmonics);
  apply_radial(make_not_null(radial_derivatives.data()),
               Spectral::differentiation_matrix(mesh.slice_through(0)), 'N',
               radial_values.data(), number_of_fields * harmonics);
  apply_angular(logical, radial_derivatives.data(), rows, angular.synthesis,
                'T', angular_size, harmonics, 0.0);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  apply_angular(make_not_null(logical.get() + rows * angular_size),
                radial_values.data(), rows, angular.trial_derivatives, 'T',
                2 * angular_size, harmonics, 0.0);
}

// sum_a (\hat D_a B)^T f_a for a batch of logical fluxes f_a stored as three
// consecutive nodal batches: the transpose of `logical_gradient`.
void logical_gradient_transpose(const gsl::not_null<double*> coefficients,
                                const double* const logical,
                                const size_t number_of_fields,
                                const Mesh<3>& mesh,
                                const ShellAngularOperators& angular) {
  const size_t rows = mesh.extents(0) * number_of_fields;
  const size_t angular_size = angular.synthesis.rows();
  const size_t harmonics = angular.synthesis.columns();
  DataVector radial_derivatives{rows * harmonics};
  DataVector radial_values{rows * harmonics};
  apply_angular(make_not_null(radial_derivatives.data()), logical, rows,
                angular.synthesis, 'N', harmonics, angular_size, 0.0);
  apply_radial(make_not_null(radial_values.data()),
               Spectral::differentiation_matrix(mesh.slice_through(0)), 'T',
               radial_derivatives.data(), number_of_fields * harmonics);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  apply_angular(make_not_null(radial_values.data()),
                logical + rows * angular_size, rows, angular.trial_derivatives,
                'N', harmonics, 2 * angular_size, 1.0);
  apply_radial(coefficients,
               Spectral::modal_to_nodal_matrix(mesh.slice_through(0)), 'T',
               radial_values.data(), number_of_fields * harmonics);
}

// Conversions between a batch and the columns of a matrix of nodal values
// (point i + N_r j) or of coefficients (row k + N_r n).
DataVector to_batch(const Matrix& columns, const size_t radial_size) {
  const size_t number_of_fields = columns.columns();
  const size_t angular_size = columns.rows() / radial_size;
  DataVector result{columns.rows() * number_of_fields};
  for (size_t j = 0; j < angular_size; ++j) {
    for (size_t field = 0; field < number_of_fields; ++field) {
      for (size_t i = 0; i < radial_size; ++i) {
        result[i + radial_size * (field + number_of_fields * j)] =
            columns(i + radial_size * j, field);
      }
    }
  }
  return result;
}

Matrix from_batch(const DataVector& batch, const size_t radial_size,
                  const size_t number_of_fields) {
  const size_t size = batch.size() / number_of_fields;
  const size_t angular_size = size / radial_size;
  Matrix result(size, number_of_fields);
  for (size_t j = 0; j < angular_size; ++j) {
    for (size_t field = 0; field < number_of_fields; ++field) {
      for (size_t i = 0; i < radial_size; ++i) {
        result(i + radial_size * j, field) =
            batch[i + radial_size * (field + number_of_fields * j)];
      }
    }
  }
  return result;
}

// Error unless the logical derivatives built from the angular operators
// reproduce `potential_gradients` on fixed pseudo-random fields.
void verify_logical_derivatives(
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const ShellAngularOperators& angular) {
  const size_t number_of_points = mesh.number_of_grid_points();
  const size_t radial_size = mesh.extents(0);
  const size_t angular_size = angular.synthesis.rows();
  const size_t rows = radial_size * number_of_potentials;
  Tags::PotentialRateTarget::type fields{number_of_points};
  std::mt19937 generator{2026};
  std::uniform_real_distribution<double> distribution{-1.0, 1.0};
  DataVector batch{number_of_points * number_of_potentials};
  for (size_t field = 0; field < number_of_potentials; ++field) {
    for (size_t point = 0; point < number_of_points; ++point) {
      fields.get(field)[point] = distribution(generator);
    }
  }
  for (size_t j = 0; j < angular_size; ++j) {
    for (size_t field = 0; field < number_of_potentials; ++field) {
      for (size_t i = 0; i < radial_size; ++i) {
        batch[i + rows * j + radial_size * field] =
            fields.get(field)[i + radial_size * j];
      }
    }
  }
  Tags::PotentialGradients::type gradients{number_of_points};
  potential_gradients(make_not_null(&gradients), fields, mesh,
                      inverse_jacobian);
  std::array<DataVector, 3> logical{};
  for (auto& derivative : logical) {
    derivative = DataVector{batch.size()};
  }
  apply_radial(make_not_null(logical[0].data()),
               Spectral::differentiation_matrix(mesh.slice_through(0)), 'N',
               batch.data(), number_of_potentials * angular_size);
  for (size_t a = 0; a < 2; ++a) {
    apply_angular(make_not_null(gsl::at(logical, a + 1).data()), batch.data(),
                  rows, gsl::at(angular.derivatives, a), 'T', angular_size,
                  angular_size, 0.0);
  }
  double largest_derivative = 0.0;
  double largest_difference = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t field = 0; field < number_of_potentials; ++field) {
      const DataVector& expected = gradients.get(i, field);
      for (size_t j = 0; j < angular_size; ++j) {
        for (size_t radial = 0; radial < radial_size; ++radial) {
          const size_t point = radial + radial_size * j;
          const size_t index = radial + rows * j + radial_size * field;
          double derivative = 0.0;
          for (size_t a = 0; a < 3; ++a) {
            derivative +=
                inverse_jacobian.get(a, i)[point] * gsl::at(logical, a)[index];
          }
          largest_derivative =
              std::max(largest_derivative, std::abs(expected[point]));
          largest_difference = std::max(largest_difference,
                                        std::abs(derivative - expected[point]));
        }
      }
    }
  }
  if (not(largest_difference <=
          shell_derivative_tolerance * largest_derivative)) {
    ERROR(
        "The shell derivatives built from the cached angular operators "
        "differ from partial_derivatives on the mesh "
        << mesh << ": largest difference " << largest_difference
        << ", largest derivative " << largest_derivative
        << ", relative tolerance " << shell_derivative_tolerance);
  }
}

// Apply the orthogonal factor Q of a QR factorization stored as LAPACK
// Householder reflectors to the columns of `values` (op = 'N' or 'T') from
// the side 'L' or 'R'.
void apply_reflectors(const gsl::not_null<double*> values,
                      const size_t leading_dimension, const char side,
                      const char op, const size_t rows, const size_t columns,
                      const Matrix& reflectors, const DataVector& factors) {
  const auto workspace_size = static_cast<blaze::blas_int_t>(
      64 * std::max(size_t{1}, side == 'L' ? columns : rows));
  std::vector<double> workspace(static_cast<size_t>(workspace_size));
  blaze::blas_int_t info = 0;
  blaze::ormqr(
      side, op, static_cast<blaze::blas_int_t>(rows),
      static_cast<blaze::blas_int_t>(columns),
      static_cast<blaze::blas_int_t>(reflectors.columns()), reflectors.data(),
      static_cast<blaze::blas_int_t>(reflectors.spacing()), factors.data(),
      values.get(), static_cast<blaze::blas_int_t>(leading_dimension),
      workspace.data(), workspace_size, &info);
  if (info != 0) {
    ERROR("Applying the shell kernel reflectors failed with info = " << info);
  }
}

// Solve with the lower-triangular `factor` or its transpose (op = 'N' or 'T').
void triangular_solve(const gsl::not_null<double*> values,
                      const size_t leading_dimension, const char op,
                      const Matrix& factor, const size_t columns) {
  blaze::blas_int_t info = 0;
  blaze::trtrs('L', op, 'N', static_cast<blaze::blas_int_t>(factor.rows()),
               static_cast<blaze::blas_int_t>(columns), factor.data(),
               static_cast<blaze::blas_int_t>(factor.spacing()), values.get(),
               static_cast<blaze::blas_int_t>(leading_dimension), &info);
  if (info != 0) {
    ERROR("Triangular solve with the shell Gram factor failed with info = "
          << info);
  }
}

// Solve with the lower-triangular `size` x `size` factor at `factor` or its
// transpose (op = 'N' or 'T').
void lower_triangular_solve(const gsl::not_null<double*> values,
                            const size_t leading_dimension, const char op,
                            const double* const factor, const size_t size,
                            const size_t factor_leading_dimension,
                            const size_t columns) {
  blaze::blas_int_t info = 0;
  blaze::trtrs('L', op, 'N', static_cast<blaze::blas_int_t>(size),
               static_cast<blaze::blas_int_t>(columns), factor,
               static_cast<blaze::blas_int_t>(factor_leading_dimension),
               values.get(), static_cast<blaze::blas_int_t>(leading_dimension),
               &info);
  if (info != 0) {
    ERROR(
        "Triangular solve with a shell velocity-filter factor failed with "
        "info = "
        << info);
  }
}

// Lower Cholesky factorization in place of the `size` x `size` matrix at
// `matrix`.
void lower_cholesky(const gsl::not_null<double*> matrix, const size_t size,
                    const size_t leading_dimension, const Mesh<3>& mesh,
                    const char* const description) {
  blaze::blas_int_t info = 0;
  blaze::potrf('L', static_cast<blaze::blas_int_t>(size), matrix.get(),
               static_cast<blaze::blas_int_t>(leading_dimension), &info);
  if (info != 0) {
    ERROR("The " << description << " of size " << size << " on the mesh "
                 << mesh
                 << " is not positive definite: Cholesky info = " << info);
  }
}

// Apply the Householder reflection I - 2 v v^T, with the unit vector
// v = `reflector`, to each column of `matrix`.
void reflect_columns(const gsl::not_null<Matrix*> matrix,
                     const DataVector& reflector) {
  ASSERT(matrix->rows() == reflector.size(),
         "Expected " << reflector.size() << " rows, got " << matrix->rows());
  for (size_t column = 0; column < matrix->columns(); ++column) {
    double projection = 0.0;
    for (size_t row = 0; row < matrix->rows(); ++row) {
      projection += reflector[row] * (*matrix)(row, column);
    }
    for (size_t row = 0; row < matrix->rows(); ++row) {
      (*matrix)(row, column) -= 2.0 * projection * reflector[row];
    }
  }
}

// Replace each column x of `values` by x - Q Q^T x, Q the orthonormal columns
// of `basis`. The dense products call BLAS through dgemm_: Blaze runs them
// with its own kernel or with whichever cblas_dgemm is linked first, which may
// be an unoptimized reference implementation, and LIBXSMM (dgemm_<true>) is
// meant for small matrices.
void subtract_projection(const gsl::not_null<Matrix*> values,
                         const Matrix& basis) {
  ASSERT(values->rows() == basis.rows(),
         "Expected " << basis.rows() << " rows, got " << values->rows());
  const size_t rows = basis.rows();
  const size_t modes = basis.columns();
  const size_t columns = values->columns();
  Matrix coefficients(modes, columns);
  dgemm_('T', 'N', modes, columns, rows, 1.0, basis.data(), basis.spacing(),
         values->data(), values->spacing(), 0.0, coefficients.data(),
         coefficients.spacing());
  dgemm_('N', 'N', rows, columns, modes, -1.0, basis.data(), basis.spacing(),
         coefficients.data(), coefficients.spacing(), 1.0, values->data(),
         values->spacing());
}

double max_abs(const Matrix& matrix) {
  double result = 0.0;
  for (size_t column = 0; column < matrix.columns(); ++column) {
    for (size_t row = 0; row < matrix.rows(); ++row) {
      result = std::max(result, std::abs(matrix(row, column)));
    }
  }
  return result;
}

void check_shell_angular_modes_to_kill(const Mesh<3>& mesh,
                                       const size_t angular_modes_to_kill) {
  if (angular_modes_to_kill + 1 > mesh.extents(1)) {
    ERROR("The number of removed angular bands ShellAngularModesToKill = "
          << angular_modes_to_kill
          << " of the shell velocity filter exceeds L = " << mesh.extents(1) - 1
          << " of the mesh " << mesh);
  }
}

void check_shell_filter_input(const Mesh<3>& mesh, const DataVector& mass,
                              const size_t angular_modes_to_kill) {
  if (mesh.basis(1) != Spectral::Basis::SphericalHarmonic or
      mesh.extents(1) < 3 or mesh.extents(2) != 2 * mesh.extents(1) - 1 or
      mass.size() != mesh.number_of_grid_points()) {
    ERROR(
        "The shell velocity filter requires a spherical-harmonic shell "
        "mesh (N_r, L + 1, 2 L + 1) with L >= 2 and one mass weight per "
        "grid point, but got the mesh "
        << mesh << " and " << mass.size() << " mass weights");
  }
  check_shell_angular_modes_to_kill(mesh, angular_modes_to_kill);
}

// The native filter I + Delta of `Structure` on the valid Spherepack
// coefficients, row and column c n_Y + n for the component c and the n-th
// valid coefficient `valid[n]`, with Delta from `fill_filter`.
template <typename Structure>
Matrix dense_native_filter(const gsl::not_null<SimpleSparseMatrix*> delta,
                           const size_t l_max,
                           const size_t angular_modes_to_kill,
                           const std::vector<size_t>& valid) {
  ylm::TensorYlm::fill_filter<Structure>(
      delta, l_max, angular_modes_to_kill, std::nullopt,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);
  const size_t components = Structure::size();
  const size_t spectral_size = ylm::Spherepack::spectral_size(l_max, l_max);
  const size_t harmonics = valid.size();
  const size_t storage_size = components * spectral_size;
  const size_t size = components * harmonics;
  std::vector<size_t> position(storage_size, size);
  for (size_t c = 0; c < components; ++c) {
    for (size_t n = 0; n < harmonics; ++n) {
      position[c * spectral_size + valid[n]] = c * harmonics + n;
    }
  }
  Matrix result(size, size, 0.0);
  std::vector<double> unit(storage_size, 0.0);
  std::vector<double> image(storage_size);
  for (size_t column = 0; column < size; ++column) {
    const size_t source =
        (column / harmonics) * spectral_size + valid[column % harmonics];
    unit[source] = 1.0;
    image = unit;
    delta->increment_multiply_on_right(make_not_null(&image), 0, 1, unit, 0, 1);
    unit[source] = 0.0;
    for (size_t row = 0; row < storage_size; ++row) {
      if (image[row] != 0.0) {
        if (position[row] == size) {
          ERROR("The native tensor spherical-harmonic filter for L = "
                << l_max << " with " << angular_modes_to_kill
                << " removed bands maps the Spherepack storage entry " << source
                << " to the unused storage entry " << row << " (value "
                << image[row] << ")");
        }
        result(position[row], column) = image[row];
      }
    }
  }
  return result;
}

// A W-orthonormal nodal basis of the range of the coefficient-space filter
// `filter` with `components` components, and the self-adjointness and
// projector errors, from the blocks of each angular degree.
void retained_space(const gsl::not_null<Matrix*> basis,
                    const gsl::not_null<double*> self_adjointness_error,
                    const gsl::not_null<double*> projector_error,
                    const Matrix& filter, const size_t components,
                    const std::vector<std::vector<size_t>>& degree_harmonics,
                    const std::vector<Matrix>& degree_gram_factors,
                    const Matrix& synthesis) {
  const size_t angular_size = synthesis.rows();
  const size_t harmonics = synthesis.columns();
  double antisymmetric_norm = 0.0;
  double norm = 0.0;
  *projector_error = 0.0;
  std::vector<Matrix> degree_bases{};
  size_t dimension = 0;
  for (size_t l = 0; l < degree_harmonics.size(); ++l) {
    const std::vector<size_t>& indices = degree_harmonics[l];
    const size_t degree_size = indices.size();
    const size_t size = components * degree_size;
    Matrix block(size, size);
    Matrix factor(size, size, 0.0);
    for (size_t c = 0; c < components; ++c) {
      for (size_t k = 0; k < degree_size; ++k) {
        for (size_t c2 = 0; c2 < components; ++c2) {
          for (size_t k2 = 0; k2 < degree_size; ++k2) {
            block(k + degree_size * c, k2 + degree_size * c2) = filter(
                indices[k] + harmonics * c, indices[k2] + harmonics * c2);
          }
        }
        for (size_t k2 = 0; k2 < degree_size; ++k2) {
          factor(k + degree_size * c, k2 + degree_size * c) =
              degree_gram_factors[l](k, k2);
        }
      }
    }
    // C^T F C^{-T}: the filter in coordinates orthonormal in W.
    Matrix weighted = blaze::trans(block);
    lower_triangular_solve(make_not_null(weighted.data()), weighted.spacing(),
                           'N', factor.data(), size, factor.spacing(), size);
    weighted = blaze::trans(factor) * blaze::trans(weighted);
    Matrix symmetric = 0.5 * (weighted + blaze::trans(weighted));
    for (size_t column = 0; column < size; ++column) {
      for (size_t row = 0; row < size; ++row) {
        norm += square(weighted(row, column));
        antisymmetric_norm +=
            square(weighted(row, column) - symmetric(row, column));
      }
    }
    blaze::DynamicVector<double> eigenvalues(size);
    blaze::syev(symmetric, eigenvalues, 'V', 'L');
    std::vector<size_t> retained{};
    for (size_t k = 0; k < size; ++k) {
      *projector_error = std::max(
          *projector_error,
          std::min(std::abs(eigenvalues[k]), std::abs(eigenvalues[k] - 1.0)));
      if (eigenvalues[k] > 0.5) {
        retained.push_back(k);
      }
    }
    // The coefficients C^{-T} V of the retained eigenvectors V, then their
    // nodal values.
    Matrix coefficients(size, retained.size());
    for (size_t a = 0; a < retained.size(); ++a) {
      for (size_t row = 0; row < size; ++row) {
        coefficients(row, a) = symmetric(row, retained[a]);
      }
    }
    lower_triangular_solve(make_not_null(coefficients.data()),
                           coefficients.spacing(), 'T', factor.data(), size,
                           factor.spacing(), retained.size());
    Matrix nodal(components * angular_size, retained.size(), 0.0);
    for (size_t a = 0; a < retained.size(); ++a) {
      for (size_t c = 0; c < components; ++c) {
        for (size_t k = 0; k < degree_size; ++k) {
          const double coefficient = coefficients(k + degree_size * c, a);
          for (size_t j = 0; j < angular_size; ++j) {
            nodal(j + angular_size * c, a) +=
                coefficient * synthesis(j, indices[k]);
          }
        }
      }
    }
    dimension += retained.size();
    degree_bases.push_back(std::move(nodal));
  }
  *self_adjointness_error =
      std::sqrt(antisymmetric_norm) / std::max(1.0, std::sqrt(norm));
  *basis = Matrix(components * angular_size, dimension);
  size_t column = 0;
  for (const Matrix& degree_basis : degree_bases) {
    blaze::submatrix(*basis, 0, column, components * angular_size,
                     degree_basis.columns()) = degree_basis;
    column += degree_basis.columns();
  }
}
}  // namespace

ShellAngularOperators::ShellAngularOperators(const size_t l_max_in)
    : l_max(l_max_in) {
  const ylm::Spherepack& ylm = ylm::get_spherepack_cache(l_max);
  const size_t angular_size = ylm.physical_size();
  const size_t harmonics = (l_max + 1) * (l_max + 1);

  synthesis = Matrix(angular_size, harmonics);
  {
    DataVector coefficients{ylm.spectral_size(), 0.0};
    size_t column = 0;
    for (ylm::SpherepackIterator iterator(l_max, l_max); iterator; ++iterator) {
      coefficients[iterator()] = 1.0;
      const DataVector values = ylm.spec_to_phys(coefficients);
      coefficients[iterator()] = 0.0;
      for (size_t j = 0; j < angular_size; ++j) {
        synthesis(j, column) = values[j];
      }
      ++column;
    }
    if (column != harmonics) {
      ERROR("Expected " << harmonics
                        << " valid Spherepack coefficients for L = " << l_max
                        << " but iterated over " << column);
    }
  }

  // The derivatives of all unit nodal vectors at once: ring `offset` of the
  // identity is the unit vector at the angular point `offset`.
  {
    DataVector identity{angular_size * angular_size, 0.0};
    for (size_t j = 0; j < angular_size; ++j) {
      identity[j + angular_size * j] = 1.0;
    }
    std::array<DataVector, 2> derivative{
        {DataVector{angular_size * angular_size},
         DataVector{angular_size * angular_size}}};
    ylm.gradient_all_offsets({{derivative[0].data(), derivative[1].data()}},
                             identity.data(), angular_size);
    for (size_t a = 0; a < 2; ++a) {
      Matrix& matrix = gsl::at(derivatives, a);
      matrix = Matrix(angular_size, angular_size);
      for (size_t column = 0; column < angular_size; ++column) {
        for (size_t j = 0; j < angular_size; ++j) {
          matrix(j, column) = gsl::at(derivative, a)[column + angular_size * j];
        }
      }
    }
  }
  trial_derivatives = Matrix(2 * angular_size, harmonics);
  for (size_t a = 0; a < 2; ++a) {
    blaze::submatrix(trial_derivatives, a * angular_size, 0, angular_size,
                     harmonics) = gsl::at(derivatives, a) * synthesis;
  }

  // The kernel of the angular derivatives within l <= L.
  Matrix decomposed = trial_derivatives;
  blaze::DynamicVector<double> singular_values{};
  Matrix right_singular_vectors_transpose{};
  blaze::gesvd(decomposed, singular_values, right_singular_vectors_transpose,
               'N', 'S');
  const double largest = singular_values[0];
  size_t rank = 0;
  while (rank < harmonics and
         singular_values[rank] > shell_kernel_tolerance * largest) {
    ++rank;
  }
  const size_t nullity = harmonics - rank;
  const size_t expected_nullity = 1 + (l_max % 2 == 0 ? l_max + 1 : l_max);
  if (nullity != expected_nullity or
      not(rank == 0 or nullity == 0 or
          singular_values[rank - 1] >=
              shell_kernel_minimum_gap * singular_values[rank])) {
    ERROR("The angular derivative kernel for L = "
          << l_max << " has dimension " << nullity << " instead of "
          << expected_nullity << ", or an ambiguous gap below "
          << shell_kernel_minimum_gap << " (threshold "
          << shell_kernel_tolerance
          << " times the largest singular value). Singular values: "
          << singular_values);
  }
  kernel = blaze::trans(blaze::submatrix(right_singular_vectors_transpose, rank,
                                         0, nullity, harmonics));
}

const ShellAngularOperators& shell_angular_operators(const size_t l_max) {
  static const auto cache = make_static_cache<CacheRange<2, 151>>(
      [](const size_t l) { return ShellAngularOperators{l}; });
  return cache(l_max);
}

ShellFilterAngularSpaces::ShellFilterAngularSpaces(
    const size_t l_max_in, const size_t angular_modes_to_kill_in)
    : l_max(l_max_in), angular_modes_to_kill(angular_modes_to_kill_in) {
  if (l_max < 2 or angular_modes_to_kill > l_max) {
    ERROR(
        "The shell velocity filter requires L >= 2 and at most L removed "
        "angular bands, but got L = "
        << l_max << " and " << angular_modes_to_kill << " removed bands");
  }
  const ylm::Spherepack& ylm = ylm::get_spherepack_cache(l_max);
  const Matrix& synthesis = shell_angular_operators(l_max).synthesis;
  const size_t angular_size = synthesis.rows();
  const size_t harmonics = synthesis.columns();
  const std::vector<double>& weights = ylm.integration_weights();

  std::vector<size_t> valid{};
  std::vector<size_t> degrees{};
  std::vector<std::vector<size_t>> degree_harmonics(l_max + 1);
  for (ylm::SpherepackIterator iterator(l_max, l_max); iterator; ++iterator) {
    degree_harmonics[iterator.l()].push_back(valid.size());
    degrees.push_back(iterator.l());
    valid.push_back(iterator());
  }

  // The Gram matrix G = S^T W S, which is block diagonal in the degree.
  Matrix weighted_synthesis = synthesis;
  for (size_t n = 0; n < harmonics; ++n) {
    for (size_t j = 0; j < angular_size; ++j) {
      weighted_synthesis(j, n) *= weights[j];
    }
  }
  const Matrix gram = blaze::trans(synthesis) * weighted_synthesis;
  double gram_coupling = 0.0;
  for (size_t n = 0; n < harmonics; ++n) {
    for (size_t n2 = 0; n2 < harmonics; ++n2) {
      if (degrees[n] != degrees[n2]) {
        gram_coupling = std::max(gram_coupling, std::abs(gram(n, n2)));
      }
    }
  }
  if (not(gram_coupling <= shell_filter_tolerance * max_abs(gram))) {
    ERROR(
        "The spherical harmonics of different degrees are not orthogonal "
        "in the Spherepack integration weights for L = "
        << l_max << ": largest Gram entry between degrees " << gram_coupling
        << ", largest Gram entry " << max_abs(gram));
  }
  std::vector<Matrix> degree_gram_factors{};
  for (size_t l = 0; l <= l_max; ++l) {
    const std::vector<size_t>& indices = degree_harmonics[l];
    Matrix factor(indices.size(), indices.size());
    for (size_t k = 0; k < indices.size(); ++k) {
      for (size_t k2 = 0; k2 < indices.size(); ++k2) {
        factor(k, k2) = gram(indices[k], indices[k2]);
      }
    }
    blaze::blas_int_t info = 0;
    blaze::potrf('L', static_cast<blaze::blas_int_t>(indices.size()),
                 factor.data(),
                 static_cast<blaze::blas_int_t>(factor.spacing()), &info);
    if (info != 0) {
      ERROR("The Gram matrix of the spherical harmonics of degree "
            << l << " for L = " << l_max
            << " is not positive definite: Cholesky info = " << info);
    }
    for (size_t k2 = 1; k2 < indices.size(); ++k2) {
      for (size_t k = 0; k < k2; ++k) {
        factor(k, k2) = 0.0;
      }
    }
    degree_gram_factors.push_back(std::move(factor));
  }

  // The Spherepack analysis A inverts the synthesis and is G^{-1} S^T W.
  {
    DataVector identity{angular_size * angular_size, 0.0};
    for (size_t j = 0; j < angular_size; ++j) {
      identity[j + angular_size * j] = 1.0;
    }
    DataVector spectral{ylm.spectral_size() * angular_size};
    const double* const identity_data = identity.data();
    ylm.phys_to_spec_all_offsets(make_not_null(spectral.data()),
                                 make_not_null(identity_data), angular_size);
    Matrix analysis(harmonics, angular_size);
    for (size_t n = 0; n < harmonics; ++n) {
      for (size_t j = 0; j < angular_size; ++j) {
        analysis(n, j) = spectral[j + angular_size * valid[n]];
      }
    }
    Matrix inversion_error = analysis * synthesis;
    for (size_t n = 0; n < harmonics; ++n) {
      inversion_error(n, n) -= 1.0;
    }
    const Matrix projection_error =
        gram * analysis - blaze::trans(weighted_synthesis);
    if (not(max_abs(inversion_error) <= shell_filter_tolerance and
            max_abs(projection_error) <=
                shell_filter_tolerance * max_abs(weighted_synthesis))) {
      ERROR("The Spherepack analysis for L = "
            << l_max
            << " is not the weighted orthogonal projection onto the "
               "harmonics: largest deviation of analysis * synthesis from "
               "the identity "
            << max_abs(inversion_error)
            << ", largest deviation of Gram * analysis from the weighted "
               "synthesis transpose "
            << max_abs(projection_error) << " (relative to "
            << max_abs(weighted_synthesis) << ")");
    }
  }

  // The native filters on the valid coefficients; rank two in the
  // trace-free coordinates of stf_encode.
  std::array<Matrix, 3> filters{};
  filters[0] = dense_native_filter<Scalar<DataVector>::structure>(
      make_not_null(&native_filters[0]), l_max, angular_modes_to_kill, valid);
  filters[1] = dense_native_filter<tnsr::i<DataVector, 3>::structure>(
      make_not_null(&native_filters[1]), l_max, angular_modes_to_kill, valid);
  const Matrix symmetric_filter =
      dense_native_filter<tnsr::ii<DataVector, 3>::structure>(
          make_not_null(&native_filters[2]), l_max, angular_modes_to_kill,
          valid);
  {
    using structure = tnsr::ii<DataVector, 3>::structure;
    constexpr auto tensor_indices = structure::storage_to_tensor_index();
    stf_decode_matrix = Matrix(6, 5);
    stf_encode_matrix = Matrix(5, 6);
    for (size_t a = 0; a < 5; ++a) {
      std::array<double, 5> unit{};
      gsl::at(unit, a) = 1.0;
      const Matrix3 decoded = stf_decode(unit.data());
      for (size_t s = 0; s < 6; ++s) {
        stf_decode_matrix(s, a) =
            decoded(tensor_indices[s][0], tensor_indices[s][1]);
        stf_encode_matrix(a, s) =
            static_cast<double>(structure::multiplicity(s)) *
            stf_decode_matrix(s, a);
      }
    }
    Matrix round_trip = stf_encode_matrix * stf_decode_matrix;
    for (size_t a = 0; a < 5; ++a) {
      round_trip(a, a) -= 1.0;
    }
    if (not(max_abs(round_trip) <= 1.0e-14)) {
      ERROR(
          "The trace-free coordinates are not orthonormal in the Frobenius "
          "norm: largest deviation "
          << max_abs(round_trip));
    }
    Matrix decoded_filter(6 * harmonics, 5 * harmonics, 0.0);
    for (size_t b = 0; b < 5; ++b) {
      for (size_t s = 0; s < 6; ++s) {
        const double decode = stf_decode_matrix(s, b);
        if (decode != 0.0) {
          blaze::submatrix(decoded_filter, 0, b * harmonics, 6 * harmonics,
                           harmonics) +=
              decode * blaze::submatrix(symmetric_filter, 0, s * harmonics,
                                        6 * harmonics, harmonics);
        }
      }
    }
    filters[2] = Matrix(5 * harmonics, 5 * harmonics, 0.0);
    for (size_t a = 0; a < 5; ++a) {
      for (size_t s = 0; s < 6; ++s) {
        const double encode = stf_encode_matrix(a, s);
        if (encode != 0.0) {
          blaze::submatrix(filters[2], a * harmonics, 0, harmonics,
                           5 * harmonics) +=
              encode * blaze::submatrix(decoded_filter, s * harmonics, 0,
                                        harmonics, 5 * harmonics);
        }
      }
    }
  }

  for (size_t rank = 0; rank < 3; ++rank) {
    const Matrix& filter = gsl::at(filters, rank);
    const size_t components = filter.rows() / harmonics;
    double coupling = 0.0;
    for (size_t column = 0; column < filter.columns(); ++column) {
      for (size_t row = 0; row < filter.rows(); ++row) {
        if (degrees[row % harmonics] != degrees[column % harmonics]) {
          coupling = std::max(coupling, std::abs(filter(row, column)));
        }
      }
    }
    if (not(coupling <= shell_filter_tolerance * max_abs(filter))) {
      ERROR("The native rank-"
            << rank << " tensor spherical-harmonic filter for L = " << l_max
            << " couples different degrees: "
            << "largest coupling " << coupling);
    }
    retained_space(make_not_null(&gsl::at(retained_bases, rank)),
                   make_not_null(&gsl::at(self_adjointness_errors, rank)),
                   make_not_null(&gsl::at(projector_errors, rank)), filter,
                   components, degree_harmonics, degree_gram_factors,
                   synthesis);
    if (not(gsl::at(self_adjointness_errors, rank) <= shell_filter_tolerance and
            gsl::at(projector_errors, rank) <= shell_filter_tolerance)) {
      ERROR("The native rank-"
            << rank << " tensor spherical-harmonic filter for L = " << l_max
            << " with " << angular_modes_to_kill
            << " removed bands is not an orthogonal projector in the "
               "Spherepack integration weights: self-adjointness error "
            << gsl::at(self_adjointness_errors, rank)
            << ", largest eigenvalue distance from 0 or 1 "
            << gsl::at(projector_errors, rank) << ", tolerance "
            << shell_filter_tolerance);
    }
  }
}

const ShellFilterAngularSpaces& shell_filter_angular_spaces(
    const size_t l_max, const size_t angular_modes_to_kill) {
  static std::mutex mutex{};
  static std::map<std::pair<size_t, size_t>,
                  std::unique_ptr<const ShellFilterAngularSpaces>>
      cache{};
  const std::lock_guard lock{mutex};
  auto& entry = cache[std::pair{l_max, angular_modes_to_kill}];
  if (entry == nullptr) {
    entry = std::make_unique<const ShellFilterAngularSpaces>(
        l_max, angular_modes_to_kill);
  }
  return *entry;
}

GeneralShellFilter::GeneralShellFilter(const Mesh<3>& mesh,
                                       const DataVector& mass,
                                       const size_t angular_modes_to_kill)
    : mesh_(mesh), angular_modes_to_kill_(angular_modes_to_kill), mass_(mass) {
  check_shell_filter_input(mesh, mass, angular_modes_to_kill);
  const size_t radial_size = mesh.extents(0);
  spaces_ =
      &shell_filter_angular_spaces(mesh.extents(1) - 1, angular_modes_to_kill);
  const ShellFilterAngularSpaces& spaces = *spaces_;
  const Matrix& nodal_to_modal =
      Spectral::nodal_to_modal_matrix(mesh.slice_through(0));
  top_radial_mode_ = DataVector{radial_size};
  for (size_t i = 0; i < radial_size; ++i) {
    top_radial_mode_[i] = nodal_to_modal(radial_size - 1, i);
  }
  const DataVector sqrt_mass = sqrt(mass);
  for (size_t rank = 0; rank < 3; ++rank) {
    const Matrix& basis = gsl::at(spaces.retained_bases, rank);
    const size_t dimension = basis.columns();
    const size_t rows = basis.rows();
    const size_t angular_size = mass.size() / radial_size;
    Matrix& factors = gsl::at(factors_, rank);
    factors = Matrix(dimension, (radial_size + 1) * dimension, 0.0);
    Matrix weighted_basis(rows, dimension);
    Matrix h(dimension, dimension, 0.0);
    for (size_t i = 0; i < radial_size; ++i) {
      for (size_t a = 0; a < dimension; ++a) {
        for (size_t row = 0; row < rows; ++row) {
          weighted_basis(row, a) =
              sqrt_mass[i + radial_size * (row % angular_size)] * basis(row, a);
        }
      }
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      double* const block = factors.data() + i * dimension * factors.spacing();
      dgemm_('T', 'N', dimension, dimension, rows, 1.0, weighted_basis.data(),
             weighted_basis.spacing(), weighted_basis.data(),
             weighted_basis.spacing(), 0.0, block, factors.spacing());
      lower_cholesky(make_not_null(block), dimension, factors.spacing(), mesh,
                     "angular mass matrix of the retained angular space");
      Matrix inverse(dimension, dimension, 0.0);
      for (size_t a = 0; a < dimension; ++a) {
        inverse(a, a) = 1.0;
      }
      lower_triangular_solve(make_not_null(inverse.data()), inverse.spacing(),
                             'N', block, dimension, factors.spacing(),
                             dimension);
      lower_triangular_solve(make_not_null(inverse.data()), inverse.spacing(),
                             'T', block, dimension, factors.spacing(),
                             dimension);
      h += square(top_radial_mode_[i]) * inverse;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    double* const h_block =
        factors.data() + radial_size * dimension * factors.spacing();
    for (size_t a = 0; a < dimension; ++a) {
      for (size_t b = 0; b < dimension; ++b) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        h_block[a + factors.spacing() * b] = 0.5 * (h(a, b) + h(b, a));
      }
    }
    lower_cholesky(make_not_null(h_block), dimension, factors.spacing(), mesh,
                   "mass matrix of the top radial degree");
  }
}

void GeneralShellFilter::apply(const ShellFilterFields& fields,
                               const double gain) const {
  apply_rank({fields[0].get(), fields[1].get(), fields[2].get(),
              fields[3].get(), fields[4].get()},
             2, gain);
  apply_rank({fields[5].get(), fields[6].get()}, 0, gain);
  apply_rank({fields[7].get(), fields[8].get(), fields[9].get(),
              fields[10].get(), fields[11].get(), fields[12].get()},
             1, gain);
}

void GeneralShellFilter::apply_rank(const std::vector<DataVector*>& fields,
                                    const size_t rank,
                                    const double gain) const {
  const size_t number_of_points = mesh_.number_of_grid_points();
  const size_t radial_size = mesh_.extents(0);
  const size_t angular_size = number_of_points / radial_size;
  const Matrix& basis = gsl::at(spaces_->retained_bases, rank);
  const Matrix& factors = gsl::at(factors_, rank);
  const size_t dimension = basis.columns();
  const size_t components = basis.rows() / angular_size;
  const size_t groups = fields.size() / components;
  const size_t rows = groups * radial_size;
  const size_t spacing = factors.spacing();
  const auto block = [&factors, dimension, spacing](const size_t i) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return factors.data() + i * dimension * spacing;
  };
  // The weighted values M v, row g + G i for the group g at the radial
  // point i, and their angular coordinates c_i = A_i^{-1} Phi^T M_i v_i,
  // column g + G i.
  Matrix values(rows, basis.rows());
  for (size_t g = 0; g < groups; ++g) {
    for (size_t c = 0; c < components; ++c) {
      const DataVector& field = *fields[c + components * g];
      ASSERT(field.size() == number_of_points, "Expected " << number_of_points
                                                           << " points but got "
                                                           << field.size());
      for (size_t j = 0; j < angular_size; ++j) {
        for (size_t i = 0; i < radial_size; ++i) {
          const size_t point = i + radial_size * j;
          values(g + groups * i, j + angular_size * c) =
              mass_[point] * field[point];
        }
      }
    }
  }
  Matrix coordinates(dimension, rows);
  dgemm_('T', 'T', dimension, rows, basis.rows(), 1.0, basis.data(),
         basis.spacing(), values.data(), values.spacing(), 0.0,
         coordinates.data(), coordinates.spacing());
  for (size_t i = 0; i < radial_size; ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    double* const radial_coordinates =
        coordinates.data() + groups * i * coordinates.spacing();
    lower_triangular_solve(make_not_null(radial_coordinates),
                           coordinates.spacing(), 'N', block(i), dimension,
                           spacing, groups);
    lower_triangular_solve(make_not_null(radial_coordinates),
                           coordinates.spacing(), 'T', block(i), dimension,
                           spacing, groups);
  }
  if (gain != 1.0) {
    // Subtract (1 - g) times the part t_i A_i^{-1} H^{-1} sum_k t_k c_k in
    // the complement of S_0.
    Matrix top(dimension, groups, 0.0);
    for (size_t i = 0; i < radial_size; ++i) {
      top += top_radial_mode_[i] *
             blaze::submatrix(coordinates, 0, groups * i, dimension, groups);
    }
    lower_triangular_solve(make_not_null(top.data()), top.spacing(), 'N',
                           block(radial_size), dimension, spacing, groups);
    lower_triangular_solve(make_not_null(top.data()), top.spacing(), 'T',
                           block(radial_size), dimension, spacing, groups);
    Matrix correction(dimension, groups);
    for (size_t i = 0; i < radial_size; ++i) {
      correction = top;
      lower_triangular_solve(make_not_null(correction.data()),
                             correction.spacing(), 'N', block(i), dimension,
                             spacing, groups);
      lower_triangular_solve(make_not_null(correction.data()),
                             correction.spacing(), 'T', block(i), dimension,
                             spacing, groups);
      blaze::submatrix(coordinates, 0, groups * i, dimension, groups) -=
          ((1.0 - gain) * top_radial_mode_[i]) * correction;
    }
  }
  dgemm_('T', 'T', rows, basis.rows(), dimension, 1.0, coordinates.data(),
         coordinates.spacing(), basis.data(), basis.spacing(), 0.0,
         values.data(), values.spacing());
  for (size_t g = 0; g < groups; ++g) {
    for (size_t c = 0; c < components; ++c) {
      DataVector& field = *fields[c + components * g];
      for (size_t j = 0; j < angular_size; ++j) {
        for (size_t i = 0; i < radial_size; ++i) {
          field[i + radial_size * j] =
              values(g + groups * i, j + angular_size * c);
        }
      }
    }
  }
}

void GeneralShellFilter::pup(PUP::er& p) {
  p | mesh_;
  p | angular_modes_to_kill_;
  p | mass_;
  p | top_radial_mode_;
  p | factors_;
  if (p.isUnpacking() and mass_.size() != 0) {
    spaces_ = &shell_filter_angular_spaces(mesh_.extents(1) - 1,
                                           angular_modes_to_kill_);
  }
}

bool operator==(const GeneralShellFilter& lhs, const GeneralShellFilter& rhs) {
  return lhs.mesh_ == rhs.mesh_ and
         lhs.angular_modes_to_kill_ == rhs.angular_modes_to_kill_ and
         lhs.mass_ == rhs.mass_ and
         lhs.top_radial_mode_ == rhs.top_radial_mode_ and
         lhs.factors_ == rhs.factors_;
}

bool operator!=(const GeneralShellFilter& lhs, const GeneralShellFilter& rhs) {
  return not(lhs == rhs);
}

SeparableShellFilter::SeparableShellFilter(const Mesh<3>& mesh,
                                           const DataVector& mass,
                                           const size_t angular_modes_to_kill)
    : mesh_(mesh), angular_modes_to_kill_(angular_modes_to_kill) {
  check_shell_filter_input(mesh, mass, angular_modes_to_kill);
  const size_t radial_size = mesh.extents(0);
  const size_t l_max = mesh.extents(1) - 1;
  // Builds and checks the angular spaces.
  spaces_ = &shell_filter_angular_spaces(l_max, angular_modes_to_kill);
  const std::vector<double>& angular_weights =
      ylm::get_spherepack_cache(l_max).integration_weights();
  const size_t angular_size = angular_weights.size();
  radial_weights_ = DataVector{radial_size, 0.0};
  for (size_t j = 0; j < angular_size; ++j) {
    for (size_t i = 0; i < radial_size; ++i) {
      radial_weights_[i] += mass[i + radial_size * j] / angular_weights[j];
    }
  }
  radial_weights_ /= static_cast<double>(angular_size);
  const Matrix& nodal_to_modal =
      Spectral::nodal_to_modal_matrix(mesh.slice_through(0));
  top_radial_mode_ = DataVector{radial_size};
  for (size_t i = 0; i < radial_size; ++i) {
    top_radial_mode_[i] =
        nodal_to_modal(radial_size - 1, i) / radial_weights_[i];
  }
  top_radial_mode_ /=
      std::sqrt(sum(radial_weights_ * square(top_radial_mode_)));
}

void SeparableShellFilter::apply(const ShellFilterFields& fields,
                                 const double gain) const {
  apply_rank({fields[0].get(), fields[1].get(), fields[2].get(),
              fields[3].get(), fields[4].get()},
             2, gain);
  apply_rank({fields[5].get(), fields[6].get()}, 0, gain);
  apply_rank({fields[7].get(), fields[8].get(), fields[9].get(),
              fields[10].get(), fields[11].get(), fields[12].get()},
             1, gain);
}

void SeparableShellFilter::apply_rank(const std::vector<DataVector*>& fields,
                                      const size_t rank,
                                      const double gain) const {
  const size_t number_of_points = mesh_.number_of_grid_points();
  const size_t radial_size = mesh_.extents(0);
  const size_t angular_size = number_of_points / radial_size;
  const size_t l_max = mesh_.extents(1) - 1;
  const ylm::Spherepack& ylm = ylm::get_spherepack_cache(l_max);
  const ShellFilterAngularSpaces& spaces = *spaces_;
  const SimpleSparseMatrix& native_filter =
      gsl::at(spaces.native_filters, rank);
  const size_t components = rank == 2 ? 5 : (rank == 1 ? 3 : 1);
  const size_t storage_components = rank == 2 ? 6 : components;
  const size_t groups = fields.size() / components;
  const size_t spectral_points = radial_size * ylm.spectral_size();
  std::vector<double> nodal(storage_components * number_of_points);
  std::vector<double> spectral(storage_components * spectral_points);
  std::vector<double> filtered(storage_components * spectral_points);
  for (size_t g = 0; g < groups; ++g) {
    // The native filter on the Cartesian (rank two: tnsr::ii) storage.
    for (size_t s = 0; s < storage_components; ++s) {
      for (size_t point = 0; point < number_of_points; ++point) {
        if (rank == 2) {
          double value = 0.0;
          for (size_t a = 0; a < components; ++a) {
            value += spaces.stf_decode_matrix(s, a) *
                     (*fields[a + components * g])[point];
          }
          nodal[point + number_of_points * s] = value;
        } else {
          nodal[point + number_of_points * s] =
              (*fields[s + components * g])[point];
        }
      }
      ylm.phys_to_spec_all_offsets(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          make_not_null(spectral.data() + spectral_points * s),
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          make_not_null(nodal.data() + number_of_points * s), radial_size);
    }
    filtered = spectral;
    for (size_t i = 0; i < radial_size; ++i) {
      native_filter.increment_multiply_on_right(
          make_not_null(&filtered), i, radial_size, spectral, i, radial_size);
    }
    for (size_t s = 0; s < storage_components; ++s) {
      ylm.spec_to_phys_all_offsets(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          make_not_null(nodal.data() + number_of_points * s),
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          make_not_null(filtered.data() + spectral_points * s), radial_size);
    }
    for (size_t c = 0; c < components; ++c) {
      DataVector& field = *fields[c + components * g];
      for (size_t point = 0; point < number_of_points; ++point) {
        if (rank == 2) {
          double value = 0.0;
          for (size_t s = 0; s < storage_components; ++s) {
            value += spaces.stf_encode_matrix(c, s) *
                     nodal[point + number_of_points * s];
          }
          field[point] = value;
        } else {
          field[point] = nodal[point + number_of_points * c];
        }
      }
      // The radial filter I - (1 - g) q (rho q)^T along each radial line.
      if (gain != 1.0) {
        for (size_t j = 0; j < angular_size; ++j) {
          double projection = 0.0;
          for (size_t i = 0; i < radial_size; ++i) {
            projection += radial_weights_[i] * top_radial_mode_[i] *
                          field[i + radial_size * j];
          }
          for (size_t i = 0; i < radial_size; ++i) {
            field[i + radial_size * j] -=
                (1.0 - gain) * projection * top_radial_mode_[i];
          }
        }
      }
    }
  }
}

void SeparableShellFilter::pup(PUP::er& p) {
  p | mesh_;
  p | angular_modes_to_kill_;
  p | radial_weights_;
  p | top_radial_mode_;
  if (p.isUnpacking() and radial_weights_.size() != 0) {
    spaces_ = &shell_filter_angular_spaces(mesh_.extents(1) - 1,
                                           angular_modes_to_kill_);
  }
}

bool operator==(const SeparableShellFilter& lhs,
                const SeparableShellFilter& rhs) {
  return lhs.mesh_ == rhs.mesh_ and
         lhs.angular_modes_to_kill_ == rhs.angular_modes_to_kill_ and
         lhs.radial_weights_ == rhs.radial_weights_ and
         lhs.top_radial_mode_ == rhs.top_radial_mode_;
}

bool operator!=(const SeparableShellFilter& lhs,
                const SeparableShellFilter& rhs) {
  return not(lhs == rhs);
}

LocalOperators::LocalOperators(
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const Scalar<DataVector>& det_inverse_jacobian,
    const size_t shell_angular_modes_to_kill)
    : mesh_(mesh),
      spherical_shell_(mesh.basis(1) == Spectral::Basis::SphericalHarmonic),
      shell_angular_modes_to_kill_(shell_angular_modes_to_kill) {
  if (spherical_shell_) {
    if (mesh.basis(0) != Spectral::Basis::Legendre or
        mesh.quadrature(0) != Spectral::Quadrature::GaussLobatto or
        mesh.basis(2) != Spectral::Basis::SphericalHarmonic or
        mesh.quadrature(1) != Spectral::Quadrature::Gauss or
        mesh.quadrature(2) != Spectral::Quadrature::Equiangular) {
      ERROR(
          "The CCZ4 gradient projection on a spherical-harmonic shell "
          "requires a Legendre Gauss-Lobatto radial dimension and a "
          "SphericalHarmonic basis with Gauss and Equiangular quadratures in "
          "the angular dimensions, but got "
          << mesh);
    }
    if (mesh.extents(0) < 2 or mesh.extents(1) < 3 or
        mesh.extents(2) != 2 * mesh.extents(1) - 1) {
      ERROR(
          "The CCZ4 gradient projection on a spherical-harmonic shell "
          "requires extents (N_r, L + 1, 2 L + 1) with N_r >= 2 and L >= 2, "
          "but got "
          << mesh);
    }
    check_shell_angular_modes_to_kill(mesh, shell_angular_modes_to_kill);
  } else {
    for (size_t d = 0; d < 3; ++d) {
      if (mesh.basis(d) != Spectral::Basis::Legendre or
          mesh.quadrature(d) != Spectral::Quadrature::GaussLobatto) {
        ERROR(
            "The CCZ4 gradient projection requires a Legendre "
            "Gauss-Lobatto mesh or a spherical-harmonic shell, but got "
            << mesh);
      }
      if (mesh.extents(d) < 2) {
        ERROR(
            "The CCZ4 gradient projection requires at least two points per "
            "dimension, but got "
            << mesh);
      }
    }
  }
  const size_t number_of_points = mesh.number_of_grid_points();

  mass_ = DataVector{number_of_points};
  if (spherical_shell_) {
    // Mass weights m = w_r w_Omega / det(J^{-1}); the Spherepack weights
    // contain the angular measure.
    const size_t radial_size = mesh.extents(0);
    const DataVector& radial_weights =
        Spectral::quadrature_weights(mesh.slice_through(0));
    const std::vector<double>& angular_weights =
        ylm::get_spherepack_cache(mesh.extents(1) - 1).integration_weights();
    for (size_t j = 0; j < angular_weights.size(); ++j) {
      for (size_t i = 0; i < radial_size; ++i) {
        const size_t index = i + radial_size * j;
        mass_[index] = radial_weights[i] * angular_weights[j] /
                       get(det_inverse_jacobian)[index];
      }
    }
  } else {
    // Mass weights m = J w_1 w_2 w_3.
    const std::array<size_t, 3> extents{
        {mesh.extents(0), mesh.extents(1), mesh.extents(2)}};
    for (size_t k = 0; k < extents[2]; ++k) {
      for (size_t j = 0; j < extents[1]; ++j) {
        for (size_t i = 0; i < extents[0]; ++i) {
          const size_t index = i + extents[0] * (j + extents[1] * k);
          mass_[index] =
              Spectral::quadrature_weights(mesh.slice_through(0))[i] *
              Spectral::quadrature_weights(mesh.slice_through(1))[j] *
              Spectral::quadrature_weights(mesh.slice_through(2))[k] /
              get(det_inverse_jacobian)[index];
        }
      }
    }
  }
  if (not(min(mass_) > 0.0 and std::isfinite(max(mass_)))) {
    ERROR(
        "The coordinate Jacobian determinant must be positive and the mass "
        "weights finite, but the mass weights range over ["
        << min(mass_) << ", " << max(mass_) << "] on the mesh " << mesh);
  }
  sqrt_mass_ = sqrt(mass_);
  volume_ = sum(mass_);
  for (size_t a = 0; a < 3; ++a) {
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(inverse_jacobian_, a + 3 * i) = inverse_jacobian.get(a, i);
    }
  }
  if (spherical_shell_) {
    build_spherical_shell(mesh, inverse_jacobian);
    general_shell_filter_ =
        GeneralShellFilter{mesh, mass_, shell_angular_modes_to_kill};
    separable_shell_filter_ =
        SeparableShellFilter{mesh, mass_, shell_angular_modes_to_kill};
    // The largest relative deviation of the mass from rho(i) w_Omega(j),
    // which decides the shell-filter dispatch and bounds the difference of
    // the two evaluations.
    const std::vector<double>& angular_weights =
        ylm::get_spherepack_cache(mesh.extents(1) - 1).integration_weights();
    const size_t radial_size = mesh.extents(0);
    DataVector radial_mass{radial_size, 0.0};
    for (size_t j = 0; j < angular_weights.size(); ++j) {
      for (size_t i = 0; i < radial_size; ++i) {
        radial_mass[i] += mass_[i + radial_size * j] / angular_weights[j];
      }
    }
    radial_mass /= static_cast<double>(angular_weights.size());
    shell_mass_separability_residual_ = 0.0;
    for (size_t j = 0; j < angular_weights.size(); ++j) {
      for (size_t i = 0; i < radial_size; ++i) {
        shell_mass_separability_residual_ =
            std::max(shell_mass_separability_residual_,
                     std::abs(mass_[i + radial_size * j] / angular_weights[j] -
                              radial_mass[i]) /
                         radial_mass[i]);
      }
    }
  } else {
    build_tensor_product(mesh);
  }
}

void LocalOperators::build_tensor_product(const Mesh<3>& mesh) {
  const size_t number_of_points = mesh.number_of_grid_points();
  const std::array<size_t, 3> extents{
      {mesh.extents(0), mesh.extents(1), mesh.extents(2)}};

  // Mass-weighted QR of the Legendre Vandermonde matrix with columns ordered
  // (constant, remaining retained modes, removed shell). Its trailing columns
  // Q_high, the only ones kept, are an orthonormal basis of the orthogonal
  // complement of the weighted retained modes.
  const std::array<const Matrix*, 3> vandermonde{
      {&Spectral::modal_to_nodal_matrix(mesh.slice_through(0)),
       &Spectral::modal_to_nodal_matrix(mesh.slice_through(1)),
       &Spectral::modal_to_nodal_matrix(mesh.slice_through(2))}};
  std::vector<std::array<size_t, 3>> modes{};
  std::vector<std::array<size_t, 3>> shell{};
  for (size_t n = 0; n < extents[2]; ++n) {
    for (size_t m = 0; m < extents[1]; ++m) {
      for (size_t l = 0; l < extents[0]; ++l) {
        if (l + 1 == extents[0] or m + 1 == extents[1] or n + 1 == extents[2]) {
          shell.push_back({{l, m, n}});
        } else {
          modes.push_back({{l, m, n}});
        }
      }
    }
  }
  // The constant mode (0, 0, 0) is the first retained mode.
  number_of_retained_modes_ = modes.size();
  modes.insert(modes.end(), shell.begin(), shell.end());
  Matrix weighted_basis(number_of_points, number_of_points);
  for (size_t column = 0; column < number_of_points; ++column) {
    const auto& [l, m, n] = modes[column];
    for (size_t k = 0; k < extents[2]; ++k) {
      for (size_t j = 0; j < extents[1]; ++j) {
        for (size_t i = 0; i < extents[0]; ++i) {
          const size_t index = i + extents[0] * (j + extents[1] * k);
          weighted_basis(index, column) =
              sqrt_mass_[index] * (*vandermonde[0])(i, l) *
              (*vandermonde[1])(j, m) * (*vandermonde[2])(k, n);
        }
      }
    }
  }
  Matrix orthonormal_basis{};
  Matrix upper_triangular{};
  blaze::qr(weighted_basis, orthonormal_basis, upper_triangular);
  removed_basis_ = blaze::submatrix(
      orthonormal_basis, 0, number_of_retained_modes_, number_of_points,
      number_of_points - number_of_retained_modes_);

  // Dense ordinary derivative matrices D_i.
  std::array<Matrix, 3> logical_derivatives{};
  for (size_t a = 0; a < 3; ++a) {
    const Matrix& d_1d =
        Spectral::differentiation_matrix(mesh.slice_through(a));
    Matrix& d = gsl::at(logical_derivatives, a);
    d = Matrix(number_of_points, number_of_points, 0.0);
    for (size_t k = 0; k < extents[2]; ++k) {
      for (size_t j = 0; j < extents[1]; ++j) {
        for (size_t i = 0; i < extents[0]; ++i) {
          const std::array<size_t, 3> point{{i, j, k}};
          const size_t row = i + extents[0] * (j + extents[1] * k);
          for (size_t s = 0; s < gsl::at(extents, a); ++s) {
            std::array<size_t, 3> other = point;
            gsl::at(other, a) = s;
            const size_t column =
                other[0] + extents[0] * (other[1] + extents[1] * other[2]);
            d(row, column) = d_1d(gsl::at(point, a), s);
          }
        }
      }
    }
  }
  // Scaled stiffness M^{-1/2} K_D M^{-1/2} = sum_i C_i^T C_i with
  // C_i = M^{1/2} D_i M^{-1/2}.
  Matrix scaled_stiffness(number_of_points, number_of_points, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    Matrix c(number_of_points, number_of_points, 0.0);
    for (size_t a = 0; a < 3; ++a) {
      const DataVector& inverse_jacobian_ai =
          gsl::at(inverse_jacobian_, a + 3 * i);
      const Matrix& d = gsl::at(logical_derivatives, a);
      for (size_t column = 0; column < number_of_points; ++column) {
        for (size_t row = 0; row < number_of_points; ++row) {
          c(row, column) += sqrt_mass_[row] * inverse_jacobian_ai[row] *
                            d(row, column) / sqrt_mass_[column];
        }
      }
    }
    scaled_stiffness += blaze::trans(c) * c;
  }
  // The mean-zero basis Z0 = M^{-1/2} H E: H = I - 2 v v^T is the Householder
  // reflection with H q = -e_0 for the unit vector q = M^{1/2} 1 / sqrt(V),
  // and E embeds R^{N-1} as the vectors orthogonal to e_0, so H E is an
  // orthonormal basis of the orthogonal complement of q. As q_0 > 0,
  // |q + e_0| >= sqrt(2) involves no cancellation.
  fit_reflector_ = sqrt_mass_ / std::sqrt(volume_);
  fit_reflector_[0] += 1.0;
  fit_reflector_ /= std::sqrt(sum(fit_reflector_ * fit_reflector_));
  // Z0^T K_D Z0 = E^T H A H E, A the scaled stiffness, is the trailing block
  // of A - 2 v w^T - 2 w v^T + 4 (v^T w) v v^T with w = A v.
  DataVector stiffness_reflector{number_of_points, 0.0};
  for (size_t column = 0; column < number_of_points; ++column) {
    for (size_t row = 0; row < number_of_points; ++row) {
      stiffness_reflector[row] +=
          scaled_stiffness(row, column) * fit_reflector_[column];
    }
  }
  const double reflector_stiffness_reflector =
      sum(fit_reflector_ * stiffness_reflector);
  const size_t reduced_size = number_of_points - 1;
  Matrix reduced_stiffness(reduced_size, reduced_size);
  for (size_t column = 1; column < number_of_points; ++column) {
    for (size_t row = 1; row < number_of_points; ++row) {
      reduced_stiffness(row - 1, column - 1) =
          scaled_stiffness(row, column) -
          2.0 * fit_reflector_[row] * stiffness_reflector[column] -
          2.0 * stiffness_reflector[row] * fit_reflector_[column] +
          4.0 * reflector_stiffness_reflector * fit_reflector_[row] *
              fit_reflector_[column];
    }
  }
  // Symmetrize the roundoff before the eigenvalue check and factorization.
  reduced_stiffness =
      0.5 * (reduced_stiffness + blaze::trans(reduced_stiffness));
  {
    Matrix eigenvectors = reduced_stiffness;
    blaze::DynamicVector<double> eigenvalues(reduced_size);
    blaze::syev(eigenvectors, eigenvalues, 'N', 'L');
    const double largest = eigenvalues[reduced_size - 1];
    if (not(eigenvalues[0] > fit_minimum_relative_eigenvalue * largest)) {
      ERROR(
          "The gradient-fit stiffness restricted to mean-zero fields is too "
          "ill-conditioned for an accurate normal-equation solve: "
          "smallest/largest eigenvalue = "
          << eigenvalues[0] << "/" << largest << " is below the bound "
          << fit_minimum_relative_eigenvalue
          << ". Check for an additional gradient nullspace or an excessively "
             "ill-conditioned element.");
    }
  }
  fit_cholesky_factor_ = reduced_stiffness;
  blaze::blas_int_t info = 0;
  blaze::potrf('L', static_cast<blaze::blas_int_t>(reduced_size),
               fit_cholesky_factor_.data(),
               static_cast<blaze::blas_int_t>(fit_cholesky_factor_.spacing()),
               &info);
  if (info != 0) {
    ERROR(
        "Cholesky factorization of the gradient-fit stiffness failed with "
        "info = "
        << info);
  }
}

void LocalOperators::build_spherical_shell(
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian) {
  const size_t number_of_points = mesh.number_of_grid_points();
  const size_t radial_size = mesh.extents(0);
  const ShellAngularOperators& angular =
      shell_angular_operators(mesh.extents(1) - 1);
  const size_t angular_size = angular.synthesis.rows();
  const size_t harmonics = angular.synthesis.columns();
  const size_t trial_size = radial_size * harmonics;
  const size_t kernel_size = angular.kernel.columns();
  const size_t reduced_size = trial_size - kernel_size;
  verify_logical_derivatives(mesh, inverse_jacobian, angular);

  // The weighted trial basis M^{1/2} B and gradients [M^{1/2} D_i B]_i, trial
  // function k + N_r n being P_k(xi) Y_n.
  const Matrix& radial_vandermonde =
      Spectral::modal_to_nodal_matrix(mesh.slice_through(0));
  const Matrix radial_derivative =
      Spectral::differentiation_matrix(mesh.slice_through(0)) *
      radial_vandermonde;
  Matrix weighted_gradients(3 * number_of_points, trial_size);
  shell_factors_ = Matrix(trial_size, trial_size);
  {
    Matrix weighted_basis(number_of_points, trial_size);
    for (size_t n = 0; n < harmonics; ++n) {
      for (size_t k = 0; k < radial_size; ++k) {
        const size_t column = k + radial_size * n;
        for (size_t j = 0; j < angular_size; ++j) {
          const double harmonic = angular.synthesis(j, n);
          const double harmonic_derivative_1 = angular.trial_derivatives(j, n);
          const double harmonic_derivative_2 =
              angular.trial_derivatives(angular_size + j, n);
          for (size_t i = 0; i < radial_size; ++i) {
            const size_t point = i + radial_size * j;
            const double radial = radial_vandermonde(i, k);
            const std::array<double, 3> logical{
                {radial_derivative(i, k) * harmonic,
                 radial * harmonic_derivative_1,
                 radial * harmonic_derivative_2}};
            weighted_basis(point, column) =
                sqrt_mass_[point] * radial * harmonic;
            for (size_t d = 0; d < 3; ++d) {
              double derivative = 0.0;
              for (size_t a = 0; a < 3; ++a) {
                derivative += gsl::at(inverse_jacobian_, a + 3 * d)[point] *
                              gsl::at(logical, a);
              }
              weighted_gradients(point + number_of_points * d, column) =
                  sqrt_mass_[point] * derivative;
            }
          }
        }
      }
    }
    // Gram matrix B^T M B and its lower Cholesky factor L_G.
    dgemm_('T', 'N', trial_size, trial_size, number_of_points, 1.0,
           weighted_basis.data(), weighted_basis.spacing(),
           weighted_basis.data(), weighted_basis.spacing(), 0.0,
           shell_factors_.data(), shell_factors_.spacing());
  }
  blaze::blas_int_t info = 0;
  blaze::potrf('L', static_cast<blaze::blas_int_t>(trial_size),
               shell_factors_.data(),
               static_cast<blaze::blas_int_t>(shell_factors_.spacing()), &info);
  if (info != 0) {
    ERROR("The Gram matrix of the shell trial basis on the mesh "
          << mesh << " is not positive definite: Cholesky info = " << info);
  }
  // Only the lower triangle holds L_G; clear the rest for the reduced factor.
  for (size_t column = 1; column < trial_size; ++column) {
    for (size_t row = 0; row < column; ++row) {
      shell_factors_(row, column) = 0.0;
    }
  }

  // The stiffness in the M-orthonormal coordinates,
  // K~_D = L_G^{-1} K_D L_G^{-T}.
  Matrix stiffness(trial_size, trial_size);
  dgemm_('T', 'N', trial_size, trial_size, 3 * number_of_points, 1.0,
         weighted_gradients.data(), weighted_gradients.spacing(),
         weighted_gradients.data(), weighted_gradients.spacing(), 0.0,
         stiffness.data(), stiffness.spacing());
  triangular_solve(make_not_null(stiffness.data()), stiffness.spacing(), 'N',
                   shell_factors_, trial_size);
  stiffness = blaze::trans(stiffness);
  triangular_solve(make_not_null(stiffness.data()), stiffness.spacing(), 'N',
                   shell_factors_, trial_size);

  // The kernel, the radially constant functions times the angular kernel, in
  // the M-orthonormal coordinates: L_G^T Z with Z the coefficients of
  // P_0 times the angular kernel vectors.
  shell_kernel_reflectors_ = Matrix(trial_size, kernel_size, 0.0);
  for (size_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
    for (size_t row = 0; row < trial_size; ++row) {
      double value = 0.0;
      for (size_t n = 0; n < harmonics; ++n) {
        const size_t coefficient = radial_size * n;
        if (coefficient >= row) {
          value += shell_factors_(coefficient, row) *
                   angular.kernel(n, kernel_index);
        }
      }
      shell_kernel_reflectors_(row, kernel_index) = value;
    }
  }
  shell_kernel_reflector_factors_ = DataVector{kernel_size};
  {
    const auto workspace_size =
        static_cast<blaze::blas_int_t>(64 * kernel_size);
    std::vector<double> workspace(static_cast<size_t>(workspace_size));
    blaze::geqrf(
        static_cast<blaze::blas_int_t>(trial_size),
        static_cast<blaze::blas_int_t>(kernel_size),
        shell_kernel_reflectors_.data(),
        static_cast<blaze::blas_int_t>(shell_kernel_reflectors_.spacing()),
        shell_kernel_reflector_factors_.data(), workspace.data(),
        workspace_size, &info);
    if (info != 0) {
      ERROR("QR factorization of the shell kernel failed with info = " << info);
    }
  }
  // The coefficients L_G^{-T} N~ of the M-orthonormal kernel functions, N~
  // the leading columns of the orthogonal factor.
  shell_kernel_ = Matrix(trial_size, kernel_size, 0.0);
  for (size_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
    shell_kernel_(kernel_index, kernel_index) = 1.0;
  }
  apply_reflectors(make_not_null(shell_kernel_.data()), shell_kernel_.spacing(),
                   'L', 'N', trial_size, kernel_size, shell_kernel_reflectors_,
                   shell_kernel_reflector_factors_);
  triangular_solve(make_not_null(shell_kernel_.data()), shell_kernel_.spacing(),
                   'T', shell_factors_, kernel_size);

  // The stiffness on the complement W of the kernel: the trailing block of
  // [N~ W]^T K~_D [N~ W].
  apply_reflectors(make_not_null(stiffness.data()), stiffness.spacing(), 'L',
                   'T', trial_size, trial_size, shell_kernel_reflectors_,
                   shell_kernel_reflector_factors_);
  apply_reflectors(make_not_null(stiffness.data()), stiffness.spacing(), 'R',
                   'N', trial_size, trial_size, shell_kernel_reflectors_,
                   shell_kernel_reflector_factors_);
  Matrix reduced_stiffness = blaze::submatrix(
      stiffness, kernel_size, kernel_size, reduced_size, reduced_size);
  // Symmetrize the roundoff before the eigenvalue check and factorization.
  reduced_stiffness =
      0.5 * (reduced_stiffness + blaze::trans(reduced_stiffness));
  double largest_eigenvalue = 0.0;
  double smallest_eigenvalue = 0.0;
  {
    Matrix eigenvectors = reduced_stiffness;
    blaze::DynamicVector<double> eigenvalues(reduced_size);
    blaze::syev(eigenvectors, eigenvalues, 'N', 'L');
    smallest_eigenvalue = eigenvalues[0];
    largest_eigenvalue = eigenvalues[reduced_size - 1];
  }

  // The kernel functions have numerically zero derivatives.
  {
    Matrix kernel_gradients(3 * number_of_points, kernel_size);
    dgemm_('N', 'N', 3 * number_of_points, kernel_size, trial_size, 1.0,
           weighted_gradients.data(), weighted_gradients.spacing(),
           shell_kernel_.data(), shell_kernel_.spacing(), 0.0,
           kernel_gradients.data(), kernel_gradients.spacing());
    double largest_kernel_gradient = 0.0;
    for (size_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
      double norm = 0.0;
      for (size_t row = 0; row < 3 * number_of_points; ++row) {
        norm += square(kernel_gradients(row, kernel_index));
      }
      largest_kernel_gradient =
          std::max(largest_kernel_gradient, std::sqrt(norm));
    }
    if (not(largest_kernel_gradient <=
            shell_kernel_tolerance * std::sqrt(largest_eigenvalue))) {
      ERROR("The shell derivative kernel is not annihilated on the mesh "
            << mesh << ": the largest weighted gradient norm of an "
            << "M-orthonormal kernel function is " << largest_kernel_gradient
            << ", larger than " << shell_kernel_tolerance
            << " times the largest singular value "
            << std::sqrt(largest_eigenvalue));
    }
  }
  if (not(smallest_eigenvalue >
          fit_minimum_relative_eigenvalue * largest_eigenvalue)) {
    ERROR(
        "The gradient-fit stiffness restricted to the complement of the "
        "shell derivative kernel is too ill-conditioned for an accurate "
        "normal-equation solve: smallest/largest eigenvalue = "
        << smallest_eigenvalue << "/" << largest_eigenvalue
        << " is below the bound " << fit_minimum_relative_eigenvalue
        << " on the mesh " << mesh
        << ". Check for an additional gradient nullspace or an excessively "
           "ill-conditioned element.");
  }
  for (size_t column = 0; column < reduced_size; ++column) {
    for (size_t row = 0; row <= column; ++row) {
      shell_factors_(row, kernel_size + column) =
          reduced_stiffness(row, column);
    }
  }
  blaze::potrf(
      'U', static_cast<blaze::blas_int_t>(reduced_size),
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      shell_factors_.data() + kernel_size * shell_factors_.spacing(),
      static_cast<blaze::blas_int_t>(shell_factors_.spacing()), &info);
  if (info != 0) {
    ERROR(
        "Cholesky factorization of the reduced shell gradient-fit stiffness "
        "failed with info = "
        << info);
  }
}

double LocalOperators::mean(const DataVector& field) const {
  return sum(mass_ * field) / volume_;
}

size_t LocalOperators::number_of_retained_modes() const {
  require_tensor_product("number_of_retained_modes");
  return number_of_retained_modes_;
}

Matrix LocalOperators::derivative_kernel() const {
  require_spherical_shell("derivative_kernel");
  return trial_functions(shell_kernel_);
}

Matrix LocalOperators::trial_functions(const Matrix& coefficients) const {
  require_spherical_shell("trial_functions");
  const size_t radial_size = mesh_.extents(0);
  const size_t number_of_fields = coefficients.columns();
  DataVector values{mesh_.number_of_grid_points() * number_of_fields};
  synthesize(make_not_null(values.data()),
             to_batch(coefficients, radial_size).data(), number_of_fields,
             mesh_, shell_angular_operators(mesh_.extents(1) - 1));
  return from_batch(values, radial_size, number_of_fields);
}

Matrix LocalOperators::trial_functions_transpose(const Matrix& values) const {
  require_spherical_shell("trial_functions_transpose");
  const size_t radial_size = mesh_.extents(0);
  const size_t number_of_fields = values.columns();
  const ShellAngularOperators& angular =
      shell_angular_operators(mesh_.extents(1) - 1);
  DataVector coefficients{radial_size * angular.synthesis.columns() *
                          number_of_fields};
  synthesize_transpose(make_not_null(coefficients.data()),
                       to_batch(values, radial_size).data(), number_of_fields,
                       mesh_, angular);
  return from_batch(coefficients, radial_size, number_of_fields);
}

std::array<Matrix, 3> LocalOperators::trial_function_gradients(
    const Matrix& coefficients) const {
  require_spherical_shell("trial_function_gradients");
  const size_t number_of_points = mesh_.number_of_grid_points();
  const size_t radial_size = mesh_.extents(0);
  const size_t number_of_fields = coefficients.columns();
  const size_t batch_size = number_of_points * number_of_fields;
  DataVector logical{3 * batch_size};
  logical_gradient(make_not_null(logical.data()),
                   to_batch(coefficients, radial_size).data(), number_of_fields,
                   mesh_, shell_angular_operators(mesh_.extents(1) - 1));
  std::array<Matrix, 3> result{};
  DataVector gradient{batch_size};
  for (size_t i = 0; i < 3; ++i) {
    gradient = 0.0;
    for (size_t a = 0; a < 3; ++a) {
      for (size_t index = 0; index < batch_size; ++index) {
        const size_t point =
            index % radial_size +
            radial_size * (index / (radial_size * number_of_fields));
        gradient[index] += gsl::at(inverse_jacobian_, a + 3 * i)[point] *
                           logical[index + a * batch_size];
      }
    }
    gsl::at(result, i) = from_batch(gradient, radial_size, number_of_fields);
  }
  return result;
}

Matrix LocalOperators::trial_function_gradients_transpose(
    const std::array<Matrix, 3>& fluxes) const {
  require_spherical_shell("trial_function_gradients_transpose");
  const size_t number_of_points = mesh_.number_of_grid_points();
  const size_t radial_size = mesh_.extents(0);
  const size_t number_of_fields = fluxes[0].columns();
  const size_t batch_size = number_of_points * number_of_fields;
  const ShellAngularOperators& angular =
      shell_angular_operators(mesh_.extents(1) - 1);
  DataVector logical{3 * batch_size, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    const DataVector flux = to_batch(gsl::at(fluxes, i), radial_size);
    for (size_t a = 0; a < 3; ++a) {
      for (size_t index = 0; index < batch_size; ++index) {
        const size_t point =
            index % radial_size +
            radial_size * (index / (radial_size * number_of_fields));
        logical[index + a * batch_size] +=
            gsl::at(inverse_jacobian_, a + 3 * i)[point] * flux[index];
      }
    }
  }
  DataVector coefficients{radial_size * angular.synthesis.columns() *
                          number_of_fields};
  logical_gradient_transpose(make_not_null(coefficients.data()), logical.data(),
                             number_of_fields, mesh_, angular);
  return from_batch(coefficients, radial_size, number_of_fields);
}

void LocalOperators::require_tensor_product(const char* const operation) const {
  if (spherical_shell_) {
    ERROR("LocalOperators::"
          << operation
          << " is not implemented on spherical-harmonic shells: the "
             "scalar-list volume filter and the mean-constrained fit exist "
             "only on tensor-product elements (shells use the grouped shell "
             "velocity filter). Mesh: "
          << mesh_);
  }
}

void LocalOperators::require_spherical_shell(
    const char* const operation) const {
  if (not spherical_shell_) {
    ERROR("LocalOperators::"
          << operation
          << " is defined only on spherical-harmonic shells, but the mesh is "
          << mesh_);
  }
}

void LocalOperators::check_mesh(const Mesh<3>& mesh) const {
  if (mesh != mesh_) {
    ERROR(
        "The CCZ4 gradient-projection local operators were built for the "
        "mesh "
        << mesh_ << " but are used with the mesh " << mesh
        << ". Only static meshes are supported.");
  }
}

template <size_t NumberOfFields>
void LocalOperators::fit(
    const gsl::not_null<std::array<DataVector, NumberOfFields>*> fields,
    const std::array<std::array<const DataVector*, NumberOfFields>, 3>&
        gradient_targets,
    const std::array<double, NumberOfFields>& means) const {
  require_tensor_product("fit");
  const size_t number_of_points = mesh_.number_of_grid_points();
  const size_t reduced_size = number_of_points - 1;
  // Right-hand sides M^{-1/2} b_D for each field, b_D = sum_a Dhat_a^T f_a,
  // f_a = sum_i Jinv^a_i M q*_i.
  Matrix scaled_rhs(number_of_points, NumberOfFields);
  DataVector logical_flux{number_of_points};
  DataVector b_d{number_of_points};
  for (size_t field = 0; field < NumberOfFields; ++field) {
    b_d = 0.0;
    for (size_t a = 0; a < 3; ++a) {
      logical_flux = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        logical_flux += gsl::at(inverse_jacobian_, a + 3 * i) *
                        *gsl::at(gsl::at(gradient_targets, i), field);
      }
      logical_flux *= mass_;
      const Matrix empty{};
      std::array<std::reference_wrapper<const Matrix>, 3> matrices{
          {std::cref(empty), std::cref(empty), std::cref(empty)}};
      gsl::at(matrices, a) = std::cref(
          Spectral::differentiation_matrix_transpose(mesh_.slice_through(a)));
      b_d += apply_matrices(matrices, logical_flux, mesh_.extents());
    }
    for (size_t point = 0; point < number_of_points; ++point) {
      scaled_rhs(point, field) = b_d[point] / sqrt_mass_[point];
    }
  }
  // The coordinates Z0^T b_D = E^T H M^{-1/2} b_D in the mean-zero basis
  // Z0 = M^{-1/2} H E.
  reflect_columns(make_not_null(&scaled_rhs), fit_reflector_);
  Matrix reduced =
      blaze::submatrix(scaled_rhs, 1, 0, reduced_size, NumberOfFields);
  blaze::blas_int_t info = 0;
  blaze::potrs('L', static_cast<blaze::blas_int_t>(reduced_size),
               static_cast<blaze::blas_int_t>(NumberOfFields),
               fit_cholesky_factor_.data(),
               static_cast<blaze::blas_int_t>(fit_cholesky_factor_.spacing()),
               reduced.data(),
               static_cast<blaze::blas_int_t>(reduced.spacing()), &info);
  if (info != 0) {
    ERROR("Cholesky solve of the gradient fit failed with info = " << info);
  }
  // M^{1/2} Z0 y = H E y, with first coordinate zero before the reflection.
  Matrix scaled_fit(number_of_points, NumberOfFields, 0.0);
  blaze::submatrix(scaled_fit, 1, 0, reduced_size, NumberOfFields) = reduced;
  reflect_columns(make_not_null(&scaled_fit), fit_reflector_);
  for (size_t field = 0; field < NumberOfFields; ++field) {
    DataVector& result = gsl::at(*fields, field);
    result.destructive_resize(number_of_points);
    for (size_t point = 0; point < number_of_points; ++point) {
      result[point] =
          gsl::at(means, field) + scaled_fit(point, field) / sqrt_mass_[point];
    }
  }
}

template <size_t NumberOfFields>
void LocalOperators::fit_from_provisional(
    const gsl::not_null<std::array<DataVector, NumberOfFields>*> fields,
    const std::array<std::array<const DataVector*, NumberOfFields>, 3>&
        gradient_targets,
    const std::array<const DataVector*, NumberOfFields>& provisional_rates)
    const {
  if (not spherical_shell_) {
    std::array<double, NumberOfFields> means{};
    for (size_t field = 0; field < NumberOfFields; ++field) {
      gsl::at(means, field) = mean(*gsl::at(provisional_rates, field));
    }
    fit(fields, gradient_targets, means);
    return;
  }
  fit_spherical_shell(fields, gradient_targets, provisional_rates);
}

template <size_t NumberOfFields>
void LocalOperators::fit_spherical_shell(
    const gsl::not_null<std::array<DataVector, NumberOfFields>*> fields,
    const std::array<std::array<const DataVector*, NumberOfFields>, 3>&
        gradient_targets,
    const std::array<const DataVector*, NumberOfFields>& provisional_rates)
    const {
  const size_t number_of_points = mesh_.number_of_grid_points();
  const size_t radial_size = mesh_.extents(0);
  const ShellAngularOperators& angular =
      shell_angular_operators(mesh_.extents(1) - 1);
  const size_t angular_size = angular.synthesis.rows();
  const size_t harmonics = angular.synthesis.columns();
  const size_t trial_size = radial_size * harmonics;
  const size_t kernel_size = shell_kernel_.columns();
  const size_t batch_size = number_of_points * NumberOfFields;
  const size_t rows = radial_size * NumberOfFields;

  // The logical fluxes sum_i Jinv^a_i M q*_i and the weighted provisional
  // rates M r*, as batches.
  DataVector logical{3 * batch_size};
  DataVector weighted_rates{batch_size};
  for (size_t field = 0; field < NumberOfFields; ++field) {
    const DataVector& rate = *gsl::at(provisional_rates, field);
    const std::array<const DataVector*, 3> targets{
        {gsl::at(gradient_targets[0], field),
         gsl::at(gradient_targets[1], field),
         gsl::at(gradient_targets[2], field)}};
    for (size_t j = 0; j < angular_size; ++j) {
      for (size_t i = 0; i < radial_size; ++i) {
        const size_t point = i + radial_size * j;
        const size_t index = i + radial_size * field + rows * j;
        weighted_rates[index] = mass_[point] * rate[point];
        for (size_t a = 0; a < 3; ++a) {
          double flux = 0.0;
          for (size_t d = 0; d < 3; ++d) {
            flux += gsl::at(inverse_jacobian_, a + 3 * d)[point] *
                    (*gsl::at(targets, d))[point];
          }
          logical[index + a * batch_size] = mass_[point] * flux;
        }
      }
    }
  }
  DataVector batch_coefficients{trial_size * NumberOfFields};
  // Right-hand sides sum_i (D_i B)^T M q*_i, one column per field.
  Matrix coefficients(trial_size, NumberOfFields);
  logical_gradient_transpose(make_not_null(batch_coefficients.data()),
                             logical.data(), NumberOfFields, mesh_, angular);
  for (size_t n = 0; n < harmonics; ++n) {
    for (size_t field = 0; field < NumberOfFields; ++field) {
      for (size_t k = 0; k < radial_size; ++k) {
        coefficients(k + radial_size * n, field) =
            batch_coefficients[k + radial_size * field + rows * n];
      }
    }
  }
  // The M-projections of the provisional rates onto the M-orthonormal
  // kernel functions, Y^T B^T M r* with Y = L_G^{-T} N~.
  synthesize_transpose(make_not_null(batch_coefficients.data()),
                       weighted_rates.data(), NumberOfFields, mesh_, angular);
  Matrix rate_coefficients(trial_size, NumberOfFields);
  for (size_t n = 0; n < harmonics; ++n) {
    for (size_t field = 0; field < NumberOfFields; ++field) {
      for (size_t k = 0; k < radial_size; ++k) {
        rate_coefficients(k + radial_size * n, field) =
            batch_coefficients[k + radial_size * field + rows * n];
      }
    }
  }
  Matrix kernel_projections(kernel_size, NumberOfFields);
  dgemm_('T', 'N', kernel_size, NumberOfFields, trial_size, 1.0,
         shell_kernel_.data(), shell_kernel_.spacing(),
         rate_coefficients.data(), rate_coefficients.spacing(), 0.0,
         kernel_projections.data(), kernel_projections.spacing());
  // The fit, completed with the kernel functions.
  solve_spherical_shell(make_not_null(coefficients.data()), NumberOfFields,
                        coefficients.spacing());
  dgemm_('N', 'N', trial_size, NumberOfFields, kernel_size, 1.0,
         shell_kernel_.data(), shell_kernel_.spacing(),
         kernel_projections.data(), kernel_projections.spacing(), 1.0,
         coefficients.data(), coefficients.spacing());
  for (size_t n = 0; n < harmonics; ++n) {
    for (size_t field = 0; field < NumberOfFields; ++field) {
      for (size_t k = 0; k < radial_size; ++k) {
        batch_coefficients[k + radial_size * field + rows * n] =
            coefficients(k + radial_size * n, field);
      }
    }
  }
  synthesize(make_not_null(weighted_rates.data()), batch_coefficients.data(),
             NumberOfFields, mesh_, angular);
  for (size_t field = 0; field < NumberOfFields; ++field) {
    DataVector& result = gsl::at(*fields, field);
    result.destructive_resize(number_of_points);
    for (size_t j = 0; j < angular_size; ++j) {
      for (size_t i = 0; i < radial_size; ++i) {
        result[i + radial_size * j] =
            weighted_rates[i + radial_size * field + rows * j];
      }
    }
  }
}

void LocalOperators::solve_spherical_shell(
    const gsl::not_null<double*> coefficients, const size_t number_of_fields,
    const size_t leading_dimension) const {
  const size_t trial_size = shell_factors_.rows();
  const size_t kernel_size = shell_kernel_.columns();
  const size_t reduced_size = trial_size - kernel_size;
  // x = W^T L_G^{-1} b: the trailing rows of [N~ W]^T L_G^{-1} b.
  triangular_solve(coefficients, leading_dimension, 'N', shell_factors_,
                   number_of_fields);
  apply_reflectors(coefficients, leading_dimension, 'L', 'T', trial_size,
                   number_of_fields, shell_kernel_reflectors_,
                   shell_kernel_reflector_factors_);
  blaze::blas_int_t info = 0;
  blaze::potrs(
      'U', static_cast<blaze::blas_int_t>(reduced_size),
      static_cast<blaze::blas_int_t>(number_of_fields),
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      shell_factors_.data() + kernel_size * shell_factors_.spacing(),
      static_cast<blaze::blas_int_t>(shell_factors_.spacing()),
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      coefficients.get() + kernel_size,
      static_cast<blaze::blas_int_t>(leading_dimension), &info);
  if (info != 0) {
    ERROR(
        "Cholesky solve of the shell gradient fit failed with info = " << info);
  }
  // L_G^{-T} [N~ W] (0, (W^T K~_D W)^{-1} x).
  for (size_t field = 0; field < number_of_fields; ++field) {
    for (size_t row = 0; row < kernel_size; ++row) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      coefficients.get()[row + leading_dimension * field] = 0.0;
    }
  }
  apply_reflectors(coefficients, leading_dimension, 'L', 'N', trial_size,
                   number_of_fields, shell_kernel_reflectors_,
                   shell_kernel_reflector_factors_);
  triangular_solve(coefficients, leading_dimension, 'T', shell_factors_,
                   number_of_fields);
}

DataVector LocalOperators::project_retained(const DataVector& field) const {
  require_tensor_product("project_retained");
  const size_t number_of_points = mesh_.number_of_grid_points();
  Matrix weighted(number_of_points, 1);
  for (size_t point = 0; point < number_of_points; ++point) {
    weighted(point, 0) = sqrt_mass_[point] * field[point];
  }
  // M^{1/2} Pi v = M^{1/2} v - Q_high Q_high^T M^{1/2} v.
  subtract_projection(make_not_null(&weighted), removed_basis_);
  DataVector result{number_of_points};
  for (size_t point = 0; point < number_of_points; ++point) {
    result[point] = weighted(point, 0) / sqrt_mass_[point];
  }
  return result;
}

void LocalOperators::apply_filter(const gsl::not_null<DataVector*> field,
                                  const double gain) const {
  require_tensor_product("apply_filter");
  const DataVector projected = project_retained(*field);
  *field = gain * (*field) + (1.0 - gain) * projected;
}

void LocalOperators::apply_filter(
    const std::vector<gsl::not_null<DataVector*>>& fields,
    const double gain) const {
  require_tensor_product("apply_filter");
  const size_t number_of_points = mesh_.number_of_grid_points();
  const size_t number_of_fields = fields.size();
  for (size_t i = 0; i < number_of_fields; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (fields[i].get() == fields[j].get()) {
        ERROR(
            "LocalOperators::apply_filter requires distinct field pointers, "
            "but entries "
            << j << " and " << i << " point to the same field.");
      }
    }
  }
  Matrix weighted(number_of_points, number_of_fields);
  for (size_t field_index = 0; field_index < number_of_fields; ++field_index) {
    const DataVector& field = *fields[field_index];
    for (size_t point = 0; point < number_of_points; ++point) {
      weighted(point, field_index) = sqrt_mass_[point] * field[point];
    }
  }
  // The weighted projections M^{1/2} Pi v of all fields, computed before any
  // field is written.
  subtract_projection(make_not_null(&weighted), removed_basis_);
  DataVector unweighted{number_of_points};
  for (size_t field_index = 0; field_index < number_of_fields; ++field_index) {
    for (size_t point = 0; point < number_of_points; ++point) {
      unweighted[point] = weighted(point, field_index) / sqrt_mass_[point];
    }
    DataVector& field = *fields[field_index];
    field = gain * field + (1.0 - gain) * unweighted;
  }
}

void LocalOperators::apply_shell_filter(const ShellFilterFields& fields,
                                        const double gain) const {
  require_spherical_shell("apply_shell_filter");
  if (shell_mass_separability_residual_ <= shell_mass_separability_threshold) {
    separable_shell_filter_.apply(fields, gain);
  } else {
    general_shell_filter_.apply(fields, gain);
  }
}

void LocalOperators::apply_shell_filter_general(const ShellFilterFields& fields,
                                                const double gain) const {
  require_spherical_shell("apply_shell_filter_general");
  general_shell_filter_.apply(fields, gain);
}

double LocalOperators::shell_mass_separability_residual() const {
  require_spherical_shell("shell_mass_separability_residual");
  return shell_mass_separability_residual_;
}

void LocalOperators::apply_shell_filter_separable(
    const ShellFilterFields& fields, const double gain) const {
  require_spherical_shell("apply_shell_filter_separable");
  separable_shell_filter_.apply(fields, gain);
}

void LocalOperators::pup(PUP::er& p) {
  p | mesh_;
  p | spherical_shell_;
  p | mass_;
  p | sqrt_mass_;
  p | volume_;
  p | inverse_jacobian_;
  p | removed_basis_;
  p | number_of_retained_modes_;
  p | fit_reflector_;
  p | fit_cholesky_factor_;
  p | shell_factors_;
  p | shell_kernel_reflectors_;
  p | shell_kernel_reflector_factors_;
  p | shell_kernel_;
  p | shell_angular_modes_to_kill_;
  p | general_shell_filter_;
  p | separable_shell_filter_;
  p | shell_mass_separability_residual_;
}

bool operator==(const LocalOperators& lhs, const LocalOperators& rhs) {
  return lhs.mesh_ == rhs.mesh_ and
         lhs.spherical_shell_ == rhs.spherical_shell_ and
         lhs.mass_ == rhs.mass_ and lhs.sqrt_mass_ == rhs.sqrt_mass_ and
         lhs.volume_ == rhs.volume_ and
         lhs.inverse_jacobian_ == rhs.inverse_jacobian_ and
         lhs.removed_basis_ == rhs.removed_basis_ and
         lhs.number_of_retained_modes_ == rhs.number_of_retained_modes_ and
         lhs.fit_reflector_ == rhs.fit_reflector_ and
         lhs.fit_cholesky_factor_ == rhs.fit_cholesky_factor_ and
         lhs.shell_factors_ == rhs.shell_factors_ and
         lhs.shell_kernel_reflectors_ == rhs.shell_kernel_reflectors_ and
         lhs.shell_kernel_reflector_factors_ ==
             rhs.shell_kernel_reflector_factors_ and
         lhs.shell_kernel_ == rhs.shell_kernel_ and
         lhs.shell_angular_modes_to_kill_ ==
             rhs.shell_angular_modes_to_kill_ and
         lhs.general_shell_filter_ == rhs.general_shell_filter_ and
         lhs.separable_shell_filter_ == rhs.separable_shell_filter_ and
         lhs.shell_mass_separability_residual_ ==
             rhs.shell_mass_separability_residual_;
}

bool operator!=(const LocalOperators& lhs, const LocalOperators& rhs) {
  return not(lhs == rhs);
}

template void LocalOperators::fit<1>(
    gsl::not_null<std::array<DataVector, 1>*>,
    const std::array<std::array<const DataVector*, 1>, 3>&,
    const std::array<double, 1>&) const;
template void LocalOperators::fit<number_of_potentials>(
    gsl::not_null<std::array<DataVector, number_of_potentials>*>,
    const std::array<std::array<const DataVector*, number_of_potentials>, 3>&,
    const std::array<double, number_of_potentials>&) const;
template void LocalOperators::fit_from_provisional<1>(
    gsl::not_null<std::array<DataVector, 1>*>,
    const std::array<std::array<const DataVector*, 1>, 3>&,
    const std::array<const DataVector*, 1>&) const;
template void LocalOperators::fit_from_provisional<number_of_potentials>(
    gsl::not_null<std::array<DataVector, number_of_potentials>*>,
    const std::array<std::array<const DataVector*, number_of_potentials>, 3>&,
    const std::array<const DataVector*, number_of_potentials>&) const;
}  // namespace Ccz4::GradientProjection
