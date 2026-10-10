// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"

#include <algorithm>
#include <array>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/Submatrix.h>
#include <blaze/math/dense/QR.h>
#include <blaze/math/lapack/clapack/potrf.h>
#include <blaze/math/lapack/clapack/potrs.h>
#include <blaze/math/lapack/syev.h>
#include <cmath>
#include <cstddef>
#include <functional>
#include <pup.h>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace Ccz4::GradientProjection {
LocalOperators::LocalOperators(
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const Scalar<DataVector>& det_inverse_jacobian)
    : mesh_(mesh) {
  for (size_t d = 0; d < 3; ++d) {
    if (mesh.basis(d) != Spectral::Basis::Legendre or
        mesh.quadrature(d) != Spectral::Quadrature::GaussLobatto) {
      ERROR(
          "The CCZ4 gradient projection requires a Legendre "
          "Gauss-Lobatto mesh, but got "
          << mesh);
    }
    if (mesh.extents(d) < 2) {
      ERROR(
          "The CCZ4 gradient projection requires at least two points per "
          "dimension, but got "
          << mesh);
    }
  }
  const size_t number_of_points = mesh.number_of_grid_points();
  const std::array<size_t, 3> extents{
      {mesh.extents(0), mesh.extents(1), mesh.extents(2)}};

  // Mass weights m = J w_1 w_2 w_3.
  mass_ = DataVector{number_of_points};
  for (size_t k = 0; k < extents[2]; ++k) {
    for (size_t j = 0; j < extents[1]; ++j) {
      for (size_t i = 0; i < extents[0]; ++i) {
        const size_t index = i + extents[0] * (j + extents[1] * k);
        mass_[index] = Spectral::quadrature_weights(mesh.slice_through(0))[i] *
                       Spectral::quadrature_weights(mesh.slice_through(1))[j] *
                       Spectral::quadrature_weights(mesh.slice_through(2))[k] /
                       get(det_inverse_jacobian)[index];
      }
    }
  }
  if (not(min(mass_) > 0.0)) {
    ERROR("The coordinate Jacobian determinant must be positive.");
  }
  sqrt_mass_ = sqrt(mass_);
  volume_ = sum(mass_);
  for (size_t a = 0; a < 3; ++a) {
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(inverse_jacobian_, a + 3 * i) = inverse_jacobian.get(a, i);
    }
  }

  // Mass-weighted QR of the Legendre Vandermonde matrix with columns ordered
  // (constant, remaining retained modes, removed shell).
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
  Matrix upper_triangular{};
  blaze::qr(weighted_basis, orthonormal_basis_, upper_triangular);

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
  const size_t reduced_size = number_of_points - 1;
  const Matrix q_1 = blaze::submatrix(orthonormal_basis_, 0, 1,
                                      number_of_points, reduced_size);
  Matrix reduced_stiffness = blaze::trans(q_1) * scaled_stiffness * q_1;
  // Symmetrize the roundoff before the eigenvalue check and factorization.
  reduced_stiffness =
      0.5 * (reduced_stiffness + blaze::trans(reduced_stiffness));
  {
    Matrix eigenvectors = reduced_stiffness;
    blaze::DynamicVector<double> eigenvalues(reduced_size);
    blaze::syev(eigenvectors, eigenvalues, 'N', 'L');
    const double largest = eigenvalues[reduced_size - 1];
    if (not(eigenvalues[0] > 1.0e-12 * largest)) {
      ERROR(
          "The gradient-fit stiffness restricted to mean-zero fields is not "
          "sufficiently positive definite: smallest/largest eigenvalue = "
          << eigenvalues[0] << "/" << largest
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

double LocalOperators::mean(const DataVector& field) const {
  return sum(mass_ * field) / volume_;
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
  const auto q_1 = blaze::submatrix(orthonormal_basis_, 0, 1, number_of_points,
                                    reduced_size);
  Matrix reduced = blaze::trans(q_1) * scaled_rhs;
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
  const Matrix scaled_fit = q_1 * reduced;
  for (size_t field = 0; field < NumberOfFields; ++field) {
    DataVector& result = gsl::at(*fields, field);
    result.destructive_resize(number_of_points);
    for (size_t point = 0; point < number_of_points; ++point) {
      result[point] =
          gsl::at(means, field) + scaled_fit(point, field) / sqrt_mass_[point];
    }
  }
}

DataVector LocalOperators::project_retained(const DataVector& field) const {
  const size_t number_of_points = mesh_.number_of_grid_points();
  blaze::DynamicVector<double> weighted(number_of_points);
  for (size_t point = 0; point < number_of_points; ++point) {
    weighted[point] = sqrt_mass_[point] * field[point];
  }
  const auto q_low = blaze::submatrix(
      orthonormal_basis_, 0, 0, number_of_points, number_of_retained_modes_);
  const blaze::DynamicVector<double> coefficients =
      blaze::trans(q_low) * weighted;
  const blaze::DynamicVector<double> projected = q_low * coefficients;
  DataVector result{number_of_points};
  for (size_t point = 0; point < number_of_points; ++point) {
    result[point] = projected[point] / sqrt_mass_[point];
  }
  return result;
}

void LocalOperators::apply_filter(
    const gsl::not_null<DataVector*> field) const {
  const DataVector projected = project_retained(*field);
  *field = shell_filter_gain * (*field) + (1.0 - shell_filter_gain) * projected;
}

void LocalOperators::pup(PUP::er& p) {
  p | mesh_;
  p | mass_;
  p | sqrt_mass_;
  p | volume_;
  p | inverse_jacobian_;
  p | orthonormal_basis_;
  p | number_of_retained_modes_;
  p | fit_cholesky_factor_;
}

bool operator==(const LocalOperators& lhs, const LocalOperators& rhs) {
  return lhs.mesh_ == rhs.mesh_ and lhs.mass_ == rhs.mass_ and
         lhs.sqrt_mass_ == rhs.sqrt_mass_ and lhs.volume_ == rhs.volume_ and
         lhs.inverse_jacobian_ == rhs.inverse_jacobian_ and
         lhs.orthonormal_basis_ == rhs.orthonormal_basis_ and
         lhs.number_of_retained_modes_ == rhs.number_of_retained_modes_ and
         lhs.fit_cholesky_factor_ == rhs.fit_cholesky_factor_;
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
}  // namespace Ccz4::GradientProjection
