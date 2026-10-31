// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <blaze/math/Column.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/Row.h>
#include <blaze/math/Submatrix.h>
#include <blaze/math/dense/Inversion.h>
#include <blaze/math/dense/QR.h>
#include <blaze/math/lapack/gesvd.h>
#include <blaze/math/lapack/syev.h>
#include <cmath>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
using TestHelpers::Ccz4GradientProjection::Geometry;
using TestHelpers::Ccz4GradientProjection::make_geometry;

double legendre(const size_t degree, const double x) {
  double previous = 1.0;
  if (degree == 0) {
    return previous;
  }
  double current = x;
  for (size_t n = 1; n < degree; ++n) {
    const double next = ((2.0 * static_cast<double>(n) + 1.0) * x * current -
                         static_cast<double>(n) * previous) /
                        (static_cast<double>(n) + 1.0);
    previous = current;
    current = next;
  }
  return current;
}

// Nodal values of the tensor-product Legendre mode (l, m, n).
DataVector mode(const Geometry& geometry, const size_t l, const size_t m,
                const size_t n) {
  DataVector result{geometry.mesh.number_of_grid_points()};
  for (size_t s = 0; s < result.size(); ++s) {
    result[s] = legendre(l, get<0>(geometry.logical)[s]) *
                legendre(m, get<1>(geometry.logical)[s]) *
                legendre(n, get<2>(geometry.logical)[s]);
  }
  return result;
}

std::array<DataVector, 3> gradient(const Geometry& geometry,
                                   const DataVector& field) {
  const auto derivative = partial_derivative(
      Scalar<DataVector>{field}, geometry.mesh, geometry.inverse_jacobian);
  return {{derivative.get(0), derivative.get(1), derivative.get(2)}};
}

DataVector random_field(const gsl::not_null<std::mt19937*> gen,
                        const size_t size) {
  std::normal_distribution<double> dist(0.0, 1.0);
  DataVector result{size};
  for (auto& value : result) {
    value = dist(*gen);
  }
  return result;
}

void test_fit(const Geometry& geometry,
              const gsl::not_null<std::mt19937*> gen) {
  const gp::LocalOperators operators{geometry.mesh, geometry.inverse_jacobian,
                                     geometry.det_inverse_jacobian, 4};
  const size_t n = geometry.mesh.number_of_grid_points();
  CHECK(sum(operators.mass()) == approx(operators.volume()));

  // Every nodal field is a representable polynomial: exact gradient targets
  // and the exact mean reproduce it.
  const DataVector exact = random_field(gen, n);
  const auto exact_gradient = gradient(geometry, exact);
  std::array<DataVector, 1> fitted{};
  operators.fit(make_not_null(&fitted),
                {{{{&exact_gradient[0]}},
                  {{&exact_gradient[1]}},
                  {{&exact_gradient[2]}}}},
                {{operators.mean(exact)}});
  CHECK_ITERABLE_CUSTOM_APPROX(fitted[0], exact,
                               Approx::custom().epsilon(1e-10).scale(1.0));

  // Arbitrary targets: the prescribed mass-weighted mean, and the normal
  // equations sum_i D_i^T M (D_i r - q_i) = 0, i.e. the residual is
  // M-orthogonal to every element gradient.
  const std::array<DataVector, 3> targets{
      {random_field(gen, n), random_field(gen, n), random_field(gen, n)}};
  const double mean = 0.37;
  operators.fit(make_not_null(&fitted),
                {{{{&targets[0]}}, {{&targets[1]}}, {{&targets[2]}}}},
                {{mean}});
  CHECK(operators.mean(fitted[0]) == approx(mean));
  const auto fitted_gradient = gradient(geometry, fitted[0]);
  double target_scale = 0.0;
  for (size_t sample = 0; sample < 4; ++sample) {
    const auto test_gradient = gradient(geometry, random_field(gen, n));
    double inner_product = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      inner_product += sum(operators.mass() * gsl::at(test_gradient, i) *
                           (gsl::at(fitted_gradient, i) - gsl::at(targets, i)));
      target_scale += sum(operators.mass() * abs(gsl::at(test_gradient, i)) *
                          abs(gsl::at(targets, i)));
    }
    CHECK(std::abs(inner_product) < 1e-11 * target_scale);
  }

  // All potentials fitted together, as in the scheme (the dense products act
  // on all fields at once), give each field's own fit.
  std::array<std::array<DataVector, gp::number_of_potentials>, 3> all_targets{};
  std::array<std::array<const DataVector*, gp::number_of_potentials>, 3>
      target_pointers{};
  std::array<double, gp::number_of_potentials> means{};
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(gsl::at(all_targets, i), a) = random_field(gen, n);
      gsl::at(gsl::at(target_pointers, i), a) =
          &gsl::at(gsl::at(all_targets, i), a);
    }
    gsl::at(means, a) = 0.1 * static_cast<double>(a) - 0.2;
  }
  std::array<DataVector, gp::number_of_potentials> fitted_together{};
  operators.fit(make_not_null(&fitted_together), target_pointers, means);
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    operators.fit(make_not_null(&fitted),
                  {{{{&gsl::at(all_targets[0], a)}},
                    {{&gsl::at(all_targets[1], a)}},
                    {{&gsl::at(all_targets[2], a)}}}},
                  {{gsl::at(means, a)}});
    CHECK_ITERABLE_CUSTOM_APPROX(gsl::at(fitted_together, a), fitted[0],
                                 Approx::custom().epsilon(1e-12).scale(1.0));
  }

  // The provisional-rate entry point is the fit with the mass-weighted means
  // of the provisional rates, bitwise.
  std::array<DataVector, gp::number_of_potentials> provisional_rates{};
  std::array<const DataVector*, gp::number_of_potentials> provisional{};
  std::array<double, gp::number_of_potentials> provisional_means{};
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    gsl::at(provisional_rates, a) = random_field(gen, n);
    gsl::at(provisional, a) = &gsl::at(provisional_rates, a);
    gsl::at(provisional_means, a) =
        operators.mean(gsl::at(provisional_rates, a));
  }
  std::array<DataVector, gp::number_of_potentials> expected{};
  operators.fit(make_not_null(&expected), target_pointers, provisional_means);
  std::array<DataVector, gp::number_of_potentials> from_provisional{};
  operators.fit_from_provisional(make_not_null(&from_provisional),
                                 target_pointers, provisional);
  CHECK(from_provisional == expected);
}

void test_filter(const Geometry& geometry,
                 const gsl::not_null<std::mt19937*> gen) {
  const gp::LocalOperators operators{geometry.mesh, geometry.inverse_jacobian,
                                     geometry.det_inverse_jacobian, 4};
  const auto& extents = geometry.mesh.extents();
  CHECK(operators.number_of_retained_modes() ==
        (extents[0] - 1) * (extents[1] - 1) * (extents[2] - 1));

  // The projector is idempotent and self-adjoint in the mass inner product.
  const size_t n = geometry.mesh.number_of_grid_points();
  const DataVector x = random_field(gen, n);
  const DataVector y = random_field(gen, n);
  const DataVector projected_x = operators.project_retained(x);
  CHECK_ITERABLE_CUSTOM_APPROX(operators.project_retained(projected_x),
                               projected_x,
                               Approx::custom().epsilon(1e-12).scale(1.0));
  CHECK(sum(operators.mass() * projected_x * y) ==
        approx(sum(operators.mass() * x * operators.project_retained(y)))
            .epsilon(1e-12));

  for (const double gain : {0.0, 0.1, 0.37, 1.0}) {
    CAPTURE(gain);
    // The retained space (degree <= p_i - 1 in every direction) is fixed
    // exactly.
    for (size_t l = 0; l + 1 < extents[0]; ++l) {
      for (size_t m = 0; m + 1 < extents[1]; ++m) {
        for (size_t k = 0; k + 1 < extents[2]; ++k) {
          DataVector field = mode(geometry, l, m, k);
          const DataVector original = field;
          operators.apply_filter(make_not_null(&field), gain);
          CHECK_ITERABLE_CUSTOM_APPROX(
              field, original, Approx::custom().epsilon(1e-12).scale(1.0));
        }
      }
    }

    // The filter is the gain on the mass-orthogonal complement.
    DataVector complement = x - projected_x;
    const DataVector original_complement = complement;
    operators.apply_filter(make_not_null(&complement), gain);
    CHECK_ITERABLE_CUSTOM_APPROX(complement, gain * original_complement,
                                 Approx::custom().epsilon(1e-12).scale(1.0));
    DataVector filtered = x;
    operators.apply_filter(make_not_null(&filtered), gain);
    CHECK_ITERABLE_CUSTOM_APPROX(
        filtered, DataVector{projected_x + gain * (x - projected_x)},
        Approx::custom().epsilon(1e-12).scale(1.0));

    // Filtering several fields at once filters each of them, up to the
    // summation order of the matrix products; an empty list is a no-op.
    const std::vector<DataVector> originals{x, y, projected_x};
    std::vector<DataVector> batch = originals;
    std::vector<gsl::not_null<DataVector*>> batch_fields{};
    for (auto& field : batch) {
      batch_fields.emplace_back(&field);
    }
    operators.apply_filter(batch_fields, gain);
    for (size_t i = 0; i < batch.size(); ++i) {
      DataVector expected = originals[i];
      operators.apply_filter(make_not_null(&expected), gain);
      CHECK_ITERABLE_CUSTOM_APPROX(batch[i], expected,
                                   Approx::custom().epsilon(1e-12).scale(1.0));
    }
    operators.apply_filter(std::vector<gsl::not_null<DataVector*>>{}, gain);

    // At unit gain both versions of the filter leave the fields unchanged.
    if (gain == 1.0) {
      CHECK(filtered == x);
      for (size_t i = 0; i < batch.size(); ++i) {
        CHECK(batch[i] == originals[i]);
      }
    }
  }
}

// On an affine element the Legendre modes are orthogonal in the LGL mass
// inner product against all lower modes, so each top mode is damped by
// exactly the gain: the top mode in each direction, not only the corner.
void test_affine_shell(const Geometry& geometry) {
  const gp::LocalOperators operators{geometry.mesh, geometry.inverse_jacobian,
                                     geometry.det_inverse_jacobian, 4};
  const auto& e = geometry.mesh.extents();
  for (const double gain : {0.0, 0.1, 0.37, 1.0}) {
    CAPTURE(gain);
    for (const auto& [l, m, n] :
         std::vector<std::array<size_t, 3>>{{{e[0] - 1, 0, 0}},
                                            {{0, e[1] - 1, 0}},
                                            {{0, 0, e[2] - 1}},
                                            {{e[0] - 1, 1, 0}},
                                            {{1, e[1] - 1, e[2] - 1}},
                                            {{e[0] - 1, e[1] - 1, e[2] - 1}}}) {
      DataVector field = mode(geometry, l, m, n);
      const DataVector original = field;
      operators.apply_filter(make_not_null(&field), gain);
      CHECK_ITERABLE_CUSTOM_APPROX(field, gain * original,
                                   Approx::custom().epsilon(1e-12).scale(1.0));
    }
  }
  CHECK(operators.volume() == approx(geometry.exact_volume));
}

void test_serialization_and_mesh_check(const Geometry& geometry) {
  const gp::LocalOperators operators{geometry.mesh, geometry.inverse_jacobian,
                                     geometry.det_inverse_jacobian, 4};
  test_serialization(operators);
  CHECK_THROWS_WITH(
      operators.check_mesh(Mesh<3>{4, Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto}),
      Catch::Matchers::ContainsSubstring("Only static meshes are supported"));
  CHECK_THROWS_WITH(
      (gp::LocalOperators{
          Mesh<3>{3, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
          geometry.inverse_jacobian, geometry.det_inverse_jacobian, 4}),
      Catch::Matchers::ContainsSubstring("Legendre Gauss-Lobatto mesh"));
  CHECK(not operators.is_spherical_shell());
  const std::string only_on_shells =
      "is defined only on spherical-harmonic shells";
  CHECK_THROWS_WITH(operators.derivative_kernel(),
                    Catch::Matchers::ContainsSubstring(only_on_shells));
  const Matrix columns(geometry.mesh.number_of_grid_points(), 1, 0.0);
  CHECK_THROWS_WITH(operators.trial_functions(columns),
                    Catch::Matchers::ContainsSubstring(only_on_shells));
  CHECK_THROWS_WITH(operators.trial_functions_transpose(columns),
                    Catch::Matchers::ContainsSubstring(only_on_shells));
  CHECK_THROWS_WITH(operators.trial_function_gradients(columns),
                    Catch::Matchers::ContainsSubstring(only_on_shells));
  CHECK_THROWS_WITH(operators.trial_function_gradients_transpose(
                        {{columns, columns, columns}}),
                    Catch::Matchers::ContainsSubstring(only_on_shells));
}

struct ShellGeometry {
  Geometry geometry;
  tnsr::I<DataVector, 3, Frame::Inertial> inertial;
  bool linear;
};

// An element of a spherical shell 1.2 <= r <= 3.1 with the map of the
// NonconformingSphericalShells domain creator.
ShellGeometry make_shell_geometry(
    const size_t radial_size, const size_t l_max,
    const domain::CoordinateMaps::Distribution distribution) {
  using Interval = domain::CoordinateMaps::Interval;
  using Identity2D = domain::CoordinateMaps::Identity<2>;
  const double inner = 1.2;
  const double outer = 3.1;
  ShellGeometry result{};
  Geometry& geometry = result.geometry;
  geometry.mesh =
      Mesh<3>{{{radial_size, l_max + 1, 2 * l_max + 1}},
              {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
                Spectral::Basis::SphericalHarmonic}},
              {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
                Spectral::Quadrature::Equiangular}}};
  geometry.logical = logical_coordinates(geometry.mesh);
  const auto map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          domain::CoordinateMaps::ProductOf2Maps<Interval, Identity2D>{
              Interval{-1.0, 1.0, inner, outer, distribution, 0.0},
              Identity2D{}},
          domain::CoordinateMaps::SphericalToCartesianPfaffian{});
  result.inertial = map(geometry.logical);
  geometry.inverse_jacobian = map.inv_jacobian(geometry.logical);
  geometry.det_inverse_jacobian = determinant(geometry.inverse_jacobian);
  geometry.exact_volume = 4.0 * M_PI / 3.0 * (cube(outer) - cube(inner));
  result.linear = distribution == domain::CoordinateMaps::Distribution::Linear;
  return result;
}

// The shell geometry with smooth non-separable perturbations of the inverse
// Jacobian and, independently, of its determinant. These are not the
// geometry of a map, but the operators use them only pointwise.
ShellGeometry make_deformed_geometry(const ShellGeometry& shell) {
  ShellGeometry result = shell;
  result.linear = false;
  Geometry& geometry = result.geometry;
  const DataVector& xi = get<0>(geometry.logical);
  const DataVector& theta = get<1>(geometry.logical);
  const DataVector& phi = get<2>(geometry.logical);
  const auto& original = shell.geometry.inverse_jacobian;
  for (size_t a = 0; a < 3; ++a) {
    for (size_t i = 0; i < 3; ++i) {
      DataVector& component = geometry.inverse_jacobian.get(a, i);
      component = original.get(a, i);
      for (size_t b = 0; b < 3; ++b) {
        const double shift_a = 1.3 * static_cast<double>(a);
        const double shift_b = 0.7 * static_cast<double>(b);
        component += 0.15 * sin(xi * theta + shift_a + shift_b) *
                     cos(phi - (1.0 + shift_b) * xi + theta * phi) *
                     original.get(b, i);
      }
    }
  }
  get(geometry.det_inverse_jacobian) =
      get(determinant(geometry.inverse_jacobian)) *
      (1.0 + 0.3 * sin(xi * theta + 2.0 * phi) * cos(theta + xi * phi));
  return result;
}

// A random element of the shell trial space: random Legendre-mode
// coefficients times random valid Spherepack coefficients.
DataVector random_trial_function(const gsl::not_null<std::mt19937*> gen,
                                 const Mesh<3>& mesh) {
  std::normal_distribution<double> dist(0.0, 1.0);
  const size_t radial_size = mesh.extents(0);
  const size_t l_max = mesh.extents(1) - 1;
  const auto& ylm = ylm::get_spherepack_cache(l_max);
  const Matrix& radial_vandermonde =
      Spectral::modal_to_nodal_matrix(mesh.slice_through(0));
  DataVector result{mesh.number_of_grid_points(), 0.0};
  for (size_t k = 0; k < radial_size; ++k) {
    DataVector coefficients{ylm.spectral_size(), 0.0};
    for (ylm::SpherepackIterator iterator(l_max, l_max); iterator; ++iterator) {
      coefficients[iterator()] = dist(*gen);
    }
    const DataVector angular = ylm.spec_to_phys(coefficients);
    for (size_t j = 0; j < angular.size(); ++j) {
      for (size_t i = 0; i < radial_size; ++i) {
        result[i + radial_size * j] += radial_vandermonde(i, k) * angular[j];
      }
    }
  }
  return result;
}

Matrix random_matrix(const gsl::not_null<std::mt19937*> gen, const size_t rows,
                     const size_t columns) {
  std::normal_distribution<double> dist(0.0, 1.0);
  Matrix result(rows, columns);
  for (size_t column = 0; column < columns; ++column) {
    for (size_t row = 0; row < rows; ++row) {
      result(row, column) = dist(*gen);
    }
  }
  return result;
}

double max_abs(const DataVector& x) { return max(abs(x)); }

double max_abs(const Matrix& x) {
  double result = 0.0;
  for (size_t column = 0; column < x.columns(); ++column) {
    for (size_t row = 0; row < x.rows(); ++row) {
      result = std::max(result, std::abs(x(row, column)));
    }
  }
  return result;
}

// The plain l2 inner product and the sum of the absolute products.
std::pair<double, double> inner_product(const Matrix& x, const Matrix& y) {
  double product = 0.0;
  double scale = 0.0;
  for (size_t column = 0; column < x.columns(); ++column) {
    for (size_t row = 0; row < x.rows(); ++row) {
      product += x(row, column) * y(row, column);
      scale += std::abs(x(row, column) * y(row, column));
    }
  }
  return {product, scale};
}

DataVector column(const Matrix& matrix, const size_t index) {
  DataVector result{matrix.rows()};
  for (size_t row = 0; row < matrix.rows(); ++row) {
    result[row] = matrix(row, index);
  }
  return result;
}

using TargetArray = std::array<DataVector, 3>;

TargetArray random_targets(const gsl::not_null<std::mt19937*> gen,
                           const size_t size) {
  return {{random_field(gen, size), random_field(gen, size),
           random_field(gen, size)}};
}

DataVector shell_fit(const gp::LocalOperators& operators,
                     const TargetArray& targets,
                     const DataVector& provisional) {
  std::array<DataVector, 1> fitted{};
  operators.fit_from_provisional(
      make_not_null(&fitted),
      {{{{&targets[0]}}, {{&targets[1]}}, {{&targets[2]}}}}, {{&provisional}});
  return fitted[0];
}

// The dense reference fit, built independently of the production operators:
// the trial basis B from Spherepack synthesis, its derivatives from
// `partial_derivatives`, the mass-weighted QR M^{1/2} B = Q R, and the
// singular value decomposition of C = [M^{1/2} D_i B R^{-1}]_i. The fit is
// B R^{-1} [C^+ (M^{1/2} q*_i)_i + N N^T Q^T M^{1/2} r*] with N the right
// singular vectors of the singular values at most 1e-11 times the largest.
struct ShellOracle {
  DataVector sqrt_mass;
  Matrix basis;
  std::array<Matrix, 3> basis_gradients;
  Matrix orthonormal_basis;
  DataVector singular_values;
  size_t rank;
  Matrix left_singular_vectors;
  Matrix right_singular_vectors;
  Matrix kernel;
};

ShellOracle make_oracle(const Geometry& geometry, const DataVector& mass) {
  const Mesh<3>& mesh = geometry.mesh;
  const size_t n = mesh.number_of_grid_points();
  const size_t radial_size = mesh.extents(0);
  const size_t l_max = mesh.extents(1) - 1;
  const auto& ylm = ylm::get_spherepack_cache(l_max);
  std::vector<DataVector> harmonics{};
  DataVector coefficients{ylm.spectral_size(), 0.0};
  for (ylm::SpherepackIterator iterator(l_max, l_max); iterator; ++iterator) {
    coefficients[iterator()] = 1.0;
    harmonics.push_back(ylm.spec_to_phys(coefficients));
    coefficients[iterator()] = 0.0;
  }
  const size_t trial_size = radial_size * harmonics.size();
  const Matrix& radial_vandermonde =
      Spectral::modal_to_nodal_matrix(mesh.slice_through(0));

  ShellOracle result{};
  result.sqrt_mass = sqrt(mass);
  result.basis = Matrix(n, trial_size);
  for (size_t c = 0; c < harmonics.size(); ++c) {
    for (size_t k = 0; k < radial_size; ++k) {
      for (size_t j = 0; j < harmonics[c].size(); ++j) {
        for (size_t i = 0; i < radial_size; ++i) {
          result.basis(i + radial_size * j, k + radial_size * c) =
              radial_vandermonde(i, k) * harmonics[c][j];
        }
      }
    }
  }
  for (auto& gradient : result.basis_gradients) {
    gradient = Matrix(n, trial_size);
  }
  gp::Tags::PotentialRateTarget::type columns{n};
  gp::Tags::PotentialGradients::type gradients{n};
  for (size_t first = 0; first < trial_size;
       first += gp::number_of_potentials) {
    const size_t batch = std::min(gp::number_of_potentials, trial_size - first);
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      columns.get(a) =
          a < batch ? column(result.basis, first + a) : DataVector{n, 0.0};
    }
    gp::potential_gradients(make_not_null(&gradients), columns, mesh,
                            geometry.inverse_jacobian);
    for (size_t a = 0; a < batch; ++a) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t point = 0; point < n; ++point) {
          gsl::at(result.basis_gradients, i)(point, first + a) =
              gradients.get(i, a)[point];
        }
      }
    }
  }

  Matrix weighted_basis(n, trial_size);
  Matrix weighted_gradients(3 * n, trial_size);
  for (size_t c = 0; c < trial_size; ++c) {
    for (size_t point = 0; point < n; ++point) {
      weighted_basis(point, c) =
          result.sqrt_mass[point] * result.basis(point, c);
      for (size_t i = 0; i < 3; ++i) {
        weighted_gradients(point + n * i, c) =
            result.sqrt_mass[point] *
            gsl::at(result.basis_gradients, i)(point, c);
      }
    }
  }
  Matrix upper_triangular{};
  blaze::qr(weighted_basis, result.orthonormal_basis, upper_triangular);
  const Matrix inverse_upper_triangular = blaze::inv(upper_triangular);
  Matrix c_matrix = weighted_gradients * inverse_upper_triangular;
  Matrix left{};
  blaze::DynamicVector<double> singular_values{};
  Matrix right_transpose{};
  blaze::gesvd(c_matrix, left, singular_values, right_transpose, 'S', 'S');
  result.singular_values = DataVector{trial_size};
  for (size_t k = 0; k < trial_size; ++k) {
    result.singular_values[k] = singular_values[k];
  }
  result.rank = 0;
  while (result.rank < trial_size and
         singular_values[result.rank] > 1.0e-11 * singular_values[0]) {
    ++result.rank;
  }
  const size_t rank = result.rank;
  result.left_singular_vectors = blaze::submatrix(left, 0, 0, 3 * n, rank);
  result.right_singular_vectors =
      blaze::trans(blaze::submatrix(right_transpose, 0, 0, rank, trial_size));
  for (size_t k = 0; k < rank; ++k) {
    blaze::column(result.right_singular_vectors, k) /= singular_values[k];
  }
  result.kernel = blaze::trans(blaze::submatrix(right_transpose, rank, 0,
                                                trial_size - rank, trial_size));
  return result;
}

DataVector oracle_fit(const ShellOracle& oracle, const TargetArray& targets,
                      const DataVector& provisional) {
  const size_t n = oracle.sqrt_mass.size();
  blaze::DynamicVector<double> weighted_targets(3 * n);
  blaze::DynamicVector<double> weighted_rates(n);
  for (size_t point = 0; point < n; ++point) {
    weighted_rates[point] = oracle.sqrt_mass[point] * provisional[point];
    for (size_t i = 0; i < 3; ++i) {
      weighted_targets[point + n * i] =
          oracle.sqrt_mass[point] * gsl::at(targets, i)[point];
    }
  }
  const blaze::DynamicVector<double> projected =
      blaze::trans(oracle.left_singular_vectors) * weighted_targets;
  const blaze::DynamicVector<double> rate_coefficients =
      blaze::trans(oracle.orthonormal_basis) * weighted_rates;
  const blaze::DynamicVector<double> kernel_coefficients =
      blaze::trans(oracle.kernel) * rate_coefficients;
  const blaze::DynamicVector<double> coefficients =
      oracle.right_singular_vectors * projected +
      oracle.kernel * kernel_coefficients;
  const blaze::DynamicVector<double> values =
      oracle.orthonormal_basis * coefficients;
  DataVector result{n};
  for (size_t point = 0; point < n; ++point) {
    result[point] = values[point] / oracle.sqrt_mass[point];
  }
  return result;
}

Matrix oracle_kernel_functions(const ShellOracle& oracle) {
  Matrix result = oracle.orthonormal_basis * oracle.kernel;
  for (size_t k = 0; k < result.columns(); ++k) {
    for (size_t point = 0; point < result.rows(); ++point) {
      result(point, k) /= oracle.sqrt_mass[point];
    }
  }
  return result;
}

size_t expected_nullity(const size_t l_max) {
  return 1 + (l_max % 2 == 0 ? l_max + 1 : l_max);
}

// The cached angular operators: the synthesis of every valid harmonic, which
// the Spherepack analysis inverts; exact plain-l2 transposes of the cached
// derivative matrices; and the angular kernel.
void test_shell_angular_operators(const size_t l_max,
                                  const gsl::not_null<std::mt19937*> gen) {
  CAPTURE(l_max);
  const gp::ShellAngularOperators& angular = gp::shell_angular_operators(l_max);
  CHECK(&angular == &gp::shell_angular_operators(l_max));
  CHECK(angular.l_max == l_max);
  const auto& ylm = ylm::get_spherepack_cache(l_max);
  const size_t angular_size = ylm.physical_size();
  const size_t harmonics = square(l_max + 1);
  REQUIRE(angular.synthesis.rows() == angular_size);
  REQUIRE(angular.synthesis.columns() == harmonics);
  REQUIRE(angular.trial_derivatives.rows() == 2 * angular_size);
  REQUIRE(angular.trial_derivatives.columns() == harmonics);
  DataVector coefficients{ylm.spectral_size(), 0.0};
  size_t index = 0;
  for (ylm::SpherepackIterator iterator(l_max, l_max); iterator; ++iterator) {
    coefficients[iterator()] = 1.0;
    CHECK(column(angular.synthesis, index) == ylm.spec_to_phys(coefficients));
    CHECK_ITERABLE_CUSTOM_APPROX(
        ylm.phys_to_spec(column(angular.synthesis, index)), coefficients,
        Approx::custom().epsilon(1e-13).scale(1.0));
    coefficients[iterator()] = 0.0;
    ++index;
  }
  for (size_t a = 0; a < 2; ++a) {
    const Matrix& derivative = gsl::at(angular.derivatives, a);
    REQUIRE(derivative.rows() == angular_size);
    REQUIRE(derivative.columns() == angular_size);
    // The cached matrices are the derivatives of gradient_all_offsets.
    const DataVector u = random_field(gen, angular_size);
    const auto expected = ylm.gradient(u);
    DataVector applied{angular_size};
    for (size_t j = 0; j < angular_size; ++j) {
      applied[j] = blaze::dot(blaze::row(derivative, j), u);
    }
    CHECK_ITERABLE_CUSTOM_APPROX(
        applied, expected.get(a),
        Approx::custom().epsilon(1e-12).scale(max_abs(expected.get(a))));
    const Matrix x = random_matrix(gen, angular_size, 2);
    const Matrix y = random_matrix(gen, angular_size, 2);
    const auto [forward, forward_scale] = inner_product(derivative * x, y);
    const auto [backward, backward_scale] =
        inner_product(x, blaze::trans(derivative) * y);
    CHECK(std::abs(forward - backward) <
          1e-14 * (forward_scale + backward_scale));
  }
  // The analysis is not the plain transpose of the synthesis.
  {
    const DataVector values = random_field(gen, angular_size);
    const DataVector analysis = ylm.phys_to_spec(values);
    double largest_difference = 0.0;
    index = 0;
    for (ylm::SpherepackIterator iterator(l_max, l_max); iterator; ++iterator) {
      largest_difference =
          std::max(largest_difference,
                   std::abs(analysis[iterator()] -
                            blaze::dot(blaze::column(angular.synthesis, index),
                                       values)));
      ++index;
    }
    CHECK(largest_difference > 1e-3);
  }
  const Matrix& kernel = angular.kernel;
  CHECK(kernel.rows() == harmonics);
  CHECK(kernel.columns() == expected_nullity(l_max));
  CHECK(max_abs(Matrix{angular.trial_derivatives * kernel}) <
        1e-12 * max_abs(angular.trial_derivatives));
  const Matrix gram = blaze::trans(kernel) * kernel;
  for (size_t k = 0; k < kernel.columns(); ++k) {
    for (size_t l = 0; l < kernel.columns(); ++l) {
      CHECK(gram(k, l) == approx(k == l ? 1.0 : 0.0));
    }
  }
  // The kernel is spanned by the constant and the harmonics with l = L and
  // even m, both their cosine and sine parts: each of these unit coefficient
  // vectors lies in it, and every other one is orthogonal to it.
  index = 0;
  for (ylm::SpherepackIterator iterator(l_max, l_max); iterator; ++iterator) {
    CAPTURE(iterator.l());
    CAPTURE(iterator.m());
    const bool in_kernel =
        iterator.l() == 0 or (iterator.l() == l_max and iterator.m() % 2 == 0);
    double projection = 0.0;
    for (size_t k = 0; k < kernel.columns(); ++k) {
      projection += square(kernel(index, k));
    }
    CHECK(projection == approx(in_kernel ? 1.0 : 0.0));
    ++index;
  }
}

void test_shell_construction(const ShellGeometry& shell,
                             const gp::LocalOperators& operators) {
  const Geometry& geometry = shell.geometry;
  CHECK(operators.is_spherical_shell());
  CHECK(min(operators.mass()) > 0.0);
  CHECK(sum(operators.mass()) == approx(operators.volume()));
  CHECK(operators.volume() ==
        approx(definite_integral(
            DataVector{1.0 / get(geometry.det_inverse_jacobian)},
            geometry.mesh)));
  // r^2 dr/dxi is quadratic for the linear map.
  if (shell.linear and geometry.mesh.extents(0) >= 3) {
    CHECK(operators.volume() == approx(geometry.exact_volume));
  }
  const std::string not_on_shells =
      "is not implemented on spherical-harmonic shells";
  DataVector field{geometry.mesh.number_of_grid_points(), 1.0};
  CHECK_THROWS_WITH(operators.apply_filter(make_not_null(&field), 0.5),
                    Catch::Matchers::ContainsSubstring(not_on_shells));
  CHECK_THROWS_WITH(
      operators.apply_filter(
          std::vector<gsl::not_null<DataVector*>>{make_not_null(&field)}, 0.5),
      Catch::Matchers::ContainsSubstring(not_on_shells));
  CHECK_THROWS_WITH(operators.project_retained(field),
                    Catch::Matchers::ContainsSubstring(not_on_shells));
  CHECK_THROWS_WITH(operators.number_of_retained_modes(),
                    Catch::Matchers::ContainsSubstring(not_on_shells));
  std::array<DataVector, 1> fitted{};
  CHECK_THROWS_WITH(
      operators.fit(make_not_null(&fitted),
                    {{{{&field}}, {{&field}}, {{&field}}}}, {{0.0}}),
      Catch::Matchers::ContainsSubstring(not_on_shells));
  test_serialization(operators);
  CHECK(operators == gp::LocalOperators{geometry.mesh,
                                        geometry.inverse_jacobian,
                                        geometry.det_inverse_jacobian, 2});
}

void test_shell_mesh_validation(const ShellGeometry& shell) {
  const Geometry& geometry = shell.geometry;
  const auto make =
      [&geometry](const std::array<size_t, 3>& extents,
                  const std::array<Spectral::Quadrature, 3>& quadratures) {
        return gp::LocalOperators{
            Mesh<3>{
                extents,
                {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
                  Spectral::Basis::SphericalHarmonic}},
                quadratures},
            geometry.inverse_jacobian, geometry.det_inverse_jacobian, 2};
      };
  const std::array<Spectral::Quadrature, 3> native{
      {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
       Spectral::Quadrature::Equiangular}};
  const std::string extents_message =
      "requires extents (N_r, L + 1, 2 L + 1) with N_r >= 2 and L >= 2";
  CHECK_THROWS_WITH(make({{4, 5, 8}}, native),
                    Catch::Matchers::ContainsSubstring(extents_message));
  CHECK_THROWS_WITH(make({{4, 2, 3}}, native),
                    Catch::Matchers::ContainsSubstring(extents_message));
  CHECK_THROWS_WITH(make({{1, 5, 9}}, native),
                    Catch::Matchers::ContainsSubstring(extents_message));
  const std::string quadrature_message =
      "a SphericalHarmonic basis with Gauss and Equiangular quadratures";
  CHECK_THROWS_WITH(make({{4, 5, 9}}, {{Spectral::Quadrature::GaussLobatto,
                                        Spectral::Quadrature::Gauss,
                                        Spectral::Quadrature::Gauss}}),
                    Catch::Matchers::ContainsSubstring(quadrature_message));
  CHECK_THROWS_WITH(make({{4, 5, 9}}, {{Spectral::Quadrature::Gauss,
                                        Spectral::Quadrature::Gauss,
                                        Spectral::Quadrature::Equiangular}}),
                    Catch::Matchers::ContainsSubstring(quadrature_message));

  // A radial derivative invisible to the inertial derivatives enlarges the
  // kernel beyond the structural one.
  auto inverse_jacobian = geometry.inverse_jacobian;
  for (size_t i = 0; i < 3; ++i) {
    inverse_jacobian.get(0, i) = 0.0;
  }
  CHECK_THROWS_WITH((gp::LocalOperators{geometry.mesh, inverse_jacobian,
                                        geometry.det_inverse_jacobian, 2}),
                    Catch::Matchers::ContainsSubstring(
                        "too ill-conditioned for an accurate"));
}

// The structured trial basis and derivatives agree with the Spherepack
// synthesis and `partial_derivatives` on every basis function, and the
// transposes are exact plain-l2 transposes.
void test_shell_structured_operators(const ShellGeometry& shell,
                                     const gp::LocalOperators& operators,
                                     const ShellOracle& oracle,
                                     const gsl::not_null<std::mt19937*> gen) {
  const size_t n = shell.geometry.mesh.number_of_grid_points();
  const size_t trial_size = oracle.basis.columns();
  Matrix identity(trial_size, trial_size, 0.0);
  for (size_t k = 0; k < trial_size; ++k) {
    identity(k, k) = 1.0;
  }
  CHECK(max_abs(Matrix{operators.trial_functions(identity) - oracle.basis}) <
        1e-13 * max_abs(oracle.basis));
  const std::array<Matrix, 3> gradients =
      operators.trial_function_gradients(identity);
  for (size_t i = 0; i < 3; ++i) {
    const Matrix& expected = gsl::at(oracle.basis_gradients, i);
    CHECK(max_abs(Matrix{gsl::at(gradients, i) - expected}) <
          1e-12 * max_abs(expected));
  }

  const Matrix coefficients = random_matrix(gen, trial_size, 2);
  const Matrix values = random_matrix(gen, n, 2);
  {
    const auto [forward, forward_scale] =
        inner_product(operators.trial_functions(coefficients), values);
    const auto [backward, backward_scale] = inner_product(
        coefficients, operators.trial_functions_transpose(values));
    CHECK(std::abs(forward - backward) <
          1e-14 * (forward_scale + backward_scale));
  }
  {
    const std::array<Matrix, 3> fluxes{{random_matrix(gen, n, 2),
                                        random_matrix(gen, n, 2),
                                        random_matrix(gen, n, 2)}};
    const std::array<Matrix, 3> coefficient_gradients =
        operators.trial_function_gradients(coefficients);
    double forward = 0.0;
    double forward_scale = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      const auto [product, scale] =
          inner_product(gsl::at(coefficient_gradients, i), gsl::at(fluxes, i));
      forward += product;
      forward_scale += scale;
    }
    const auto [backward, backward_scale] = inner_product(
        coefficients, operators.trial_function_gradients_transpose(fluxes));
    CHECK(std::abs(forward - backward) <
          1e-14 * (forward_scale + backward_scale));
  }
}

// The production fit agrees with the dense reference fit, and the kernels
// span the same functions.
void test_shell_oracle(const ShellGeometry& shell,
                       const gp::LocalOperators& operators,
                       const ShellOracle& oracle,
                       const gsl::not_null<std::mt19937*> gen) {
  const Mesh<3>& mesh = shell.geometry.mesh;
  const size_t n = mesh.number_of_grid_points();
  const size_t l_max = mesh.extents(1) - 1;
  const size_t trial_size = oracle.basis.columns();
  const DataVector& mass = operators.mass();
  const size_t rank = oracle.rank;
  REQUIRE(rank < trial_size);
  const double gap =
      oracle.singular_values[rank - 1] / oracle.singular_values[rank];
  const double condition_number =
      oracle.singular_values[0] / oracle.singular_values[rank - 1];
  CAPTURE(rank);
  CAPTURE(gap);
  CAPTURE(condition_number);
  CHECK(trial_size - rank == expected_nullity(l_max));
  CHECK(gap >= 100.0);

  double largest_difference = 0.0;
  for (size_t sample = 0; sample < 3; ++sample) {
    const TargetArray targets = random_targets(gen, n);
    const DataVector provisional = random_field(gen, n);
    const DataVector expected = oracle_fit(oracle, targets, provisional);
    largest_difference =
        std::max(largest_difference,
                 max_abs(DataVector{shell_fit(operators, targets, provisional) -
                                    expected}) /
                     max_abs(expected));
  }
  CAPTURE(largest_difference);
  CHECK(largest_difference < 1e-12);

  const Matrix kernel = operators.derivative_kernel();
  const Matrix reference_kernel = oracle_kernel_functions(oracle);
  REQUIRE(kernel.columns() == expected_nullity(l_max));
  REQUIRE(reference_kernel.columns() == kernel.columns());
  for (const auto& [functions, basis] :
       {std::pair{&kernel, &reference_kernel},
        std::pair{&reference_kernel, &kernel}}) {
    for (size_t k = 0; k < functions->columns(); ++k) {
      DataVector residual = column(*functions, k);
      const DataVector function = residual;
      for (size_t l = 0; l < basis->columns(); ++l) {
        const DataVector basis_function = column(*basis, l);
        residual -= sum(mass * basis_function * function) * basis_function;
      }
      CHECK(max_abs(residual) < 1e-12 * max_abs(function));
    }
  }
}

// Exact recovery of trial-space fields from their production gradients, and
// of a linear function from its analytic gradient.
void test_shell_exact_recovery(const ShellGeometry& shell,
                               const gp::LocalOperators& operators,
                               const ShellOracle& oracle,
                               const gsl::not_null<std::mt19937*> gen) {
  const Geometry& geometry = shell.geometry;
  const size_t n = geometry.mesh.number_of_grid_points();
  gp::Tags::PotentialRateTarget::type exact{n};
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    exact.get(a) = random_trial_function(gen, geometry.mesh);
  }
  gp::Tags::GradientTarget::type targets{n};
  gp::potential_gradients(make_not_null(&targets), exact, geometry.mesh,
                          geometry.inverse_jacobian);
  std::array<std::array<const DataVector*, gp::number_of_potentials>, 3>
      target_pointers{};
  std::array<const DataVector*, gp::number_of_potentials> provisional{};
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    gsl::at(provisional, a) = &exact.get(a);
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(gsl::at(target_pointers, i), a) = &targets.get(i, a);
    }
  }
  std::array<DataVector, gp::number_of_potentials> fitted{};
  operators.fit_from_provisional(make_not_null(&fitted), target_pointers,
                                 provisional);
  double production_error = 0.0;
  double oracle_error = 0.0;
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    const double scale = max_abs(exact.get(a));
    production_error = std::max(
        production_error,
        max_abs(DataVector{gsl::at(fitted, a) - exact.get(a)}) / scale);
    const DataVector reference = oracle_fit(
        oracle, {{targets.get(0, a), targets.get(1, a), targets.get(2, a)}},
        exact.get(a));
    oracle_error = std::max(
        oracle_error, max_abs(DataVector{reference - exact.get(a)}) / scale);
  }
  CAPTURE(production_error);
  CAPTURE(oracle_error);
  CHECK(production_error < 1e-11);
  CHECK(oracle_error < 1e-11);

  if (shell.linear) {
    const auto& x = shell.inertial;
    const DataVector linear =
        0.3 + get<0>(x) - 2.0 * get<1>(x) + 0.5 * get<2>(x);
    const TargetArray gradient{
        {DataVector{n, 1.0}, DataVector{n, -2.0}, DataVector{n, 0.5}}};
    CHECK_ITERABLE_CUSTOM_APPROX(
        shell_fit(operators, gradient, linear), linear,
        Approx::custom().epsilon(1e-12).scale(max_abs(linear)));
  }
}

// The mean rule, the kernel M-projection identity, the kernel completion and
// the least-squares optimality for inconsistent targets and provisional
// rates outside the trial space.
void test_shell_completion(const ShellGeometry& shell,
                           const gp::LocalOperators& operators,
                           const gsl::not_null<std::mt19937*> gen) {
  const Geometry& geometry = shell.geometry;
  const Mesh<3>& mesh = geometry.mesh;
  const size_t n = mesh.number_of_grid_points();
  const DataVector& mass = operators.mass();
  const TargetArray targets = random_targets(gen, n);
  const DataVector provisional = random_field(gen, n);
  const DataVector fitted = shell_fit(operators, targets, provisional);
  const double scale = max_abs(fitted) + max_abs(provisional);

  // Mean rule.
  CHECK(operators.mean(fitted) ==
        approx(operators.mean(provisional)).scale(scale));

  // The kernel functions are M-orthonormal with numerically zero
  // derivatives, and the fit has the M-projections of the provisional rate
  // onto them.
  const Matrix kernel = operators.derivative_kernel();
  double reference_derivative = 0.0;
  {
    DataVector reference = random_trial_function(gen, mesh);
    reference /= std::sqrt(sum(mass * square(reference)));
    const auto derivative = partial_derivative(Scalar<DataVector>{reference},
                                               mesh, geometry.inverse_jacobian);
    for (size_t i = 0; i < 3; ++i) {
      reference_derivative =
          std::max(reference_derivative, max_abs(derivative.get(i)));
    }
  }
  std::vector<DataVector> kernel_functions{};
  for (size_t k = 0; k < kernel.columns(); ++k) {
    const DataVector function = column(kernel, k);
    const auto derivative = partial_derivative(Scalar<DataVector>{function},
                                               mesh, geometry.inverse_jacobian);
    for (size_t i = 0; i < 3; ++i) {
      CHECK(max_abs(derivative.get(i)) < 1e-10 * reference_derivative);
    }
    for (size_t l = 0; l < k; ++l) {
      CHECK(std::abs(sum(mass * function * kernel_functions[l])) < 1e-12);
    }
    CHECK(sum(mass * square(function)) == approx(1.0));
    CHECK(sum(mass * function * fitted) ==
          approx(sum(mass * function * provisional)).scale(scale));
    kernel_functions.push_back(function);
  }

  // Kernel content of the provisional rate is copied; a range-space trial
  // function in it changes nothing.
  std::normal_distribution<double> dist(0.0, 1.0);
  DataVector kernel_function{n, 0.0};
  for (const auto& function : kernel_functions) {
    kernel_function += dist(*gen) * function;
  }
  CHECK_ITERABLE_CUSTOM_APPROX(
      DataVector{shell_fit(operators, targets, provisional + kernel_function) -
                 fitted},
      kernel_function,
      Approx::custom().epsilon(1e-12).scale(scale + max_abs(kernel_function)));
  DataVector range_function = random_trial_function(gen, mesh);
  for (const auto& function : kernel_functions) {
    range_function -= sum(mass * function * range_function) * function;
  }
  CHECK_ITERABLE_CUSTOM_APPROX(
      shell_fit(operators, targets, provisional + range_function), fitted,
      Approx::custom().epsilon(1e-12).scale(scale + max_abs(range_function)));
  // The completion does not depend on the gradient targets.
  const TargetArray other_targets = random_targets(gen, n);
  const DataVector other_provisional = random_field(gen, n);
  CHECK_ITERABLE_CUSTOM_APPROX(
      DataVector{shell_fit(operators, targets, other_provisional) - fitted},
      DataVector{shell_fit(operators, other_targets, other_provisional) -
                 shell_fit(operators, other_targets, provisional)},
      Approx::custom().epsilon(1e-12).scale(scale));

  // Least-squares optimality: the weighted residual is orthogonal to the
  // gradients of the trial space, sum_i (D_i B)^T M (D_i r - q*_i) = 0.
  const auto fitted_gradient = partial_derivative(
      Scalar<DataVector>{fitted}, mesh, geometry.inverse_jacobian);
  std::array<Matrix, 3> residuals{};
  std::array<Matrix, 3> residual_scales{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(residuals, i) = Matrix(n, 1);
    gsl::at(residual_scales, i) = Matrix(n, 1);
    for (size_t point = 0; point < n; ++point) {
      gsl::at(residuals, i)(point, 0) =
          mass[point] *
          (fitted_gradient.get(i)[point] - gsl::at(targets, i)[point]);
      gsl::at(residual_scales, i)(point, 0) =
          mass[point] * (std::abs(fitted_gradient.get(i)[point]) +
                         std::abs(gsl::at(targets, i)[point]));
    }
  }
  const Matrix normal_residual =
      operators.trial_function_gradients_transpose(residuals);
  const Matrix normal_scale =
      operators.trial_function_gradients_transpose(residual_scales);
  CHECK(max_abs(normal_residual) < 1e-12 * max_abs(normal_scale));
  for (size_t sample = 0; sample < 4; ++sample) {
    const auto test_gradient =
        partial_derivative(Scalar<DataVector>{random_trial_function(gen, mesh)},
                           mesh, geometry.inverse_jacobian);
    double inner = 0.0;
    double target_scale = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      inner += sum(mass * test_gradient.get(i) *
                   (fitted_gradient.get(i) - gsl::at(targets, i)));
      target_scale +=
          sum(mass * abs(test_gradient.get(i)) *
              (abs(fitted_gradient.get(i)) + abs(gsl::at(targets, i))));
    }
    CHECK(std::abs(inner) < 1e-12 * target_scale);
  }
}

// The 6 x 5 decoding and 5 x 6 encoding matrices between the trace-free
// coordinates and the tnsr::ii storage, from stf_decode and stf_encode.
std::pair<Matrix, Matrix> stf_matrices() {
  using structure = tnsr::ii<DataVector, 3>::structure;
  const auto indices = structure::storage_to_tensor_index();
  Matrix decode(6, 5);
  Matrix encode(5, 6);
  for (size_t a = 0; a < 5; ++a) {
    std::array<double, 5> unit{};
    gsl::at(unit, a) = 1.0;
    const gp::Matrix3 decoded = gp::stf_decode(unit.data());
    for (size_t s = 0; s < 6; ++s) {
      decode(s, a) = decoded(indices[s][0], indices[s][1]);
    }
  }
  for (size_t s = 0; s < 6; ++s) {
    gp::Matrix3 unit_tensor(0.0);
    unit_tensor(indices[s][0], indices[s][1]) = 1.0;
    unit_tensor(indices[s][1], indices[s][0]) = 1.0;
    const auto encoded = gp::stf_encode(unit_tensor);
    for (size_t a = 0; a < 5; ++a) {
      encode(a, s) = gsl::at(encoded, a);
    }
  }
  return {decode, encode};
}

// The native nodal filter of the given rank on the Cartesian components
// (rank two: the trace-free coordinates), built independently of the
// production cache: Spherepack analysis, I + Delta from fill_filter and
// synthesis applied to every unit nodal vector. Row and column j + n_Omega c.
Matrix native_nodal_filter(const size_t rank, const size_t l_max,
                           const size_t modes_to_kill) {
  const auto& ylm = ylm::get_spherepack_cache(l_max);
  const size_t angular_size = ylm.physical_size();
  const size_t spectral_size = ylm.spectral_size();
  const size_t storage_components = rank == 0 ? 1 : (rank == 1 ? 3 : 6);
  SimpleSparseMatrix delta{};
  const auto normalization =
      ylm::TensorYlm::CoefficientNormalization::Spherepack;
  if (rank == 0) {
    ylm::TensorYlm::fill_filter<Scalar<DataVector>::structure>(
        make_not_null(&delta), l_max, modes_to_kill, std::nullopt,
        normalization);
  } else if (rank == 1) {
    ylm::TensorYlm::fill_filter<tnsr::i<DataVector, 3>::structure>(
        make_not_null(&delta), l_max, modes_to_kill, std::nullopt,
        normalization);
  } else {
    ylm::TensorYlm::fill_filter<tnsr::ii<DataVector, 3>::structure>(
        make_not_null(&delta), l_max, modes_to_kill, std::nullopt,
        normalization);
  }
  const size_t size = storage_components * angular_size;
  Matrix storage_filter(size, size);
  std::vector<double> spectral(storage_components * spectral_size);
  for (size_t column = 0; column < size; ++column) {
    for (size_t s = 0; s < storage_components; ++s) {
      DataVector values{angular_size, 0.0};
      if (column / angular_size == s) {
        values[column % angular_size] = 1.0;
      }
      const DataVector coefficients = ylm.phys_to_spec(values);
      for (size_t k = 0; k < spectral_size; ++k) {
        spectral[k + spectral_size * s] = coefficients[k];
      }
    }
    std::vector<double> filtered_spectral = spectral;
    delta.increment_multiply_on_right(make_not_null(&filtered_spectral), 0, 1,
                                      spectral, 0, 1);
    for (size_t s = 0; s < storage_components; ++s) {
      DataVector coefficients{spectral_size};
      for (size_t k = 0; k < spectral_size; ++k) {
        coefficients[k] = filtered_spectral[k + spectral_size * s];
      }
      const DataVector values = ylm.spec_to_phys(coefficients);
      for (size_t j = 0; j < angular_size; ++j) {
        storage_filter(j + angular_size * s, column) = values[j];
      }
    }
  }
  if (rank < 2) {
    return storage_filter;
  }
  const auto [decode, encode] = stf_matrices();
  Matrix result(5 * angular_size, 5 * angular_size, 0.0);
  for (size_t a = 0; a < 5; ++a) {
    for (size_t b = 0; b < 5; ++b) {
      for (size_t s = 0; s < 6; ++s) {
        for (size_t s2 = 0; s2 < 6; ++s2) {
          const double factor = encode(a, s) * decode(s2, b);
          if (factor != 0.0) {
            blaze::submatrix(result, a * angular_size, b * angular_size,
                             angular_size, angular_size) +=
                factor * blaze::submatrix(storage_filter, s * angular_size,
                                          s2 * angular_size, angular_size,
                                          angular_size);
          }
        }
      }
    }
  }
  return result;
}

// A W-orthonormal basis of the range of a nodal filter with `components`
// components, from the eigenvectors of W^{1/2} N W^{-1/2}, which must be
// symmetric with eigenvalues 0 and 1.
Matrix reference_retained_basis(const Matrix& nodal_filter,
                                const size_t l_max) {
  const std::vector<double>& weights =
      ylm::get_spherepack_cache(l_max).integration_weights();
  const size_t angular_size = weights.size();
  const size_t size = nodal_filter.rows();
  Matrix weighted(size, size);
  for (size_t column = 0; column < size; ++column) {
    for (size_t row = 0; row < size; ++row) {
      weighted(row, column) = std::sqrt(weights[row % angular_size]) *
                              nodal_filter(row, column) /
                              std::sqrt(weights[column % angular_size]);
    }
  }
  const Matrix antisymmetric = weighted - blaze::trans(weighted);
  CHECK(max_abs(antisymmetric) < 1e-11);
  Matrix symmetric = 0.5 * (weighted + blaze::trans(weighted));
  blaze::DynamicVector<double> eigenvalues(size);
  blaze::syev(symmetric, eigenvalues, 'V', 'L');
  std::vector<size_t> retained{};
  for (size_t k = 0; k < size; ++k) {
    CHECK(std::min(std::abs(eigenvalues[k]), std::abs(eigenvalues[k] - 1.0)) <
          1e-11);
    if (eigenvalues[k] > 0.5) {
      retained.push_back(k);
    }
  }
  Matrix result(size, retained.size());
  for (size_t a = 0; a < retained.size(); ++a) {
    for (size_t row = 0; row < size; ++row) {
      result(row, a) =
          symmetric(row, retained[a]) / std::sqrt(weights[row % angular_size]);
    }
  }
  return result;
}

// The production retained angular spaces agree with the independently built
// native nodal filters: the production bases are W-orthonormal and span the
// range of the native filter.
void test_shell_filter_angular_spaces(const size_t l_max,
                                      const size_t modes_to_kill) {
  CAPTURE(l_max);
  CAPTURE(modes_to_kill);
  const gp::ShellFilterAngularSpaces& spaces =
      gp::shell_filter_angular_spaces(l_max, modes_to_kill);
  CHECK(&spaces == &gp::shell_filter_angular_spaces(l_max, modes_to_kill));
  CHECK(spaces.l_max == l_max);
  CHECK(spaces.angular_modes_to_kill == modes_to_kill);
  const auto [decode, encode] = stf_matrices();
  CHECK(max_abs(Matrix{spaces.stf_decode_matrix - decode}) < 1e-15);
  CHECK(max_abs(Matrix{spaces.stf_encode_matrix - encode}) < 1e-15);
  const std::vector<double>& weights =
      ylm::get_spherepack_cache(l_max).integration_weights();
  const size_t angular_size = weights.size();
  const size_t harmonics = square(l_max + 1);
  for (size_t rank = 0; rank < 3; ++rank) {
    CAPTURE(rank);
    CAPTURE(gsl::at(spaces.self_adjointness_errors, rank));
    CAPTURE(gsl::at(spaces.projector_errors, rank));
    CHECK(gsl::at(spaces.self_adjointness_errors, rank) < 1e-11);
    CHECK(gsl::at(spaces.projector_errors, rank) < 1e-11);
    const Matrix& basis = gsl::at(spaces.retained_bases, rank);
    const size_t components = rank == 0 ? 1 : (rank == 1 ? 3 : 5);
    REQUIRE(basis.rows() == components * angular_size);
    // Zero removed bands is the identity only for scalars.
    if (rank == 0 and modes_to_kill == 0) {
      CHECK(basis.columns() == harmonics);
    } else {
      CHECK(basis.columns() < components * harmonics);
    }
    Matrix weighted_basis = basis;
    for (size_t a = 0; a < basis.columns(); ++a) {
      for (size_t row = 0; row < basis.rows(); ++row) {
        weighted_basis(row, a) *= weights[row % angular_size];
      }
    }
    Matrix gram = blaze::trans(basis) * weighted_basis;
    for (size_t a = 0; a < basis.columns(); ++a) {
      gram(a, a) -= 1.0;
    }
    CHECK(max_abs(gram) < 1e-12);
    const Matrix nodal_filter = native_nodal_filter(rank, l_max, modes_to_kill);
    const Matrix reference = reference_retained_basis(nodal_filter, l_max);
    CHECK(reference.columns() == basis.columns());
    // The native filter fixes the production basis, and the production
    // projector reproduces the reference basis.
    CHECK(max_abs(Matrix{nodal_filter * basis - basis}) < 1e-11);
    const Matrix projected =
        basis * Matrix{blaze::trans(weighted_basis) * reference};
    CHECK(max_abs(Matrix{projected - reference}) < 1e-11 * max_abs(reference));
  }
  // The trace-free coordinates carry the Frobenius norm.
  MAKE_GENERATOR(gen);
  std::normal_distribution<double> dist(0.0, 1.0);
  for (size_t sample = 0; sample < 5; ++sample) {
    gp::Matrix3 tensor{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        tensor(i, j) = tensor(j, i) = dist(gen);
      }
    }
    const double trace = tensor(0, 0) + tensor(1, 1) + tensor(2, 2);
    for (size_t i = 0; i < 3; ++i) {
      tensor(i, i) -= trace / 3.0;
    }
    const auto coordinates = gp::stf_encode(tensor);
    double coordinate_norm = 0.0;
    for (const double coordinate : coordinates) {
      coordinate_norm += square(coordinate);
    }
    double frobenius_norm = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        frobenius_norm += square(tensor(i, j));
      }
    }
    CHECK(coordinate_norm == approx(frobenius_norm));
  }
}

// The groups of the shell velocity filter: (first field, number of
// components, rank).
const std::array<std::array<size_t, 3>, 5> shell_filter_groups{
    {{{0, 5, 2}}, {{5, 1, 0}}, {{6, 1, 0}}, {{7, 3, 1}}, {{10, 3, 1}}}};

using FilterFields = std::array<DataVector, gp::number_of_shell_filter_fields>;

FilterFields random_filter_fields(const gsl::not_null<std::mt19937*> gen,
                                  const size_t size) {
  FilterFields result{};
  for (auto& field : result) {
    field = random_field(gen, size);
  }
  return result;
}

gp::ShellFilterFields pointers(const gsl::not_null<FilterFields*> fields) {
  return {{&(*fields)[0], &(*fields)[1], &(*fields)[2], &(*fields)[3],
           &(*fields)[4], &(*fields)[5], &(*fields)[6], &(*fields)[7],
           &(*fields)[8], &(*fields)[9], &(*fields)[10], &(*fields)[11],
           &(*fields)[12]}};
}

// The mass inner product summed over the groups.
double inner(const DataVector& mass, const FilterFields& x,
             const FilterFields& y) {
  double result = 0.0;
  for (size_t c = 0; c < x.size(); ++c) {
    result += sum(mass * gsl::at(x, c) * gsl::at(y, c));
  }
  return result;
}

double max_abs(const FilterFields& x) {
  double result = 0.0;
  for (const auto& field : x) {
    result = std::max(result, max_abs(field));
  }
  return result;
}

FilterFields difference(const FilterFields& x, const FilterFields& y) {
  FilterFields result{};
  for (size_t c = 0; c < x.size(); ++c) {
    gsl::at(result, c) = gsl::at(x, c) - gsl::at(y, c);
  }
  return result;
}

// The dense reference projectors of one group: the M-orthonormal factor Q of
// the weighted QR of the basis of S_A, ordered with the lower radial degrees
// first, and the replicated square root of the mass. Rows c N + p.
struct ReferenceProjector {
  Matrix q;
  size_t lower_size;
  DataVector sqrt_mass;
};

ReferenceProjector make_reference_projector(const Mesh<3>& mesh,
                                            const DataVector& mass,
                                            const Matrix& angular_basis) {
  const size_t n = mesh.number_of_grid_points();
  const size_t radial_size = mesh.extents(0);
  const size_t angular_size = n / radial_size;
  const size_t components = angular_basis.rows() / angular_size;
  const size_t dimension = angular_basis.columns();
  const Matrix& radial_vandermonde =
      Spectral::modal_to_nodal_matrix(mesh.slice_through(0));
  ReferenceProjector result{};
  result.sqrt_mass = DataVector{components * n};
  for (size_t c = 0; c < components; ++c) {
    for (size_t p = 0; p < n; ++p) {
      result.sqrt_mass[p + n * c] = std::sqrt(mass[p]);
    }
  }
  Matrix weighted_basis(components * n, radial_size * dimension);
  for (size_t k = 0; k < radial_size; ++k) {
    for (size_t a = 0; a < dimension; ++a) {
      for (size_t c = 0; c < components; ++c) {
        for (size_t j = 0; j < angular_size; ++j) {
          for (size_t i = 0; i < radial_size; ++i) {
            const size_t row = i + radial_size * j + n * c;
            weighted_basis(row, a + dimension * k) =
                result.sqrt_mass[row] * radial_vandermonde(i, k) *
                angular_basis(j + angular_size * c, a);
          }
        }
      }
    }
  }
  Matrix upper_triangular{};
  blaze::qr(weighted_basis, result.q, upper_triangular);
  result.lower_size = (radial_size - 1) * dimension;
  return result;
}

// Pi_0 x + gain (Pi_A - Pi_0) x for the group values x.
DataVector reference_filter(const ReferenceProjector& projector,
                            const DataVector& x, const double gain) {
  const blaze::DynamicVector<double> weighted = projector.sqrt_mass * x;
  blaze::DynamicVector<double> coefficients =
      blaze::trans(projector.q) * weighted;
  for (size_t k = projector.lower_size; k < coefficients.size(); ++k) {
    coefficients[k] *= gain;
  }
  const blaze::DynamicVector<double> projected = projector.q * coefficients;
  DataVector result{x.size()};
  for (size_t row = 0; row < x.size(); ++row) {
    result[row] = projected[row] / projector.sqrt_mass[row];
  }
  return result;
}

FilterFields reference_filter(const std::array<ReferenceProjector, 3>& ranks,
                              const FilterFields& x, const double gain) {
  const size_t n = x[0].size();
  FilterFields result{};
  for (const auto& [first, components, rank] : shell_filter_groups) {
    DataVector group{components * n};
    for (size_t c = 0; c < components; ++c) {
      for (size_t p = 0; p < n; ++p) {
        group[p + n * c] = gsl::at(x, first + c)[p];
      }
    }
    const DataVector filtered_group =
        reference_filter(gsl::at(ranks, rank), group, gain);
    for (size_t c = 0; c < components; ++c) {
      gsl::at(result, first + c) = DataVector{n};
      for (size_t p = 0; p < n; ++p) {
        gsl::at(result, first + c)[p] = filtered_group[p + n * c];
      }
    }
  }
  return result;
}

template <typename Filter>
FilterFields filtered(const Filter& filter, FilterFields x, const double gain) {
  filter.apply(pointers(make_not_null(&x)), gain);
  return x;
}

// The general shell velocity filter: the projector identities, the
// contraction and agreement with the dense reference for gains 0, 0.1 and 1,
// for every group; and, on round shells, agreement with the separable
// evaluation.
void test_shell_velocity_filter(const ShellGeometry& shell,
                                const size_t modes_to_kill, const bool deformed,
                                const gsl::not_null<std::mt19937*> gen) {
  const Mesh<3>& mesh = shell.geometry.mesh;
  const size_t n = mesh.number_of_grid_points();
  const size_t l_max = mesh.extents(1) - 1;
  CAPTURE(modes_to_kill);
  const gp::LocalOperators operators{mesh, shell.geometry.inverse_jacobian,
                                     shell.geometry.det_inverse_jacobian,
                                     modes_to_kill};
  CHECK(operators.shell_angular_modes_to_kill() == modes_to_kill);
  const DataVector& mass = operators.mass();
  const gp::GeneralShellFilter general{mesh, mass, modes_to_kill};
  const gp::SeparableShellFilter separable{mesh, mass, modes_to_kill};
  test_serialization(general);
  test_serialization(separable);
  if (deformed) {
    // The mass is not of the form rho(i) w_Omega(j).
    const auto& weights =
        ylm::get_spherepack_cache(l_max).integration_weights();
    const size_t radial_size = mesh.extents(0);
    double largest_deviation = 0.0;
    for (size_t i = 0; i < radial_size; ++i) {
      const double reference = mass[i] / weights[0];
      for (size_t j = 0; j < weights.size(); ++j) {
        largest_deviation = std::max(
            largest_deviation,
            std::abs(mass[i + radial_size * j] / weights[j] - reference) /
                reference);
      }
    }
    CAPTURE(largest_deviation);
    CHECK(largest_deviation > 0.01);
  }

  std::array<ReferenceProjector, 3> references{};
  for (size_t rank = 0; rank < 3; ++rank) {
    gsl::at(references, rank) = make_reference_projector(
        mesh, mass,
        reference_retained_basis(
            native_nodal_filter(rank, l_max, modes_to_kill), l_max));
  }

  const FilterFields x = random_filter_fields(gen, n);
  const FilterFields y = random_filter_fields(gen, n);
  // A deserialized filter must re-resolve its angular-space pointer, which
  // `operator==` does not compare: its action must match the original's
  // exactly, as must that of a deserialized shell `LocalOperators`.
  {
    CHECK(filtered(serialize_and_deserialize(general), x, 0.1) ==
          filtered(general, x, 0.1));
    CHECK(filtered(serialize_and_deserialize(separable), x, 0.1) ==
          filtered(separable, x, 0.1));
    const gp::LocalOperators deserialized_operators =
        serialize_and_deserialize(operators);
    FilterFields through_deserialized = x;
    deserialized_operators.apply_shell_filter(
        pointers(make_not_null(&through_deserialized)), 0.1);
    FilterFields through_original = x;
    operators.apply_shell_filter(pointers(make_not_null(&through_original)),
                                 0.1);
    CHECK(through_deserialized == through_original);
  }
  const double scale = max_abs(x);
  const auto inner_approx =
      Approx::custom().epsilon(1e-12).scale(inner(mass, x, x));
  const FilterFields projected = filtered(general, x, 1.0);
  const FilterFields lower = filtered(general, x, 0.0);
  // The reference projectors are M-self-adjoint, idempotent and nested.
  {
    const FilterFields reference_projected =
        reference_filter(references, x, 1.0);
    const FilterFields reference_lower = reference_filter(references, x, 0.0);
    CHECK(inner(mass, reference_projected, y) ==
          inner_approx(inner(mass, x, reference_filter(references, y, 1.0))));
    CHECK(inner(mass, reference_lower, y) ==
          inner_approx(inner(mass, x, reference_filter(references, y, 0.0))));
    CHECK(max_abs(
              difference(reference_filter(references, reference_projected, 1.0),
                         reference_projected)) < 1e-12 * scale);
    CHECK(max_abs(difference(reference_filter(references, reference_lower, 1.0),
                             reference_lower)) < 1e-12 * scale);
    CHECK(max_abs(
              difference(reference_filter(references, reference_projected, 0.0),
                         reference_lower)) < 1e-12 * scale);
  }
  double reference_difference = 0.0;
  for (const double gain : {0.0, 0.1, 1.0}) {
    CAPTURE(gain);
    const FilterFields result = filtered(general, x, gain);
    const FilterFields expected = reference_filter(references, x, gain);
    reference_difference =
        std::max(reference_difference,
                 max_abs(difference(result, expected)) / max_abs(expected));
    // Through LocalOperators: the explicit evaluations, and the dispatching
    // entry point, which must select the separable evaluation exactly when
    // the mass separability residual is at roundoff.
    FilterFields through_operators = x;
    operators.apply_shell_filter_general(
        pointers(make_not_null(&through_operators)), gain);
    CHECK(through_operators == result);
    FilterFields through_separable = x;
    operators.apply_shell_filter_separable(
        pointers(make_not_null(&through_separable)), gain);
    CHECK(through_separable == filtered(separable, x, gain));
    FilterFields through_dispatch = x;
    operators.apply_shell_filter(pointers(make_not_null(&through_dispatch)),
                                 gain);
    if (deformed) {
      CHECK(operators.shell_mass_separability_residual() > 0.01);
      CHECK(through_dispatch == through_operators);
    } else {
      CHECK(operators.shell_mass_separability_residual() <= 1.0e-13);
      CHECK(through_dispatch == through_separable);
    }
    // Contraction, summed over the groups.
    const FilterFields top = difference(projected, lower);
    CHECK(inner(mass, result, result) ==
          inner_approx(inner(mass, lower, lower) +
                       square(gain) * inner(mass, top, top)));
    CHECK(inner(mass, result, result) <= inner(mass, x, x));
    // Every gain removes the content outside S_A.
    CHECK(max_abs(difference(result, x)) > 0.1 * scale);
  }
  CAPTURE(reference_difference);
  CHECK(reference_difference < 1e-12);
  // The production projectors are M-self-adjoint, idempotent and nested.
  CHECK(inner(mass, projected, y) ==
        inner_approx(inner(mass, x, filtered(general, y, 1.0))));
  CHECK(inner(mass, lower, y) ==
        inner_approx(inner(mass, x, filtered(general, y, 0.0))));
  CHECK(max_abs(difference(filtered(general, projected, 1.0), projected)) <
        1e-12 * scale);
  CHECK(max_abs(difference(filtered(general, lower, 0.0), lower)) <
        1e-12 * scale);
  CHECK(max_abs(difference(filtered(general, projected, 0.0), lower)) <
        1e-12 * scale);
  CHECK(max_abs(difference(filtered(general, lower, 1.0), lower)) <
        1e-12 * scale);

  // The separable evaluation agrees on round shells and differs on the
  // deformed geometry.
  double separable_difference = 0.0;
  for (const double gain : {0.0, 0.1, 1.0}) {
    const FilterFields expected = filtered(general, x, gain);
    const FilterFields result = filtered(separable, x, gain);
    FilterFields through_operators = x;
    operators.apply_shell_filter_separable(
        pointers(make_not_null(&through_operators)), gain);
    CHECK(through_operators == result);
    separable_difference =
        std::max(separable_difference,
                 max_abs(difference(result, expected)) / max_abs(expected));
  }
  CAPTURE(separable_difference);
  if (deformed) {
    CHECK(separable_difference > 1e-4);
  } else {
    CHECK(separable_difference < 1e-12);
  }
}

void test_shell_filter_errors() {
  const ShellGeometry shell =
      make_shell_geometry(3, 3, domain::CoordinateMaps::Distribution::Linear);
  const Geometry& geometry = shell.geometry;
  CHECK_THROWS_WITH(
      (gp::LocalOperators{geometry.mesh, geometry.inverse_jacobian,
                          geometry.det_inverse_jacobian, 4}),
      Catch::Matchers::ContainsSubstring(
          "ShellAngularModesToKill = 4 of the shell velocity filter exceeds "
          "L = 3"));
  CHECK_THROWS_WITH(gp::shell_filter_angular_spaces(3, 4),
                    Catch::Matchers::ContainsSubstring(
                        "at most L removed angular bands, but got L = 3"));
  const gp::LocalOperators operators{geometry.mesh, geometry.inverse_jacobian,
                                     geometry.det_inverse_jacobian, 3};
  CHECK(operators != gp::LocalOperators{geometry.mesh,
                                        geometry.inverse_jacobian,
                                        geometry.det_inverse_jacobian, 2});
  test_serialization(operators);
  const Geometry cube = make_geometry({{3, 3, 3}}, true);
  const gp::LocalOperators cube_operators{cube.mesh, cube.inverse_jacobian,
                                          cube.det_inverse_jacobian, 7};
  CHECK(cube_operators.shell_angular_modes_to_kill() == 7);
  FilterFields fields{};
  for (auto& field : fields) {
    field = DataVector{cube.mesh.number_of_grid_points(), 1.0};
  }
  CHECK_THROWS_WITH(
      cube_operators.apply_shell_filter(pointers(make_not_null(&fields)), 0.5),
      Catch::Matchers::ContainsSubstring(
          "is defined only on spherical-harmonic shells"));
  CHECK_THROWS_WITH(cube_operators.apply_shell_filter_separable(
                        pointers(make_not_null(&fields)), 0.5),
                    Catch::Matchers::ContainsSubstring(
                        "is defined only on spherical-harmonic shells"));
  CHECK_THROWS_WITH(cube_operators.apply_shell_filter_general(
                        pointers(make_not_null(&fields)), 0.5),
                    Catch::Matchers::ContainsSubstring(
                        "is defined only on spherical-harmonic shells"));
  CHECK_THROWS_WITH(cube_operators.shell_mass_separability_residual(),
                    Catch::Matchers::ContainsSubstring(
                        "is defined only on spherical-harmonic shells"));
}

// Dense references of the tensor-product operators for the production mass,
// built otherwise independently of the production operators from the gradient
// matrices G_i of `partial_derivative` and from the retained Legendre modes of
// `mode`. The fit is the solution of the Lagrange system of the constrained
// least-squares problem,
//
//   [K_D  m] [r     ]   [b_D ]
//   [m^T  0] [lambda] = [V mu],
//
// with K_D = sum_i G_i^T M G_i and b_D = sum_i G_i^T M q*_i, which uses no
// basis of the mean-zero subspace. The retained projection is the
// mass-weighted least-squares projection M^{-1/2} Q Q^T M^{1/2} with
// M^{1/2} B = Q R, B the retained modes.
struct TensorProductReference {
  DataVector mass;
  std::array<Matrix, 3> gradients;
  Matrix lagrange_inverse;
  Matrix retained_basis;
};

TensorProductReference make_tensor_product_reference(const Geometry& geometry,
                                                     const DataVector& mass) {
  const size_t n = geometry.mesh.number_of_grid_points();
  const auto& extents = geometry.mesh.extents();
  TensorProductReference result{};
  result.mass = mass;
  for (auto& gradient_matrix : result.gradients) {
    gradient_matrix = Matrix(n, n);
  }
  for (size_t column = 0; column < n; ++column) {
    DataVector unit{n, 0.0};
    unit[column] = 1.0;
    const auto unit_gradient = gradient(geometry, unit);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t point = 0; point < n; ++point) {
        gsl::at(result.gradients, i)(point, column) =
            gsl::at(unit_gradient, i)[point];
      }
    }
  }
  Matrix lagrange(n + 1, n + 1, 0.0);
  for (const Matrix& gradient_matrix : result.gradients) {
    for (size_t column = 0; column < n; ++column) {
      for (size_t row = 0; row < n; ++row) {
        for (size_t point = 0; point < n; ++point) {
          lagrange(row, column) += gradient_matrix(point, row) * mass[point] *
                                   gradient_matrix(point, column);
        }
      }
    }
  }
  for (size_t point = 0; point < n; ++point) {
    lagrange(point, n) = mass[point];
    lagrange(n, point) = mass[point];
  }
  result.lagrange_inverse = blaze::inv(lagrange);

  const DataVector sqrt_mass = sqrt(mass);
  Matrix weighted_modes(n,
                        (extents[0] - 1) * (extents[1] - 1) * (extents[2] - 1));
  size_t mode_index = 0;
  for (size_t l = 0; l + 1 < extents[0]; ++l) {
    for (size_t m = 0; m + 1 < extents[1]; ++m) {
      for (size_t k = 0; k + 1 < extents[2]; ++k) {
        const DataVector weighted_mode = sqrt_mass * mode(geometry, l, m, k);
        for (size_t point = 0; point < n; ++point) {
          weighted_modes(point, mode_index) = weighted_mode[point];
        }
        ++mode_index;
      }
    }
  }
  Matrix upper_triangular{};
  blaze::qr(weighted_modes, result.retained_basis, upper_triangular);
  return result;
}

DataVector reference_fit(const TensorProductReference& reference,
                         const TargetArray& targets, const double mean) {
  const size_t n = reference.mass.size();
  blaze::DynamicVector<double> rhs(n + 1, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    const Matrix& gradient_matrix = gsl::at(reference.gradients, i);
    for (size_t column = 0; column < n; ++column) {
      for (size_t point = 0; point < n; ++point) {
        rhs[column] += gradient_matrix(point, column) * reference.mass[point] *
                       gsl::at(targets, i)[point];
      }
    }
  }
  rhs[n] = sum(reference.mass) * mean;
  const blaze::DynamicVector<double> solution =
      reference.lagrange_inverse * rhs;
  DataVector result{n};
  for (size_t point = 0; point < n; ++point) {
    result[point] = solution[point];
  }
  return result;
}

DataVector reference_projection(const TensorProductReference& reference,
                                const DataVector& field) {
  const size_t n = field.size();
  const DataVector sqrt_mass = sqrt(reference.mass);
  blaze::DynamicVector<double> weighted(n);
  for (size_t point = 0; point < n; ++point) {
    weighted[point] = sqrt_mass[point] * field[point];
  }
  const blaze::DynamicVector<double> coefficients =
      blaze::trans(reference.retained_basis) * weighted;
  const blaze::DynamicVector<double> projected =
      reference.retained_basis * coefficients;
  DataVector result{n};
  for (size_t point = 0; point < n; ++point) {
    result[point] = projected[point] / sqrt_mass[point];
  }
  return result;
}

// The tensor-product fit and filter against the dense references, for the
// operators as built and after serialization. Both sides are direct dense
// solves of small well-conditioned systems, compared with the 1e-12 of the
// consistency checks of `test_fit` and `test_filter`.
void test_tensor_product_reference(const Geometry& geometry,
                                   const gsl::not_null<std::mt19937*> gen) {
  const gp::LocalOperators built{geometry.mesh, geometry.inverse_jacobian,
                                 geometry.det_inverse_jacobian, 4};
  const auto deserialized = serialize_and_deserialize(built);
  const TensorProductReference reference =
      make_tensor_product_reference(geometry, built.mass());
  const size_t n = geometry.mesh.number_of_grid_points();
  const auto custom = Approx::custom().epsilon(1e-12).scale(1.0);

  // Arbitrary gradient targets and nonzero means of all potentials.
  std::array<TargetArray, gp::number_of_potentials> targets{};
  std::array<std::array<const DataVector*, gp::number_of_potentials>, 3>
      target_pointers{};
  std::array<double, gp::number_of_potentials> means{};
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    gsl::at(targets, a) = random_targets(gen, n);
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(gsl::at(target_pointers, i), a) =
          &gsl::at(gsl::at(targets, a), i);
    }
    gsl::at(means, a) = 0.3 * static_cast<double>(a) - 1.1;
  }
  // Fields with retained and removed content, split by the reference
  // projection.
  std::array<DataVector, 2> retained{};
  std::array<DataVector, 2> removed{};
  for (size_t field_index = 0; field_index < 2; ++field_index) {
    gsl::at(retained, field_index) =
        reference_projection(reference, random_field(gen, n));
    const DataVector field = random_field(gen, n);
    gsl::at(removed, field_index) =
        field - reference_projection(reference, field);
  }

  for (const auto* operators : {&built, &deserialized}) {
    std::array<DataVector, gp::number_of_potentials> fitted{};
    operators->fit(make_not_null(&fitted), target_pointers, means);
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      CHECK_ITERABLE_CUSTOM_APPROX(
          gsl::at(fitted, a),
          reference_fit(reference, gsl::at(targets, a), gsl::at(means, a)),
          custom);
    }
    std::array<DataVector, 1> fitted_single{};
    operators->fit(
        make_not_null(&fitted_single),
        {{{{&targets[1][0]}}, {{&targets[1][1]}}, {{&targets[1][2]}}}},
        {{means[1]}});
    CHECK_ITERABLE_CUSTOM_APPROX(fitted_single[0],
                                 reference_fit(reference, targets[1], means[1]),
                                 custom);

    const DataVector field = retained[0] + removed[0];
    CHECK_ITERABLE_CUSTOM_APPROX(operators->project_retained(field),
                                 retained[0], custom);
    for (const double gain : {0.0, 0.37, 1.0}) {
      CAPTURE(gain);
      DataVector filtered = field;
      operators->apply_filter(make_not_null(&filtered), gain);
      CHECK_ITERABLE_CUSTOM_APPROX(
          filtered, DataVector{retained[0] + gain * removed[0]}, custom);
      // Distinct fields with identical values are valid inputs.
      std::array<DataVector, 3> batch{
          {field, DataVector{retained[1] + removed[1]}, field}};
      const std::vector<gsl::not_null<DataVector*>> batch_fields{
          make_not_null(&batch[0]), make_not_null(&batch[1]),
          make_not_null(&batch[2])};
      operators->apply_filter(batch_fields, gain);
      CHECK_ITERABLE_CUSTOM_APPROX(
          batch[0], DataVector{retained[0] + gain * removed[0]}, custom);
      CHECK_ITERABLE_CUSTOM_APPROX(
          batch[1], DataVector{retained[1] + gain * removed[1]}, custom);
      CHECK_ITERABLE_CUSTOM_APPROX(
          batch[2], DataVector{retained[0] + gain * removed[0]}, custom);

      // Reject both adjacent and nonadjacent duplicates before modifying any
      // field, including when the gain is one.
      const auto before_error = batch;
      for (const size_t duplicate_index : std::array<size_t, 2>{{0, 2}}) {
        CAPTURE(duplicate_index);
        auto duplicate_fields = batch_fields;
        duplicate_fields.push_back(batch_fields[duplicate_index]);
        CHECK_THROWS_WITH(operators->apply_filter(duplicate_fields, gain),
                          Catch::Matchers::ContainsSubstring(
                              "requires distinct field pointers"));
        CHECK(batch == before_error);
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection.LocalOperators",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  for (const auto& extents : std::vector<std::array<size_t, 3>>{
           {{2, 2, 2}}, {{3, 3, 3}}, {{3, 4, 5}}, {{6, 6, 6}}}) {
    for (const bool curved : {false, true}) {
      CAPTURE(extents);
      CAPTURE(curved);
      const Geometry geometry = make_geometry(extents, curved);
      test_fit(geometry, make_not_null(&gen));
      test_filter(geometry, make_not_null(&gen));
      if (not curved) {
        test_affine_shell(geometry);
      }
    }
  }
  test_serialization_and_mesh_check(make_geometry({{3, 3, 3}}, true));

  for (const size_t l_max : {2_st, 3_st, 4_st, 5_st, 6_st}) {
    test_shell_angular_operators(l_max, make_not_null(&gen));
  }
  // The parity rule of the angular derivative kernel at values between the
  // tested grids and the candidate resolution; the construction itself
  // ERRORs on a count or gap violation.
  for (const size_t l_max : {7_st, 8_st, 10_st}) {
    CHECK(gp::shell_angular_operators(l_max).kernel.columns() ==
          1 + (l_max % 2 == 0 ? l_max + 1 : l_max));
  }
  using domain::CoordinateMaps::Distribution;
  for (const auto& [radial_size, l_max, distribution, deformed] :
       std::vector<std::tuple<size_t, size_t, Distribution, bool>>{
           {2, 3, Distribution::Linear, false},
           {3, 2, Distribution::Linear, false},
           {3, 3, Distribution::Logarithmic, false},
           {4, 4, Distribution::Logarithmic, false},
           {5, 3, Distribution::Linear, false},
           {3, 6, Distribution::Logarithmic, false},
           {4, 3, Distribution::Logarithmic, true},
           {3, 4, Distribution::Linear, true}}) {
    CAPTURE(radial_size);
    CAPTURE(l_max);
    CAPTURE(distribution);
    CAPTURE(deformed);
    const ShellGeometry round_shell =
        make_shell_geometry(radial_size, l_max, distribution);
    const ShellGeometry shell =
        deformed ? make_deformed_geometry(round_shell) : round_shell;
    const gp::LocalOperators operators{shell.geometry.mesh,
                                       shell.geometry.inverse_jacobian,
                                       shell.geometry.det_inverse_jacobian, 2};
    const ShellOracle oracle = make_oracle(shell.geometry, operators.mass());
    test_shell_construction(shell, operators);
    test_shell_structured_operators(shell, operators, oracle,
                                    make_not_null(&gen));
    test_shell_oracle(shell, operators, oracle, make_not_null(&gen));
    test_shell_exact_recovery(shell, operators, oracle, make_not_null(&gen));
    test_shell_completion(shell, operators, make_not_null(&gen));
  }
  test_shell_mesh_validation(
      make_shell_geometry(3, 3, domain::CoordinateMaps::Distribution::Linear));
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection.ShellVelocityFilter",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  using domain::CoordinateMaps::Distribution;
  for (const auto& [l_max, modes_to_kill] :
       std::vector<std::pair<size_t, size_t>>{{2, 2}, {3, 0}, {4, 2}, {5, 1}}) {
    test_shell_filter_angular_spaces(l_max, modes_to_kill);
  }
  {
    // The retained dimensions at the candidate resolution.
    const auto& spaces = gp::shell_filter_angular_spaces(10, 4);
    CHECK(spaces.retained_bases[0].columns() == 49);
    CHECK(spaces.retained_bases[1].columns() == 145);
    CHECK(spaces.retained_bases[2].columns() == 235);
  }
  for (const auto& [radial_size, l_max, distribution, deformed, modes_to_kill] :
       std::vector<std::tuple<size_t, size_t, Distribution, bool, size_t>>{
           {3, 2, Distribution::Linear, false, 2},
           {3, 7, Distribution::Linear, false, 4},
           {4, 4, Distribution::Logarithmic, false, 2},
           {3, 3, Distribution::Logarithmic, false, 0},
           {2, 4, Distribution::Linear, false, 1},
           {4, 3, Distribution::Logarithmic, true, 1},
           {3, 4, Distribution::Linear, true, 2}}) {
    CAPTURE(radial_size);
    CAPTURE(l_max);
    CAPTURE(distribution);
    CAPTURE(deformed);
    const ShellGeometry round_shell =
        make_shell_geometry(radial_size, l_max, distribution);
    test_shell_velocity_filter(
        deformed ? make_deformed_geometry(round_shell) : round_shell,
        modes_to_kill, deformed, make_not_null(&gen));
  }
  test_shell_filter_errors();
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection."
    "LocalOperatorsTensorProductReference",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  // The minimal mesh, whose only retained mode is the constant, and an
  // anisotropic one, both with the nonseparable mass of the curved map.
  for (const auto& extents :
       std::vector<std::array<size_t, 3>>{{{2, 2, 2}}, {{3, 4, 5}}}) {
    CAPTURE(extents);
    test_tensor_product_reference(make_geometry(extents, true),
                                  make_not_null(&gen));
  }
}
