// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"

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
                                     geometry.det_inverse_jacobian};
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
}

void test_filter(const Geometry& geometry,
                 const gsl::not_null<std::mt19937*> gen) {
  const gp::LocalOperators operators{geometry.mesh, geometry.inverse_jacobian,
                                     geometry.det_inverse_jacobian};
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
                                     geometry.det_inverse_jacobian};
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
                                     geometry.det_inverse_jacobian};
  test_serialization(operators);
  CHECK_THROWS_WITH(
      operators.check_mesh(Mesh<3>{4, Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto}),
      Catch::Matchers::ContainsSubstring("Only static meshes are supported"));
  CHECK_THROWS_WITH(
      (gp::LocalOperators{
          Mesh<3>{3, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
          geometry.inverse_jacobian, geometry.det_inverse_jacobian}),
      Catch::Matchers::ContainsSubstring("Legendre Gauss-Lobatto mesh"));
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
}
