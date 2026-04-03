// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Filtering.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral::filtering {
Matrix exponential_filter(const Mesh<1>& mesh, const double alpha,
                          const unsigned half_power) {
  if (UNLIKELY(mesh.number_of_grid_points() == 1)) {
    return Matrix(1, 1, 1.0);
  }
  const Matrix& nodal_to_modal = Spectral::nodal_to_modal_matrix(mesh);
  const Matrix& modal_to_nodal = Spectral::modal_to_nodal_matrix(mesh);
  Matrix filter_matrix(mesh.number_of_grid_points(),
                       mesh.number_of_grid_points(), 0.0);
  const double order = mesh.number_of_grid_points() - 1.0;
  for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
    filter_matrix(i, i) = exp(-alpha * pow(i / order, 2 * half_power));
  }
  return modal_to_nodal * filter_matrix * nodal_to_modal;
}

Matrix cg_filter(const Mesh<1>& mesh, const double alpha,
                 const unsigned half_power) {
  ASSERT(mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto,
         "CG filtering only works with Gauss-Lobatto points, but got "
             << mesh.quadrature(0));

  const size_t npts = mesh.number_of_grid_points();
  const double order = static_cast<double>(npts - 1);

  const Matrix& nodal_to_modal = Spectral::nodal_to_modal_matrix(mesh);
  const Matrix& modal_to_nodal = Spectral::modal_to_nodal_matrix(mesh);
  const auto& xi = Spectral::collocation_points(mesh);

  auto sigma = [alpha, half_power, order](const size_t j) -> double {
    return exp(-alpha * pow(static_cast<double>(j) / order,
                            2.0 * static_cast<double>(half_power)));
  };

  // Matrix that builds the endpoint-preserving linear polynomial
  // p_i = ((1 - xi_i)/2) * u_L + ((1 + xi_i)/2) * u_R
  auto lift_matrix = [&xi, npts]() -> Matrix {
    Matrix lift(npts, npts, 0.0);
    for (size_t i = 0; i < npts; ++i) {
      lift(i, 0) = 0.5 * (1.0 - xi[i]);
      lift(i, npts - 1) = 0.5 * (1.0 + xi[i]);
    }
    return lift;
  };

  auto identity_matrix = [npts]() -> Matrix {
    Matrix id(npts, npts, 0.0);
    for (size_t i = 0; i < npts; ++i) {
      id(i, i) = 1.0;
    }
    return id;
  };

  // Boyd recursion: unfiltered modal coefficients a_j -> filtered coefficients
  // \bar{a}_j
  auto boyd_modal_coefficients =
      [npts, &sigma](const std::vector<double>& a) -> std::vector<double> {
    std::vector<double> a_bar = a;  // leaves a_0 and a_1 unchanged

    for (size_t parity = 0; parity < 2; ++parity) {
      int N = static_cast<int>(npts) - 1;
      if (N % 2 != static_cast<int>(parity)) {
        --N;
      }

      if (N <= 1) {
        continue;
      }

      // Eq. (8): lambda = sigma_{N-2} a_N, b_{N-2} = a_N, ā_N = lambda, rho =
      // lambda
      double b_j = a[static_cast<size_t>(N)];
      double rho = sigma(static_cast<size_t>(N - 2)) * b_j;
      a_bar[static_cast<size_t>(N)] = rho;

      // Eq. (9): j = N-2, N-4, ...
      for (int j = N - 2; j > 1; j -= 2) {
        b_j += a[static_cast<size_t>(j)];
        const double lambda = sigma(static_cast<size_t>(j - 2)) * b_j;
        a_bar[static_cast<size_t>(j)] = lambda - rho;
        rho = lambda;
      }
    }

    return a_bar;
  };

  // Build the modal-space operator that maps a -> a_bar
  auto boyd_modal_filter_matrix = [&boyd_modal_coefficients, npts]() -> Matrix {
    Matrix modal_filter(npts, npts, 0.0);
    for (size_t col = 0; col < npts; ++col) {
      std::vector<double> a(npts, 0.0);
      a[col] = 1.0;
      const auto a_bar = boyd_modal_coefficients(a);
      for (size_t row = 0; row < npts; ++row) {
        modal_filter(row, col) = a_bar[row];
      }
    }
    return modal_filter;
  };

  const Matrix lift = lift_matrix();
  const Matrix identity = identity_matrix();
  const Matrix residual_projector = identity - lift;
  const Matrix modal_filter = boyd_modal_filter_matrix();

  // u -> p + F(u - p)
  return lift +
         modal_to_nodal * modal_filter * nodal_to_modal * residual_projector;
}

namespace {
template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
struct ZeroLowestModesImpl {
  Matrix operator()(const size_t num_points,
                    const size_t num_modes_to_zero) const {
    const Matrix& nodal_to_modal =
        Spectral::nodal_to_modal_matrix<BasisType, QuadratureType>(num_points);
    const Matrix& modal_to_nodal =
        Spectral::modal_to_nodal_matrix<BasisType, QuadratureType>(num_points);
    Matrix filter_matrix(num_points, num_points, 0.0);
    for (size_t i = num_modes_to_zero; i < num_points; ++i) {
      filter_matrix(i, i) = 1.0;
    }
    return Matrix(modal_to_nodal * filter_matrix * nodal_to_modal);
  }
};
}  // namespace

const Matrix& zero_lowest_modes(const Mesh<1>& mesh,
                                const size_t number_of_modes_to_zero) {
  ASSERT(number_of_modes_to_zero < mesh.extents(0),
         "For a 1d mesh with " << mesh.extents(0)
                               << " grid points, you cannot zero "
                               << number_of_modes_to_zero << " modes.");

  switch (mesh.basis(0)) {
    case Basis::Legendre:
      switch (mesh.quadrature(0)) {
        case Spectral::Quadrature::GaussLobatto: {
          constexpr size_t max_num_points =
              Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
          constexpr size_t min_num_points = Spectral::minimum_number_of_points<
              Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>;
          const auto cache =
              make_static_cache<CacheRange<min_num_points, max_num_points + 1>,
                                CacheRange<0_st, max_num_points>>(
                  ZeroLowestModesImpl<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto>{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero);
        }
        case Spectral::Quadrature::Gauss: {
          constexpr size_t max_num_points =
              Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
          constexpr size_t min_num_points =
              Spectral::minimum_number_of_points<Spectral::Basis::Legendre,
                                                 Spectral::Quadrature::Gauss>;
          const auto cache =
              make_static_cache<CacheRange<min_num_points, max_num_points + 1>,
                                CacheRange<0_st, max_num_points>>(
                  ZeroLowestModesImpl<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::Gauss>{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero);
        }
        default:
          ERROR("Unsupported quadrature type in filtering lowest modes: "
                << mesh.quadrature(0));
      };
    case Basis::Chebyshev:
      switch (mesh.quadrature(0)) {
        case Spectral::Quadrature::GaussLobatto: {
          constexpr size_t max_num_points =
              Spectral::maximum_number_of_points<Spectral::Basis::Chebyshev>;
          constexpr size_t min_num_points = Spectral::minimum_number_of_points<
              Spectral::Basis::Chebyshev, Spectral::Quadrature::GaussLobatto>;
          const auto cache =
              make_static_cache<CacheRange<min_num_points, max_num_points + 1>,
                                CacheRange<0_st, max_num_points>>(
                  ZeroLowestModesImpl<Spectral::Basis::Chebyshev,
                                      Spectral::Quadrature::GaussLobatto>{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero);
        }
        case Spectral::Quadrature::Gauss: {
          constexpr size_t max_num_points =
              Spectral::maximum_number_of_points<Spectral::Basis::Chebyshev>;
          constexpr size_t min_num_points =
              Spectral::minimum_number_of_points<Spectral::Basis::Chebyshev,
                                                 Spectral::Quadrature::Gauss>;
          const auto cache =
              make_static_cache<CacheRange<min_num_points, max_num_points + 1>,
                                CacheRange<0_st, max_num_points>>(
                  ZeroLowestModesImpl<Spectral::Basis::Chebyshev,
                                      Spectral::Quadrature::Gauss>{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero);
        }
        default:
          ERROR("Unsupported quadrature type in filtering lowest modes: "
                << mesh.quadrature(0));
      };
    default:
      ERROR("Cannot filter basis type: " << mesh.basis(0));
  };
}
}  // namespace Spectral::filtering
