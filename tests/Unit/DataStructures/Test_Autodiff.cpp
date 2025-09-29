// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/utils/derivative.hpp>
#include <autodiff/forward/utils/gradient.hpp>
#include <autodiff/reverse/var.hpp>
#include <chrono>
#include <type_traits>
#include <xsimd/xsimd.hpp>

#include <iostream>

#include "DataStructures/DynamicVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {

using Affine = domain::CoordinateMaps::Affine;
using Interval = domain::CoordinateMaps::Interval;
using Wedge3D = domain::CoordinateMaps::Wedge<3>;
using Distribution = domain::CoordinateMaps::Distribution;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
using Interval3D =
    domain::CoordinateMaps::ProductOf3Maps<Interval, Interval, Interval>;

template <typename DataType>
DataType f1(const DataType& x, const double param) {
  return square(x) * param;
}

template <typename DataType>
DataType f2(const std::array<DataType, 2>& x) {
  return x[0] * square(x[1]);
}
template <typename DataType>
std::array<DataType, 2> f3(const std::array<DataType, 2>& x) {
  return {{square(x[0]) * x[1], cube(x[1])}};
}
template <typename DataType>
DataType f4(const tnsr::I<DataType, 2>& x, const Scalar<DataType>& y) {
  const auto f = tenex::evaluate<ti::I>(x(ti::I) * y());
  return get<0>(f);
}
template <typename DataType>
blaze::DynamicVector<autodiff::real> f5(
    const blaze::DynamicVector<autodiff::real>& x) {
  return {{square(x[0]) * x[1], cube(x[1]), x[0] + x[1]}};
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Autodiff", "[Unit][DataStructures]") {
  // Test that autodiff works with the autodiff::real type in forward mode.
  // The autodiff::dual type supports higher cross-derivatives in forward mode.
  // Reverse mode with the autodiff::var type needs some adaptors to work with
  // Blaze vectors and tensors, similar to how it is implemented for Eigen in
  // the autodiff library (it works fine with single numbers though, if needed).
  {
    INFO("Profiling Forward Mode");
    const size_t points_per_dimension = 40;
    const Mesh<3> mesh{points_per_dimension, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
    std::array<DataVector, 3> logical_coords{};
    for (size_t i = 0; i < 3; ++i) {
      logical_coords[i] = logical_coordinates(mesh).get(i);
    }
    // const auto coord_map =
    //   Affine3D{Affine{-1.0, 1.0, -1.0, 1.0}, Affine{-1.0, 1.0, 0.0, 2.0},
    //            Affine{-1.0, 1.0, -2.0, 2.0}};
    // const auto coord_map =
    //   Interval3D{Interval{-1.0, 1.0, -1.0, 1.0, Distribution::Equiangular},
    //              Interval{-1.0, 1.0, 0.0, 2.0, Distribution::Equiangular},
    //              Interval{-1.0, 1.0, -2.0, 2.0, Distribution::Equiangular}};
    const auto coord_map =
        Wedge3D{1.0, 2.0, 0.0, 1.0, OrientationMap<3>::create_aligned(), true};

auto total_start = std::chrono::high_resolution_clock::now();
    const auto expected_jacobian = coord_map.jacobian(logical_coords);
auto total_end = std::chrono::high_resolution_clock::now();
auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
    total_end - total_start);
std::cout << "SpECTRE jacobian - Total time: " << total_time.count() / 1000.0
          << " ms" << std::endl;

total_start = std::chrono::high_resolution_clock::now();
    auto actual_jacobian =
        make_with_value<tnsr::Ij<DataVector, 3>>(expected_jacobian, 0.0);

    using b_type = xsimd::batch<double>;
    using FirstOrderDual = autodiff::HigherOrderDual<1, b_type>;

    for (size_t i = 0; i < logical_coords[0].size(); i += b_type::size) {
      for (size_t k = 0; k < 3; ++k) {
        std::array<FirstOrderDual, 3>
          log_coords{b_type::load_aligned(&logical_coords[0][i]),
                    b_type::load_aligned(&logical_coords[1][i]),
                    b_type::load_aligned(&logical_coords[2][i])};

        // Seed the k-th coordinate for differentiation
        autodiff::detail::seed<1>(log_coords[k], 1.0);

        const auto y = coord_map(log_coords);
        for (size_t j = 0; j < 3; ++j) {
          // Extract the derivative part directly from the coordinate map
          const auto deriv_jk = autodiff::derivative(gsl::at(y, j));
          deriv_jk.store_aligned(&actual_jacobian.get(j, k)[i]);
        }
      }
    }

total_end = std::chrono::high_resolution_clock::now();
total_time = std::chrono::duration_cast<std::chrono::microseconds>(
        total_end - total_start);

std::cout << "Forward Mode - Total time: " << total_time.count() / 1000.0
          << " ms" << std::endl;

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        CHECK_ITERABLE_APPROX(actual_jacobian.get(i, j),
                              expected_jacobian.get(i, j));
      }
    }
  }
}
