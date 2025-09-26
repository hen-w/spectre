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
    INFO("Profiling Reverse Mode");
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
    const auto expected_jacobian = coord_map.jacobian(logical_coords);

    auto total_start = std::chrono::high_resolution_clock::now();
    auto actual_jacobian =
        make_with_value<tnsr::Ij<DataVector, 3>>(expected_jacobian, 0.0);

    for (size_t i = 0; i < logical_coords[0].size(); ++i) {
      std::array<autodiff::var, 3> autodiff_vars{
          logical_coords[0][i], logical_coords[1][i], logical_coords[2][i]};

      const auto y = coord_map(autodiff_vars);

      for (size_t j = 0; j < 3; ++j) {
        const auto deriv_j = autodiff::derivatives(
            gsl::at(y, j),
            autodiff::reverse::detail::wrt(autodiff_vars[0], autodiff_vars[1],
                                           autodiff_vars[2]));

        actual_jacobian.get(j, 0)[i] = gsl::at(deriv_j, 0);
        actual_jacobian.get(j, 1)[i] = gsl::at(deriv_j, 1);
        actual_jacobian.get(j, 2)[i] = gsl::at(deriv_j, 2);
      }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
        total_end - total_start);

    std::cout << "Reverse Mode - Total time: " << total_time.count() / 1000.0
              << " ms" << std::endl;

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        CHECK_ITERABLE_APPROX(actual_jacobian.get(i, j),
                              expected_jacobian.get(i, j));
      }
    }
  }
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
    const auto expected_jacobian = coord_map.jacobian(logical_coords);

    auto total_start = std::chrono::high_resolution_clock::now();
    auto actual_jacobian =
        make_with_value<tnsr::Ij<DataVector, 3>>(expected_jacobian, 0.0);

    for (size_t i = 0; i < logical_coords[0].size(); ++i) {
      // We will do 3 forward-mode evaluations (one per input direction k)
      for (size_t k = 0; k < 3; ++k) {
        // Set up dual numbers once for this k
        std::array<autodiff::dual, 3> autodiff_vars{
            logical_coords[0][i], logical_coords[1][i], logical_coords[2][i]};

        // seed only the k-th variable
        autodiff::seed<1>(autodiff_vars[k], 1.0);

        // single evaluation for this seed
        const auto y = coord_map(autodiff_vars);

        // extract derivative of each output wrt input k and store it
        for (size_t j = 0; j < 3; ++j) {
          const auto deriv_jk = autodiff::derivative<1>(gsl::at(y, j));
          actual_jacobian.get(j, k)[i] = deriv_jk;
        }
      }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
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
  {
    INFO("Single numbers");
    autodiff::dual x = 2.0;
    const auto df_dx = autodiff::derivative(
        &f1<autodiff::dual>, autodiff::detail::wrt(x), autodiff::at(x, 3.0));
    CHECK(df_dx == approx(12.0));
  }
  {
    INFO("Vectorization");
    blaze::DynamicVector<autodiff::real> x{2.0, 3.0, 4.0};
    const auto df_dx =
        autodiff::derivative(&f1<blaze::DynamicVector<autodiff::real>>,
                             autodiff::detail::wrt(x), autodiff::at(x, 3.0));
    const blaze::DynamicVector<double> df_dx_expected{12.0, 18.0, 24.0};
    CHECK_ITERABLE_APPROX(df_dx, df_dx_expected);
  }
  {
    INFO("Vectorization2");
    blaze::DynamicVector<autodiff::real> x{2.0, 3.0};
    std::cout << autodiff::derivative(&f5<blaze::DynamicVector<autodiff::real>>,
                                      autodiff::detail::wrt(x), autodiff::at(x))
              << std::endl;
  }
  {
    INFO("Gradient");
    std::array<autodiff::real, 2> x = {2.0, 3.0};
    const auto df_dx = autodiff::derivative(
        &f2<autodiff::real>, autodiff::detail::wrt(x[0]), autodiff::at(x));
    const auto df_dy = autodiff::derivative(
        &f2<autodiff::real>, autodiff::detail::wrt(x[1]), autodiff::at(x));
    CHECK(df_dx == approx(9.0));
    CHECK(df_dy == approx(12.0));
    // Same as above, but using the gradient convenience function
    autodiff::real F{};
    std::vector<double> grad{};
    autodiff::gradient(&f2<autodiff::real>, autodiff::detail::wrt(x[0], x[1]),
                       autodiff::at(x), F, grad);
    CHECK(grad[0] == approx(9.0));
    CHECK(grad[1] == approx(12.0));
  }
  {
    INFO("Jacobian");
    std::array<autodiff::real, 2> x = {2.0, 3.0};
    std::array<autodiff::real, 2> F{};
    blaze::DynamicMatrix<double> J{};
    autodiff::jacobian(&f3<autodiff::real>, autodiff::detail::wrt(x[0], x[1]),
                       autodiff::at(x), F, J);
  }
  {
    INFO("Tensors");
    tnsr::I<blaze::DynamicVector<autodiff::real>, 2> x{};
    get<0>(x) = blaze::DynamicVector<autodiff::real>{2.0, 3.0};
    get<1>(x) = blaze::DynamicVector<autodiff::real>{4.0, 5.0};
    Scalar<blaze::DynamicVector<autodiff::real>> y{};
    get(y) = blaze::DynamicVector<autodiff::real>{6.0, 7.0};
    const auto df_dx1 = autodiff::derivative(
        &f4<blaze::DynamicVector<autodiff::real>>,
        autodiff::detail::wrt(get<0>(x)), autodiff::at(x, y));
    const blaze::DynamicVector<double> df_dx1_expected{6.0, 7.0};
    CHECK_ITERABLE_APPROX(df_dx1, df_dx1_expected);
  }
}
