// Distributed under the MIT License.
// See LICENSE.txt for details.

// Ensure unconstrained FP here so Enzyme doesn't see constrained intrinsics
#pragma clang fp exceptions(ignore)

#include "Framework/TestingFramework.hpp"

#include <chrono>
#include <iostream>
#include <xsimd/xsimd.hpp>

#include "DataStructures/DynamicVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Wedge.cpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/MakeWithValue.hpp"

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template <typename return_type, typename... T>
return_type __enzyme_fwddiff(void*, T...);

template <typename return_type, typename... T>
return_type __enzyme_autodiff(void*, T...);

namespace {
using Wedge3D = domain::CoordinateMaps::Wedge<3>;

std::array<double, 2> f(const double x) {
  return std::array<double, 2>{x * x, x};
}

inline std::array<double, 3> coord_component(const Wedge3D& map, double ksi,
                                             double eta, double zeta) {
  std::array<double, 3> lc{ksi, eta, zeta};
  const auto result = map.operator()(lc);
  return result;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Autodiff.Enzyme",
                  "[Unit][DataStructures]") {
  {
    INFO("test enzyme");
    double x = 3.0;
    double dx = 1.0;
    auto df_dx =
        __enzyme_fwddiff<std::array<double, 2>>((void*)f, enzyme_dup, x, dx);
    std::cout << "df_dx = " << df_dx << std::endl;
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

    auto total_start = std::chrono::high_resolution_clock::now();
    const auto expected_jacobian = coord_map.jacobian(logical_coords);
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
        total_end - total_start);
    std::cout << "SpECTRE jacobian - Total time: "
              << total_time.count() / 1000.0 << " ms" << std::endl;

    total_start = std::chrono::high_resolution_clock::now();
    auto actual_jacobian =
        make_with_value<tnsr::Ij<DataVector, 3>>(expected_jacobian, 0.0);

    for (size_t i = 0; i < logical_coords[0].size(); i += 1) {
      const auto d_comp_dksi = __enzyme_fwddiff<std::array<double, 3>>(
          (void*)coord_component, enzyme_const, coord_map, enzyme_dup,
          logical_coords[0][i], 1.0, enzyme_dup, logical_coords[1][i], 0.0,
          enzyme_dup, logical_coords[2][i], 0.0);
      const auto d_comp_deta = __enzyme_fwddiff<std::array<double, 3>>(
          (void*)coord_component, enzyme_const, coord_map, enzyme_dup,
          logical_coords[0][i], 0.0, enzyme_dup, logical_coords[1][i], 1.0,
          enzyme_dup, logical_coords[2][i], 0.0);
      const auto d_comp_dzeta = __enzyme_fwddiff<std::array<double, 3>>(
          (void*)coord_component, enzyme_const, coord_map, enzyme_dup,
          logical_coords[0][i], 0.0, enzyme_dup, logical_coords[1][i], 0.0,
          enzyme_dup, logical_coords[2][i], 1.0);

      for (size_t j = 0; j < 3; ++j) {
        actual_jacobian.get(j, 0)[i] = d_comp_dksi[j];
        actual_jacobian.get(j, 1)[i] = d_comp_deta[j];
        actual_jacobian.get(j, 2)[i] = d_comp_dzeta[j];
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

  //   {
  //     INFO("Profiling Forward Mode");
  //     const size_t points_per_dimension = 40;
  //     const Mesh<3> mesh{points_per_dimension,
  //     Spectral::Basis::FiniteDifference,
  //                        Spectral::Quadrature::CellCentered};
  //     std::array<DataVector, 3> logical_coords{};
  //     for (size_t i = 0; i < 3; ++i) {
  //       logical_coords[i] = logical_coordinates(mesh).get(i);
  //     }
  //     // const auto coord_map =
  //     //   Affine3D{Affine{-1.0, 1.0, -1.0, 1.0}, Affine{-1.0, 1.0,
  //     0.0, 2.0},
  //     //              Affine{-1.0, 1.0, -2.0, 2.0}};
  //     // const auto coord_map =
  //     //   Interval3D{Interval{-1.0, 1.0, -1.0, 1.0,
  //     Distribution::Equiangular},
  //     //              Interval{-1.0, 1.0, 0.0, 2.0,
  //     Distribution::Equiangular},
  //     //              Interval{-1.0, 1.0, -2.0, 2.0,
  //     Distribution::Equiangular}}; const auto coord_map =
  //         Wedge3D{1.0, 2.0, 0.0, 1.0, OrientationMap<3>::create_aligned(),
  //         true};

  // auto total_start = std::chrono::high_resolution_clock::now();
  //     const auto expected_jacobian = coord_map.jacobian(logical_coords);
  // auto total_end = std::chrono::high_resolution_clock::now();
  // auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
  //     total_end - total_start);
  // std::cout << "SpECTRE jacobian - Total time: " << total_time.count() /
  // 1000.0
  //           << " ms" << std::endl;

  // total_start = std::chrono::high_resolution_clock::now();
  //     auto actual_jacobian =
  //         make_with_value<tnsr::Ij<DataVector, 3>>(expected_jacobian, 0.0);

  //     using b_type = xsimd::batch<double>;

  //     for (size_t i = 0; i < logical_coords[0].size(); i += b_type::size) {
  //       std::array<b_type, 3> log_coords{
  //           b_type::load_aligned(&logical_coords[0][i]),
  //           b_type::load_aligned(&logical_coords[1][i]),
  //           b_type::load_aligned(&logical_coords[2][i])};

  //       const auto J = coord_map.jacobian(log_coords);

  //       for (size_t j = 0; j < 3; ++j) {
  //         for (size_t k = 0; k < 3; ++k) {
  //           J.get(j, k).store_aligned(&actual_jacobian.get(j, k)[i]);
  //         }
  //       }
  //     }

  // total_end = std::chrono::high_resolution_clock::now();
  // total_time = std::chrono::duration_cast<std::chrono::microseconds>(
  //         total_end - total_start);

  //     std::cout << "Vectorized SpECTRE jacobian " << total_time.count() /
  //     1000.0
  //               << " ms" << std::endl;

  //     for (size_t i = 0; i < 3; ++i) {
  //       for (size_t j = 0; j < 3; ++j) {
  //         CHECK_ITERABLE_APPROX(actual_jacobian.get(i, j),
  //                               expected_jacobian.get(i, j));
  //       }
  //     }
  //   }
}
