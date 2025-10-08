// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <chrono>
// #include <clad/Differentiator/Differentiator.h>
// #include <clad/Differentiator/STLBuiltins.h>

#include <iostream>

#include "DataStructures/DynamicVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.cpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Interval.cpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.cpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {
// using Affine = domain::CoordinateMaps::Affine;
// using Interval = domain::CoordinateMaps::Interval;
// using Wedge3D = domain::CoordinateMaps::Wedge<3>;
// using Distribution = domain::CoordinateMaps::Distribution;
// using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine,
// Affine>; using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine,
// Affine>; using Interval3D =
//     domain::CoordinateMaps::ProductOf3Maps<Interval, Interval, Interval>;

// // Helper free function to avoid Clad issues with overloaded
// Wedge::operator() template < typename MapType > inline std::array<double, 1>
// coord_component_Wedge3D(const MapType& map,
//                                       const double mama) {
//   std::array<double, 1> lc{mama};
//   const auto result = map.operator()(lc);
//   return result;
// }

// std::array<double, 2> fn(double x, double y) {
//   std::array<double, 2> res{x, y};
//   return std::array<double, 2> {res[0]*res[1], res[0]*x};
// }

class Simple {
 public:
  double x, y;
  Simple(double px = 0.0, double py = 0.0) : x(px), y(py) {}

  // f(i, j) = (x + y) * i + i * j * j
  double f(double i, double j) { return (x + y) * i + i * j * j; }

  // Template operator() - like Affine::operator()
  template <typename T>
  T operator()(const T& source_coord) const {
    return (x + y) * source_coord;
  }
};

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Autodiff", "[Unit][DataStructures]") {
  {
    std::cout << "Testing Interval inlined functions with Clad:" << std::endl;

    // Test Linear distribution
    const double A = -1.0, B = 1.0, a = 0.0, b = 2.0;
    auto d_linear = clad::differentiate(
        domain::CoordinateMaps::interval_linear_map, "source_coord");
    std::cout << "Linear map derivative:" << std::endl;
    d_linear.dump();
    double linear_result = d_linear.execute(A, B, a, b, 0.5);
    std::cout << "Linear derivative result: " << linear_result << std::endl;
    CHECK(linear_result == (b - a) / (B - A));  // Should be 1.0

    // Test Equiangular distribution
    auto d_equiangular = clad::differentiate(
        domain::CoordinateMaps::interval_equiangular_map, "source_coord");
    std::cout << "\nEquiangular map derivative:" << std::endl;
    d_equiangular.dump();
    double equiangular_result = d_equiangular.execute(A, B, a, b, 0.0);
    std::cout << "Equiangular derivative result: " << equiangular_result
              << std::endl;
  }
  // {
  //   INFO("test clad");
  //   // differentiate 'fn' w.r.t 'x'.
  //   auto d_fn_1 = clad::differentiate(fn, "x");
  //   d_fn_1.dump();
  //   std::array<double, 2> expected{2.0, 2.0};
  //   CHECK_ITERABLE_APPROX(d_fn_1.execute(1.0, 2.0), expected);
  // }

  {
    INFO("test clad with template operator() like Affine");
    Simple s(1.0, 2.0);

    // Test 1: Direct member function differentiation (this should work)
    auto d_template_direct =
        clad::differentiate(&Simple::operator()<double>, "source_coord");
    std::cout << "Direct template operator() differentiation:" << std::endl;
    d_template_direct.dump();
    CHECK(d_template_direct.execute(s, 5.0) == 3.0);

    // Test 2: Lambda wrapper calling operator() (this should break like Affine)
    const auto simple_wrapper = [](const double mama) {
      Simple obj(1.0, 2.0);
      return obj(mama);  // Call operator() through lambda
    };

    auto d_template_lambda = clad::differentiate(simple_wrapper, "mama");
    std::cout << "Lambda-wrapped template operator() differentiation:"
              << std::endl;
    d_template_lambda.dump();
    CHECK(d_template_lambda.execute(5.0) == 3.0);
  }

  {
    INFO("test clad with Wedge 3D free function");
    std::cout << "Testing Wedge 3D function with Clad:" << std::endl;

    // Test the wedge_3d_map function
    const double xi = 0.5, eta = 0.3, zeta = 0.0;
    const double radius_inner = 1.0, radius_outer = 2.0;
    const double sphericity_inner = 1.0, sphericity_outer = 1.0;

    // Call the function directly
    auto result = domain::CoordinateMaps::wedge_3d_map(
        xi, eta, zeta, radius_inner, radius_outer, sphericity_inner,
        sphericity_outer);
    std::cout << "Direct function result: [" << result[0] << ", " << result[1]
              << ", " << result[2] << "]" << std::endl;

    // Create a Wedge object to get analytical jacobian for comparison
    using Wedge3D = domain::CoordinateMaps::Wedge<3>;
    const auto wedge_map = Wedge3D{radius_inner,
                                   radius_outer,
                                   sphericity_inner,
                                   sphericity_outer,
                                   OrientationMap<3>::create_aligned(),
                                   false};  // no equiangular map for simplicity

    // Get analytical jacobian at the test point
    const std::array<double, 3> source_coords = {xi, eta, zeta};
    const auto analytical_jacobian = wedge_map.jacobian(source_coords);

    // Extract the partial derivatives w.r.t xi (polar coordinate, index 0)
    const auto expected_dx_dxi = analytical_jacobian.get(0, 0);  // ∂x/∂ξ
    const auto expected_dy_dxi = analytical_jacobian.get(1, 0);  // ∂y/∂ξ
    const auto expected_dz_dxi = analytical_jacobian.get(2, 0);  // ∂z/∂ξ

    std::array<double, 3> expected_derivatives = {
        expected_dx_dxi, expected_dy_dxi, expected_dz_dxi};
    std::cout << "Expected derivatives from analytical jacobian: ["
              << expected_dx_dxi << ", " << expected_dy_dxi << ", "
              << expected_dz_dxi << "]" << std::endl;

    // Test Clad differentiation with respect to xi (polar angle)
    auto d_wedge_dxi =
        clad::differentiate(domain::CoordinateMaps::wedge_3d_map, "xi");
    std::cout << "\nClad derivative with respect to xi:" << std::endl;
    d_wedge_dxi.dump();

    auto deriv_result =
        d_wedge_dxi.execute(xi, eta, zeta, radius_inner, radius_outer,
                            sphericity_inner, sphericity_outer);
    std::cout << "Derivative w.r.t xi from Clad: [" << deriv_result[0] << ", "
              << deriv_result[1] << ", " << deriv_result[2] << "]" << std::endl;

    // Basic sanity checks
    CHECK(std::isfinite(result[0]));
    CHECK(std::isfinite(result[1]));
    CHECK(std::isfinite(result[2]));
    CHECK(std::isfinite(deriv_result[0]));
    CHECK(std::isfinite(deriv_result[1]));
    CHECK(std::isfinite(deriv_result[2]));

    // Compare Clad results with analytical jacobian using SpECTRE's testing
    // framework
    CHECK_ITERABLE_APPROX(deriv_result, expected_derivatives);
  }

  //   {
  //     INFO("Profiling Forward Mode");
  //     const size_t points_per_dimension = 3;
  //     const Mesh<1> mesh{points_per_dimension,
  //     Spectral::Basis::FiniteDifference,
  //                        Spectral::Quadrature::CellCentered};
  //     std::array<DataVector, 1> logical_coords{};
  //     for (size_t i = 0; i < 1; ++i) {
  //       logical_coords[i] = logical_coordinates(mesh).get(i);
  //     }
  //     const auto coord_map = Affine{-10.0, 10.0, 0.0, 40.0};
  //     // const auto coord_map =
  //     //   Interval3D{Interval{-1.0, 1.0, -1.0, 1.0,
  //     Distribution::Equiangular},
  //     //              Interval{-1.0, 1.0, 0.0, 2.0,
  //     Distribution::Equiangular},
  //     //              Interval{-1.0, 1.0, -2.0, 2.0,
  //     Distribution::Equiangular}};
  //     // const auto coord_map =
  //     //     Wedge3D{1.0, 2.0, 0.0, 1.0, OrientationMap<3>::create_aligned(),
  //     true};

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

  //     // Build derivative functors once for the non-capturing helper
  //     const auto d_comp_dksi =
  //     clad::differentiate(coord_component_Wedge3D<decltype(coord_map)>,
  //     "mama"); d_comp_dksi.dump();

  //     for (size_t i = 0; i < logical_coords[0].size(); i += 1) {
  //       std::array<double, 1> log_coords{
  //           logical_coords[0][i]};

  //       for (size_t j = 0; j < 1; ++j) {
  //         actual_jacobian.get(j, 0)[i] = d_comp_dksi.execute(
  //             coord_map, log_coords[0])[j];
  //       }
  //     }
  // total_end = std::chrono::high_resolution_clock::now();
  // total_time = std::chrono::duration_cast<std::chrono::microseconds>(
  //         total_end - total_start);
  // std::cout << "Forward Mode - Total time: " << total_time.count() / 1000.0
  //           << " ms" << std::endl;

  //     for (size_t i = 0; i < 1; ++i) {
  //       for (size_t j = 0; j < 1; ++j) {
  //         CAPTURE(i, j);
  //         CHECK_ITERABLE_APPROX(actual_jacobian.get(i, j),
  //                               expected_jacobian.get(i, j));
  //       }
  //     }
  //   }

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
