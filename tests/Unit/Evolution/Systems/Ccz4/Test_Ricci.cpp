// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <climits>
#include <cstddef>
#include <limits>
#include <random>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/Ricci.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <typename Solution>
void test_compute_spatial_ricci_tensor(
    const Solution& solution, size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) {
  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });
  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();
  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim>>(vars);
  const auto det_spatial_metric = determinant_and_inverse(spatial_metric).first;
  const auto& d_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<DataVector, SpatialDim>,
                      tmpl::size_t<SpatialDim>, Frame::Inertial>>(vars);
  const auto d_det_spatial_metric =
      get<gr::Tags::DerivDetSpatialMetric<DataVector, SpatialDim>>(
          solution.variables(
              x, t,
              tmpl::list<
                  gr::Tags::DerivDetSpatialMetric<DataVector, SpatialDim>>{}));

  // Compute arguments for `spatial_ricci_tensor` function to test
  const auto conformal_factor = pow(get(det_spatial_metric), -1. / 6.);

  tnsr::ii<DataVector, SpatialDim, Frame::Inertial> conformal_spatial_metric{};
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = i; j < SpatialDim; j++) {
      conformal_spatial_metric.get(i, j) =
          square(conformal_factor) * spatial_metric.get(i, j);
    }
  }

  tnsr::ijj<DataVector, SpatialDim, Frame::Inertial>
      d_conformal_spatial_metric{};
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        d_conformal_spatial_metric.get(k, i, j) =
            pow<2>(conformal_factor) * d_spatial_metric.get(k, i, j) -
            pow<8>(conformal_factor) * d_det_spatial_metric.get(k) *
                spatial_metric.get(i, j) / 3.;
      }
    }
  }

  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(conformal_spatial_metric).second;

  tnsr::ijj<DataVector, SpatialDim, Frame::Inertial> field_d{};
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        field_d.get(k, i, j) = 0.5 * d_conformal_spatial_metric.get(k, i, j);
      }
    }
  }

  auto field_d_up = gr::deriv_inverse_spatial_metric(
      inverse_conformal_spatial_metric, field_d);
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        field_d_up.get(k, i, j) *= -1.0;
      }
    }
  }

  const auto d_field_d =
      partial_derivative(field_d, mesh, coord_map.inv_jacobian(x_logical));
  const auto d_conformal_christoffel_second_kind =
      Ccz4::deriv_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d, d_field_d, field_d_up);

  tnsr::i<DataVector, SpatialDim, Frame::Inertial> field_p{};
  for (size_t i = 0; i < SpatialDim; i++) {
    field_p.get(i) =
        -d_det_spatial_metric.get(i) / (6. * get(det_spatial_metric));
  }

  const auto d_field_p =
      partial_derivative(field_p, mesh, coord_map.inv_jacobian(x_logical));
  // testing spatial_ricci_tensor() with a symmetric d_field_p
  tnsr::ii<DataVector, SpatialDim, Frame::Inertial> d_field_p_symm{};
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = i; j < SpatialDim; j++) {
        d_field_p_symm.get(i, j) = d_field_p.get(i, j);
    }
  }

  const auto conformal_christoffel_second_kind =
      Ccz4::conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                              field_d);

  const auto christoffel_second_kind = Ccz4::christoffel_second_kind(
      conformal_spatial_metric, inverse_conformal_spatial_metric, field_p,
      conformal_christoffel_second_kind);

  const auto d_christoffel_second_kind = partial_derivative(
      christoffel_second_kind, mesh, coord_map.inv_jacobian(x_logical));
  const auto contracted_d_conformal_christoffel_difference =
      ::tenex::evaluate<ti::i, ti::j>(
          d_conformal_christoffel_second_kind(ti::m, ti::M, ti::i, ti::j) -
          d_conformal_christoffel_second_kind(ti::j, ti::M, ti::i, ti::m));

  // Compute expected and actual ricci tensors using above computed arguments
  const auto expected_py_ricci_tensor{
      pypp::call<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          "Ricci", "spatial_ricci_tensor", christoffel_second_kind,
          d_conformal_christoffel_second_kind, conformal_spatial_metric,
          inverse_conformal_spatial_metric, field_d, field_d_up, field_p,
          d_field_p)};
  const auto expected_py_ricci_tensor_with_symmetric_d_field_p{
      pypp::call<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          "Ricci", "spatial_ricci_tensor", christoffel_second_kind,
          d_conformal_christoffel_second_kind, conformal_spatial_metric,
          inverse_conformal_spatial_metric, field_d, field_d_up, field_p,
          d_field_p_symm)};

  const auto expected_cpp_ricci_tensor =
      gr::ricci_tensor(christoffel_second_kind, d_christoffel_second_kind);

  const auto actual_cpp_ricci_tensor = Ccz4::spatial_ricci_tensor(
      conformal_christoffel_second_kind,
      contracted_d_conformal_christoffel_difference, conformal_spatial_metric,
      inverse_conformal_spatial_metric, field_p, d_field_p);
  const auto actual_cpp_ricci_tensor_with_symmetric_d_field_p =
      Ccz4::spatial_ricci_tensor(conformal_christoffel_second_kind,
                                 contracted_d_conformal_christoffel_difference,
                                 conformal_spatial_metric,
                                 inverse_conformal_spatial_metric, field_p,
                                 d_field_p_symm);

  CHECK_ITERABLE_APPROX(expected_py_ricci_tensor, actual_cpp_ricci_tensor);
  CHECK_ITERABLE_APPROX(expected_py_ricci_tensor_with_symmetric_d_field_p,
    actual_cpp_ricci_tensor_with_symmetric_d_field_p);
  // A custom epsilon is used here because the Legendre polynomials don't fit
  // the derivative of 1 / r well. This was looked at for various box sizes and
  // number of 1D grid points.
  Approx approx = Approx::custom().epsilon(1e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_cpp_ricci_tensor,
                               actual_cpp_ricci_tensor, approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_cpp_ricci_tensor,
                               actual_cpp_ricci_tensor_with_symmetric_d_field_p,
                               approx);
}

// Random data against the unchanged Python expansion in the spatial
// christoffel symbols: dense non-flat conformal metrics
// \tilde{\gamma}_{ij} = \delta_{ij} + a_{ik} a_{jk}, whose determinant is at
// least 1 + a_{ij} a_{ij}, and D_{kij}, P_i, \partial_j P_i and
// \partial_l \tilde{\Gamma}^k_{ij}, independent of the metric and of each
// other, with only the symmetries of their types. The inverse metric,
// D_k^{ij}, and the conformal and physical christoffel symbols are computed
// from them as in the evolution systems. P_i is zero or random, and
// \partial_j P_i symmetric or not.
template <size_t Dim, typename DataType>
void test_random_data(const gsl::not_null<std::mt19937*> generator,
                      const DataType& used_for_size) {
  CAPTURE(Dim);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const auto a = make_with_random_values<tnsr::ij<DataType, Dim>>(
      generator, make_not_null(&distribution), used_for_size);
  auto conformal_spatial_metric =
      make_with_value<tnsr::ii<DataType, Dim>>(used_for_size, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    conformal_spatial_metric.get(i, i) = 1.0;
    for (size_t j = i; j < Dim; ++j) {
      for (size_t k = 0; k < Dim; ++k) {
        conformal_spatial_metric.get(i, j) += a.get(i, k) * a.get(j, k);
      }
    }
  }
  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(conformal_spatial_metric).second;
  const auto field_d = make_with_random_values<tnsr::ijj<DataType, Dim>>(
      generator, make_not_null(&distribution), used_for_size);
  tnsr::iJJ<DataType, Dim> field_d_up{};
  tenex::evaluate<ti::k, ti::I, ti::J>(
      make_not_null(&field_d_up),
      inverse_conformal_spatial_metric(ti::I, ti::N) *
          inverse_conformal_spatial_metric(ti::M, ti::J) *
          field_d(ti::k, ti::n, ti::m));
  const auto conformal_christoffel_second_kind =
      Ccz4::conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                              field_d);
  const auto d_conformal_christoffel_second_kind =
      make_with_random_values<tnsr::iJkk<DataType, Dim>>(
          generator, make_not_null(&distribution), used_for_size);
  tnsr::ij<DataType, Dim> contracted_d_conformal_christoffel_difference{};
  tenex::evaluate<ti::i, ti::j>(
      make_not_null(&contracted_d_conformal_christoffel_difference),
      d_conformal_christoffel_second_kind(ti::m, ti::M, ti::i, ti::j) -
          d_conformal_christoffel_second_kind(ti::j, ti::M, ti::i, ti::m));

  for (const bool zero_field_p : {true, false}) {
    CAPTURE(zero_field_p);
    const auto field_p =
        zero_field_p
            ? make_with_value<tnsr::i<DataType, Dim>>(used_for_size, 0.0)
            : make_with_random_values<tnsr::i<DataType, Dim>>(
                  generator, make_not_null(&distribution), used_for_size);
    const auto christoffel_second_kind = Ccz4::christoffel_second_kind(
        conformal_spatial_metric, inverse_conformal_spatial_metric, field_p,
        conformal_christoffel_second_kind);
    const auto d_field_p = make_with_random_values<tnsr::ij<DataType, Dim>>(
        generator, make_not_null(&distribution), used_for_size);
    tnsr::ii<DataType, Dim> symmetric_d_field_p{};
    tenex::evaluate<ti::i, ti::j>(
        make_not_null(&symmetric_d_field_p),
        0.5 * (d_field_p(ti::i, ti::j) + d_field_p(ti::j, ti::i)));

    const auto check = [&](const auto& d_field_p_to_use) {
      const auto expected = pypp::call<tnsr::ij<DataType, Dim>>(
          "Ricci", "spatial_ricci_tensor", christoffel_second_kind,
          d_conformal_christoffel_second_kind, conformal_spatial_metric,
          inverse_conformal_spatial_metric, field_d, field_d_up, field_p,
          d_field_p_to_use);
      const auto ricci = Ccz4::spatial_ricci_tensor(
          conformal_christoffel_second_kind,
          contracted_d_conformal_christoffel_difference,
          conformal_spatial_metric, inverse_conformal_spatial_metric, field_p,
          d_field_p_to_use);
      // The difference of the derivatives of the conformal christoffel
      // symbols is not symmetric, so neither is the expected tensor: tenex
      // evaluates the symmetric result at the canonical index of each
      // stored component.
      for (size_t storage_index = 0; storage_index < ricci.size();
           ++storage_index) {
        const auto index = ricci.get_tensor_index(storage_index);
        CAPTURE(index);
        CHECK_ITERABLE_APPROX(ricci[storage_index],
                              expected.get(index[0], index[1]));
      }
    };
    check(d_field_p);
    check(symmetric_d_field_p);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Ricci", "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env("Evolution/Systems/Ccz4/");

  const double mass = 2.;
  const std::array<double, 3> spin{{0.3, 0.5, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};

  test_compute_spatial_ricci_tensor(solution, grid_size, lower_bound,
                                    upper_bound);

  MAKE_GENERATOR(generator);
  const double used_for_size_double =
      std::numeric_limits<double>::signaling_NaN();
  const DataVector used_for_size_data_vector(
      5, std::numeric_limits<double>::signaling_NaN());
  test_random_data<1>(make_not_null(&generator), used_for_size_double);
  test_random_data<2>(make_not_null(&generator), used_for_size_double);
  test_random_data<3>(make_not_null(&generator), used_for_size_double);
  test_random_data<1>(make_not_null(&generator), used_for_size_data_vector);
  test_random_data<2>(make_not_null(&generator), used_for_size_data_vector);
  test_random_data<3>(make_not_null(&generator), used_for_size_data_vector);
}
