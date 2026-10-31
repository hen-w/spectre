// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstring>
#include <random>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.tpp"
#include "NumericalAlgorithms/FiniteDifference/SecondPartialDerivatives.hpp"
#include "NumericalAlgorithms/FiniteDifference/SecondPartialDerivatives.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Compare the stored double representations, including the sign of zero.
template <typename T>
void check_bitwise_equal(const T& actual, const T& expected) {
  REQUIRE(actual.size() == expected.size());
  CHECK(std::memcmp(actual.data(), expected.data(),
                    actual.size() * sizeof(double)) == 0);
}

// Random volume data of `number_of_vars` variables and random ghost data
// `ghost_zone_size` points deep in all directions.
template <size_t Dim>
std::pair<DataVector, DirectionMap<Dim, DataVector>> random_volume_and_ghosts(
    const gsl::not_null<std::mt19937*> generator, const Mesh<Dim>& mesh,
    const size_t number_of_vars, const size_t ghost_zone_size) {
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  auto volume_vars = make_with_random_values<DataVector>(
      generator, dist,
      DataVector(number_of_vars * mesh.number_of_grid_points()));
  DirectionMap<Dim, DataVector> ghost_data{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    ghost_data[direction] = make_with_random_values<DataVector>(
        generator, dist,
        DataVector(number_of_vars * ghost_zone_size *
                   mesh.slice_away(0).number_of_grid_points()));
  }
  return {std::move(volume_vars), std::move(ghost_data)};
}

template <size_t Dim>
DirectionMap<Dim, gsl::span<const double>> spans(
    const DirectionMap<Dim, DataVector>& ghost_data) {
  DirectionMap<Dim, gsl::span<const double>> result{};
  for (const auto& [direction, data] : ghost_data) {
    result[direction] = gsl::make_span(data.data(), data.size());
  }
  return result;
}

// The first partial derivatives are bitwise those of the overload that does
// not keep the logical derivatives, and the kept logical derivatives are
// bitwise those of `fd::logical_partial_derivatives`.
template <size_t Dim>
void test_partial_derivatives_keeping_logical(
    const gsl::not_null<std::mt19937*> generator, const size_t fd_order) {
  CAPTURE(Dim);
  CAPTURE(fd_order);
  using derivative_tags =
      tmpl::list<Tags::TempScalar<0>, Tags::TempI<1, Dim, Frame::Inertial>>;
  constexpr size_t number_of_vars =
      Variables<derivative_tags>::number_of_independent_components;
  using DerivVars =
      Variables<db::wrap_tags_in<Tags::deriv, derivative_tags,
                                 tmpl::size_t<Dim>, Frame::Inertial>>;
  const Mesh<Dim> mesh{fd_order + 3, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
  const size_t number_of_points = mesh.number_of_grid_points();
  const auto [volume_vars, ghost_data] = random_volume_and_ghosts(
      generator, mesh, number_of_vars, fd_order / 2 + 1);
  const auto ghost_cell_vars = spans(ghost_data);
  const auto volume_span =
      gsl::make_span(volume_vars.data(), volume_vars.size());
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto inverse_jacobian = make_with_random_values<
      InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>>(
      generator, dist, DataVector(number_of_points));

  DerivVars expected_derivatives{number_of_points};
  ::fd::partial_derivatives<derivative_tags>(
      make_not_null(&expected_derivatives), volume_span, ghost_cell_vars, mesh,
      number_of_vars, fd_order, inverse_jacobian);
  DataVector expected_logical_buffer(Dim * volume_vars.size());
  std::array<gsl::span<double>, Dim> expected_logical{};
  DataVector logical_buffer(Dim * volume_vars.size());
  std::array<gsl::span<double>, Dim> logical{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(expected_logical, d) = gsl::make_span(
        &expected_logical_buffer[d * volume_vars.size()], volume_vars.size());
    gsl::at(logical, d) = gsl::make_span(
        &logical_buffer[d * volume_vars.size()], volume_vars.size());
  }
  ::fd::logical_partial_derivatives(make_not_null(&expected_logical),
                                    volume_span, ghost_cell_vars, mesh,
                                    number_of_vars, fd_order);

  DerivVars derivatives{number_of_points};
  ::fd::partial_derivatives<derivative_tags>(
      make_not_null(&derivatives), make_not_null(&logical), volume_span,
      ghost_cell_vars, mesh, number_of_vars, fd_order, inverse_jacobian);
  check_bitwise_equal(derivatives, expected_derivatives);
  check_bitwise_equal(logical_buffer, expected_logical_buffer);
}

// Tensors of different ranks, one of them symmetric, so that the
// transformation loops over several tensor components.
using tensor_tags =
    tmpl::list<Tags::TempScalar<0>, Tags::Tempii<1, 3, Frame::Inertial>,
               Tags::TempI<2, 3, Frame::Inertial>>;
using Vars = Variables<tensor_tags>;
using SecondDerivVars =
    Variables<db::wrap_tags_in<Tags::second_deriv, tensor_tags, tmpl::size_t<3>,
                               Frame::Inertial>>;
using InverseJacobianType =
    InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>;
using InverseHessianType =
    InverseHessian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>;

std::array<gsl::span<const double>, 3> const_spans(
    const std::array<DataVector, 3>& logical_derivatives) {
  std::array<gsl::span<const double>, 3> result{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(result, d) = gsl::make_span(gsl::at(logical_derivatives, d).data(),
                                        gsl::at(logical_derivatives, d).size());
  }
  return result;
}

// The chain rule
//   d_i d_j u = sum_{a,b} J^a_i J^b_j d_a d_b u + sum_d H^d_{ij} d_d u
// written with the tensor index interface, where d_a d_b u is the second
// logical derivative computed from the volume and ghost data, and d_d u is the
// given first logical derivative.
SecondDerivVars chain_rule(
    const std::array<DataVector, 3>& first_logical_derivatives,
    const DataVector& volume_vars,
    const DirectionMap<3, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<3>& mesh, const InverseJacobianType& inverse_jacobian,
    const InverseHessianType& inverse_hessian) {
  constexpr size_t number_of_vars = Vars::number_of_independent_components;
  const size_t number_of_points = mesh.number_of_grid_points();
  DataVector pure_buffer(3 * volume_vars.size());
  DataVector mixed_buffer(3 * volume_vars.size());
  std::array<gsl::span<double>, 3> pure{};
  std::array<gsl::span<double>, 3> mixed{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(pure, d) = gsl::make_span(&pure_buffer[d * volume_vars.size()],
                                      volume_vars.size());
    gsl::at(mixed, d) = gsl::make_span(&mixed_buffer[d * volume_vars.size()],
                                       volume_vars.size());
  }
  ::fd::second_logical_partial_derivatives(
      make_not_null(&pure), make_not_null(&mixed),
      gsl::make_span(volume_vars.data(), volume_vars.size()), ghost_cell_vars,
      mesh, number_of_vars, 4);

  // Views of the logical derivatives with the layout of `Vars`.
  const auto views = [](const auto& logical_derivatives) {
    const auto view = [&logical_derivatives](const size_t d) {
      const auto& derivative = gsl::at(logical_derivatives, d);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      return Vars{const_cast<double*>(derivative.data()), derivative.size()};
    };
    return std::array<Vars, 3>{{view(0), view(1), view(2)}};
  };
  const auto first_vars = views(first_logical_derivatives);
  const auto pure_vars = views(pure);
  const auto mixed_vars = views(mixed);

  SecondDerivVars result{number_of_points};
  tmpl::for_each<tensor_tags>([&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
    auto& second_derivative =
        get<Tags::second_deriv<Tag, tmpl::size_t<3>, Frame::Inertial>>(result);
    for (size_t storage_index = 0; storage_index < second_derivative.size();
         ++storage_index) {
      const auto index = second_derivative.get_tensor_index(storage_index);
      const size_t i = gsl::at(index, 0);
      const size_t j = gsl::at(index, 1);
      std::array<size_t, Tag::type::rank()> tensor_index{};
      for (size_t r = 0; r < tensor_index.size(); ++r) {
        gsl::at(tensor_index, r) = gsl::at(index, r + 2);
      }
      const size_t component = Tag::type::get_storage_index(tensor_index);
      // The mixed logical derivatives are stored in the order xy, yz, xz.
      const auto second_logical = [&](const size_t a,
                                      const size_t b) -> const DataVector& {
        if (a == b) {
          return get<Tag>(gsl::at(pure_vars, a))[component];
        }
        const size_t mixed_index = a + b == 1 ? 0 : (a + b == 3 ? 1 : 2);
        return get<Tag>(gsl::at(mixed_vars, mixed_index))[component];
      };
      DataVector& value = second_derivative[storage_index];
      value = 0.0;
      for (size_t a = 0; a < 3; ++a) {
        for (size_t b = 0; b < 3; ++b) {
          value += inverse_jacobian.get(a, i) * inverse_jacobian.get(b, j) *
                   second_logical(a, b);
        }
      }
      for (size_t d = 0; d < 3; ++d) {
        value += inverse_hessian.get(d, i, j) *
                 get<Tag>(gsl::at(first_vars, d))[component];
      }
    }
  });
  return result;
}

// Compare `fd::second_partial_derivatives` given the first logical
// derivatives with the overload that computes them, and with the chain rule
// for dense, spatially varying geometry and first logical derivatives
// unrelated to the volume data (so that they must be the given ones).
void test_second_partial_derivatives_from_logical(
    const gsl::not_null<std::mt19937*> generator,
    const size_t points_per_dimension, const size_t ghost_zone_size) {
  CAPTURE(points_per_dimension);
  CAPTURE(ghost_zone_size);
  constexpr size_t fd_order = 4;
  constexpr size_t number_of_vars = Vars::number_of_independent_components;
  const Mesh<3> mesh{points_per_dimension, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered};
  const size_t number_of_points = mesh.number_of_grid_points();
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto [volume_vars, ghost_data] = random_volume_and_ghosts(
      generator, mesh, number_of_vars, ghost_zone_size);
  const auto ghost_cell_vars = spans(ghost_data);
  const auto volume_span =
      gsl::make_span(volume_vars.data(), volume_vars.size());
  const DataVector volume_vars_copy = volume_vars;
  const DirectionMap<3, DataVector> ghost_data_copy = ghost_data;

  Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0e3);
  // The same output is reused with new geometry and first logical
  // derivatives, so that values left from the previous call are caught.
  SecondDerivVars second_derivatives{number_of_points};
  SecondDerivVars expected_second_derivatives{number_of_points};
  for (size_t call = 0; call < 2; ++call) {
    CAPTURE(call);
    const auto inverse_jacobian = make_with_random_values<InverseJacobianType>(
        generator, dist, DataVector(number_of_points));
    const auto inverse_hessian = make_with_random_values<InverseHessianType>(
        generator, dist, DataVector(number_of_points));

    // The logical derivatives of the volume data reproduce the overload that
    // computes them, bitwise.
    std::array<DataVector, 3> logical_derivatives{};
    std::array<gsl::span<double>, 3> logical_derivative_spans{};
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(logical_derivatives, d) = DataVector(volume_vars.size());
      gsl::at(logical_derivative_spans, d) =
          gsl::make_span(gsl::at(logical_derivatives, d).data(),
                         gsl::at(logical_derivatives, d).size());
    }
    ::fd::logical_partial_derivatives(make_not_null(&logical_derivative_spans),
                                      volume_span, ghost_cell_vars, mesh,
                                      number_of_vars, fd_order);
    ::fd::second_partial_derivatives<tensor_tags>(
        make_not_null(&expected_second_derivatives), volume_span,
        ghost_cell_vars, mesh, number_of_vars, fd_order, inverse_jacobian,
        inverse_hessian);
    ::fd::second_partial_derivatives<tensor_tags>(
        make_not_null(&second_derivatives), const_spans(logical_derivatives),
        volume_span, ghost_cell_vars, mesh, number_of_vars, fd_order,
        inverse_jacobian, inverse_hessian);
    check_bitwise_equal(second_derivatives, expected_second_derivatives);
    CHECK_VARIABLES_CUSTOM_APPROX(
        second_derivatives,
        chain_rule(logical_derivatives, volume_vars, ghost_cell_vars, mesh,
                   inverse_jacobian, inverse_hessian),
        custom_approx);

    // Unrelated first logical derivatives enter the inverse-Hessian terms.
    std::array<DataVector, 3> random_logical_derivatives{};
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(random_logical_derivatives, d) =
          make_with_random_values<DataVector>(generator, dist,
                                              DataVector(volume_vars.size()));
    }
    const auto random_logical_derivatives_copy = random_logical_derivatives;
    ::fd::second_partial_derivatives<tensor_tags>(
        make_not_null(&second_derivatives),
        const_spans(random_logical_derivatives), volume_span, ghost_cell_vars,
        mesh, number_of_vars, fd_order, inverse_jacobian, inverse_hessian);
    CHECK_VARIABLES_CUSTOM_APPROX(
        second_derivatives,
        chain_rule(random_logical_derivatives, volume_vars, ghost_cell_vars,
                   mesh, inverse_jacobian, inverse_hessian),
        custom_approx);
    CHECK(random_logical_derivatives == random_logical_derivatives_copy);
  }
  CHECK(volume_vars == volume_vars_copy);
  CHECK(ghost_data == ghost_data_copy);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.SecondPartialDerivativesFromLogical",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  for (const size_t fd_order : {2_st, 4_st, 6_st, 8_st}) {
    test_partial_derivatives_keeping_logical<1>(make_not_null(&generator),
                                                fd_order);
    test_partial_derivatives_keeping_logical<2>(make_not_null(&generator),
                                                fd_order);
    test_partial_derivatives_keeping_logical<3>(make_not_null(&generator),
                                                fd_order);
  }
  // Odd extents give point counts that are not multiples of the SIMD width.
  for (const size_t points_per_dimension : {5_st, 6_st, 7_st, 9_st}) {
    for (const size_t ghost_zone_size : {2_st, 3_st, 4_st}) {
      test_second_partial_derivatives_from_logical(
          make_not_null(&generator), points_per_dimension, ghost_zone_size);
    }
  }
}
