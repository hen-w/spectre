// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstring>
#include <random>
#include <string>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Compare the stored double representations, including the sign of zero.
template <typename T>
void check_bitwise_equal(const T& actual, const T& expected) {
  REQUIRE(actual.size() == expected.size());
  CHECK(std::memcmp(actual.data(), expected.data(),
                    actual.size() * sizeof(double)) == 0);
}

using System = Ccz4::fd::System;
using AllVariables = Variables<System::variables_tag_list>;
using deriv_tags = db::wrap_tags_in<Tags::deriv, System::gradients_tags,
                                    tmpl::size_t<3>, Frame::Inertial>;
using second_deriv_tags =
    db::wrap_tags_in<Tags::second_deriv, System::gradients_tags,
                     tmpl::size_t<3>, Frame::Inertial>;
using selected_deriv_tags =
    db::wrap_tags_in<Tags::deriv, System::second_derivative_tags,
                     tmpl::size_t<3>, Frame::Inertial>;
using selected_second_deriv_tags =
    db::wrap_tags_in<Tags::second_deriv, System::second_derivative_tags,
                     tmpl::size_t<3>, Frame::Inertial>;
constexpr size_t fd_order = 4;

struct Inputs {
  AllVariables volume_vars;
  DirectionalIdMap<3, evolution::dg::subcell::GhostData> ghost_data;
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian;
  InverseHessian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inverse_hessian;
};

// Random values in all the evolved components (the physical fields and the
// stored gradients that follow them) in the volume and in the ghost data, so
// that wrong component offsets change the derivatives, and random dense
// geometry.
Inputs random_inputs(const gsl::not_null<std::mt19937*> generator,
                     const Mesh<3>& mesh, const size_t ghost_zone_size) {
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const size_t number_of_points = mesh.number_of_grid_points();
  Inputs inputs{
      make_with_random_values<AllVariables>(generator, dist, number_of_points),
      {},
      make_with_random_values<InverseJacobian<
          DataVector, 3, Frame::ElementLogical, Frame::Inertial>>(
          generator, dist, DataVector(number_of_points)),
      make_with_random_values<InverseHessian<
          DataVector, 3, Frame::ElementLogical, Frame::Inertial>>(
          generator, dist, DataVector(number_of_points))};
  for (const auto& direction : Direction<3>::all_directions()) {
    const DirectionalId<3> id{direction, ElementId<3>{1}};
    inputs.ghost_data[id] = evolution::dg::subcell::GhostData{1};
    inputs.ghost_data[id].neighbor_ghost_data_for_reconstruction() =
        make_with_random_values<DataVector>(
            generator, dist,
            DataVector(AllVariables::number_of_independent_components *
                       ghost_zone_size *
                       mesh.slice_away(0).number_of_grid_points()));
  }
  return inputs;
}

// The first derivatives of the fields `System::second_derivative_tags`, and
// the first derivatives of all the physical fields with the second derivatives
// of `System::second_derivative_tags`, are bitwise the corresponding ones of
// `spacetime_derivatives` and `second_spacetime_derivatives` of all the
// physical fields, and leave the inputs unchanged.
void check_against_all_field_derivatives(
    const gsl::not_null<Variables<selected_deriv_tags>*> selected_derivatives,
    const gsl::not_null<Variables<deriv_tags>*> first_derivatives,
    const gsl::not_null<Variables<selected_second_deriv_tags>*>
        second_derivatives,
    const Inputs& inputs, const Mesh<3>& mesh) {
  Variables<deriv_tags> expected_derivatives{};
  Ccz4::fd::spacetime_derivatives(make_not_null(&expected_derivatives),
                                  inputs.volume_vars, inputs.ghost_data,
                                  fd_order, mesh, inputs.inverse_jacobian);
  Variables<second_deriv_tags> expected_second_derivatives{};
  Ccz4::fd::second_spacetime_derivatives(
      make_not_null(&expected_second_derivatives), inputs.volume_vars,
      inputs.ghost_data, fd_order, mesh, inputs.inverse_jacobian,
      inputs.inverse_hessian);

  const Inputs inputs_copy = inputs;
  Ccz4::fd::spacetime_derivatives(selected_derivatives, inputs.volume_vars,
                                  inputs.ghost_data, fd_order, mesh,
                                  inputs.inverse_jacobian);
  Ccz4::fd::first_and_second_spacetime_derivatives(
      first_derivatives, second_derivatives, inputs.volume_vars,
      inputs.ghost_data, fd_order, mesh, inputs.inverse_jacobian,
      inputs.inverse_hessian);
  CHECK(inputs.volume_vars == inputs_copy.volume_vars);
  CHECK(inputs.ghost_data == inputs_copy.ghost_data);
  CHECK(inputs.inverse_jacobian == inputs_copy.inverse_jacobian);
  CHECK(inputs.inverse_hessian == inputs_copy.inverse_hessian);

  check_bitwise_equal(*first_derivatives, expected_derivatives);
  check_bitwise_equal(
      *selected_derivatives,
      expected_derivatives.extract_subset<selected_deriv_tags>());
  check_bitwise_equal(
      *second_derivatives,
      expected_second_derivatives.extract_subset<selected_second_deriv_tags>());
}

void test(const gsl::not_null<std::mt19937*> generator,
          const size_t points_per_dimension, const size_t ghost_zone_size) {
  CAPTURE(points_per_dimension);
  CAPTURE(ghost_zone_size);
  const Mesh<3> mesh{points_per_dimension, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered};
  // The empty outputs are resized by the first call and then reused with new
  // fields and geometry, so that values left from the first call are caught.
  Variables<selected_deriv_tags> selected_derivatives{};
  Variables<deriv_tags> first_derivatives{};
  Variables<selected_second_deriv_tags> second_derivatives{};
  for (size_t call = 0; call < 2; ++call) {
    CAPTURE(call);
    check_against_all_field_derivatives(
        make_not_null(&selected_derivatives), make_not_null(&first_derivatives),
        make_not_null(&second_derivatives),
        random_inputs(generator, mesh, ghost_zone_size), mesh);
  }
}

void test_bad_ghost_data(const gsl::not_null<std::mt19937*> generator) {
#ifdef SPECTRE_DEBUG
  const Mesh<3> mesh{5, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered};
  for (const auto& direction : Direction<3>::all_directions()) {
    auto inputs = random_inputs(generator, mesh, 3);
    const DirectionalId<3> id{direction, ElementId<3>{1}};
    inputs.ghost_data.at(id).neighbor_ghost_data_for_reconstruction() =
        DataVector(2, 0.0);
    const std::string match_string{
        MakeString{}
        << "Amount of reconstruction data sent (2) from " << id
        << " is not a multiple of the number of reconstruction variables "
        << AllVariables::number_of_independent_components};
    Variables<selected_deriv_tags> selected_derivatives{};
    Variables<deriv_tags> first_derivatives{};
    Variables<selected_second_deriv_tags> second_derivatives{};
    CHECK_THROWS_WITH(
        Ccz4::fd::spacetime_derivatives(
            make_not_null(&selected_derivatives), inputs.volume_vars,
            inputs.ghost_data, fd_order, mesh, inputs.inverse_jacobian),
        Catch::Matchers::ContainsSubstring(match_string));
    CHECK_THROWS_WITH(Ccz4::fd::first_and_second_spacetime_derivatives(
                          make_not_null(&first_derivatives),
                          make_not_null(&second_derivatives),
                          inputs.volume_vars, inputs.ghost_data, fd_order, mesh,
                          inputs.inverse_jacobian, inputs.inverse_hessian),
                      Catch::Matchers::ContainsSubstring(match_string));
  }
#else
  (void)generator;
#endif  // SPECTRE_DEBUG
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.FiniteDifference.FirstAndSecondDerivatives",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  // Odd extents give point counts that are not multiples of the SIMD width.
  for (const size_t points_per_dimension : {5_st, 7_st}) {
    for (const size_t ghost_zone_size : {2_st, 3_st, 4_st}) {
      test(make_not_null(&generator), points_per_dimension, ghost_zone_size);
    }
  }
  test_bad_ghost_data(make_not_null(&generator));
}
