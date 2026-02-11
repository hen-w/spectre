// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Characteristics.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ConstraintCharacteristics.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/Z4Constraint.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"

namespace {
void test_constraint_characteristics(
    const size_t points_per_dimension,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>&
        normal_one_form) {
  ASSERT(pow<3>(points_per_dimension) == get<0>(normal_one_form).size(),
         "The size of the unit normal one form must match the number of grid "
         "points per dimension.");

  const size_t ghost_zone_size = 3;
  const size_t fd_deriv_order = 4;
  const Mesh<::Ccz4::fd::System::volume_dim> subcell_mesh{
      points_per_dimension, Spectral::Basis::FiniteDifference,
      Spectral::Quadrature::CellCentered};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  const std::array<double, ::Ccz4::fd::System::volume_dim> lower_bound{5.8, 5.0,
                                                                       2.3};
  const std::array<double, ::Ccz4::fd::System::volume_dim> upper_bound{6.2, 5.2,
                                                                       2.4};
  const std::array<double, ::Ccz4::fd::System::volume_dim> coords_range =
      upper_bound - lower_bound;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });
  // set up displaced logical coords
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = coord_map(logical_coords);
  InverseJacobian<DataVector, ::Ccz4::fd::System::volume_dim,
                  Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < ::Ccz4::fd::System::volume_dim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  const Element<::Ccz4::fd::System::volume_dim> element =
      TestHelpers::Ccz4::fd::detail::set_element();

  // Setup solution
  const double mass = 2.0;
  const std::array<double, ::Ccz4::fd::System::volume_dim> spin{
      {0.2, 0.4, 0.8}};
  const std::array<double, ::Ccz4::fd::System::volume_dim> center{
      {0.2, 0.5, 0.1}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  const double f = Ccz4::fd::System::f;
  const bool evolve_shift = true;
  const DirectionalIdMap<::Ccz4::fd::System::volume_dim,
                         evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::KerrSchild::
                  compute_prim_solution_for_KerrSchild,
              coords_range, t, f, evolve_shift, solution);

  auto volume_evolved_vars = TestHelpers::Ccz4::fd::detail::KerrSchild::
      compute_prim_solution_for_KerrSchild(x, t, f, evolve_shift, solution);

  // Change theta and gamma_hat to some dummy values to test constraint
  // characteristics
  get(get<Ccz4::Tags::Theta<DataVector>>(volume_evolved_vars)) =
      DataVector(subcell_mesh.number_of_grid_points(), 42.0);
  for (size_t i = 0; i < ::Ccz4::fd::System::volume_dim; ++i) {
    get<Ccz4::Tags::GammaHat<DataVector, ::Ccz4::fd::System::volume_dim>>(
        volume_evolved_vars)
        .get(i) = DataVector(subcell_mesh.number_of_grid_points(), -42.0 + i);
  }

  Variables<db::wrap_tags_in<
      Tags::deriv, typename Ccz4::fd::System::gradients_tags,
      tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame::Inertial>>
      deriv_of_Ccz4_vars{subcell_mesh.number_of_grid_points()};

  ::Ccz4::fd::spacetime_derivatives(
      make_not_null(&deriv_of_Ccz4_vars), volume_evolved_vars, all_ghost_data,
      fd_deriv_order, subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian);

  // Compute unit_normal_one_form
  const auto& conformal_spatial_metric = get<::Ccz4::Tags::ConformalMetric<
      DataVector, ::Ccz4::fd::System::volume_dim>>(volume_evolved_vars);
  const auto& conformal_factor =
      get<::Ccz4::Tags::ConformalFactor<DataVector>>(volume_evolved_vars);
  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(conformal_spatial_metric).second;
  tnsr::II<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>
      inverse_spatial_metric{};
  ::tenex::evaluate<ti::I, ti::J>(
      make_not_null(&inverse_spatial_metric),
      inverse_conformal_spatial_metric(ti::I, ti::J) * conformal_factor() *
          conformal_factor());
  const DataVector magnitude = sqrt(
      get(::tenex::evaluate(inverse_spatial_metric(ti::I, ti::J) *
                            normal_one_form(ti::i) * normal_one_form(ti::j))));
  tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>
      unit_normal_one_form{};
  for (size_t i = 0; i < ::Ccz4::fd::System::volume_dim; ++i) {
    unit_normal_one_form.get(i) = normal_one_form.get(i) / magnitude;
  }
  const auto unit_normal_vector =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);

  // Test constraint characteristic speeds by numerical diagnoalization of the
  // principal symbol.
  std::array<DataVector, 3> constraint_char_speeds{};
  ::Ccz4::fd::ConstraintCharacteristicSpeedsCompute<Frame::Inertial>::function(
      make_not_null(&constraint_char_speeds),
      get<gr::Tags::Lapse<DataVector>>(volume_evolved_vars),
      get<gr::Tags::Shift<DataVector, ::Ccz4::fd::System::volume_dim,
                          Frame::Inertial>>(volume_evolved_vars),
      unit_normal_one_form);
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(volume_evolved_vars);
  const auto& shift =
      get<gr::Tags::Shift<DataVector, ::Ccz4::fd::System::volume_dim,
                          Frame::Inertial>>(volume_evolved_vars);

  CHECK_ITERABLE_APPROX(
      constraint_char_speeds,
      (pypp::call<std::array<DataVector, 3>>(
          "TestFunctions", "constraint_characteristic_speeds", lapse, shift,
          unit_normal_vector, unit_normal_one_form)));

  // Test constraint characteristic fields by computing from evolved variables.
  const auto& d_conformal_spatial_metric = get<Tags::deriv<
      Ccz4::Tags::ConformalMetric<DataVector, ::Ccz4::fd::System::volume_dim,
                                  Frame::Inertial>,
      tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame::Inertial>>(
      deriv_of_Ccz4_vars);
  const auto field_d = ::tenex::evaluate<ti::k, ti::i, ti::j>(
      0.5 * d_conformal_spatial_metric(ti::k, ti::i, ti::j));
  const auto conformal_christoffel = ::Ccz4::conformal_christoffel_second_kind(
      inverse_conformal_spatial_metric, field_d);
  const auto contracted_conformal_christoffel =
      ::Ccz4::contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, conformal_christoffel);
  const auto& gamma_hat =
      get<::Ccz4::Tags::GammaHat<DataVector, ::Ccz4::fd::System::volume_dim>>(
          volume_evolved_vars);
  const auto gamma_hat_minus_contracted_conformal_christoffel =
      ::tenex::evaluate<ti::I>(gamma_hat(ti::I) -
                               contracted_conformal_christoffel(ti::I));
  const Scalar<DataVector> half_conformal_factor_squared =
      ::tenex::evaluate(0.5 * conformal_factor() * conformal_factor());
  const auto upper_spatial_z4 = ::Ccz4::upper_spatial_z4_constraint(
      half_conformal_factor_squared,
      gamma_hat_minus_contracted_conformal_christoffel);

  tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>
      spatial_metric{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&spatial_metric),
      conformal_spatial_metric(ti::i, ti::j) /
          (conformal_factor() * conformal_factor()));

  typename ::Ccz4::fd::Tags::ConstraintCharacteristicFields<
      DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>::type
      constraint_char_fields{};
  ::Ccz4::fd::ConstraintCharacteristicFieldsCompute<Frame::Inertial>::function(
      make_not_null(&constraint_char_fields),
      get<::Ccz4::Tags::Theta<DataVector>>(volume_evolved_vars),
      upper_spatial_z4, spatial_metric, unit_normal_one_form);

  typename ::Ccz4::fd::Tags::ConstraintCharacteristicFields<
      DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>::type
      expected_constraint_char_fields{get<0>(normal_one_form).size()};

  const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>
      q_dd = gr::transverse_projection_operator(spatial_metric,
                                                unit_normal_one_form);
  auto& expected_c_vector_zero = get<::Ccz4::fd::Tags::CVectorZero<
      DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>>(
      expected_constraint_char_fields);
  ::tenex::evaluate<ti::i>(make_not_null(&expected_c_vector_zero),
                           q_dd(ti::i, ti::j) * upper_spatial_z4(ti::J));

  const auto& theta = get<::Ccz4::Tags::Theta<DataVector>>(volume_evolved_vars);
  auto& expected_c_scalar_plus = get<::Ccz4::fd::Tags::CScalarPlus<DataVector>>(
      expected_constraint_char_fields);
  auto& expected_c_scalar_minus =
      get<::Ccz4::fd::Tags::CScalarMinus<DataVector>>(
          expected_constraint_char_fields);
  ::tenex::evaluate(
      make_not_null(&expected_c_scalar_plus),
      -theta() + unit_normal_one_form(ti::i) * upper_spatial_z4(ti::I));
  ::tenex::evaluate(
      make_not_null(&expected_c_scalar_minus),
      theta() + unit_normal_one_form(ti::i) * upper_spatial_z4(ti::I));

  const Approx custom_approx =
      Approx::custom().epsilon(1.0e-12).scale(*std::max_element(
          volume_evolved_vars.data(),
          volume_evolved_vars.data() + volume_evolved_vars.size()));

  tmpl::for_each<typename ::Ccz4::fd::Tags::ConstraintCharacteristicFields<
      DataVector, ::Ccz4::fd::System::volume_dim,
      Frame::Inertial>::type::tags_list>(
      [&constraint_char_fields, &expected_constraint_char_fields,
       &custom_approx]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const std::string tag_name = db::tag_name<Tag>();
        CAPTURE(tag_name);
        CHECK_ITERABLE_CUSTOM_APPROX(get<Tag>(constraint_char_fields),
                                     get<Tag>(expected_constraint_char_fields),
                                     custom_approx);
      });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.ConstraintCharacteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/Ccz4/FiniteDifference"};

  MAKE_GENERATOR(generator);
  const std::uniform_real_distribution<> distribution(1.0, 2.0);
  const size_t points_per_dimension = 5;
  auto normal_one_form = make_with_random_values<
      tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>>(
      make_not_null(&generator), distribution,
      DataVector(pow<3>(points_per_dimension),
                 std::numeric_limits<double>::signaling_NaN()));

  test_constraint_characteristics(points_per_dimension, normal_one_form);
}
