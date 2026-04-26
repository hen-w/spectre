// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Evolution/Systems/SoScalarWave/Characteristics.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SoPlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "DataStructures/TaggedTuple.hpp"

namespace {
template <size_t Index, size_t Dim>
Scalar<DataVector> speed_with_index(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal) {
  return Scalar<DataVector>{
      SoScalarWave::characteristic_speeds<Dim>(normal)[Index]};
}

template <size_t Dim>
void test_characteristic_speeds() {
  TestHelpers::db::test_compute_tag<
      SoScalarWave::Tags::CharacteristicSpeedsCompute<Dim>>(
      "CharacteristicSpeeds");
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(speed_with_index<0, Dim>, "Characteristics",
                                    "char_speed_vpsi", {{{-10.0, 10.0}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<1, Dim>, "Characteristics",
                                    "char_speed_vzero", {{{-10.0, 10.0}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<3, Dim>, "Characteristics",
                                    "char_speed_vminus", {{{-10.0, 10.0}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<2, Dim>, "Characteristics",
                                    "char_speed_vplus", {{{-10.0, 10.0}}},
                                    used_for_size);
}

template <size_t Index, size_t Dim>
Scalar<DataVector> speed_with_index_mesh_velocity(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& mesh_velocity) {
  return Scalar<DataVector>{
      SoScalarWave::characteristic_speeds<Dim>(
          normal, std::optional{mesh_velocity})[Index]};
}

template <size_t Dim>
void test_characteristic_speeds_with_mesh_velocity() {
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(
      speed_with_index_mesh_velocity<0, Dim>, "Characteristics",
      "char_speed_vpsi_mesh_velocity", {{{-10.0, 10.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      speed_with_index_mesh_velocity<1, Dim>, "Characteristics",
      "char_speed_vzero_mesh_velocity", {{{-10.0, 10.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      speed_with_index_mesh_velocity<2, Dim>, "Characteristics",
      "char_speed_vplus_mesh_velocity", {{{-10.0, 10.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      speed_with_index_mesh_velocity<3, Dim>, "Characteristics",
      "char_speed_vminus_mesh_velocity", {{{-10.0, 10.0}}}, used_for_size);
}

// Test that nullopt mesh velocity gives the same result as the no-mesh-velocity
// overload
template <size_t Dim>
void test_characteristic_speeds_nullopt_matches() {
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-10., 10.);
  const size_t n_pts = 5;

  auto normal = make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
      DataVector(n_pts), 0.);
  fill_with_random_values(make_not_null(&normal), make_not_null(&gen),
                          make_not_null(&dist));
  const auto mag = magnitude(normal);
  for (size_t i = 0; i < Dim; ++i) {
    normal.get(i) /= get(mag);
  }

  const auto speeds_no_arg =
      SoScalarWave::characteristic_speeds(normal);
  const auto speeds_nullopt =
      SoScalarWave::characteristic_speeds(
          normal,
          std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>{
              std::nullopt});

  for (size_t i = 0; i < 4; ++i) {
    CHECK_ITERABLE_APPROX(gsl::at(speeds_no_arg, i),
                          gsl::at(speeds_nullopt, i));
  }
}

// Test return-by-reference char speeds by comparing to analytic solution
template <size_t Dim>
void test_characteristic_speeds_analytic(
    const size_t grid_size_each_dimension) {
  // Setup mesh
  Mesh<Dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
  // Get ingredients
  const size_t n_pts = mesh.number_of_grid_points();
  const tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal_one_form{
      DataVector(n_pts, 1. / sqrt(Dim))};

  const auto vpsi_speed_expected =
      make_with_value<Scalar<DataVector>>(unit_normal_one_form, 0.);
  const auto vzero_speed_expected =
      make_with_value<Scalar<DataVector>>(unit_normal_one_form, 0.);
  const auto vplus_speed_expected =
      make_with_value<Scalar<DataVector>>(unit_normal_one_form, 1.);
  const auto vminus_speed_expected =
      make_with_value<Scalar<DataVector>>(unit_normal_one_form, -1.);

  std::array<DataVector, 4> char_speeds{};
  SoScalarWave::Tags::CharacteristicSpeedsCompute<Dim>::function(
      &char_speeds, unit_normal_one_form);
  const auto& vpsi_speed = char_speeds[0];
  const auto& vzero_speed = char_speeds[1];
  const auto& vplus_speed = char_speeds[2];
  const auto& vminus_speed = char_speeds[3];

  CHECK_ITERABLE_APPROX(vpsi_speed_expected.get(), vpsi_speed);
  CHECK_ITERABLE_APPROX(vzero_speed_expected.get(), vzero_speed);
  CHECK_ITERABLE_APPROX(vplus_speed_expected.get(), vplus_speed);
  CHECK_ITERABLE_APPROX(vminus_speed_expected.get(), vminus_speed);
}
}  // namespace

namespace {
template <typename Tag, size_t Dim>
typename Tag::type field_with_tag(
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_one_form) {
  Variables<
      tmpl::list<SoScalarWave::Tags::VPsi, SoScalarWave::Tags::VZero<Dim>,
                 SoScalarWave::Tags::VPlus, SoScalarWave::Tags::VMinus>>
      char_fields{};
  SoScalarWave::Tags::CharacteristicFieldsCompute<Dim>::function(
      make_not_null(&char_fields), psi, pi, phi, normal_one_form);
  return get<Tag>(char_fields);
}

template <size_t Dim>
void test_characteristic_fields() {
  TestHelpers::db::test_compute_tag<
      SoScalarWave::Tags::CharacteristicFieldsCompute<Dim>>(
      "CharacteristicFields");
  const DataVector used_for_size(5);
  // VPsi
  pypp::check_with_random_values<1>(
      field_with_tag<SoScalarWave::Tags::VPsi, Dim>, "Characteristics",
      "char_field_vpsi", {{{-10., 10.}}}, used_for_size);
  // VZero
  pypp::check_with_random_values<1>(
      field_with_tag<SoScalarWave::Tags::VZero<Dim>, Dim>, "Characteristics",
      "char_field_vzero", {{{-10., 10.}}}, used_for_size, 1.e-11);
  // VPlus
  pypp::check_with_random_values<1>(
      field_with_tag<SoScalarWave::Tags::VPlus, Dim>, "Characteristics",
      "char_field_vplus", {{{-10., 10.}}}, used_for_size);
  // VMinus
  pypp::check_with_random_values<1>(
      field_with_tag<SoScalarWave::Tags::VMinus, Dim>, "Characteristics",
      "char_field_vminus", {{{-10., 10.}}}, used_for_size);
}

// Test return-by-reference char fields by comparing to analytic solution
template <size_t Dim, typename Solution>
void test_characteristic_fields_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, Dim>& lower_bound,
    const std::array<double, Dim>& upper_bound) {
  // Set up grid
  Mesh<Dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  const double t = 0.;

  // Evaluate analytic solution
  const auto vars = solution.variables(
      x, t,
      tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                 SoScalarWave::Tags::Phi<Dim>>{});
  // Get ingredients
  const size_t n_pts = mesh.number_of_grid_points();
  const auto& psi = get<SoScalarWave::Tags::Psi>(vars);
  const auto& pi = get<SoScalarWave::Tags::Pi>(vars);
  const auto& phi = get<SoScalarWave::Tags::Phi<Dim>>(vars);
  const auto unit_normal_one_form =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          x, 1. / sqrt(Dim));

  // Compute characteristic fields locally
  const auto phi_dot_normal = dot_product(unit_normal_one_form, phi);

  tnsr::i<DataVector, Dim, Frame::Inertial> phi_dot_projection_tensor{
      DataVector(n_pts)};
  for (size_t i = 0; i < Dim; ++i) {
    phi_dot_projection_tensor.get(i) =
        phi.get(i) - unit_normal_one_form.get(i) * get(phi_dot_normal);
  }

  const auto& vpsi_expected = psi;
  const auto& vzero_expected = phi_dot_projection_tensor;
  const Scalar<DataVector> vplus_expected{get(pi) + get(phi_dot_normal)};
  const Scalar<DataVector> vminus_expected{get(pi) - get(phi_dot_normal)};

  // Check that locally computed fields match returned ones
  Variables<
      tmpl::list<SoScalarWave::Tags::VPsi, SoScalarWave::Tags::VZero<Dim>,
                 SoScalarWave::Tags::VPlus, SoScalarWave::Tags::VMinus>>
      uvars{};
  SoScalarWave::Tags::CharacteristicFieldsCompute<Dim>::function(
      make_not_null(&uvars), psi, pi, phi, unit_normal_one_form);

  const auto& vpsi = get<SoScalarWave::Tags::VPsi>(uvars);
  const auto& vzero = get<SoScalarWave::Tags::VZero<Dim>>(uvars);
  const auto& vplus = get<SoScalarWave::Tags::VPlus>(uvars);
  const auto& vminus = get<SoScalarWave::Tags::VMinus>(uvars);

  CHECK_ITERABLE_APPROX(vpsi_expected, vpsi);
  CHECK_ITERABLE_APPROX(vzero_expected, vzero);
  CHECK_ITERABLE_APPROX(vplus_expected, vplus);
  CHECK_ITERABLE_APPROX(vminus_expected, vminus);
}
}  // namespace

namespace {
template <typename Tag, size_t Dim>
typename Tag::type evol_field_with_tag(
    const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  Variables<tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                       SoScalarWave::Tags::Phi<Dim>>>
      evolved_vars{};
  SoScalarWave::Tags::EvolvedFieldsFromCharacteristicFieldsCompute<
      Dim>::function(make_not_null(&evolved_vars), v_psi, v_zero, v_plus,
                     v_minus, unit_normal_one_form);
  return get<Tag>(evolved_vars);
}

template <size_t Dim>
void test_evolved_from_characteristic_fields() {
  TestHelpers::db::test_compute_tag<
      SoScalarWave::Tags::EvolvedFieldsFromCharacteristicFieldsCompute<Dim>>(
      "EvolvedFieldsFromCharacteristicFields");
  const DataVector used_for_size(5);
  // Psi
  pypp::check_with_random_values<1>(
      evol_field_with_tag<SoScalarWave::Tags::Psi, Dim>, "Characteristics",
      "evol_field_psi", {{{-10., 10.}}}, used_for_size);
  // Pi
  pypp::check_with_random_values<1>(
      evol_field_with_tag<SoScalarWave::Tags::Pi, Dim>, "Characteristics",
      "evol_field_pi", {{{-10., 10.}}}, used_for_size);
  // Phi
  pypp::check_with_random_values<1>(
      evol_field_with_tag<SoScalarWave::Tags::Phi<Dim>, Dim>,
      "Characteristics", "evol_field_phi", {{{-10., 10.}}}, used_for_size);
}

// Test return-by-reference evolved fields by comparing to analytic solution
template <size_t Dim, typename Solution>
void test_evolved_from_characteristic_fields_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, Dim>& lower_bound,
    const std::array<double, Dim>& upper_bound) {
  // Set up grid
  Mesh<Dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  const double t = 0.;

  // Evaluate analytic solution
  const auto vars = solution.variables(
      x, t,
      tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                 SoScalarWave::Tags::Phi<Dim>>{});
  // Get ingredients
  const size_t n_pts = mesh.number_of_grid_points();
  const auto& psi_expected = get<SoScalarWave::Tags::Psi>(vars);
  const auto& pi_expected = get<SoScalarWave::Tags::Pi>(vars);
  const auto& phi_expected = get<SoScalarWave::Tags::Phi<Dim>>(vars);
  const auto unit_normal_one_form =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          x, 1. / sqrt(Dim));

  // Compute characteristic fields locally
  const auto phi_dot_normal = dot_product(unit_normal_one_form, phi_expected);

  tnsr::i<DataVector, Dim, Frame::Inertial> phi_dot_projection_tensor{
      DataVector(n_pts)};
  for (size_t i = 0; i < Dim; ++i) {
    phi_dot_projection_tensor.get(i) =
        phi_expected.get(i) - unit_normal_one_form.get(i) * get(phi_dot_normal);
  }

  const auto& vpsi = psi_expected;
  const auto& vzero = phi_dot_projection_tensor;
  const Scalar<DataVector> vplus{get(pi_expected) + get(phi_dot_normal)};
  const Scalar<DataVector> vminus{get(pi_expected) - get(phi_dot_normal)};
  // Obtain evolved fields using compute tag
  {
    Variables<tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                         SoScalarWave::Tags::Phi<Dim>>>
        fields{};
    SoScalarWave::Tags::EvolvedFieldsFromCharacteristicFieldsCompute<
        Dim>::function(make_not_null(&fields), vpsi, vzero, vplus, vminus,
                       unit_normal_one_form);
    const auto& psi = get<SoScalarWave::Tags::Psi>(fields);
    const auto& pi = get<SoScalarWave::Tags::Pi>(fields);
    const auto& phi = get<SoScalarWave::Tags::Phi<Dim>>(fields);

    CHECK_ITERABLE_APPROX(psi_expected, psi);
    CHECK_ITERABLE_APPROX(pi_expected, pi);
    CHECK_ITERABLE_APPROX(phi_expected, phi);
  }
  // Obtain evolved fields using function
  {
    const auto fields =
        SoScalarWave::evolved_fields_from_characteristic_fields(
            vpsi, vzero, vplus, vminus, unit_normal_one_form);
    const auto& psi = get<SoScalarWave::Tags::Psi>(fields);
    const auto& pi = get<SoScalarWave::Tags::Pi>(fields);
    const auto& phi = get<SoScalarWave::Tags::Phi<Dim>>(fields);

    CHECK_ITERABLE_APPROX(psi_expected, psi);
    CHECK_ITERABLE_APPROX(pi_expected, pi);
    CHECK_ITERABLE_APPROX(phi_expected, phi);
  }
}
}  // namespace

// Test that characteristic_fields followed by
// evolved_fields_from_characteristic_fields is the identity, using random data
// and a random unit normal.
template <size_t Dim>
void test_roundtrip() {
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-10., 10.);
  const size_t n_pts = 5;

  // Random evolved fields
  auto psi = make_with_value<Scalar<DataVector>>(DataVector(n_pts), 0.);
  auto pi = make_with_value<Scalar<DataVector>>(DataVector(n_pts), 0.);
  auto phi =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          DataVector(n_pts), 0.);
  fill_with_random_values(make_not_null(&psi), make_not_null(&gen),
                          make_not_null(&dist));
  fill_with_random_values(make_not_null(&pi), make_not_null(&gen),
                          make_not_null(&dist));
  fill_with_random_values(make_not_null(&phi), make_not_null(&gen),
                          make_not_null(&dist));

  // Random unit normal: generate random direction, then normalize
  auto normal =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          DataVector(n_pts), 0.);
  fill_with_random_values(make_not_null(&normal), make_not_null(&gen),
                          make_not_null(&dist));
  const auto mag = magnitude(normal);
  for (size_t i = 0; i < Dim; ++i) {
    normal.get(i) /= get(mag);
  }

  // Forward: evolved -> characteristic
  const auto char_fields =
      SoScalarWave::characteristic_fields(psi, pi, phi, normal);
  const auto& v_psi = get<SoScalarWave::Tags::VPsi>(char_fields);
  const auto& v_zero = get<SoScalarWave::Tags::VZero<Dim>>(char_fields);
  const auto& v_plus = get<SoScalarWave::Tags::VPlus>(char_fields);
  const auto& v_minus = get<SoScalarWave::Tags::VMinus>(char_fields);

  // Inverse: characteristic -> evolved
  const auto recovered =
      SoScalarWave::evolved_fields_from_characteristic_fields(
          v_psi, v_zero, v_plus, v_minus, normal);

  CHECK_ITERABLE_APPROX(psi, get<SoScalarWave::Tags::Psi>(recovered));
  CHECK_ITERABLE_APPROX(pi, get<SoScalarWave::Tags::Pi>(recovered));
  CHECK_ITERABLE_APPROX(phi, get<SoScalarWave::Tags::Phi<Dim>>(recovered));
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.SoScalarWave.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/SoScalarWave/"};

  test_characteristic_speeds<1>();
  test_characteristic_speeds<2>();
  test_characteristic_speeds<3>();

  test_characteristic_speeds_with_mesh_velocity<1>();
  test_characteristic_speeds_with_mesh_velocity<2>();
  test_characteristic_speeds_with_mesh_velocity<3>();

  test_characteristic_speeds_nullopt_matches<1>();
  test_characteristic_speeds_nullopt_matches<2>();
  test_characteristic_speeds_nullopt_matches<3>();

  test_characteristic_fields<1>();
  test_characteristic_fields<2>();
  test_characteristic_fields<3>();

  test_evolved_from_characteristic_fields<1>();
  test_evolved_from_characteristic_fields<2>();
  test_evolved_from_characteristic_fields<3>();

  test_roundtrip<1>();
  test_roundtrip<2>();
  test_roundtrip<3>();

  // Test characteristics against 3D plane wave
  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.22, 1.32}};
  const std::array<double, 3> upper_bound{{0.78, 1.18, 1.28}};

  const double kx = 1.5;
  const double ky = -7.2;
  const double kz = 2.7;
  const double center_x = 2.4;
  const double center_y = -4.8;
  const double center_z = 8.4;
  const SoScalarWave::Solutions::SoPlaneWave<3> plane_wave_solution(
      {{kx, ky, kz}}, {{center_x, center_y, center_z}},
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(3));

  test_characteristic_speeds_analytic<3>(grid_size);
  test_characteristic_fields_analytic<3>(plane_wave_solution, grid_size,
                                         lower_bound, upper_bound);
  test_evolved_from_characteristic_fields_analytic<3>(
      plane_wave_solution, grid_size, lower_bound, upper_bound);

  double largest_characteristic_speed =
      std::numeric_limits<double>::signaling_NaN();
  SoScalarWave::Tags::ComputeLargestCharacteristicSpeed::function(
      make_not_null(&largest_characteristic_speed));
  CHECK(largest_characteristic_speed == 1.0);
}
