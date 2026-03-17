// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/Ccz4/ATilde.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/ConstraintsRadiationPreserving.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Sommerfeld.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/DummyReconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SoTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Ccz4WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugePlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
namespace {

template <bool EnableSubcell>
struct DummyEvolutionMetaVars {
  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary = EnableSubcell;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<BoundaryConditions::BoundaryCondition,
                             BoundaryConditions::standard_boundary_conditions>>;
  };
};

using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

// Test second order CCZ4 in Minkowski spacetime
void test_minkowski(const bool evolve_lapse_and_shift) {
  // set up subcell grid
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 5;
  const Ccz4::fd::DummyReconstructor recons{};
  const size_t ghost_zone_size = recons.ghost_zone_size();
  const Mesh<SpatialDim> subcell_mesh{points_per_dimension,
                                      Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered};

  const std::array<double, SpatialDim> lower_bound{-2., 0., -0.5};
  const std::array<double, SpatialDim> upper_bound{2., 2., -0.1};
  const std::array<double, SpatialDim> coords_range = upper_bound - lower_bound;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });
  // set up displaced logical coords
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = coord_map(logical_coords);
  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                  Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < SpatialDim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  InverseHessian<DataVector, SpatialDim, Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_hessian{
          subcell_mesh.number_of_grid_points(), 0.0};

  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element();

  const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::Minkowski::
                  compute_prim_solution_for_Minkowski<false>,
              coords_range);

  // Get system evolved variables
  // Use the physical inertial coords
  const auto evolved_vars = TestHelpers::Ccz4::fd::detail::Minkowski::
      compute_prim_solution_for_Minkowski(x);

  const DataVector used_for_size =
      DataVector(subcell_mesh.number_of_grid_points(),
                 std::numeric_limits<double>::signaling_NaN());
  const auto k_0 = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, 3>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  // put needed quantities into databox
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, Ccz4::fd::System::variables_tag>;

  auto box = db::create<db::AddSimpleTags<
      ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
      ::Ccz4::fd::Tags::EvolveLapseAndShift, domain::Tags::Element<SpatialDim>,
      fd::Tags::Reconstructor,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<false>>,
      Ccz4::fd::System::variables_tag, ::Ccz4::Tags::Eta<DataVector>,
      ::Ccz4::Tags::K0<DataVector>,
      ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>,
      ::Ccz4::fd::Tags::ObserverCharacteristicFieldsTag<SpatialDim,
                                                         Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverConstraintCharacteristicFieldsTag<
          SpatialDim, Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverRadiationCharacteristicFieldsTag<
          SpatialDim, Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::ObserverConstraintCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::ObserverRadiationCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::InitialBoundaryCharacteristicFields<SpatialDim,
                                                             Frame::Inertial>,
      dt_variables_tag,
      evolution::dg::subcell::Tags::Mesh<SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseHessianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
      domain::Tags::ExternalBoundaryConditions<SpatialDim>,
      evolution::dg::subcell::Tags::Coordinates<SpatialDim, Frame::Inertial>>>(
      kappa_1, kappa_2, kappa_3, evolve_lapse_and_shift, element,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      DummyEvolutionMetaVars<false>{}, evolved_vars, eta, k_0,
      upper_spatial_z4_constraint,
      Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::constraint_characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::radiation_characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::constraint_characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::radiation_characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      std::optional<Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>>{std::nullopt},
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
      cell_centered_logical_to_inertial_inv_hessian, all_ghost_data,
      std::vector<DirectionMap<
          SpatialDim,
          std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>{},
      x);
  // Check that all time derivatives are 0
  ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));
  const auto zero = DataVector(used_for_size.size(), 0.0);

  tmpl::for_each<Ccz4::fd::System::variables_tag_list>(
    [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const std::string tag_name = db::tag_name<::Tags::dt<Tag>>();
        CAPTURE(tag_name);
        for (auto& component : get<::Tags::dt<Tag>>(box)) {
            CHECK_ITERABLE_APPROX(component, zero);
        }
  });
}

// Test second-order CCZ4 in KerrSchild spacetime
//
// evolve_shift: whether or not to evolve the shift (always true for SO-CCZ4);
// slicing_condition_type: which slicing condition to use (always 1+log for
// SO-CCZ4)
void test_kerrschild(const bool evolve_lapse_and_shift) {
  const bool evolve_shift = evolve_lapse_and_shift;
  const Ccz4::SlicingConditionType slicing_condition_type =
      Ccz4::SlicingConditionType::Log;  // always use 1+log slicing

  // set up subcell grid
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 20;
  const Ccz4::fd::DummyReconstructor recons{};
  const size_t ghost_zone_size = recons.ghost_zone_size();
  const Mesh<SpatialDim> subcell_mesh{points_per_dimension,
                                      Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered};

  const std::array<double, SpatialDim> lower_bound{0.8, 1., 1.3};
  const std::array<double, SpatialDim> upper_bound{1.2, 1.2, 1.4};
  const std::array<double, SpatialDim> coords_range = upper_bound - lower_bound;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });
  // set up displaced logical coords
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = coord_map(logical_coords);
  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                  Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < SpatialDim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  InverseHessian<DataVector, SpatialDim, Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_hessian{
          subcell_mesh.number_of_grid_points(), 0.0};

  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element();

  // Setup solution
  const double mass = 2.0;
  const std::array<double, SpatialDim> spin{{0.2, 0.4, 0.8}};
  const std::array<double, SpatialDim> center{{0.2, 0.5, 0.1}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  const double f = Ccz4::fd::System::f;

  const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::KerrSchild::
                  compute_prim_solution_for_KerrSchild,
              coords_range, t, f, evolve_shift, solution);

  const auto evolved_vars = TestHelpers::Ccz4::fd::detail::KerrSchild::
      compute_prim_solution_for_KerrSchild(x, t, f, evolve_shift, solution);

  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(evolved_vars);
  const auto d_lapse = partial_derivative(
      lapse, subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);
  const DataVector used_for_size =
      DataVector(subcell_mesh.number_of_grid_points(),
                 std::numeric_limits<double>::signaling_NaN());
  const auto eta = make_with_value<Scalar<DataVector>>(
      used_for_size, 0.1);                      // change eta to non-zero later
  const Scalar<DataVector> slicing_condition =  // g(\alpha)
      TestHelpers::Ccz4::fd::detail::KerrSchild::get_slicing_condition(
          slicing_condition_type, lapse);
  const auto k_0 = TestHelpers::Ccz4::fd::detail::KerrSchild::get_k_0_kerr(
      get<gr::Tags::Shift<DataVector, SpatialDim>>(evolved_vars), lapse,
      d_lapse, slicing_condition,
      get<::Ccz4::Tags::Theta<DataVector>>(evolved_vars),
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars));
  const auto upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, 3>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  // put needed quantities into databox
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, Ccz4::fd::System::variables_tag>;

  auto box = db::create<db::AddSimpleTags<
      ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
      ::Ccz4::fd::Tags::EvolveLapseAndShift, domain::Tags::Element<SpatialDim>,
      fd::Tags::Reconstructor,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<false>>,
      Ccz4::fd::System::variables_tag, ::Ccz4::Tags::Eta<DataVector>,
      ::Ccz4::Tags::K0<DataVector>,
      ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>,
      ::Ccz4::fd::Tags::ObserverCharacteristicFieldsTag<SpatialDim,
                                                         Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverConstraintCharacteristicFieldsTag<
          SpatialDim, Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverRadiationCharacteristicFieldsTag<
          SpatialDim, Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::ObserverConstraintCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::ObserverRadiationCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::InitialBoundaryCharacteristicFields<SpatialDim,
                                                             Frame::Inertial>,
      dt_variables_tag,
      evolution::dg::subcell::Tags::Mesh<SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseHessianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
      domain::Tags::ExternalBoundaryConditions<SpatialDim>,
      evolution::dg::subcell::Tags::Coordinates<SpatialDim, Frame::Inertial>>>(
      kappa_1, kappa_2, kappa_3, evolve_lapse_and_shift, element,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      DummyEvolutionMetaVars<false>{}, evolved_vars, eta, k_0,
      upper_spatial_z4_constraint,
      Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::constraint_characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::radiation_characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::constraint_characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::radiation_characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      std::optional<Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>>{std::nullopt},
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
      cell_centered_logical_to_inertial_inv_hessian, all_ghost_data,
      std::vector<DirectionMap<
          SpatialDim,
          std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>{},
      x);
  // Check that all time derivatives are 0
  ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));
  const auto zero = DataVector(used_for_size.size(), 0.0);
  const Approx custom_approx =
      Approx::custom().epsilon(1.0e-9).scale(*std::max_element(
          evolved_vars.data(), evolved_vars.data() + evolved_vars.size() - 1));

  // Remove dt_b from the list, which will be checked separately below
  tmpl::for_each<tmpl::remove<Ccz4::fd::System::variables_tag_list,
    ::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>>(
      [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const std::string tag_name = db::tag_name<::Tags::dt<Tag>>();
        CAPTURE(tag_name);
        for (auto& component : get<::Tags::dt<Tag>>(box)) {
          CHECK_ITERABLE_CUSTOM_APPROX(component, zero, custom_approx);
        }
      });

  // eq 12i
  // \partial_t b will not be 0 for KerrSchild if evolve_shift == true
  // since we assume the shift is time-independent for testing
  // but KerrSchild is not stationary under 1+log slicing
  const auto d_gamma_hat = partial_derivative(
      get<::Ccz4::Tags::GammaHat<DataVector, SpatialDim>>(box), subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian);
  const auto d_b = partial_derivative(
      get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>(box),
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);
  const tnsr::I<DataVector, SpatialDim, FrameType> dt_b_expected =
      TestHelpers::Ccz4::fd::detail::KerrSchild::get_dt_b_kerr_expected(
          evolve_shift, get<::Ccz4::Tags::Eta<DataVector>>(box),
          get<gr::Tags::Shift<DataVector, SpatialDim>>(box), d_gamma_hat,
          get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>(box), d_b);
  const auto& dt_b_actual =
      get<::Tags::dt<::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>>(
          box);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_b_actual, dt_b_expected, custom_approx);
}

// Test second-order CCZ4 in a GaugePlaneWave spacetime
void test_gauge_plane_wave(
    const std::array<double, 3>& wave_vector,
    std::unique_ptr<MathFunction<1, Frame::Inertial>> profile, const double t,
    const bool evolve_lapse_and_shift) {
  // set up subcell grid
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 20;
  const Ccz4::fd::DummyReconstructor recons{};
  const size_t ghost_zone_size = recons.ghost_zone_size();
  const Mesh<SpatialDim> subcell_mesh{points_per_dimension,
                                      Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered};
  const std::array<double, SpatialDim> lower_bound{-0.5, -2., 1.};
  const std::array<double, SpatialDim> upper_bound{0.0, 2., 2.};
  const std::array<double, SpatialDim> coords_range = upper_bound - lower_bound;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });
  // set up displaced logical coords
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = coord_map(logical_coords);
  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                  Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < SpatialDim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  InverseHessian<DataVector, SpatialDim, Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_hessian{
          subcell_mesh.number_of_grid_points(), 0.0};

  double omega = 0.0;
  for (const auto& k : wave_vector) {
    omega += square(k);
  }
  omega = pow(omega, 1.0 / 2.0);

  const DataVector used_for_size =
      DataVector(subcell_mesh.number_of_grid_points(),
                 std::numeric_limits<double>::signaling_NaN());

  tnsr::i<DataVector, SpatialDim, FrameType> k_tnsr{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    k_tnsr.get(i) =
        make_with_value<DataVector>(used_for_size, gsl::at(wave_vector, i));
  }

  const gr::Solutions::GaugePlaneWave<SpatialDim>::IntermediateVars<DataVector>
      intermediate_sol(wave_vector, profile, omega, x, t);
  const Scalar<DataVector> h{intermediate_sol.h};
  const Scalar<DataVector> du_h{intermediate_sol.du_h};
  const Scalar<DataVector> du_du_h{intermediate_sol.du_du_h};

  // Setup solutions
  const gr::Solutions::GaugePlaneWave<SpatialDim> solution(wave_vector,
                                                           std::move(profile));
  const auto gauge_plane_wave_vars = solution.variables(
      x, t,
      typename gr::Solutions::GaugePlaneWave<SpatialDim>::tags<DataVector>{});

  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element();

  const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
      all_ghost_data = TestHelpers::Ccz4::fd::detail::compute_ghost_data(
          subcell_mesh, x, element.neighbors(), ghost_zone_size,
          TestHelpers::Ccz4::fd::detail::GaugePlaneWave::
              compute_prim_solution_for_GaugePlaneWave,
          coords_range, t, solution, intermediate_sol);

  const auto evolved_vars = TestHelpers::Ccz4::fd::detail::GaugePlaneWave::
      compute_prim_solution_for_GaugePlaneWave(x, t, solution,
                                               intermediate_sol);

  const auto& k_0 =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars);

  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, 3>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  // put needed quantities into databox
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, Ccz4::fd::System::variables_tag>;

  auto box = db::create<db::AddSimpleTags<
      ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
      ::Ccz4::fd::Tags::EvolveLapseAndShift, domain::Tags::Element<SpatialDim>,
      fd::Tags::Reconstructor,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<false>>,
      Ccz4::fd::System::variables_tag, ::Ccz4::Tags::Eta<DataVector>,
      ::Ccz4::Tags::K0<DataVector>,
      ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>,
      ::Ccz4::fd::Tags::ObserverCharacteristicFieldsTag<SpatialDim,
                                                         Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverConstraintCharacteristicFieldsTag<
          SpatialDim, Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverRadiationCharacteristicFieldsTag<
          SpatialDim, Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::ObserverConstraintCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::ObserverRadiationCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::InitialBoundaryCharacteristicFields<SpatialDim,
                                                             Frame::Inertial>,
      dt_variables_tag,
      evolution::dg::subcell::Tags::Mesh<SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseHessianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
      domain::Tags::ExternalBoundaryConditions<SpatialDim>,
      evolution::dg::subcell::Tags::Coordinates<SpatialDim, Frame::Inertial>>>(
      kappa_1, kappa_2, kappa_3, evolve_lapse_and_shift, element,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      DummyEvolutionMetaVars<false>{}, evolved_vars, eta, k_0,
      upper_spatial_z4_constraint,
      Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::constraint_characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::radiation_characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::constraint_characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::radiation_characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      std::optional<Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>>{std::nullopt},
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
      cell_centered_logical_to_inertial_inv_hessian, all_ghost_data,
      std::vector<DirectionMap<
          SpatialDim,
          std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>{},
      x);
  // Check all time derivatives
  ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));

  const Approx custom_approx =
      Approx::custom().epsilon(1.0e-9).scale(*std::max_element(
          evolved_vars.data(), evolved_vars.data() + evolved_vars.size() - 1));
  // eq 12a
  const auto& dt_conformal_spatial_metric_actual =
      get<::Tags::dt<::Ccz4::Tags::ConformalMetric<DataVector, SpatialDim>>>(
          box);

  const auto conformal_factor =
      get(get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars));
  Scalar<DataVector> conformal_factor_squared{};
  get(conformal_factor_squared) = square(conformal_factor);
  const auto& dt_spatial_metric =
      get<::Tags::dt<gr::Tags::SpatialMetric<DataVector, SpatialDim>>>(
          gauge_plane_wave_vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim, FrameType>>(
          gauge_plane_wave_vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, SpatialDim, FrameType>>(
          gauge_plane_wave_vars);
  const auto dt_conformal_spatial_metric_expected = TestHelpers::Ccz4::fd::
      detail::GaugePlaneWave::get_dt_conformal_spatial_metric_gauge_plane_wave(
          conformal_factor_squared, spatial_metric, inverse_spatial_metric,
          dt_spatial_metric);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_conformal_spatial_metric_actual,
                               dt_conformal_spatial_metric_expected,
                               custom_approx);

  // eq 12b
  const auto& dt_lapse_actual =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(box);
  const auto& d_lapse =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<SpatialDim>,
                        FrameType>>(gauge_plane_wave_vars);
  auto dt_lapse_expected = make_with_value<Scalar<DataVector>>(
          used_for_size, 0.0);
  if (evolve_lapse_and_shift) {
      dt_lapse_expected = TestHelpers::Ccz4::fd::detail::GaugePlaneWave::
      get_dt_lapse_gauge_plane_wave(
          d_lapse, get<gr::Tags::Shift<DataVector, SpatialDim, FrameType>>(
                       gauge_plane_wave_vars));
  }
  CHECK_ITERABLE_CUSTOM_APPROX(dt_lapse_actual, dt_lapse_expected,
                               custom_approx);

  // eq 12c
  const auto& dt_shift_actual =
      get<::Tags::dt<gr::Tags::Shift<DataVector, SpatialDim>>>(box);
  const auto& d_shift =
      get<::Tags::deriv<gr::Tags::Shift<DataVector, SpatialDim, FrameType>,
                        tmpl::size_t<SpatialDim>, FrameType>>(
          gauge_plane_wave_vars);
  auto dt_shift_expected = make_with_value<tnsr::I<DataVector, 3>>(
        used_for_size, 0.0);
  if (evolve_lapse_and_shift) {
    dt_shift_expected = TestHelpers::Ccz4::fd::detail::GaugePlaneWave::
        get_dt_shift_gauge_plane_wave(
            get<gr::Tags::Shift<DataVector, SpatialDim, FrameType>>(
                gauge_plane_wave_vars),
            d_shift);
  }
  CHECK_ITERABLE_CUSTOM_APPROX(dt_shift_actual, dt_shift_expected,
                               custom_approx);

  // eq 12d
  const auto& dt_conformal_factor_actual =
      get<::Tags::dt<::Ccz4::Tags::ConformalFactor<DataVector>>>(box);
  const auto dt_conformal_factor_expected = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_conformal_factor_gauge_plane_wave(
          inverse_spatial_metric, dt_spatial_metric,
          Scalar<DataVector>(conformal_factor));
  CHECK_ITERABLE_CUSTOM_APPROX(dt_conformal_factor_actual,
                               dt_conformal_factor_expected, custom_approx);

  // eq 12e
  const auto& dt_a_tilde_actual =
      get<::Tags::dt<::Ccz4::Tags::ATilde<DataVector, SpatialDim>>>(box);
  const auto dt_conformal_factor = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_conformal_factor_gauge_plane_wave(
          inverse_spatial_metric, dt_spatial_metric,
          Scalar<DataVector>(conformal_factor));
  const auto one_plus_h_times_omega_squared = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_one_plus_h_times_omega_squared(h, omega);
  const auto dt_extrinsic_curvature = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_extrinsic_curvature_gauge_plane_wave(
          k_tnsr, du_h, du_du_h, one_plus_h_times_omega_squared, omega);

  const auto dt_inverse_spatial_metric = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_inverse_spatial_metric(inverse_spatial_metric,
                                                    dt_spatial_metric);
  const auto dt_trace_extrinsic_curvature_expected = TestHelpers::Ccz4::fd::
      detail::GaugePlaneWave::get_dt_trace_extrinsic_curvature_gauge_plane_wave(
          get<gr::Tags::ExtrinsicCurvature<DataVector, SpatialDim>>(
              gauge_plane_wave_vars),
          dt_extrinsic_curvature, inverse_spatial_metric,
          dt_inverse_spatial_metric);
  const auto dt_a_tilde_expected = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_a_tilde_gauge_plane_wave(
          Scalar<DataVector>(conformal_factor), conformal_factor_squared,
          dt_conformal_factor, spatial_metric, dt_spatial_metric,
          get<gr::Tags::ExtrinsicCurvature<DataVector, SpatialDim>>(
              gauge_plane_wave_vars),
          dt_extrinsic_curvature,
          get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars),
          dt_trace_extrinsic_curvature_expected);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_a_tilde_actual, dt_a_tilde_expected,
                               custom_approx);

  // eq 12f
  const auto& dt_trace_extrinsic_curvature_actual =
      get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(box);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_trace_extrinsic_curvature_actual,
                               dt_trace_extrinsic_curvature_expected,
                               custom_approx);

  // eq 12g
  const auto& dt_theta_actual =
      get<::Tags::dt<::Ccz4::Tags::Theta<DataVector>>>(box);
  const auto& dt_theta_expected =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_theta_actual, dt_theta_expected,
                               custom_approx);

  // eq 12h
  const auto& dt_gamma_hat_actual =
      get<::Tags::dt<::Ccz4::Tags::GammaHat<DataVector, SpatialDim>>>(box);
  const auto dt_conformal_spatial_metric = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_conformal_spatial_metric_gauge_plane_wave(
          conformal_factor_squared, spatial_metric, inverse_spatial_metric,
          dt_spatial_metric);
  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(
          get<::Ccz4::Tags::ConformalMetric<DataVector, SpatialDim>>(
              evolved_vars))
          .second;
  const auto dt_inverse_conformal_spatial_metric = TestHelpers::Ccz4::fd::
      detail::GaugePlaneWave::get_dt_inverse_conformal_spatial_metric(
          inverse_conformal_spatial_metric, dt_conformal_spatial_metric);
  const auto dt_d_spatial_metric =
      TestHelpers::Ccz4::fd::detail::GaugePlaneWave::
          get_dt_d_spatial_metric_gauge_plane_wave(k_tnsr, du_du_h, omega);
  const auto& d_spatial_metric =
      get<::Tags::deriv<gr::Tags::SpatialMetric<DataVector, SpatialDim>,
                        tmpl::size_t<SpatialDim>, FrameType>>(
          gauge_plane_wave_vars);
  const auto d_conformal_factor = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_d_conformal_factor_gauge_plane_wave(
          inverse_spatial_metric, d_spatial_metric,
          Scalar<DataVector>(conformal_factor));
  const auto dt_d_conformal_factor = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_d_conformal_factor_gauge_plane_wave(
          Scalar<DataVector>(conformal_factor), dt_conformal_factor,
          inverse_spatial_metric, dt_inverse_spatial_metric, d_spatial_metric,
          dt_d_spatial_metric);
  const auto dt_d_conformal_spatial_metric = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_d_conformal_spatial_metric_gauge_plane_wave(
          spatial_metric, dt_spatial_metric, d_spatial_metric,
          dt_d_spatial_metric, Scalar<DataVector>(conformal_factor),
          dt_conformal_factor, d_conformal_factor, dt_d_conformal_factor);
  const auto det_spatial_metric = determinant(spatial_metric);
  const auto d_det_spatial_metric = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_d_det_spatial_metric_gauge_plane_wave(
          det_spatial_metric,
          get<gr::Tags::InverseSpatialMetric<DataVector, SpatialDim,
                                             FrameType>>(gauge_plane_wave_vars),
          get<::Tags::deriv<gr::Tags::SpatialMetric<DataVector, SpatialDim>,
                            tmpl::size_t<SpatialDim>, FrameType>>(
              gauge_plane_wave_vars));
  const tnsr::ijj<DataVector, SpatialDim, FrameType>
      d_conformal_spatial_metric = TestHelpers::Ccz4::fd::detail::KerrSchild::
          get_d_conformal_spatial_metric(conformal_factor_squared,
                                         spatial_metric, d_spatial_metric,
                                         d_det_spatial_metric);
  const auto dt_gamma_hat_expected = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_gamma_hat_gauge_plane_wave(
          inverse_conformal_spatial_metric, dt_inverse_conformal_spatial_metric,
          d_conformal_spatial_metric, dt_d_conformal_spatial_metric);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_gamma_hat_actual, dt_gamma_hat_expected,
                               custom_approx);

  // eq 12i
  auto dt_b_expected =
        make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 0.0);
  if (evolve_lapse_and_shift) {
    const tnsr::ijj<DataVector, SpatialDim, FrameType> field_d =
        TestHelpers::Ccz4::fd::detail::KerrSchild::get_field_d(
            d_conformal_spatial_metric);
    const tnsr::iJJ<DataVector, SpatialDim, FrameType> field_d_up =
        TestHelpers::Ccz4::fd::detail::GaugePlaneWave::get_field_d_up(
            inverse_conformal_spatial_metric, field_d);
    const auto d_field_d = partial_derivative(
        field_d, subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);
    const auto d_conformal_christoffel_second_kind =
        ::Ccz4::deriv_conformal_christoffel_second_kind(
            inverse_conformal_spatial_metric, field_d, d_field_d, field_d_up);
    const auto conformal_christoffel_second_kind =
        ::Ccz4::conformal_christoffel_second_kind(
            inverse_conformal_spatial_metric, field_d);
    const auto d_gamma_hat =
        ::Ccz4::deriv_contracted_conformal_christoffel_second_kind(
            inverse_conformal_spatial_metric, field_d_up,
            conformal_christoffel_second_kind,
            d_conformal_christoffel_second_kind);
    dt_b_expected = TestHelpers::
        Ccz4::fd::detail::GaugePlaneWave::get_dt_b_gauge_plane_wave_expected(
            dt_gamma_hat_expected, d_gamma_hat,
            get<gr::Tags::Shift<DataVector, SpatialDim, FrameType>>(
                gauge_plane_wave_vars));
  }
  const auto& dt_b_actual =
      get<::Tags::dt<::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>>(
          box);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_b_actual, dt_b_expected, custom_approx);
}

void test_constraint_radiation_preserving_bc(
    const bool evolve_lapse_and_shift,
    std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>
        boundary_condition) {
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 10;
  const Ccz4::fd::DummyReconstructor recons{};
  const size_t ghost_zone_size = recons.ghost_zone_size();
  const Mesh<SpatialDim> subcell_mesh{points_per_dimension,
                                      Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered};

  const std::array<double, ::Ccz4::fd::System::volume_dim> lower_bound{5.8, 5.0,
                                                                       2.3};
  const std::array<double, ::Ccz4::fd::System::volume_dim> upper_bound{6.2, 5.2,
                                                                       2.4};
  const std::array<double, SpatialDim> coords_range = upper_bound - lower_bound;

  // Create an element with an external boundary by omitting the last neighbor
  // in the upper_zeta direction.
  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element(true);

  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);

  const auto grid_to_inertial_map =
      domain::make_coordinate_map<Frame::Grid, FrameType>(
          domain::CoordinateMaps::Identity<3>{});

  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                  Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < SpatialDim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  InverseHessian<DataVector, SpatialDim, Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_hessian{
          subcell_mesh.number_of_grid_points(), 0.0};

  const DataVector used_for_size(subcell_mesh.number_of_grid_points(),
                                 std::numeric_limits<double>::signaling_NaN());

  const auto k_0 = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, 3>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, Ccz4::fd::System::variables_tag>;

  const size_t num_pts = subcell_mesh.number_of_grid_points();
  const size_t num_face_pts = subcell_mesh.extents(0) * subcell_mesh.extents(1);

  const Approx approx = Approx::custom().epsilon(1.0e-9).scale(1.0);

  // Test Minkowski case
  {
    INFO("Testing constraint-preserving BCs in Minkowski spacetime");
    // We cannot declare element_map const because it is not copyable into the
    // box NOLINTNEXTLINE(misc-const-correctness)
    ElementMap element_map{
        element.id(),
        domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(
            Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                     Affine{-1., 1., lower_bound[1], upper_bound[1]},
                     Affine{-1., 1., lower_bound[2], upper_bound[2]}})
            .get_clone()};
    const auto x = grid_to_inertial_map(element_map(logical_coords));

    auto evolved_vars = TestHelpers::Ccz4::fd::detail::Minkowski::
        compute_prim_solution_for_Minkowski(x);

    // Ghost data from interior neighbors (none for the external face, which is
    // set by BCs)
    const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
        all_ghost_data =
            TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
                subcell_mesh, x, element.neighbors(), ghost_zone_size,
                TestHelpers::Ccz4::fd::detail::Minkowski::
                    compute_prim_solution_for_Minkowski<false>,
                coords_range);

    // Provide BC on the external face (upper_zeta)
    std::vector<DirectionMap<
        SpatialDim,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
        external_bcs_per_block(1);
    external_bcs_per_block[0][Direction<SpatialDim>::upper_zeta()] =
        boundary_condition->get_clone();

    // NOLINTNEXTLINE(misc-const-correctness)
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time{};

    auto box = db::create<db::AddSimpleTags<
        ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
        ::Ccz4::fd::Tags::EvolveLapseAndShift,
        domain::Tags::Element<SpatialDim>, fd::Tags::Reconstructor,
        Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<true>>,
        Ccz4::fd::System::variables_tag, ::Ccz4::Tags::Eta<DataVector>,
        ::Ccz4::Tags::K0<DataVector>,
        ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>,
        ::Ccz4::fd::Tags::ObserverCharacteristicFieldsTag<SpatialDim,
                                                           Frame::Inertial>,
        ::Ccz4::fd::Tags::ObserverConstraintCharacteristicFieldsTag<
            SpatialDim, Frame::Inertial>,
        ::Ccz4::fd::Tags::ObserverRadiationCharacteristicFieldsTag<
            SpatialDim, Frame::Inertial>,
        ::Ccz4::fd::Tags::ObserverCharacteristicSpeedsTag,
        ::Ccz4::fd::Tags::ObserverConstraintCharacteristicSpeedsTag,
        ::Ccz4::fd::Tags::ObserverRadiationCharacteristicSpeedsTag,
        ::Ccz4::fd::Tags::InitialBoundaryCharacteristicFields<SpatialDim,
                                                               Frame::Inertial>,
        dt_variables_tag,
        evolution::dg::subcell::Tags::Mesh<SpatialDim>,
        evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
            SpatialDim>,
        evolution::dg::subcell::fd::Tags::InverseHessianLogicalToInertial<
            SpatialDim>,
        evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
        domain::Tags::ExternalBoundaryConditions<SpatialDim>,
        evolution::dg::subcell::Tags::Coordinates<SpatialDim, Frame::Inertial>,
        ::Tags::Time, domain::Tags::FunctionsOfTime,
        domain::Tags::ElementMap<SpatialDim, Frame::Grid>,
        domain::CoordinateMaps::Tags::CoordinateMap<SpatialDim, Frame::Grid,
                                                    Frame::Inertial>>>(
        kappa_1, kappa_2, kappa_3, evolve_lapse_and_shift, element,
        std::unique_ptr<Ccz4::fd::Reconstructor>{
            std::make_unique<std::decay_t<decltype(recons)>>(recons)},
        DummyEvolutionMetaVars<true>{}, evolved_vars, eta, k_0,
        upper_spatial_z4_constraint,
        Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
            DataVector, SpatialDim, Frame::Inertial>>{
            subcell_mesh.number_of_grid_points(), 0.0},
        Variables<::Ccz4::fd::Tags::constraint_characteristic_fields_tags_list<
            DataVector, SpatialDim, Frame::Inertial>>{
            subcell_mesh.number_of_grid_points(), 0.0},
        Variables<::Ccz4::fd::Tags::radiation_characteristic_fields_tags_list<
            DataVector, SpatialDim, Frame::Inertial>>{
            subcell_mesh.number_of_grid_points(), 0.0},
        Variables<::Ccz4::fd::Tags::characteristic_speeds_tags_list>{
            subcell_mesh.number_of_grid_points(), 0.0},
        Variables<::Ccz4::fd::Tags::constraint_characteristic_speeds_tags_list>{
            subcell_mesh.number_of_grid_points(), 0.0},
        Variables<::Ccz4::fd::Tags::radiation_characteristic_speeds_tags_list>{
            subcell_mesh.number_of_grid_points(), 0.0},
        std::optional<
            Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
                DataVector, SpatialDim, Frame::Inertial>>>{std::nullopt},
        Variables<typename dt_variables_tag::tags_list>{
            subcell_mesh.number_of_grid_points()},
        subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
        cell_centered_logical_to_inertial_inv_hessian, all_ghost_data,
        std::move(external_bcs_per_block), x, 0.0, std::move(functions_of_time),
        std::move(element_map), grid_to_inertial_map.get_clone());

    ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));

    tmpl::for_each<Ccz4::fd::System::variables_tag_list>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          const std::string tag_name = db::tag_name<::Tags::dt<Tag>>();
          CAPTURE(tag_name);
          for (auto& component : get<::Tags::dt<Tag>>(box)) {
            CHECK_ITERABLE_CUSTOM_APPROX(component, DataVector(num_pts, 0.0),
                                         approx);
          }
        });
  }
  // Test KerrSchild case
  {
    INFO("Testing constraint-preserving BCs in KerrSchild spacetime");
    const double t = std::numeric_limits<double>::signaling_NaN();
    const double f = Ccz4::fd::System::f;
    // Setup solution
    const double mass = 2.0;
    const std::array<double, Dim> spin{{0.2, 0.4, 0.8}};
    const std::array<double, Dim> center{{0.2, 0.5, 0.1}};
    const gr::Solutions::KerrSchild solution(mass, spin, center);

    // We cannot declare element_map const because it is not copyable into the
    // box NOLINTNEXTLINE(misc-const-correctness)
    ElementMap element_map{
        element.id(),
        domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(
            Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                     Affine{-1., 1., lower_bound[1], upper_bound[1]},
                     Affine{-1., 1., lower_bound[2], upper_bound[2]}})
            .get_clone()};
    const auto x = grid_to_inertial_map(element_map(logical_coords));

    const auto evolved_vars = TestHelpers::Ccz4::fd::detail::KerrSchild::
        compute_prim_solution_for_KerrSchild(x, t, f, evolve_lapse_and_shift,
                                             solution);
    // Ghost data from interior neighbors (none for the external face, which is
    // set by BCs)
    const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
        all_ghost_data =
            TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
                subcell_mesh, x, element.neighbors(), ghost_zone_size,
                TestHelpers::Ccz4::fd::detail::KerrSchild::
                    compute_prim_solution_for_KerrSchild,
                coords_range, t, f, evolve_lapse_and_shift, solution);

    // Provide BC on the external face (upper_zeta)
    std::vector<DirectionMap<
        SpatialDim,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
        external_bcs_per_block(1);
    external_bcs_per_block[0][Direction<SpatialDim>::upper_zeta()] =
        boundary_condition->get_clone();

    // NOLINTNEXTLINE(misc-const-correctness)
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time{};

    auto box = db::create<db::AddSimpleTags<
        ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
        ::Ccz4::fd::Tags::EvolveLapseAndShift,
        domain::Tags::Element<SpatialDim>, fd::Tags::Reconstructor,
        Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<true>>,
        Ccz4::fd::System::variables_tag, ::Ccz4::Tags::Eta<DataVector>,
        ::Ccz4::Tags::K0<DataVector>,
        ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>,
        ::Ccz4::fd::Tags::ObserverCharacteristicFieldsTag<SpatialDim,
                                                           Frame::Inertial>,
        ::Ccz4::fd::Tags::ObserverConstraintCharacteristicFieldsTag<
            SpatialDim, Frame::Inertial>,
        ::Ccz4::fd::Tags::ObserverRadiationCharacteristicFieldsTag<
            SpatialDim, Frame::Inertial>,
        ::Ccz4::fd::Tags::ObserverCharacteristicSpeedsTag,
        ::Ccz4::fd::Tags::ObserverConstraintCharacteristicSpeedsTag,
        ::Ccz4::fd::Tags::ObserverRadiationCharacteristicSpeedsTag,
        ::Ccz4::fd::Tags::InitialBoundaryCharacteristicFields<SpatialDim,
                                                               Frame::Inertial>,
        dt_variables_tag,
        evolution::dg::subcell::Tags::Mesh<SpatialDim>,
        evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
            SpatialDim>,
        evolution::dg::subcell::fd::Tags::InverseHessianLogicalToInertial<
            SpatialDim>,
        evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
        domain::Tags::ExternalBoundaryConditions<SpatialDim>,
        evolution::dg::subcell::Tags::Coordinates<SpatialDim, Frame::Inertial>,
        ::Tags::Time, domain::Tags::FunctionsOfTime,
        domain::Tags::ElementMap<SpatialDim, Frame::Grid>,
        domain::CoordinateMaps::Tags::CoordinateMap<SpatialDim, Frame::Grid,
                                                    Frame::Inertial>>>(
        kappa_1, kappa_2, kappa_3, evolve_lapse_and_shift, element,
        std::unique_ptr<Ccz4::fd::Reconstructor>{
            std::make_unique<std::decay_t<decltype(recons)>>(recons)},
        DummyEvolutionMetaVars<true>{}, evolved_vars, eta, k_0,
        upper_spatial_z4_constraint,
        Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
            DataVector, SpatialDim, Frame::Inertial>>{
            subcell_mesh.number_of_grid_points(), 0.0},
        Variables<::Ccz4::fd::Tags::constraint_characteristic_fields_tags_list<
            DataVector, SpatialDim, Frame::Inertial>>{
            subcell_mesh.number_of_grid_points(), 0.0},
        Variables<::Ccz4::fd::Tags::radiation_characteristic_fields_tags_list<
            DataVector, SpatialDim, Frame::Inertial>>{
            subcell_mesh.number_of_grid_points(), 0.0},
        Variables<::Ccz4::fd::Tags::characteristic_speeds_tags_list>{
            subcell_mesh.number_of_grid_points(), 0.0},
        Variables<::Ccz4::fd::Tags::constraint_characteristic_speeds_tags_list>{
            subcell_mesh.number_of_grid_points(), 0.0},
        Variables<::Ccz4::fd::Tags::radiation_characteristic_speeds_tags_list>{
            subcell_mesh.number_of_grid_points(), 0.0},
        std::optional<
            Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
                DataVector, SpatialDim, Frame::Inertial>>>{std::nullopt},
        Variables<typename dt_variables_tag::tags_list>{
            subcell_mesh.number_of_grid_points()},
        subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
        cell_centered_logical_to_inertial_inv_hessian, all_ghost_data,
        std::move(external_bcs_per_block), x, 0.0, std::move(functions_of_time),
        std::move(element_map), grid_to_inertial_map.get_clone());

    ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));

    using deriv_var_tag =
        db::wrap_tags_in<::Tags::deriv, System::gradients_tags,
                         tmpl::size_t<Dim>, Frame::Inertial>;
    Variables<deriv_var_tag> deriv_vars{num_pts};
    ::Ccz4::fd::spacetime_derivatives(
        make_not_null(&deriv_vars), evolved_vars,
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<
            SpatialDim>>(box),
        4, subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);

    using second_deriv_var_tag =
        db::wrap_tags_in<::Tags::second_deriv, System::gradients_tags,
                         tmpl::size_t<Dim>, Frame::Inertial>;
    Variables<second_deriv_var_tag> second_deriv_vars{num_pts};
    Ccz4::fd::second_spacetime_derivatives(
        make_not_null(&second_deriv_vars), evolved_vars,
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<
            SpatialDim>>(box),
        4, subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
        cell_centered_logical_to_inertial_inv_hessian);

    const auto& conformal_spatial_metric =
        get<::Ccz4::Tags::ConformalMetric<DataVector, SpatialDim>>(
            evolved_vars);
    const auto& conformal_factor =
        get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars);
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(evolved_vars);
    const auto& shift =
        get<gr::Tags::Shift<DataVector, SpatialDim>>(evolved_vars);
    const auto& trace_extrinsic_curvature =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars);
    const auto& theta = get<::Ccz4::Tags::Theta<DataVector>>(evolved_vars);

    auto dt_conformal_spatial_metric =
        make_with_value<tnsr::ii<DataVector, SpatialDim>>(used_for_size, 0.0);
    auto dt_a_tilde =
        make_with_value<tnsr::ii<DataVector, SpatialDim>>(used_for_size, 0.0);
    auto dt_conformal_factor =
        make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
    auto dt_trace_extrinsic_curvature =
        make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
    auto dt_theta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
    auto dt_gamma_hat =
        make_with_value<tnsr::I<DataVector, SpatialDim>>(used_for_size, 0.0);
    auto dt_lapse = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
    const auto& d_lapse =
        get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<SpatialDim>,
                          FrameType>>(deriv_vars);
    ::tenex::evaluate(
        make_not_null(&dt_lapse),
        -2.0 * lapse() * (trace_extrinsic_curvature() - k_0() - 2.0 * theta()) +
            shift(ti::K) * d_lapse(ti::k));
    auto dt_shift =
        make_with_value<tnsr::I<DataVector, SpatialDim>>(used_for_size, 0.0);
    auto dt_d_conformal_spatial_metric =
        make_with_value<tnsr::ijj<DataVector, SpatialDim>>(used_for_size, 0.0);
    auto dt_d_conformal_factor =
        make_with_value<tnsr::i<DataVector, SpatialDim>>(used_for_size, 0.0);
    const auto& d_trace_extrinsic_curvature =
        get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                          tmpl::size_t<SpatialDim>, FrameType>>(deriv_vars);
    const auto& d_theta =
        get<::Tags::deriv<::Ccz4::Tags::Theta<DataVector>,
                          tmpl::size_t<SpatialDim>, FrameType>>(deriv_vars);
    const auto& d_shift =
        get<::Tags::deriv<gr::Tags::Shift<DataVector, SpatialDim, FrameType>,
                          tmpl::size_t<SpatialDim>, FrameType>>(deriv_vars);
    const auto& d_d_lapse =
        get<::Tags::second_deriv<gr::Tags::Lapse<DataVector>,
                                 tmpl::size_t<SpatialDim>, FrameType>>(
            second_deriv_vars);
    auto dt_d_lapse =
        make_with_value<tnsr::i<DataVector, SpatialDim>>(used_for_size, 0.0);
    ::tenex::evaluate<ti::i>(
        make_not_null(&dt_d_lapse),
        -2.0 * lapse() *
                (d_trace_extrinsic_curvature(ti::i) - 2.0 * d_theta(ti::i)) -
            2.0 * d_lapse(ti::i) *
                (trace_extrinsic_curvature() - k_0() - 2.0 * theta()) +
            d_shift(ti::i, ti::K) * d_lapse(ti::k) +
            shift(ti::K) * d_d_lapse(ti::i, ti::k));
    auto dt_d_shift =
        make_with_value<tnsr::iJ<DataVector, SpatialDim>>(used_for_size, 0.0);
    const auto& d_b =
        get<::Tags::deriv<::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>,
                          tmpl::size_t<SpatialDim>, FrameType>>(deriv_vars);
    const auto& d_d_shift =
        get<::Tags::second_deriv<gr::Tags::Shift<DataVector, SpatialDim>,
                                 tmpl::size_t<SpatialDim>, FrameType>>(
            second_deriv_vars);
    ::tenex::evaluate<ti::k, ti::I>(make_not_null(&dt_d_shift),
                                    f * d_b(ti::k, ti::I));
    if (System::shifting_shift) {
      ::tenex::update<ti::k, ti::I>(
          make_not_null(&dt_d_shift),
          dt_d_shift(ti::k, ti::I) +
              d_shift(ti::k, ti::L) * d_shift(ti::l, ti::I) +
              shift(ti::L) * d_d_shift(ti::k, ti::l, ti::I));
    }

    const auto& d_gamma_hat =
        get<::Tags::deriv<::Ccz4::Tags::GammaHat<DataVector, SpatialDim>,
                          tmpl::size_t<SpatialDim>, FrameType>>(deriv_vars);
    const auto& auxiliary_field_b =
        get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>(
            evolved_vars);
    const tnsr::I<DataVector, SpatialDim, FrameType> dt_auxiliary_field_b =
        TestHelpers::Ccz4::fd::detail::KerrSchild::get_dt_b_kerr_expected(
            evolve_lapse_and_shift, eta, shift, d_gamma_hat, auxiliary_field_b,
            d_b);
    const auto normal_vector = x;
    tnsr::ii<DataVector, SpatialDim, FrameType> spatial_metric;
    ::tenex::evaluate<ti::i, ti::j>(
        make_not_null(&spatial_metric),
        conformal_spatial_metric(ti::i, ti::j) /
            (conformal_factor() * conformal_factor()));
    Scalar<DataVector> normal_vector_magnitude =
        ::tenex::evaluate(spatial_metric(ti::i, ti::j) * normal_vector(ti::I) *
                          normal_vector(ti::J));
    get(normal_vector_magnitude) = sqrt(get(normal_vector_magnitude));
    const auto unit_normal_vector = ::tenex::evaluate<ti::I>(
        normal_vector(ti::I) / normal_vector_magnitude());
    const auto unit_normal_one_form = ::tenex::evaluate<ti::i>(
        spatial_metric(ti::i, ti::j) * unit_normal_vector(ti::J));

    // Compute dt of characteristic variables
    auto dt_char_fields = dt_characteristic_fields(
        unit_normal_one_form, conformal_spatial_metric, conformal_factor, lapse,
        shift, dt_trace_extrinsic_curvature, dt_a_tilde, dt_theta, dt_gamma_hat,
        dt_auxiliary_field_b, dt_d_conformal_spatial_metric,
        dt_d_conformal_factor, dt_d_lapse, dt_d_shift, f);

    // Compute characteristic variables
    const auto& a_tilde =
        get<::Ccz4::Tags::ATilde<DataVector, SpatialDim>>(evolved_vars);
    const auto& d_conformal_spatial_metric =
        get<::Tags::deriv<::Ccz4::Tags::ConformalMetric<DataVector, SpatialDim>,
                          tmpl::size_t<SpatialDim>, FrameType>>(deriv_vars);
    const auto& d_conformal_factor =
        get<::Tags::deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                          tmpl::size_t<SpatialDim>, FrameType>>(deriv_vars);

    // Compute constraint characteristic fields
    const auto field_d = ::tenex::evaluate<ti::k, ti::i, ti::j>(
        0.5 * d_conformal_spatial_metric(ti::k, ti::i, ti::j));
    const auto inverse_conformal_spatial_metric =
        determinant_and_inverse(conformal_spatial_metric).second;
    const auto conformal_christoffel =
        ::Ccz4::conformal_christoffel_second_kind(
            inverse_conformal_spatial_metric, field_d);
    const auto contracted_conformal_christoffel =
        ::Ccz4::contracted_conformal_christoffel_second_kind(
            inverse_conformal_spatial_metric, conformal_christoffel);
    const auto& gamma_hat =
        get<::Ccz4::Tags::GammaHat<DataVector, ::Ccz4::fd::System::volume_dim>>(
            evolved_vars);
    const auto gamma_hat_minus_contracted_conformal_christoffel =
        ::tenex::evaluate<ti::I>(gamma_hat(ti::I) -
                                 contracted_conformal_christoffel(ti::I));
    const Scalar<DataVector> half_conformal_factor_squared =
        ::tenex::evaluate(0.5 * conformal_factor() * conformal_factor());
    const auto upper_spatial_z4 = ::Ccz4::upper_spatial_z4_constraint(
        half_conformal_factor_squared,
        gamma_hat_minus_contracted_conformal_christoffel);
    const auto constraint_char_fields = constraint_characteristic_fields(
        get<::Ccz4::Tags::Theta<DataVector>>(evolved_vars), upper_spatial_z4,
        spatial_metric, unit_normal_one_form);

    // Compute radiation characteristic fields
    const auto& conformal_factor_squared =
        ::tenex::evaluate(conformal_factor() * conformal_factor());
    const auto inverse_spatial_metric =
        determinant_and_inverse(spatial_metric).second;
    const auto& d_a_tilde =
        get<::Tags::deriv<::Ccz4::Tags::ATilde<DataVector, SpatialDim>,
                          tmpl::size_t<SpatialDim>, FrameType>>(deriv_vars);
    tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FrameType> field_p;
    ::tenex::evaluate<ti::i>(make_not_null(&field_p),
                             d_conformal_factor(ti::i) / conformal_factor());
    const auto christoffel = christoffel_second_kind(
        conformal_spatial_metric, inverse_conformal_spatial_metric, field_p,
        conformal_christoffel);
    const auto contracted_christoffel =
        ::tenex::evaluate<ti::l>(christoffel(ti::M, ti::l, ti::m));
    tnsr::iJkk<DataVector, ::Ccz4::fd::System::volume_dim>
        d_conformal_christoffel{};
    const auto d_d_conformal_metric = get<::Tags::second_deriv<
        ::Ccz4::Tags::ConformalMetric<
            DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>,
        tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame::Inertial>>(
        second_deriv_vars);
    tnsr::iijj<DataVector, ::Ccz4::fd::System::volume_dim> d_field_d{};
    ::tenex::evaluate<ti::i, ti::j, ti::k, ti::l>(
        make_not_null(&d_field_d),
        0.5 * d_d_conformal_metric(ti::i, ti::j, ti::k, ti::l));
    tnsr::iJJ<DataVector, ::Ccz4::fd::System::volume_dim> field_d_up{};
    ::tenex::evaluate<ti::k, ti::I, ti::J>(
        make_not_null(&field_d_up),
        (inverse_conformal_spatial_metric)(ti::I, ti::N) *
            (inverse_conformal_spatial_metric)(ti::M, ti::J) *
            field_d(ti::k, ti::n, ti::m));
    ::Ccz4::deriv_conformal_christoffel_second_kind(
        make_not_null(&d_conformal_christoffel),
        inverse_conformal_spatial_metric, field_d, d_field_d, field_d_up);
    const auto contracted_d_conformal_christoffel_difference =
        ::tenex::evaluate<ti::i, ti::j>(
            (d_conformal_christoffel)(ti::m, ti::M, ti::i, ti::j) -
            (d_conformal_christoffel)(ti::j, ti::M, ti::i, ti::m));
    const auto contracted_field_d_up =
        ::tenex::evaluate<ti::L>((field_d_up)(ti::m, ti::M, ti::L));
    const auto& d_d_conformal_factor =
        get<::Tags::second_deriv<Ccz4::Tags::ConformalFactor<DataVector>,
                                 tmpl::size_t<::Ccz4::fd::System::volume_dim>,
                                 Frame::Inertial>>(second_deriv_vars);
    tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim> d_field_p{};
    ::tenex::evaluate<ti::i, ti::j>(
        make_not_null(&d_field_p),
        (d_d_conformal_factor)(ti::i, ti::j) / conformal_factor() -
            (d_conformal_factor(ti::i) * d_conformal_factor(ti::j)) /
                (conformal_factor() * conformal_factor()));
    tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim> spatial_ricci_tensor{};
    ::Ccz4::spatial_ricci_tensor(
        make_not_null(&spatial_ricci_tensor), christoffel,
        contracted_christoffel, contracted_d_conformal_christoffel_difference,
        conformal_spatial_metric, inverse_conformal_spatial_metric, field_d,
        field_d_up, contracted_field_d_up, field_p, d_field_p);
    const auto radiation_char_fields = radiation_characteristic_fields(
        conformal_factor, conformal_factor_squared, conformal_spatial_metric,
        spatial_metric, inverse_spatial_metric, trace_extrinsic_curvature,
        a_tilde, d_conformal_factor, d_trace_extrinsic_curvature,
        d_conformal_spatial_metric, d_a_tilde, spatial_ricci_tensor,
        christoffel, unit_normal_one_form);

    // Modify incoming expected dt char fields at the external boundary
    auto& dt_u_vector1_zero =
        get<::Tags::dt<Tags::UVector1Zero<DataVector, SpatialDim, FrameType>>>(
            dt_char_fields);
    const auto& dt_u_vector3_plus =
        get<::Tags::dt<Tags::UVector3Plus<DataVector, SpatialDim, FrameType>>>(
            dt_char_fields);
    auto& dt_u_vector3_minus =
        get<::Tags::dt<Tags::UVector3Minus<DataVector, SpatialDim, FrameType>>>(
            dt_char_fields);
    auto& dt_u_scalar1_zero =
        get<::Tags::dt<Tags::UScalar1Zero<DataVector>>>(dt_char_fields);
    const auto& dt_u_scalar3_plus =
        get<::Tags::dt<Tags::UScalar3Plus<DataVector>>>(dt_char_fields);
    auto& dt_u_scalar3_minus =
        get<::Tags::dt<Tags::UScalar3Minus<DataVector>>>(dt_char_fields);
    const auto& dt_u_scalar4_plus =
        get<::Tags::dt<Tags::UScalar4Plus<DataVector>>>(dt_char_fields);
    auto& dt_u_scalar4_minus =
        get<::Tags::dt<Tags::UScalar4Minus<DataVector>>>(dt_char_fields);
    const auto& dt_u_scalar5_plus =
        get<::Tags::dt<Tags::UScalar5Plus<DataVector>>>(dt_char_fields);
    auto& dt_u_scalar5_minus =
        get<::Tags::dt<Tags::UScalar5Minus<DataVector>>>(dt_char_fields);

    const auto set_zero_at_boundary =
        [num_pts, num_face_pts]<typename TensorType>(TensorType& dt_tensor) {
          for (size_t tensor_index = 0; tensor_index < dt_tensor.size();
               ++tensor_index) {
            for (size_t i = 0; i < num_face_pts; ++i) {
              dt_tensor[tensor_index][num_pts - num_face_pts + i] = 0.0;
            }
          }
        };
    if (System::shifting_shift) {
      set_zero_at_boundary(dt_u_vector1_zero);
      set_zero_at_boundary(dt_u_scalar1_zero);
    }
    set_zero_at_boundary(dt_u_vector3_minus);
    set_zero_at_boundary(dt_u_scalar3_minus);
    set_zero_at_boundary(dt_u_scalar4_minus);
    set_zero_at_boundary(dt_u_scalar5_minus);

    const auto shift_n =
        ::tenex::evaluate(shift(ti::K) * unit_normal_one_form(ti::k));
    for (size_t i = 0; i < num_face_pts; ++i) {
      CHECK(get(shift_n)[num_pts - num_face_pts + i] > 0.0);
    }

    Scalar<DataVector> inertial_radial_coords;
    get(inertial_radial_coords) =
        sqrt(square(get<0>(x)) + square(get<1>(x)) + square(get<2>(x)));
    auto outermost_radial_spacing = make_with_value<Scalar<DataVector>>(
        DataVector{num_face_pts, std::numeric_limits<double>::signaling_NaN()},
        0.0);
    for (size_t i = 0; i < num_face_pts; ++i) {
      get(outermost_radial_spacing)[i] =
          get(inertial_radial_coords)[num_pts - num_face_pts + i] -
          get(inertial_radial_coords)[num_pts - 2 * num_face_pts + i];
    }
    auto penalty_strength = make_with_value<Scalar<DataVector>>(
        DataVector{num_face_pts, std::numeric_limits<double>::signaling_NaN()},
        0.0);
    for (size_t i = 0; i < num_face_pts; ++i) {
      get(penalty_strength)[i] = (get(lapse)[num_pts - num_face_pts + i] +
                                  get(shift_n)[num_pts - num_face_pts + i]) /
                                 get(outermost_radial_spacing)[i];
    }
    const auto& dt_u_vector2_plus =
        get<::Tags::dt<Tags::UVector2Plus<DataVector, SpatialDim, FrameType>>>(
            dt_char_fields);
    auto& dt_u_vector2_minus =
        get<::Tags::dt<Tags::UVector2Minus<DataVector, SpatialDim, FrameType>>>(
            dt_char_fields);
    const auto& c_vector_zero =
        get<Tags::CVectorZero<DataVector, SpatialDim, FrameType>>(
            constraint_char_fields);
    for (size_t tensor_index = 0; tensor_index < dt_u_vector2_minus.size();
         ++tensor_index) {
      for (size_t i = 0; i < num_face_pts; ++i) {
        dt_u_vector2_minus[tensor_index][num_pts - num_face_pts + i] =
            -4.0 * get(penalty_strength)[i] /
            square(get(conformal_factor)[num_pts - num_face_pts + i]) *
            c_vector_zero[tensor_index][num_pts - num_face_pts + i];
      }
    }
    const auto& dt_u_scalar2_plus =
        get<::Tags::dt<Tags::UScalar2Plus<DataVector>>>(dt_char_fields);
    auto& dt_u_scalar2_minus =
        get<::Tags::dt<Tags::UScalar2Minus<DataVector>>>(dt_char_fields);
    const auto& c_scalar_minus =
        get<Tags::CScalarMinus<DataVector>>(constraint_char_fields);
    for (size_t tensor_index = 0; tensor_index < dt_u_scalar2_minus.size();
         ++tensor_index) {
      for (size_t i = 0; i < num_face_pts; ++i) {
        dt_u_scalar2_minus[tensor_index][num_pts - num_face_pts + i] =
            -2.0 * get(penalty_strength)[i] *
            square(get(conformal_factor)[num_pts - num_face_pts + i]) *
            c_scalar_minus[tensor_index][num_pts - num_face_pts + i];
      }
    }
    const auto& dt_u_tensor_plus =
        get<::Tags::dt<Tags::UTensorPlus<DataVector, SpatialDim, FrameType>>>(
            dt_char_fields);
    auto& dt_u_tensor_minus =
        get<::Tags::dt<Tags::UTensorMinus<DataVector, SpatialDim, FrameType>>>(
            dt_char_fields);
    const auto& c_tensor_minus =
        get<Tags::CTensorMinus<DataVector, SpatialDim, FrameType>>(
            radiation_char_fields);
    for (size_t tensor_index = 0; tensor_index < dt_u_tensor_minus.size();
         ++tensor_index) {
      for (size_t i = 0; i < num_face_pts; ++i) {
        dt_u_tensor_minus[tensor_index][num_pts - num_face_pts + i] -=
            (get(lapse)[num_pts - num_face_pts + i] +
             get(shift_n)[num_pts - num_face_pts + i]) *
            square(get(conformal_factor)[num_pts - num_face_pts + i]) *
            c_tensor_minus[tensor_index][num_pts - num_face_pts + i];
      }
    }

    // Compute dt of evolved space
    const auto modified_dt_vars =
        dt_evolved_space_from_dt_characteristic_fields(
            dt_u_tensor_plus, dt_u_tensor_minus, dt_u_vector1_zero,
            dt_u_vector2_plus, dt_u_vector2_minus, dt_u_vector3_plus,
            dt_u_vector3_minus, dt_u_scalar1_zero, dt_u_scalar2_plus,
            dt_u_scalar2_minus, dt_u_scalar3_plus, dt_u_scalar3_minus,
            dt_u_scalar4_plus, dt_u_scalar4_minus, dt_u_scalar5_plus,
            dt_u_scalar5_minus, unit_normal_one_form, conformal_spatial_metric,
            conformal_factor, lapse, shift, f);

    CHECK_ITERABLE_CUSTOM_APPROX(
        (get<::Tags::dt<::Ccz4::Tags::ATilde<DataVector, SpatialDim>>>(box)),
        (get<::Tags::dt<::Ccz4::Tags::ATilde<DataVector, SpatialDim>>>(
            modified_dt_vars)),
        approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        (get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(box)),
        (get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
            modified_dt_vars)),
        approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        (get<::Tags::dt<::Ccz4::Tags::Theta<DataVector>>>(box)),
        (get<::Tags::dt<::Ccz4::Tags::Theta<DataVector>>>(modified_dt_vars)),
        approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        (get<::Tags::dt<::Ccz4::Tags::GammaHat<DataVector, SpatialDim>>>(box)),
        (get<::Tags::dt<::Ccz4::Tags::GammaHat<DataVector, SpatialDim>>>(
            modified_dt_vars)),
        approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        (get<::Tags::dt<::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>>(
            box)),
        (get<::Tags::dt<::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>>(
            modified_dt_vars)),
        approx);

    const double logical_cell_size = 2.0 / points_per_dimension;
    const auto n_dot_inv_jac = ::tenex::evaluate<ti::I>(
        unit_normal_vector(ti::J) *
        cell_centered_logical_to_inertial_inv_jacobian(ti::I, ti::j));
    Scalar<DataVector> jacobian_factor;
    get(jacobian_factor) = n_dot_inv_jac.get(2);

    const auto reconstruct_tensor_from_derivative =
        [num_pts, num_face_pts, logical_cell_size,
         &jacobian_factor]<typename TensorType>(const TensorType& dt_dn_tensor,
                                                TensorType& dt_tensor) {
          for (size_t tensor_index = 0; tensor_index < dt_tensor.size();
               ++tensor_index) {
            for (size_t i = 0; i < num_face_pts; ++i) {
              dt_tensor[tensor_index][num_pts - num_face_pts + i] =
                  (12.0 * logical_cell_size *
                       dt_dn_tensor[tensor_index][num_pts - num_face_pts + i] /
                       get(jacobian_factor)[num_pts - num_face_pts + i] +
                   48.0 *
                       dt_tensor[tensor_index][num_pts - 2 * num_face_pts + i] -
                   36.0 *
                       dt_tensor[tensor_index][num_pts - 3 * num_face_pts + i] +
                   16.0 *
                       dt_tensor[tensor_index][num_pts - 4 * num_face_pts + i] -
                   3.0 * dt_tensor[tensor_index]
                                  [num_pts - 5 * num_face_pts + i]) /
                  25.0;
            }
          }
        };

    reconstruct_tensor_from_derivative(
        get<::Tags::dt<
            Tags::DnConformalMetric<DataVector, SpatialDim, FrameType>>>(
            modified_dt_vars),
        dt_conformal_spatial_metric);
    reconstruct_tensor_from_derivative(
        get<::Tags::dt<Tags::DnConformalFactor<DataVector>>>(modified_dt_vars),
        dt_conformal_factor);
    reconstruct_tensor_from_derivative(
        get<::Tags::dt<Tags::DnLapse<DataVector>>>(modified_dt_vars), dt_lapse);
    reconstruct_tensor_from_derivative(
        get<::Tags::dt<Tags::DnShift<DataVector, SpatialDim, FrameType>>>(
            modified_dt_vars),
        dt_shift);
    CHECK_ITERABLE_CUSTOM_APPROX(
        (get<::Tags::dt<::Ccz4::Tags::ConformalMetric<DataVector, SpatialDim>>>(
            box)),
        dt_conformal_spatial_metric, approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        (get<::Tags::dt<::Ccz4::Tags::ConformalFactor<DataVector>>>(box)),
        dt_conformal_factor, approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        (get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(box)), dt_lapse, approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        (get<::Tags::dt<gr::Tags::Shift<DataVector, SpatialDim>>>(box)),
        dt_shift, approx);
  }
}

void test_sommerfeld(
    const bool evolve_lapse_and_shift,
    std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>
        boundary_condition) {
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 5;
  const Ccz4::fd::DummyReconstructor recons{};
  const size_t ghost_zone_size = recons.ghost_zone_size();
  const Mesh<SpatialDim> subcell_mesh{points_per_dimension,
                                      Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered};

  const std::array<double, SpatialDim> lower_bound{0.8, 0.7, -2.1};
  const std::array<double, SpatialDim> upper_bound{1.0, 1.5, -0.9};
  const std::array<double, SpatialDim> coords_range = upper_bound - lower_bound;

  // Create an element with an external boundary by omitting the last neighbor
  // in the upper_zeta direction.
  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element(true);

  // we cannot declare element_map const because it is not copyable into the box
  // NOLINTNEXTLINE(misc-const-correctness)
  ElementMap element_map{
      element.id(),
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(
          Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                   Affine{-1., 1., lower_bound[1], upper_bound[1]},
                   Affine{-1., 1., lower_bound[2], upper_bound[2]}})
          .get_clone()};

  const auto grid_to_inertial_map =
      domain::make_coordinate_map<Frame::Grid, FrameType>(
          domain::CoordinateMaps::Identity<3>{});

  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = grid_to_inertial_map(element_map(logical_coords));

  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                  Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < SpatialDim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  InverseHessian<DataVector, SpatialDim, Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_hessian{
          subcell_mesh.number_of_grid_points(), 0.0};

  // Ghost data from interior neighbors (none for the external face, which is
  // set by BCs)
  const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::Minkowski::
                  compute_prim_solution_for_Minkowski<false>,
              coords_range);

  // Minkowski evolved variables
  auto evolved_vars = TestHelpers::Ccz4::fd::detail::Minkowski::
      compute_prim_solution_for_Minkowski(x);

  const DataVector radial_coords =
      sqrt(square(get<0>(x)) + square(get<1>(x)) + square(get<2>(x)));

  const DataVector used_for_size(subcell_mesh.number_of_grid_points(),
                                 std::numeric_limits<double>::signaling_NaN());
  // set dummy k_0 value to get non-trivial lapse evolution for testing
  const auto k_0 = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, 3>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, Ccz4::fd::System::variables_tag>;

  // Provide BC on the external face (upper_zeta)
  std::vector<DirectionMap<
      SpatialDim,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      external_bcs_per_block(1);
  external_bcs_per_block[0][Direction<SpatialDim>::upper_zeta()] =
      std::move(boundary_condition);

  // NOLINTNEXTLINE(misc-const-correctness)
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  auto box = db::create<db::AddSimpleTags<
      ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
      ::Ccz4::fd::Tags::EvolveLapseAndShift, domain::Tags::Element<SpatialDim>,
      fd::Tags::Reconstructor,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<true>>,
      Ccz4::fd::System::variables_tag, ::Ccz4::Tags::Eta<DataVector>,
      ::Ccz4::Tags::K0<DataVector>,
      ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>,
      ::Ccz4::fd::Tags::ObserverCharacteristicFieldsTag<SpatialDim,
                                                         Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverConstraintCharacteristicFieldsTag<
          SpatialDim, Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverRadiationCharacteristicFieldsTag<
          SpatialDim, Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::ObserverConstraintCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::ObserverRadiationCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::InitialBoundaryCharacteristicFields<SpatialDim,
                                                             Frame::Inertial>,
      dt_variables_tag,
      evolution::dg::subcell::Tags::Mesh<SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseHessianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
      domain::Tags::ExternalBoundaryConditions<SpatialDim>,
      evolution::dg::subcell::Tags::Coordinates<SpatialDim, Frame::Inertial>,
      ::Tags::Time,
      domain::Tags::FunctionsOfTime,
      domain::Tags::ElementMap<SpatialDim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<SpatialDim, Frame::Grid,
                                                  Frame::Inertial>>>(
      kappa_1, kappa_2, kappa_3, evolve_lapse_and_shift, element,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      DummyEvolutionMetaVars<true>{}, evolved_vars, eta, k_0,
      upper_spatial_z4_constraint,
      Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::constraint_characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::radiation_characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::constraint_characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::radiation_characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      std::optional<Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>>{std::nullopt},
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
      cell_centered_logical_to_inertial_inv_hessian,
      all_ghost_data, std::move(external_bcs_per_block), x, 0.0,
      std::move(functions_of_time), std::move(element_map),
      grid_to_inertial_map.get_clone());

  ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));

  const size_t num_pts = subcell_mesh.number_of_grid_points();
  const size_t num_face_pts = subcell_mesh.extents(0) * subcell_mesh.extents(1);

  // Compute r on the upper zeta slice
  const size_t num_unaffected_pts = num_pts - num_face_pts;
  DataVector r_affected{};
  make_const_view<DataVector>(make_not_null(&r_affected), radial_coords,
                              num_unaffected_pts, num_face_pts);

  {
    // Points not in upper zeta slice should be unaffected (zero dt for
    // Minkowski)
    const auto unaffected_expected = DataVector(num_pts - num_face_pts, 0.0);
    const auto& dt_lapse = get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(box);

    DataVector unaffected_actual{};
    make_const_view<DataVector>(make_not_null(&unaffected_actual),
                                get(dt_lapse), 0, num_unaffected_pts);
    CHECK_ITERABLE_APPROX(unaffected_actual, unaffected_expected);

    DataVector affected_actual{};
    make_const_view<DataVector>(make_not_null(&affected_actual), get(dt_lapse),
                                num_unaffected_pts, num_face_pts);
    // Points in upper zeta slice has expected dt = -1/r for lapse in Minkowski
    const DataVector affected_expected = -1.0 / r_affected;
    CHECK_ITERABLE_APPROX(affected_actual, affected_expected);
  }

  {
    // Points not in upper zeta slice should be unaffected (zero dt for
    // Minkowski)
    const auto unaffected_expected = DataVector(num_pts - num_face_pts, 0.0);
    const auto& dt_conformal_factor =
        get<::Tags::dt<::Ccz4::Tags::ConformalFactor<DataVector>>>(box);

    DataVector unaffected_actual{};
    make_const_view<DataVector>(make_not_null(&unaffected_actual),
                                get(dt_conformal_factor), 0,
                                num_unaffected_pts);
    CHECK_ITERABLE_APPROX(unaffected_actual, unaffected_expected);

    DataVector affected_actual{};
    make_const_view<DataVector>(make_not_null(&affected_actual),
                                get(dt_conformal_factor), num_unaffected_pts,
                                num_face_pts);
    // Points in upper zeta slice has expected dt = -1/r for conformal factor in
    // Minkowski
    const DataVector affected_expected = -1.0 / r_affected;
    CHECK_ITERABLE_APPROX(affected_actual, affected_expected);
  }

  {
    // Points not in upper zeta slice should be unaffected (zero dt for
    // Minkowski)
    const auto unaffected_expected = DataVector(num_pts - num_face_pts, 0.0);
    const auto& dt_conformal_metric =
        get<::Tags::dt<::Ccz4::Tags::ConformalMetric<DataVector, SpatialDim>>>(
            box);
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        DataVector unaffected_actual{};
        make_const_view<DataVector>(make_not_null(&unaffected_actual),
                                    dt_conformal_metric.get(i, j), 0,
                                    num_unaffected_pts);
        CHECK_ITERABLE_APPROX(unaffected_actual, unaffected_expected);

        DataVector affected_actual{};
        make_const_view<DataVector>(make_not_null(&affected_actual),
                                    dt_conformal_metric.get(i, j),
                                    num_unaffected_pts, num_face_pts);
        // Points in upper zeta slice has expected dt = -1/r for diagonal in
        // Minkowski
        const DataVector affected_expected =
            (i == j) ? (-1.0 / r_affected) : DataVector(num_face_pts, 0.0);
        CHECK_ITERABLE_APPROX(affected_actual, affected_expected);
      }
    }
  }

  // all other evolved variables must have zero dt everywhere
  {
    const Approx custom_approx = Approx::custom().epsilon(1.0e-13).scale(1.0);
    const DataVector all_expected{num_pts, 0.0};
    tmpl::for_each<Ccz4::fd::System::variables_tag_list>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          // Skip variables already checked above: lapse, conformal factor,
          // conformal metric
          if constexpr (std::is_same_v<Tag, gr::Tags::Lapse<DataVector>> ||
                        std::is_same_v<
                            Tag, ::Ccz4::Tags::ConformalFactor<DataVector>> ||
                        std::is_same_v<Tag, ::Ccz4::Tags::ConformalMetric<
                                                DataVector, SpatialDim>>) {
            return;
          }
          const std::string tag_name = db::tag_name<::Tags::dt<Tag>>();
          CAPTURE(tag_name);
          for (auto& component : get<::Tags::dt<Tag>>(box)) {
            // dt_theta and dt_trace_extrinsic_curvature are slightly non-zero
            // which is likely due to round-off error.
            CHECK_ITERABLE_CUSTOM_APPROX(component, all_expected,
                                         custom_approx);
          }
        });
  }

  // now change the lapse to some dummy values to test the radial deriv term in
  // Sommerfeld BC
  db::mutate<
      Ccz4::fd::System::variables_tag, ::Ccz4::Tags::K0<DataVector>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>>(
      [&](const auto evolved_var_ptr, const auto k_0_ptr,
          const auto all_ghost_data_ptr) {
        get(get<gr::Tags::Lapse<DataVector>>(*evolved_var_ptr)) = get<0>(x);
        get(*k_0_ptr) = DataVector(num_pts, 1.0);
        *all_ghost_data_ptr =
            TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
                subcell_mesh, x, element.neighbors(), ghost_zone_size,
                TestHelpers::Ccz4::fd::detail::Minkowski::
                    compute_prim_solution_for_Minkowski<true>,
                coords_range);
      },
      make_not_null(&box));

  ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));

  {
    // Points not in upper zeta slice should be unaffected (value dependent on
    // evolve_lapse_and_shift)
    DataVector dt_lapse_expected = -2.0 * get<0>(x) / radial_coords;
    for (size_t i = 0; i < num_unaffected_pts; ++i) {
      dt_lapse_expected[i] = evolve_lapse_and_shift ? 2.0 * get<0>(x)[i] : 0.0;
    }
    DataVector dt_lapse_unaffected_expected{};
    make_const_view<DataVector>(make_not_null(&dt_lapse_unaffected_expected),
                                dt_lapse_expected, 0, num_unaffected_pts);

    const auto& dt_lapse = get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(box);

    DataVector unaffected_actual{};
    make_const_view<DataVector>(make_not_null(&unaffected_actual),
                                get(dt_lapse), 0, num_unaffected_pts);
    CHECK_ITERABLE_APPROX(unaffected_actual, dt_lapse_unaffected_expected);

    DataVector affected_actual{};
    make_const_view<DataVector>(make_not_null(&affected_actual), get(dt_lapse),
                                num_unaffected_pts, num_face_pts);
    // Points in upper zeta slice has expected dt = -2*x/r for dummy lapse in
    // Minkowski
    DataVector dt_lapse_affected_expected{};
    make_const_view<DataVector>(make_not_null(&dt_lapse_affected_expected),
                                dt_lapse_expected, num_unaffected_pts,
                                num_face_pts);
    CHECK_ITERABLE_APPROX(affected_actual, dt_lapse_affected_expected);
  }
}

void test_dirichlet_analytic_bc(const bool evolve_lapse_and_shift) {
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 5;
  const Ccz4::fd::DummyReconstructor recons{};
  const size_t ghost_zone_size = recons.ghost_zone_size();
  const Mesh<SpatialDim> subcell_mesh{points_per_dimension,
                                      Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered};

  const std::array<double, SpatialDim> lower_bound{0.8, 0.7, -2.1};
  const std::array<double, SpatialDim> upper_bound{1.0, 1.5, -0.9};
  const std::array<double, SpatialDim> coords_range = upper_bound - lower_bound;

  // Create an element with an external boundary by omitting the last neighbor
  // in the upper_zeta direction.
  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element(true);

  // we cannot declare element_map const because it is not copyable into the box
  // NOLINTNEXTLINE(misc-const-correctness)
  ElementMap element_map{
      element.id(),
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(
          Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                   Affine{-1., 1., lower_bound[1], upper_bound[1]},
                   Affine{-1., 1., lower_bound[2], upper_bound[2]}})
          .get_clone()};

  const auto grid_to_inertial_map =
      domain::make_coordinate_map<Frame::Grid, FrameType>(
          domain::CoordinateMaps::Identity<3>{});

  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = grid_to_inertial_map(element_map(logical_coords));

  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                  Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < SpatialDim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  InverseHessian<DataVector, SpatialDim, Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_hessian{
          subcell_mesh.number_of_grid_points(), 0.0};

  // Ghost data from interior neighbors (none for the external face, which is
  // set by BCs)
  const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::Minkowski::
                  compute_prim_solution_for_Minkowski<false>,
              coords_range);

  // Minkowski evolved variables
  auto evolved_vars = TestHelpers::Ccz4::fd::detail::Minkowski::
      compute_prim_solution_for_Minkowski(x);

  const DataVector used_for_size(subcell_mesh.number_of_grid_points(),
                                 std::numeric_limits<double>::signaling_NaN());
  // set dummy k_0 value to get non-trivial lapse evolution for testing
  const auto k_0 = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, 3>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, Ccz4::fd::System::variables_tag>;

  // Provide BC on the external face (upper_zeta)
  std::vector<DirectionMap<
      SpatialDim,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      external_bcs_per_block(1);
  external_bcs_per_block[0][Direction<SpatialDim>::upper_zeta()] =
      std::make_unique<Ccz4::BoundaryConditions::DirichletAnalytic>(
          std::make_unique<
              Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::Minkowski<3>>>(
              Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::Minkowski<3>>{}));

  // NOLINTNEXTLINE(misc-const-correctness)
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  auto box = db::create<db::AddSimpleTags<
      ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
      ::Ccz4::fd::Tags::EvolveLapseAndShift, domain::Tags::Element<SpatialDim>,
      fd::Tags::Reconstructor,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<true>>,
      Ccz4::fd::System::variables_tag, ::Ccz4::Tags::Eta<DataVector>,
      ::Ccz4::Tags::K0<DataVector>,
      ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>,
      ::Ccz4::fd::Tags::ObserverCharacteristicFieldsTag<SpatialDim,
                                                         Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverConstraintCharacteristicFieldsTag<
          SpatialDim, Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverRadiationCharacteristicFieldsTag<
          SpatialDim, Frame::Inertial>,
      ::Ccz4::fd::Tags::ObserverCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::ObserverConstraintCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::ObserverRadiationCharacteristicSpeedsTag,
      ::Ccz4::fd::Tags::InitialBoundaryCharacteristicFields<SpatialDim,
                                                             Frame::Inertial>,
      dt_variables_tag,
      evolution::dg::subcell::Tags::Mesh<SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseHessianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
      domain::Tags::ExternalBoundaryConditions<SpatialDim>,
      evolution::dg::subcell::Tags::Coordinates<SpatialDim, Frame::Inertial>,
      ::Tags::Time,
      domain::Tags::FunctionsOfTime,
      domain::Tags::ElementMap<SpatialDim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<SpatialDim, Frame::Grid,
                                                  Frame::Inertial>>>(
      kappa_1, kappa_2, kappa_3, evolve_lapse_and_shift, element,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      DummyEvolutionMetaVars<true>{}, evolved_vars, eta, k_0,
      upper_spatial_z4_constraint,
      Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::constraint_characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::radiation_characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::constraint_characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      Variables<::Ccz4::fd::Tags::radiation_characteristic_speeds_tags_list>{
          subcell_mesh.number_of_grid_points(), 0.0},
      std::optional<Variables<::Ccz4::fd::Tags::characteristic_fields_tags_list<
          DataVector, SpatialDim, Frame::Inertial>>>{std::nullopt},
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
      cell_centered_logical_to_inertial_inv_hessian,
      all_ghost_data, std::move(external_bcs_per_block), x, 0.0,
      std::move(functions_of_time), std::move(element_map),
      grid_to_inertial_map.get_clone());

  ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));

  {
    const size_t num_pts = subcell_mesh.number_of_grid_points();
    const Approx custom_approx = Approx::custom().epsilon(1.0e-13).scale(1.0);
    const DataVector all_expected{num_pts, 0.0};
    tmpl::for_each<Ccz4::fd::System::variables_tag_list>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          const std::string tag_name = db::tag_name<::Tags::dt<Tag>>();
          CAPTURE(tag_name);
          for (auto& component : get<::Tags::dt<Tag>>(box)) {
            // dt_theta and dt_trace_extrinsic_curvature are slightly non-zero
            // which is likely due to round-off error.
            CHECK_ITERABLE_CUSTOM_APPROX(component, all_expected,
                                         custom_approx);
          }
        });
  }
}

void test() {
  test_minkowski(true);
  test_kerrschild(true);
  test_minkowski(false);
  test_kerrschild(false);

  const std::array<double, 3> k{{0.5, 0.1, -0.2}};
  test_gauge_plane_wave(
      k,
      std::make_unique<MathFunctions::Sinusoid<1, Frame::Inertial>>(0.6, 0.8,
                                                                    2.0),
      0.4, true);
  test_gauge_plane_wave(
      k,
      std::make_unique<MathFunctions::Sinusoid<1, Frame::Inertial>>(0.6, 0.8,
                                                                    2.0),
      0.4, false);

  // Run Diridchlet BC test
  test_dirichlet_analytic_bc(true);
  test_dirichlet_analytic_bc(false);
  // Run Sommerfeld BC test
  test_sommerfeld(true,
                  std::make_unique<Ccz4::BoundaryConditions::Sommerfeld>(2));
  test_sommerfeld(false,
                  std::make_unique<Ccz4::BoundaryConditions::Sommerfeld>(2));
  // Run constraint radiation preserving BC test
  test_constraint_radiation_preserving_bc(
      true, std::make_unique<
                Ccz4::BoundaryConditions::ConstraintsRadiationPreserving>(4));
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      test_constraint_radiation_preserving_bc(
          false,
          std::make_unique<
              Ccz4::BoundaryConditions::ConstraintsRadiationPreserving>(4)),
      Catch::Matchers::ContainsSubstring(
          "ConstraintsRadiationPreserving BC is not implemented"));
#endif
}
}  // namespace

// The tests run relatively long as we use much higher spatial
// resolution (~8000 grid points per element) to reach a relative
// error of 1e-9.
// [[TimeOut, 40]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.FiniteDifference.TimeDerivative",
                  "[Unit][Evolution]") {
  test();
}
}  // namespace Ccz4::fd
