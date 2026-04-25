// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Factory.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UnlimitedDeg4Prim.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UpdateAuxiliaryVariablesFd.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

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

struct DummyParallelComponent {};

using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

// Test UpdateAuxiliaryVariablesFd with Minkowski spacetime.
// In Minkowski: lapse=1, shift=0, conformal_factor=1, conformal_metric=delta,
// so all spatial derivatives are zero and FieldA/B/D/P should all be zero.
void test_minkowski() {
  constexpr size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 5;
  const Ccz4::fd::UnlimitedDeg4Prim recons{};
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

  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element();

  const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::Minkowski::
                  compute_prim_solution_for_Minkowski<false>,
              coords_range);

  const auto evolved_vars = TestHelpers::Ccz4::fd::detail::Minkowski::
      compute_prim_solution_for_Minkowski(x);

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Element<SpatialDim>, fd::Tags::Reconstructor,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<false>>,
      Ccz4::fd::System::variables_tag,
      evolution::dg::subcell::Tags::Mesh<SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
      domain::Tags::ExternalBoundaryConditions<SpatialDim>>>(
      element,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      DummyEvolutionMetaVars<false>{}, evolved_vars, subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian, all_ghost_data,
      std::vector<DirectionMap<
          SpatialDim,
          std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>{});

  // Call UpdateAuxiliaryVariablesFd
  tuples::TaggedTuple<> empty_inboxes{};
  Parallel::GlobalCache<DummyEvolutionMetaVars<false>>* null_cache = nullptr;
  const int array_index = 0;
  const DummyParallelComponent* null_component = nullptr;
  const auto result = UpdateAuxiliaryVariablesFd::apply(
      box, empty_inboxes, *null_cache, array_index, tmpl::list<>{},
      null_component);
  CHECK(std::get<0>(result) == Parallel::AlgorithmExecution::Continue);

  // In Minkowski, all derivatives are zero, so FieldA/B/D/P should be zero
  const DataVector zero(subcell_mesh.number_of_grid_points(), 0.0);

  const auto& field_a =
      get<::Ccz4::Tags::FieldA<DataVector, 3>>(box);
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_APPROX(field_a.get(i), zero);
  }

  const auto& field_b =
      get<::Ccz4::Tags::FieldB<DataVector, 3>>(box);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      CHECK_ITERABLE_APPROX(field_b.get(i, j), zero);
    }
  }

  const auto& field_d =
      get<::Ccz4::Tags::FieldD<DataVector, 3>>(box);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = j; k < 3; ++k) {
        CHECK_ITERABLE_APPROX(field_d.get(i, j, k), zero);
      }
    }
  }

  const auto& field_p =
      get<::Ccz4::Tags::FieldP<DataVector, 3>>(box);
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_APPROX(field_p.get(i), zero);
  }
}

// Test UpdateAuxiliaryVariablesFd with KerrSchild spacetime.
// Non-trivial metric — FieldA/B/D/P should be non-zero and match
// the values computed by SoTimeDerivative (which uses the same FD
// derivative + tenex formulas).
void test_kerrschild() {
  constexpr size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 20;
  const Ccz4::fd::UnlimitedDeg4Prim recons{};
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

  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element();

  const double mass = 2.0;
  const std::array<double, SpatialDim> spin{{0.2, 0.4, 0.8}};
  const std::array<double, SpatialDim> center{{0.2, 0.5, 0.1}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);
  const double t = std::numeric_limits<double>::signaling_NaN();
  const double f = Ccz4::fd::System::f;
  constexpr bool evolve_shift = false;

  const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::KerrSchild::
                  compute_prim_solution_for_KerrSchild,
              coords_range, t, f, evolve_shift, solution);

  const auto evolved_vars = TestHelpers::Ccz4::fd::detail::KerrSchild::
      compute_prim_solution_for_KerrSchild(x, t, f, evolve_shift, solution);

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Element<SpatialDim>, fd::Tags::Reconstructor,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<false>>,
      Ccz4::fd::System::variables_tag,
      evolution::dg::subcell::Tags::Mesh<SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
      domain::Tags::ExternalBoundaryConditions<SpatialDim>>>(
      element,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      DummyEvolutionMetaVars<false>{}, evolved_vars, subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian, all_ghost_data,
      std::vector<DirectionMap<
          SpatialDim,
          std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>{});

  // Call UpdateAuxiliaryVariablesFd
  tuples::TaggedTuple<> empty_inboxes{};
  Parallel::GlobalCache<DummyEvolutionMetaVars<false>>* null_cache = nullptr;
  const int array_index = 0;
  const DummyParallelComponent* null_component = nullptr;
  UpdateAuxiliaryVariablesFd::apply(box, empty_inboxes, *null_cache,
                                    array_index, tmpl::list<>{},
                                    null_component);

  // Compute expected FieldA/B/D/P using spectral derivatives
  // (the DG UpdateAuxiliaryVariables mutator uses spectral derivatives,
  // but for FD we compute from the same first derivatives)
  // Instead, we verify that the auxiliary fields match what
  // SoTimeDerivative would compute: recompute FD derivatives and
  // apply the same formulas manually.
  const size_t num_pts = subcell_mesh.number_of_grid_points();
  using gradients_tags = typename System::gradients_tags;
  using deriv_var_tag = db::wrap_tags_in<::Tags::deriv, gradients_tags,
                                         tmpl::size_t<3>, Frame::Inertial>;
  Variables<deriv_var_tag> cell_centered_derivs{num_pts};
  Ccz4::fd::spacetime_derivatives(
      make_not_null(&cell_centered_derivs), evolved_vars, all_ghost_data, 4,
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);

  const auto& d_lapse =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>>(cell_centered_derivs);
  const auto& d_shift =
      get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(cell_centered_derivs);
  const auto& d_conformal_metric =
      get<::Tags::deriv<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                        tmpl::size_t<3>, Frame::Inertial>>(
          cell_centered_derivs);
  const auto& d_conformal_factor =
      get<::Tags::deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>(
          cell_centered_derivs);

  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(evolved_vars);
  auto expected_field_a =
      ::tenex::evaluate<ti::i>(d_lapse(ti::i) / lapse());
  const auto& expected_field_b = d_shift;
  tnsr::ijj<DataVector, 3> expected_field_d;
  ::tenex::evaluate<ti::i, ti::j, ti::k>(
      make_not_null(&expected_field_d),
      0.5 * d_conformal_metric(ti::i, ti::j, ti::k));
  const auto& conformal_factor =
      get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars);
  auto expected_field_p = ::tenex::evaluate<ti::i>(
      d_conformal_factor(ti::i) / conformal_factor());

  // FieldA/B/D/P should be non-trivial for KerrSchild
  bool any_nonzero = false;
  for (size_t i = 0; i < 3; ++i) {
    if (max(abs(expected_field_a.get(i))) > 1.0e-10) {
      any_nonzero = true;
    }
  }
  REQUIRE(any_nonzero);

  const auto& actual_field_a =
      get<::Ccz4::Tags::FieldA<DataVector, 3>>(box);
  const auto& actual_field_b =
      get<::Ccz4::Tags::FieldB<DataVector, 3>>(box);
  const auto& actual_field_d =
      get<::Ccz4::Tags::FieldD<DataVector, 3>>(box);
  const auto& actual_field_p =
      get<::Ccz4::Tags::FieldP<DataVector, 3>>(box);
  CHECK_ITERABLE_APPROX(actual_field_a, expected_field_a);
  CHECK_ITERABLE_APPROX(actual_field_b, expected_field_b);
  CHECK_ITERABLE_APPROX(actual_field_d, expected_field_d);
  CHECK_ITERABLE_APPROX(actual_field_p, expected_field_p);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.UpdateAuxiliaryVariablesFd",
                  "[Unit][Evolution]") {
  test_minkowski();
  test_kerrschild();
}
}  // namespace
}  // namespace Ccz4::fd
