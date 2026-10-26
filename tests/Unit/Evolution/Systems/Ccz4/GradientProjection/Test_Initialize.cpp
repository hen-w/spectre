// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/Systems/Ccz4/Ccz4WrappedGr.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Initialize.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
using Solution = Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::KerrSchild>;
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <bool UseSubcell>
struct Metavariables {
  using component_list = tmpl::list<>;
  struct SubcellOptions {
    static constexpr bool subcell_enabled = UseSubcell;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<evolution::initial_data::InitialData, tmpl::list<Solution>>>;
  };
};
struct Component {};

Solution solution() {
  return Solution{
      gr::Solutions::KerrSchild{1.0, {{0.1, 0.2, 0.3}}, {{0.0, 0.0, 0.0}}}};
}

// SetInitialData sets the physical fields from the analytic data at the
// points of the active grid (with DG-subcell, the subcell grid of elements
// using finite differences) and zeroes the stored gradients.
template <bool UseSubcell>
void test(const evolution::dg::subcell::ActiveGrid active_grid) {
  CAPTURE(UseSubcell);
  CAPTURE(active_grid);
  const Mesh<3> dg_mesh{4, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const auto dg_logical = logical_coordinates(dg_mesh);
  const auto subcell_logical = logical_coordinates(subcell_mesh);
  const std::array<double, 3> lower{{3.0, 1.0, -1.0}};
  const std::array<double, 3> upper{{4.5, 2.0, 0.5}};
  const double time = 0.7;
  using FunctionsOfTimeMap = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;

  auto box = db::create<db::AddSimpleTags<
      ::Tags::Time, evolution::initial_data::Tags::InitialData,
      evolution::dg::subcell::Tags::ActiveGrid,
      evolution::dg::subcell::Tags::Coordinates<3, Frame::ElementLogical>,
      domain::Tags::Coordinates<3, Frame::ElementLogical>,
      domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                  Frame::Inertial>,
      domain::Tags::ElementMap<3, Frame::Grid>, domain::Tags::FunctionsOfTime,
      gp::System::variables_tag>>(
      time,
      std::unique_ptr<evolution::initial_data::InitialData>{
          std::make_unique<Solution>(solution())},
      active_grid, subcell_logical, dg_logical,
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{}),
      ElementMap<3, Frame::Grid>{
          ElementId<3>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              Affine3D{Affine{-1.0, 1.0, lower[0], upper[0]},
                       Affine{-1.0, 1.0, lower[1], upper[1]},
                       Affine{-1.0, 1.0, lower[2], upper[2]}})},
      FunctionsOfTimeMap{},
      Variables<gp::System::variables_tag_list>{1, 3.0e3});
  tuples::TaggedTuple<> inboxes{};
  const Parallel::GlobalCache<Metavariables<UseSubcell>> cache{};
  gp::Actions::SetInitialData::apply(box, inboxes, cache, 0, tmpl::list<>{},
                                     static_cast<const Component*>(nullptr));

  const bool on_subcells =
      UseSubcell and active_grid == evolution::dg::subcell::ActiveGrid::Subcell;
  const auto& logical = on_subcells ? subcell_logical : dg_logical;
  const size_t number_of_points = get<0>(logical).size();
  tnsr::I<DataVector, 3> x{number_of_points};
  for (size_t i = 0; i < 3; ++i) {
    x.get(i) = gsl::at(lower, i) + 0.5 *
                                       (gsl::at(upper, i) - gsl::at(lower, i)) *
                                       (logical.get(i) + 1.0);
  }
  const auto expected =
      solution().variables(x, time, gp::System::physical_tags{});
  const auto& vars = db::get<gp::System::variables_tag>(box);
  REQUIRE(vars.number_of_grid_points() == number_of_points);
  tmpl::for_each<gp::System::physical_tags>(
      [&vars, &expected]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        CAPTURE(db::tag_name<Tag>());
        CHECK_ITERABLE_APPROX(get<Tag>(vars), get<Tag>(expected));
      });
  for (const auto& component : get<gp::Tags::PotentialGradients>(vars)) {
    CHECK(component == DataVector(number_of_points, 0.0));
  }
}

// CheckFiniteDifferenceOptions requires ConstrainedEvolution and
// EvolveLapseAndShift on elements using finite differences, for all four
// option combinations, and ignores both options on elements using DG.
void test_check_finite_difference_options() {
  const ElementId<3> id{0};
  const Element<3> element{id, DirectionMap<3, Neighbors<3>>{}};
  for (const auto active_grid : {evolution::dg::subcell::ActiveGrid::Dg,
                                 evolution::dg::subcell::ActiveGrid::Subcell}) {
    for (const bool constrained_evolution : {true, false}) {
      for (const bool evolve_lapse_and_shift : {true, false}) {
        CAPTURE(active_grid);
        CAPTURE(constrained_evolution);
        CAPTURE(evolve_lapse_and_shift);
        auto box = db::create<db::AddSimpleTags<
            evolution::dg::subcell::Tags::ActiveGrid, domain::Tags::Element<3>,
            Ccz4::fd::Tags::ConstrainedEvolution,
            Ccz4::fd::Tags::EvolveLapseAndShift>>(active_grid, element,
                                                  constrained_evolution,
                                                  evolve_lapse_and_shift);
        tuples::TaggedTuple<> inboxes{};
        const Parallel::GlobalCache<Metavariables<true>> cache{};
        const auto apply = [&box, &inboxes, &cache]() {
          return gp::Actions::CheckFiniteDifferenceOptions::apply(
              box, inboxes, cache, 0, tmpl::list<>{},
              static_cast<const Component*>(nullptr));
        };
        if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
          CHECK(std::get<0>(apply()) == Parallel::AlgorithmExecution::Continue);
        } else if (not constrained_evolution) {
          CHECK_THROWS_WITH(apply(), Catch::Matchers::ContainsSubstring(
                                         "require ConstrainedEvolution: true"));
        } else if (not evolve_lapse_and_shift) {
          CHECK_THROWS_WITH(apply(), Catch::Matchers::ContainsSubstring(
                                         "require EvolveLapseAndShift: true"));
        } else {
          CHECK(std::get<0>(apply()) == Parallel::AlgorithmExecution::Continue);
        }
      }
    }
  }
}

// InitializeLocalOperators passes the number of removed angular bands to the
// operators; on a spherical-harmonic shell it must not exceed L.
void test_initialize_local_operators() {
  const auto identity_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});
  const auto make_inverse_jacobian = [](const size_t size) {
    InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
        result{size, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      result.get(i, i) = 1.0;
    }
    return result;
  };
  const Mesh<3> cube{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  gp::LocalOperators operators{};
  gp::InitializeLocalOperators::apply(
      make_not_null(&operators), cube,
      make_inverse_jacobian(cube.number_of_grid_points()),
      Scalar<DataVector>{cube.number_of_grid_points(), 1.0}, *identity_map, 6);
  CHECK(operators.shell_angular_modes_to_kill() == 6);
  CHECK(not operators.is_spherical_shell());
  const Mesh<3> shell{
      {{3, 4, 7}},
      {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
        Spectral::Quadrature::Equiangular}}};
  CHECK_THROWS_WITH(
      gp::InitializeLocalOperators::apply(
          make_not_null(&operators), shell,
          make_inverse_jacobian(shell.number_of_grid_points()),
          Scalar<DataVector>{shell.number_of_grid_points(), 1.0}, *identity_map,
          4),
      Catch::Matchers::ContainsSubstring(
          "ShellAngularModesToKill = 4 of the shell velocity filter exceeds "
          "L = 3"));
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.GradientProjection.Initialize",
                  "[Unit][Evolution]") {
  test_initialize_local_operators();
  test<true>(evolution::dg::subcell::ActiveGrid::Dg);
  test<true>(evolution::dg::subcell::ActiveGrid::Subcell);
  // Without DG-subcell the active grid is ignored.
  test<false>(evolution::dg::subcell::ActiveGrid::Subcell);
  test_check_finite_difference_options();
}
}  // namespace
