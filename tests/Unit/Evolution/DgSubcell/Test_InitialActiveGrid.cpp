// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/InitialActiveGrid.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace subcell = evolution::dg::subcell;
using subcell::ActiveGrid;

// A spherical shell (radially two elements) whose outer face borders the
// lower xi face of a cube nonconformingly, followed along xi by two more
// cubes; the last one is rotated by 90 degrees about xi. Each cube has two
// elements along xi and is refined transversally, the first one more than the
// others, so that the outer shell element has 64 neighbors in one direction
// and the elements of the second cube have four (conforming) neighbors in the
// first cube. The cubes have external boundaries in their transverse
// directions, the shell at its inner face and the last cube at its upper xi
// face.
const std::vector<std::string> names{"Shell", "Cube1", "Cube2", "Cube3"};
const std::vector<std::array<size_t, 3>> refinement_levels{
    {{1, 0, 0}}, {{1, 3, 3}}, {{1, 2, 2}}, {{1, 2, 2}}};

std::vector<Block<3>> blocks() {
  const auto aligned = OrientationMap<3>::create_aligned();
  const OrientationMap<3> rotated{std::array{Direction<3>::upper_xi(),
                                             Direction<3>::upper_zeta(),
                                             Direction<3>::lower_eta()}};
  std::vector<Block<3>> result{};
  result.emplace_back(
      nullptr, 0,
      DirectionMap<3, BlockNeighbors<3>>{
          {Direction<3>::upper_xi(),
           BlockNeighbors<3>{{1}, {{1, aligned}}, false}}},
      names[0],
      std::array{domain::Topology::I1, domain::Topology::S2Colatitude,
                 domain::Topology::S2Longitude});
  result.emplace_back(
      nullptr, 1,
      DirectionMap<3, BlockNeighbors<3>>{
          {Direction<3>::lower_xi(),
           BlockNeighbors<3>{{0}, {{0, aligned}}, false}},
          {Direction<3>::upper_xi(), BlockNeighbors<3>{2, aligned}}},
      names[1]);
  result.emplace_back(
      nullptr, 2,
      DirectionMap<3, BlockNeighbors<3>>{
          {Direction<3>::lower_xi(), BlockNeighbors<3>{1, aligned}},
          {Direction<3>::upper_xi(), BlockNeighbors<3>{3, rotated}}},
      names[2]);
  result.emplace_back(nullptr, 3,
                      DirectionMap<3, BlockNeighbors<3>>{
                          {Direction<3>::lower_xi(),
                           BlockNeighbors<3>{2, rotated.inverse_map()}}},
                      names[3]);
  return result;
}

// Only converts the names of the DG-only blocks into block ids.
class BlockNamesCreator : public DomainCreator<3> {
 public:
  Domain<3> create_domain() const override { return Domain<3>{}; }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override {
    return {};
  }
  std::vector<std::string> block_names() const override { return names; }
  std::vector<std::array<size_t, 3>> initial_extents() const override {
    return {};
  }
  std::vector<std::array<size_t, 3>> initial_refinement_levels()
      const override {
    return refinement_levels;
  }
};

subcell::SubcellOptions subcell_options(const bool shell_is_dg_only) {
  std::vector<std::string> only_dg_blocks{"Cube3"};
  if (shell_is_dg_only) {
    only_dg_blocks.emplace_back("Shell");
  }
  return subcell::SubcellOptions{
      subcell::SubcellOptions{4.0, 1_st, 1.0e-3, 1.0e-4, false, false,
                              subcell::fd::ReconstructionMethod::DimByDim,
                              false, only_dg_blocks, ::fd::DerivativeOrder::Two,
                              1, 1, 1},
      BlockNamesCreator{}};
}

bool at_block_boundary(const SegmentId& segment) {
  return segment.index() == 0 or
         segment.index() + 1 == two_to_the(segment.refinement_level());
}

// The rule of SetSubcellGrid applied by hand to the layout above: DG in the
// DG-only blocks (Cube3, and the shell if `shell_is_dg_only`) and in the
// elements bordering them (the first xi layer of Cube1 if the shell is
// DG-only, the last xi layer of Cube2), and at external boundaries unless
// `subcell_at_external_boundary`.
ActiveGrid expected_active_grid(const ElementId<3>& id,
                                const bool shell_is_dg_only,
                                const bool subcell_at_external_boundary) {
  const size_t block = id.block_id();
  const size_t xi_index = id.segment_id(0).index();
  if (block == 3 or (block == 0 and shell_is_dg_only)) {
    return ActiveGrid::Dg;
  }
  if ((block == 1 and xi_index == 0 and shell_is_dg_only) or
      (block == 2 and xi_index == 1)) {
    return ActiveGrid::Dg;
  }
  const bool at_external_boundary =
      block == 0 ? xi_index == 0
                 : (at_block_boundary(id.segment_id(1)) or
                    at_block_boundary(id.segment_id(2)));
  return (at_external_boundary and not subcell_at_external_boundary)
             ? ActiveGrid::Dg
             : ActiveGrid::Subcell;
}

// Every element is classified as by hand, and the neighbor active grids of
// every element hold one entry per neighbor, equal to the classification of
// that neighbor's own element.
void test_classification(const bool shell_is_dg_only,
                         const bool subcell_at_external_boundary) {
  CAPTURE(shell_is_dg_only);
  CAPTURE(subcell_at_external_boundary);
  const auto domain_blocks = blocks();
  const subcell::SubcellOptions options = subcell_options(shell_is_dg_only);
  const auto element_ids = initial_element_ids(refinement_levels);
  REQUIRE(element_ids.size() == 2 + 128 + 32 + 32);
  for (const auto& id : element_ids) {
    CAPTURE(id);
    const Element<3> element =
        domain::create_initial_element(id, domain_blocks, refinement_levels);
    CHECK(subcell::initial_active_grid(element, options,
                                       subcell_at_external_boundary) ==
          expected_active_grid(id, shell_is_dg_only,
                               subcell_at_external_boundary));
    const auto neighbor_active_grids = subcell::initial_neighbor_active_grids(
        element, domain_blocks, refinement_levels, options,
        subcell_at_external_boundary);
    CHECK(neighbor_active_grids.size() == element.number_of_neighbors());
    for (const auto& [direction, neighbors] : element.neighbors()) {
      for (const auto& neighbor : neighbors) {
        CAPTURE(neighbor);
        const DirectionalId<3> neighbor_key{direction, neighbor};
        REQUIRE(neighbor_active_grids.contains(neighbor_key));
        CHECK(neighbor_active_grids.at(neighbor_key) ==
              expected_active_grid(neighbor, shell_is_dg_only,
                                   subcell_at_external_boundary));
      }
    }
  }

  // The outer shell element has more neighbors in one direction than a
  // DirectionalIdMap holds in total; 28 of the 64 are at the external
  // boundaries of Cube1.
  const Element<3> outer_shell = domain::create_initial_element(
      ElementId<3>{0, {{{1, 1}, {0, 0}, {0, 0}}}}, domain_blocks,
      refinement_levels);
  REQUIRE(outer_shell.neighbors().at(Direction<3>::upper_xi()).size() == 64);
  CHECK_FALSE(
      outer_shell.neighbors().at(Direction<3>::upper_xi()).are_conforming());
  CHECK(maximum_number_of_neighbors(3) < 64);
  const auto neighbor_active_grids = subcell::initial_neighbor_active_grids(
      outer_shell, domain_blocks, refinement_levels, options,
      subcell_at_external_boundary);
  CHECK(neighbor_active_grids.size() == 65);
  size_t number_of_subcell_neighbors = 0;
  for (const auto& [neighbor, active_grid] : neighbor_active_grids) {
    if (neighbor.direction() == Direction<3>::upper_xi() and
        active_grid == ActiveGrid::Subcell) {
      ++number_of_subcell_neighbors;
    }
  }
  CHECK(number_of_subcell_neighbors ==
        (shell_is_dg_only ? 0 : (subcell_at_external_boundary ? 64 : 36)));

  // The layout has the interfaces it is meant to have: four conforming
  // neighbors in one direction, and a rotated interface.
  const Element<3> second_cube = domain::create_initial_element(
      ElementId<3>{2, {{{1, 0}, {2, 1}, {2, 1}}}}, domain_blocks,
      refinement_levels);
  CHECK(second_cube.neighbors().at(Direction<3>::lower_xi()).size() == 4);
  const Element<3> collar = domain::create_initial_element(
      ElementId<3>{2, {{{1, 1}, {2, 1}, {2, 1}}}}, domain_blocks,
      refinement_levels);
  const auto& rotated_neighbors =
      collar.neighbors().at(Direction<3>::upper_xi());
  CHECK_FALSE(
      rotated_neighbors.orientation(*rotated_neighbors.begin()).is_aligned());
}

void test_neighbor_uses_subcell() {
  const DirectionalId<3> subcell_neighbor{Direction<3>::upper_xi(),
                                          ElementId<3>{1}};
  const DirectionalId<3> dg_neighbor{Direction<3>::lower_xi(), ElementId<3>{2}};
  const DirectionalId<3> unknown_neighbor{Direction<3>::upper_eta(),
                                          ElementId<3>{3}};
  const subcell::Tags::NeighborActiveGrids<3>::type neighbor_active_grids{
      {subcell_neighbor, ActiveGrid::Subcell}, {dg_neighbor, ActiveGrid::Dg}};
  const auto box =
      db::create<tmpl::list<domain::Tags::Element<3>,
                            subcell::Tags::NeighborActiveGrids<3>>>(
          Element<3>{ElementId<3>{0}, {}}, neighbor_active_grids);
  CHECK(subcell::neighbor_uses_subcell(box, subcell_neighbor));
  CHECK_FALSE(subcell::neighbor_uses_subcell(box, dg_neighbor));
  CHECK_THROWS_WITH(subcell::neighbor_uses_subcell(box, unknown_neighbor),
                    Catch::Matchers::ContainsSubstring(
                        "has no active grid for its neighbor"));
  // The map is stored in the DataBox, so it is checkpointed.
  test_serialization(neighbor_active_grids);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.InitialActiveGrid",
                  "[Evolution][Unit]") {
  for (const bool shell_is_dg_only : {true, false}) {
    for (const bool subcell_at_external_boundary : {true, false}) {
      test_classification(shell_is_dg_only, subcell_at_external_boundary);
    }
  }
  test_neighbor_uses_subcell();
}
