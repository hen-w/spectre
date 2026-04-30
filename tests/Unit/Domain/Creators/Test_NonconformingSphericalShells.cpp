// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Creators/NonconformingSphericalShells.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Options/Context.hpp"
#include "Utilities/Gsl.hpp"

namespace {
using Excision =
    domain::creators::NonconformingSphericalShells_detail::Excision;
using InnerCube =
    domain::creators::NonconformingSphericalShells_detail::InnerCube;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition(const bool outer) {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      outer ? Direction<3>::upper_xi() : Direction<3>::lower_zeta(), 50);
}

std::string excised_option_string(
    const double inner_radius, const double interface_radius,
    const double outer_radius, const std::vector<double>& wedges_partitioning,
    const std::vector<double>& shells_partitioning,
    const size_t radial_refinement, const size_t angular_refinement,
    const size_t radial_extents, const size_t spherical_harmonic_l,
    const size_t angular_extents, const bool use_equiangular_map,
    const bool with_boundary_conditions) {
  const std::string interior_option =
      with_boundary_conditions
          ? "  Interior:\n"
            "    ExciseWithBoundaryCondition:\n"
            "      TestBoundaryCondition:\n"
            "        Direction: lower-xi\n"
            "        BlockId: 50\n"
          : "  Interior: Excise\n";
  const std::string outer_bc_option = with_boundary_conditions
                                          ? "  OuterBoundaryCondition:\n"
                                            "    TestBoundaryCondition:\n"
                                            "      Direction: upper-xi\n"
                                            "      BlockId: 50\n"
                                          : "";
  std::string wedges_part_str = "  WedgesRadialPartitioning: [";
  for (size_t i = 0; i < wedges_partitioning.size(); ++i) {
    if (i > 0) {
      wedges_part_str += ", ";
    }
    wedges_part_str += std::to_string(wedges_partitioning[i]);
  }
  wedges_part_str += "]\n";
  std::string shells_part_str = "  ShellsRadialPartitioning: [";
  for (size_t i = 0; i < shells_partitioning.size(); ++i) {
    if (i > 0) {
      shells_part_str += ", ";
    }
    shells_part_str += std::to_string(shells_partitioning[i]);
  }
  shells_part_str += "]\n";
  return "NonconformingSphericalShells:\n"
         "  InnerRadius: " +
         std::to_string(inner_radius) +
         "\n"
         "  InterfaceRadius: " +
         std::to_string(interface_radius) +
         "\n"
         "  OuterRadius: " +
         std::to_string(outer_radius) + "\n" + wedges_part_str +
         shells_part_str +
         "  InitialRadialRefinement: " +
         std::to_string(radial_refinement) +
         "\n"
         "  InitialAngularRefinementOfWedges: " +
         std::to_string(angular_refinement) +
         "\n"
         "  InitialNumberOfRadialGridPoints: " +
         std::to_string(radial_extents) +
         "\n"
         "  InitialSphericalHarmonicL: " +
         std::to_string(spherical_harmonic_l) +
         "\n"
         "  InitialNumberOfAngularGridPointsOfWedges: " +
         std::to_string(angular_extents) +
         "\n" + interior_option +
         "  UseEquiangularMap: " +
         (use_equiangular_map ? "true" : "false") + "\n" + outer_bc_option;
}

std::string filled_option_string(
    const double inner_radius, const double interface_radius,
    const double outer_radius, const std::vector<double>& wedges_partitioning,
    const std::vector<double>& shells_partitioning,
    const size_t radial_refinement, const size_t angular_refinement,
    const size_t radial_extents, const size_t spherical_harmonic_l,
    const size_t angular_extents, const double sphericity,
    const bool use_equiangular_map, const bool with_boundary_conditions) {
  const std::string outer_bc_option = with_boundary_conditions
                                          ? "  OuterBoundaryCondition:\n"
                                            "    TestBoundaryCondition:\n"
                                            "      Direction: upper-xi\n"
                                            "      BlockId: 50\n"
                                          : "";
  std::string wedges_part_str = "  WedgesRadialPartitioning: [";
  for (size_t i = 0; i < wedges_partitioning.size(); ++i) {
    if (i > 0) {
      wedges_part_str += ", ";
    }
    wedges_part_str += std::to_string(wedges_partitioning[i]);
  }
  wedges_part_str += "]\n";
  std::string shells_part_str = "  ShellsRadialPartitioning: [";
  for (size_t i = 0; i < shells_partitioning.size(); ++i) {
    if (i > 0) {
      shells_part_str += ", ";
    }
    shells_part_str += std::to_string(shells_partitioning[i]);
  }
  shells_part_str += "]\n";
  return "NonconformingSphericalShells:\n"
         "  InnerRadius: " +
         std::to_string(inner_radius) +
         "\n"
         "  InterfaceRadius: " +
         std::to_string(interface_radius) +
         "\n"
         "  OuterRadius: " +
         std::to_string(outer_radius) + "\n" + wedges_part_str +
         shells_part_str +
         "  InitialRadialRefinement: " +
         std::to_string(radial_refinement) +
         "\n"
         "  InitialAngularRefinementOfWedges: " +
         std::to_string(angular_refinement) +
         "\n"
         "  InitialNumberOfRadialGridPoints: " +
         std::to_string(radial_extents) +
         "\n"
         "  InitialSphericalHarmonicL: " +
         std::to_string(spherical_harmonic_l) +
         "\n"
         "  InitialNumberOfAngularGridPointsOfWedges: " +
         std::to_string(angular_extents) +
         "\n"
         "  Interior:\n"
         "    FillWithSphericity: " +
         std::to_string(sphericity) +
         "\n"
         "  UseEquiangularMap: " +
         (use_equiangular_map ? "true" : "false") + "\n" + outer_bc_option;
}

void test_parse_errors() {
  INFO("NonconformingSphericalShells check throws");
  const double inner_radius = 1.9;
  const double interface_radius = 2.4;
  const double outer_radius = 2.9;
  const size_t radial_refinement = 0;
  const size_t angular_refinement = 1;
  const size_t radial_extents = 12;
  const size_t l = 9;
  const size_t angular_extents = 11;

  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, 0.5 * inner_radius, outer_radius, {}, {},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Inner radius must be smaller than interface radius"));

  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, 1.5 * outer_radius, outer_radius, {}, {},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Interface radius must be smaller than outer radius"));

  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents, Excision{create_boundary_condition(false)}, true,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Must specify either both inner and outer boundary conditions "
          "or neither."));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents, Excision{create_boundary_condition(false)}, true,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with "
          "NonconformingSphericalShells"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents,
          Excision{std::make_unique<TestHelpers::domain::BoundaryConditions::
                                        TestPeriodicBoundaryCondition<3>>()},
          true, create_boundary_condition(true),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with "
          "NonconformingSphericalShells"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents, Excision{create_boundary_condition(false)}, true,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents,
          Excision{std::make_unique<TestHelpers::domain::BoundaryConditions::
                                        TestNoneBoundaryCondition<3>>()},
          true, create_boundary_condition(true),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));

  // Wedges partitioning parse errors
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {2.2, 2.0}, {},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify radial partitioning in ascending order"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {1.5}, {},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "First radial partition must be larger than the inner radius"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {2.5}, {},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Last radial partition must be smaller than the interface radius"));

  // Shells partitioning parse errors
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {2.7, 2.5},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify radial partitioning in ascending order"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {2.3},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "First radial partition must be larger than the interface radius"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {3.0},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Last radial partition must be smaller than the outer radius"));
}

template <typename Generator>
void test_excised_construction(
    const gsl::not_null<Generator*> gen,
    const domain::creators::NonconformingSphericalShells& creator,
    const double inner_radius, const double interface_radius,
    const double outer_radius,
    const std::vector<double>& wedges_partitioning,
    const std::vector<double>& shells_partitioning,
    const bool expect_boundary_conditions = true) {
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      creator, expect_boundary_conditions);
  const auto& grid_anchors = creator.grid_anchors();
  CHECK(grid_anchors.size() == 1);
  CHECK(grid_anchors.count("Center") == 1);
  CHECK(grid_anchors.at("Center") ==
        tnsr::I<double, 3, Frame::Grid>{std::array{0.0, 0.0, 0.0}});

  const size_t num_wedge_layers = 1 + wedges_partitioning.size();
  const size_t num_shells = 1 + shells_partitioning.size();
  const size_t num_wedge_blocks = 6 * num_wedge_layers;
  const size_t expected_num_blocks = num_wedge_blocks + num_shells;

  const auto& blocks = domain.blocks();
  const auto block_names = creator.block_names();
  const size_t num_blocks = blocks.size();
  CAPTURE(num_blocks);
  CHECK(num_blocks == expected_num_blocks);
  const auto all_boundary_conditions = creator.external_boundary_conditions();

  // Check total number of external boundaries: 6 inner + 1 outer
  const size_t num_external_boundaries =
      alg::accumulate(blocks, 0_st, [](const size_t count, const auto& block) {
        return count + block.external_boundaries().size();
      });
  CHECK(num_external_boundaries == 7);

  // Build wedge radii (including inner/interface)
  std::vector<double> wedge_radii;
  wedge_radii.push_back(inner_radius);
  for (const auto& r : wedges_partitioning) {
    wedge_radii.push_back(r);
  }
  wedge_radii.push_back(interface_radius);

  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  for (size_t layer = 0; layer < num_wedge_layers; ++layer) {
    const double layer_inner = wedge_radii[layer];
    const double layer_outer = wedge_radii[layer + 1];
    for (size_t wedge = 0; wedge < 6; ++wedge) {
      const size_t block_id = layer * 6 + wedge;
      CAPTURE(block_id);
      const auto& block = blocks[block_id];
      const ElementMap<3, Frame::Inertial> inertial_element_map{
          ElementId<3>{block_id}, block};
      {
        INFO("Radius of random point on lower face of wedge");
        const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
            {{xi_distribution(*gen), xi_distribution(*gen), -1.0}}};
        auto x_inertial = inertial_element_map(x_logical);
        CHECK(get(magnitude(x_inertial)) == approx(layer_inner));
      }
      {
        INFO("Radius of random point on upper face of wedge");
        const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
            {{xi_distribution(*gen), xi_distribution(*gen), 1.0}}};
        auto x_inertial = inertial_element_map(x_logical);
        CHECK(get(magnitude(x_inertial)) == approx(layer_outer));
      }
      if (layer == 0) {
        INFO("External boundaries of innermost wedges");
        const auto& external_boundaries = block.external_boundaries();
        CHECK(external_boundaries.size() == 1);
        CHECK(alg::found(external_boundaries, Direction<3>::lower_zeta()));
        if (expect_boundary_conditions) {
          const auto& boundary_conditions =
              all_boundary_conditions[block_id];
          for (const auto& direction : block.external_boundaries()) {
            CAPTURE(direction);
            const auto& boundary_condition =
                dynamic_cast<const TestHelpers::domain::BoundaryConditions::
                                 TestBoundaryCondition<3>&>(
                    *boundary_conditions.at(direction));
            CHECK(boundary_condition.direction() == direction);
          }
        }
      } else {
        INFO("Non-innermost wedges have no external boundaries");
        CHECK(block.external_boundaries().empty());
      }
    }
  }

  // Build shell radii
  std::vector<double> shell_radii;
  shell_radii.push_back(interface_radius);
  for (const auto& r : shells_partitioning) {
    shell_radii.push_back(r);
  }
  shell_radii.push_back(outer_radius);

  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> theta_distribution(0.0, M_PI);
  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);

  for (size_t i = 0; i < num_shells; ++i) {
    const size_t shell_block_id = num_wedge_blocks + i;
    CAPTURE(shell_block_id);
    const auto& block = blocks[shell_block_id];
    const ElementMap<3, Frame::Inertial> inertial_element_map{
        ElementId<3>{shell_block_id}, block};
    {
      INFO("Radius of random point on lower face of shell");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{-1.0, theta_distribution(*gen), phi_distribution(*gen)}}};
      auto x_inertial = inertial_element_map(x_logical);
      CHECK(get(magnitude(x_inertial)) == approx(shell_radii[i]));
    }
    {
      INFO("Radius of random point on upper face of shell");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{1.0, theta_distribution(*gen), phi_distribution(*gen)}}};
      auto x_inertial = inertial_element_map(x_logical);
      CHECK(get(magnitude(x_inertial)) == approx(shell_radii[i + 1]));
    }
    if (i == num_shells - 1) {
      INFO("External boundaries of outermost shell");
      const auto& external_boundaries = block.external_boundaries();
      CHECK(external_boundaries.size() == 1);
      CHECK(alg::found(external_boundaries, Direction<3>::upper_xi()));
      if (expect_boundary_conditions) {
        const auto& boundary_conditions =
            all_boundary_conditions[shell_block_id];
        for (const auto& direction : block.external_boundaries()) {
          CAPTURE(direction);
          const auto& boundary_condition =
              dynamic_cast<const TestHelpers::domain::BoundaryConditions::
                               TestBoundaryCondition<3>&>(
                  *boundary_conditions.at(direction));
          CHECK(boundary_condition.direction() == direction);
        }
      }
    } else {
      INFO("Non-outermost shells have no external boundaries");
      CHECK(block.external_boundaries().empty());
    }
  }
}

template <typename Generator>
void test_filled_construction(
    const gsl::not_null<Generator*> gen,
    const domain::creators::NonconformingSphericalShells& creator,
    const double inner_radius, const double interface_radius,
    const double outer_radius,
    const std::vector<double>& wedges_partitioning,
    const std::vector<double>& shells_partitioning,
    const bool expect_boundary_conditions = true) {
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      creator, expect_boundary_conditions);
  const auto& grid_anchors = creator.grid_anchors();
  CHECK(grid_anchors.size() == 1);
  CHECK(grid_anchors.count("Center") == 1);

  const size_t num_wedge_layers = 1 + wedges_partitioning.size();
  const size_t num_shells = 1 + shells_partitioning.size();
  const size_t num_wedge_blocks = 6 * num_wedge_layers;
  const size_t expected_num_blocks = num_wedge_blocks + 1 + num_shells;

  const auto& blocks = domain.blocks();
  const auto block_names = creator.block_names();
  const size_t num_blocks = blocks.size();
  CAPTURE(num_blocks);
  CHECK(num_blocks == expected_num_blocks);
  CHECK(block_names[num_wedge_blocks] == "InnerCube");
  CHECK(block_names[num_wedge_blocks + 1] == "Shell0");

  const auto all_boundary_conditions = creator.external_boundary_conditions();

  // Check total number of external boundaries: only the outermost shell's
  // outer face
  const size_t num_external_boundaries =
      alg::accumulate(blocks, 0_st, [](const size_t count, const auto& block) {
        return count + block.external_boundaries().size();
      });
  CHECK(num_external_boundaries == 1);

  // Build wedge radii
  std::vector<double> wedge_radii;
  wedge_radii.push_back(inner_radius);
  for (const auto& r : wedges_partitioning) {
    wedge_radii.push_back(r);
  }
  wedge_radii.push_back(interface_radius);

  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  for (size_t layer = 0; layer < num_wedge_layers; ++layer) {
    const double layer_outer = wedge_radii[layer + 1];
    for (size_t wedge = 0; wedge < 6; ++wedge) {
      const size_t block_id = layer * 6 + wedge;
      CAPTURE(block_id);
      const auto& block = blocks[block_id];
      const ElementMap<3, Frame::Inertial> inertial_element_map{
          ElementId<3>{block_id}, block};
      {
        INFO("Radius of random point on upper face of wedge");
        const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
            {{xi_distribution(*gen), xi_distribution(*gen), 1.0}}};
        auto x_inertial = inertial_element_map(x_logical);
        CHECK(get(magnitude(x_inertial)) == approx(layer_outer));
      }
      {
        INFO("Wedges have no external boundaries when filled");
        CHECK(block.external_boundaries().empty());
      }
    }
  }

  // Inner cube block
  {
    INFO("Inner cube block");
    const auto& cube_block = blocks[num_wedge_blocks];
    CHECK(cube_block.external_boundaries().empty());
  }

  // Build shell radii
  std::vector<double> shell_radii;
  shell_radii.push_back(interface_radius);
  for (const auto& r : shells_partitioning) {
    shell_radii.push_back(r);
  }
  shell_radii.push_back(outer_radius);

  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> theta_distribution(0.0, M_PI);
  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);

  const size_t first_shell_id = num_wedge_blocks + 1;
  for (size_t i = 0; i < num_shells; ++i) {
    const size_t shell_block_id = first_shell_id + i;
    CAPTURE(shell_block_id);
    const auto& shell_block = blocks[shell_block_id];
    const ElementMap<3, Frame::Inertial> shell_element_map{
        ElementId<3>{shell_block_id}, shell_block};
    {
      INFO("Radius of random point on lower face of shell");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{-1.0, theta_distribution(*gen), phi_distribution(*gen)}}};
      auto x_inertial = shell_element_map(x_logical);
      CHECK(get(magnitude(x_inertial)) == approx(shell_radii[i]));
    }
    {
      INFO("Radius of random point on upper face of shell");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{1.0, theta_distribution(*gen), phi_distribution(*gen)}}};
      auto x_inertial = shell_element_map(x_logical);
      CHECK(get(magnitude(x_inertial)) == approx(shell_radii[i + 1]));
    }
    if (i == num_shells - 1) {
      INFO("External boundaries of outermost shell");
      const auto& external_boundaries = shell_block.external_boundaries();
      CHECK(external_boundaries.size() == 1);
      CHECK(alg::found(external_boundaries, Direction<3>::upper_xi()));
      if (expect_boundary_conditions) {
        const auto& boundary_conditions =
            all_boundary_conditions[shell_block_id];
        for (const auto& direction : shell_block.external_boundaries()) {
          CAPTURE(direction);
          const auto& boundary_condition =
              dynamic_cast<const TestHelpers::domain::BoundaryConditions::
                               TestBoundaryCondition<3>&>(
                  *boundary_conditions.at(direction));
          CHECK(boundary_condition.direction() == direction);
        }
      }
    } else {
      INFO("Non-outermost shells have no external boundaries");
      CHECK(shell_block.external_boundaries().empty());
    }
  }
}

template <typename Generator>
void test_excised(const gsl::not_null<Generator*> gen) {
  INFO("Excised interior");
  const double inner_radius = 1.0;
  const double interface_radius = 1.5;
  const double outer_radius = 2.0;
  const size_t radial_refinement = 3;
  const size_t angular_refinement = 2;
  const size_t radial_extents = 5;
  const size_t l = 6;
  const size_t angular_extents = 7;
  for (const bool with_boundary_conditions : {true, false}) {
    CAPTURE(with_boundary_conditions);
    // No partitioning (backward compatibility)
    {
      INFO("No partitioning");
      const domain::creators::NonconformingSphericalShells creator{
          inner_radius,
          interface_radius,
          outer_radius,
          {},
          {},
          radial_refinement,
          angular_refinement,
          radial_extents,
          l,
          angular_extents,
          with_boundary_conditions ? Excision{create_boundary_condition(false)}
                                  : Excision{nullptr},
          true,
          with_boundary_conditions ? create_boundary_condition(true) : nullptr};
      test_excised_construction(gen, creator, inner_radius, interface_radius,
                                outer_radius, {}, {},
                                with_boundary_conditions);
      TestHelpers::domain::creators::test_creation(
          excised_option_string(inner_radius, interface_radius, outer_radius,
                                {}, {}, radial_refinement, angular_refinement,
                                radial_extents, l, angular_extents, true,
                                with_boundary_conditions),
          creator, with_boundary_conditions);
    }
    // Wedges partitioning only
    {
      INFO("Wedges partitioning only");
      const std::vector<double> wedges_part{1.2};
      const domain::creators::NonconformingSphericalShells creator{
          inner_radius,
          interface_radius,
          outer_radius,
          wedges_part,
          {},
          radial_refinement,
          angular_refinement,
          radial_extents,
          l,
          angular_extents,
          with_boundary_conditions ? Excision{create_boundary_condition(false)}
                                  : Excision{nullptr},
          true,
          with_boundary_conditions ? create_boundary_condition(true) : nullptr};
      test_excised_construction(gen, creator, inner_radius, interface_radius,
                                outer_radius, wedges_part, {},
                                with_boundary_conditions);
      TestHelpers::domain::creators::test_creation(
          excised_option_string(inner_radius, interface_radius, outer_radius,
                                wedges_part, {}, radial_refinement,
                                angular_refinement, radial_extents, l,
                                angular_extents, true,
                                with_boundary_conditions),
          creator, with_boundary_conditions);
    }
    // Shells partitioning only
    {
      INFO("Shells partitioning only");
      const std::vector<double> shells_part{1.7};
      const domain::creators::NonconformingSphericalShells creator{
          inner_radius,
          interface_radius,
          outer_radius,
          {},
          shells_part,
          radial_refinement,
          angular_refinement,
          radial_extents,
          l,
          angular_extents,
          with_boundary_conditions ? Excision{create_boundary_condition(false)}
                                  : Excision{nullptr},
          true,
          with_boundary_conditions ? create_boundary_condition(true) : nullptr};
      test_excised_construction(gen, creator, inner_radius, interface_radius,
                                outer_radius, {}, shells_part,
                                with_boundary_conditions);
      TestHelpers::domain::creators::test_creation(
          excised_option_string(inner_radius, interface_radius, outer_radius,
                                {}, shells_part, radial_refinement,
                                angular_refinement, radial_extents, l,
                                angular_extents, true,
                                with_boundary_conditions),
          creator, with_boundary_conditions);
    }
    // Both partitioning
    {
      INFO("Both partitioning");
      const std::vector<double> wedges_part{1.2};
      const std::vector<double> shells_part{1.7};
      const domain::creators::NonconformingSphericalShells creator{
          inner_radius,
          interface_radius,
          outer_radius,
          wedges_part,
          shells_part,
          radial_refinement,
          angular_refinement,
          radial_extents,
          l,
          angular_extents,
          with_boundary_conditions ? Excision{create_boundary_condition(false)}
                                  : Excision{nullptr},
          true,
          with_boundary_conditions ? create_boundary_condition(true) : nullptr};
      test_excised_construction(gen, creator, inner_radius, interface_radius,
                                outer_radius, wedges_part, shells_part,
                                with_boundary_conditions);
      TestHelpers::domain::creators::test_creation(
          excised_option_string(inner_radius, interface_radius, outer_radius,
                                wedges_part, shells_part, radial_refinement,
                                angular_refinement, radial_extents, l,
                                angular_extents, true,
                                with_boundary_conditions),
          creator, with_boundary_conditions);
    }
  }
}

template <typename Generator>
void test_filled(const gsl::not_null<Generator*> gen) {
  INFO("Filled interior");
  const double inner_radius = 1.0;
  const double interface_radius = 1.5;
  const double outer_radius = 2.0;
  const size_t radial_refinement = 3;
  const size_t angular_refinement = 2;
  const size_t radial_extents = 5;
  const size_t l = 6;
  const size_t angular_extents = 7;
  for (const double sphericity : {0.0, 0.5}) {
    CAPTURE(sphericity);
    for (const bool use_equiangular_map : {true, false}) {
      CAPTURE(use_equiangular_map);
      for (const bool with_boundary_conditions : {true, false}) {
        CAPTURE(with_boundary_conditions);
        // No partitioning
        {
          INFO("No partitioning");
          const domain::creators::NonconformingSphericalShells creator{
              inner_radius,
              interface_radius,
              outer_radius,
              {},
              {},
              radial_refinement,
              angular_refinement,
              radial_extents,
              l,
              angular_extents,
              InnerCube{sphericity},
              use_equiangular_map,
              with_boundary_conditions ? create_boundary_condition(true)
                                      : nullptr};
          test_filled_construction(gen, creator, inner_radius,
                                   interface_radius, outer_radius, {}, {},
                                   with_boundary_conditions);
          TestHelpers::domain::creators::test_creation(
              filled_option_string(
                  inner_radius, interface_radius, outer_radius, {}, {},
                  radial_refinement, angular_refinement, radial_extents, l,
                  angular_extents, sphericity, use_equiangular_map,
                  with_boundary_conditions),
              creator, with_boundary_conditions);
        }
        // Both partitioning
        {
          INFO("Both partitioning");
          const std::vector<double> wedges_part{1.2};
          const std::vector<double> shells_part{1.7};
          const domain::creators::NonconformingSphericalShells creator{
              inner_radius,
              interface_radius,
              outer_radius,
              wedges_part,
              shells_part,
              radial_refinement,
              angular_refinement,
              radial_extents,
              l,
              angular_extents,
              InnerCube{sphericity},
              use_equiangular_map,
              with_boundary_conditions ? create_boundary_condition(true)
                                      : nullptr};
          test_filled_construction(gen, creator, inner_radius,
                                   interface_radius, outer_radius,
                                   wedges_part, shells_part,
                                   with_boundary_conditions);
          TestHelpers::domain::creators::test_creation(
              filled_option_string(
                  inner_radius, interface_radius, outer_radius, wedges_part,
                  shells_part, radial_refinement, angular_refinement,
                  radial_extents, l, angular_extents, sphericity,
                  use_equiangular_map, with_boundary_conditions),
              creator, with_boundary_conditions);
        }
      }
    }
  }
}
}  // namespace

// [[TimeOut, 30]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.NonconformingSphericalShells",
                  "[Domain][Unit]") {
  MAKE_GENERATOR(gen);
  domain::creators::time_dependence::register_derived_with_charm();
  test_parse_errors();
  test_excised(make_not_null(&gen));
  test_filled(make_not_null(&gen));
}
