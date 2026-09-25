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
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
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
using Distribution = domain::CoordinateMaps::Distribution;

// Build a vector of `size` Linear distributions.
std::vector<Distribution> linear(const size_t size) {
  return std::vector<Distribution>(size, Distribution::Linear);
}

// Build an all-Linear RadialDistribution option with `num_wedge_layers`
// entries for the wedges and `num_shells` entries for the spherical-harmonic
// shells.
std::array<std::vector<Distribution>, 2> all_linear(
    const size_t num_wedge_layers, const size_t num_shells) {
  return {{linear(num_wedge_layers), linear(num_shells)}};
}

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
  // All-Linear RadialDistribution sized to match the partitionings.
  std::string radial_dist_str = "  RadialDistribution: [[";
  const size_t num_wedge_layers = 1 + wedges_partitioning.size();
  const size_t num_shells = 1 + shells_partitioning.size();
  for (size_t i = 0; i < num_wedge_layers; ++i) {
    if (i > 0) {
      radial_dist_str += ", ";
    }
    radial_dist_str += "Linear";
  }
  radial_dist_str += "], [";
  for (size_t i = 0; i < num_shells; ++i) {
    if (i > 0) {
      radial_dist_str += ", ";
    }
    radial_dist_str += "Linear";
  }
  radial_dist_str += "]]\n";
  return "NonconformingSphericalShells:\n"
         "  InnerRadius: " +
         std::to_string(inner_radius) +
         "\n"
         "  InterfaceRadius: " +
         std::to_string(interface_radius) +
         "\n"
         "  OuterRadius: " +
         std::to_string(outer_radius) + "\n" + wedges_part_str +
         shells_part_str + radial_dist_str +
         "  InitialRadialRefinement: " + std::to_string(radial_refinement) +
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
         std::to_string(angular_extents) + "\n" + interior_option +
         "  UseEquiangularMap: " + (use_equiangular_map ? "true" : "false") +
         "\n" + outer_bc_option;
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
  // All-Linear RadialDistribution sized to match the partitionings.
  std::string radial_dist_str = "  RadialDistribution: [[";
  const size_t num_wedge_layers = 1 + wedges_partitioning.size();
  const size_t num_shells = 1 + shells_partitioning.size();
  for (size_t i = 0; i < num_wedge_layers; ++i) {
    if (i > 0) {
      radial_dist_str += ", ";
    }
    radial_dist_str += "Linear";
  }
  radial_dist_str += "], [";
  for (size_t i = 0; i < num_shells; ++i) {
    if (i > 0) {
      radial_dist_str += ", ";
    }
    radial_dist_str += "Linear";
  }
  radial_dist_str += "]]\n";
  return "NonconformingSphericalShells:\n"
         "  InnerRadius: " +
         std::to_string(inner_radius) +
         "\n"
         "  InterfaceRadius: " +
         std::to_string(interface_radius) +
         "\n"
         "  OuterRadius: " +
         std::to_string(outer_radius) + "\n" + wedges_part_str +
         shells_part_str + radial_dist_str +
         "  InitialRadialRefinement: " + std::to_string(radial_refinement) +
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

  CHECK_THROWS_WITH(domain::creators::NonconformingSphericalShells(
                        inner_radius, 0.5 * inner_radius, outer_radius, {}, {},
                        all_linear(1, 1), radial_refinement, angular_refinement,
                        radial_extents, l, angular_extents, Excision{nullptr},
                        true, nullptr, Options::Context{false, {}, 1, 1}),
                    Catch::Matchers::ContainsSubstring(
                        "Inner radius must be smaller than interface radius"));

  CHECK_THROWS_WITH(domain::creators::NonconformingSphericalShells(
                        inner_radius, 1.5 * outer_radius, outer_radius, {}, {},
                        all_linear(1, 1), radial_refinement, angular_refinement,
                        radial_extents, l, angular_extents, Excision{nullptr},
                        true, nullptr, Options::Context{false, {}, 1, 1}),
                    Catch::Matchers::ContainsSubstring(
                        "Interface radius must be smaller than outer radius"));

  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {},
          all_linear(1, 1), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents,
          Excision{create_boundary_condition(false)}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Must specify either both inner and outer boundary conditions "
          "or neither."));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {},
          all_linear(1, 1), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents,
          Excision{create_boundary_condition(false)}, true,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with "
          "NonconformingSphericalShells"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {},
          all_linear(1, 1), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents,
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
          all_linear(1, 1), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents,
          Excision{create_boundary_condition(false)}, true,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {},
          all_linear(1, 1), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents,
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
          all_linear(3, 1), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify radial partitioning in ascending order"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {1.5}, {},
          all_linear(2, 1), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "First radial partition must be larger than the inner radius"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {2.5}, {},
          all_linear(2, 1), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Last radial partition must be smaller than the interface radius"));

  // Shells partitioning parse errors
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {2.7, 2.5},
          all_linear(1, 3), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify radial partitioning in ascending order"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {2.3},
          all_linear(1, 2), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "First radial partition must be larger than the interface radius"));
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {3.0},
          all_linear(1, 2), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Last radial partition must be smaller than the outer radius"));

  // RadialDistribution parse errors.
  // (i) Wrong number of distributions for the wedge layers: with one wedge
  // partition there are two wedge layers, but only one distribution is given.
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {2.1}, {},
          all_linear(1, 1), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify a 'RadialDistribution' for every spherical shell"));
  // (ii) Wrong number of distributions for the shells: with one shell partition
  // there are two shells, but only one distribution is given.
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {2.6},
          all_linear(1, 1), radial_refinement, angular_refinement,
          radial_extents, l, angular_extents, Excision{nullptr}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify a 'RadialDistribution' for every spherical shell"));
  // (iii) Non-Linear innermost wedge layer with a filled-cube interior.
  CHECK_THROWS_WITH(
      domain::creators::NonconformingSphericalShells(
          inner_radius, interface_radius, outer_radius, {}, {},
          std::array<std::vector<Distribution>, 2>{
              {std::vector<Distribution>{Distribution::Logarithmic},
               std::vector<Distribution>{Distribution::Linear}}},
          radial_refinement, angular_refinement, radial_extents, l,
          angular_extents, InnerCube{0.0}, true, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "must be 'Linear' for the innermost wedge layer when the interior is "
          "filled with a cube"));
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
          all_linear(1, 1),
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
          all_linear(2, 1),
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
          all_linear(1, 2),
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
          all_linear(2, 2),
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
              all_linear(1, 1),
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
              all_linear(2, 2),
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

// Pin (a): an all-Linear domain must reproduce the exact affine radii at
// sample logical points. Catches Interval(Linear) silently changing the old
// (Affine) behavior for either the shells or the wedges.
void test_linear_unchanged() {
  INFO("Linear distribution reproduces the affine radii");
  const double inner_radius = 1.0;
  const double interface_radius = 1.5;
  const double outer_radius = 2.0;
  const domain::creators::NonconformingSphericalShells creator{
      inner_radius,
      interface_radius,
      outer_radius,
      {},
      {},
      all_linear(1, 1),
      3,
      2,
      5,
      6,
      7,
      Excision{nullptr},
      true,
      nullptr};
  const auto domain = creator.create_domain();
  const auto& blocks = domain.blocks();
  {
    INFO("Wedge block spans [inner, interface] linearly along zeta");
    const size_t wedge_block_id = 0;
    const ElementMap<3, Frame::Inertial> element_map{
        ElementId<3>{wedge_block_id}, blocks[wedge_block_id]};
    const auto radius_at = [&element_map](const double zeta) {
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{0.0, 0.0, zeta}}};
      return get(magnitude(element_map(x_logical)));
    };
    CHECK(radius_at(-1.0) == approx(inner_radius));
    CHECK(radius_at(0.0) == approx(0.5 * (inner_radius + interface_radius)));
    CHECK(radius_at(1.0) == approx(interface_radius));
  }
  {
    INFO("Shell block spans [interface, outer] linearly along xi");
    const size_t shell_block_id = 6;  // 6 wedges (1 layer), excised, 1st shell
    const ElementMap<3, Frame::Inertial> element_map{
        ElementId<3>{shell_block_id}, blocks[shell_block_id]};
    const double theta = 1.0;
    const double phi = 2.0;
    const auto radius_at = [&element_map, theta, phi](const double xi) {
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
          {{xi, theta, phi}}};
      return get(magnitude(element_map(x_logical)));
    };
    CHECK(radius_at(-1.0) == approx(interface_radius));
    CHECK(radius_at(0.0) == approx(0.5 * (interface_radius + outer_radius)));
    CHECK(radius_at(1.0) == approx(outer_radius));
  }
}

// Pin (b): a Logarithmic spherical-harmonic shell must reproduce the radii of
// a directly-constructed Interval(Logarithmic) map. Catches the shell radial
// distribution option being dropped (the shell would fall back to Linear).
void test_logarithmic_shells() {
  INFO("Logarithmic shell reproduces the Interval(Logarithmic) radii");
  const double inner_radius = 1.0;
  const double interface_radius = 1.5;
  const double outer_radius = 2.0;
  const domain::creators::NonconformingSphericalShells creator{
      inner_radius,
      interface_radius,
      outer_radius,
      {},
      {},
      std::array<std::vector<Distribution>, 2>{
          {std::vector<Distribution>{Distribution::Linear},
           std::vector<Distribution>{Distribution::Logarithmic}}},
      3,
      2,
      5,
      6,
      7,
      Excision{nullptr},
      true,
      nullptr};
  const auto domain = creator.create_domain();
  const auto& blocks = domain.blocks();
  const size_t shell_block_id = 6;  // 6 wedges (1 layer), excised, 1st shell
  const ElementMap<3, Frame::Inertial> element_map{ElementId<3>{shell_block_id},
                                                   blocks[shell_block_id]};
  // Reference is exactly the Interval the creator builds for this shell.
  const domain::CoordinateMaps::Interval reference{
      -1.0, 1.0, interface_radius, outer_radius, Distribution::Logarithmic,
      0.0};
  const double theta = 1.0;
  const double phi = 2.0;
  for (const double xi : {-1.0, 0.0, 1.0}) {
    CAPTURE(xi);
    const double reference_radius = reference(std::array<double, 1>{{xi}})[0];
    const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
        {{xi, theta, phi}}};
    CHECK(get(magnitude(element_map(x_logical))) == approx(reference_radius));
  }
}

// Pin (c): a Logarithmic OUTER wedge layer (filled interior, Linear innermost
// layer) must reproduce the radii of a directly-constructed
// Interval(Logarithmic) map, and its midpoint radius must differ from the
// linear midpoint. Catches the wedge radial distribution option being dropped.
void test_logarithmic_wedge() {
  INFO(
      "Logarithmic outer wedge layer reproduces the Interval(Logarithmic) "
      "radii");
  const double inner_radius = 1.0;
  const double interface_radius = 1.5;
  const double outer_radius = 2.0;
  const double wedge_partition = 1.2;
  const domain::creators::NonconformingSphericalShells creator{
      inner_radius,
      interface_radius,
      outer_radius,
      {wedge_partition},
      {},
      std::array<std::vector<Distribution>, 2>{
          {std::vector<Distribution>{Distribution::Linear,
                                     Distribution::Logarithmic},
           std::vector<Distribution>{Distribution::Linear}}},
      3,
      2,
      5,
      6,
      7,
      InnerCube{0.0},
      true,
      nullptr};
  const auto domain = creator.create_domain();
  const auto& blocks = domain.blocks();
  // Two wedge layers (12 wedge blocks). The outer layer is blocks 6..11 and
  // spans [wedge_partition, interface_radius] with both faces spherical, so at
  // xi = eta = 0 the mapped radius follows the radial distribution exactly.
  const size_t wedge_block_id = 6;
  const ElementMap<3, Frame::Inertial> element_map{ElementId<3>{wedge_block_id},
                                                   blocks[wedge_block_id]};
  const domain::CoordinateMaps::Interval reference{
      -1.0, 1.0, wedge_partition, interface_radius, Distribution::Logarithmic,
      0.0};
  const auto radius_at = [&element_map](const double zeta) {
    const tnsr::I<double, 3, Frame::ElementLogical> x_logical{
        {{0.0, 0.0, zeta}}};
    return get(magnitude(element_map(x_logical)));
  };
  for (const double zeta : {-1.0, 0.0, 1.0}) {
    CAPTURE(zeta);
    const double reference_radius = reference(std::array<double, 1>{{zeta}})[0];
    CHECK(radius_at(zeta) == approx(reference_radius));
  }
  // The log-distributed midpoint must differ from the linear midpoint,
  // otherwise the wedge distribution option was silently ignored.
  CHECK(radius_at(0.0) != approx(0.5 * (wedge_partition + interface_radius)));
}

// Pin (e): a creator built with a mixed (non-Linear) distribution must produce
// block maps that differ from the all-Linear ones. Guards against the option
// being parsed but never plumbed into the maps.
void test_mixed_differs_from_linear() {
  INFO("Mixed distributions differ from all-Linear");
  const double inner_radius = 1.0;
  const double interface_radius = 1.5;
  const double outer_radius = 2.0;
  const auto make_creator = [&](const Distribution shell_distribution) {
    return domain::creators::NonconformingSphericalShells{
        inner_radius,
        interface_radius,
        outer_radius,
        {},
        {},
        std::array<std::vector<Distribution>, 2>{
            {std::vector<Distribution>{Distribution::Linear},
             std::vector<Distribution>{shell_distribution}}},
        3,
        2,
        5,
        6,
        7,
        Excision{nullptr},
        true,
        nullptr};
  };
  const auto linear_creator = make_creator(Distribution::Linear);
  const auto log_creator = make_creator(Distribution::Logarithmic);
  const auto linear_domain = linear_creator.create_domain();
  const auto log_domain = log_creator.create_domain();
  const size_t shell_block_id = 6;  // 6 wedges (1 layer), excised, 1st shell
  const ElementMap<3, Frame::Inertial> linear_map{
      ElementId<3>{shell_block_id}, linear_domain.blocks()[shell_block_id]};
  const ElementMap<3, Frame::Inertial> log_map{
      ElementId<3>{shell_block_id}, log_domain.blocks()[shell_block_id]};
  const tnsr::I<double, 3, Frame::ElementLogical> midpoint_logical{
      {{0.0, 1.0, 2.0}}};
  const double linear_radius = get(magnitude(linear_map(midpoint_logical)));
  const double log_radius = get(magnitude(log_map(midpoint_logical)));
  CHECK(linear_radius != approx(log_radius));
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
  test_linear_unchanged();
  test_logarithmic_shells();
  test_logarithmic_wedge();
  test_mixed_differs_from_linear();
}
