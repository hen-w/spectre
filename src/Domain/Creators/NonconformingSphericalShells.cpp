// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/NonconformingSphericalShells.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/ShellDistribution.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::creators {

NonconformingSphericalShells::NonconformingSphericalShells(
    const double inner_radius, const double interface_radius,
    const double outer_radius,
    std::vector<double> wedges_radial_partitioning,
    std::vector<double> shells_radial_partitioning,
    const size_t initial_radial_refinement,
    const size_t initial_angular_refinement,
    const size_t initial_number_of_radial_grid_points,
    const size_t initial_spherical_harmonic_l,
    const size_t initial_number_of_angular_grid_points_of_wedges,
    std::variant<Excision, InnerCube> interior,
    const bool use_equiangular_map,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      interface_radius_(interface_radius),
      outer_radius_(outer_radius),
      wedges_radial_partitioning_(std::move(wedges_radial_partitioning)),
      shells_radial_partitioning_(std::move(shells_radial_partitioning)),
      initial_radial_refinement_(initial_radial_refinement),
      initial_angular_refinement_(initial_angular_refinement),
      initial_number_of_radial_grid_points_(
          initial_number_of_radial_grid_points),
      initial_spherical_harmonic_l_(initial_spherical_harmonic_l),
      initial_number_of_angular_grid_points_of_wedges_(
          initial_number_of_angular_grid_points_of_wedges),
      interior_(std::move(interior)),
      fill_interior_(std::holds_alternative<InnerCube>(interior_)),
      use_equiangular_map_(use_equiangular_map),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      grid_anchors_{{{"Center", tnsr::I<double, 3, Frame::Grid>{
                                    std::array{0.0, 0.0, 0.0}}}}} {
  if (inner_radius_ > interface_radius_) {
    PARSE_ERROR(context,
                "Inner radius must be smaller than interface radius, but inner "
                "radius is " +
                    std::to_string(inner_radius_) +
                    " and interface radius is " +
                    std::to_string(interface_radius_) + ".");
  }

  if (interface_radius_ > outer_radius_) {
    PARSE_ERROR(
        context,
        "Interface radius must be smaller than outer radius, but interface "
        "radius is " +
            std::to_string(interface_radius_) + " and outer radius is " +
            std::to_string(outer_radius_) + ".");
  }

  // Validate wedges partitioning
  {
    std::vector<CoordinateMaps::Distribution> wedge_radial_dist;
    set_shell_distribution(
        make_not_null(&num_wedge_layers_),
        make_not_null(&wedge_radial_dist), wedges_radial_partitioning_,
        CoordinateMaps::Distribution::Linear, inner_radius_,
        interface_radius_, "inner", "interface", context);
  }

  // Validate shells partitioning
  {
    std::vector<CoordinateMaps::Distribution> shell_radial_dist;
    set_shell_distribution(
        make_not_null(&num_shells_), make_not_null(&shell_radial_dist),
        shells_radial_partitioning_, CoordinateMaps::Distribution::Linear,
        interface_radius_, outer_radius_, "interface", "outer", context);
  }

  const size_t num_wedge_blocks = 6 * num_wedge_layers_;
  const size_t num_blocks =
      num_wedge_blocks + (fill_interior_ ? 1 : 0) + num_shells_;

  // Initialize refinement and grid points
  initial_refinement_levels_.assign(
      num_blocks, {{initial_angular_refinement_, initial_angular_refinement_,
                    initial_radial_refinement_}});
  initial_number_of_grid_points_.assign(
      num_blocks,
      {{initial_number_of_angular_grid_points_of_wedges_,
        initial_number_of_angular_grid_points_of_wedges_,
        initial_number_of_radial_grid_points_}});

  // Correct for inner cube block (z-refinement = y-refinement)
  if (fill_interior_) {
    const size_t cube_index = num_wedge_blocks;
    initial_refinement_levels_[cube_index][2] =
        initial_refinement_levels_[cube_index][1];
    initial_number_of_grid_points_[cube_index][2] =
        initial_number_of_grid_points_[cube_index][1];
  }

  // Correct for all shell blocks
  const size_t first_shell_index =
      num_wedge_blocks + (fill_interior_ ? 1 : 0);
  for (size_t i = 0; i < num_shells_; ++i) {
    const size_t shell_index = first_shell_index + i;
    initial_number_of_grid_points_[shell_index] = {
        {initial_number_of_radial_grid_points_,
         initial_spherical_harmonic_l_ + 1,
         2 * initial_spherical_harmonic_l_ + 1}};
    initial_refinement_levels_[shell_index] = {
        {initial_radial_refinement_, 0_st, 0_st}};
  }

  // Validate boundary conditions
  using domain::BoundaryConditions::is_none;
  using domain::BoundaryConditions::is_periodic;
  if (not fill_interior_) {
    const auto& inner_bc =
        std::get<Excision>(interior_).boundary_condition;
    if (is_none(inner_bc) or is_none(outer_boundary_condition_)) {
      PARSE_ERROR(
          context,
          "None boundary condition is not supported. If you would like an "
          "outflow-type boundary condition, you must use that.");
    }
    if (is_periodic(inner_bc) or is_periodic(outer_boundary_condition_)) {
      PARSE_ERROR(context,
                  "Cannot have periodic boundary conditions with "
                  "NonconformingSphericalShells");
    }
    if ((inner_bc == nullptr) != (outer_boundary_condition_ == nullptr)) {
      PARSE_ERROR(context,
                  "Must specify either both inner and outer boundary conditions "
                  "or neither.");
    }
  } else {
    if (is_none(outer_boundary_condition_)) {
      PARSE_ERROR(
          context,
          "None boundary condition is not supported. If you would like an "
          "outflow-type boundary condition, you must use that.");
    }
    if (is_periodic(outer_boundary_condition_)) {
      PARSE_ERROR(context,
                  "Cannot have periodic boundary conditions with "
                  "NonconformingSphericalShells");
    }
  }

  // Build block names and groups
  block_names_.reserve(num_blocks);
  for (size_t layer = 0; layer < num_wedge_layers_; ++layer) {
    const std::string group_name = "WedgedShell" + std::to_string(layer);
    for (size_t wedge = 0; wedge < 6; ++wedge) {
      const std::string name =
          "Wedge" + std::to_string(layer * 6 + wedge);
      block_names_.emplace_back(name);
      block_groups_[group_name].insert(name);
    }
  }
  if (fill_interior_) {
    block_names_.emplace_back("InnerCube");
  }
  for (size_t i = 0; i < num_shells_; ++i) {
    const std::string name = "Shell" + std::to_string(i);
    block_names_.emplace_back(name);
    block_groups_[name].insert(name);
  }
}

Domain<3> NonconformingSphericalShells::create_domain() const {
  const size_t num_wedge_blocks = 6 * num_wedge_layers_;
  const size_t num_blocks =
      num_wedge_blocks + (fill_interior_ ? 1 : 0) + num_shells_;
  std::vector<Block<3>> blocks;
  blocks.reserve(num_blocks);

  const std::vector<std::array<size_t, 8>> corners =
      corners_for_radially_layered_domains(num_wedge_layers_, fill_interior_);
  std::vector<DirectionMap<3, BlockNeighbors<3>>> neighbors_of_all_blocks{};
  set_internal_boundaries<3>(make_not_null(&neighbors_of_all_blocks), corners);

  // Set up nonconforming shell neighbors.
  // The outermost wedge layer connects to the innermost shell.
  const size_t outermost_wedge_start = 6 * (num_wedge_layers_ - 1);
  const size_t first_shell_id =
      num_wedge_blocks + (fill_interior_ ? 1 : 0);
  const OrientationMap<3> shell_to_wedge{
      {{Direction<3>::upper_zeta(), Direction<3>::self(),
        Direction<3>::self()}}};

  // Innermost shell's lower_xi neighbor = outermost wedge layer
  DirectionMap<3, BlockNeighbors<3>> neighbors_of_innermost_shell{};
  {
    std::unordered_set<size_t> wedge_ids;
    std::unordered_map<size_t, OrientationMap<3>> orientations;
    for (size_t i = 0; i < 6; ++i) {
      const size_t wedge_id = outermost_wedge_start + i;
      wedge_ids.insert(wedge_id);
      orientations.emplace(wedge_id, shell_to_wedge);
    }
    neighbors_of_innermost_shell.emplace(std::pair{
        Direction<3>::lower_xi(),
        BlockNeighbors<3>{std::move(wedge_ids), std::move(orientations),
                          false}});
  }

  // Outermost wedge layer's upper_zeta neighbor = innermost shell
  for (size_t i = 0; i < 6; ++i) {
    neighbors_of_all_blocks[outermost_wedge_start + i].emplace(std::pair{
        Direction<3>::upper_zeta(),
        BlockNeighbors<3>{
            {first_shell_id},
            {{first_shell_id, shell_to_wedge.inverse_map()}},
            false}});
  }

  // Create wedge coordinate maps
  const double inner_sphericity =
      fill_interior_ ? std::get<InnerCube>(interior_).sphericity : 1.0;
  auto wedge_coord_maps =
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial, 3>(
          sph_wedge_coordinate_maps(
              inner_radius_, interface_radius_, inner_sphericity, 1.0,
              use_equiangular_map_, std::nullopt, false,
              wedges_radial_partitioning_,
              std::vector<CoordinateMaps::Distribution>(
                  num_wedge_layers_, CoordinateMaps::Distribution::Linear)));

  // Build wedge blocks
  for (size_t i = 0; i < num_wedge_blocks; ++i) {
    blocks.emplace_back(
        std::move(wedge_coord_maps[i]), i,
        std::move(neighbors_of_all_blocks[i]),
        block_names_.at(i), domain::topologies::hypercube<3>);
  }

  // Build inner cube block (if filled)
  if (fill_interior_) {
    const size_t cube_id = num_wedge_blocks;
    const double inner_cube_sphericity =
        std::get<InnerCube>(interior_).sphericity;
    std::unique_ptr<
        CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>
        inner_cube_map;
    if (inner_cube_sphericity == 0.0) {
      if (use_equiangular_map_) {
        inner_cube_map =
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                Equiangular3D{
                    Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0)),
                    Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0)),
                    Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0))});
      } else {
        inner_cube_map =
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0)),
                         Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0)),
                         Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0))});
      }
    } else {
      inner_cube_map =
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              BulgedCube{inner_radius_, inner_cube_sphericity,
                         use_equiangular_map_});
    }
    blocks.emplace_back(std::move(inner_cube_map), cube_id,
                        std::move(neighbors_of_all_blocks[cube_id]),
                        block_names_.at(cube_id),
                        domain::topologies::hypercube<3>);
  }

  // Build shell blocks
  const auto aligned = OrientationMap<3>::create_aligned();
  for (size_t i = 0; i < num_shells_; ++i) {
    const size_t shell_id = first_shell_id + i;
    const double shell_inner =
        (i == 0) ? interface_radius_ : shells_radial_partitioning_[i - 1];
    const double shell_outer =
        (i == num_shells_ - 1) ? outer_radius_
                               : shells_radial_partitioning_[i];
    auto shell_map =
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                           CoordinateMaps::Identity<2>>{
                CoordinateMaps::Affine{-1.0, 1.0, shell_inner, shell_outer},
                CoordinateMaps::Identity<2>{}},
            CoordinateMaps::SphericalToCartesianPfaffian{});

    DirectionMap<3, BlockNeighbors<3>> neighbors;
    if (i == 0) {
      neighbors = std::move(neighbors_of_innermost_shell);
    } else {
      // Connect to previous shell conformingly
      neighbors.emplace(std::pair{Direction<3>::lower_xi(),
                                  BlockNeighbors<3>{shell_id - 1, aligned}});
    }
    if (i < num_shells_ - 1) {
      // Connect to next shell conformingly
      neighbors.emplace(std::pair{Direction<3>::upper_xi(),
                                  BlockNeighbors<3>{shell_id + 1, aligned}});
    }

    blocks.emplace_back(std::move(shell_map), shell_id, std::move(neighbors),
                        block_names_.at(shell_id),
                        domain::topologies::spherical_shell);
  }

  return Domain(std::move(blocks), {}, block_groups_);
}

std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
NonconformingSphericalShells::grid_anchors() const {
  return grid_anchors_;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
NonconformingSphericalShells::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }
  const size_t num_wedge_blocks = 6 * num_wedge_layers_;
  const size_t num_blocks =
      num_wedge_blocks + (fill_interior_ ? 1 : 0) + num_shells_;
  const size_t last_shell_index = num_blocks - 1;
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{num_blocks};
  if (not fill_interior_) {
    const auto& inner_bc =
        std::get<Excision>(interior_).boundary_condition;
    // Inner BC on lower_zeta of the 6 innermost wedges (indices 0-5)
    for (size_t i = 0; i < 6; ++i) {
      boundary_conditions[i][Direction<3>::lower_zeta()] =
          inner_bc->get_clone();
    }
  }
  // Outer BC on upper_xi of the outermost shell
  boundary_conditions[last_shell_index][Direction<3>::upper_xi()] =
      outer_boundary_condition_->get_clone();
  return boundary_conditions;
}

std::vector<std::string> NonconformingSphericalShells::block_names() const {
  return block_names_;
}

std::unordered_map<std::string, std::unordered_set<std::string>>
NonconformingSphericalShells::block_groups() const {
  return block_groups_;
}

std::vector<std::array<size_t, 3>>
NonconformingSphericalShells::initial_extents() const {
  return initial_number_of_grid_points_;
}

std::vector<std::array<size_t, 3>>
NonconformingSphericalShells::initial_refinement_levels() const {
  return initial_refinement_levels_;
}
}  // namespace domain::creators
