// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/AsAccess.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/Ccz4/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Ccz4/BoundaryCorrections/LaxFriedrichs.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/NeighborPackagedData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UnlimitedDeg4Prim.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

using system = Ccz4::fd::System;
using variables_tag_list = system::variables_tag_list;
using recons_tags = Ccz4::fd::tags_list_for_reconstruct;
using GhostData = evolution::dg::subcell::GhostData;

DirectionalIdMap<3, GhostData> compute_ghost_data(
    const Mesh<3>& subcell_mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& /*volume_logical_coords*/,
    const DirectionMap<3, Neighbors<3>>& neighbors,
    const size_t ghost_zone_size,
    const gsl::not_null<std::mt19937*> gen,
    const gsl::not_null<std::uniform_real_distribution<>*> dist) {
  DirectionalIdMap<3, GhostData> ghost_data{};

  for (const auto& [direction, neighbors_in_direction] : neighbors) {
    REQUIRE(neighbors_in_direction.size() == 1);
    const ElementId<3>& neighbor_id = *neighbors_in_direction.begin();

    // Create random Variables<variables_tag_list> for this neighbor
    auto neighbor_vars =
        make_with_random_values<Variables<variables_tag_list>>(
            gen, dist, subcell_mesh.number_of_grid_points());

    // Slice ghost zone data from the neighbor
    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars.data(), neighbor_vars.size()),
        subcell_mesh.extents(), ghost_zone_size,
        std::unordered_set{direction.opposite()}, 0, {});

    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));

    ghost_data[DirectionalId<3>{direction, neighbor_id}] = GhostData{1};
    ghost_data.at(DirectionalId<3>{direction, neighbor_id})
        .neighbor_ghost_data_for_reconstruction() =
        sliced_data.at(direction.opposite());
  }
  return ghost_data;
}

template <bool IsAuxiliary>
void test_neighbor_packaged_data(const gsl::not_null<std::mt19937*> gen) {
  using LaxFriedrichs = Ccz4::BoundaryCorrections::LaxFriedrichs<3>;
  using Reconstructor = Ccz4::fd::UnlimitedDeg4Prim;

  std::uniform_real_distribution<> dist(-1.0, 1.0);

  // Create element with neighbors in all 6 directions
  DirectionMap<3, Neighbors<3>> element_neighbors{};
  for (size_t i = 0; i < 6; ++i) {
    element_neighbors[gsl::at(Direction<3>::all_directions(), i)] =
        Neighbors<3>{{ElementId<3>{i + 1, {}}},
                     OrientationMap<3>::create_aligned()};
  }
  const Element<3> element{ElementId<3>{0, {}}, element_neighbors};

  // Meshes
  const size_t num_dg_pts_per_dimension = 5;
  const Mesh<3> dg_mesh{num_dg_pts_per_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  // Random DG volume data
  auto dg_evolved_vars = make_with_random_values<Variables<variables_tag_list>>(
      gen, make_not_null(&dist), dg_mesh.number_of_grid_points());

  // Project to subcell for independent computation
  const auto volume_vars_subcell = evolution::dg::subcell::fd::project(
      dg_evolved_vars, dg_mesh, subcell_mesh.extents());

  // Ghost data
  const Reconstructor reconstructor{};
  const auto subcell_logical_coords = logical_coordinates(subcell_mesh);
  auto ghost_data = compute_ghost_data(subcell_mesh, subcell_logical_coords,
                                       element.neighbors(),
                                       reconstructor.ghost_zone_size(), gen,
                                       make_not_null(&dist));

  // Normal covectors on DG faces (identity map: unit normal = ±e_dim)
  DirectionMap<3, std::optional<Variables<
                      tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<3>>>>>
      normal_vectors{};
  for (const auto& direction : Direction<3>::all_directions()) {
    const Mesh<2> face_mesh = dg_mesh.slice_away(direction.dimension());
    const size_t num_face_pts = face_mesh.number_of_grid_points();
    Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                         evolution::dg::Tags::NormalCovector<3>>>
        face_data{num_face_pts};
    get(get<evolution::dg::Tags::MagnitudeOfNormal>(face_data)) = 1.0;
    auto& normal = get<evolution::dg::Tags::NormalCovector<3>>(face_data);
    for (size_t i = 0; i < 3; ++i) {
      normal.get(i) = 0.0;
    }
    normal.get(direction.dimension()) =
        static_cast<double>(direction.sign());
    normal_vectors[direction] = std::move(face_data);
  }

  // Build the DataBox
  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Element<3>, domain::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Mesh<3>, system::variables_tag,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
      Ccz4::fd::Tags::Reconstructor, evolution::Tags::BoundaryCorrection,
      domain::Tags::MeshVelocity<3>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<3>,
      evolution::dg::subcell::Tags::SubcellOptions<3>>>(
      element, dg_mesh, subcell_mesh, dg_evolved_vars, ghost_data,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<Reconstructor>()},
      std::unique_ptr<evolution::BoundaryCorrection>{
          std::make_unique<LaxFriedrichs>()},
      std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>{},
      normal_vectors,
      evolution::dg::subcell::SubcellOptions{
          4.0, 1_st, 1.0e-3, 1.0e-4, false, false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
          std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1});

  // Build mortar list
  std::vector<DirectionalId<3>> mortars_to_reconstruct_to{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    mortars_to_reconstruct_to.emplace_back(
        DirectionalId<3>{direction, *neighbors.begin()});
  }

  // Call the function under test
  const auto packaged_data =
      Ccz4::fd::NeighborPackagedDataImpl<IsAuxiliary>::apply(
          db::as_access(box), mortars_to_reconstruct_to);

  // Determine expected packaged data size per face point
  using pkg_field_tags = tmpl::conditional_t<
      IsAuxiliary, typename LaxFriedrichs::dg_auxiliary_package_field_tags,
      typename LaxFriedrichs::dg_package_field_tags>;
  using dg_package_data_projected_tags = variables_tag_list;

  LaxFriedrichs boundary_corr{};

  // Independently compute expected values for each mortar and compare
  for (const auto& mortar_id : mortars_to_reconstruct_to) {
    const Direction<3>& direction = mortar_id.direction();
    const size_t dim = direction.dimension();

    const size_t num_face_pts =
        subcell_mesh.extents().slice_away(dim).product();

    // 1. Reconstruct to FD face
    Variables<variables_tag_list> vars_on_face{num_face_pts, 0.0};
    auto recons_face =
        vars_on_face.template reference_subset<recons_tags>();
    reconstructor.reconstruct_fd_neighbor(
        make_not_null(&recons_face), volume_vars_subcell, element, ghost_data,
        subcell_mesh, direction);

    // 2. Negate normal covector and project to FD face
    tnsr::i<DataVector, 3, Frame::Inertial> normal_covector =
        get<evolution::dg::Tags::NormalCovector<3>>(
            *normal_vectors.at(direction));
    for (auto& t : normal_covector) {
      t *= -1.0;
    }
    const auto dg_normal_covector = normal_covector;
    for (size_t i = 0; i < 3; ++i) {
      normal_covector.get(i) = evolution::dg::subcell::fd::project(
          dg_normal_covector.get(i), dg_mesh.slice_away(dim),
          subcell_mesh.extents().slice_away(dim));
    }

    // 3. Package data on FD face
    Variables<pkg_field_tags> expected_packaged{num_face_pts, 0.0};
    if constexpr (IsAuxiliary) {
      evolution::dg::Actions::detail::dg_auxiliary_package_data<system>(
          make_not_null(&expected_packaged), boundary_corr, vars_on_face,
          normal_covector, {std::nullopt}, direction,
          dg_package_data_projected_tags{});
    } else {
      evolution::dg::Actions::detail::dg_package_data<system>(
          make_not_null(&expected_packaged), boundary_corr, vars_on_face,
          normal_covector, {std::nullopt}, direction,
          dg_package_data_projected_tags{});
    }

    // 4. Interpolate to DG face
    auto expected_dg_packaged = evolution::dg::subcell::fd::reconstruct(
        expected_packaged, dg_mesh.slice_away(dim),
        subcell_mesh.extents().slice_away(dim),
        evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);

    const DataVector expected_dv{expected_dg_packaged.data(),
                                 expected_dg_packaged.size()};

    REQUIRE(packaged_data.contains(mortar_id));
    CHECK_ITERABLE_APPROX(expected_dv, packaged_data.at(mortar_id));
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.NeighborPackagedData",
                  "[Unit][Evolution]") {
  // Test empty-mortars early return
  {
    auto box = db::create<db::AddSimpleTags<>>();
    const std::vector<DirectionalId<3>> empty_mortars{};
    const DirectionalIdMap<3, DataVector> result =
        Ccz4::fd::NeighborPackagedData::apply(db::as_access(box),
                                              empty_mortars);
    CHECK(result.empty());
  }

  // Test with actual data
  MAKE_GENERATOR(gen);
  {
    INFO("Physical pass (IsAuxiliary=false)");
    test_neighbor_packaged_data<false>(make_not_null(&gen));
  }
  {
    INFO("Auxiliary pass (IsAuxiliary=true)");
    test_neighbor_packaged_data<true>(make_not_null(&gen));
  }
}
}  // namespace
