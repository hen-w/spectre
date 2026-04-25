// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <unordered_set>

#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UnlimitedDeg4Prim.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
namespace {

using GhostData = evolution::dg::subcell::GhostData;

// Compute ghost data by slicing neighbor volume data to the ghost zone region.
// Follows the same pattern as TestHelpers::ForceFree::fd::compute_ghost_data.
template <typename F>
DirectionalIdMap<3, GhostData> compute_ghost_data(
    const Mesh<3>& subcell_mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& volume_logical_coords,
    const DirectionMap<3, Neighbors<3>>& neighbors,
    const size_t ghost_zone_size, const F& compute_variables) {
  DirectionalIdMap<3, GhostData> ghost_data{};

  for (const auto& [direction, neighbors_in_direction] : neighbors) {
    REQUIRE(neighbors_in_direction.size() == 1);
    const ElementId<3>& neighbor_id = *neighbors_in_direction.begin();

    // Shift logical coordinates to neighbor element
    auto neighbor_logical_coords = volume_logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    const auto neighbor_vars = compute_variables(neighbor_logical_coords);

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

// Fill Variables<variables_tag_list> with a linear function of coordinates.
// Only fills tags_list_for_reconstruct (13 tags); boundary second-order tags
// stay zero.  Linear data is exact for the degree-4 unlimited reconstructor.
Variables<System::variables_tag_list> linear_solution(
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& coords) {
  Variables<System::variables_tag_list> vars{get<0>(coords).size(), 0.0};

  size_t tag_counter = 0;
  tmpl::for_each<tags_list_for_reconstruct>([&](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    auto& tensor = get<tag>(vars);
    for (size_t comp = 0; comp < tensor.size(); ++comp) {
      const double offset = 1.0 + 0.1 * static_cast<double>(tag_counter + comp);
      tensor[comp] = offset;
      for (size_t d = 0; d < 3; ++d) {
        tensor[comp] +=
            (0.5 + 0.05 * static_cast<double>(tag_counter + comp + d)) *
            coords.get(d);
      }
    }
    tag_counter += tensor.size();
  });
  return vars;
}

void test_reconstruct() {
  const size_t points_per_dimension = 7;
  const Ccz4::fd::UnlimitedDeg4Prim reconstructor{};

  // Create element with neighbors in all 6 directions
  DirectionMap<3, Neighbors<3>> neighbors{};
  for (size_t i = 0; i < 6; ++i) {
    neighbors[gsl::at(Direction<3>::all_directions(), i)] = Neighbors<3>{
        {ElementId<3>{i + 1, {}}}, OrientationMap<3>::create_aligned()};
  }
  const Element<3> element{ElementId<3>{0, {}}, neighbors};

  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(subcell_mesh);
  // Offset coordinates so each direction is distinguishable
  for (size_t i = 1; i < 3; ++i) {
    logical_coords.get(i) += 4.0 * static_cast<double>(i);
  }

  const Variables<System::variables_tag_list> volume_vars =
      linear_solution(logical_coords);

  const DirectionalIdMap<3, GhostData> ghost_data = compute_ghost_data(
      subcell_mesh, logical_coords, element.neighbors(),
      reconstructor.ghost_zone_size(), linear_solution);

  // --- Test reconstruct() ---
  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();
  using recons_tags = tags_list_for_reconstruct;

  std::array<Variables<recons_tags>, 3> vars_on_lower_face =
      make_array<3>(Variables<recons_tags>(reconstructed_num_pts));
  std::array<Variables<recons_tags>, 3> vars_on_upper_face =
      make_array<3>(Variables<recons_tags>(reconstructed_num_pts));

  reconstructor.reconstruct(make_not_null(&vars_on_lower_face),
                            make_not_null(&vars_on_upper_face), volume_vars,
                            element, ghost_data, subcell_mesh);

  for (size_t dim = 0; dim < 3; ++dim) {
    CAPTURE(dim);

    // Face-centered coordinates
    const auto basis = make_array<3>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<3>(Spectral::Quadrature::CellCentered);
    auto extents = make_array<3>(points_per_dimension);
    gsl::at(extents, dim) = points_per_dimension + 1;
    gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
    const Mesh<3> face_centered_mesh{extents, basis, quadrature};
    auto face_coords = logical_coordinates(face_centered_mesh);
    for (size_t i = 1; i < 3; ++i) {
      face_coords.get(i) += 4.0 * static_cast<double>(i);
    }

    // Expected face values from the analytic linear solution
    const auto full_expected = linear_solution(face_coords);

    tmpl::for_each<recons_tags>(
        [dim, &full_expected, &vars_on_lower_face,
         &vars_on_upper_face](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          CAPTURE(db::tag_name<tag>());
          CHECK_ITERABLE_APPROX(
              get<tag>(gsl::at(vars_on_lower_face, dim)),
              get<tag>(full_expected));
          CHECK_ITERABLE_APPROX(
              get<tag>(gsl::at(vars_on_upper_face, dim)),
              get<tag>(full_expected));
        });

    // --- Test reconstruct_fd_neighbor() ---
    const size_t num_pts_on_mortar =
        face_centered_mesh.slice_away(dim).number_of_grid_points();

    Variables<recons_tags> upper_mortar{num_pts_on_mortar};
    reconstructor.reconstruct_fd_neighbor(
        make_not_null(&upper_mortar), volume_vars, element, ghost_data,
        subcell_mesh, Direction<3>{dim, Side::Upper});

    Variables<recons_tags> lower_mortar{num_pts_on_mortar};
    reconstructor.reconstruct_fd_neighbor(
        make_not_null(&lower_mortar), volume_vars, element, ghost_data,
        subcell_mesh, Direction<3>{dim, Side::Lower});

    tmpl::for_each<recons_tags>(
        [dim, &full_expected, &lower_mortar, &upper_mortar,
         &face_centered_mesh](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          CAPTURE(db::tag_name<tag>());
          CHECK_ITERABLE_APPROX(
              get<tag>(lower_mortar),
              data_on_slice(get<tag>(full_expected),
                            face_centered_mesh.extents(), dim, 0));
          CHECK_ITERABLE_APPROX(
              get<tag>(upper_mortar),
              data_on_slice(get<tag>(full_expected),
                            face_centered_mesh.extents(), dim,
                            face_centered_mesh.extents(dim) - 1));
        });
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.UnlimitedDeg4Prim",
                  "[Unit][Evolution]") {
  const auto recons_from_options_base =
      TestHelpers::test_factory_creation<Ccz4::fd::Reconstructor,
                                         Ccz4::fd::OptionTags::Reconstructor>(
          "UnlimitedDeg4Prim:\n");
  auto* const recons_from_options =
      dynamic_cast<const Ccz4::fd::UnlimitedDeg4Prim*>(
          recons_from_options_base.get());
  REQUIRE(recons_from_options != nullptr);
  CHECK(recons_from_options->ghost_zone_size() == 3);

  test_reconstruct();
}

}  // namespace
}  // namespace Ccz4::fd
