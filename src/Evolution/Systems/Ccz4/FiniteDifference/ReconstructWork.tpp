// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {

template <typename TagsList, typename Reconstructor>
void reconstruct_work(
    const gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_upper_face,
    const Reconstructor& reconstruct,
    const Variables<System::variables_tag_list>& volume_vars,
    const Element<3>& element,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& neighbor_data,
    const Mesh<3>& subcell_mesh, const size_t ghost_zone_size) {
  ASSERT(is_isotropic(subcell_mesh),
         "The subcell mesh should be isotropic but got " << subcell_mesh);

  const size_t volume_num_pts = subcell_mesh.number_of_grid_points();
  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();
  const size_t neighbor_num_pts =
      ghost_zone_size * subcell_mesh.extents().slice_away(0).product();

  // Ghost data is stored as Variables<variables_tag_list> (17 tags), but we
  // only reconstruct the 13 non-boundary tags.  We track the offset into
  // the neighbor data buffer via vars_in_neighbor_count, which counts
  // independent components of tags_list_for_reconstruct.
  size_t vars_in_neighbor_count = 0;
  tmpl::for_each<tags_list_for_reconstruct>([&](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;

    const auto& volume_tensor = get<tag>(volume_vars);
    const size_t number_of_variables = volume_tensor.size();

    const gsl::span<const double> volume_vars_span = gsl::make_span(
        volume_tensor[0].data(), number_of_variables * volume_num_pts);

    std::array<gsl::span<double>, 3> upper_face_vars{};
    std::array<gsl::span<double>, 3> lower_face_vars{};
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(upper_face_vars, i) =
          gsl::make_span(get<tag>(gsl::at(*vars_on_upper_face, i))[0].data(),
                         number_of_variables * reconstructed_num_pts);
      gsl::at(lower_face_vars, i) =
          gsl::make_span(get<tag>(gsl::at(*vars_on_lower_face, i))[0].data(),
                         number_of_variables * reconstructed_num_pts);
    }

    DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};

    for (const auto& direction : Direction<3>::all_directions()) {
      if (element.neighbors().contains(direction)) {
        const auto& neighbors_in_direction = element.neighbors().at(direction);

        ASSERT(neighbors_in_direction.size() == 1,
               "Currently only support one neighbor in each direction, but "
               "got "
                   << neighbors_in_direction.size() << " in direction "
                   << direction);

        const DataVector& neighbor_data_dv =
            neighbor_data
                .at(DirectionalId<3>{direction,
                                     *neighbors_in_direction.begin()})
                .neighbor_ghost_data_for_reconstruction();

        ASSERT(neighbor_data_dv.size() != 0,
               "The neighbor data is empty in direction "
                   << direction << " on element id " << element.id());

        ghost_cell_vars[direction] = gsl::make_span(
            &neighbor_data_dv[vars_in_neighbor_count * neighbor_num_pts],
            number_of_variables * neighbor_num_pts);
      } else {
        ASSERT(
            element.external_boundaries().count(direction) == 1,
            "Element has neither neighbor nor external boundary to direction: "
                << direction);

        const DataVector& neighbor_data_dv =
            neighbor_data
                .at(DirectionalId<3>{direction,
                                     ElementId<3>::external_boundary_id()})
                .neighbor_ghost_data_for_reconstruction();

        ghost_cell_vars[direction] = gsl::make_span(
            &neighbor_data_dv[0], number_of_variables * neighbor_num_pts);
      }
    }

    reconstruct(make_not_null(&upper_face_vars),
                make_not_null(&lower_face_vars), volume_vars_span,
                ghost_cell_vars, subcell_mesh.extents(), number_of_variables);

    vars_in_neighbor_count += number_of_variables;
  });
}

template <typename TagsList, typename ReconstructLower,
          typename ReconstructUpper>
void reconstruct_fd_neighbor_work(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const ReconstructLower& reconstruct_lower_neighbor,
    const ReconstructUpper& reconstruct_upper_neighbor,
    const Variables<System::variables_tag_list>& subcell_volume_vars,
    const Element<3>& element,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<3>& subcell_mesh,
    const Direction<3>& direction_to_reconstruct,
    const size_t ghost_zone_size) {
  const DirectionalId<3> mortar_id{
      direction_to_reconstruct,
      *element.neighbors().at(direction_to_reconstruct).begin()};

  Index<3> ghost_data_extents = subcell_mesh.extents();
  ghost_data_extents[direction_to_reconstruct.dimension()] = ghost_zone_size;

  // Ghost data buffer is Variables<variables_tag_list> (17 tags).  We
  // interpret only the first 13 non-boundary tags via set_data_ref.
  Variables<tags_list_for_reconstruct> neighbor_vars{};
  {
    ASSERT(ghost_data.contains(mortar_id),
           "The neighbor data does not contain the mortar: " << mortar_id);
    const DataVector& neighbor_data_on_mortar =
        ghost_data.at(mortar_id).neighbor_ghost_data_for_reconstruction();
    neighbor_vars.set_data_ref(
        const_cast<double*>(neighbor_data_on_mortar.data()),
        neighbor_vars.number_of_independent_components *
            ghost_data_extents.product());
  }

  tmpl::for_each<tags_list_for_reconstruct>(
      [&direction_to_reconstruct, &ghost_data_extents, &neighbor_vars,
       &reconstruct_lower_neighbor, &reconstruct_upper_neighbor, &subcell_mesh,
       &subcell_volume_vars, &vars_on_face](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;

        const auto& volume_tensor = get<tag>(subcell_volume_vars);
        const auto& tensor_neighbor = get<tag>(neighbor_vars);
        auto& tensor_on_face = get<tag>(*vars_on_face);

        if (direction_to_reconstruct.side() == Side::Upper) {
          for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
               ++tensor_index) {
            reconstruct_upper_neighbor(
                make_not_null(&tensor_on_face[tensor_index]),
                volume_tensor[tensor_index], tensor_neighbor[tensor_index],
                subcell_mesh.extents(), ghost_data_extents,
                direction_to_reconstruct);
          }
        } else {
          for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
               ++tensor_index) {
            reconstruct_lower_neighbor(
                make_not_null(&tensor_on_face[tensor_index]),
                volume_tensor[tensor_index], tensor_neighbor[tensor_index],
                subcell_mesh.extents(), ghost_data_extents,
                direction_to_reconstruct);
          }
        }
      });
}

}  // namespace Ccz4::fd
