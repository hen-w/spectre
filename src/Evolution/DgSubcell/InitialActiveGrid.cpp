// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/InitialActiveGrid.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::subcell {
template <size_t Dim>
ActiveGrid initial_active_grid(
    const Element<Dim>& element, const SubcellOptions& subcell_options,
    const bool subcell_enabled_at_external_boundary) {
  // Loop over block neighbors and if neighbor id is inside of
  // subcell_options.only_dg_block_ids(), then bordering DG-only block
  const bool bordering_dg_block = alg::any_of(
      element.neighbors(),
      [&subcell_options](const auto& direction_and_neighbor) {
        const size_t first_block_id =
            direction_and_neighbor.second.ids().begin()->block_id();
        return alg::found(subcell_options.only_dg_block_ids(), first_block_id);
      });

  const bool subcell_allowed_in_element =
      not alg::found(subcell_options.only_dg_block_ids(),
                     element.id().block_id()) and
      not bordering_dg_block;
  const bool cell_is_not_on_external_boundary =
      element.external_boundaries().empty();

  return (cell_is_not_on_external_boundary or
          subcell_enabled_at_external_boundary) and
                 subcell_allowed_in_element
             ? ActiveGrid::Subcell
             : ActiveGrid::Dg;
}

template <size_t Dim>
std::unordered_map<DirectionalId<Dim>, ActiveGrid>
initial_neighbor_active_grids(
    const Element<Dim>& element, const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const SubcellOptions& subcell_options,
    const bool subcell_enabled_at_external_boundary) {
  std::unordered_map<DirectionalId<Dim>, ActiveGrid> result{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    for (const ElementId<Dim>& neighbor_id : neighbors) {
      result.emplace(
          DirectionalId<Dim>{direction, neighbor_id},
          initial_active_grid(
              domain::create_initial_element(neighbor_id, blocks,
                                             initial_refinement_levels),
              subcell_options, subcell_enabled_at_external_boundary));
    }
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template ActiveGrid initial_active_grid(                          \
      const Element<DIM(data)>& element,                            \
      const SubcellOptions& subcell_options,                        \
      bool subcell_enabled_at_external_boundary);                   \
  template std::unordered_map<DirectionalId<DIM(data)>, ActiveGrid> \
  initial_neighbor_active_grids(                                    \
      const Element<DIM(data)>& element,                            \
      const std::vector<Block<DIM(data)>>& blocks,                  \
      const std::vector<std::array<size_t, DIM(data)>>&             \
          initial_refinement_levels,                                \
      const SubcellOptions& subcell_options,                        \
      bool subcell_enabled_at_external_boundary);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::dg::subcell
