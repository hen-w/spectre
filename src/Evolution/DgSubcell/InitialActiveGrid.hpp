// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/ExpectsAndEnsures.hpp"

/// \cond
template <size_t Dim>
class Block;
namespace evolution::dg::subcell {
class SubcellOptions;
}  // namespace evolution::dg::subcell
/// \endcond

namespace evolution::dg::subcell {
/*!
 * \brief The active grid that `Actions::SetSubcellGrid` selects for `element`
 * at initialization.
 *
 * \details The element uses finite differences (`ActiveGrid::Subcell`) unless
 * it is in a DG-only block (`SubcellOptions::only_dg_block_ids()`), the first
 * neighbor in one of its directions is in a DG-only block, or it has external
 * boundaries and `subcell_enabled_at_external_boundary` is `false`.
 */
template <size_t Dim>
ActiveGrid initial_active_grid(const Element<Dim>& element,
                               const SubcellOptions& subcell_options,
                               bool subcell_enabled_at_external_boundary);

/*!
 * \brief The active grid that `Actions::SetSubcellGrid` selects at
 * initialization for every neighbor of `element`, keyed by the neighbor's
 * `DirectionalId`.
 *
 * \details The `Element` of each neighbor is created from the initial domain
 * topology with `domain::create_initial_element`, as in
 * `evolution::dg::Initialization::Domain`, and passed to
 * `initial_active_grid`. No coordinates or maps are evaluated.
 */
template <size_t Dim>
std::unordered_map<DirectionalId<Dim>, ActiveGrid>
initial_neighbor_active_grids(
    const Element<Dim>& element, const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const SubcellOptions& subcell_options,
    bool subcell_enabled_at_external_boundary);

/*!
 * \brief Whether the neighbor `neighbor_id` of the element uses finite
 * differences, according to `Tags::NeighborActiveGrids`.
 *
 * \details It is an error if the neighbor has no entry.
 */
template <typename DbTagsList, size_t Dim>
bool neighbor_uses_subcell(const db::DataBox<DbTagsList>& box,
                           const DirectionalId<Dim>& neighbor_id) {
  const auto& neighbor_active_grids =
      db::get<Tags::NeighborActiveGrids<Dim>>(box);
  const auto neighbor_active_grid = neighbor_active_grids.find(neighbor_id);
  if (UNLIKELY(neighbor_active_grid == neighbor_active_grids.end())) {
    ERROR("Element " << db::get<::domain::Tags::Element<Dim>>(box).id()
                     << " has no active grid for its neighbor " << neighbor_id
                     << ". The active grids of the neighbors are set at "
                        "initialization by SetNeighborActiveGrids.");
  }
  return neighbor_active_grid->second == ActiveGrid::Subcell;
}
}  // namespace evolution::dg::subcell
