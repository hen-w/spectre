// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/OptionTags.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell {
namespace OptionTags {
struct ActiveGrid {
  using type = evolution::dg::subcell::ActiveGrid;
  static constexpr Options::String help = {
      "The type of the active grid. Either 'Dg' or 'Subcell'."};
  using group = SpatialDiscretization::OptionTags::SpatialDiscretizationGroup;
};
}  // namespace OptionTags
namespace Tags {

/// The grid currently used for the DG-subcell evolution on the element.
struct ActiveGrid : db::SimpleTag {
  using type = evolution::dg::subcell::ActiveGrid;
  using option_tags =
      tmpl::list<evolution::dg::subcell::OptionTags::ActiveGrid>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type option) { return option; }
};

/*!
 * \brief The active grid of every neighbor of the element, keyed by the
 * neighbor's `DirectionalId`, as selected at initialization by
 * `Actions::SetSubcellGrid` on the neighbor.
 *
 * \details Set by `Actions::SetNeighborActiveGrids` from the initial domain
 * topology, and valid only while the topology and the active grids are fixed
 * (no AMR, no troubled-cell indicator), see
 * `evolution::dg::active_grids_fixed_at_initialization`. Unlike a
 * `DirectionalIdMap`, the map holds any number of neighbors, as a face with
 * several nonconforming neighbors may have.
 */
template <size_t Dim>
struct NeighborActiveGrids : db::SimpleTag {
  using type = std::unordered_map<DirectionalId<Dim>,
                                  evolution::dg::subcell::ActiveGrid>;
};
}  // namespace Tags
}  // namespace evolution::dg::subcell
