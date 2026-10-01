// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/InterfaceOrientation.hpp"

#include <cstddef>
#include <optional>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg {
template <size_t Dim>
InterfaceOrientation interface_orientation(
    const ElementId<Dim>& element_id, const Direction<Dim>& direction,
    const Neighbors<Dim, ElementId<Dim>>& neighbors_in_direction,
    const ElementId<Dim>& neighbor_id) {
  const Side side_me = direction.side();
  // A mortar with multiple non-conforming neighbors is labeled by the host
  // element's own id and aggregates every neighbor on this face, so the
  // neighbor side of the interface is determined from all the neighbor
  // orientations, which must agree. A single-neighbor mortar reduces to its
  // one orientation entry.
  const auto& orientations = neighbors_in_direction.orientations();
  ASSERT(not orientations.empty(), "No neighbor orientations across "
                                       << direction << " of element "
                                       << element_id);
  std::optional<Side> side_nb{};
  for (const auto& [neighbor_block_id, orientation_map] : orientations) {
    (void)neighbor_block_id;
    const Side side = orientation_map(direction.opposite()).side();
    if (side_nb.has_value() and side != side_nb.value()) {
      ERROR("interface_orientation requires all neighbors across "
            << direction << " of element " << element_id
            << " to lie on a single side of the interface, but the neighbor "
               "orientations disagree, so no single Upper/Lower labeling of "
               "this mortar exists.");
    }
    side_nb = side;
  }
  if (side_me != side_nb.value()) {
    return side_me == Side::Lower ? InterfaceOrientation::InteriorIsUpper
                                  : InterfaceOrientation::InteriorIsLower;
  }
  if (element_id == neighbor_id) {
    ERROR("interface_orientation cannot label the element "
          << element_id << " against its own id across " << direction
          << " on a twisted interface (both faces are on the same side "
          << side_me
          << "): this is either a periodic self-neighbor or a host-labeled "
             "mortar with multiple non-conforming neighbors, and no "
             "antisymmetric Upper/Lower labeling of an element against itself "
             "exists, so this configuration is not supported.");
  }
  return element_id < neighbor_id ? InterfaceOrientation::InteriorIsUpper
                                  : InterfaceOrientation::InteriorIsLower;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                            \
  template InterfaceOrientation interface_orientation(  \
      const ElementId<DIM(data)>& element_id,           \
      const Direction<DIM(data)>& direction,            \
      const Neighbors<DIM(data), ElementId<DIM(data)>>& \
          neighbors_in_direction,                       \
      const ElementId<DIM(data)>& neighbor_id);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::dg
