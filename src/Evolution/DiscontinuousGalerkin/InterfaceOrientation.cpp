// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/InterfaceOrientation.hpp"

#include <cstddef>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg {
template <size_t Dim>
InterfaceOrientation interface_orientation(
    const ElementId<Dim>& element_id, const Direction<Dim>& direction,
    const Neighbors<Dim, ElementId<Dim>>& neighbors_in_direction,
    const ElementId<Dim>& neighbor_id) {
  const Side side_me = direction.side();
  const OrientationMap<Dim>& orientation_map =
      neighbors_in_direction.orientation(neighbor_id);
  const Side side_nb = orientation_map(direction.opposite()).side();
  if (side_me != side_nb) {
    return side_me == Side::Lower ? InterfaceOrientation::InteriorIsUpper
                                  : InterfaceOrientation::InteriorIsLower;
  }
  if (element_id == neighbor_id) {
    ERROR(
        "interface_orientation cannot label a periodic self-neighbor whose "
        "orientation reverses the interface: the element "
        << element_id << " identifies with its neighbor across " << direction
        << " but the interface is twisted (both faces are on the same side "
        << side_me
        << "). No antisymmetric Upper/Lower labeling of an element against "
           "itself exists, so this configuration is not supported.");
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
