// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

/// \cond
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class ElementId;
template <size_t VolumeDim, typename IdType>
class Neighbors;
/// \endcond

namespace evolution::dg {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief A consistent Upper/Lower labeling of the two elements that share a
 * mortar, used to construct single-valued one-sided numerical fluxes.
 *
 * Some numerical fluxes (for example the alternating fluxes of a local
 * discontinuous Galerkin scheme) take their trace entirely from one fixed side
 * of each interior interface. To make such a scheme conservative the two
 * elements that abut an interface must agree on which of them is the "Upper"
 * element and which is the "Lower" element, and they must arrive at opposite
 * conclusions so that the single-valued flux each of them computes is the same.
 * `InterfaceOrientation` records, from the point of view of the interior
 * element of a mortar, whether that interior element is the Upper or the Lower
 * element of the interface.
 *
 * - `InteriorIsUpper`: the interior element sits on the `Side::Upper` side of
 *   the interface.
 * - `InteriorIsLower`: the interior element sits on the `Side::Lower` side of
 *   the interface.
 * - `ExternalBoundary`: the face has no neighbor and boundary conditions supply
 *   the exterior data. `interface_orientation` never returns this value; it is
 *   provided so that external-boundary call sites can carry an orientation.
 */
enum class InterfaceOrientation {
  InteriorIsUpper,
  InteriorIsLower,
  ExternalBoundary
};

/// Output operator for InterfaceOrientation.
inline std::ostream& operator<<(std::ostream& os,
                                const InterfaceOrientation orientation) {
  switch (orientation) {
    case InterfaceOrientation::InteriorIsUpper:
      return os << "InteriorIsUpper";
    case InterfaceOrientation::InteriorIsLower:
      return os << "InteriorIsLower";
    case InterfaceOrientation::ExternalBoundary:
      return os << "ExternalBoundary";
    default:
      ERROR("Unknown InterfaceOrientation.");
  }
}

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Label the interior element of a mortar as the Upper or Lower element
 * of the shared interface.
 *
 * Determines whether the element `element_id`, looking across the interface in
 * `direction` at its neighbor `neighbor_id`, is the Upper or the Lower element
 * of that interface. The result is used to build single-valued one-sided
 * numerical fluxes (see `InterfaceOrientation`), so it is guaranteed to be
 * antisymmetric: the two elements that share an interface always compute
 * opposite values (one `InteriorIsUpper`, the other `InteriorIsLower`).
 *
 * The interior element's face lies on `direction.side()`. The neighbor's face
 * toward the same interface lies on `orientation_map(direction.opposite())`,
 * where `orientation_map` is the neighbor's `OrientationMap` obtained from
 * `neighbors_in_direction` (the map takes objects in the interior element's
 * logical frame to the neighbor's logical frame; the idiom
 * `orientation_map(direction.opposite())` is the neighbor's face direction
 * toward this interface). Call these two sides `side_me` and `side_nb`.
 *
 * - If `side_me != side_nb` the two blocks agree on a common Upper/Lower
 *   labeling of the interface (the aligned case, which includes a
 *   trivial-orientation periodic self-neighbor). The interior element sits on
 *   `opposite(side_me)`, so the result is `InteriorIsUpper` when
 *   `side_me == Side::Lower` and `InteriorIsLower` when `side_me ==
 *   Side::Upper`.
 * - A mortar with multiple non-conforming neighbors is labeled by the host
 *   element's own id and aggregates every neighbor on the face. The neighbor
 *   side is then determined from all the neighbor orientations across the
 *   direction, which must agree; disagreement is an error. Single-neighbor
 *   mortars reduce to their one orientation entry, so their behavior is
 *   unchanged.
 * - If `side_me == side_nb` the interface is twisted by an
 *   orientation-reversing discrete rotation between the two blocks and no
 *   consistent Upper/Lower labeling of the interface exists. The label is then
 *   broken by the tie-break `element_id < neighbor_id`: the smaller id is the
 *   Upper element. This choice is arbitrary-but-fixed; it is antisymmetric by
 *   construction because
 *   `element_id < neighbor_id` and `neighbor_id < element_id` cannot both hold.
 *
 * \returns `InteriorIsUpper` or `InteriorIsLower`; never `ExternalBoundary`.
 *
 * It is an error for the twisted case to occur with `element_id ==
 * neighbor_id` (a periodic self-neighbor whose orientation reverses the
 * interface): no antisymmetric labeling of a single element against itself
 * exists, and this pathological configuration is not supported.
 */
template <size_t Dim>
InterfaceOrientation interface_orientation(
    const ElementId<Dim>& element_id, const Direction<Dim>& direction,
    const Neighbors<Dim, ElementId<Dim>>& neighbors_in_direction,
    const ElementId<Dim>& neighbor_id);
}  // namespace evolution::dg
