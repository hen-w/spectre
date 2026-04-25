// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Direction;
template <size_t Dim>
class Element;
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
namespace gsl {
template <typename>
class not_null;
}  // namespace gsl
/// \endcond

namespace Ccz4::fd {

/// Tags reconstructed to faces: 9 original evolved + 4 auxiliary.
/// Boundary second-order tags are not reconstructed (always zero at internal
/// interfaces).
using tags_list_for_reconstruct =
    tmpl::append<System::original_evolved_variables_tags,
                 System::auxiliary_variables_tags>;

template <typename TagsList, typename Reconstructor>
void reconstruct_work(
    gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_lower_face,
    gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_upper_face,
    const Reconstructor& reconstruct,
    const Variables<System::variables_tag_list>& volume_vars,
    const Element<3>& element,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<3>& subcell_mesh, size_t ghost_zone_size);

template <typename TagsList, typename ReconstructLower,
          typename ReconstructUpper>
void reconstruct_fd_neighbor_work(
    gsl::not_null<Variables<TagsList>*> vars_on_face,
    const ReconstructLower& reconstruct_lower_neighbor,
    const ReconstructUpper& reconstruct_upper_neighbor,
    const Variables<System::variables_tag_list>& subcell_volume_vars,
    const Element<3>& element,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<3>& subcell_mesh,
    const Direction<3>& direction_to_reconstruct,
    size_t ghost_zone_size);

}  // namespace Ccz4::fd
