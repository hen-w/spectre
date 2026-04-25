// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/UnlimitedDeg4Prim.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ReconstructWork.tpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "NumericalAlgorithms/FiniteDifference/NeighborDataAsVariables.hpp"
#include "NumericalAlgorithms/FiniteDifference/Unlimited.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::fd {

UnlimitedDeg4Prim::UnlimitedDeg4Prim(CkMigrateMessage* const msg)
    : Reconstructor(msg) {}

std::unique_ptr<Reconstructor> UnlimitedDeg4Prim::get_clone() const {
  return std::make_unique<UnlimitedDeg4Prim>(*this);
}

void UnlimitedDeg4Prim::pup(PUP::er& p) { Reconstructor::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID UnlimitedDeg4Prim::my_PUP_ID = 0;

void UnlimitedDeg4Prim::reconstruct(
    const gsl::not_null<std::array<Variables<recons_tags>, dim>*>
        vars_on_lower_face,
    const gsl::not_null<std::array<Variables<recons_tags>, dim>*>
        vars_on_upper_face,
    const Variables<volume_vars_tags>& volume_vars,
    const Element<dim>& element,
    const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<dim>& subcell_mesh) const {
  reconstruct_work(
      vars_on_lower_face, vars_on_upper_face,
      [](auto upper_face_vars_ptr, auto lower_face_vars_ptr,
         const auto& volume_variables, const auto& ghost_cell_vars,
         const auto& subcell_extents, const size_t number_of_variables) {
        ::fd::reconstruction::unlimited<4, 3>(
            upper_face_vars_ptr, lower_face_vars_ptr, volume_variables,
            ghost_cell_vars, subcell_extents, number_of_variables);
      },
      volume_vars, element, ghost_data, subcell_mesh, ghost_zone_size());
}

void UnlimitedDeg4Prim::reconstruct_fd_neighbor(
    const gsl::not_null<Variables<recons_tags>*> vars_on_face,
    const Variables<volume_vars_tags>& subcell_volume_vars,
    const Element<dim>& element,
    const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<dim>& subcell_mesh,
    const Direction<dim> direction_to_reconstruct) const {
  reconstruct_fd_neighbor_work(
      vars_on_face,
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor, const auto& subcell_extents,
         const auto& ghost_data_extents,
         const auto& local_direction_to_reconstruct) {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Lower,
            ::fd::reconstruction::detail::UnlimitedReconstructor<4>>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      [](const auto tensor_component_on_face_ptr,
         const auto& tensor_component_volume,
         const auto& tensor_component_neighbor, const auto& subcell_extents,
         const auto& ghost_data_extents,
         const auto& local_direction_to_reconstruct) {
        ::fd::reconstruction::reconstruct_neighbor<
            Side::Upper,
            ::fd::reconstruction::detail::UnlimitedReconstructor<4>>(
            tensor_component_on_face_ptr, tensor_component_volume,
            tensor_component_neighbor, subcell_extents, ghost_data_extents,
            local_direction_to_reconstruct);
      },
      subcell_volume_vars, element, ghost_data, subcell_mesh,
      direction_to_reconstruct, ghost_zone_size());
}

bool operator==(const UnlimitedDeg4Prim& /*lhs*/,
                const UnlimitedDeg4Prim& /*rhs*/) {
  return true;
}

bool operator!=(const UnlimitedDeg4Prim& lhs, const UnlimitedDeg4Prim& rhs) {
  return not(lhs == rhs);
}

}  // namespace Ccz4::fd
