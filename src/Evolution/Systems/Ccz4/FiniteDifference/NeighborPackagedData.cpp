// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/NeighborPackagedData.hpp"

#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.tpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace Ccz4::fd {
DirectionalIdMap<3, DataVector> NeighborPackagedData::apply(
    const db::Access& /*box*/,
    const std::vector<DirectionalId<3>>& mortars_to_reconstruct_to) {
  // Phase 1 stub: no subcell neighbors expected when all elements are DG.
  // Phase 3 will implement actual face reconstruction from subcell ghost data.
  ASSERT(mortars_to_reconstruct_to.empty(),
         "NeighborPackagedData not yet implemented. Expected 0 mortars to "
         "reconstruct to, got "
             << mortars_to_reconstruct_to.size());
  return {};
}
}  // namespace Ccz4::fd

// Explicit template instantiation required by the DG-subcell framework.
// ApplyBoundaryCorrections calls neighbor_reconstructed_face_solution
// when using_subcell_v is true. The .tpp is included above; this
// instantiation makes the linker find the symbol.
template void evolution::dg::subcell::neighbor_reconstructed_face_solution<
    3, Ccz4::fd::NeighborPackagedData>(gsl::not_null<db::Access*> box);
