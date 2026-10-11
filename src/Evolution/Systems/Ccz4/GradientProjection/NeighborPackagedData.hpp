// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryCorrections/UpwindGradientProjection.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"

/// \cond
namespace db {
class Access;
}  // namespace db
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
struct DirectionalId;
template <size_t Dim, typename T>
class DirectionalIdMap;
/// \endcond

namespace Ccz4::GradientProjection {
/// The packaged face data of `BoundaryCorrections::UpwindGradientProjection`.
using PackagedFaceData = Variables<
    BoundaryCorrections::UpwindGradientProjection::dg_package_field_tags>;

/*!
 * \brief The package of the reconstructed values of a finite-difference
 * neighbor at the points of the shared finite-difference face.
 *
 * \details The reconstructed conformal metric and \f$\tilde A\f$ are first
 * projected onto the algebraic-constraint surface with
 * `Ccz4::fd::enforce_algebraic_constraints` (the reconstruction of
 * on-surface cell values is off the surface at truncation level), then
 * packaged by `BoundaryCorrections::UpwindGradientProjection::dg_package_data`
 * (which constructs \f$H\f$ and \f$a\f$ and checks the surface). Only these
 * temporary face values are projected; the gradients are packaged as
 * reconstructed. It is an `ERROR` if the reconstructed metric has a
 * non-positive determinant or the conformal factor or lapse is not positive.
 */
PackagedFaceData package_finite_difference_face(
    Variables<System::variables_tag_list> reconstructed_vars,
    const tnsr::i<DataVector, 3>& normal_covector,
    const BoundaryCorrections::UpwindGradientProjection& boundary_correction,
    const Direction<3>& direction);

/*!
 * \brief On elements using DG, the packaged data of the neighboring
 * finite-difference elements' sides of the shared faces, the
 * `DgComputeSubcellNeighborPackagedData` of the CCZ4 gradient-projection
 * DG-FD scheme.
 *
 * \details For each mortar: the DG volume variables (physical fields and the
 * gradients computed in the current right-hand side) are projected to the
 * subcells, the neighbor's face values are reconstructed from them and the
 * received finite-difference ghost data (physical fields and gradients), the
 * face is packaged by `package_finite_difference_face` with the reversed and
 * projected DG normal (the packaged neighbor normal is not used by the
 * correction), and the package is reconstructed onto the DG face points.
 *
 * It is an `ERROR` if a mortar's ghost data were not sent by a
 * finite-difference element (DG ghost data carry no gradients), if the
 * neighbor is not conforming of equal order (the mortar mesh is not the local
 * face mesh or the ghost data are not on the local subcell mesh: the package is
 * stored on the local face mesh, see
 * `evolution::dg::subcell::neighbor_reconstructed_face_solution`), or if the
 * conformal factor or lapse of the package on the DG face is not positive.
 */
struct NeighborPackagedData {
  static DirectionalIdMap<3, DataVector> apply(
      const db::Access& box,
      const std::vector<DirectionalId<3>>& mortars_to_reconstruct_to);
};
}  // namespace Ccz4::GradientProjection
