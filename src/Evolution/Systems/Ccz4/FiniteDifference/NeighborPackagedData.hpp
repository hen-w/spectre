// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

/// \cond
namespace db {
class Access;
}  // namespace db
class DataVector;
template <size_t VolumeDim>
struct DirectionalId;
template <size_t Dim, typename T>
class DirectionalIdMap;
/// \endcond

namespace Ccz4::fd {

/*!
 * \brief On elements using DG, reconstructs the interface data from a
 * neighboring element doing subcell.
 *
 * - `NeighborPackagedDataImpl<false>` (physical pass): packages data using
 *   `dg_package_data`.
 * - `NeighborPackagedDataImpl<true>` (auxiliary pass): packages data using
 *   `dg_auxiliary_package_data`.
 */
template <bool IsAuxiliary>
struct NeighborPackagedDataImpl {
  static DirectionalIdMap<3, DataVector> apply(
      const db::Access& box,
      const std::vector<DirectionalId<3>>& mortars_to_reconstruct_to);
};

/// Physical pass: used as DgComputeSubcellNeighborPackagedData.
using NeighborPackagedData = NeighborPackagedDataImpl<false>;

}  // namespace Ccz4::fd
