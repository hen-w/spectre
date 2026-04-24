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
struct NeighborPackagedData {
  static DirectionalIdMap<3, DataVector> apply(
      const db::Access& box,
      const std::vector<DirectionalId<3>>& mortars_to_reconstruct_to);
};
}  // namespace Ccz4::fd
