// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <vector>

#include "DataStructures/DataBox/AsAccess.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/NeighborPackagedData.hpp"

namespace {
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.NeighborPackagedData",
                  "[Unit][Evolution]") {
  // Phase 1 stub: only the empty-mortars case is supported.
  // Create a minimal DataBox (stub ignores it).
  auto box = db::create<db::AddSimpleTags<>>();

  const std::vector<DirectionalId<3>> empty_mortars{};
  const DirectionalIdMap<3, DataVector> result =
      Ccz4::fd::NeighborPackagedData::apply(db::as_access(box),
                                            empty_mortars);
  CHECK(result.empty());
}
}  // namespace
