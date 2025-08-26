// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.Tag", "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<Ccz4::fd::Tags::Reconstructor>(
      "Reconstructor");
}
