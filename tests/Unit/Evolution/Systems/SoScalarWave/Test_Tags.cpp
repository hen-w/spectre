// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.SoScalarWave.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::Psi>("Psi");
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::Pi>("Pi");
}
