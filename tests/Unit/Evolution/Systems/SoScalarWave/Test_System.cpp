// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/SoScalarWave/System.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.SoScalarWave.System",
                  "[Unit][Evolution]") {
  CHECK(SoScalarWave::System<1>::name() == "SoScalarWave");
  CHECK(SoScalarWave::System<2>::name() == "SoScalarWave");
  CHECK(SoScalarWave::System<3>::name() == "SoScalarWave");
}
