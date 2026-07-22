// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.SoScalarWave.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::Psi>("Psi");
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::Pi>("Pi");
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::Phi<3>>("Phi");
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::BoundaryPsi>(
      "BoundaryPsi");
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::NormalDotPhi>(
      "NormalDotPhi");
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::PsiTimesNormal<3>>(
      "PsiTimesNormal");
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::VPsi>("VPsi");
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::VZero<3>>("VZero");
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::VPlus>("VPlus");
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::VMinus>("VMinus");
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::CharacteristicSpeeds<3>>(
      "CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<SoScalarWave::Tags::CharacteristicFields<3>>(
      "CharacteristicFields");
  TestHelpers::db::test_simple_tag<
      SoScalarWave::Tags::EvolvedFieldsFromCharacteristicFields<3>>(
      "EvolvedFieldsFromCharacteristicFields");
}
