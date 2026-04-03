// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <pup.h>
#include <string>

#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryCorrections/CgCollocation.hpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace {
template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen, const size_t num_pts) {
  PUPable_reg(SoScalarWave::BoundaryCorrections::CgCollocation<Dim>);

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      SoScalarWave::System<Dim>>(
      gen, "CgCollocation", "dg_package_data", "dg_boundary_terms",
      SoScalarWave::BoundaryCorrections::CgCollocation<Dim>{},
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::GaussLobatto},
      {}, {});

  const auto cg_collocation = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      SoScalarWave::BoundaryCorrections::CgCollocation<Dim>>("CgCollocation:");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      SoScalarWave::System<Dim>>(
      gen, "CgCollocation", "dg_package_data", "dg_boundary_terms",
      dynamic_cast<
          const SoScalarWave::BoundaryCorrections::CgCollocation<Dim>&>(
          *cg_collocation),
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::GaussLobatto},
      {}, {});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.SoScalarWave.BoundaryCorrections.CgCollocation",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/SoScalarWave/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  test<1>(make_not_null(&gen), 1);
  test<2>(make_not_null(&gen), 5);
  test<3>(make_not_null(&gen), 5);
}
