// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryCorrections/LaxFriedrichs.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
Mesh<Dim - 1> face_mesh() {
  if constexpr (Dim == 1) {
    return Mesh<0>{};
  } else {
    return Mesh<Dim - 1>{5, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
  }
}

template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen) {
  CAPTURE(Dim);
  PUPable_reg(SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>);

  std::uniform_real_distribution<> dist(0.0, 2.0);
  const double tau = dist(*gen);
  const SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>
      correction{tau};

  const auto mesh = face_mesh<Dim>();

  // The correction is a handful of O(1) multiply-adds, so C++ and python
  // agree to a few ULP; 1.0e-14 is tight while leaving ~10x flake margin.
  constexpr double eps = 1.0e-14;

  TestHelpers::evolution::dg::test_boundary_correction_conservation<
      SecondOrderScalarWave::System<Dim>>(
      gen, correction, mesh, {}, {},
      TestHelpers::evolution::dg::ZeroOnSmoothSolution::Yes, eps);
  TestHelpers::evolution::dg::test_auxiliary_boundary_correction_conservation<
      SecondOrderScalarWave::System<Dim>>(
      gen, correction, mesh, {}, {},
      TestHelpers::evolution::dg::ZeroOnSmoothSolution::Yes, eps);

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      SecondOrderScalarWave::System<Dim>>(
      gen, "LaxFriedrichs", "dg_package_data", "dg_boundary_terms", correction,
      mesh, {}, {}, eps, std::make_tuple(tau));
  TestHelpers::evolution::dg::test_auxiliary_boundary_correction_with_python<
      SecondOrderScalarWave::System<Dim>>(
      gen, "LaxFriedrichs", "dg_auxiliary_package_data",
      "dg_auxiliary_boundary_terms", correction, mesh, {}, {}, eps,
      std::make_tuple(tau));

  // Factory creation round-trips the options into an equal object.
  const auto created = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>>(
      "LaxFriedrichs:\n"
      "  Tau: 1.5\n");
  const auto& downcast = dynamic_cast<
      const SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>&>(
      *created);
  CHECK(downcast ==
        SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>{1.5});

  // Pin the return values of the two package-data functions (the physical
  // pass returns the largest characteristic-speed magnitude on the face,
  // 1 + max(|n.v|); the auxiliary pass a signaling NaN) and the packaged
  // Psi and n.v fields.
  const size_t num_pts = 2;
  const Scalar<DataVector> psi{num_pts, 0.5};
  const Scalar<DataVector> pi{num_pts, 0.5};
  const tnsr::i<DataVector, Dim, Frame::Inertial> phi{num_pts, 0.5};
  tnsr::i<DataVector, Dim, Frame::Inertial> normal_covector{num_pts, 0.0};
  get<0>(normal_covector) = 1.0;

  Scalar<DataVector> packaged_pi{num_pts};
  Scalar<DataVector> packaged_normal_dot_phi{num_pts};
  Scalar<DataVector> packaged_psi{num_pts};
  Scalar<DataVector> packaged_normal_dot_mesh_velocity{num_pts};
  const double max_char_speed = correction.dg_package_data(
      make_not_null(&packaged_pi), make_not_null(&packaged_normal_dot_phi),
      make_not_null(&packaged_psi),
      make_not_null(&packaged_normal_dot_mesh_velocity), psi, pi, phi,
      normal_covector, std::nullopt, std::nullopt);
  CHECK(max_char_speed == 1.0);
  CHECK(get(packaged_psi) == get(psi));
  CHECK(get(packaged_normal_dot_mesh_velocity) == DataVector(num_pts, 0.0));

  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{
      Scalar<DataVector>{DataVector{0.4, -1.5}}};
  const double moving_max_char_speed = correction.dg_package_data(
      make_not_null(&packaged_pi), make_not_null(&packaged_normal_dot_phi),
      make_not_null(&packaged_psi),
      make_not_null(&packaged_normal_dot_mesh_velocity), psi, pi, phi,
      normal_covector, std::nullopt, normal_dot_mesh_velocity);
  // Largest characteristic-speed magnitude on the face is
  // 1 + max(|n.v|) = 1 + 1.5 = 2.5.
  CHECK(moving_max_char_speed == 2.5);
  CHECK_ITERABLE_APPROX(get(packaged_normal_dot_mesh_velocity),
                        (DataVector{0.4, -1.5}));

  // Hand-computed pin of the advection-consistency boundary terms
  //   G_X = 0.5 (ndotv_int X_int + ndotv_ext X_ext),  X in {Psi, Pi},
  // with each side's n.v packaged with its own normal (consistent mesh
  // velocity: ndotv_ext = -ndotv_int). For Psi (psi_int = 2, psi_ext = 6):
  //   point 0: 0.5 (0.4 * 2 + (-0.4) * 6) = -0.8
  //   point 1: 0.5 ((-1.5) * 2 + 1.5 * 6) = 3.0
  // For Pi, the penalty coefficient is tau * lambda_max with
  //   lambda_max = 1 + |ndotv_int|:  point 0: 1.4, point 1: 2.5.
  // With tau = 1 the flux part of the Pi correction is
  //   -0.5 (1 + 3) - 0.5 lambda_max (5 - 2):
  //   point 0: -2 - 0.5 * 1.4 * 3 = -2 - 2.1  = -4.1
  //   point 1: -2 - 0.5 * 2.5 * 3 = -2 - 3.75 = -5.75,
  // and G_Pi (pi_int = 2, pi_ext = 5) adds
  //   point 0: 0.5 (0.4 * 2 + (-0.4) * 5) = -0.6  -> -4.7
  //   point 1: 0.5 ((-1.5) * 2 + 1.5 * 5) = 2.25  -> -3.5
  {
    const SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>
        pinned_correction{1.0};
    const Scalar<DataVector> pi_int{DataVector{2.0, 2.0}};
    const Scalar<DataVector> normal_dot_phi_int{DataVector{1.0, 1.0}};
    const Scalar<DataVector> psi_int{DataVector{2.0, 2.0}};
    const Scalar<DataVector> ndotv_int{DataVector{0.4, -1.5}};
    const Scalar<DataVector> pi_ext{DataVector{5.0, 5.0}};
    const Scalar<DataVector> normal_dot_phi_ext{DataVector{3.0, 3.0}};
    const Scalar<DataVector> psi_ext{DataVector{6.0, 6.0}};
    const Scalar<DataVector> ndotv_ext{DataVector{-0.4, 1.5}};
    Scalar<DataVector> psi_correction{num_pts};
    Scalar<DataVector> pi_correction{num_pts};
    pinned_correction.dg_boundary_terms(
        make_not_null(&psi_correction), make_not_null(&pi_correction), pi_int,
        normal_dot_phi_int, psi_int, ndotv_int, pi_ext, normal_dot_phi_ext,
        psi_ext, ndotv_ext, dg::Formulation::StrongInertial);
    CHECK_ITERABLE_APPROX(get(psi_correction), (DataVector{-0.8, 3.0}));
    CHECK_ITERABLE_APPROX(get(pi_correction), (DataVector{-4.7, -3.5}));

    // Static mesh (both packaged n.v zero): the Psi correction vanishes, the
    // penalty coefficient reduces to tau * (1 + 0) = 1, and the Pi correction
    // reduces to its static value -0.5 (1 + 3) - 0.5 (5 - 2) = -3.5.
    const Scalar<DataVector> zero_ndotv{DataVector{0.0, 0.0}};
    pinned_correction.dg_boundary_terms(
        make_not_null(&psi_correction), make_not_null(&pi_correction), pi_int,
        normal_dot_phi_int, psi_int, zero_ndotv, pi_ext, normal_dot_phi_ext,
        psi_ext, zero_ndotv, dg::Formulation::StrongInertial);
    CHECK(get(psi_correction) == DataVector(num_pts, 0.0));
    CHECK_ITERABLE_APPROX(get(pi_correction), (DataVector{-3.5, -3.5}));
  }

  tnsr::i<DataVector, Dim, Frame::Inertial> psi_times_normal{num_pts};
  const double auxiliary_speed = correction.dg_auxiliary_package_data(
      make_not_null(&psi_times_normal), psi, pi, normal_covector, std::nullopt,
      std::nullopt);
  {
    const ScopedFpeState disable_fpes(false);
    CHECK(std::isnan(auxiliary_speed));
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.SecondOrderScalarWave.BoundaryCorrections.LaxFriedrichs",
    "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/SecondOrderScalarWave/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  test<1>(make_not_null(&gen));
  test<2>(make_not_null(&gen));
  test<3>(make_not_null(&gen));
}
