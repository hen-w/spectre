// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryCorrections/LaxFriedrichs.hpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <size_t Dim>
void test_package_data(const gsl::not_null<std::mt19937*> gen) {
  CAPTURE(Dim);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const size_t num_pts = 5;

  const auto psi = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  const auto pi = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  const auto phi =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          gen, make_not_null(&dist), num_pts);
  const auto boundary_psi = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  const auto normal_covector =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          gen, make_not_null(&dist), num_pts);

  const SoScalarWave::BoundaryCorrections::LaxFriedrichs<Dim> correction{1.5,
                                                                         0.5};

  Scalar<DataVector> packaged_pi{num_pts};
  Scalar<DataVector> packaged_normal_dot_phi{num_pts};
  correction.dg_package_data(
      make_not_null(&packaged_pi), make_not_null(&packaged_normal_dot_phi), psi,
      pi, phi, boundary_psi, normal_covector, std::nullopt, std::nullopt);

  // Check pi is just copied
  CHECK_ITERABLE_APPROX(get(packaged_pi), get(pi));

  // Check normal_dot_phi = n_i * phi^i
  DataVector expected_ndphi(num_pts, 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    expected_ndphi += normal_covector.get(d) * phi.get(d);
  }
  CHECK_ITERABLE_APPROX(get(packaged_normal_dot_phi), expected_ndphi);
}

template <size_t Dim>
void test_boundary_terms(const gsl::not_null<std::mt19937*> gen) {
  CAPTURE(Dim);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const size_t num_pts = 5;

  const double tau1 = 1.5;
  const double tau2 = 0.5;
  const SoScalarWave::BoundaryCorrections::LaxFriedrichs<Dim> correction{tau1,
                                                                         tau2};

  const auto pi_int = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  const auto ndphi_int = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  const auto pi_ext = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  const auto ndphi_ext = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);

  Scalar<DataVector> psi_corr{num_pts};
  Scalar<DataVector> pi_corr{num_pts};
  tnsr::i<DataVector, Dim, Frame::Inertial> phi_corr{num_pts};
  Scalar<DataVector> boundary_psi_corr{num_pts};

  correction.dg_boundary_terms(
      make_not_null(&psi_corr), make_not_null(&pi_corr),
      make_not_null(&phi_corr), make_not_null(&boundary_psi_corr), pi_int,
      ndphi_int, pi_ext, ndphi_ext, dg::Formulation::StrongInertial);

  // psi correction = 0
  CHECK_ITERABLE_APPROX(get(psi_corr), DataVector(num_pts, 0.0));

  // pi correction = -0.5*(ndphi_int + ndphi_ext) - tau1*0.5*(pi_ext - pi_int)
  const DataVector expected_pi_corr = -0.5 * (get(ndphi_int) + get(ndphi_ext)) -
                                      tau1 * 0.5 * (get(pi_ext) - get(pi_int));
  CHECK_ITERABLE_APPROX(get(pi_corr), expected_pi_corr);

  // phi correction = 0
  for (size_t d = 0; d < Dim; ++d) {
    CHECK_ITERABLE_APPROX(phi_corr.get(d), DataVector(num_pts, 0.0));
  }

  // boundary_psi correction = 0
  CHECK_ITERABLE_APPROX(get(boundary_psi_corr), DataVector(num_pts, 0.0));
}

template <size_t Dim>
void test_auxiliary_package_data(const gsl::not_null<std::mt19937*> gen) {
  CAPTURE(Dim);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const size_t num_pts = 5;

  const auto psi = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  const auto pi = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  const auto phi =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          gen, make_not_null(&dist), num_pts);
  const auto boundary_psi = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  const auto normal_covector =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          gen, make_not_null(&dist), num_pts);

  const SoScalarWave::BoundaryCorrections::LaxFriedrichs<Dim> correction{1.5,
                                                                         0.5};

  Scalar<DataVector> packaged_psi{num_pts};
  tnsr::i<DataVector, Dim, Frame::Inertial> psi_times_normal{num_pts};

  correction.dg_auxiliary_package_data(
      make_not_null(&packaged_psi), make_not_null(&psi_times_normal), psi, pi,
      phi, boundary_psi, normal_covector, std::nullopt, std::nullopt);

  // Check psi is just copied
  CHECK_ITERABLE_APPROX(get(packaged_psi), get(psi));

  // Check psi_times_normal_i = psi * normal_i
  for (size_t d = 0; d < Dim; ++d) {
    CHECK_ITERABLE_APPROX(psi_times_normal.get(d),
                          get(psi) * normal_covector.get(d));
  }
}

template <size_t Dim>
void test_auxiliary_boundary_terms(const gsl::not_null<std::mt19937*> gen) {
  CAPTURE(Dim);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const size_t num_pts = 5;

  const double tau1 = 1.5;
  const double tau2 = 0.5;
  const SoScalarWave::BoundaryCorrections::LaxFriedrichs<Dim> correction{tau1,
                                                                         tau2};

  const auto psi_int = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  const auto psn_int =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          gen, make_not_null(&dist), num_pts);
  const auto psi_ext = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  const auto psn_ext =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          gen, make_not_null(&dist), num_pts);

  Scalar<DataVector> psi_corr{num_pts};
  Scalar<DataVector> pi_corr{num_pts};
  tnsr::i<DataVector, Dim, Frame::Inertial> phi_corr{num_pts};
  Scalar<DataVector> boundary_psi_corr{num_pts};

  correction.dg_auxiliary_boundary_terms(
      make_not_null(&psi_corr), make_not_null(&pi_corr),
      make_not_null(&phi_corr), make_not_null(&boundary_psi_corr), psi_int,
      psn_int, psi_ext, psn_ext, dg::Formulation::StrongInertial);

  // psi correction = 0
  CHECK_ITERABLE_APPROX(get(psi_corr), DataVector(num_pts, 0.0));

  // pi correction = 0
  CHECK_ITERABLE_APPROX(get(pi_corr), DataVector(num_pts, 0.0));

  // phi_i correction = 0.5*(psn_int_i + psn_ext_i) - 0.5*tau2*(psi_ext -
  // psi_int)
  for (size_t d = 0; d < Dim; ++d) {
    const DataVector expected = 0.5 * (psn_int.get(d) + psn_ext.get(d)) -
                                0.5 * tau2 * (get(psi_ext) - get(psi_int));
    CHECK_ITERABLE_APPROX(phi_corr.get(d), expected);
  }

  // boundary_psi correction = 0
  CHECK_ITERABLE_APPROX(get(boundary_psi_corr), DataVector(num_pts, 0.0));
}

template <size_t Dim>
void test_factory_creation() {
  CAPTURE(Dim);
  PUPable_reg(SoScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>);

  const auto lax_friedrichs = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      SoScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>>(
      "LaxFriedrichs:\n"
      "  Tau1: 1.5\n"
      "  Tau2: 0.5\n");
  CHECK(lax_friedrichs != nullptr);

  // Test serialization
  const auto serialized = serialize_and_deserialize(
      SoScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>{1.5, 0.5});
  // Verify the deserialized object produces the same results
  constexpr size_t npts = 1;
  Scalar<DataVector> packaged_pi{npts};
  Scalar<DataVector> packaged_ndphi{npts};
  const Scalar<DataVector> psi{{{{1.0}}}};
  const Scalar<DataVector> pi{{{{2.0}}}};
  tnsr::i<DataVector, Dim, Frame::Inertial> phi{npts, 0.0};
  phi.get(0) = 3.0;
  const Scalar<DataVector> boundary_psi{{{{0.5}}}};
  tnsr::i<DataVector, Dim, Frame::Inertial> normal{npts, 0.0};
  normal.get(0) = 1.0;
  serialized.dg_package_data(make_not_null(&packaged_pi),
                             make_not_null(&packaged_ndphi), psi, pi, phi,
                             boundary_psi, normal, std::nullopt, std::nullopt);
  CHECK(get(packaged_pi)[0] == approx(2.0));
  CHECK(get(packaged_ndphi)[0] == approx(3.0));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.SoScalarWave.BoundaryCorrections.LaxFriedrichs",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);

  test_package_data<1>(make_not_null(&gen));
  test_package_data<2>(make_not_null(&gen));
  test_package_data<3>(make_not_null(&gen));

  test_boundary_terms<1>(make_not_null(&gen));
  test_boundary_terms<2>(make_not_null(&gen));
  test_boundary_terms<3>(make_not_null(&gen));

  test_auxiliary_package_data<1>(make_not_null(&gen));
  test_auxiliary_package_data<2>(make_not_null(&gen));
  test_auxiliary_package_data<3>(make_not_null(&gen));

  test_auxiliary_boundary_terms<1>(make_not_null(&gen));
  test_auxiliary_boundary_terms<2>(make_not_null(&gen));
  test_auxiliary_boundary_terms<3>(make_not_null(&gen));

  test_factory_creation<1>();
  test_factory_creation<2>();
  test_factory_creation<3>();
}
