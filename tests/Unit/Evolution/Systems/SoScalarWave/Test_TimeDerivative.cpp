// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Evolution/Systems/SoScalarWave/TimeDerivative.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SoPlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
template <size_t Dim>
void check_du_dt(const size_t npts, const double time) {
  CAPTURE(Dim);
  const Mesh<Dim> mesh{npts, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const size_t num_pts = mesh.number_of_grid_points();

  const auto wave_vector = make_array<Dim>(0.1);
  const auto center = make_array<Dim>(0.0);
  SoScalarWave::Solutions::SoPlaneWave<Dim> solution(
      wave_vector, center,
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2));

  tnsr::I<DataVector, Dim, Frame::Inertial> x{num_pts};
  {
    auto logical_coords = logical_coordinates(mesh);
    for (size_t i = 0; i < Dim; ++i) {
      x.get(i) = std::move(logical_coords.get(i));
    }
  }

  // Identity Jacobian (logical = inertial)
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      inv_jac{num_pts};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      inv_jac.get(i, j) = (i == j ? 1.0 : 0.0);
    }
  }

  // Evolved variables from analytic solution
  const Scalar<DataVector> pi{-1.0 * solution.dpsi_dt(x, time).get()};

  // Compute analytic derivatives for PowX(2): f(u) = u^2, f'(u) = 2u,
  // f''(u) = 2.  u = sum_i k_i*(x_i - c_i) - omega*t
  double omega_sq = 0.0;
  for (size_t i = 0; i < Dim; ++i) {
    omega_sq += gsl::at(wave_vector, i) * gsl::at(wave_vector, i);
  }

  DataVector u(num_pts, -sqrt(omega_sq) * time);
  for (size_t i = 0; i < Dim; ++i) {
    u += gsl::at(wave_vector, i) * (x.get(i) - gsl::at(center, i));
  }
  const DataVector f_prime = 2.0 * u;

  // phi_i = k_i * f'(u)
  tnsr::i<DataVector, Dim, Frame::Inertial> phi{num_pts};
  for (size_t i = 0; i < Dim; ++i) {
    phi.get(i) = gsl::at(wave_vector, i) * f_prime;
  }

  // d_psi_i = k_i * f'(u) (unused by apply)
  tnsr::i<DataVector, Dim, Frame::Inertial> d_psi{num_pts, 0.0};

  // d_pi_i (unused by apply)
  tnsr::i<DataVector, Dim, Frame::Inertial> d_pi{num_pts, 0.0};

  // d_phi_{ij} = d_i(k_j * f'(u)) = k_i * k_j * f''(u) = 2 * k_i * k_j
  tnsr::ij<DataVector, Dim, Frame::Inertial> d_phi{num_pts};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      d_phi.get(i, j) = 2.0 * gsl::at(wave_vector, i) * gsl::at(wave_vector, j);
    }
  }

  Scalar<DataVector> dt_psi{num_pts};
  Scalar<DataVector> dt_pi{num_pts};
  tnsr::i<DataVector, Dim, Frame::Inertial> dt_phi{num_pts};
  Scalar<DataVector> dt_boundary_psi{num_pts};

  // d_boundary_psi_i (unused by apply)
  tnsr::i<DataVector, Dim, Frame::Inertial> d_boundary_psi{num_pts, 0.0};

  SoScalarWave::TimeDerivative<Dim>::apply(
      make_not_null(&dt_psi), make_not_null(&dt_pi), make_not_null(&dt_phi),
      make_not_null(&dt_boundary_psi), d_psi, d_pi, d_phi, d_boundary_psi, pi,
      phi, mesh, inv_jac, x, time);

  // dt_psi = -pi = dpsi/dt
  CHECK_ITERABLE_APPROX(dt_psi, solution.dpsi_dt(x, time));

  // dt_pi = -trace(d_phi) = -d2psi/dt2
  CHECK_ITERABLE_APPROX(
      dt_pi, Scalar<DataVector>(-1.0 * solution.d2psi_dt2(x, time).get()));

  // dt_phi = 0 (not evolved in LDG)
  for (size_t d = 0; d < Dim; ++d) {
    CHECK_ITERABLE_APPROX(dt_phi.get(d), DataVector(num_pts, 0.0));
  }

  // dt_boundary_psi = 0 (no volume evolution)
  CHECK_ITERABLE_APPROX(dt_boundary_psi, Scalar<DataVector>(num_pts, 0.0));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.SoScalarWave.TimeDerivative",
                  "[Unit][Evolution]") {
  constexpr double time = 0.7;
  check_du_dt<1>(3, time);
  check_du_dt<2>(3, time);
  check_du_dt<3>(3, time);
}
