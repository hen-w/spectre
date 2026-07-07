// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/SoScalarWave/UpdateAuxiliaryVariables.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
template <size_t Dim>
void check_update_aux(const size_t npts, const double time) {
  CAPTURE(Dim);
  const Mesh<Dim> mesh{npts, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const size_t num_pts = mesh.number_of_grid_points();

  const auto wave_vector = make_array<Dim>(0.1);
  const auto center = make_array<Dim>(0.0);

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

  // Build psi on the grid for PowX(2): f(u) = u^2, f'(u) = 2u.
  // u = sum_i k_i*(x_i - c_i) - omega*t
  double omega_sq = 0.0;
  for (size_t i = 0; i < Dim; ++i) {
    omega_sq += gsl::at(wave_vector, i) * gsl::at(wave_vector, i);
  }
  DataVector u(num_pts, -sqrt(omega_sq) * time);
  for (size_t i = 0; i < Dim; ++i) {
    u += gsl::at(wave_vector, i) * (x.get(i) - gsl::at(center, i));
  }
  const Scalar<DataVector> psi{u * u};

  // phi_i = d_i psi = k_i * f'(u) = k_i * 2u
  const DataVector f_prime = 2.0 * u;
  tnsr::i<DataVector, Dim, Frame::Inertial> expected_phi{num_pts};
  for (size_t i = 0; i < Dim; ++i) {
    expected_phi.get(i) = gsl::at(wave_vector, i) * f_prime;
  }

  tnsr::i<DataVector, Dim, Frame::Inertial> phi_computed{num_pts};
  SoScalarWave::UpdateAuxiliaryVariables<Dim>::apply(
      make_not_null(&phi_computed), psi, mesh, inv_jac);

  for (size_t d = 0; d < Dim; ++d) {
    CHECK_ITERABLE_APPROX(phi_computed.get(d), expected_phi.get(d));
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.SoScalarWave.UpdateAuxiliaryVariables",
    "[Unit][Evolution]") {
  constexpr double time = 0.7;
  check_update_aux<1>(3, time);
  check_update_aux<2>(3, time);
  check_update_aux<3>(3, time);
}
