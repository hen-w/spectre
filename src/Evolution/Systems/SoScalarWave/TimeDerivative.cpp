// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SoScalarWave/TimeDerivative.hpp"

#include <cstddef>
#include <iostream>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Element.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MetricIdentityJacobian.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/WeakDivergence.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace SoScalarWave {
template <size_t Dim>
evolution::dg::TimeDerivativeDecisions<Dim> TimeDerivative<Dim>::apply(
    // Time derivatives returned by reference. All the tags in the
    // variables_tag in the system struct.
    gsl::not_null<Scalar<DataVector>*> dt_psi,
    gsl::not_null<Scalar<DataVector>*> dt_pi,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,
    gsl::not_null<Scalar<DataVector>*> dt_boundary_psi,

    // Partial derivative arguments. Listed in the system struct as
    // gradient_variables.
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*d_psi*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_pi,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*d_boundary_psi*/,

    // Terms list in argument_tags above
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*phi*/,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
    const double& time) {
  get(*dt_psi) = -get(pi);
  get(*dt_pi) = 0.0;
  for (size_t d = 0; d < Dim; ++d) {
    get(*dt_pi) -= d_phi.get(d, d);
  }
  // We do not evolve reduction variables in LDG
  for (size_t d = 0; d < Dim; ++d) {
    dt_phi->get(d) = 0.0;
  }
  // BoundaryPsi has zero volume time derivative; driven by BC correction only
  get(*dt_boundary_psi) = 0.0;

  //   std::cout << "coords: " << inertial_coords.get(0) << std::endl;
  //   std::cout << "dt_pi max error: "
  //             << (abs(get(*dt_pi) - sin(inertial_coords.get(0) - time)))
  //             << std::endl;

  return {false};
}

template class TimeDerivative<1>;
template class TimeDerivative<2>;
template class TimeDerivative<3>;
}  // namespace SoScalarWave
