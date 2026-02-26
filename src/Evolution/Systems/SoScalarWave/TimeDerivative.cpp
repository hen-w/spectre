// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SoScalarWave/TimeDerivative.hpp"

#include <cstddef>
#include <iostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Element.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace SoScalarWave {
template <size_t Dim>
evolution::dg::TimeDerivativeDecisions<Dim> TimeDerivative<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const gsl::not_null<Scalar<DataVector>*> dt_pi,

    const gsl::not_null<Scalar<DataVector>*> temp_dt_psi,
    const gsl::not_null<Scalar<DataVector>*> temp_dt_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> temp_d_psi,

    const Variables<tmpl::list<Tags::Psi, Tags::Pi>>& evolved_vars,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
    const double& time) {
  // Compute first derivatives of Psi
  using first_deriv_var_tag =
      tmpl::list<::Tags::deriv<Tags::Psi, tmpl::size_t<Dim>, Frame::Inertial>,
                 ::Tags::deriv<Tags::Pi, tmpl::size_t<Dim>, Frame::Inertial>>;
  Variables<first_deriv_var_tag> first_derivs(mesh.number_of_grid_points());
  partial_derivatives(make_not_null(&first_derivs), evolved_vars, mesh,
                      inverse_jacobian);

  // Compute second derivatives of Psi by differentiating the first derivatives.
  // We have to use ::Tags::deriv twice instead of ::Tags::second_deriv as here
  // we are taking the derivative of the first derivatives.
  using second_deriv_var_tag = tmpl::list<
      ::Tags::deriv<
          ::Tags::deriv<Tags::Psi, tmpl::size_t<Dim>, Frame::Inertial>,
          tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<::Tags::deriv<Tags::Pi, tmpl::size_t<Dim>, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>>;
  Variables<second_deriv_var_tag> second_derivs(mesh.number_of_grid_points());
  partial_derivatives(make_not_null(&second_derivs), first_derivs, mesh,
                      inverse_jacobian);

  // Compute time derivative
  const auto& d_d_psi = get<::Tags::deriv<
      ::Tags::deriv<Tags::Psi, tmpl::size_t<Dim>, Frame::Inertial>,
      tmpl::size_t<Dim>, Frame::Inertial>>(second_derivs);
  get(*dt_pi) = 0.0;
  for (size_t d = 0; d < Dim; ++d) {
    get(*dt_pi) -= d_d_psi.get(d, d);
  }
  const auto& pi = get<Tags::Pi>(evolved_vars);
  get(*dt_psi) = -get(pi);

  std::cout << "coords: " << inertial_coords.get(0) << std::endl;
  std::cout << "dt_pi max error: "
            << (abs(get(*dt_pi) - sin(inertial_coords.get(0) - time)))
            << std::endl;

  // Copy time derivatives to temporary tags so they can be projected to faces
  // for CG collocation boundary corrections
  get(*temp_dt_psi) = get(*dt_psi);
  get(*temp_dt_pi) = get(*dt_pi);
  *temp_d_psi =
      get<::Tags::deriv<Tags::Psi, tmpl::size_t<Dim>, Frame::Inertial>>(
          first_derivs);

  // No flux divergence for non-conservative system, so
  // it does not matter whether we return true or false.
  return {false};
}

template class TimeDerivative<1>;
template class TimeDerivative<2>;
template class TimeDerivative<3>;
}  // namespace SoScalarWave
