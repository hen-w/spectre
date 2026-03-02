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
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const gsl::not_null<Scalar<DataVector>*> dt_pi,

    const gsl::not_null<Scalar<DataVector>*> temp_dt_psi,
    const gsl::not_null<Scalar<DataVector>*> temp_dt_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> temp_d_psi,
    const gsl::not_null<Scalar<DataVector>*> temp_det_jacobian,

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

  Variables<tmpl::list<Tags::Flux<DataVector, Dim, Frame::Inertial>>> psi_flux(
      mesh.number_of_grid_points());
  for (size_t d = 0; d < Dim; ++d) {
    get<Tags::Flux<DataVector, Dim, Frame::Inertial>>(psi_flux).get(d) =
        get<::Tags::deriv<Tags::Psi, tmpl::size_t<Dim>, Frame::Inertial>>(
            first_derivs)
            .get(d);
  }

  const auto [det_inverse_jacobian, jacobian] =
      determinant_and_inverse(inverse_jacobian);
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      det_jac_times_inverse_jacobian{};
  ::dg::metric_identity_det_jac_times_inv_jac(
      make_not_null(&det_jac_times_inverse_jacobian), mesh, inertial_coords,
      jacobian);

  Variables<tmpl::list<Tags::Psi>> divergence_of_psi_flux(
      mesh.number_of_grid_points());
  weak_divergence(make_not_null(&divergence_of_psi_flux), psi_flux, mesh,
                  det_jac_times_inverse_jacobian);

  get(*dt_pi) =
      get(get<Tags::Psi>(divergence_of_psi_flux)) * get(det_inverse_jacobian);
  const auto& pi = get<Tags::Pi>(evolved_vars);
  get(*dt_psi) = -get(pi);

  // Filter evolved variables
  std::array<Matrix, Dim> filter;
  for (size_t d = 0; d < Dim; ++d) {
    filter[d] = Spectral::filtering::exponential_filter(mesh.slice_through(d),
                                                        36.0, 36);
  }

  //   // Filter the time derivatives
  //   std::array<Matrix, Dim> filter;
  //   for (size_t d = 0; d < Dim; ++d) {
  //     filter[d] =
  //     Spectral::filtering::exponential_filter(mesh.slice_through(d), 36.0,
  //                                                         36);
  //   }
  //   get(*dt_psi) = apply_matrices(filter, get(*dt_psi), mesh.extents());
  //   get(*dt_pi) = apply_matrices(filter, get(*dt_pi), mesh.extents());

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
  get(*temp_det_jacobian) = 1.0 / get(det_inverse_jacobian);

  // No flux divergence for non-conservative system, so
  // it does not matter whether we return true or false.
  return {false};
}

template class TimeDerivative<1>;
template class TimeDerivative<2>;
template class TimeDerivative<3>;
}  // namespace SoScalarWave
