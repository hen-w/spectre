// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeDerivativeDecisions.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

namespace SoScalarWave {
/*!
 * \brief Compute the time derivatives for the second-order scalar wave system
 */
template <size_t Dim>
struct TimeDerivative {
  // Include time derivatives as temporary tags so they can be
  // projected to faces and sent to neighbors for CG boundary corrections
  using temporary_tags = tmpl::list<>;

  using argument_tags =
      tmpl::list<Tags::Pi, Tags::Phi<Dim>, domain::Tags::Mesh<Dim>,
                 domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>, ::Tags::Time>;

  static evolution::dg::TimeDerivativeDecisions<Dim> apply(
      // Time derivatives returned by reference. All the tags in the
      // variables_tag in the system struct.
      gsl::not_null<Scalar<DataVector>*> dt_psi,
      gsl::not_null<Scalar<DataVector>*> dt_pi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,

      // Partial derivative arguments. Listed in the system struct as
      // gradient_variables.
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*d_psi*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_pi,
      const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi,

      // Terms list in argument_tags above
      const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*phi*/,
      const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
      const double& time);
};
}  // namespace SoScalarWave
