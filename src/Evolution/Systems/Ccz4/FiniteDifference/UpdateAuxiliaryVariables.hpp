// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Computes the auxiliary variables FieldA, FieldB, FieldD, FieldP
 * from the current evolved variables and stores them back into the evolved
 * variables.
 *
 * \details In the LDG scheme, the auxiliary variables are not evolved in time
 * but are recomputed from the evolved variables at each substep:
 * - \f$A_i = \partial_i \ln\alpha\f$
 * - \f$B_{iJ} = \partial_i \beta^J\f$
 * - \f$D_{ijk} = \frac{1}{2}\partial_i \tilde\gamma_{jk}\f$
 * - \f$P_i = \partial_i \ln\phi\f$
 *
 * After this mutator runs, the boundary correction from
 * `ApplyAuxiliaryBoundaryCorrectionsToVariables` will fix up the auxiliary
 * fields at element interfaces.
 */
struct UpdateAuxiliaryVariables {
  using return_tags = tmpl::list<
      ::Ccz4::Tags::FieldA<DataVector, 3>, ::Ccz4::Tags::FieldB<DataVector, 3>,
      ::Ccz4::Tags::FieldD<DataVector, 3>, ::Ccz4::Tags::FieldP<DataVector, 3>>;
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      ::Ccz4::Tags::ConformalMetric<DataVector, 3>,
      ::Ccz4::Tags::ConformalFactor<DataVector>, domain::Tags::Mesh<3>,
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>>;

  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> field_a,
      const gsl::not_null<tnsr::iJ<DataVector, 3, Frame::Inertial>*> field_b,
      const gsl::not_null<tnsr::ijj<DataVector, 3, Frame::Inertial>*> field_d,
      const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> field_p,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& conformal_metric,
      const Scalar<DataVector>& conformal_factor, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian) {
    // field_a = d_i(ln(alpha))
    Scalar<DataVector> ln_lapse(get(lapse).size());
    get(ln_lapse) = log(get(lapse));
    *field_a = partial_derivative(ln_lapse, mesh, inverse_jacobian);

    // field_b = d_i(beta^J)
    *field_b = partial_derivative(shift, mesh, inverse_jacobian);

    // field_d = 0.5 * d_i(conformal_metric_{jk})
    const auto d_conformal_metric =
        partial_derivative(conformal_metric, mesh, inverse_jacobian);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = j; k < 3; ++k) {
          field_d->get(i, j, k) = 0.5 * d_conformal_metric.get(i, j, k);
        }
      }
    }

    // field_p = d_i(ln(phi))
    Scalar<DataVector> ln_phi(get(conformal_factor).size());
    get(ln_phi) = log(get(conformal_factor));
    *field_p = partial_derivative(ln_phi, mesh, inverse_jacobian);
  }
};
}  // namespace Ccz4::fd
