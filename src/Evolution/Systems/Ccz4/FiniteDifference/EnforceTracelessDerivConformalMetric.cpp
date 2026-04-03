// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceTracelessDerivConformalMetric.hpp"

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::fd {

void EnforceTracelessDerivConformalMetric::apply(
    const gsl::not_null<tnsr::ijj<DataVector, dim>*> field_d,
    const tnsr::ii<DataVector, dim>& conformal_metric,
    const bool constrained_evolution) {
  if (constrained_evolution) {
    const auto inv_conformal_metric =
        determinant_and_inverse(conformal_metric).second;

    // Compute residual_k = gamma_tilde^{ij} D_{kij}
    tnsr::i<DataVector, dim> residual{};
    ::tenex::evaluate<ti::k>(
        make_not_null(&residual),
        inv_conformal_metric(ti::I, ti::J) * (*field_d)(ti::k, ti::i, ti::j));

    // D_{kij} -> D_{kij} - (1/3) * residual_k * gamma_tilde_{ij}
    ::tenex::update<ti::k, ti::i, ti::j>(
        field_d, (*field_d)(ti::k, ti::i, ti::j) -
                     residual(ti::k) * conformal_metric(ti::i, ti::j) / 3.0);
  }
}

}  // namespace Ccz4::fd
