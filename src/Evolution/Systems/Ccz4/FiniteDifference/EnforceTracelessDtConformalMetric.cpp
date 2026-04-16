// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceTracelessDtConformalMetric.hpp"

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::fd {

void EnforceTracelessDtConformalMetric::apply(
    const gsl::not_null<tnsr::ii<DataVector, dim>*> dt_conformal_metric,
    const gsl::not_null<tnsr::ii<DataVector, dim>*>
        dt_boundary_conformal_metric,
    const tnsr::ii<DataVector, dim>& conformal_metric,
    const tnsr::ii<DataVector, dim>& boundary_conformal_metric,
    const bool constrained_evolution) {
  if (constrained_evolution) {
    const auto inv_conformal_metric =
        determinant_and_inverse(conformal_metric).second;

    // Compute residual r = gamma_tilde^{ij} dt_gamma_tilde_{ij}
    Scalar<DataVector> residual{};
    ::tenex::evaluate(make_not_null(&residual),
                      inv_conformal_metric(ti::I, ti::J) *
                          (*dt_conformal_metric)(ti::i, ti::j));

    // dt_gamma_tilde_{ij} -> dt_gamma_tilde_{ij} - (1/3) r gamma_tilde_{ij}
    ::tenex::update<ti::i, ti::j>(
        dt_conformal_metric,
        (*dt_conformal_metric)(ti::i, ti::j) -
            residual() * conformal_metric(ti::i, ti::j) / 3.0);

    // Same correction for the boundary conformal metric
    const auto inv_boundary_conformal_metric =
        determinant_and_inverse(boundary_conformal_metric).second;

    Scalar<DataVector> boundary_residual{};
    ::tenex::evaluate(make_not_null(&boundary_residual),
                      inv_boundary_conformal_metric(ti::I, ti::J) *
                          (*dt_boundary_conformal_metric)(ti::i, ti::j));

    ::tenex::update<ti::i, ti::j>(
        dt_boundary_conformal_metric,
        (*dt_boundary_conformal_metric)(ti::i, ti::j) -
            boundary_residual() * boundary_conformal_metric(ti::i, ti::j) /
                3.0);
  }
}

}  // namespace Ccz4::fd
