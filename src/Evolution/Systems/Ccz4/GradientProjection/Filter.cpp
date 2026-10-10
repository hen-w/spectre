// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/Filter.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceConstrainedEvolution.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/SelfStart.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::GradientProjection {
void EnforceAlgebraicConstraints::apply(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde) {
  ::Ccz4::fd::enforce_algebraic_constraints(conformal_metric, a_tilde);
}

void FilterAndEnforceConstraints::apply(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde,
    const gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
    const gsl::not_null<Scalar<DataVector>*> theta,
    const gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat,
    const gsl::not_null<tnsr::I<DataVector, 3>*> b, const Mesh<3>& mesh,
    const LocalOperators& local_operators, const TimeStepId& time_step_id,
    const TimeStepId& next_time_step_id) {
  if (::SelfStart::step_unused(time_step_id, next_time_step_id)) {
    return;
  }
  local_operators.check_mesh(mesh);
  for (auto& component : *a_tilde) {
    local_operators.apply_filter(make_not_null(&component));
  }
  local_operators.apply_filter(make_not_null(&get(*trace_extrinsic_curvature)));
  local_operators.apply_filter(make_not_null(&get(*theta)));
  for (auto& component : *gamma_hat) {
    local_operators.apply_filter(make_not_null(&component));
  }
  for (auto& component : *b) {
    local_operators.apply_filter(make_not_null(&component));
  }
  ::Ccz4::fd::enforce_algebraic_constraints(conformal_metric, a_tilde);
}
}  // namespace Ccz4::GradientProjection
