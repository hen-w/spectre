// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/Filter.hpp"

#include <cstddef>
#include <vector>

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
  std::vector<gsl::not_null<DataVector*>> fields{};
  fields.reserve(a_tilde->size() + 2 + gamma_hat->size() + b->size());
  for (auto& component : *a_tilde) {
    fields.emplace_back(&component);
  }
  fields.emplace_back(&get(*trace_extrinsic_curvature));
  fields.emplace_back(&get(*theta));
  for (auto& component : *gamma_hat) {
    fields.emplace_back(&component);
  }
  for (auto& component : *b) {
    fields.emplace_back(&component);
  }
  local_operators.apply_filter(fields);
  ::Ccz4::fd::enforce_algebraic_constraints(conformal_metric, a_tilde);
}
}  // namespace Ccz4::GradientProjection
