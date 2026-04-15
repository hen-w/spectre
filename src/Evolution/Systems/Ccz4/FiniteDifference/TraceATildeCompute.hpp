// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Compute tag for \f$\mathrm{tr}\tilde{A} = \tilde{\gamma}^{ij}
 * \tilde{A}_{ij}\f$, the trace of the trace-free part of the extrinsic
 * curvature.
 *
 * \details Recomputed on demand from the evolved variables
 * `Ccz4::Tags::ConformalMetric` and `Ccz4::Tags::ATilde`. The CCZ4 formulation
 * enforces \f$\mathrm{tr}\tilde{A} = 0\f$ analytically, so this quantity can
 * be observed to monitor how well that algebraic constraint is preserved
 * during the evolution.
 */
struct TraceATildeCompute : ::Ccz4::Tags::TraceATilde<DataVector>,
                            db::ComputeTag {
  using base = ::Ccz4::Tags::TraceATilde<DataVector>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                 ::Ccz4::Tags::ATilde<DataVector, 3>>;

  static void function(const gsl::not_null<return_type*> trace_a_tilde,
                       const tnsr::ii<DataVector, 3>& conformal_metric,
                       const tnsr::ii<DataVector, 3>& a_tilde) {
    const auto det_and_inv = determinant_and_inverse(conformal_metric);
    const tnsr::II<DataVector, 3>& inv_conformal_metric = det_and_inv.second;
    ::tenex::evaluate(trace_a_tilde, inv_conformal_metric(ti::I, ti::J) *
                                         a_tilde(ti::i, ti::j));
  }
};
}  // namespace Ccz4::fd
