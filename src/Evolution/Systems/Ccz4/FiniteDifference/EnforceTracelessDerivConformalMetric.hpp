// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Mutator to enforce the trace-free condition on FieldD
 *
 * \details Togglable via the `ConstrainedEvolution` option in the input file.
 * Since the conformal metric satisfies \f$\det(\tilde{\gamma}_{ij})
 * = 1\f$, the auxiliary variable \f$D_{kij} = \frac{1}{2}\partial_k
 * \tilde{\gamma}_{ij}\f$ must satisfy the trace-free condition
 * \f$\tilde{\gamma}^{ij} D_{kij} = 0\f$.
 *
 * This mutator computes the residual
 * \f$r_k = \tilde{\gamma}^{ij} D_{kij}\f$
 * and removes the trace by
 * \f$D_{kij} \to D_{kij} - \frac{1}{3} r_k \tilde{\gamma}_{ij}\f$.
 */
struct EnforceTracelessDerivConformalMetric
    : tt::ConformsTo<db::protocols::Mutator> {
  static constexpr size_t dim = System::volume_dim;
  using return_tags = tmpl::list<::Ccz4::Tags::FieldD<DataVector, dim>>;
  using argument_tags =
      tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, dim>,
                 ::Ccz4::fd::Tags::ConstrainedEvolution>;

  static void apply(gsl::not_null<tnsr::ijj<DataVector, dim>*> field_d,
                    const tnsr::ii<DataVector, dim>& conformal_metric,
                    bool constrained_evolution);
};
}  // namespace Ccz4::fd
