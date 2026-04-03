// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Mutator to enforce the trace-free condition on
 * \f$\partial_t \tilde{\gamma}_{ij}\f$
 *
 * \details Togglable via the `ConstrainedEvolution` option in the input file.
 * Since the conformal metric satisfies \f$\det(\tilde{\gamma}_{ij})
 * = 1\f$, its time derivative must be trace-free:
 * \f$\tilde{\gamma}^{ij} \partial_t \tilde{\gamma}_{ij} = 0\f$.
 *
 * This mutator computes the residual
 * \f$r = \tilde{\gamma}^{ij} \partial_t \tilde{\gamma}_{ij}\f$
 * and removes the trace by
 * \f$\partial_t \tilde{\gamma}_{ij} \to \partial_t \tilde{\gamma}_{ij}
 *   - \frac{1}{3} r \, \tilde{\gamma}_{ij}\f$.
 */
struct EnforceTracelessDtConformalMetric
    : tt::ConformsTo<db::protocols::Mutator> {
  static constexpr size_t dim = System::volume_dim;
  using return_tags =
      tmpl::list<::Tags::dt<::Ccz4::Tags::ConformalMetric<DataVector, dim>>>;
  using argument_tags =
      tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, dim>,
                 ::Ccz4::fd::Tags::ConstrainedEvolution>;

  static void apply(
      gsl::not_null<tnsr::ii<DataVector, dim>*> dt_conformal_metric,
      const tnsr::ii<DataVector, dim>& conformal_metric,
      bool constrained_evolution);
};
}  // namespace Ccz4::fd
