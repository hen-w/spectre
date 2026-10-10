// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
class TimeStepId;
/// \endcond

namespace Ccz4::GradientProjection {
/*!
 * \brief Enforce the algebraic constraints (unit conformal-metric determinant
 * and \f$\tilde\gamma\f$-trace-free \f$\tilde A\f$) on the physical state.
 *
 * \details Used once at initialization, after any perturbation of the initial
 * data, so that the first right-hand side starts on the constraint surface.
 */
struct EnforceAlgebraicConstraints : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                                 ::Ccz4::Tags::ATilde<DataVector, 3>>;
  using argument_tags = tmpl::list<>;

  static void apply(gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
                    gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde);
};

/*!
 * \brief The post-step filter and algebraic projection.
 *
 * \details After the time stepper has advanced the physical state, each
 * stored component of the physical \f$\tilde A_{ij}\f$, \f$K\f$, \f$\Theta\f$,
 * \f$\hat\Gamma^i\f$ and \f$b^i\f$ is filtered with
 * `LocalOperators::apply_filter` (top tensor shell gain
 * `shell_filter_gain`); \f$\tilde\gamma\f$, \f$\phi\f$, \f$\alpha\f$ and
 * \f$\beta\f$ are not filtered. The algebraic constraints are then enforced,
 * which restores the \f$\tilde\gamma\f$-trace of \f$\tilde A\f$ that the
 * componentwise filter does not preserve on a varying metric. The time-stepper
 * history is not modified. Like `UpdateU`, this does nothing on the unused
 * substeps of the self-start procedure, where the state is not advanced.
 */
struct FilterAndEnforceConstraints : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                                 ::Ccz4::Tags::ATilde<DataVector, 3>,
                                 gr::Tags::TraceExtrinsicCurvature<DataVector>,
                                 ::Ccz4::Tags::Theta<DataVector>,
                                 ::Ccz4::Tags::GammaHat<DataVector, 3>,
                                 ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<3>, Tags::LocalOperators,
                 ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>>;

  static void apply(
      gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
      gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde,
      gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> theta,
      gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat,
      gsl::not_null<tnsr::I<DataVector, 3>*> b, const Mesh<3>& mesh,
      const LocalOperators& local_operators, const TimeStepId& time_step_id,
      const TimeStepId& next_time_step_id);
};
}  // namespace Ccz4::GradientProjection
