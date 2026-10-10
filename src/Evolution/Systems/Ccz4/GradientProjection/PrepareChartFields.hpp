// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace Ccz4::GradientProjection {
/// The relative tolerance for the Euclidean traces of \f$H\f$ and \f$a\f$
/// that the chart encoding discards.
constexpr double chart_trace_tolerance = 1.0e-10;

/*!
 * \brief The element-local ordinary gradients \f$D_iu\f$ of a field with the
 * layout of `Tags::ChartU`.
 */
void potential_gradients(
    gsl::not_null<Tags::ChartQ::type*> gradients,
    const Tags::ChartU::type& potentials, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian);

/*!
 * \brief Overwrite the chart fields \f$u,w\f$ from the current physical state
 * and reconstruct \f$q_i=D_iu\f$.
 *
 * \details This runs at the start of every right-hand-side evaluation, so the
 * chart scratch fields (which the time stepper advances mechanically) never
 * carry independently evolved information into a gradient consumer. It is an
 * `ERROR` if the physical state is off the algebraic-constraint surface by
 * more than `chart_trace_tolerance` (the projection after each step and at
 * initialization guarantees roundoff-level traces), so off-constraint data is
 * never silently projected by the encoding.
 */
struct PrepareChartFields : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<Tags::ChartU, Tags::ChartW, Tags::ChartQ>;
  using argument_tags = tmpl::list<
      ::Ccz4::Tags::ConformalMetric<DataVector, 3>,
      ::Ccz4::Tags::ConformalFactor<DataVector>,
      ::Ccz4::Tags::ATilde<DataVector, 3>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Ccz4::Tags::Theta<DataVector>, ::Ccz4::Tags::GammaHat<DataVector, 3>,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>, domain::Tags::Mesh<3>,
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>>;

  static void apply(gsl::not_null<Tags::ChartU::type*> u,
                    gsl::not_null<Tags::ChartW::type*> w,
                    gsl::not_null<Tags::ChartQ::type*> q,
                    const tnsr::ii<DataVector, 3>& conformal_metric,
                    const Scalar<DataVector>& conformal_factor,
                    const tnsr::ii<DataVector, 3>& a_tilde,
                    const Scalar<DataVector>& trace_extrinsic_curvature,
                    const Scalar<DataVector>& theta,
                    const tnsr::I<DataVector, 3>& gamma_hat,
                    const Scalar<DataVector>& lapse,
                    const tnsr::I<DataVector, 3>& shift,
                    const tnsr::I<DataVector, 3>& b, const Mesh<3>& mesh,
                    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                                          Frame::Inertial>& inverse_jacobian);
};
}  // namespace Ccz4::GradientProjection
