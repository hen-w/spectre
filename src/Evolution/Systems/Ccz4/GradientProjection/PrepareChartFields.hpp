// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace Ccz4::GradientProjection {
/// The relative tolerance for the Euclidean traces of \f$H\f$ and \f$a\f$
/// that the chart encoding discards.
constexpr double chart_trace_tolerance = 1.0e-10;

/// The local chart fields \f$(H,a)\f$ constructed from the physical fields.
using LocalChartFields =
    Variables<tmpl::list<Tags::LogConformalMetric, Tags::TransformedATilde>>;

/*!
 * \brief `chart_from_physical`, with an `ERROR` if the state is off the
 * algebraic-constraint surface by more than `chart_trace_tolerance`.
 *
 * \details The encoding of \f$H\f$ and \f$a\f$ discards their Euclidean
 * traces, \f$\frac12\log\det\tilde\gamma\f$ and
 * \f$\tilde\gamma^{ij}\tilde A_{ij}\f$; this check guarantees that no
 * off-constraint information is silently projected away. `context` is
 * included in the error message.
 */
void checked_chart_from_physical(gsl::not_null<PotentialVector*> u,
                                 gsl::not_null<VelocityVector*> w,
                                 const PhysicalPointState& state,
                                 const char* context);

/*!
 * \brief The local chart fields \f$H=\frac12\log\tilde\gamma\f$ and
 * \f$a=2\mathcal T_H^{-1}\tilde A\f$ at every grid point, see
 * `checked_chart_from_physical`.
 *
 * \details These are the only transformed fields of the scheme; the other
 * potentials and velocities are the physical fields themselves.
 */
void local_chart_fields(gsl::not_null<LocalChartFields*> chart,
                        const tnsr::ii<DataVector, 3>& conformal_metric,
                        const Scalar<DataVector>& conformal_factor,
                        const tnsr::ii<DataVector, 3>& a_tilde,
                        const Scalar<DataVector>& lapse);

/// The `LogMetricChart` of the logarithmic conformal metric \f$H\f$ of `u` at
/// every grid point.
std::vector<LogMetricChart> log_metric_charts(const PotentialFields& u);

/*!
 * \brief Assemble \f$q_i=D_iu=(D_iH,D_i\phi,D_i\alpha,D_i\beta)\f$ from the
 * ordinary derivatives of the nodal \f$H\f$ and of the physical fields.
 */
void assemble_potential_gradients(
    gsl::not_null<Tags::PotentialGradients::type*> gradients,
    const TensorMetafunctions::prepend_spatial_index<
        Tags::LogConformalMetric::type, 3, UpLo::Lo, Frame::Inertial>&
        d_log_conformal_metric,
    const tnsr::i<DataVector, 3>& d_conformal_factor,
    const tnsr::i<DataVector, 3>& d_lapse,
    const tnsr::iJ<DataVector, 3>& d_shift);

/*!
 * \brief The element-local ordinary gradients \f$D_iv\f$ of a field \f$v\f$
 * with the layout of the potentials.
 */
void potential_gradients(
    gsl::not_null<Tags::PotentialGradients::type*> gradients,
    const Tags::PotentialRateTarget::type& potentials, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian);
}  // namespace Ccz4::GradientProjection
