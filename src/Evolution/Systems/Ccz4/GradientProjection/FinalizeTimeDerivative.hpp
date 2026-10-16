// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
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
/*!
 * \brief Fit the potential rates, apply the compatibility correction and
 * convert all rates to the physical fields.
 *
 * \details After all boundary corrections have been lifted, `Tags::RhsTargets`
 * holds the corrected targets \f$w_t^*\f$, \f$q_{i,t}^*\f$ and, in the
 * potential slots, \f$R-S_uu\f$ (the original potential right-hand side with
 * the lifted internal value-jump damping). The chart potentials and velocities
 * are the physical fields with the local \f$H\f$ and \f$a\f$ of
 * `Tags::StepChartFields` (computed by `ComputeVolumeTargets`, as are the
 * charts `Tags::StepCharts`), and \f$q\f$ the stored gradients. Then
 *
 * 1. \f$r\f$ is the element-local constrained least-squares fit of
 *    `LocalOperators::fit` to the gradient targets \f$q_{i,t}^*\f$ with the
 *    mass-weighted mean \f$\overline{R-S_uu}\f$ of the time derivative;
 * 2. \f$u_t=r\f$ and
 *    \f$w_t=w_t^*+P\{\sum_iE_i(q_{i,t}^*-D_ir)+R_u(R-r)\}\f$ with \f$R\f$ the
 *    original pointwise potential right-hand side and \f$P\f$ of
 *    `apply_rhs_inverse` at the Gamma-driver parameter `Ccz4::fd::System::f`
 *    (both terms make
 *    \f$dR/dt=R_ur+Lw_t+\sum_iE_iD_ir=R_uR+Lw_t^*+\sum_iE_iq_{i,t}^*\f$ hold);
 * 3. replace the volume damping \f$-\eta b^i\f$ in the \f$b^i\f$ rate by
 *    \f$-\eta r_\beta^i/f\f$, by adding
 *    \f$-\eta(r_\beta^i/f-b^i)\f$ to that rate only. Here \f$r_\beta\f$ is
 *    the complete fitted shift rate, including its prescribed mean. This
 *    correction vanishes in the continuum for the non-shifting Gamma driver
 *    \f$\partial_t\beta^i=fb^i\f$. The compatibility identity in step 2 holds
 *    before this damping replacement;
 * 4. the physical rates are \f$\tilde\gamma_t=\mathcal T_Hr_H\f$,
 *    \f$\phi_t=r_\phi\f$,
 *    \f$\tilde A_t=\frac12\{\mathcal T_Ha_t+\mathcal S_H[r_H,a]\}\f$,
 *    \f$\alpha_t=r_\alpha\f$, \f$\beta_t=r_\beta\f$, and the
 *    \f$K,\Theta,\hat\Gamma,b\f$ rates are the corresponding components of
 *    \f$w_t\f$.
 *
 * These finalized physical rates are the only nonzero rates the time stepper
 * uses; the rates of the stored gradients stay zero.
 */
struct FinalizeTimeDerivative : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags =
      tmpl::list<::Tags::dt<::Ccz4::Tags::ConformalMetric<DataVector, 3>>,
                 ::Tags::dt<::Ccz4::Tags::ConformalFactor<DataVector>>,
                 ::Tags::dt<::Ccz4::Tags::ATilde<DataVector, 3>>,
                 ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>,
                 ::Tags::dt<::Ccz4::Tags::Theta<DataVector>>,
                 ::Tags::dt<::Ccz4::Tags::GammaHat<DataVector, 3>>,
                 ::Tags::dt<gr::Tags::Lapse<DataVector>>,
                 ::Tags::dt<gr::Tags::Shift<DataVector, 3>>,
                 ::Tags::dt<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>;
  using argument_tags = tmpl::list<
      Tags::StepChartFields, Tags::StepCharts,
      ::Ccz4::Tags::ConformalFactor<DataVector>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Ccz4::Tags::Theta<DataVector>, ::Ccz4::Tags::GammaHat<DataVector, 3>,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>, Tags::PotentialGradients,
      Tags::PotentialRateTarget, Tags::VelocityTarget, Tags::GradientTarget,
      domain::Tags::Mesh<3>,
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>,
      Tags::LocalOperators, ::Ccz4::Tags::Eta<DataVector>>;

  static void apply(
      gsl::not_null<tnsr::ii<DataVector, 3>*> dt_conformal_metric,
      gsl::not_null<Scalar<DataVector>*> dt_conformal_factor,
      gsl::not_null<tnsr::ii<DataVector, 3>*> dt_a_tilde,
      gsl::not_null<Scalar<DataVector>*> dt_trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> dt_theta,
      gsl::not_null<tnsr::I<DataVector, 3>*> dt_gamma_hat,
      gsl::not_null<Scalar<DataVector>*> dt_lapse,
      gsl::not_null<tnsr::I<DataVector, 3>*> dt_shift,
      gsl::not_null<tnsr::I<DataVector, 3>*> dt_b,
      const typename Tags::StepChartFields::type& chart_fields,
      const std::vector<LogMetricChart>& charts,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const Scalar<DataVector>& theta, const tnsr::I<DataVector, 3>& gamma_hat,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
      const tnsr::I<DataVector, 3>& b, const Tags::PotentialGradients::type& q,
      const Tags::PotentialRateTarget::type& dt_u_target,
      const Tags::VelocityTarget::type& dt_w_target,
      const Tags::GradientTarget::type& dt_q_target, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian,
      const LocalOperators& local_operators, const Scalar<DataVector>& eta);

  /// `apply` with the chart fields and charts computed from the physical
  /// fields.
  static void apply(
      gsl::not_null<tnsr::ii<DataVector, 3>*> dt_conformal_metric,
      gsl::not_null<Scalar<DataVector>*> dt_conformal_factor,
      gsl::not_null<tnsr::ii<DataVector, 3>*> dt_a_tilde,
      gsl::not_null<Scalar<DataVector>*> dt_trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> dt_theta,
      gsl::not_null<tnsr::I<DataVector, 3>*> dt_gamma_hat,
      gsl::not_null<Scalar<DataVector>*> dt_lapse,
      gsl::not_null<tnsr::I<DataVector, 3>*> dt_shift,
      gsl::not_null<tnsr::I<DataVector, 3>*> dt_b,
      const tnsr::ii<DataVector, 3>& conformal_metric,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::ii<DataVector, 3>& a_tilde,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const Scalar<DataVector>& theta, const tnsr::I<DataVector, 3>& gamma_hat,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
      const tnsr::I<DataVector, 3>& b, const Tags::PotentialGradients::type& q,
      const Tags::PotentialRateTarget::type& dt_u_target,
      const Tags::VelocityTarget::type& dt_w_target,
      const Tags::GradientTarget::type& dt_q_target, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian,
      const LocalOperators& local_operators, const Scalar<DataVector>& eta);
};

/*!
 * \brief The finalized chart rates \f$(r, w_t)\f$ of `FinalizeTimeDerivative`
 * for the Gamma-driver parameter \f$f\f$ before the conversion to physical
 * rates, including the fitted shift damping. The input velocity target must
 * already contain the volume source \f$-\eta b\f$.
 */
void finalized_chart_rates(
    gsl::not_null<Tags::PotentialRateTarget::type*> u_rate,
    gsl::not_null<Tags::VelocityTarget::type*> w_rate,
    gsl::not_null<Tags::GradientTarget::type*> fitted_gradient,
    const PotentialFields& u, const VelocityFields& w,
    const Tags::PotentialGradients::type& q,
    const Tags::PotentialRateTarget::type& dt_u_target,
    const Tags::VelocityTarget::type& dt_w_target,
    const Tags::GradientTarget::type& dt_q_target, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const LocalOperators& local_operators, const Scalar<DataVector>& eta,
    double f);

/// `finalized_chart_rates` with the `log_metric_charts` of `u` already
/// computed.
void finalized_chart_rates(
    gsl::not_null<Tags::PotentialRateTarget::type*> u_rate,
    gsl::not_null<Tags::VelocityTarget::type*> w_rate,
    gsl::not_null<Tags::GradientTarget::type*> fitted_gradient,
    const PotentialFields& u, const std::vector<LogMetricChart>& charts,
    const VelocityFields& w, const Tags::PotentialGradients::type& q,
    const Tags::PotentialRateTarget::type& dt_u_target,
    const Tags::VelocityTarget::type& dt_w_target,
    const Tags::GradientTarget::type& dt_q_target, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const LocalOperators& local_operators, const Scalar<DataVector>& eta,
    double f);
}  // namespace Ccz4::GradientProjection
