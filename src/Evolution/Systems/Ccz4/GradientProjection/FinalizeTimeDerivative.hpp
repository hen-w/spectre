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
 * \details After all boundary corrections have been lifted, the time
 * derivatives of the chart fields hold the corrected targets
 * \f$w_t^*\f$, \f$q_{i,t}^*\f$ and, in the potential slots,
 * \f$R-S_uu\f$ (the original potential right-hand side with the lifted
 * internal value-jump damping). Then
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
 *    \f$\phi_t=\phi\,r_\ell\f$,
 *    \f$\tilde A_t=\frac12\{\mathcal T_Ha_t+\mathcal S_H[r_H,a]\}\f$,
 *    \f$\alpha_t=r_\alpha\f$, \f$\beta_t=r_\beta\f$, and the
 *    \f$K,\Theta,\hat\Gamma,b\f$ rates are the corresponding components of
 *    \f$w_t\f$.
 *
 * These finalized physical rates are the only rates the time stepper uses for
 * the physical fields; the chart time derivatives are left holding the
 * targets.
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
      Tags::ChartU, Tags::ChartW, Tags::ChartQ, ::Tags::dt<Tags::ChartU>,
      ::Tags::dt<Tags::ChartW>, ::Tags::dt<Tags::ChartQ>, domain::Tags::Mesh<3>,
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
      gsl::not_null<tnsr::I<DataVector, 3>*> dt_b, const Tags::ChartU::type& u,
      const Tags::ChartW::type& w, const Tags::ChartQ::type& q,
      const Tags::ChartU::type& dt_u_target,
      const Tags::ChartW::type& dt_w_target,
      const Tags::ChartQ::type& dt_q_target, const Mesh<3>& mesh,
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
    gsl::not_null<Tags::ChartU::type*> u_rate,
    gsl::not_null<Tags::ChartW::type*> w_rate,
    gsl::not_null<Tags::ChartQ::type*> fitted_gradient,
    const Tags::ChartU::type& u, const Tags::ChartW::type& w,
    const Tags::ChartQ::type& q, const Tags::ChartU::type& dt_u_target,
    const Tags::ChartW::type& dt_w_target,
    const Tags::ChartQ::type& dt_q_target, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const LocalOperators& local_operators, const Scalar<DataVector>& eta,
    double f);
}  // namespace Ccz4::GradientProjection
