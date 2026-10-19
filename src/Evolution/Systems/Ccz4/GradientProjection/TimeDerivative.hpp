// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace Ccz4::GradientProjection {
/*!
 * \brief The derived inputs of the production second-order CCZ4 kernel
 * `Ccz4::fd::detail::apply`; the physical fields and the derivatives
 * \f$D_kK\f$, \f$D_k\Theta\f$, \f$D_k\hat\Gamma\f$, \f$D_kb\f$ are passed to
 * the kernel directly.
 *
 * \details From \f$H\f$, the physical \f$\phi,\alpha\f$, the element-local
 * gradients \f$q_i=D_iu\f$, the ordinary derivatives \f$D_ia\f$ of the nodal
 * \f$a\f$ and \f$D_iq_j\f$:
 * \f{align*}{
 * \mathrm{FieldA}_i&=q_{\alpha,i}/\alpha,&
 * \mathrm{FieldB}_i{}^j&=q_{\beta,i}{}^j,&
 * \mathrm{FieldD}_{ijk}&=\tfrac12(\mathcal T_Hq_{H,i})_{jk},&
 * \mathrm{FieldP}_i&=q_{\phi,i}/\phi,
 * \f}
 * and, symmetrized over the two derivative indices as the kernel expects,
 * \f{align*}{
 * \partial_{(i}\mathrm{FieldA}_{j)}&=D_{(i}q_{\alpha,j)}/\alpha
 *   -q_{\alpha,i}q_{\alpha,j}/\alpha^2,\qquad
 * \partial_{(i}\mathrm{FieldB}_{j)}{}^k=D_{(i}B_{j)}{}^k,\\
 * \partial_{(i}\mathrm{FieldD}_{j)kl}&=\tfrac12\left(\mathcal T_H
 *   D_{(i}q_{H,j)}+\mathcal S_H[q_{H,i},q_{H,j}]\right)_{kl},\qquad
 * \partial_{(i}\mathrm{FieldP}_{j)}=D_{(i}q_{\phi,j)}/\phi
 *   -q_{\phi,i}q_{\phi,j}/\phi^2,\\
 * \partial_k\tilde A&=\tfrac12\left(\mathcal T_HD_ka
 *   +\mathcal S_H[q_{H,k},a]\right).
 * \f}
 * `field_b` does not own its data: it references the gradients \f$q\f$.
 */
struct KernelInputs {
  tnsr::i<DataVector, 3> field_a;
  tnsr::iJ<DataVector, 3> field_b;
  tnsr::ijj<DataVector, 3> field_d;
  tnsr::i<DataVector, 3> field_p;
  tnsr::ii<DataVector, 3> d_field_a;
  tnsr::iiJ<DataVector, 3> d_field_b;
  tnsr::iijj<DataVector, 3> d_field_d;
  tnsr::ii<DataVector, 3> d_field_p;
  tnsr::ijj<DataVector, 3> d_a_tilde;
};

/// Compute the `KernelInputs`, see its documentation.
KernelInputs kernel_inputs(const PotentialFields& u,
                           const Tags::TransformedATilde::type& a,
                           const Tags::PotentialGradients::type& q,
                           const DerivTransformedATilde::type& d_a,
                           const DerivPotentialGradients::type& d_q);

/// `kernel_inputs` with the `log_metric_charts` of `u` already computed.
KernelInputs kernel_inputs(const PotentialFields& u,
                           const std::vector<LogMetricChart>& charts,
                           const Tags::TransformedATilde::type& a,
                           const Tags::PotentialGradients::type& q,
                           const DerivTransformedATilde::type& d_a,
                           const DerivPotentialGradients::type& d_q);

/// The physical rates of the production kernel.
struct KernelRates {
  tnsr::ii<DataVector, 3> conformal_metric;
  Scalar<DataVector> conformal_factor;
  tnsr::ii<DataVector, 3> a_tilde;
  Scalar<DataVector> trace_extrinsic_curvature;
  Scalar<DataVector> theta;
  tnsr::I<DataVector, 3> gamma_hat;
  Scalar<DataVector> lapse;
  tnsr::I<DataVector, 3> shift;
  tnsr::I<DataVector, 3> b;
};

/*!
 * \brief Evaluate the production second-order CCZ4 kernel
 * `Ccz4::fd::detail::apply` on the physical fields (passed by reference),
 * `KernelInputs` and the derivatives of \f$K,\Theta,\hat\Gamma,b\f$, with
 * \f$c=1\f$, cleaning speed 1, no algebraic relaxation, the Gamma-driver
 * parameter \f$f\f$, a non-shifting Gamma driver and evolved lapse and shift.
 *
 * \details The scheme uses only the \f$\tilde A,K,\Theta,\hat\Gamma,b\f$ rates;
 * the potential rates of the kernel equal the original potential right-hand
 * side \f$R\f$ at \f$K_0=0\f$ and are returned for verification.
 */
KernelRates kernel_rates(const KernelInputs& inputs,
                         const tnsr::ii<DataVector, 3>& conformal_metric,
                         const tnsr::ii<DataVector, 3>& a_tilde,
                         const PotentialFields& u, const VelocityFields& w,
                         const VelocityDerivatives& d_w,
                         const Scalar<DataVector>& kappa_1,
                         const Scalar<DataVector>& kappa_2, double kappa_3,
                         const Scalar<DataVector>& eta,
                         const Scalar<DataVector>& k_0, double f);

/*!
 * \brief The volume targets \f$(w_t^{\rm vol},q_{i,t}^{\rm vol})\f$ and the
 * original potential right-hand side \f$R\f$ (`potential_rhs`), for the
 * Gamma-driver parameter \f$f\f$.
 *
 * \details The production kernel is evaluated by `kernel_rates` on the
 * physical fields. Its physical \f$\tilde A_t\f$ is converted to the chart rate
 * \f$a_t=2\mathcal T_H^{-1}\tilde A_t-\mathcal T_H^{-1}
 * \mathcal S_H[R_H,a]\f$; its \f$K,\Theta,\hat\Gamma,b\f$ rates are the chart
 * rates, to which the continuum-zero ordering term
 * \f$\Delta\hat\Gamma_t^k=\Delta b_t^k=\frac16g^{ki}(D_iB_j{}^j-D_jB_i{}^j)\f$
 * with raw (unsymmetrized) \f$D_iB_j{}^k=D_iq_{\beta,j}{}^k\f$ is added. The
 * gradient targets are \f$q_{i,t}^{\rm vol}=R_uq_i+R(u,D_iw,D_iq)\f$, i.e.
 * \f$R_uq_i+LD_iw+\sum_jE_jD_iq_j\f$, and `dt_u` is set to \f$R(u,w,q)\f$.
 */
void volume_targets(gsl::not_null<Tags::VelocityTarget::type*> dt_w,
                    gsl::not_null<Tags::GradientTarget::type*> dt_q,
                    gsl::not_null<Tags::PotentialRateTarget::type*> dt_u,
                    const tnsr::ii<DataVector, 3>& conformal_metric,
                    const tnsr::ii<DataVector, 3>& a_tilde,
                    const PotentialFields& u, const VelocityFields& w,
                    const Tags::PotentialGradients::type& q,
                    const VelocityDerivatives& d_w,
                    const DerivPotentialGradients::type& d_q,
                    const Scalar<DataVector>& kappa_1,
                    const Scalar<DataVector>& kappa_2, double kappa_3,
                    const Scalar<DataVector>& eta,
                    const Scalar<DataVector>& k_0, double f);

/// `volume_targets` with the `log_metric_charts` of `u` already computed.
void volume_targets(
    gsl::not_null<Tags::VelocityTarget::type*> dt_w,
    gsl::not_null<Tags::GradientTarget::type*> dt_q,
    gsl::not_null<Tags::PotentialRateTarget::type*> dt_u,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::ii<DataVector, 3>& a_tilde, const PotentialFields& u,
    const std::vector<LogMetricChart>& charts, const VelocityFields& w,
    const Tags::PotentialGradients::type& q, const VelocityDerivatives& d_w,
    const DerivPotentialGradients::type& d_q, const Scalar<DataVector>& kappa_1,
    const Scalar<DataVector>& kappa_2, double kappa_3,
    const Scalar<DataVector>& eta, const Scalar<DataVector>& k_0, double f);

/*!
 * \brief The volume evaluation of a right-hand side, run before
 * `evolution::dg::Actions::ComputeTimeDerivative` (and also the
 * `compute_volume_time_derivative_terms` of
 * `Ccz4::GradientProjection::System`, which has no generic volume terms).
 *
 * \details From the physical fields:
 * 1. the local nodal \f$H\f$ and \f$a\f$ are constructed (`local_chart_fields`,
 *    an `ERROR` off the algebraic-constraint surface) into
 *    `Tags::StepChartFields`, and the charts of \f$H\f$ from the same
 *    eigendecompositions into `Tags::StepCharts`, which
 *    `FinalizeTimeDerivative` reuses;
 * 2. the stored gradients are overwritten by
 *    \f$q_i=(D_iH,D_i\phi,D_i\alpha,D_i\beta)\f$ with the ordinary
 *    element-local derivative \f$D_i\f$;
 * 3. the velocities \f$(a,K,\Theta,\hat\Gamma,b)\f$ of the state (the local
 *    \f$a\f$ and the physical \f$\tilde A\f$, \f$K\f$, \f$\Theta\f$,
 *    \f$\hat\Gamma\f$ and \f$b\f$) are filtered in place by
 *    `filter_velocities` at the gain `Tags::TopShellFilterGain`, at the fixed
 *    potentials, gradients and charts of 1 and 2 (nothing at gain 1);
 * 4. \f$D_ia\f$, \f$D_iK\f$, \f$D_i\Theta\f$, \f$D_i\hat\Gamma\f$, \f$D_ib\f$
 *    and \f$D_iq_j\f$ are computed, the physical fields being differentiated
 *    in place through non-owning views;
 * 5. `Tags::RhsTargets` is set to the volume targets of `volume_targets` (the
 *    potential slot to \f$R\f$) for the Gamma-driver parameter
 *    `Ccz4::fd::System::f`;
 * 6. all time derivatives of the evolved variables are zeroed: the physical
 *    rates are computed only by `FinalizeTimeDerivative`, and the gradients are
 *    not evolved.
 *
 * The filter is the only modification of the physical fields: the state it
 * leaves is the one the right-hand side is evaluated at and the time stepper
 * records, so this is the end-of-step filter of the scheme applied before the
 * next right-hand side. It is an `ERROR` if \f$K_0\neq0\f$:
 * the potential right-hand side \f$R_\alpha\f$ (and so the lapse rate recovered
 * by `FinalizeTimeDerivative`) omits \f$K_0\f$, so the scheme supports only
 * \f$K_0=0\f$.
 */
struct ComputeVolumeTargets : tt::ConformsTo<db::protocols::Mutator> {
  static_assert(
      not ::Ccz4::fd::System::shifting_shift,
      "CCZ4 GradientProjection requires shifting_shift=false. Its fitted "
      "shift damping assumes dt(beta^i)=f*b^i; revisit the damping repair "
      "and characteristic operator before changing the gauge.");
  using variables_tag =
      ::Tags::Variables<::Ccz4::fd::System::physical_and_gradient_tags>;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  /// The generic DG action allocates no volume temporaries.
  using temporary_tags = tmpl::list<>;
  using return_tags =
      tmpl::list<variables_tag, dt_variables_tag, Tags::RhsTargets,
                 Tags::StepChartFields, Tags::StepCharts>;
  using argument_tags = tmpl::list<
      domain::Tags::Mesh<3>,
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>,
      ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
      ::Ccz4::Tags::Eta<DataVector>, ::Ccz4::Tags::K0<DataVector>,
      Tags::LocalOperators, Tags::TopShellFilterGain>;
  using const_global_cache_tags = tmpl::list<Tags::TopShellFilterGain>;

  static void apply(
      gsl::not_null<typename variables_tag::type*> evolved_variables,
      gsl::not_null<typename dt_variables_tag::type*> dt_evolved_variables,
      gsl::not_null<typename Tags::RhsTargets::type*> targets,
      gsl::not_null<typename Tags::StepChartFields::type*> chart_fields,
      gsl::not_null<std::vector<LogMetricChart>*> charts, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian,
      const Scalar<DataVector>& kappa_1, const Scalar<DataVector>& kappa_2,
      double kappa_3, const Scalar<DataVector>& eta,
      const Scalar<DataVector>& k_0, const LocalOperators& local_operators,
      double top_shell_filter_gain);

  /// `apply` with the chart fields and charts in local buffers.
  static void apply(
      gsl::not_null<typename variables_tag::type*> evolved_variables,
      gsl::not_null<typename dt_variables_tag::type*> dt_evolved_variables,
      gsl::not_null<typename Tags::RhsTargets::type*> targets,
      const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian,
      const Scalar<DataVector>& kappa_1, const Scalar<DataVector>& kappa_2,
      double kappa_3, const Scalar<DataVector>& eta,
      const Scalar<DataVector>& k_0, const LocalOperators& local_operators,
      double top_shell_filter_gain);
};
}  // namespace Ccz4::GradientProjection
