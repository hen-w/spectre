// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeDerivativeDecisions.hpp"
#include "Evolution/PassVariables.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection {
/// The tensor types of the ordinary derivatives of the chart fields.
using DerivChartW =
    ::Tags::deriv<Tags::ChartW, tmpl::size_t<3>, Frame::Inertial>;
using DerivChartQ =
    ::Tags::deriv<Tags::ChartQ, tmpl::size_t<3>, Frame::Inertial>;

/*!
 * \brief The physical fields and their chain-rule derivatives that are passed
 * to the production second-order CCZ4 kernel `Ccz4::fd::detail::apply`.
 *
 * \details From the chart fields \f$u,w\f$, the element-local gradients
 * \f$q_i=D_iu\f$ and the ordinary derivatives \f$D_iw\f$, \f$D_iq_j\f$:
 * \f$g=e^{2H}\f$, \f$\phi=e^\ell\f$, \f$\tilde A=\frac12\mathcal T_Ha\f$,
 * \f{align*}{
 * \mathrm{FieldA}_i&=q_{\alpha,i}/\alpha,&
 * \mathrm{FieldB}_i{}^j&=q_{\beta,i}{}^j,&
 * \mathrm{FieldD}_{ijk}&=\tfrac12(\mathcal T_Hq_{H,i})_{jk},&
 * \mathrm{FieldP}_i&=q_{\ell,i},
 * \f}
 * and, symmetrized over the two derivative indices as the kernel expects,
 * \f{align*}{
 * \partial_{(i}\mathrm{FieldA}_{j)}&=D_{(i}q_{\alpha,j)}/\alpha
 *   -q_{\alpha,i}q_{\alpha,j}/\alpha^2,\qquad
 * \partial_{(i}\mathrm{FieldB}_{j)}{}^k=D_{(i}B_{j)}{}^k,\\
 * \partial_{(i}\mathrm{FieldD}_{j)kl}&=\tfrac12\left(\mathcal T_H
 *   D_{(i}q_{H,j)}+\mathcal S_H[q_{H,i},q_{H,j}]\right)_{kl},\qquad
 * \partial_{(i}\mathrm{FieldP}_{j)}=D_{(i}q_{\ell,j)},\\
 * \partial_k\tilde A&=\tfrac12\left(\mathcal T_HD_ka
 *   +\mathcal S_H[q_{H,k},a]\right),
 * \f}
 * with \f$D_kK\f$, \f$D_k\Theta\f$, \f$D_k\hat\Gamma\f$, \f$D_kb\f$ taken
 * from \f$D_kw\f$.
 */
struct KernelInputs {
  tnsr::ii<DataVector, 3> conformal_metric;
  Scalar<DataVector> conformal_factor;
  tnsr::ii<DataVector, 3> a_tilde;
  Scalar<DataVector> trace_extrinsic_curvature;
  Scalar<DataVector> theta;
  tnsr::I<DataVector, 3> gamma_hat;
  Scalar<DataVector> lapse;
  tnsr::I<DataVector, 3> shift;
  tnsr::I<DataVector, 3> b;
  tnsr::i<DataVector, 3> field_a;
  tnsr::iJ<DataVector, 3> field_b;
  tnsr::ijj<DataVector, 3> field_d;
  tnsr::i<DataVector, 3> field_p;
  tnsr::ii<DataVector, 3> d_field_a;
  tnsr::iiJ<DataVector, 3> d_field_b;
  tnsr::iijj<DataVector, 3> d_field_d;
  tnsr::ii<DataVector, 3> d_field_p;
  tnsr::ijj<DataVector, 3> d_a_tilde;
  tnsr::i<DataVector, 3> d_trace_extrinsic_curvature;
  tnsr::i<DataVector, 3> d_theta;
  tnsr::iJ<DataVector, 3> d_gamma_hat;
  tnsr::iJ<DataVector, 3> d_b;
};

/// Compute the `KernelInputs` from the chart fields and their derivatives.
KernelInputs kernel_inputs(const Tags::ChartU::type& u,
                           const Tags::ChartW::type& w,
                           const Tags::ChartQ::type& q,
                           const DerivChartW::type& d_w,
                           const DerivChartQ::type& d_q);

/// The physical rates of the production kernel that the scheme uses.
struct KernelRates {
  tnsr::ii<DataVector, 3> a_tilde;
  Scalar<DataVector> trace_extrinsic_curvature;
  Scalar<DataVector> theta;
  tnsr::I<DataVector, 3> gamma_hat;
  tnsr::I<DataVector, 3> b;
};

/*!
 * \brief Evaluate the production second-order CCZ4 kernel
 * `Ccz4::fd::detail::apply` on `KernelInputs`, with \f$c=1\f$, cleaning speed
 * 1, no algebraic relaxation, the Gamma-driver parameter \f$f\f$, a
 * non-shifting Gamma driver and evolved lapse and shift.
 *
 * \details Only the \f$\tilde A,K,\Theta,\hat\Gamma,b\f$ rates are returned;
 * the conformal metric, conformal factor, lapse and shift rates of the kernel
 * are not used by the scheme (the potential rates come from the gradient
 * fit).
 */
KernelRates kernel_rates(const KernelInputs& inputs,
                         const Scalar<DataVector>& kappa_1,
                         const Scalar<DataVector>& kappa_2, double kappa_3,
                         const Scalar<DataVector>& eta,
                         const Scalar<DataVector>& k_0, double f);

/*!
 * \brief The volume targets \f$(w_t^{\rm vol},q_{i,t}^{\rm vol})\f$ and the
 * original potential right-hand side \f$R\f$ (`potential_rhs`), for the
 * Gamma-driver parameter \f$f\f$.
 *
 * \details The production kernel is evaluated by `kernel_rates`. Its
 * physical \f$\tilde A_t\f$ is converted to the chart rate
 * \f$a_t=2\mathcal T_H^{-1}\tilde A_t-\mathcal T_H^{-1}
 * \mathcal S_H[R_H,a]\f$; its \f$K,\Theta,\hat\Gamma,b\f$ rates are the chart
 * rates, to which the continuum-zero ordering term
 * \f$\Delta\hat\Gamma_t^k=\Delta b_t^k=\frac16g^{ki}(D_iB_j{}^j-D_jB_i{}^j)\f$
 * with raw (unsymmetrized) \f$D_iB_j{}^k=D_iq_{\beta,j}{}^k\f$ is added. The
 * gradient targets are \f$q_{i,t}^{\rm vol}=R_uq_i+R(u,D_iw,D_iq)\f$, i.e.
 * \f$R_uq_i+LD_iw+\sum_jE_jD_iq_j\f$, and `dt_u` is set to \f$R(u,w,q)\f$.
 */
void volume_targets(gsl::not_null<Tags::ChartW::type*> dt_w,
                    gsl::not_null<Tags::ChartQ::type*> dt_q,
                    gsl::not_null<Tags::ChartU::type*> dt_u,
                    const Tags::ChartU::type& u, const Tags::ChartW::type& w,
                    const Tags::ChartQ::type& q, const DerivChartW::type& d_w,
                    const DerivChartQ::type& d_q,
                    const Scalar<DataVector>& kappa_1,
                    const Scalar<DataVector>& kappa_2, double kappa_3,
                    const Scalar<DataVector>& eta,
                    const Scalar<DataVector>& k_0, double f);

/*!
 * \brief The `compute_volume_time_derivative_terms` of
 * `Ccz4::GradientProjection::System`.
 *
 * \details Fills the time derivatives of the chart fields with the volume
 * targets of `volume_targets` (the potential slots with \f$R\f$) for the
 * Gamma-driver parameter `Ccz4::fd::System::f` and leaves the physical time
 * derivatives zero; the physical rates are computed only by
 * `FinalizeTimeDerivative` after all boundary corrections are lifted.
 */
struct TimeDerivative : evolution::PassVariables {
  static_assert(
      not ::Ccz4::fd::System::shifting_shift,
      "CCZ4 GradientProjection requires shifting_shift=false. Its fitted "
      "shift damping assumes dt(beta^i)=f*b^i; revisit the damping repair "
      "and characteristic operator before changing the gauge.");
  using temporary_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<Tags::ChartU, Tags::ChartW, Tags::ChartQ, ::Ccz4::Tags::Kappa1,
                 ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
                 ::Ccz4::Tags::Eta<DataVector>, ::Ccz4::Tags::K0<DataVector>>;

  template <typename DtTags>
  static evolution::dg::TimeDerivativeDecisions<3> apply(
      const gsl::not_null<Variables<DtTags>*> dt_vars,
      const gsl::not_null<Variables<tmpl::list<>>*> /*temporaries*/,
      const DerivChartW::type& d_w, const DerivChartQ::type& d_q,
      const Tags::ChartU::type& u, const Tags::ChartW::type& w,
      const Tags::ChartQ::type& q, const Scalar<DataVector>& kappa_1,
      const Scalar<DataVector>& kappa_2, const double kappa_3,
      const Scalar<DataVector>& eta, const Scalar<DataVector>& k_0) {
    volume_targets(make_not_null(&get<::Tags::dt<Tags::ChartW>>(*dt_vars)),
                   make_not_null(&get<::Tags::dt<Tags::ChartQ>>(*dt_vars)),
                   make_not_null(&get<::Tags::dt<Tags::ChartU>>(*dt_vars)), u,
                   w, q, d_w, d_q, kappa_1, kappa_2, kappa_3, eta, k_0,
                   ::Ccz4::fd::System::f);
    return {true};
  }
};
}  // namespace Ccz4::GradientProjection
