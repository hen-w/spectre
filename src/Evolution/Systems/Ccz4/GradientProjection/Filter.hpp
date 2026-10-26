// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
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
 * \brief Enforce the algebraic constraints (unit conformal-metric determinant
 * and \f$\tilde\gamma\f$-trace-free \f$\tilde A\f$) on the physical state.
 *
 * \details Used once at initialization, after any perturbation of the initial
 * data, and after every time step, so that each right-hand side starts on the
 * constraint surface.
 */
struct EnforceAlgebraicConstraints : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                                 ::Ccz4::Tags::ATilde<DataVector, 3>>;
  using argument_tags = tmpl::list<>;

  static void apply(gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
                    gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde);
};

/*!
 * \brief The top-shell filter of the chart velocities at fixed potentials and
 * gradients.
 *
 * \details With the potentials \f$u\f$, their stored gradients \f$q\f$ and
 * the charts of \f$H\f$ held fixed, the velocities
 * \f$w=(a,K,\Theta,\hat\Gamma,b)\f$ are replaced by the \f$w^+\f$ with
 *
 * \f{align*}{
 * R(u,w^+,q)&=F\,R(u,w,q),\\
 * \hat\Gamma^+-b^+&=F\,(\hat\Gamma-b),
 * \f}
 *
 * where \f$R\f$ is `potential_rhs` (the potential rates) and \f$F\f$ the
 * velocity filter at the gain \f$g\f$: on tensor-product elements
 * \f$F=\Pi+g(I-\Pi)\f$ (`LocalOperators::apply_filter`) applied to each of
 * these 13 fields over the element, on spherical-harmonic shells the grouped
 * tensor filter `LocalOperators::apply_shell_filter` (the
 * `GeneralShellFilter`) of the 13 fields. Since \f$R\f$ is
 * linear in \f$w\f$ with no constant term at fixed \f$u\f$,
 * \f$w^+=P\left(F\,R(u,w,q)-R(u,0,q)\right)\f$ with \f$P\f$ the
 * `apply_rhs_inverse`, plus the filtered \f$\hat\Gamma-b\f$ on
 * \f$\hat\Gamma\f$. The local \f$a\f$ and the physical \f$\tilde A=\frac12
 * \mathcal T_Ha\f$ (which stays \f$\tilde\gamma\f$-trace-free),
 * \f$K\f$, \f$\Theta\f$, \f$\hat\Gamma\f$ and \f$b\f$ are overwritten in
 * place; nothing else changes. In the linearization about unit Minkowski,
 * \f$R=Lw+\sum_iE_iq_i\f$ with constant \f$L\f$ and \f$E_i\f$, this is
 * \f$w^+=Fw+P\sum_iE_i(F-I)q_i\f$, the filter of the linear model. At gain 1
 * nothing is done on tensor-product elements; on shells the filter still
 * removes the angular content outside the retained angular space.
 */
void filter_velocities(
    gsl::not_null<Tags::TransformedATilde::type*> a,
    gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde,
    gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
    gsl::not_null<Scalar<DataVector>*> theta,
    gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat,
    gsl::not_null<tnsr::I<DataVector, 3>*> b, const PotentialFields& u,
    const std::vector<LogMetricChart>& charts,
    const Tags::PotentialGradients::type& q, const Mesh<3>& mesh,
    const LocalOperators& local_operators, double gain, double f);
}  // namespace Ccz4::GradientProjection
