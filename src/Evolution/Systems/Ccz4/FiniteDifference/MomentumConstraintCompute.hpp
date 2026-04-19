// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Compute tag for the momentum constraint from CCZ4 variables.
 *
 * \details Computes the vacuum ADM momentum constraint:
 *
 * \f{align}{
 *   \mathcal{M}^i = \nabla_j S^{ij}
 *     = \partial_j S^{ij} + \Gamma^i_{jk} S^{kj} + \Gamma^j_{jk} S^{ik}
 * \f}
 *
 * where \f$S^{ij} = K^{ij} - \gamma^{ij} K
 * = \phi^2(\tilde{A}^{ij} - \frac{2}{3} K \tilde{\gamma}^{ij})\f$
 * and \f$\Gamma^k_{ij}\f$ is the physical Christoffel symbol.
 */
struct MomentumConstraintCompute
    : gr::Tags::MomentumConstraint<DataVector, 3, Frame::Inertial>,
      db::ComputeTag {
  using base = gr::Tags::MomentumConstraint<DataVector, 3, Frame::Inertial>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      Ccz4::Tags::ConformalFactor<DataVector>,
      Ccz4::Tags::ConformalMetric<DataVector, 3>,
      Ccz4::Tags::ATilde<DataVector, 3>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Ccz4::Tags::FieldD<DataVector, 3>,
      Ccz4::Tags::FieldP<DataVector, 3>,
      domain::Tags::Mesh<3>,
      domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                    Frame::Inertial>>;

  static void function(
      const gsl::not_null<return_type*> momentum_constraint,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::ii<DataVector, 3>& conformal_metric,
      const tnsr::ii<DataVector, 3>& a_tilde,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const tnsr::ijj<DataVector, 3>& field_d,
      const tnsr::i<DataVector, 3>& field_p, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian) {
    const size_t num_pts = get(conformal_factor).size();

    // Inverse conformal metric
    const auto [det_conformal, inv_conformal_metric] =
        determinant_and_inverse(conformal_metric);

    // Conformal Christoffel second kind
    tnsr::Ijj<DataVector, 3> conformal_christoffel(num_pts);
    ::Ccz4::conformal_christoffel_second_kind(
        make_not_null(&conformal_christoffel), inv_conformal_metric, field_d);

    // Physical Christoffel second kind
    tnsr::Ijj<DataVector, 3> christoffel(num_pts);
    ::Ccz4::christoffel_second_kind(make_not_null(&christoffel),
                                    conformal_metric, inv_conformal_metric,
                                    field_p, conformal_christoffel);

    // A_tilde^{ij} = gamma_tilde^{ik} gamma_tilde^{jl} A_tilde_{kl}
    tnsr::II<DataVector, 3> a_tilde_up(num_pts);
    ::tenex::evaluate<ti::I, ti::J>(
        make_not_null(&a_tilde_up),
        inv_conformal_metric(ti::I, ti::K) *
            inv_conformal_metric(ti::J, ti::L) * a_tilde(ti::k, ti::l));

    // S^{ij} = phi^2 (A_tilde^{ij} - (2/3) K gamma_tilde^{ij})
    tnsr::II<DataVector, 3> s_ij(num_pts);
    ::tenex::evaluate<ti::I, ti::J>(
        make_not_null(&s_ij),
        conformal_factor() * conformal_factor() *
            (a_tilde_up(ti::I, ti::J) -
             (2.0 / 3.0) * trace_extrinsic_curvature() *
                 inv_conformal_metric(ti::I, ti::J)));

    // d_S = partial_derivative(S^{ij})  ->  d_k S^{ij}
    const auto d_s = partial_derivative(s_ij, mesh, inverse_jacobian);

    // M^i = d_j S^{ij} + Gamma^i_{jk} S^{kj} + Gamma^j_{jk} S^{ik}
    ::tenex::evaluate<ti::I>(
        momentum_constraint,
        d_s(ti::j, ti::I, ti::J) +
            christoffel(ti::I, ti::j, ti::k) * s_ij(ti::K, ti::J) +
            christoffel(ti::J, ti::j, ti::k) * s_ij(ti::I, ti::K));
  }
};
}  // namespace Ccz4::fd
