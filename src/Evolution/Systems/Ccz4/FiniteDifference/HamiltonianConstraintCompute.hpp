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
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/Ricci.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Compute tag for the Hamiltonian constraint from CCZ4 variables.
 *
 * \details Computes the vacuum ADM Hamiltonian constraint with the 1/2
 * convention matching `gr::Tags::HamiltonianConstraint`:
 *
 * \f{align}{
 *   \mathcal{H} = \frac{1}{2}\left(R + \frac{2}{3}K^2
 *     - \tilde{\gamma}^{ik}\tilde{\gamma}^{jl}
 *       \tilde{A}_{ij}\tilde{A}_{kl}\right)
 * \f}
 *
 * where \f$R = \phi^2 \tilde{\gamma}^{ij} R_{ij}\f$ is the physical Ricci
 * scalar. The Ricci tensor is computed from conformal quantities via
 * `Ccz4::spatial_ricci_tensor`, which requires spectral derivatives of
 * `FieldD` and `FieldP`.
 */
struct HamiltonianConstraintCompute
    : gr::Tags::HamiltonianConstraint<DataVector>,
      db::ComputeTag {
  using base = gr::Tags::HamiltonianConstraint<DataVector>;
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
      const gsl::not_null<return_type*> hamiltonian_constraint,
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

    // D_k^{ij} = gamma_tilde^{im} gamma_tilde^{jn} D_{kmn}
    tnsr::iJJ<DataVector, 3> field_d_up(num_pts);
    ::tenex::evaluate<ti::k, ti::I, ti::J>(
        make_not_null(&field_d_up),
        inv_conformal_metric(ti::I, ti::N) *
            inv_conformal_metric(ti::M, ti::J) * field_d(ti::k, ti::n, ti::m));

    // Conformal Christoffel second kind
    tnsr::Ijj<DataVector, 3> conformal_christoffel(num_pts);
    ::Ccz4::conformal_christoffel_second_kind(
        make_not_null(&conformal_christoffel), inv_conformal_metric, field_d);

    // Spectral derivative of field_d, symmetrized in first two indices:
    // d_l D_{kij} = (1/2) d_l d_k gamma_tilde_{ij} is symmetric in l,k
    const auto d_field_d_raw =
        partial_derivative(field_d, mesh, inverse_jacobian);
    tnsr::iijj<DataVector, 3> d_field_d(num_pts);
    ::tenex::evaluate<ti::i, ti::j, ti::k, ti::l>(
        make_not_null(&d_field_d),
        0.5 * (d_field_d_raw(ti::i, ti::j, ti::k, ti::l) +
               d_field_d_raw(ti::j, ti::i, ti::k, ti::l)));

    // Derivative of conformal Christoffel
    tnsr::iJkk<DataVector, 3> d_conformal_christoffel(num_pts);
    ::Ccz4::deriv_conformal_christoffel_second_kind(
        make_not_null(&d_conformal_christoffel), inv_conformal_metric, field_d,
        d_field_d, field_d_up);

    // Physical Christoffel second kind
    tnsr::Ijj<DataVector, 3> christoffel(num_pts);
    ::Ccz4::christoffel_second_kind(make_not_null(&christoffel),
                                    conformal_metric, inv_conformal_metric,
                                    field_p, conformal_christoffel);

    // Contracted physical Christoffel: Gamma^m_{lm}
    tnsr::i<DataVector, 3> contracted_christoffel(num_pts);
    ::tenex::evaluate<ti::l>(make_not_null(&contracted_christoffel),
                             christoffel(ti::M, ti::l, ti::m));

    // Contracted d_conformal_christoffel difference:
    //   d_m Gamma_tilde^m_{ij} - d_j Gamma_tilde^m_{im}
    tnsr::ij<DataVector, 3> contracted_d_conformal_christoffel_diff(num_pts);
    ::tenex::evaluate<ti::i, ti::j>(
        make_not_null(&contracted_d_conformal_christoffel_diff),
        d_conformal_christoffel(ti::m, ti::M, ti::i, ti::j) -
            d_conformal_christoffel(ti::j, ti::M, ti::i, ti::m));

    // Contracted field_d_up: D_m^{ml}
    tnsr::I<DataVector, 3> contracted_field_d_up(num_pts);
    ::tenex::evaluate<ti::L>(make_not_null(&contracted_field_d_up),
                             field_d_up(ti::m, ti::M, ti::L));

    // Spectral derivative of field_p, symmetrized:
    // d_j P_i = d_j d_i ln(phi) is symmetric in i,j
    const auto d_field_p_raw =
        partial_derivative(field_p, mesh, inverse_jacobian);
    tnsr::ii<DataVector, 3> d_field_p(num_pts);
    ::tenex::evaluate<ti::i, ti::j>(
        make_not_null(&d_field_p),
        0.5 * (d_field_p_raw(ti::i, ti::j) +
               d_field_p_raw(ti::j, ti::i)));

    // Physical Ricci tensor
    tnsr::ii<DataVector, 3> ricci(num_pts);
    ::Ccz4::spatial_ricci_tensor(
        make_not_null(&ricci), christoffel, contracted_christoffel,
        contracted_d_conformal_christoffel_diff, conformal_metric,
        inv_conformal_metric, field_d, field_d_up, contracted_field_d_up,
        field_p, d_field_p);

    // Ricci scalar: R = phi^2 gamma_tilde^{ij} R_ij
    Scalar<DataVector> ricci_scalar(num_pts);
    ::tenex::evaluate(
        make_not_null(&ricci_scalar),
        conformal_factor() * conformal_factor() *
            inv_conformal_metric(ti::I, ti::J) * ricci(ti::i, ti::j));

    // A_tilde squared: gamma_tilde^{ik} gamma_tilde^{jl} A_tilde_{ij}
    // A_tilde_{kl}
    Scalar<DataVector> a_tilde_squared(num_pts);
    ::tenex::evaluate(make_not_null(&a_tilde_squared),
                      inv_conformal_metric(ti::I, ti::K) *
                          inv_conformal_metric(ti::J, ti::L) *
                          a_tilde(ti::i, ti::j) * a_tilde(ti::k, ti::l));

    // H = 0.5 * (R + (2/3) K^2 - A_tilde^{ij} A_tilde_{ij})
    get(*hamiltonian_constraint) =
        0.5 * (get(ricci_scalar) +
               (2.0 / 3.0) * square(get(trace_extrinsic_curvature)) -
               get(a_tilde_squared));
  }
};
}  // namespace Ccz4::fd
