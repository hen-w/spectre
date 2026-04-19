// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
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
#include "Evolution/Systems/Ccz4/Z4Constraint.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Compute tag for the pointwise CCZ4 constraint energy.
 *
 * \details Computes
 *
 * \f{align}{
 *   E = \sqrt{\Theta^2 + \gamma^{ij} Z_i Z_j
 *     + \mathcal{H}^2 + \gamma_{ij}\mathcal{M}^i\mathcal{M}^j}
 * \f}
 *
 * where \f$\gamma_{ij} = \phi^{-2}\tilde{\gamma}_{ij}\f$ is the physical
 * spatial metric.  All four constraint quantities (Theta, Z_i, Hamiltonian,
 * Momentum) are computed internally so that expensive intermediates
 * (inverse conformal metric, Christoffel symbols, Ricci tensor) are shared.
 */
struct ConstraintEnergyCompute
    : Ccz4::Tags::ConstraintEnergy<DataVector>,
      db::ComputeTag {
  using base = Ccz4::Tags::ConstraintEnergy<DataVector>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      Ccz4::Tags::ConformalFactor<DataVector>,
      Ccz4::Tags::ConformalMetric<DataVector, 3>,
      Ccz4::Tags::ATilde<DataVector, 3>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Ccz4::Tags::FieldD<DataVector, 3>,
      Ccz4::Tags::FieldP<DataVector, 3>,
      Ccz4::Tags::Theta<DataVector>,
      Ccz4::Tags::GammaHat<DataVector, 3>,
      domain::Tags::Mesh<3>,
      domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                    Frame::Inertial>>;

  static void function(
      const gsl::not_null<return_type*> constraint_energy,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::ii<DataVector, 3>& conformal_metric,
      const tnsr::ii<DataVector, 3>& a_tilde,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const tnsr::ijj<DataVector, 3>& field_d,
      const tnsr::i<DataVector, 3>& field_p,
      const Scalar<DataVector>& theta,
      const tnsr::I<DataVector, 3>& gamma_hat,
      const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian) {
    const size_t num_pts = get(conformal_factor).size();

    // --- Shared intermediates ---

    // Inverse conformal metric
    const auto [det_conformal, inv_conformal_metric] =
        determinant_and_inverse(conformal_metric);

    // D_k^{ij}
    tnsr::iJJ<DataVector, 3> field_d_up(num_pts);
    ::tenex::evaluate<ti::k, ti::I, ti::J>(
        make_not_null(&field_d_up),
        inv_conformal_metric(ti::I, ti::N) *
            inv_conformal_metric(ti::M, ti::J) * field_d(ti::k, ti::n, ti::m));

    // Conformal Christoffel
    tnsr::Ijj<DataVector, 3> conformal_christoffel(num_pts);
    ::Ccz4::conformal_christoffel_second_kind(
        make_not_null(&conformal_christoffel), inv_conformal_metric, field_d);

    // Contracted conformal Christoffel (needed for Z_i)
    tnsr::I<DataVector, 3> contracted_conformal_christoffel(num_pts);
    ::Ccz4::contracted_conformal_christoffel_second_kind(
        make_not_null(&contracted_conformal_christoffel), inv_conformal_metric,
        conformal_christoffel);

    // Physical Christoffel (needed for Ricci and momentum)
    tnsr::Ijj<DataVector, 3> christoffel(num_pts);
    ::Ccz4::christoffel_second_kind(make_not_null(&christoffel),
                                    conformal_metric, inv_conformal_metric,
                                    field_p, conformal_christoffel);

    // --- Z_i Z^i term ---
    // Use existing functions for Z_i (lower) and Z^i (upper),
    // then contract: Z_i Z^i is just the direct index contraction.
    tnsr::I<DataVector, 3> gamma_hat_minus_contracted_conformal_christoffel(
        num_pts);
    ::tenex::evaluate<ti::I>(
        make_not_null(&gamma_hat_minus_contracted_conformal_christoffel),
        gamma_hat(ti::I) - contracted_conformal_christoffel(ti::I));

    // Z_i = (1/2) gamma_tilde_{ij} (gamma_hat^j - Gamma_tilde^j)
    tnsr::i<DataVector, 3> z4_constraint_lower(num_pts);
    ::Ccz4::spatial_z4_constraint(
        make_not_null(&z4_constraint_lower), conformal_metric,
        gamma_hat_minus_contracted_conformal_christoffel);

    // Z^i = (1/2) phi^2 (gamma_hat^i - Gamma_tilde^i)
    Scalar<DataVector> half_conformal_factor_squared(num_pts);
    get(half_conformal_factor_squared) =
        0.5 * square(get(conformal_factor));
    tnsr::I<DataVector, 3> z4_constraint_upper(num_pts);
    ::Ccz4::upper_spatial_z4_constraint(
        make_not_null(&z4_constraint_upper), half_conformal_factor_squared,
        gamma_hat_minus_contracted_conformal_christoffel);

    // Z_i Z^i (direct contraction of lower and upper)
    Scalar<DataVector> z_squared(num_pts);
    ::tenex::evaluate(make_not_null(&z_squared),
                      z4_constraint_lower(ti::i) *
                          z4_constraint_upper(ti::I));

    // --- Hamiltonian constraint ---

    // Spectral derivative of field_d, symmetrized in first two indices
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

    // Contracted physical Christoffel: Gamma^m_{lm}
    tnsr::i<DataVector, 3> contracted_christoffel(num_pts);
    ::tenex::evaluate<ti::l>(make_not_null(&contracted_christoffel),
                             christoffel(ti::M, ti::l, ti::m));

    // Contracted d_conformal_christoffel difference
    tnsr::ij<DataVector, 3> contracted_d_conformal_christoffel_diff(num_pts);
    ::tenex::evaluate<ti::i, ti::j>(
        make_not_null(&contracted_d_conformal_christoffel_diff),
        d_conformal_christoffel(ti::m, ti::M, ti::i, ti::j) -
            d_conformal_christoffel(ti::j, ti::M, ti::i, ti::m));

    // Contracted field_d_up: D_m^{ml}
    tnsr::I<DataVector, 3> contracted_field_d_up(num_pts);
    ::tenex::evaluate<ti::L>(make_not_null(&contracted_field_d_up),
                             field_d_up(ti::m, ti::M, ti::L));

    // Spectral derivative of field_p, symmetrized
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
    DataVector h_squared =
        square(0.5 * (get(ricci_scalar) +
                      (2.0 / 3.0) * square(get(trace_extrinsic_curvature)) -
                      get(a_tilde_squared)));

    // --- Momentum constraint ---
    // Reuses christoffel and inv_conformal_metric from above

    // A_tilde^{ij}
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

    // d_k S^{ij}
    const auto d_s = partial_derivative(s_ij, mesh, inverse_jacobian);

    // M^i = d_j S^{ij} + Gamma^i_{jk} S^{kj} + Gamma^j_{jk} S^{ik}
    tnsr::I<DataVector, 3> momentum(num_pts);
    ::tenex::evaluate<ti::I>(
        make_not_null(&momentum),
        d_s(ti::j, ti::I, ti::J) +
            christoffel(ti::I, ti::j, ti::k) * s_ij(ti::K, ti::J) +
            christoffel(ti::J, ti::j, ti::k) * s_ij(ti::I, ti::K));

    // M_i M^i = gamma_{ij} M^i M^j = phi^{-2} gamma_tilde_{ij} M^i M^j
    Scalar<DataVector> m_squared(num_pts);
    ::tenex::evaluate(
        make_not_null(&m_squared),
        conformal_metric(ti::i, ti::j) * momentum(ti::I) * momentum(ti::J) /
            (conformal_factor() * conformal_factor()));

    // --- Assemble constraint energy ---
    // E = sqrt(Theta^2 + Z_i Z^i + H^2 + M_i M^i)
    get(*constraint_energy) =
        sqrt(square(get(theta)) + get(z_squared) + h_squared + get(m_squared));
  }
};
}  // namespace Ccz4::fd
