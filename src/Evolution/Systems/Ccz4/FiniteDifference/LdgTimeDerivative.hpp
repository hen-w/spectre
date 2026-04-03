// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeDerivativeDecisions.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SoTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TempTags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Computes the time derivatives for the second-order CCZ4 system
 * using the LDG (Local Discontinuous Galerkin) discretization.
 *
 * \details This is the `compute_volume_time_derivative_terms` used by
 * `evolution::dg::Actions::ComputeTimeDerivative`. Spatial derivatives
 * of the gradient variables are provided by the DG infrastructure's
 * `partial_derivative`. This function symmetrizes the raw derivatives
 * of auxiliary fields (FieldA, FieldB, FieldD, FieldP) and then delegates
 * to `Ccz4::fd::detail::apply()` which computes the actual RHS.
 */
struct LdgTimeDerivative {
  using temporary_tags = tmpl::list<>;

  using argument_tags = tmpl::list<
      ::Ccz4::Tags::ConformalMetric<DataVector, 3>,
      ::Ccz4::Tags::ConformalFactor<DataVector>,
      ::Ccz4::Tags::ATilde<DataVector, 3>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Ccz4::Tags::Theta<DataVector>, ::Ccz4::Tags::GammaHat<DataVector, 3>,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>,
      ::Ccz4::Tags::FieldA<DataVector, 3>, ::Ccz4::Tags::FieldB<DataVector, 3>,
      ::Ccz4::Tags::FieldD<DataVector, 3>, ::Ccz4::Tags::FieldP<DataVector, 3>,
      ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
      ::Ccz4::Tags::Eta<DataVector>, ::Ccz4::Tags::K0<DataVector>,
      ::Ccz4::fd::Tags::EvolveLapseAndShift, domain::Tags::Element<3>,
      domain::Tags::Mesh<3>, domain::Tags::ExternalBoundaryConditions<3>,
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>>;

  static evolution::dg::TimeDerivativeDecisions<3> apply(
      // dt outputs for ALL variables in variables_tag (17 total),
      // in variables_tag_list order: gradient_variables then boundary_modes.
      //   original evolved (9):
      gsl::not_null<tnsr::ii<DataVector, 3>*> dt_conformal_metric,
      gsl::not_null<Scalar<DataVector>*> dt_conformal_factor,
      gsl::not_null<tnsr::ii<DataVector, 3>*> dt_a_tilde,
      gsl::not_null<Scalar<DataVector>*> dt_trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> dt_theta,
      gsl::not_null<tnsr::I<DataVector, 3>*> dt_gamma_hat,
      gsl::not_null<Scalar<DataVector>*> dt_lapse,
      gsl::not_null<tnsr::I<DataVector, 3>*> dt_shift,
      gsl::not_null<tnsr::I<DataVector, 3>*> dt_b,
      //   auxiliary (4):
      gsl::not_null<tnsr::i<DataVector, 3>*> dt_field_a,
      gsl::not_null<tnsr::iJ<DataVector, 3>*> dt_field_b,
      gsl::not_null<tnsr::ijj<DataVector, 3>*> dt_field_d,
      gsl::not_null<tnsr::i<DataVector, 3>*> dt_field_p,
      //   boundary modes (4):
      gsl::not_null<Scalar<DataVector>*> dt_u_scalar3_minus,
      gsl::not_null<tnsr::i<DataVector, 3>*> dt_u_vector2_minus,
      gsl::not_null<Scalar<DataVector>*> dt_u_scalar2_minus,
      gsl::not_null<tnsr::ii<DataVector, 3>*> dt_u_tensor_minus,

      // Partial derivatives of gradient_variables (17 total):
      // ConformalMetric, ConformalFactor, ATilde, K, Theta, GammaHat,
      // Lapse, Shift, b, FieldA, FieldB, FieldD, FieldP,
      // UScalar3Minus, UVector2Minus, UScalar2Minus, UTensorMinus
      const tnsr::ijj<DataVector, 3>& /*d_conformal_metric*/,
      const tnsr::i<DataVector, 3>& /*d_conformal_factor*/,
      const tnsr::ijj<DataVector, 3>& d_a_tilde,
      const tnsr::i<DataVector, 3>& d_trace_extrinsic_curvature,
      const tnsr::i<DataVector, 3>& d_theta,
      const tnsr::iJ<DataVector, 3>& d_gamma_hat,
      const tnsr::i<DataVector, 3>& /*d_lapse*/,
      const tnsr::iJ<DataVector, 3>& /*d_shift*/,
      const tnsr::iJ<DataVector, 3>& d_b,
      const tnsr::ij<DataVector, 3>& d_field_a_raw,
      const tnsr::ijK<DataVector, 3>& d_field_b_raw,
      const tnsr::ijkk<DataVector, 3>& d_field_d_raw,
      const tnsr::ij<DataVector, 3>& d_field_p_raw,
      // Boundary mode derivatives (zero, included for DG infrastructure)
      const tnsr::i<DataVector, 3>& /*d_u_scalar3_minus*/,
      const tnsr::ij<DataVector, 3>& /*d_u_vector2_minus*/,
      const tnsr::i<DataVector, 3>& /*d_u_scalar2_minus*/,
      const tnsr::ijj<DataVector, 3>& /*d_u_tensor_minus*/,

      // argument_tags
      const tnsr::ii<DataVector, 3>& conformal_metric,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::ii<DataVector, 3>& a_tilde,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const Scalar<DataVector>& theta, const tnsr::I<DataVector, 3>& gamma_hat,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
      const tnsr::I<DataVector, 3>& b, const tnsr::i<DataVector, 3>& field_a,
      const tnsr::iJ<DataVector, 3>& field_b,
      const tnsr::ijj<DataVector, 3>& field_d,
      const tnsr::i<DataVector, 3>& field_p, const double& kappa_1,
      const double& kappa_2, const double& kappa_3,
      const Scalar<DataVector>& eta, const Scalar<DataVector>& k_0,
      const bool& evolve_lapse_and_shift, const Element<3>& /*element*/,
      const Mesh<3>& /*mesh*/,
      const std::vector<DirectionMap<
          3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>&
      /*all_boundary_conditions*/,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& /*inv_jacobian*/) {
    const size_t num_pts = get(lapse).size();

    // Initialize boundary mode dt to zero everywhere
    get(*dt_u_scalar3_minus) = 0.0;
    for (auto& component : *dt_u_vector2_minus) {
      component = 0.0;
    }
    get(*dt_u_scalar2_minus) = 0.0;
    for (auto& component : *dt_u_tensor_minus) {
      component = 0.0;
    }

    // Symmetrize derivatives of auxiliary fields.
    // detail::apply expects symmetric first-two-lower-index derivatives.

    // d_field_a_sym(i,j) = 0.5 * (d_field_a_raw(i,j) + d_field_a_raw(j,i))
    tnsr::ii<DataVector, Dim> d_field_a(num_pts);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        d_field_a.get(i, j) =
            0.5 * (d_field_a_raw.get(i, j) + d_field_a_raw.get(j, i));
      }
    }

    // d_field_b_sym: symmetrize first two lower indices
    // raw is tnsr with indices (lo, lo, up) all independent
    // target is tnsr::iiJ (first two symmetric)
    tnsr::iiJ<DataVector, Dim> d_field_b(num_pts);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        for (size_t k = 0; k < Dim; ++k) {
          d_field_b.get(i, j, k) =
              0.5 * (d_field_b_raw.get(i, j, k) + d_field_b_raw.get(j, i, k));
        }
      }
    }

    // d_field_d_sym: symmetrize first two lower indices
    // raw has indices (lo, lo, lo, lo) with last two symmetric
    // target is tnsr::iijj (first two symmetric, last two symmetric)
    tnsr::iijj<DataVector, Dim> d_field_d(num_pts);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        for (size_t k = 0; k < Dim; ++k) {
          for (size_t l = k; l < Dim; ++l) {
            d_field_d.get(i, j, k, l) = 0.5 * (d_field_d_raw.get(i, j, k, l) +
                                               d_field_d_raw.get(j, i, k, l));
          }
        }
      }
    }

    // d_field_p_sym(i,j) = 0.5 * (d_field_p_raw(i,j) + d_field_p_raw(j,i))
    tnsr::ii<DataVector, Dim> d_field_p(num_pts);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        d_field_p.get(i, j) =
            0.5 * (d_field_p_raw.get(i, j) + d_field_p_raw.get(j, i));
      }
    }

    // Allocate temporaries (same as SoTimeDerivative.hpp)
    using TempVars = Variables<tmpl::list<
        ::Ccz4::Tags::ConformalFactorSquared<DataVector>,
        ::Ccz4::Tags::DetConformalSpatialMetric<DataVector>,
        ::Ccz4::Tags::InverseConformalMetric<DataVector, Dim>,
        gr::Tags::InverseSpatialMetric<DataVector, Dim>,
        ::Ccz4::Tags::InvATilde<DataVector, Dim>,
        ::Ccz4::Tags::ATildeTimesFieldB<DataVector, Dim>,
        ::Ccz4::Tags::ATildeMinusOneThirdConformalMetricTimesTraceATilde<
            DataVector, Dim>,
        ::Ccz4::Tags::ContractedFieldB<DataVector>,
        ::Ccz4::Tags::SymmetrizedDerivFieldB<DataVector, Dim>,
        ::Ccz4::Tags::ContractedSymmetrizedDerivFieldB<DataVector, Dim>,
        ::Ccz4::Tags::FieldDUpTimesATilde<DataVector, Dim>,
        ::Ccz4::Tags::ContractedFieldDUp<DataVector, Dim>,
        ::Ccz4::Tags::HalfConformalFactorSquared<DataVector>,
        ::Ccz4::Tags::ConformalMetricTimesFieldB<DataVector, Dim>,
        ::Ccz4::Tags::ConformalMetricTimesTraceATilde<DataVector, Dim>,
        ::Ccz4::Tags::InverseConformalMetricTimesDerivATilde<DataVector, Dim>,
        ::Ccz4::Tags::GammaHatMinusContractedConformalChristoffel<DataVector,
                                                                  Dim>,
        ::Ccz4::Tags::DerivGammaHatMinusContractedConformalChristoffel<
            DataVector, Dim>,
        ::Ccz4::Tags::ContractedChristoffelSecondKind<DataVector, Dim>,
        ::Ccz4::Tags::ContractedDerivConformalChristoffelDifference<DataVector,
                                                                    Dim>,
        ::Ccz4::Tags::KMinus2ThetaC<DataVector>,
        ::Ccz4::Tags::KMinusK0Minus2ThetaC<DataVector>,
        ::Ccz4::Tags::LapseTimesATilde<DataVector, Dim>,
        ::Ccz4::Tags::LapseTimesDerivATilde<DataVector, Dim>,
        ::Ccz4::Tags::LapseTimesFieldA<DataVector, Dim>,
        ::Ccz4::Tags::LapseTimesConformalMetric<DataVector, Dim>,
        ::Ccz4::Tags::LapseTimesSlicingCondition<DataVector>,
        ::Ccz4::Tags::LapseTimesRicciScalarPlus2DivergenceZ4Constraint<
            DataVector>,
        ::Ccz4::Tags::ShiftTimesDerivGammaHat<DataVector, Dim>,
        ::Ccz4::Tags::InverseTauTimesConformalMetric<DataVector, Dim>,
        ::Ccz4::Tags::TraceATilde<DataVector>,
        ::Ccz4::Tags::FieldDUp<DataVector, Dim>,
        ::Ccz4::Tags::ConformalChristoffelSecondKind<DataVector, Dim>,
        ::Ccz4::Tags::DerivConformalChristoffelSecondKind<DataVector, Dim>,
        ::Ccz4::Tags::ChristoffelSecondKind<DataVector, Dim>,
        ::Ccz4::Tags::SpatialRicciTensor<DataVector, Dim>,
        ::Ccz4::Tags::GradGradLapse<DataVector, Dim>,
        ::Ccz4::Tags::DivergenceLapse<DataVector>,
        ::Ccz4::Tags::ContractedConformalChristoffelSecondKind<DataVector, Dim>,
        ::Ccz4::Tags::DerivContractedConformalChristoffelSecondKind<DataVector,
                                                                    Dim>,
        ::Ccz4::Tags::SpatialZ4Constraint<DataVector, Dim>,
        ::Ccz4::Tags::GradSpatialZ4Constraint<DataVector, Dim>,
        ::Ccz4::Tags::RicciScalarPlusDivergenceZ4Constraint<DataVector>>>;

    TempVars temp_vars(num_pts);
    tnsr::I<DataVector, Dim> upper_spatial_z4_constraint(num_pts);

    // Fixed params for SO-CCZ4
    const double c = 1.0;
    const double cleaning_speed = 1.0;
    const double one_over_relaxation_time = 0.0;

    const double f_param = System::f;

    // Analytic time derivatives of auxiliary fields.
    // Used by ComputeCrpbcBoundaryModeDt instead of spectral differentiation.
    // Convention: dt_field_d = (1/2) ∂_k[∂_t γ̃_{ij}], etc.
    {
      constexpr double one_third = 1.0 / 3.0;

      // dt_field_d(k,i,j) = (1/2) * ∂_k[∂_t γ̃_{ij}]
      ::tenex::evaluate<ti::k, ti::i, ti::j>(
          dt_field_d, -(d_a_tilde(ti::k, ti::i, ti::j) * lapse() +
                        a_tilde(ti::i, ti::j) * lapse() * field_a(ti::k)) +
                          field_d(ti::k, ti::l, ti::i) * field_b(ti::j, ti::L) +
                          field_d(ti::k, ti::l, ti::j) * field_b(ti::i, ti::L) +
                          0.5 * (conformal_metric(ti::i, ti::l) *
                                     d_field_b(ti::k, ti::j, ti::L) +
                                 conformal_metric(ti::j, ti::l) *
                                     d_field_b(ti::k, ti::i, ti::L)) -
                          one_third * (conformal_metric(ti::i, ti::j) *
                                           d_field_b(ti::k, ti::l, ti::L) +
                                       2.0 * field_b(ti::l, ti::L) *
                                           field_d(ti::k, ti::i, ti::j)) +
                          shift(ti::L) * d_field_d(ti::k, ti::l, ti::i, ti::j) +
                          field_b(ti::k, ti::L) * field_d(ti::l, ti::i, ti::j));

      // dt_field_b(k,^i) = ∂_k[∂_t β^i]
      ::tenex::evaluate<ti::k, ti::I>(dt_field_b, f_param * d_b(ti::k, ti::I));
      if constexpr (System::shifting_shift) {
        ::tenex::update<ti::k, ti::I>(
            dt_field_b, (*dt_field_b)(ti::k, ti::I) +
                            field_b(ti::k, ti::L) * field_b(ti::l, ti::I) +
                            shift(ti::L) * d_field_b(ti::k, ti::l, ti::I));
      }

      // dt_field_a(k) = ∂_t A_k = ∂_k[∂_t ln α]
      // From ∂_t ln α = -2(K-K0-2θ) + β^l A_l, differentiating (∂_k K0 = 0):
      ::tenex::evaluate<ti::k>(
          dt_field_a,
          -2.0 * (d_trace_extrinsic_curvature(ti::k) - 2.0 * d_theta(ti::k)) +
              field_b(ti::k, ti::L) * field_a(ti::l) +
              shift(ti::L) * d_field_a(ti::k, ti::l));

      // dt_field_p(k) = ∂_t P_k = ∂_k[∂_t ln φ]
      // From ∂_t ln φ = (1/3)(α K - ∂_l β^l) + β^l P_l, differentiating:
      ::tenex::evaluate<ti::k>(
          dt_field_p,
          one_third * (lapse() * field_a(ti::k) * trace_extrinsic_curvature() +
                       lapse() * d_trace_extrinsic_curvature(ti::k) -
                       d_field_b(ti::k, ti::l, ti::L)) +
              field_b(ti::k, ti::L) * field_p(ti::l) +
              shift(ti::L) * d_field_p(ti::k, ti::l));
    }

    auto& [conformal_factor_squared, det_conformal_spatial_metric,
           inv_conformal_spatial_metric, inv_spatial_metric, inv_a_tilde,
           a_tilde_times_field_b,
           a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde,
           contracted_field_b, symmetrized_d_field_b,
           contracted_symmetrized_d_field_b, field_d_up_times_a_tilde,
           contracted_field_d_up, half_conformal_factor_squared,
           conformal_metric_times_field_b, conformal_metric_times_trace_a_tilde,
           inv_conformal_metric_times_d_a_tilde,
           gamma_hat_minus_contracted_conformal_christoffel,
           d_gamma_hat_minus_contracted_conformal_christoffel,
           contracted_christoffel_second_kind,
           contracted_d_conformal_christoffel_difference, k_minus_2_theta_c,
           k_minus_k0_minus_2_theta_c, lapse_times_a_tilde,
           lapse_times_d_a_tilde, lapse_times_field_a,
           lapse_times_conformal_spatial_metric, lapse_times_slicing_condition,
           lapse_times_ricci_scalar_plus_divergence_z4_constraint,
           shift_times_deriv_gamma_hat, inv_tau_times_conformal_metric,
           trace_a_tilde, field_d_up, conformal_christoffel_second_kind,
           d_conformal_christoffel_second_kind, christoffel_second_kind,
           spatial_ricci_tensor, grad_grad_lapse, divergence_lapse,
           contracted_conformal_christoffel_second_kind,
           d_contracted_conformal_christoffel_second_kind,
           spatial_z4_constraint, grad_spatial_z4_constraint,
           ricci_scalar_plus_divergence_z4_constraint] = temp_vars;

    // const double tol2 = 1.0e-9;

    // const auto [det_conformal_metric, inv_conformal_metric] =
    //   determinant_and_inverse(conformal_metric);

    // tnsr::i<DataVector, Dim, Frame::Inertial>
    // trace_d_conformal_metric_check{};
    // ::tenex::evaluate<ti::k>(
    //     make_not_null(&trace_d_conformal_metric_check),
    //     2.0 * inv_conformal_metric(ti::I, ti::J) *
    //         field_d(ti::k, ti::i, ti::j));
    // double max_abs_trace_d_cm =
    //     max(abs(trace_d_conformal_metric_check.get(0)));
    // for (size_t k = 1; k < Dim; ++k) {
    //   max_abs_trace_d_cm = std::max(
    //       max_abs_trace_d_cm,
    //       max(abs(trace_d_conformal_metric_check.get(k))));
    // }
    // ASSERT(max_abs_trace_d_cm < tol2,
    //         "Interior trace(d_conformal_metric) deviates from 0 at place 2 by
    //         "
    //             << max_abs_trace_d_cm << " (tolerance " << tol2 << ")");

    ::Ccz4::fd::detail::apply<Dim>(
        // LHS time derivatives of evolved variables
        dt_conformal_metric, dt_lapse, dt_shift, dt_conformal_factor,
        dt_a_tilde, dt_trace_extrinsic_curvature, dt_theta, dt_gamma_hat, dt_b,
        // temporaries
        make_not_null(&conformal_factor_squared),
        make_not_null(&det_conformal_spatial_metric),
        make_not_null(&inv_conformal_spatial_metric),
        make_not_null(&inv_spatial_metric), make_not_null(&inv_a_tilde),
        make_not_null(&a_tilde_times_field_b),
        make_not_null(
            &a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde),
        make_not_null(&contracted_field_b),
        make_not_null(&symmetrized_d_field_b),
        make_not_null(&contracted_symmetrized_d_field_b),
        make_not_null(&field_d_up_times_a_tilde),
        make_not_null(&contracted_field_d_up),
        make_not_null(&half_conformal_factor_squared),
        make_not_null(&conformal_metric_times_field_b),
        make_not_null(&conformal_metric_times_trace_a_tilde),
        make_not_null(&inv_conformal_metric_times_d_a_tilde),
        make_not_null(&gamma_hat_minus_contracted_conformal_christoffel),
        make_not_null(&d_gamma_hat_minus_contracted_conformal_christoffel),
        make_not_null(&contracted_christoffel_second_kind),
        make_not_null(&contracted_d_conformal_christoffel_difference),
        make_not_null(&k_minus_2_theta_c),
        make_not_null(&k_minus_k0_minus_2_theta_c),
        make_not_null(&lapse_times_a_tilde),
        make_not_null(&lapse_times_d_a_tilde),
        make_not_null(&lapse_times_field_a),
        make_not_null(&lapse_times_conformal_spatial_metric),
        make_not_null(&lapse_times_slicing_condition),
        make_not_null(&lapse_times_ricci_scalar_plus_divergence_z4_constraint),
        make_not_null(&shift_times_deriv_gamma_hat),
        make_not_null(&inv_tau_times_conformal_metric),
        make_not_null(&trace_a_tilde), make_not_null(&field_d_up),
        make_not_null(&conformal_christoffel_second_kind),
        make_not_null(&d_conformal_christoffel_second_kind),
        make_not_null(&christoffel_second_kind),
        make_not_null(&spatial_ricci_tensor), make_not_null(&grad_grad_lapse),
        make_not_null(&divergence_lapse),
        make_not_null(&contracted_conformal_christoffel_second_kind),
        make_not_null(&d_contracted_conformal_christoffel_second_kind),
        make_not_null(&spatial_z4_constraint),
        make_not_null(&upper_spatial_z4_constraint),
        make_not_null(&grad_spatial_z4_constraint),
        make_not_null(&ricci_scalar_plus_divergence_z4_constraint),
        // fixed params
        c, cleaning_speed, one_over_relaxation_time,
        // free params
        eta, f_param, kappa_1, kappa_2, kappa_3, k_0,
        // evolved variables
        conformal_metric, lapse, shift, conformal_factor, a_tilde,
        trace_extrinsic_curvature, theta, gamma_hat, b,
        // auxiliary fields
        field_a, field_b, field_d, field_p,
        // symmetrized derivatives of auxiliary fields
        d_field_a, d_field_b, d_field_d, d_field_p,
        // derivatives of other evolved variables
        d_a_tilde, d_trace_extrinsic_curvature, d_theta, d_gamma_hat, d_b,
        // flags
        System::shifting_shift, evolve_lapse_and_shift);

    return {false};
  }
};
}  // namespace Ccz4::fd
