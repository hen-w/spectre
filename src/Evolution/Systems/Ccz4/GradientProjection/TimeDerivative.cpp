// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/TimeDerivative.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SoTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Filter.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TempTags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection {
KernelInputs kernel_inputs(const PotentialFields& u,
                           const Tags::TransformedATilde::type& a,
                           const Tags::PotentialGradients::type& q,
                           const DerivTransformedATilde::type& d_a,
                           const DerivPotentialGradients::type& d_q) {
  return kernel_inputs(u, log_metric_charts(u), a, q, d_a, d_q);
}

KernelInputs kernel_inputs(const PotentialFields& u,
                           const std::vector<LogMetricChart>& charts,
                           const Tags::TransformedATilde::type& a,
                           const Tags::PotentialGradients::type& q,
                           const DerivTransformedATilde::type& d_a,
                           const DerivPotentialGradients::type& d_q) {
  const size_t number_of_points = get(u.lapse).size();
  KernelInputs result{};
  const auto allocate = [number_of_points](auto& tensor) {
    for (auto& component : tensor) {
      component = DataVector{number_of_points};
    }
  };
  allocate(result.field_a);
  allocate(result.field_d);
  allocate(result.field_p);
  allocate(result.d_field_a);
  allocate(result.d_field_b);
  allocate(result.d_field_d);
  allocate(result.d_field_p);
  allocate(result.d_a_tilde);
  // FieldB is the shift gradient itself.
  for (size_t i = 0; i < 3; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      result.field_b.get(i, k).set_data_ref(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<double*>(q.get(i, PotentialIndex::shift + k).data()),
          number_of_points);
    }
  }

  for (size_t s = 0; s < number_of_points; ++s) {
    const PotentialVector u_s = potentials_at(u, s);
    const GradientVector q_s = gradients_at(q, s);
    const LogMetricChart& chart = charts[s];
    std::array<double, 5> a_s{};
    for (size_t c = 0; c < 5; ++c) {
      gsl::at(a_s, c) = a.get(c)[s];
    }
    const double lapse = u_s[PotentialIndex::lapse];
    const double conformal_factor = u_s[PotentialIndex::conformal_factor];
    // a and the D_i H in the eigenbasis of H, each converted once.
    const Matrix3 a_hat = chart.to_eigenbasis(stf_decode(a_s.data()));
    std::array<Matrix3, 3> q_h_hat{};
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(q_h_hat, i) = chart.to_eigenbasis(
          stf_decode(&gsl::at(q_s, i)[PotentialIndex::log_metric]));
    }

    // Auxiliary fields.
    for (size_t i = 0; i < 3; ++i) {
      const PotentialVector& q_i = gsl::at(q_s, i);
      result.field_a.get(i)[s] = q_i[PotentialIndex::lapse] / lapse;
      result.field_p.get(i)[s] =
          q_i[PotentialIndex::conformal_factor] / conformal_factor;
      const Matrix3 field_d =
          0.5 *
          chart.from_eigenbasis(chart.eigenbasis_tangent(gsl::at(q_h_hat, i)));
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = j; k < 3; ++k) {
          result.field_d.get(i, j, k)[s] = field_d(j, k);
        }
      }
    }

    // Symmetrized derivatives of the auxiliary fields.
    const auto dq = [&d_q, s](const size_t i, const size_t j,
                              const size_t component) {
      return d_q.get(i, j, component)[s];
    };
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        result.d_field_a.get(i, j)[s] =
            0.5 *
                (dq(i, j, PotentialIndex::lapse) +
                 dq(j, i, PotentialIndex::lapse)) /
                lapse -
            gsl::at(q_s, i)[PotentialIndex::lapse] *
                gsl::at(q_s, j)[PotentialIndex::lapse] / square(lapse);
        result.d_field_p.get(i, j)[s] =
            0.5 *
                (dq(i, j, PotentialIndex::conformal_factor) +
                 dq(j, i, PotentialIndex::conformal_factor)) /
                conformal_factor -
            gsl::at(q_s, i)[PotentialIndex::conformal_factor] *
                gsl::at(q_s, j)[PotentialIndex::conformal_factor] /
                square(conformal_factor);
        for (size_t k = 0; k < 3; ++k) {
          result.d_field_b.get(i, j, k)[s] =
              0.5 * (dq(i, j, PotentialIndex::shift + k) +
                     dq(j, i, PotentialIndex::shift + k));
        }
        std::array<double, 5> d_q_h_ij{};
        std::array<double, 5> d_q_h_ji{};
        for (size_t c = 0; c < 5; ++c) {
          gsl::at(d_q_h_ij, c) = dq(i, j, PotentialIndex::log_metric + c);
          gsl::at(d_q_h_ji, c) = dq(j, i, PotentialIndex::log_metric + c);
        }
        const Matrix3 d_field_d =
            0.5 * chart.from_eigenbasis(
                      0.5 * chart.eigenbasis_tangent(chart.to_eigenbasis(
                                stf_decode(d_q_h_ij.data()) +
                                stf_decode(d_q_h_ji.data()))) +
                      chart.eigenbasis_second_derivative(gsl::at(q_h_hat, i),
                                                         gsl::at(q_h_hat, j)));
        for (size_t k = 0; k < 3; ++k) {
          for (size_t l = k; l < 3; ++l) {
            result.d_field_d.get(i, j, k, l)[s] = d_field_d(k, l);
          }
        }
      }
    }

    // The derivative of the curvature from that of the nodal a.
    for (size_t k = 0; k < 3; ++k) {
      std::array<double, 5> d_a_k{};
      for (size_t c = 0; c < 5; ++c) {
        gsl::at(d_a_k, c) = d_a.get(k, c)[s];
      }
      const Matrix3 d_a_tilde =
          0.5 *
          chart.from_eigenbasis(
              chart.eigenbasis_tangent(
                  chart.to_eigenbasis(stf_decode(d_a_k.data()))) +
              chart.eigenbasis_second_derivative(gsl::at(q_h_hat, k), a_hat));
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          result.d_a_tilde.get(k, i, j)[s] = d_a_tilde(i, j);
        }
      }
    }
  }
  return result;
}

KernelRates kernel_rates(const KernelInputs& inputs,
                         const tnsr::ii<DataVector, 3>& conformal_metric,
                         const tnsr::ii<DataVector, 3>& a_tilde,
                         const PotentialFields& u, const VelocityFields& w,
                         const VelocityDerivatives& d_w,
                         const Scalar<DataVector>& kappa_1,
                         const Scalar<DataVector>& kappa_2,
                         const double kappa_3, const Scalar<DataVector>& eta,
                         const Scalar<DataVector>& k_0, const double f) {
  constexpr size_t Dim = 3;
  const size_t number_of_points = get(u.lapse).size();

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
      ::Ccz4::Tags::DerivGammaHatMinusContractedConformalChristoffel<DataVector,
                                                                     Dim>,
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
  TempVars temp_vars(number_of_points);
  tnsr::I<DataVector, Dim> upper_spatial_z4_constraint(number_of_points);
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
         k_minus_k0_minus_2_theta_c, lapse_times_a_tilde, lapse_times_d_a_tilde,
         lapse_times_field_a, lapse_times_conformal_spatial_metric,
         lapse_times_slicing_condition,
         lapse_times_ricci_scalar_plus_divergence_z4_constraint,
         shift_times_deriv_gamma_hat, inv_tau_times_conformal_metric,
         trace_a_tilde, field_d_up, conformal_christoffel_second_kind,
         d_conformal_christoffel_second_kind, christoffel_second_kind,
         spatial_ricci_tensor, grad_grad_lapse, divergence_lapse,
         contracted_conformal_christoffel_second_kind,
         d_contracted_conformal_christoffel_second_kind, spatial_z4_constraint,
         grad_spatial_z4_constraint,
         ricci_scalar_plus_divergence_z4_constraint] = temp_vars;

  KernelRates rates{tnsr::ii<DataVector, Dim>(number_of_points),
                    Scalar<DataVector>(number_of_points),
                    tnsr::ii<DataVector, Dim>(number_of_points),
                    Scalar<DataVector>(number_of_points),
                    Scalar<DataVector>(number_of_points),
                    tnsr::I<DataVector, Dim>(number_of_points),
                    Scalar<DataVector>(number_of_points),
                    tnsr::I<DataVector, Dim>(number_of_points),
                    tnsr::I<DataVector, Dim>(number_of_points)};

  constexpr double c = 1.0;
  constexpr double cleaning_speed = 1.0;
  constexpr double one_over_relaxation_time = 0.0;
  constexpr bool shifting_shift = false;
  constexpr bool evolve_lapse_and_shift = true;
  ::Ccz4::fd::detail::apply<Dim>(
      make_not_null(&rates.conformal_metric), make_not_null(&rates.lapse),
      make_not_null(&rates.shift), make_not_null(&rates.conformal_factor),
      make_not_null(&rates.a_tilde),
      make_not_null(&rates.trace_extrinsic_curvature),
      make_not_null(&rates.theta), make_not_null(&rates.gamma_hat),
      make_not_null(&rates.b), make_not_null(&conformal_factor_squared),
      make_not_null(&det_conformal_spatial_metric),
      make_not_null(&inv_conformal_spatial_metric),
      make_not_null(&inv_spatial_metric), make_not_null(&inv_a_tilde),
      make_not_null(&a_tilde_times_field_b),
      make_not_null(
          &a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde),
      make_not_null(&contracted_field_b), make_not_null(&symmetrized_d_field_b),
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
      make_not_null(&ricci_scalar_plus_divergence_z4_constraint), c,
      cleaning_speed, one_over_relaxation_time, eta, f, kappa_1, kappa_2,
      kappa_3, k_0, conformal_metric, u.lapse, u.shift, u.conformal_factor,
      a_tilde, w.trace_extrinsic_curvature, w.theta, w.gamma_hat, w.b,
      inputs.field_a, inputs.field_b, inputs.field_d, inputs.field_p,
      inputs.d_field_a, inputs.d_field_b, inputs.d_field_d, inputs.d_field_p,
      inputs.d_a_tilde, d_w.trace_extrinsic_curvature, d_w.theta, d_w.gamma_hat,
      d_w.b, shifting_shift, evolve_lapse_and_shift);

  return rates;
}

void volume_targets(const gsl::not_null<Tags::VelocityTarget::type*> dt_w,
                    const gsl::not_null<Tags::GradientTarget::type*> dt_q,
                    const gsl::not_null<Tags::PotentialRateTarget::type*> dt_u,
                    const tnsr::ii<DataVector, 3>& conformal_metric,
                    const tnsr::ii<DataVector, 3>& a_tilde,
                    const PotentialFields& u, const VelocityFields& w,
                    const Tags::PotentialGradients::type& q,
                    const VelocityDerivatives& d_w,
                    const DerivPotentialGradients::type& d_q,
                    const Scalar<DataVector>& kappa_1,
                    const Scalar<DataVector>& kappa_2, const double kappa_3,
                    const Scalar<DataVector>& eta,
                    const Scalar<DataVector>& k_0, const double f) {
  volume_targets(dt_w, dt_q, dt_u, conformal_metric, a_tilde, u,
                 log_metric_charts(u), w, q, d_w, d_q, kappa_1, kappa_2,
                 kappa_3, eta, k_0, f);
}

void volume_targets(
    const gsl::not_null<Tags::VelocityTarget::type*> dt_w,
    const gsl::not_null<Tags::GradientTarget::type*> dt_q,
    const gsl::not_null<Tags::PotentialRateTarget::type*> dt_u,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::ii<DataVector, 3>& a_tilde, const PotentialFields& u,
    const std::vector<LogMetricChart>& charts, const VelocityFields& w,
    const Tags::PotentialGradients::type& q, const VelocityDerivatives& d_w,
    const DerivPotentialGradients::type& d_q, const Scalar<DataVector>& kappa_1,
    const Scalar<DataVector>& kappa_2, const double kappa_3,
    const Scalar<DataVector>& eta, const Scalar<DataVector>& k_0,
    const double f) {
  const size_t number_of_points = get(u.lapse).size();
  const KernelRates rates = kernel_rates(
      kernel_inputs(u, charts, w.a, q, d_w.a, d_q), conformal_metric, a_tilde,
      u, w, d_w, kappa_1, kappa_2, kappa_3, eta, k_0, f);

  for (size_t s = 0; s < number_of_points; ++s) {
    const PotentialVector u_s = potentials_at(u, s);
    const VelocityVector w_s = velocities_at(w, s);
    const GradientVector q_s = gradients_at(q, s);
    const LogMetricChart& chart = charts[s];
    const ShiftGradientTerms shift_terms{chart, q_s};
    const PotentialVector rhs =
        potential_rhs(chart, shift_terms, u_s, w_s, q_s, f);

    // Chart rate of a from the physical ATilde rate, in the eigenbasis of H.
    Matrix3 physical_a_tilde_rate{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        physical_a_tilde_rate(i, j) = rates.a_tilde.get(i, j)[s];
      }
    }
    const std::array<double, 5> a_rate =
        stf_encode(chart.from_eigenbasis(chart.eigenbasis_inverse_tangent(
            2.0 * chart.to_eigenbasis(physical_a_tilde_rate) -
            chart.eigenbasis_second_derivative(
                chart.to_eigenbasis(
                    stf_decode(&rhs[PotentialIndex::log_metric])),
                chart.to_eigenbasis(stf_decode(&w_s[VelocityIndex::a]))))));

    // Ordering term (1/6) g^{ki} (D_i B_j^j - D_j B_i^j).
    std::array<double, 3> curl_covector{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        gsl::at(curl_covector, i) +=
            d_q.get(i, j, PotentialIndex::shift + j)[s] -
            d_q.get(j, i, PotentialIndex::shift + j)[s];
      }
    }
    const Matrix3 inverse_metric = chart.inverse_metric();

    for (size_t c_index = 0; c_index < 5; ++c_index) {
      dt_w->get(VelocityIndex::a + c_index)[s] = gsl::at(a_rate, c_index);
    }
    dt_w->get(VelocityIndex::trace_extrinsic_curvature)[s] =
        get(rates.trace_extrinsic_curvature)[s];
    dt_w->get(VelocityIndex::theta)[s] = get(rates.theta)[s];
    for (size_t k = 0; k < 3; ++k) {
      double ordering = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        ordering += inverse_metric(k, i) * gsl::at(curl_covector, i);
      }
      ordering /= 6.0;
      dt_w->get(VelocityIndex::gamma_hat + k)[s] =
          rates.gamma_hat.get(k)[s] + ordering;
      dt_w->get(VelocityIndex::b + k)[s] = rates.b.get(k)[s] + ordering;
    }

    // Gradient targets R_u q_i + R(u, D_i w, D_i q).
    for (size_t i = 0; i < 3; ++i) {
      const PotentialVector principal =
          potential_rhs(chart, u_s, velocity_derivatives_at(d_w, i, s),
                        gradient_derivatives_at(d_q, i, s), f);
      const PotentialVector lower_order = potential_rhs_potential_derivative(
          chart, shift_terms, u_s, w_s, q_s, gsl::at(q_s, i));
      for (size_t a_index = 0; a_index < number_of_potentials; ++a_index) {
        dt_q->get(i, a_index)[s] =
            gsl::at(principal, a_index) + gsl::at(lower_order, a_index);
      }
    }
    for (size_t a_index = 0; a_index < number_of_potentials; ++a_index) {
      dt_u->get(a_index)[s] = gsl::at(rhs, a_index);
    }
  }
}

void ComputeVolumeTargets::apply(
    const gsl::not_null<typename variables_tag::type*> evolved_variables,
    const gsl::not_null<typename dt_variables_tag::type*> dt_evolved_variables,
    const gsl::not_null<typename Tags::RhsTargets::type*> targets,
    const gsl::not_null<typename Tags::StepChartFields::type*> chart_fields,
    const gsl::not_null<std::vector<LogMetricChart>*> charts,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const Scalar<DataVector>& kappa_1, const Scalar<DataVector>& kappa_2,
    const double kappa_3, const Scalar<DataVector>& eta,
    const Scalar<DataVector>& k_0, const LocalOperators& local_operators,
    const double top_shell_filter_gain) {
  if (max(abs(get(k_0))) != 0.0) {
    ERROR(
        "The CCZ4 gradient-projection scheme requires K0 = 0: its potential "
        "right-hand side and the lapse rate recovered from it omit K0. Got "
        "max|K0| = "
        << max(abs(get(k_0))));
  }
  using ConformalFactor = ::Ccz4::Tags::ConformalFactor<DataVector>;
  using potential_gauge_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>>;
  using curvature_tags =
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 ::Ccz4::Tags::Theta<DataVector>,
                 ::Ccz4::Tags::GammaHat<DataVector, 3>>;
  using b_tags = tmpl::list<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>;
  const size_t number_of_points = mesh.number_of_grid_points();
  auto& vars = *evolved_variables;
  const auto& conformal_metric =
      get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(vars);
  const auto& conformal_factor = get<ConformalFactor>(vars);
  const auto& a_tilde = get<::Ccz4::Tags::ATilde<DataVector, 3>>(vars);
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);

  // 1. The local nodal H and a, and the charts of H.
  local_chart_fields(chart_fields, charts, conformal_metric, conformal_factor,
                     a_tilde, lapse);
  const LocalChartFields& chart = *chart_fields;

  // 2. The gradients of the potentials, differentiating the physical fields
  // in place.
  const auto d_log_conformal_metric = partial_derivatives<
      tmpl::list<Tags::LogConformalMetric>>(
      chart.template reference_subset<tmpl::list<Tags::LogConformalMetric>>(),
      mesh, inverse_jacobian);
  const auto d_conformal_factor =
      partial_derivatives<tmpl::list<ConformalFactor>>(
          vars.template reference_subset<tmpl::list<ConformalFactor>>(), mesh,
          inverse_jacobian);
  const auto d_potential_gauge = partial_derivatives<potential_gauge_tags>(
      vars.template reference_subset<potential_gauge_tags>(), mesh,
      inverse_jacobian);
  auto& gradients = get<Tags::PotentialGradients>(vars);
  assemble_potential_gradients(
      make_not_null(&gradients),
      get<::Tags::deriv<Tags::LogConformalMetric, tmpl::size_t<3>,
                        Frame::Inertial>>(d_log_conformal_metric),
      get<::Tags::deriv<ConformalFactor, tmpl::size_t<3>, Frame::Inertial>>(
          d_conformal_factor),
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>>(d_potential_gauge),
      get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(d_potential_gauge));

  // 3. The filter of the velocities at fixed potentials and gradients.
  const PotentialFields u{get<Tags::LogConformalMetric>(chart),
                          conformal_factor, lapse,
                          get<gr::Tags::Shift<DataVector, 3>>(vars)};
  filter_velocities(
      make_not_null(&get<Tags::TransformedATilde>(*chart_fields)),
      make_not_null(&get<::Ccz4::Tags::ATilde<DataVector, 3>>(vars)),
      make_not_null(&get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars)),
      make_not_null(&get<::Ccz4::Tags::Theta<DataVector>>(vars)),
      make_not_null(&get<::Ccz4::Tags::GammaHat<DataVector, 3>>(vars)),
      make_not_null(&get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(vars)),
      u, *charts, gradients, mesh, local_operators, top_shell_filter_gain,
      ::Ccz4::fd::System::f);

  // 4. The remaining derivatives, of the filtered velocities and of the
  // gradients.
  const auto d_a = partial_derivatives<tmpl::list<Tags::TransformedATilde>>(
      chart.template reference_subset<tmpl::list<Tags::TransformedATilde>>(),
      mesh, inverse_jacobian);
  const auto d_curvature = partial_derivatives<curvature_tags>(
      vars.template reference_subset<curvature_tags>(), mesh, inverse_jacobian);
  const auto d_b = partial_derivatives<b_tags>(
      vars.template reference_subset<b_tags>(), mesh, inverse_jacobian);
  const auto d_gradients = partial_derivatives<
      tmpl::list<Tags::PotentialGradients>>(
      vars.template reference_subset<tmpl::list<Tags::PotentialGradients>>(),
      mesh, inverse_jacobian);

  // 5. The volume targets.
  const VelocityFields w{
      get<Tags::TransformedATilde>(chart),
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars),
      get<::Ccz4::Tags::Theta<DataVector>>(vars),
      get<::Ccz4::Tags::GammaHat<DataVector, 3>>(vars),
      get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(vars)};
  const VelocityDerivatives d_w{
      get<DerivTransformedATilde>(d_a),
      get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>(d_curvature),
      get<::Tags::deriv<::Ccz4::Tags::Theta<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>>(d_curvature),
      get<::Tags::deriv<::Ccz4::Tags::GammaHat<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(d_curvature),
      get<::Tags::deriv<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>,
                        tmpl::size_t<3>, Frame::Inertial>>(d_b)};
  targets->initialize(number_of_points);
  volume_targets(make_not_null(&get<Tags::VelocityTarget>(*targets)),
                 make_not_null(&get<Tags::GradientTarget>(*targets)),
                 make_not_null(&get<Tags::PotentialRateTarget>(*targets)),
                 conformal_metric, a_tilde, u, *charts, w, gradients, d_w,
                 get<DerivPotentialGradients>(d_gradients), kappa_1, kappa_2,
                 kappa_3, eta, k_0, ::Ccz4::fd::System::f);

  // 6. Only FinalizeTimeDerivative writes physical rates.
  dt_evolved_variables->initialize(number_of_points, 0.0);
}

void ComputeVolumeTargets::apply(
    const gsl::not_null<typename variables_tag::type*> evolved_variables,
    const gsl::not_null<typename dt_variables_tag::type*> dt_evolved_variables,
    const gsl::not_null<typename Tags::RhsTargets::type*> targets,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const Scalar<DataVector>& kappa_1, const Scalar<DataVector>& kappa_2,
    const double kappa_3, const Scalar<DataVector>& eta,
    const Scalar<DataVector>& k_0, const LocalOperators& local_operators,
    const double top_shell_filter_gain) {
  typename Tags::StepChartFields::type chart_fields{};
  std::vector<LogMetricChart> charts{};
  apply(evolved_variables, dt_evolved_variables, targets,
        make_not_null(&chart_fields), make_not_null(&charts), mesh,
        inverse_jacobian, kappa_1, kappa_2, kappa_3, eta, k_0, local_operators,
        top_shell_filter_gain);
}
}  // namespace Ccz4::GradientProjection
