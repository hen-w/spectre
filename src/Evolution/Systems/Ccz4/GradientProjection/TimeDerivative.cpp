// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/TimeDerivative.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SoTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TempTags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection {
KernelInputs kernel_inputs(const Tags::ChartU::type& u,
                           const Tags::ChartW::type& w,
                           const Tags::ChartQ::type& q,
                           const DerivChartW::type& d_w,
                           const DerivChartQ::type& d_q) {
  const size_t number_of_points = get<0>(u).size();
  KernelInputs result{};
  const auto allocate = [number_of_points](auto& tensor) {
    for (auto& component : tensor) {
      component = DataVector{number_of_points};
    }
  };
  allocate(result.conformal_metric);
  allocate(result.conformal_factor);
  allocate(result.a_tilde);
  allocate(result.trace_extrinsic_curvature);
  allocate(result.theta);
  allocate(result.gamma_hat);
  allocate(result.lapse);
  allocate(result.shift);
  allocate(result.b);
  allocate(result.field_a);
  allocate(result.field_b);
  allocate(result.field_d);
  allocate(result.field_p);
  allocate(result.d_field_a);
  allocate(result.d_field_b);
  allocate(result.d_field_d);
  allocate(result.d_field_p);
  allocate(result.d_a_tilde);
  allocate(result.d_trace_extrinsic_curvature);
  allocate(result.d_theta);
  allocate(result.d_gamma_hat);
  allocate(result.d_b);

  for (size_t s = 0; s < number_of_points; ++s) {
    const PotentialVector u_s = potentials_at(u, s);
    const VelocityVector w_s = velocities_at(w, s);
    const GradientVector q_s = gradients_at(q, s);
    const LogMetricChart chart{stf_decode(&u_s[PotentialIndex::log_metric])};
    const Matrix3 metric = chart.metric();
    const Matrix3 a = stf_decode(&w_s[VelocityIndex::a]);
    const Matrix3 a_tilde = 0.5 * chart.tangent(a);
    const double lapse = u_s[PotentialIndex::lapse];
    std::array<Matrix3, 3> q_h{};
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(q_h, i) =
          stf_decode(&gsl::at(q_s, i)[PotentialIndex::log_metric]);
    }

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        result.conformal_metric.get(i, j)[s] = metric(i, j);
        result.a_tilde.get(i, j)[s] = a_tilde(i, j);
      }
    }
    get(result.conformal_factor)[s] =
        std::exp(u_s[PotentialIndex::log_conformal_factor]);
    get(result.lapse)[s] = lapse;
    get(result.trace_extrinsic_curvature)[s] =
        w_s[VelocityIndex::trace_extrinsic_curvature];
    get(result.theta)[s] = w_s[VelocityIndex::theta];
    for (size_t i = 0; i < 3; ++i) {
      result.shift.get(i)[s] = gsl::at(u_s, PotentialIndex::shift + i);
      result.gamma_hat.get(i)[s] = gsl::at(w_s, VelocityIndex::gamma_hat + i);
      result.b.get(i)[s] = gsl::at(w_s, VelocityIndex::b + i);
    }

    // Auxiliary fields.
    for (size_t i = 0; i < 3; ++i) {
      const PotentialVector& q_i = gsl::at(q_s, i);
      result.field_a.get(i)[s] = q_i[PotentialIndex::lapse] / lapse;
      result.field_p.get(i)[s] = q_i[PotentialIndex::log_conformal_factor];
      for (size_t k = 0; k < 3; ++k) {
        result.field_b.get(i, k)[s] = gsl::at(q_i, PotentialIndex::shift + k);
      }
      const Matrix3 field_d = 0.5 * chart.tangent(gsl::at(q_h, i));
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
            0.5 * (dq(i, j, PotentialIndex::log_conformal_factor) +
                   dq(j, i, PotentialIndex::log_conformal_factor));
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
            0.5 * (0.5 * chart.tangent(stf_decode(d_q_h_ij.data()) +
                                       stf_decode(d_q_h_ji.data())) +
                   chart.second_derivative(gsl::at(q_h, i), gsl::at(q_h, j)));
        for (size_t k = 0; k < 3; ++k) {
          for (size_t l = k; l < 3; ++l) {
            result.d_field_d.get(i, j, k, l)[s] = d_field_d(k, l);
          }
        }
      }
    }

    // Derivatives of the chart velocities.
    for (size_t k = 0; k < 3; ++k) {
      std::array<double, 5> d_a{};
      for (size_t c = 0; c < 5; ++c) {
        gsl::at(d_a, c) = d_w.get(k, VelocityIndex::a + c)[s];
      }
      const Matrix3 d_a_tilde =
          0.5 * (chart.tangent(stf_decode(d_a.data())) +
                 chart.second_derivative(gsl::at(q_h, k), a));
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          result.d_a_tilde.get(k, i, j)[s] = d_a_tilde(i, j);
        }
      }
      result.d_trace_extrinsic_curvature.get(k)[s] =
          d_w.get(k, VelocityIndex::trace_extrinsic_curvature)[s];
      result.d_theta.get(k)[s] = d_w.get(k, VelocityIndex::theta)[s];
      for (size_t i = 0; i < 3; ++i) {
        result.d_gamma_hat.get(k, i)[s] =
            d_w.get(k, VelocityIndex::gamma_hat + i)[s];
        result.d_b.get(k, i)[s] = d_w.get(k, VelocityIndex::b + i)[s];
      }
    }
  }
  return result;
}

KernelRates kernel_rates(const KernelInputs& inputs,
                         const Scalar<DataVector>& kappa_1,
                         const Scalar<DataVector>& kappa_2,
                         const double kappa_3, const Scalar<DataVector>& eta,
                         const Scalar<DataVector>& k_0, const double f) {
  constexpr size_t Dim = 3;
  const size_t number_of_points = get(inputs.lapse).size();

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

  tnsr::ii<DataVector, Dim> dt_conformal_metric(number_of_points);
  Scalar<DataVector> dt_lapse(number_of_points);
  tnsr::I<DataVector, Dim> dt_shift(number_of_points);
  Scalar<DataVector> dt_conformal_factor(number_of_points);
  tnsr::ii<DataVector, Dim> dt_a_tilde(number_of_points);
  Scalar<DataVector> dt_trace_extrinsic_curvature(number_of_points);
  Scalar<DataVector> dt_theta(number_of_points);
  tnsr::I<DataVector, Dim> dt_gamma_hat(number_of_points);
  tnsr::I<DataVector, Dim> dt_b(number_of_points);

  constexpr double c = 1.0;
  constexpr double cleaning_speed = 1.0;
  constexpr double one_over_relaxation_time = 0.0;
  constexpr bool shifting_shift = false;
  constexpr bool evolve_lapse_and_shift = true;
  ::Ccz4::fd::detail::apply<Dim>(
      make_not_null(&dt_conformal_metric), make_not_null(&dt_lapse),
      make_not_null(&dt_shift), make_not_null(&dt_conformal_factor),
      make_not_null(&dt_a_tilde), make_not_null(&dt_trace_extrinsic_curvature),
      make_not_null(&dt_theta), make_not_null(&dt_gamma_hat),
      make_not_null(&dt_b), make_not_null(&conformal_factor_squared),
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
      kappa_3, k_0, inputs.conformal_metric, inputs.lapse, inputs.shift,
      inputs.conformal_factor, inputs.a_tilde, inputs.trace_extrinsic_curvature,
      inputs.theta, inputs.gamma_hat, inputs.b, inputs.field_a, inputs.field_b,
      inputs.field_d, inputs.field_p, inputs.d_field_a, inputs.d_field_b,
      inputs.d_field_d, inputs.d_field_p, inputs.d_a_tilde,
      inputs.d_trace_extrinsic_curvature, inputs.d_theta, inputs.d_gamma_hat,
      inputs.d_b, shifting_shift, evolve_lapse_and_shift);

  return {std::move(dt_a_tilde), std::move(dt_trace_extrinsic_curvature),
          std::move(dt_theta), std::move(dt_gamma_hat), std::move(dt_b)};
}

void volume_targets(const gsl::not_null<Tags::ChartW::type*> dt_w,
                    const gsl::not_null<Tags::ChartQ::type*> dt_q,
                    const gsl::not_null<Tags::ChartU::type*> dt_u,
                    const Tags::ChartU::type& u, const Tags::ChartW::type& w,
                    const Tags::ChartQ::type& q, const DerivChartW::type& d_w,
                    const DerivChartQ::type& d_q,
                    const Scalar<DataVector>& kappa_1,
                    const Scalar<DataVector>& kappa_2, const double kappa_3,
                    const Scalar<DataVector>& eta,
                    const Scalar<DataVector>& k_0, const double f) {
  const size_t number_of_points = get<0>(u).size();
  const KernelRates rates = kernel_rates(
      kernel_inputs(u, w, q, d_w, d_q), kappa_1, kappa_2, kappa_3, eta, k_0, f);
  const auto& dt_a_tilde = rates.a_tilde;
  const auto& dt_trace_extrinsic_curvature = rates.trace_extrinsic_curvature;
  const auto& dt_theta = rates.theta;
  const auto& dt_gamma_hat = rates.gamma_hat;
  const auto& dt_b = rates.b;

  for (size_t s = 0; s < number_of_points; ++s) {
    const PotentialVector u_s = potentials_at(u, s);
    const VelocityVector w_s = velocities_at(w, s);
    const GradientVector q_s = gradients_at(q, s);
    const LogMetricChart chart{stf_decode(&u_s[PotentialIndex::log_metric])};
    const PotentialVector rhs = potential_rhs(chart, u_s, w_s, q_s, f);

    // Chart rate of a from the physical ATilde rate.
    Matrix3 physical_a_tilde_rate{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        physical_a_tilde_rate(i, j) = dt_a_tilde.get(i, j)[s];
      }
    }
    const std::array<double, 5> a_rate = stf_encode(chart.inverse_tangent(
        2.0 * physical_a_tilde_rate -
        chart.second_derivative(stf_decode(&rhs[PotentialIndex::log_metric]),
                                stf_decode(&w_s[VelocityIndex::a]))));

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
        get(dt_trace_extrinsic_curvature)[s];
    dt_w->get(VelocityIndex::theta)[s] = get(dt_theta)[s];
    for (size_t k = 0; k < 3; ++k) {
      double ordering = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        ordering += inverse_metric(k, i) * gsl::at(curl_covector, i);
      }
      ordering /= 6.0;
      dt_w->get(VelocityIndex::gamma_hat + k)[s] =
          dt_gamma_hat.get(k)[s] + ordering;
      dt_w->get(VelocityIndex::b + k)[s] = dt_b.get(k)[s] + ordering;
    }

    // Gradient targets R_u q_i + R(u, D_i w, D_i q).
    for (size_t i = 0; i < 3; ++i) {
      VelocityVector d_i_w{};
      for (size_t a_index = 0; a_index < number_of_velocities; ++a_index) {
        gsl::at(d_i_w, a_index) = d_w.get(i, a_index)[s];
      }
      GradientVector d_i_q{};
      for (size_t j = 0; j < 3; ++j) {
        for (size_t a_index = 0; a_index < number_of_potentials; ++a_index) {
          gsl::at(gsl::at(d_i_q, j), a_index) = d_q.get(i, j, a_index)[s];
        }
      }
      const PotentialVector principal =
          potential_rhs(chart, u_s, d_i_w, d_i_q, f);
      const PotentialVector lower_order = potential_rhs_potential_derivative(
          chart, u_s, w_s, q_s, gsl::at(q_s, i));
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
}  // namespace Ccz4::GradientProjection
