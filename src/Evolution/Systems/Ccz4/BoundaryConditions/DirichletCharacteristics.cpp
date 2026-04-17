// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/BoundaryConditions/DirichletCharacteristics.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Characteristics.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
// Helper: compute time derivatives of the four second-order fields
// (conformal_metric, conformal_factor, lapse, shift) using the CCZ4 evolution
// equations. Replicates the logic from SoTimeDerivative.hpp lines 396-431.
// Hardcoded constants: c=1.0, lapse_times_slicing_condition=2.0,
// one_over_relaxation_time=0.0 (relaxation term omitted).
void compute_dt_second_order_fields(
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        dt_conformal_metric,
    const gsl::not_null<Scalar<DataVector>*> dt_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*> dt_lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> dt_shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_input,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift_input,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& a_tilde,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& theta,
    const tnsr::I<DataVector, 3, Frame::Inertial>& b,
    const tnsr::i<DataVector, 3, Frame::Inertial>& field_a,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& field_b,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& field_d,
    const tnsr::i<DataVector, 3, Frame::Inertial>& field_p,
    const Scalar<DataVector>& k_0, const double f, const bool shifting_shift) {
  static constexpr double one_third = 1.0 / 3.0;
  // c = 1.0, lapse_times_slicing_condition = 2.0
  static constexpr double c = 1.0;
  static constexpr double lapse_times_slicing_cond = 2.0;

  // Intermediates
  Scalar<DataVector> contracted_field_b{};
  ::tenex::evaluate(make_not_null(&contracted_field_b), field_b(ti::k, ti::K));

  tnsr::ij<DataVector, 3, Frame::Inertial> conformal_metric_times_field_b{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&conformal_metric_times_field_b),
      conformal_metric(ti::k, ti::i) * field_b(ti::j, ti::K));

  // trace_a_tilde = 0 (SO-CCZ4 identity, a_tilde is trace-free)
  // So a_tilde_minus_one_third_cm_trace = a_tilde

  // k_minus_k0_minus_2_theta_c = K - k_0 - 2*c*theta
  Scalar<DataVector> k_minus_k0_minus_2_theta_c{};
  ::tenex::evaluate(make_not_null(&k_minus_k0_minus_2_theta_c),
                    trace_extrinsic_curvature() - k_0() - 2.0 * c * theta());

  // inv_tau_times_conformal_metric = 0 (one_over_relaxation_time = 0)
  // det_conformal_metric term vanishes

  // eq 12a: dt conformal metric (with tau^{-1}=0)
  ::tenex::evaluate<ti::i, ti::j>(
      dt_conformal_metric,
      2.0 * shift_input(ti::K) * field_d(ti::k, ti::i, ti::j) +
          conformal_metric_times_field_b(ti::i, ti::j) +
          conformal_metric_times_field_b(ti::j, ti::i) -
          2.0 * one_third * conformal_metric(ti::i, ti::j) *
              contracted_field_b() -
          2.0 * lapse_input() * a_tilde(ti::i, ti::j));

  // dt lapse
  ::tenex::evaluate<>(
      dt_lapse, (shift_input(ti::K) * field_a(ti::k) -
                 lapse_times_slicing_cond * k_minus_k0_minus_2_theta_c()) *
                    lapse_input());

  // dt shift
  ::tenex::evaluate<ti::I>(dt_shift, f * b(ti::I));
  if (shifting_shift) {
    ::tenex::update<ti::I>(
        dt_shift,
        (*dt_shift)(ti::I) + shift_input(ti::K) * field_b(ti::k, ti::I));
  }

  // dt conformal factor
  ::tenex::evaluate(dt_conformal_factor,
                    (shift_input(ti::K) * field_p(ti::k) +
                     one_third * (lapse_input() * trace_extrinsic_curvature() -
                                  contracted_field_b())) *
                        conformal_factor());
}

// Helper: perform characteristic mode mixing. Replaces incoming (or outgoing)
// modes in char_fields with the corresponding modes from ghost_char_fields.
template <typename CharFieldsType>
void mix_characteristic_modes(CharFieldsType& char_fields,
                              const CharFieldsType& ghost_char_fields,
                              const std::array<DataVector, 16>& char_speeds,
                              const bool prescribe_outgoing,
                              const size_t num_pts) {
  using namespace ::Ccz4::fd::Tags;
  static constexpr size_t Dim = 3;

  auto& u_tnsr_plus =
      get<UTensorPlus<DataVector, Dim, Frame::Inertial>>(char_fields);
  auto& u_tnsr_minus =
      get<UTensorMinus<DataVector, Dim, Frame::Inertial>>(char_fields);
  auto& u_vector1_zero =
      get<UVector1Zero<DataVector, Dim, Frame::Inertial>>(char_fields);
  auto& u_vector2_plus =
      get<UVector2Plus<DataVector, Dim, Frame::Inertial>>(char_fields);
  auto& u_vector2_minus =
      get<UVector2Minus<DataVector, Dim, Frame::Inertial>>(char_fields);
  auto& u_vector3_plus =
      get<UVector3Plus<DataVector, Dim, Frame::Inertial>>(char_fields);
  auto& u_vector3_minus =
      get<UVector3Minus<DataVector, Dim, Frame::Inertial>>(char_fields);
  auto& u_scalar1_zero = get<UScalar1Zero<DataVector>>(char_fields);
  auto& u_scalar2_plus = get<UScalar2Plus<DataVector>>(char_fields);
  auto& u_scalar2_minus = get<UScalar2Minus<DataVector>>(char_fields);
  auto& u_scalar3_plus = get<UScalar3Plus<DataVector>>(char_fields);
  auto& u_scalar3_minus = get<UScalar3Minus<DataVector>>(char_fields);
  auto& u_scalar4_plus = get<UScalar4Plus<DataVector>>(char_fields);
  auto& u_scalar4_minus = get<UScalar4Minus<DataVector>>(char_fields);
  auto& u_scalar5_plus = get<UScalar5Plus<DataVector>>(char_fields);
  auto& u_scalar5_minus = get<UScalar5Minus<DataVector>>(char_fields);

  const auto& g_u_tnsr_plus =
      get<UTensorPlus<DataVector, Dim, Frame::Inertial>>(ghost_char_fields);
  const auto& g_u_tnsr_minus =
      get<UTensorMinus<DataVector, Dim, Frame::Inertial>>(ghost_char_fields);
  const auto& g_u_vector1_zero =
      get<UVector1Zero<DataVector, Dim, Frame::Inertial>>(ghost_char_fields);
  const auto& g_u_vector2_plus =
      get<UVector2Plus<DataVector, Dim, Frame::Inertial>>(ghost_char_fields);
  const auto& g_u_vector2_minus =
      get<UVector2Minus<DataVector, Dim, Frame::Inertial>>(ghost_char_fields);
  const auto& g_u_vector3_plus =
      get<UVector3Plus<DataVector, Dim, Frame::Inertial>>(ghost_char_fields);
  const auto& g_u_vector3_minus =
      get<UVector3Minus<DataVector, Dim, Frame::Inertial>>(ghost_char_fields);
  const auto& g_u_scalar1_zero =
      get<UScalar1Zero<DataVector>>(ghost_char_fields);
  const auto& g_u_scalar2_plus =
      get<UScalar2Plus<DataVector>>(ghost_char_fields);
  const auto& g_u_scalar2_minus =
      get<UScalar2Minus<DataVector>>(ghost_char_fields);
  const auto& g_u_scalar3_plus =
      get<UScalar3Plus<DataVector>>(ghost_char_fields);
  const auto& g_u_scalar3_minus =
      get<UScalar3Minus<DataVector>>(ghost_char_fields);
  const auto& g_u_scalar4_plus =
      get<UScalar4Plus<DataVector>>(ghost_char_fields);
  const auto& g_u_scalar4_minus =
      get<UScalar4Minus<DataVector>>(ghost_char_fields);
  const auto& g_u_scalar5_plus =
      get<UScalar5Plus<DataVector>>(ghost_char_fields);
  const auto& g_u_scalar5_minus =
      get<UScalar5Minus<DataVector>>(ghost_char_fields);

  if (not prescribe_outgoing) {
    // Replace INCOMING modes with ghost values
    u_tnsr_minus = g_u_tnsr_minus;
    for (size_t s = 0; s < num_pts; ++s) {
      if (char_speeds[2][s] < 0.0) {
        for (size_t i = 0; i < Dim; ++i) {
          u_vector1_zero.get(i)[s] = g_u_vector1_zero.get(i)[s];
        }
      }
    }
    u_vector2_minus = g_u_vector2_minus;
    u_vector3_minus = g_u_vector3_minus;
    for (size_t s = 0; s < num_pts; ++s) {
      if (char_speeds[7][s] < 0.0) {
        get(u_scalar1_zero)[s] = get(g_u_scalar1_zero)[s];
      }
    }
    u_scalar2_minus = g_u_scalar2_minus;
    u_scalar3_minus = g_u_scalar3_minus;
    u_scalar4_minus = g_u_scalar4_minus;
    u_scalar5_minus = g_u_scalar5_minus;
  } else {
    // Replace OUTGOING modes with ghost values
    u_tnsr_plus = g_u_tnsr_plus;
    for (size_t s = 0; s < num_pts; ++s) {
      if (char_speeds[2][s] >= 0.0) {
        for (size_t i = 0; i < Dim; ++i) {
          u_vector1_zero.get(i)[s] = g_u_vector1_zero.get(i)[s];
        }
      }
    }
    u_vector2_plus = g_u_vector2_plus;
    u_vector3_plus = g_u_vector3_plus;
    for (size_t s = 0; s < num_pts; ++s) {
      if (char_speeds[7][s] >= 0.0) {
        get(u_scalar1_zero)[s] = get(g_u_scalar1_zero)[s];
      }
    }
    u_scalar2_plus = g_u_scalar2_plus;
    u_scalar3_plus = g_u_scalar3_plus;
    u_scalar4_plus = g_u_scalar4_plus;
    u_scalar5_plus = g_u_scalar5_plus;
  }
}
// Result of the characteristic decomposition + mode mixing + inverse transform
// + auxiliary reconstruction pipeline.
struct CharMixedState {
  std::array<DataVector, 16> char_speeds;
  // Evolved vars + normal derivatives from inverse char transform
  ::Ccz4::fd::Tags::EvolvedSpaceFromCharacteristicFields<
      DataVector, 3, Frame::Inertial>::type evolved_space;
  // Reconstructed auxiliary fields
  tnsr::i<DataVector, 3, Frame::Inertial> field_a;
  tnsr::iJ<DataVector, 3, Frame::Inertial> field_b;
  tnsr::ijj<DataVector, 3, Frame::Inertial> field_d;
  tnsr::i<DataVector, 3, Frame::Inertial> field_p;
};

// Performs the full characteristic mode-mixing pipeline shared by dg_ghost
// and dg_time_derivative:
// 1. Reconstructs interior spatial derivatives from interior auxiliary fields
// 2. Evaluates analytic solution
// 3. Computes analytic unit normal (one form + vector)
// 4. Reconstructs analytic spatial derivatives
// 5. Computes char speeds (interior lapse/shift/cf, analytic unit normal)
// 6. Interior forward char transform (analytic four fields, interior evolved/deriv)
// 7. Ghost forward char transform + mode mixing
// 8. Inverse char transform (analytic four fields + analytic unit normal)
// 9. Reconstructs auxiliary fields (analytic unit normal, boundary-integrated lapse/cf)
CharMixedState characteristic_decomposition_pipeline(
    const evolution::initial_data::InitialData& analytic_prescription,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const Scalar<DataVector>& interior_conformal_factor,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_a_tilde,
    const Scalar<DataVector>& interior_trace_extrinsic_curvature,
    const Scalar<DataVector>& interior_theta,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_gamma_hat,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_auxiliary_shift_b,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interior_field_a,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& interior_field_b,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& interior_field_d,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interior_field_p,
    const Scalar<DataVector>& boundary_integrated_lapse,
    const Scalar<DataVector>& boundary_integrated_conformal_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
    const double f, const bool prescribe_outgoing) {
  static constexpr size_t Dim = 3;

  // Step 1: Reconstruct interior spatial derivatives from auxiliary fields
  tnsr::ijj<DataVector, Dim, Frame::Inertial> d_conformal_metric{};
  ::tenex::evaluate<ti::k, ti::i, ti::j>(
      make_not_null(&d_conformal_metric),
      2.0 * interior_field_d(ti::k, ti::i, ti::j));

  tnsr::i<DataVector, Dim, Frame::Inertial> d_conformal_factor{};
  ::tenex::evaluate<ti::i>(
      make_not_null(&d_conformal_factor),
      interior_conformal_factor() * interior_field_p(ti::i));

  tnsr::i<DataVector, Dim, Frame::Inertial> d_lapse{};
  ::tenex::evaluate<ti::i>(make_not_null(&d_lapse),
                           interior_lapse() * interior_field_a(ti::i));

  tnsr::iJ<DataVector, Dim, Frame::Inertial> d_shift{};
  ::tenex::evaluate<ti::i, ti::J>(make_not_null(&d_shift),
                                  interior_field_b(ti::i, ti::J));

  const size_t num_pts = get(interior_conformal_factor).size();

  // Step 2: Evaluate analytic solution
  using all_tags = tmpl::list<
      Ccz4::Tags::ConformalMetric<DataVector, 3>,
      Ccz4::Tags::ConformalFactor<DataVector>,
      Ccz4::Tags::ATilde<DataVector, 3>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Ccz4::Tags::Theta<DataVector>, Ccz4::Tags::GammaHat<DataVector, 3>,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>,
      Ccz4::Tags::FieldA<DataVector, 3>, Ccz4::Tags::FieldB<DataVector, 3>,
      Ccz4::Tags::FieldD<DataVector, 3>, Ccz4::Tags::FieldP<DataVector, 3>>;
  auto analytic_values = call_with_dynamic_type<
      tuples::TaggedTuple<
          Ccz4::Tags::ConformalMetric<DataVector, 3>,
          Ccz4::Tags::ConformalFactor<DataVector>,
          Ccz4::Tags::ATilde<DataVector, 3>,
          gr::Tags::TraceExtrinsicCurvature<DataVector>,
          Ccz4::Tags::Theta<DataVector>, Ccz4::Tags::GammaHat<DataVector, 3>,
          gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
          Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>,
          Ccz4::Tags::FieldA<DataVector, 3>, Ccz4::Tags::FieldB<DataVector, 3>,
          Ccz4::Tags::FieldD<DataVector, 3>, Ccz4::Tags::FieldP<DataVector, 3>>,
      Ccz4::Solutions::all_solutions>(
      &analytic_prescription, [&coords, &time](const auto* const initial_data) {
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*initial_data)>>) {
          return initial_data->variables(coords, time, all_tags{});
        } else if constexpr (evolution::is_numeric_initial_data_v<
                                 std::decay_t<decltype(*initial_data)>>) {
          ERROR(
              "Cannot currently use numeric initial data as an analytic "
              "prescription for boundary conditions.");
        } else {
          (void)time;
          return initial_data->variables(coords, all_tags{});
        }
      });

  const auto& analytic_conformal_metric =
      get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(analytic_values);
  const auto& analytic_conformal_factor =
      get<Ccz4::Tags::ConformalFactor<DataVector>>(analytic_values);
  const auto& analytic_lapse =
      get<gr::Tags::Lapse<DataVector>>(analytic_values);
  const auto& analytic_shift =
      get<gr::Tags::Shift<DataVector, 3>>(analytic_values);
  const auto& analytic_a_tilde =
      get<Ccz4::Tags::ATilde<DataVector, 3>>(analytic_values);
  const auto& analytic_trace_K =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(analytic_values);
  const auto& analytic_theta =
      get<Ccz4::Tags::Theta<DataVector>>(analytic_values);
  const auto& analytic_gamma_hat =
      get<Ccz4::Tags::GammaHat<DataVector, 3>>(analytic_values);
  const auto& analytic_b =
      get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(analytic_values);
  const auto& analytic_field_a =
      get<Ccz4::Tags::FieldA<DataVector, 3>>(analytic_values);
  const auto& analytic_field_b =
      get<Ccz4::Tags::FieldB<DataVector, 3>>(analytic_values);
  const auto& analytic_field_d =
      get<Ccz4::Tags::FieldD<DataVector, 3>>(analytic_values);
  const auto& analytic_field_p =
      get<Ccz4::Tags::FieldP<DataVector, 3>>(analytic_values);

  // Step 3: Compute analytic unit normal (one form + vector)
  const auto [det_analytic_cm, inv_analytic_cm] =
      determinant_and_inverse(analytic_conformal_metric);

  tnsr::II<DataVector, Dim, Frame::Inertial> inv_analytic_spatial_metric{};
  ::tenex::evaluate<ti::I, ti::J>(make_not_null(&inv_analytic_spatial_metric),
                                  analytic_conformal_factor() *
                                      analytic_conformal_factor() *
                                      inv_analytic_cm(ti::I, ti::J));

  const Scalar<DataVector> analytic_mag_sq =
      dot_product(normal_covector, normal_covector,
                  inv_analytic_spatial_metric);
  const DataVector analytic_inv_mag = 1.0 / sqrt(get(analytic_mag_sq));

  tnsr::i<DataVector, Dim, Frame::Inertial> analytic_unit_normal_one_form(
      num_pts);
  for (size_t i = 0; i < Dim; ++i) {
    analytic_unit_normal_one_form.get(i) =
        normal_covector.get(i) * analytic_inv_mag;
  }

  tnsr::I<DataVector, Dim, Frame::Inertial> analytic_unit_normal_vector{};
  ::tenex::evaluate<ti::I>(make_not_null(&analytic_unit_normal_vector),
                           inv_analytic_spatial_metric(ti::I, ti::J) *
                               analytic_unit_normal_one_form(ti::j));

  // Step 4: Analytic spatial derivatives
  tnsr::ijj<DataVector, Dim, Frame::Inertial> analytic_d_cm{};
  ::tenex::evaluate<ti::k, ti::i, ti::j>(
      make_not_null(&analytic_d_cm),
      2.0 * analytic_field_d(ti::k, ti::i, ti::j));

  tnsr::i<DataVector, Dim, Frame::Inertial> analytic_d_cf{};
  ::tenex::evaluate<ti::i>(
      make_not_null(&analytic_d_cf),
      analytic_conformal_factor() * analytic_field_p(ti::i));

  tnsr::i<DataVector, Dim, Frame::Inertial> analytic_d_lapse{};
  ::tenex::evaluate<ti::i>(make_not_null(&analytic_d_lapse),
                           analytic_lapse() * analytic_field_a(ti::i));

  tnsr::iJ<DataVector, Dim, Frame::Inertial> analytic_d_shift{};
  ::tenex::evaluate<ti::i, ti::J>(make_not_null(&analytic_d_shift),
                                  analytic_field_b(ti::i, ti::J));

  // Step 5: Char speeds (interior lapse/shift/cf, analytic unit normal)
  auto char_speeds = ::Ccz4::fd::characteristic_speeds(
      interior_lapse, interior_shift, interior_conformal_factor, f,
      analytic_unit_normal_one_form);

  // Step 6: Interior forward char transform (analytic four fields for
  // metric/gauge coefficients, interior evolved vars and derivatives)
  auto char_fields = ::Ccz4::fd::characteristic_fields(
      analytic_unit_normal_one_form, analytic_conformal_metric,
      analytic_conformal_factor, analytic_lapse, analytic_shift,
      interior_trace_extrinsic_curvature, interior_a_tilde, interior_theta,
      interior_gamma_hat, interior_auxiliary_shift_b, d_conformal_metric,
      d_conformal_factor, d_lapse, d_shift, f);

  // Step 7: Ghost char fields + mode mixing
  const auto ghost_char_fields = ::Ccz4::fd::characteristic_fields(
      analytic_unit_normal_one_form, analytic_conformal_metric,
      analytic_conformal_factor, analytic_lapse, analytic_shift,
      analytic_trace_K, analytic_a_tilde, analytic_theta, analytic_gamma_hat,
      analytic_b, analytic_d_cm, analytic_d_cf, analytic_d_lapse, analytic_d_shift, f);

  mix_characteristic_modes(char_fields, ghost_char_fields, char_speeds,
                           prescribe_outgoing, num_pts);

  // Step 7: Inverse char transform
  const auto& u_tnsr_plus =
      get<::Ccz4::fd::Tags::UTensorPlus<DataVector, Dim, Frame::Inertial>>(
          char_fields);
  const auto& u_tnsr_minus =
      get<::Ccz4::fd::Tags::UTensorMinus<DataVector, Dim, Frame::Inertial>>(
          char_fields);
  const auto& u_vector1_zero =
      get<::Ccz4::fd::Tags::UVector1Zero<DataVector, Dim, Frame::Inertial>>(
          char_fields);
  const auto& u_vector2_plus =
      get<::Ccz4::fd::Tags::UVector2Plus<DataVector, Dim, Frame::Inertial>>(
          char_fields);
  const auto& u_vector2_minus =
      get<::Ccz4::fd::Tags::UVector2Minus<DataVector, Dim, Frame::Inertial>>(
          char_fields);
  const auto& u_vector3_plus =
      get<::Ccz4::fd::Tags::UVector3Plus<DataVector, Dim, Frame::Inertial>>(
          char_fields);
  const auto& u_vector3_minus =
      get<::Ccz4::fd::Tags::UVector3Minus<DataVector, Dim, Frame::Inertial>>(
          char_fields);
  const auto& u_scalar1_zero =
      get<::Ccz4::fd::Tags::UScalar1Zero<DataVector>>(char_fields);
  const auto& u_scalar2_plus =
      get<::Ccz4::fd::Tags::UScalar2Plus<DataVector>>(char_fields);
  const auto& u_scalar2_minus =
      get<::Ccz4::fd::Tags::UScalar2Minus<DataVector>>(char_fields);
  const auto& u_scalar3_plus =
      get<::Ccz4::fd::Tags::UScalar3Plus<DataVector>>(char_fields);
  const auto& u_scalar3_minus =
      get<::Ccz4::fd::Tags::UScalar3Minus<DataVector>>(char_fields);
  const auto& u_scalar4_plus =
      get<::Ccz4::fd::Tags::UScalar4Plus<DataVector>>(char_fields);
  const auto& u_scalar4_minus =
      get<::Ccz4::fd::Tags::UScalar4Minus<DataVector>>(char_fields);
  const auto& u_scalar5_plus =
      get<::Ccz4::fd::Tags::UScalar5Plus<DataVector>>(char_fields);
  const auto& u_scalar5_minus =
      get<::Ccz4::fd::Tags::UScalar5Minus<DataVector>>(char_fields);

  auto evolved_space = ::Ccz4::fd::evolved_space_from_characteristic_fields(
      u_tnsr_plus, u_tnsr_minus, u_vector1_zero, u_vector2_plus,
      u_vector2_minus, u_vector3_plus, u_vector3_minus, u_scalar1_zero,
      u_scalar2_plus, u_scalar2_minus, u_scalar3_plus, u_scalar3_minus,
      u_scalar4_plus, u_scalar4_minus, u_scalar5_plus, u_scalar5_minus,
      analytic_unit_normal_one_form, analytic_conformal_metric,
      analytic_conformal_factor, analytic_lapse, analytic_shift, f);

  // Step 8: Auxiliary field reconstruction from normal derivatives
  using DnCM =
      ::Ccz4::fd::Tags::DnConformalMetric<DataVector, Dim, Frame::Inertial>;
  using DnL = ::Ccz4::fd::Tags::DnLapse<DataVector>;
  using DnS = ::Ccz4::fd::Tags::DnShift<DataVector, Dim, Frame::Inertial>;
  using DnCF = ::Ccz4::fd::Tags::DnConformalFactor<DataVector>;

  const auto& dn_cm = get<DnCM>(evolved_space);
  const auto& dn_lapse = get<DnL>(evolved_space);
  const auto& dn_shift = get<DnS>(evolved_space);
  const auto& dn_cf = get<DnCF>(evolved_space);

  Scalar<DataVector> n_dot_d_lapse{};
  ::tenex::evaluate(make_not_null(&n_dot_d_lapse),
                    analytic_unit_normal_vector(ti::I) * d_lapse(ti::i));

  Scalar<DataVector> n_dot_d_cf{};
  ::tenex::evaluate(
      make_not_null(&n_dot_d_cf),
      analytic_unit_normal_vector(ti::I) * d_conformal_factor(ti::i));

  tnsr::i<DataVector, Dim, Frame::Inertial> result_field_a{};
  ::tenex::evaluate<ti::i>(
      make_not_null(&result_field_a),
      (d_lapse(ti::i) - analytic_unit_normal_one_form(ti::i) * n_dot_d_lapse() +
       analytic_unit_normal_one_form(ti::i) * dn_lapse()) /
          boundary_integrated_lapse());

  tnsr::i<DataVector, Dim, Frame::Inertial> result_field_p{};
  ::tenex::evaluate<ti::i>(make_not_null(&result_field_p),
                           (d_conformal_factor(ti::i) -
                            analytic_unit_normal_one_form(ti::i) * n_dot_d_cf() +
                            analytic_unit_normal_one_form(ti::i) * dn_cf()) /
                               boundary_integrated_conformal_factor());

  tnsr::ii<DataVector, Dim, Frame::Inertial> n_dot_d_cm{};
  ::tenex::evaluate<ti::j, ti::k>(make_not_null(&n_dot_d_cm),
                                  analytic_unit_normal_vector(ti::M) *
                                      d_conformal_metric(ti::m, ti::j, ti::k));

  tnsr::ijj<DataVector, Dim, Frame::Inertial> result_field_d{};
  ::tenex::evaluate<ti::i, ti::j, ti::k>(
      make_not_null(&result_field_d),
      0.5 * (d_conformal_metric(ti::i, ti::j, ti::k) -
             analytic_unit_normal_one_form(ti::i) * n_dot_d_cm(ti::j, ti::k) +
             analytic_unit_normal_one_form(ti::i) * dn_cm(ti::j, ti::k)));

  tnsr::I<DataVector, Dim, Frame::Inertial> n_dot_d_shift{};
  ::tenex::evaluate<ti::J>(
      make_not_null(&n_dot_d_shift),
      analytic_unit_normal_vector(ti::M) * d_shift(ti::m, ti::J));

  tnsr::iJ<DataVector, Dim, Frame::Inertial> result_field_b{};
  ::tenex::evaluate<ti::i, ti::J>(
      make_not_null(&result_field_b),
      d_shift(ti::i, ti::J) -
          analytic_unit_normal_one_form(ti::i) * n_dot_d_shift(ti::J) +
          analytic_unit_normal_one_form(ti::i) * dn_shift(ti::J));

  return {std::move(char_speeds),    std::move(evolved_space),
          std::move(result_field_a), std::move(result_field_b),
          std::move(result_field_d), std::move(result_field_p)};
}
}  // namespace

namespace Ccz4::BoundaryConditions {

// LCOV_EXCL_START
DirichletCharacteristics::DirichletCharacteristics(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}
// LCOV_EXCL_STOP
DirichletCharacteristics::DirichletCharacteristics(
    const DirichletCharacteristics& rhs)
    : BoundaryCondition{dynamic_cast<const BoundaryCondition&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()),
      prescribe_outgoing_(rhs.prescribe_outgoing_) {}

DirichletCharacteristics& DirichletCharacteristics::operator=(
    const DirichletCharacteristics& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  prescribe_outgoing_ = rhs.prescribe_outgoing_;
  return *this;
}

DirichletCharacteristics::DirichletCharacteristics(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription,
    bool prescribe_outgoing)
    : analytic_prescription_(std::move(analytic_prescription)),
      prescribe_outgoing_(prescribe_outgoing) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletCharacteristics::get_clone() const {
  return std::make_unique<DirichletCharacteristics>(*this);
}

void DirichletCharacteristics::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | analytic_prescription_;
  p | prescribe_outgoing_;
}
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletCharacteristics::my_PUP_ID = 0;

std::optional<std::string> DirichletCharacteristics::dg_ghost(
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        conformal_metric,
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> a_tilde,
    const gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
    const gsl::not_null<Scalar<DataVector>*> theta,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> gamma_hat,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        auxiliary_shift_b,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> field_a,
    const gsl::not_null<tnsr::iJ<DataVector, 3, Frame::Inertial>*> field_b,
    const gsl::not_null<tnsr::ijj<DataVector, 3, Frame::Inertial>*> field_d,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> field_p,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        bm_u_tensor_minus,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        boundary_conformal_metric,
    const gsl::not_null<Scalar<DataVector>*> boundary_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*> boundary_lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_shift,
    const gsl::not_null<Scalar<DataVector>*> boundary_theta,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> boundary_z,
    const std::optional<
        tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& /*interior_conformal_metric*/,
    const Scalar<DataVector>& interior_conformal_factor,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_a_tilde,
    const Scalar<DataVector>& interior_trace_extrinsic_curvature,
    const Scalar<DataVector>& interior_theta,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_gamma_hat,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_auxiliary_shift_b,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interior_field_a,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& interior_field_b,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& interior_field_d,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interior_field_p,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& /*interior_boundary_u_tensor_minus*/,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        interior_boundary_conformal_metric,
    const Scalar<DataVector>& interior_boundary_conformal_factor,
    const Scalar<DataVector>& interior_boundary_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_boundary_shift,
    const Scalar<DataVector>& interior_boundary_theta,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interior_boundary_z,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
    const bool evolve_lapse_and_shift) const {
  static constexpr size_t Dim = 3;
  static constexpr double f = ::Ccz4::fd::System::f;

  ASSERT(evolve_lapse_and_shift,
         "DirichletCharacteristics BC is not implemented"
         " when lapse and shift are not evolved, as the SoCcz4 system"
         " is only weakly hyperbolic in that case");
  const size_t num_pts = get(interior_conformal_factor).size();

  // Run the full characteristic decomposition pipeline
  auto result = characteristic_decomposition_pipeline(
      *analytic_prescription_, normal_covector,
      interior_conformal_factor, interior_a_tilde,
      interior_trace_extrinsic_curvature, interior_theta, interior_gamma_hat,
      interior_lapse, interior_shift, interior_auxiliary_shift_b,
      interior_field_a, interior_field_b, interior_field_d, interior_field_p,
      interior_boundary_lapse, interior_boundary_conformal_factor, coords, time,
      f, prescribe_outgoing_);

  // Validate characteristic speed signs
  for (size_t s = 0; s < num_pts; ++s) {
    if (result.char_speeds[0][s] < 0.0 || result.char_speeds[1][s] >= 0.0 ||
        result.char_speeds[5][s] < 0.0 || result.char_speeds[6][s] >= 0.0 ||
        result.char_speeds[12][s] < 0.0 || result.char_speeds[13][s] >= 0.0 ||
        result.char_speeds[14][s] < 0.0 || result.char_speeds[15][s] >= 0.0) {
      ERROR(
          "DirichletCharacteristics dg_ghost: characteristic speed sign "
          "violation at face point "
          << s << ".\n"
          << "  char_speeds[0]  (UTensorPlus,  expect>=0): "
          << result.char_speeds[0][s] << "\n"
          << "  char_speeds[1]  (UTensorMinus,  expect<0): "
          << result.char_speeds[1][s] << "\n"
          << "  char_speeds[5]  (UVector3Plus,  expect>=0): "
          << result.char_speeds[5][s] << "\n"
          << "  char_speeds[6]  (UVector3Minus, expect<0): "
          << result.char_speeds[6][s] << "\n"
          << "  char_speeds[12] (UScalar4Plus,  expect>=0): "
          << result.char_speeds[12][s] << "\n"
          << "  char_speeds[13] (UScalar4Minus, expect<0): "
          << result.char_speeds[13][s] << "\n"
          << "  char_speeds[14] (UScalar5Plus,  expect>=0): "
          << result.char_speeds[14][s] << "\n"
          << "  char_speeds[15] (UScalar5Minus, expect<0): "
          << result.char_speeds[15][s]);
    }
  }

  // Write exterior ghost state
  // Ghost four fields = boundary-integrated
  *conformal_metric = interior_boundary_conformal_metric;
  *conformal_factor = interior_boundary_conformal_factor;
  *lapse = interior_boundary_lapse;
  *shift = interior_boundary_shift;

  // Evolved variables from inverse transform
  *a_tilde = get<::Ccz4::Tags::ATilde<DataVector, Dim, Frame::Inertial>>(
      result.evolved_space);
  *trace_extrinsic_curvature =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(result.evolved_space);
  *theta = get<::Ccz4::Tags::Theta<DataVector>>(result.evolved_space);
  *gamma_hat = get<::Ccz4::Tags::GammaHat<DataVector, Dim, Frame::Inertial>>(
      result.evolved_space);
  *auxiliary_shift_b =
      get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim, Frame::Inertial>>(
          result.evolved_space);

  // Auxiliary fields from reconstruction
  *field_a = std::move(result.field_a);
  *field_b = std::move(result.field_b);
  *field_d = std::move(result.field_d);
  *field_p = std::move(result.field_p);

  // Enforce g_tilde^{jk} D_{ijk} = 0 w.r.t. boundary-integrated conformal metric
  // (Jacobi formula for det(g_tilde) = 1)
  {
    const auto [det_bnd_cm, inv_bnd_cm] =
        determinant_and_inverse(interior_boundary_conformal_metric);
    tnsr::i<DataVector, Dim, Frame::Inertial> residual{};
    ::tenex::evaluate<ti::k>(
        make_not_null(&residual),
        inv_bnd_cm(ti::I, ti::J) * (*field_d)(ti::k, ti::i, ti::j));
    ::tenex::update<ti::k, ti::i, ti::j>(
        field_d,
        (*field_d)(ti::k, ti::i, ti::j) -
            residual(ti::k) *
                interior_boundary_conformal_metric(ti::i, ti::j) / 3.0);
  }

  // Boundary mode exterior values: zero
  for (auto& component : *bm_u_tensor_minus) {
    component = 0.0;
  }

  // Boundary second-order field exterior values: pass through interior
  *boundary_conformal_metric = interior_boundary_conformal_metric;
  *boundary_conformal_factor = interior_boundary_conformal_factor;
  *boundary_lapse = interior_boundary_lapse;
  *boundary_shift = interior_boundary_shift;
  *boundary_theta = interior_boundary_theta;
  *boundary_z = interior_boundary_z;

  return {};
}

std::optional<std::string> DirichletCharacteristics::dg_time_derivative(
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        dt_conformal_metric_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_conformal_factor_correction,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        dt_a_tilde_correction,
    const gsl::not_null<Scalar<DataVector>*>
        dt_trace_extrinsic_curvature_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_theta_correction,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        dt_gamma_hat_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_lapse_correction,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        dt_shift_correction,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        dt_auxiliary_shift_b_correction,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        dt_field_a_correction,
    const gsl::not_null<tnsr::iJ<DataVector, 3, Frame::Inertial>*>
        dt_field_b_correction,
    const gsl::not_null<tnsr::ijj<DataVector, 3, Frame::Inertial>*>
        dt_field_d_correction,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        dt_field_p_correction,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        dt_u_tensor_minus_correction,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        dt_boundary_conformal_metric_correction,
    const gsl::not_null<Scalar<DataVector>*>
        dt_boundary_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_boundary_lapse_correction,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        dt_boundary_shift_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_boundary_theta_correction,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        dt_boundary_z_correction,
    const std::optional<
        tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& /*interior_conformal_metric*/,
    const Scalar<DataVector>& interior_conformal_factor,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_a_tilde,
    const Scalar<DataVector>& interior_trace_extrinsic_curvature,
    const Scalar<DataVector>& interior_theta,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_gamma_hat,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_auxiliary_shift_b,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interior_field_a,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& interior_field_b,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& interior_field_d,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interior_field_p,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& /*interior_boundary_u_tensor_minus*/,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        interior_boundary_conformal_metric,
    const Scalar<DataVector>& interior_boundary_conformal_factor,
    const Scalar<DataVector>& interior_boundary_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_boundary_shift,
    const Scalar<DataVector>& /*interior_boundary_theta*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*interior_boundary_z*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
    const bool evolve_lapse_and_shift) const {
  static constexpr size_t Dim = 3;
  static constexpr double f_val = ::Ccz4::fd::System::f;
  static constexpr bool shifting_shift = ::Ccz4::fd::System::shifting_shift;

  ASSERT(evolve_lapse_and_shift,
         "DirichletCharacteristics BC is not implemented"
         " when lapse and shift are not evolved");

  // Zero all 17 original evolved + auxiliary dt corrections
  for (auto& component : *dt_conformal_metric_correction) {
    component = 0.0;
  }
  get(*dt_conformal_factor_correction) = 0.0;
  for (auto& component : *dt_a_tilde_correction) {
    component = 0.0;
  }
  get(*dt_trace_extrinsic_curvature_correction) = 0.0;
  get(*dt_theta_correction) = 0.0;
  for (auto& component : *dt_gamma_hat_correction) {
    component = 0.0;
  }
  get(*dt_lapse_correction) = 0.0;
  for (auto& component : *dt_shift_correction) {
    component = 0.0;
  }
  for (auto& component : *dt_auxiliary_shift_b_correction) {
    component = 0.0;
  }
  for (auto& component : *dt_field_a_correction) {
    component = 0.0;
  }
  for (auto& component : *dt_field_b_correction) {
    component = 0.0;
  }
  for (auto& component : *dt_field_d_correction) {
    component = 0.0;
  }
  for (auto& component : *dt_field_p_correction) {
    component = 0.0;
  }
  // Zero boundary mode dt corrections
  for (auto& component : *dt_u_tensor_minus_correction) {
    component = 0.0;
  }

  // Run the full characteristic decomposition pipeline
  auto result = characteristic_decomposition_pipeline(
      *analytic_prescription_, normal_covector,
      interior_conformal_factor, interior_a_tilde,
      interior_trace_extrinsic_curvature, interior_theta, interior_gamma_hat,
      interior_lapse, interior_shift, interior_auxiliary_shift_b,
      interior_field_a, interior_field_b, interior_field_d, interior_field_p,
      interior_boundary_lapse, interior_boundary_conformal_factor, coords, time,
      f_val, prescribe_outgoing_);

  // Extract char-mixed evolved variables
  const auto& mixed_a_tilde =
      get<::Ccz4::Tags::ATilde<DataVector, Dim, Frame::Inertial>>(
          result.evolved_space);
  const auto& mixed_K =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(result.evolved_space);
  const auto& mixed_theta =
      get<::Ccz4::Tags::Theta<DataVector>>(result.evolved_space);
  const auto& mixed_b =
      get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim, Frame::Inertial>>(
          result.evolved_space);

  // Compute dt of second-order fields using char-mixed state.
  // K0 = 0 hardcoded (see SetK0.hpp: always set to zero in SO-CCZ4).
  const auto k_0 = make_with_value<Scalar<DataVector>>(
      get(interior_boundary_conformal_factor), 0.0);

  // Enforce g_tilde^{jk} D_{ijk} = 0 w.r.t. boundary conformal metric
  // (Jacobi formula for det(g_tilde) = 1)
  const auto [det_bnd_cm, inv_bnd_cm] =
      determinant_and_inverse(interior_boundary_conformal_metric);
  {
    tnsr::i<DataVector, Dim, Frame::Inertial> residual{};
    ::tenex::evaluate<ti::k>(
        make_not_null(&residual),
        inv_bnd_cm(ti::I, ti::J) * result.field_d(ti::k, ti::i, ti::j));
    ::tenex::update<ti::k, ti::i, ti::j>(
        make_not_null(&result.field_d),
        result.field_d(ti::k, ti::i, ti::j) -
            residual(ti::k) *
                interior_boundary_conformal_metric(ti::i, ti::j) / 3.0);
  }

  compute_dt_second_order_fields(
      dt_boundary_conformal_metric_correction,
      dt_boundary_conformal_factor_correction, dt_boundary_lapse_correction,
      dt_boundary_shift_correction, interior_boundary_conformal_metric,
      interior_boundary_conformal_factor, interior_boundary_lapse,
      interior_boundary_shift, mixed_a_tilde, mixed_K, mixed_theta, mixed_b,
      result.field_a, result.field_b, result.field_d, result.field_p, k_0,
      f_val, shifting_shift);

  // Zero boundary theta/z dt corrections (not evolved by this BC)
  get(*dt_boundary_theta_correction) = 0.0;
  for (auto& component : *dt_boundary_z_correction) {
    component = 0.0;
  }

  return {};
}

void DirichletCharacteristics::fd_ghost(
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>,
    const gsl::not_null<Scalar<DataVector>*>,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>,
    const gsl::not_null<Scalar<DataVector>*>,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>,
    const gsl::not_null<Scalar<DataVector>*>,
    const gsl::not_null<Scalar<DataVector>*>,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>,
    const Direction<3>& /*direction*/) const {
  ERROR(
      "DirichletCharacteristics fd_ghost is not implemented. "
      "This BC is only available for the DG (LDG) path.");
}
}  // namespace Ccz4::BoundaryConditions
