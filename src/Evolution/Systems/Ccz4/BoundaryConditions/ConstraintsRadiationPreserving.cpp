// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/BoundaryConditions/ConstraintsRadiationPreserving.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Characteristics.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
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
  static constexpr double c = 1.0;
  static constexpr double lapse_times_slicing_cond = 2.0;

  Scalar<DataVector> contracted_field_b{};
  ::tenex::evaluate(make_not_null(&contracted_field_b), field_b(ti::k, ti::K));

  tnsr::ij<DataVector, 3, Frame::Inertial> conformal_metric_times_field_b{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&conformal_metric_times_field_b),
      conformal_metric(ti::k, ti::i) * field_b(ti::j, ti::K));

  Scalar<DataVector> k_minus_k0_minus_2_theta_c{};
  ::tenex::evaluate(make_not_null(&k_minus_k0_minus_2_theta_c),
                    trace_extrinsic_curvature() - k_0() - 2.0 * c * theta());

  // eq 12a: dt conformal metric
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

// Result of the CRPBC characteristic decomposition + mode mixing + inverse
// transform + auxiliary reconstruction pipeline.
struct CrpbcMixedState {
  std::array<DataVector, 16> char_speeds;
  ::Ccz4::fd::Tags::EvolvedSpaceFromCharacteristicFields<DataVector, 3,
                                                         Frame::Inertial>::type
      evolved_space;
  tnsr::i<DataVector, 3, Frame::Inertial> field_a;
  tnsr::iJ<DataVector, 3, Frame::Inertial> field_b;
  tnsr::ijj<DataVector, 3, Frame::Inertial> field_d;
  tnsr::i<DataVector, 3, Frame::Inertial> field_p;
};

// Performs the full CRPBC characteristic mode-mixing pipeline shared by
// dg_ghost and dg_time_derivative:
// 1. Reconstructs interior spatial derivatives from interior auxiliary fields
// 2. Computes interior unit normal from normal_covector + interior metric
// 3. Computes interior char speeds and char fields
// 4. Evaluates analytic solution
// 5. Reconstructs ghost spatial derivatives (coeff four fields * analytic aux)
// 6. Computes ghost char fields (ghost unit normal + coeff four fields)
// 7. CRPBC mode mixing (4 boundary modes from time-integrated values,
//    gauge modes from ghost, zero-speed modes conditionally from ghost)
// 8. Inverse char transform (ghost unit normal + coeff four fields)
// 9. Reconstructs auxiliary fields from normal derivatives
CrpbcMixedState crpbc_characteristic_pipeline(
    const evolution::initial_data::InitialData& analytic_prescription,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_conformal_metric,
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
    const tnsr::ii<DataVector, 3, Frame::Inertial>& coeff_conformal_metric,
    const Scalar<DataVector>& coeff_conformal_factor,
    const Scalar<DataVector>& coeff_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coeff_shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& ghost_unit_normal_one_form,
    const tnsr::I<DataVector, 3, Frame::Inertial>& ghost_unit_normal_vector,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        interior_boundary_u_tensor_minus,
    const Scalar<DataVector>& interior_boundary_theta,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interior_boundary_z,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double time,
    const double f, const bool use_analytic_for_all) {
  static constexpr size_t Dim = 3;
  const size_t num_pts = get(interior_conformal_factor).size();

  // Step 1: Interior spatial derivatives from auxiliary fields
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

  // Step 2: Interior unit normal
  const auto [det_cm, inv_cm] =
      determinant_and_inverse(interior_conformal_metric);

  tnsr::II<DataVector, Dim, Frame::Inertial> inv_spatial_metric{};
  ::tenex::evaluate<ti::I, ti::J>(make_not_null(&inv_spatial_metric),
                                  interior_conformal_factor() *
                                      interior_conformal_factor() *
                                      inv_cm(ti::I, ti::J));

  const Scalar<DataVector> mag_sq =
      dot_product(normal_covector, normal_covector, inv_spatial_metric);
  const DataVector inv_mag = 1.0 / sqrt(get(mag_sq));

  tnsr::i<DataVector, Dim, Frame::Inertial> interior_unit_normal_one_form(
      num_pts);
  for (size_t i = 0; i < Dim; ++i) {
    interior_unit_normal_one_form.get(i) = normal_covector.get(i) * inv_mag;
  }

  // Step 3: Interior char speeds and char fields
  auto char_speeds = ::Ccz4::fd::characteristic_speeds(
      interior_lapse, interior_shift, interior_conformal_factor, f,
      interior_unit_normal_one_form);

  const auto interior_char_fields = ::Ccz4::fd::characteristic_fields(
      interior_unit_normal_one_form, interior_conformal_metric,
      interior_conformal_factor, interior_lapse, interior_shift,
      interior_trace_extrinsic_curvature, interior_a_tilde, interior_theta,
      interior_gamma_hat, interior_auxiliary_shift_b, d_conformal_metric,
      d_conformal_factor, d_lapse, d_shift, f);

  // Step 4: Evaluate analytic solution
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
          Ccz4::Tags::FieldD<DataVector, 3>,
          Ccz4::Tags::FieldP<DataVector, 3>>,
      Ccz4::Solutions::all_solutions>(
      &analytic_prescription,
      [&coords, &time](const auto* const initial_data) {
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

  // Step 5: Ghost spatial derivatives (coeff four fields * analytic aux)
  tnsr::ijj<DataVector, Dim, Frame::Inertial> ghost_d_cm{};
  ::tenex::evaluate<ti::k, ti::i, ti::j>(
      make_not_null(&ghost_d_cm), 2.0 * analytic_field_d(ti::k, ti::i, ti::j));

  tnsr::i<DataVector, Dim, Frame::Inertial> ghost_d_cf{};
  ::tenex::evaluate<ti::i>(make_not_null(&ghost_d_cf),
                           coeff_conformal_factor() * analytic_field_p(ti::i));

  tnsr::i<DataVector, Dim, Frame::Inertial> ghost_d_lapse{};
  ::tenex::evaluate<ti::i>(make_not_null(&ghost_d_lapse),
                           coeff_lapse() * analytic_field_a(ti::i));

  tnsr::iJ<DataVector, Dim, Frame::Inertial> ghost_d_shift{};
  ::tenex::evaluate<ti::i, ti::J>(make_not_null(&ghost_d_shift),
                                  analytic_field_b(ti::i, ti::J));

  // Step 6: Ghost char fields (ghost unit normal + coeff four fields +
  // analytic first-order vars)
  const auto ghost_char_fields = ::Ccz4::fd::characteristic_fields(
      ghost_unit_normal_one_form, coeff_conformal_metric,
      coeff_conformal_factor, coeff_lapse, coeff_shift, analytic_trace_K,
      analytic_a_tilde, analytic_theta, analytic_gamma_hat, analytic_b,
      ghost_d_cm, ghost_d_cf, ghost_d_lapse, ghost_d_shift, f);

  // Step 7: CRPBC mode mixing. The incoming (-) modes and zero-speed modes
  // start as copies of the interior values and are selectively overwritten
  // below; outgoing (+) modes remain at their interior values.
  using namespace ::Ccz4::fd::Tags;

  auto u_tnsr_minus =
      get<UTensorMinus<DataVector, Dim, Frame::Inertial>>(interior_char_fields);
  auto u_vector1_zero =
      get<UVector1Zero<DataVector, Dim, Frame::Inertial>>(interior_char_fields);
  auto u_vector2_minus_field =
      get<UVector2Minus<DataVector, Dim, Frame::Inertial>>(interior_char_fields);
  auto u_vector3_minus_field =
      get<UVector3Minus<DataVector, Dim, Frame::Inertial>>(interior_char_fields);
  auto u_scalar1_zero = get<UScalar1Zero<DataVector>>(interior_char_fields);
  auto u_scalar2_minus_field =
      get<UScalar2Minus<DataVector>>(interior_char_fields);
  auto u_scalar3_minus_field =
      get<UScalar3Minus<DataVector>>(interior_char_fields);
  auto u_scalar4_minus_field =
      get<UScalar4Minus<DataVector>>(interior_char_fields);
  auto u_scalar5_minus_field =
      get<UScalar5Minus<DataVector>>(interior_char_fields);

  const auto& ghost_u_vector1_zero =
      get<UVector1Zero<DataVector, Dim, Frame::Inertial>>(ghost_char_fields);
  const auto& ghost_u_scalar1_zero =
      get<UScalar1Zero<DataVector>>(ghost_char_fields);

  if (use_analytic_for_all) {
    // Debug mode: ALL incoming modes from ghost char fields
    // (same behavior as DirichletCharacteristics)
    u_tnsr_minus = get<UTensorMinus<DataVector, Dim, Frame::Inertial>>(
        ghost_char_fields);
    u_vector2_minus_field =
        get<UVector2Minus<DataVector, Dim, Frame::Inertial>>(
            ghost_char_fields);
    u_scalar2_minus_field =
        get<UScalar2Minus<DataVector>>(ghost_char_fields);
    u_scalar3_minus_field =
        get<UScalar3Minus<DataVector>>(ghost_char_fields);
    u_vector3_minus_field =
        get<UVector3Minus<DataVector, Dim, Frame::Inertial>>(
            ghost_char_fields);
    u_scalar4_minus_field =
        get<UScalar4Minus<DataVector>>(ghost_char_fields);
    u_scalar5_minus_field =
        get<UScalar5Minus<DataVector>>(ghost_char_fields);
  } else {
    // Normal CRPBC mode:
    //   - UTensorMinus is evolved as a boundary mode.
    //   - UScalar3Minus / UVector2Minus / UScalar2Minus are reconstructed
    //     algebraically from BoundaryTheta and BoundaryZ (evolved as
    //     advection+damping ODEs on the CRPBC face) combined with the
    //     interior plus modes, per the inverse relations in `crpbc.tex`.
    //   - The three gauge (-) modes come from the analytic (ghost) char
    //     fields as before.
    u_tnsr_minus = interior_boundary_u_tensor_minus;

    // Inverse of the coefficient (boundary-integrated) conformal metric.
    const auto [det_coeff_cm_local, inv_coeff_cm] =
        determinant_and_inverse(coeff_conformal_metric);
    (void)det_coeff_cm_local;

    // Coefficient conformal factor squared and its square.
    Scalar<DataVector> coeff_conformal_factor_squared{};
    ::tenex::evaluate(make_not_null(&coeff_conformal_factor_squared),
                      coeff_conformal_factor() * coeff_conformal_factor());
    Scalar<DataVector> coeff_phi4{};
    ::tenex::evaluate(
        make_not_null(&coeff_phi4),
        coeff_conformal_factor_squared() * coeff_conformal_factor_squared());

    // Outgoing (+) modes from the interior char fields.
    const auto& u_scalar3_plus_in =
        get<UScalar3Plus<DataVector>>(interior_char_fields);
    const auto& u_vector2_plus_in =
        get<UVector2Plus<DataVector, Dim, Frame::Inertial>>(
            interior_char_fields);
    const auto& u_scalar2_plus_in =
        get<UScalar2Plus<DataVector>>(interior_char_fields);

    // UScalar3Minus_rec = UScalar3Plus + 4·Θ_bdry / φ²
    ::tenex::evaluate(make_not_null(&u_scalar3_minus_field),
                      u_scalar3_plus_in() + 4.0 * interior_boundary_theta() /
                                                coeff_conformal_factor_squared());

    // Transverse projector q^I_j from ghost unit normal (used for T^i,
    // which is built from ghost-side spatial derivatives of gamma-tilde).
    const auto q_mixed = gr::transverse_projection_operator(
        ghost_unit_normal_vector, ghost_unit_normal_one_form);

    // Interior-side unit normal vector and projector: used to decompose
    // interior_boundary_z into its normal + transverse parts. Z_i is an
    // interior-evolved quantity, so its split must match the interior
    // normal; otherwise interior and ghost Z_i components get mixed.
    tnsr::I<DataVector, Dim, Frame::Inertial> interior_unit_normal_vector{};
    ::tenex::evaluate<ti::I>(make_not_null(&interior_unit_normal_vector),
                             inv_spatial_metric(ti::I, ti::J) *
                                 interior_unit_normal_one_form(ti::j));
    const auto q_mixed_interior = gr::transverse_projection_operator(
        interior_unit_normal_vector, interior_unit_normal_one_form);

    // T^i = γ̃^{ij} γ̃^{kl} q^m_l (2·analytic_field_d)_{m,j,k}
    //     = γ̃^{ij} γ̃^{kl} q^m_l · ghost_d_cm(m,j,k)
    tnsr::I<DataVector, Dim, Frame::Inertial> T_up{};
    ::tenex::evaluate<ti::I>(
        make_not_null(&T_up),
        inv_coeff_cm(ti::I, ti::J) * inv_coeff_cm(ti::K, ti::L) *
            q_mixed(ti::M, ti::l) * ghost_d_cm(ti::m, ti::j, ti::k));

    // T^⊥_i = q_{ij} T^j = γ_{ij} q^j_k T^k   (physical metric lowering).
    // γ_{ij} = γ̃_{ij} / φ², so divide by φ² after lowering with γ̃.
    tnsr::i<DataVector, Dim, Frame::Inertial> T_perp_lo{};
    ::tenex::evaluate<ti::i>(make_not_null(&T_perp_lo),
                             coeff_conformal_metric(ti::i, ti::j) *
                                 q_mixed(ti::J, ti::k) * T_up(ti::K) /
                                 coeff_conformal_factor_squared());

    // Z^⊥_i = q^j_i Z_j_bdry   (lower, transverse-projected BoundaryZ).
    // Use the interior projector because interior_boundary_z is an
    // interior-side quantity.
    tnsr::i<DataVector, Dim, Frame::Inertial> Z_perp_lo{};
    ::tenex::evaluate<ti::i>(
        make_not_null(&Z_perp_lo),
        q_mixed_interior(ti::J, ti::i) * interior_boundary_z(ti::j));

    // UVector2Minus_rec_i = -UVector2Plus_i + 4·Z^⊥_i / φ² + 2·T^⊥_i
    ::tenex::evaluate<ti::i>(
        make_not_null(&u_vector2_minus_field),
        -u_vector2_plus_in(ti::i) +
            4.0 * Z_perp_lo(ti::i) / coeff_conformal_factor_squared() +
            2.0 * T_perp_lo(ti::i));

    // T^n = n_i T^i
    Scalar<DataVector> T_n{};
    ::tenex::evaluate(
        make_not_null(&T_n),
        ghost_unit_normal_one_form(ti::i) * T_up(ti::I));

    // Z^n = n^i Z_i_bdry   (interior normal, for the same reason as Z^⊥_i).
    Scalar<DataVector> Z_n{};
    ::tenex::evaluate(
        make_not_null(&Z_n),
        interior_unit_normal_vector(ti::I) * interior_boundary_z(ti::i));

    // UScalar2Minus_rec = UScalar2Plus
    //   - (φ⁴/2)(UScalar3Plus + UScalar3Minus_rec)
    //   + φ⁴·T^n + 2·φ²·Z^n
    ::tenex::evaluate(
        make_not_null(&u_scalar2_minus_field),
        u_scalar2_plus_in() -
            0.5 * coeff_phi4() *
                (u_scalar3_plus_in() + u_scalar3_minus_field()) +
            coeff_phi4() * T_n() +
            2.0 * coeff_conformal_factor_squared() * Z_n());

    // 3 gauge modes from ghost char fields
    u_vector3_minus_field =
        get<UVector3Minus<DataVector, Dim, Frame::Inertial>>(
            ghost_char_fields);
    u_scalar4_minus_field =
        get<UScalar4Minus<DataVector>>(ghost_char_fields);
    u_scalar5_minus_field =
        get<UScalar5Minus<DataVector>>(ghost_char_fields);
  }

  // Zero-speed modes: conditionally from ghost
  for (size_t s = 0; s < num_pts; ++s) {
    if (char_speeds[2][s] < 0.0) {
      for (size_t i = 0; i < Dim; ++i) {
        u_vector1_zero.get(i)[s] = ghost_u_vector1_zero.get(i)[s];
      }
    }
    if (char_speeds[7][s] < 0.0) {
      get(u_scalar1_zero)[s] = get(ghost_u_scalar1_zero)[s];
    }
  }

  // Step 8: Inverse char transform (ghost unit normal + coeff four fields).
  // Outgoing (+) modes come from the interior char fields unchanged.
  const auto& u_tnsr_plus =
      get<UTensorPlus<DataVector, Dim, Frame::Inertial>>(interior_char_fields);
  const auto& u_vector2_plus =
      get<UVector2Plus<DataVector, Dim, Frame::Inertial>>(interior_char_fields);
  const auto& u_vector3_plus =
      get<UVector3Plus<DataVector, Dim, Frame::Inertial>>(interior_char_fields);
  const auto& u_scalar2_plus =
      get<UScalar2Plus<DataVector>>(interior_char_fields);
  const auto& u_scalar3_plus =
      get<UScalar3Plus<DataVector>>(interior_char_fields);
  const auto& u_scalar4_plus =
      get<UScalar4Plus<DataVector>>(interior_char_fields);
  const auto& u_scalar5_plus =
      get<UScalar5Plus<DataVector>>(interior_char_fields);

  auto evolved_space = ::Ccz4::fd::evolved_space_from_characteristic_fields(
      u_tnsr_plus, u_tnsr_minus, u_vector1_zero, u_vector2_plus,
      u_vector2_minus_field, u_vector3_plus, u_vector3_minus_field,
      u_scalar1_zero, u_scalar2_plus, u_scalar2_minus_field, u_scalar3_plus,
      u_scalar3_minus_field, u_scalar4_plus, u_scalar4_minus_field,
      u_scalar5_plus, u_scalar5_minus_field, ghost_unit_normal_one_form,
      coeff_conformal_metric, coeff_conformal_factor, coeff_lapse, coeff_shift,
      f);

  // Step 9: Auxiliary field reconstruction from normal derivatives
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
                    ghost_unit_normal_vector(ti::I) * d_lapse(ti::i));

  Scalar<DataVector> n_dot_d_cf{};
  ::tenex::evaluate(
      make_not_null(&n_dot_d_cf),
      ghost_unit_normal_vector(ti::I) * d_conformal_factor(ti::i));

  tnsr::i<DataVector, Dim, Frame::Inertial> result_field_a{};
  ::tenex::evaluate<ti::i>(
      make_not_null(&result_field_a),
      (d_lapse(ti::i) - ghost_unit_normal_one_form(ti::i) * n_dot_d_lapse() +
       ghost_unit_normal_one_form(ti::i) * dn_lapse()) /
          coeff_lapse());

  tnsr::i<DataVector, Dim, Frame::Inertial> result_field_p{};
  ::tenex::evaluate<ti::i>(make_not_null(&result_field_p),
                           (d_conformal_factor(ti::i) -
                            ghost_unit_normal_one_form(ti::i) * n_dot_d_cf() +
                            ghost_unit_normal_one_form(ti::i) * dn_cf()) /
                               coeff_conformal_factor());

  tnsr::ii<DataVector, Dim, Frame::Inertial> n_dot_d_cm{};
  ::tenex::evaluate<ti::j, ti::k>(make_not_null(&n_dot_d_cm),
                                  ghost_unit_normal_vector(ti::M) *
                                      d_conformal_metric(ti::m, ti::j, ti::k));

  tnsr::ijj<DataVector, Dim, Frame::Inertial> result_field_d{};
  ::tenex::evaluate<ti::i, ti::j, ti::k>(
      make_not_null(&result_field_d),
      0.5 * (d_conformal_metric(ti::i, ti::j, ti::k) -
             ghost_unit_normal_one_form(ti::i) * n_dot_d_cm(ti::j, ti::k) +
             ghost_unit_normal_one_form(ti::i) * dn_cm(ti::j, ti::k)));

  tnsr::I<DataVector, Dim, Frame::Inertial> n_dot_d_shift{};
  ::tenex::evaluate<ti::J>(
      make_not_null(&n_dot_d_shift),
      ghost_unit_normal_vector(ti::M) * d_shift(ti::m, ti::J));

  tnsr::iJ<DataVector, Dim, Frame::Inertial> result_field_b{};
  ::tenex::evaluate<ti::i, ti::J>(
      make_not_null(&result_field_b),
      d_shift(ti::i, ti::J) -
          ghost_unit_normal_one_form(ti::i) * n_dot_d_shift(ti::J) +
          ghost_unit_normal_one_form(ti::i) * dn_shift(ti::J));

  return {std::move(char_speeds),    std::move(evolved_space),
          std::move(result_field_a), std::move(result_field_b),
          std::move(result_field_d), std::move(result_field_p)};
}
}  // namespace

namespace Ccz4::BoundaryConditions {

// LCOV_EXCL_START
ConstraintsRadiationPreserving::ConstraintsRadiationPreserving(
    CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}
// LCOV_EXCL_STOP

ConstraintsRadiationPreserving::ConstraintsRadiationPreserving(
    const ConstraintsRadiationPreserving& rhs)
    : BoundaryCondition{dynamic_cast<const BoundaryCondition&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()),
      use_analytic_for_all_(rhs.use_analytic_for_all_),
      penalty_multiplier_(rhs.penalty_multiplier_) {}

ConstraintsRadiationPreserving& ConstraintsRadiationPreserving::operator=(
    const ConstraintsRadiationPreserving& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  use_analytic_for_all_ = rhs.use_analytic_for_all_;
  penalty_multiplier_ = rhs.penalty_multiplier_;
  return *this;
}

ConstraintsRadiationPreserving::ConstraintsRadiationPreserving(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription,
    bool use_analytic_for_all, double penalty_multiplier)
    : analytic_prescription_(std::move(analytic_prescription)),
      use_analytic_for_all_(use_analytic_for_all),
      penalty_multiplier_(penalty_multiplier) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
ConstraintsRadiationPreserving::get_clone() const {
  return std::make_unique<ConstraintsRadiationPreserving>(*this);
}

void ConstraintsRadiationPreserving::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | analytic_prescription_;
  p | use_analytic_for_all_;
  p | penalty_multiplier_;
}
// NOLINTNEXTLINE
PUP::able::PUP_ID ConstraintsRadiationPreserving::my_PUP_ID = 0;

std::optional<std::string> ConstraintsRadiationPreserving::dg_ghost(
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
        u_tensor_minus,
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
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_conformal_metric,
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
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        interior_boundary_u_tensor_minus,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        interior_boundary_conformal_metric,
    const Scalar<DataVector>& interior_boundary_conformal_factor,
    const Scalar<DataVector>& interior_boundary_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_boundary_shift,
    const Scalar<DataVector>& interior_boundary_theta,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interior_boundary_z,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
    [[maybe_unused]] const double time,
    [[maybe_unused]] const bool evolve_lapse_and_shift) const {
  static constexpr size_t Dim = 3;
  static constexpr double f = ::Ccz4::fd::System::f;

  ASSERT(evolve_lapse_and_shift,
         "ConstraintsRadiationPreserving BC requires evolving lapse and "
         "shift.");

  const size_t num_pts = get(interior_conformal_factor).size();

  // Coeff four fields = boundary-integrated fields
  const auto& coeff_conformal_metric = interior_boundary_conformal_metric;
  const auto& coeff_conformal_factor = interior_boundary_conformal_factor;
  const auto& coeff_lapse = interior_boundary_lapse;
  const auto& coeff_shift = interior_boundary_shift;

  // Compute ghost-side unit normal from coeff four fields
  const auto [det_coeff_cm, inv_coeff_cm] =
      determinant_and_inverse(coeff_conformal_metric);

  tnsr::II<DataVector, Dim, Frame::Inertial> inv_coeff_spatial_metric{};
  ::tenex::evaluate<ti::I, ti::J>(make_not_null(&inv_coeff_spatial_metric),
                                  coeff_conformal_factor() *
                                      coeff_conformal_factor() *
                                      inv_coeff_cm(ti::I, ti::J));

  const Scalar<DataVector> coeff_mag_sq =
      dot_product(normal_covector, normal_covector, inv_coeff_spatial_metric);
  const DataVector coeff_inv_mag = 1.0 / sqrt(get(coeff_mag_sq));

  tnsr::i<DataVector, Dim, Frame::Inertial> ghost_unit_normal_one_form(
      num_pts);
  for (size_t i = 0; i < Dim; ++i) {
    ghost_unit_normal_one_form.get(i) = normal_covector.get(i) * coeff_inv_mag;
  }

  tnsr::I<DataVector, Dim, Frame::Inertial> ghost_unit_normal_vector{};
  ::tenex::evaluate<ti::I>(make_not_null(&ghost_unit_normal_vector),
                           inv_coeff_spatial_metric(ti::I, ti::J) *
                               ghost_unit_normal_one_form(ti::j));

  // Run CRPBC pipeline
  auto result = crpbc_characteristic_pipeline(
      *analytic_prescription_, normal_covector, interior_conformal_metric,
      interior_conformal_factor, interior_a_tilde,
      interior_trace_extrinsic_curvature, interior_theta, interior_gamma_hat,
      interior_lapse, interior_shift, interior_auxiliary_shift_b,
      interior_field_a, interior_field_b, interior_field_d, interior_field_p,
      coeff_conformal_metric, coeff_conformal_factor, coeff_lapse, coeff_shift,
      ghost_unit_normal_one_form, ghost_unit_normal_vector,
      interior_boundary_u_tensor_minus, interior_boundary_theta,
      interior_boundary_z, coords, time, f, use_analytic_for_all_);

  // Validate characteristic speed signs
  for (size_t s = 0; s < num_pts; ++s) {
    if (result.char_speeds[0][s] < 0.0 || result.char_speeds[1][s] >= 0.0 ||
        result.char_speeds[5][s] < 0.0 || result.char_speeds[6][s] >= 0.0 ||
        result.char_speeds[12][s] < 0.0 || result.char_speeds[13][s] >= 0.0 ||
        result.char_speeds[14][s] < 0.0 || result.char_speeds[15][s] >= 0.0) {
      ERROR(
          "ConstraintsRadiationPreserving dg_ghost: characteristic speed sign "
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
  *conformal_metric = coeff_conformal_metric;
  *conformal_factor = coeff_conformal_factor;
  *lapse = coeff_lapse;
  *shift = coeff_shift;

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

  *field_a = std::move(result.field_a);
  *field_b = std::move(result.field_b);
  *field_d = std::move(result.field_d);
  *field_p = std::move(result.field_p);

  // Boundary mode exterior values: zero (unused by LF)
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      u_tensor_minus->get(i, j) = 0.0;
    }
  }

  // Boundary second-order fields: passthrough (zero jump)
  *boundary_conformal_metric = interior_boundary_conformal_metric;
  *boundary_conformal_factor = interior_boundary_conformal_factor;
  *boundary_lapse = interior_boundary_lapse;
  *boundary_shift = interior_boundary_shift;
  *boundary_theta = interior_boundary_theta;
  *boundary_z = interior_boundary_z;

  return {};
}

std::optional<std::string> ConstraintsRadiationPreserving::dg_time_derivative(
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
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_conformal_metric,
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
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        interior_boundary_u_tensor_minus,
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
  static constexpr double f_val = ::Ccz4::fd::System::f;
  static constexpr bool shifting_shift = ::Ccz4::fd::System::shifting_shift;

  ASSERT(evolve_lapse_and_shift,
         "ConstraintsRadiationPreserving BC requires evolving lapse and "
         "shift.");

  // Zero all 17 evolved/aux dt corrections
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
  // BoundaryTheta and BoundaryZ dt corrections are zero (no DG jump).
  get(*dt_boundary_theta_correction) = 0.0;
  for (auto& component : *dt_boundary_z_correction) {
    component = 0.0;
  }

  // Compute ghost unit normal from boundary-integrated metric
  const size_t num_pts = get(interior_conformal_factor).size();
  const auto [det_bnd_cm, inv_bnd_cm] =
      determinant_and_inverse(interior_boundary_conformal_metric);

  tnsr::II<DataVector, Dim, Frame::Inertial> inv_bnd_spatial_metric{};
  ::tenex::evaluate<ti::I, ti::J>(make_not_null(&inv_bnd_spatial_metric),
                                  interior_boundary_conformal_factor() *
                                      interior_boundary_conformal_factor() *
                                      inv_bnd_cm(ti::I, ti::J));

  const Scalar<DataVector> bnd_mag_sq =
      dot_product(normal_covector, normal_covector, inv_bnd_spatial_metric);
  const DataVector bnd_inv_mag = 1.0 / sqrt(get(bnd_mag_sq));

  tnsr::i<DataVector, Dim, Frame::Inertial> boundary_unit_normal_one_form(
      num_pts);
  for (size_t i = 0; i < Dim; ++i) {
    boundary_unit_normal_one_form.get(i) =
        normal_covector.get(i) * bnd_inv_mag;
  }

  tnsr::I<DataVector, Dim, Frame::Inertial> boundary_unit_normal_vector{};
  ::tenex::evaluate<ti::I>(make_not_null(&boundary_unit_normal_vector),
                           inv_bnd_spatial_metric(ti::I, ti::J) *
                               boundary_unit_normal_one_form(ti::j));

  // Run CRPBC pipeline
  auto result = crpbc_characteristic_pipeline(
      *analytic_prescription_, normal_covector, interior_conformal_metric,
      interior_conformal_factor, interior_a_tilde,
      interior_trace_extrinsic_curvature, interior_theta, interior_gamma_hat,
      interior_lapse, interior_shift, interior_auxiliary_shift_b,
      interior_field_a, interior_field_b, interior_field_d, interior_field_p,
      interior_boundary_conformal_metric, interior_boundary_conformal_factor,
      interior_boundary_lapse, interior_boundary_shift,
      boundary_unit_normal_one_form, boundary_unit_normal_vector,
      interior_boundary_u_tensor_minus, interior_boundary_theta,
      interior_boundary_z, coords, time, f_val, use_analytic_for_all_);

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
  compute_dt_second_order_fields(
      dt_boundary_conformal_metric_correction,
      dt_boundary_conformal_factor_correction, dt_boundary_lapse_correction,
      dt_boundary_shift_correction, interior_boundary_conformal_metric,
      interior_boundary_conformal_factor, interior_boundary_lapse,
      interior_boundary_shift, mixed_a_tilde, mixed_K, mixed_theta, mixed_b,
      result.field_a, result.field_b, result.field_d, result.field_p, k_0,
      f_val, shifting_shift);

  return {};
}

void ConstraintsRadiationPreserving::fd_ghost(
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
      "ConstraintsRadiationPreserving fd_ghost is not implemented. "
      "This BC is only available for the DG (LDG) path.");
}
}  // namespace Ccz4::BoundaryConditions
