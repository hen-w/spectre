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
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

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
    const gsl::not_null<Scalar<DataVector>*> u_scalar3_minus,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        u_vector2_minus,
    const gsl::not_null<Scalar<DataVector>*> u_scalar2_minus,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        u_tensor_minus,
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
    const Scalar<DataVector>& interior_u_scalar3_minus,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interior_u_vector2_minus,
    const Scalar<DataVector>& interior_u_scalar2_minus,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_u_tensor_minus,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
    [[maybe_unused]] const double time,
    [[maybe_unused]] const bool evolve_lapse_and_shift) const {
  static constexpr size_t Dim = 3;
  static constexpr double f = ::Ccz4::fd::System::f;

  ASSERT(evolve_lapse_and_shift,
         "ConstraintsRadiationPreserving BC requires evolving lapse and "
         "shift.");

  const size_t num_pts = get(interior_conformal_factor).size();

  // ============================================================
  // Phase A: Copy interior metric/gauge to exterior
  // ============================================================
  *conformal_metric = interior_conformal_metric;
  *conformal_factor = interior_conformal_factor;
  *lapse = interior_lapse;
  *shift = interior_shift;

  // A2: Boundary mode exterior values are unused (LaxFriedrichs zeroes the
  // boundary corrections for these tags), so just set them to zero.
  get(*u_scalar3_minus) = 0.0;
  get(*u_scalar2_minus) = 0.0;
  for (size_t i = 0; i < Dim; ++i) {
    u_vector2_minus->get(i) = 0.0;
  }
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      u_tensor_minus->get(i, j) = 0.0;
    }
  }

  // ============================================================
  // Phase B: Reconstruct spatial derivatives from interior auxiliary fields
  // ============================================================
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

  // ============================================================
  // Phase C: Compute unit normal one-form
  // ============================================================
  const auto [det_conformal_metric, inv_conformal_metric] =
      determinant_and_inverse(interior_conformal_metric);

  tnsr::II<DataVector, Dim, Frame::Inertial> inv_spatial_metric{};
  ::tenex::evaluate<ti::I, ti::J>(make_not_null(&inv_spatial_metric),
                                  interior_conformal_factor() *
                                      interior_conformal_factor() *
                                      inv_conformal_metric(ti::I, ti::J));

  const Scalar<DataVector> magnitude_sq =
      dot_product(normal_covector, normal_covector, inv_spatial_metric);
  const DataVector inv_magnitude = 1.0 / sqrt(get(magnitude_sq));

  tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal_one_form(num_pts);
  for (size_t i = 0; i < Dim; ++i) {
    unit_normal_one_form.get(i) = normal_covector.get(i) * inv_magnitude;
  }

  tnsr::I<DataVector, Dim, Frame::Inertial> unit_normal_vector{};
  ::tenex::evaluate<ti::I>(
      make_not_null(&unit_normal_vector),
      inv_spatial_metric(ti::I, ti::J) * unit_normal_one_form(ti::j));

  // Validate interior CCZ4 constraints on the boundary face
  {
    constexpr double tol1 = 1.0e-6;
    const Scalar<DataVector> trace_a_tilde_check =
        trace(interior_a_tilde, inv_conformal_metric);

    ASSERT(max(abs(get(det_conformal_metric) - 1.0)) < tol1,
           "Interior det(conformal_metric) deviates from 1 by "
               << max(abs(get(det_conformal_metric) - 1.0)) << " (tolerance "
               << tol1 << ")");
    ASSERT(max(abs(get(trace_a_tilde_check))) < tol1,
           "Interior trace(A_tilde) deviates from 0 by "
               << max(abs(get(trace_a_tilde_check))) << " (tolerance " << tol1
               << ")");
  }

  // ============================================================
  // Phase D: Compute interior characteristic speeds and fields
  // ============================================================
  const auto char_speeds = ::Ccz4::fd::characteristic_speeds(
      interior_lapse, interior_shift, interior_conformal_factor, f,
      unit_normal_one_form);

  auto char_fields = ::Ccz4::fd::characteristic_fields(
      unit_normal_one_form, interior_conformal_metric,
      interior_conformal_factor, interior_lapse, interior_shift,
      interior_trace_extrinsic_curvature, interior_a_tilde, interior_theta,
      interior_gamma_hat, interior_auxiliary_shift_b, d_conformal_metric,
      d_conformal_factor, d_lapse, d_shift, f);

  // ============================================================
  // Phase D2: Validate characteristic speeds
  // ============================================================
  for (size_t s = 0; s < num_pts; ++s) {
    if (char_speeds[0][s] < 0.0 || char_speeds[1][s] >= 0.0 ||
        char_speeds[5][s] < 0.0 || char_speeds[6][s] >= 0.0 ||
        char_speeds[12][s] < 0.0 || char_speeds[13][s] >= 0.0 ||
        char_speeds[14][s] < 0.0 || char_speeds[15][s] >= 0.0) {
      ERROR(
          "ConstraintsRadiationPreserving dg_ghost: characteristic speed sign "
          "violation at face point "
          << s << ".\n"
          << "  char_speeds[0]  (UTensorPlus,  expect>=0): "
          << char_speeds[0][s] << "\n"
          << "  char_speeds[1]  (UTensorMinus,  expect<0): "
          << char_speeds[1][s] << "\n"
          << "  char_speeds[5]  (UVector3Plus,  expect>=0): "
          << char_speeds[5][s] << "\n"
          << "  char_speeds[6]  (UVector3Minus, expect<0): "
          << char_speeds[6][s] << "\n"
          << "  char_speeds[12] (UScalar4Plus,  expect>=0): "
          << char_speeds[12][s] << "\n"
          << "  char_speeds[13] (UScalar4Minus, expect<0): "
          << char_speeds[13][s] << "\n"
          << "  char_speeds[14] (UScalar5Plus,  expect>=0): "
          << char_speeds[14][s] << "\n"
          << "  char_speeds[15] (UScalar5Minus, expect<0): "
          << char_speeds[15][s]);
    }
  }

  // ============================================================
  // Phase E: Replace incoming modes with time-integrated boundary mode values
  // ============================================================
  using namespace ::Ccz4::fd::Tags;

  auto& u_tnsr_plus =
      get<UTensorPlus<DataVector, Dim, Frame::Inertial>>(char_fields);
  auto& u_tnsr_minus =
      get<UTensorMinus<DataVector, Dim, Frame::Inertial>>(char_fields);
  auto& u_vector1_zero =
      get<UVector1Zero<DataVector, Dim, Frame::Inertial>>(char_fields);
  auto& u_vector2_minus_field =
      get<UVector2Minus<DataVector, Dim, Frame::Inertial>>(char_fields);
  auto& u_vector3_minus_field =
      get<UVector3Minus<DataVector, Dim, Frame::Inertial>>(char_fields);
  auto& u_scalar1_zero = get<UScalar1Zero<DataVector>>(char_fields);
  auto& u_scalar2_minus_field = get<UScalar2Minus<DataVector>>(char_fields);
  auto& u_scalar3_minus_field = get<UScalar3Minus<DataVector>>(char_fields);
  auto& u_scalar4_minus_field = get<UScalar4Minus<DataVector>>(char_fields);
  auto& u_scalar5_minus_field = get<UScalar5Minus<DataVector>>(char_fields);

  // Gauge modes: get from analytic prescription
  using all_tags =
      tmpl::list<Tags::ConformalMetric<DataVector, 3>,
                 Tags::ConformalFactor<DataVector>, Tags::ATilde<DataVector, 3>,
                 gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 Tags::Theta<DataVector>, Tags::GammaHat<DataVector, 3>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 Tags::AuxiliaryShiftB<DataVector, 3>,
                 Tags::FieldA<DataVector, 3>, Tags::FieldB<DataVector, 3>,
                 Tags::FieldD<DataVector, 3>, Tags::FieldP<DataVector, 3>>;
  auto analytic_values = call_with_dynamic_type<
      tuples::TaggedTuple<
          Tags::ConformalMetric<DataVector, 3>,
          Tags::ConformalFactor<DataVector>, Tags::ATilde<DataVector, 3>,
          gr::Tags::TraceExtrinsicCurvature<DataVector>,
          Tags::Theta<DataVector>, Tags::GammaHat<DataVector, 3>,
          gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
          Tags::AuxiliaryShiftB<DataVector, 3>, Tags::FieldA<DataVector, 3>,
          Tags::FieldB<DataVector, 3>, Tags::FieldD<DataVector, 3>,
          Tags::FieldP<DataVector, 3>>,
      Ccz4::Solutions::all_solutions>(
      analytic_prescription_.get(),
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

  // Compute analytic characteristic fields for gauge modes
  const auto& analytic_conformal_metric =
      get<Tags::ConformalMetric<DataVector, 3>>(analytic_values);
  const auto& analytic_conformal_factor =
      get<Tags::ConformalFactor<DataVector>>(analytic_values);
  const auto& analytic_a_tilde =
      get<Tags::ATilde<DataVector, 3>>(analytic_values);
  const auto& analytic_trace_extrinsic_curvature =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(analytic_values);
  const auto& analytic_theta = get<Tags::Theta<DataVector>>(analytic_values);
  const auto& analytic_gamma_hat =
      get<Tags::GammaHat<DataVector, 3>>(analytic_values);
  const auto& analytic_lapse =
      get<gr::Tags::Lapse<DataVector>>(analytic_values);
  const auto& analytic_shift =
      get<gr::Tags::Shift<DataVector, 3>>(analytic_values);
  const auto& analytic_auxiliary_shift_b =
      get<Tags::AuxiliaryShiftB<DataVector, 3>>(analytic_values);
  const auto& analytic_field_a =
      get<Tags::FieldA<DataVector, 3>>(analytic_values);
  const auto& analytic_field_b =
      get<Tags::FieldB<DataVector, 3>>(analytic_values);
  const auto& analytic_field_d =
      get<Tags::FieldD<DataVector, 3>>(analytic_values);
  const auto& analytic_field_p =
      get<Tags::FieldP<DataVector, 3>>(analytic_values);

  tnsr::ijj<DataVector, Dim, Frame::Inertial> analytic_d_conformal_metric{};
  ::tenex::evaluate<ti::k, ti::i, ti::j>(
      make_not_null(&analytic_d_conformal_metric),
      2.0 * analytic_field_d(ti::k, ti::i, ti::j));
  tnsr::i<DataVector, Dim, Frame::Inertial> analytic_d_conformal_factor{};
  ::tenex::evaluate<ti::i>(
      make_not_null(&analytic_d_conformal_factor),
      analytic_conformal_factor() * analytic_field_p(ti::i));
  tnsr::i<DataVector, Dim, Frame::Inertial> analytic_d_lapse{};
  ::tenex::evaluate<ti::i>(make_not_null(&analytic_d_lapse),
                           analytic_lapse() * analytic_field_a(ti::i));
  tnsr::iJ<DataVector, Dim, Frame::Inertial> analytic_d_shift{};
  ::tenex::evaluate<ti::i, ti::J>(make_not_null(&analytic_d_shift),
                                  analytic_field_b(ti::i, ti::J));

  const auto analytic_char_fields = ::Ccz4::fd::characteristic_fields(
      unit_normal_one_form, analytic_conformal_metric,
      analytic_conformal_factor, analytic_lapse, analytic_shift,
      analytic_trace_extrinsic_curvature, analytic_a_tilde, analytic_theta,
      analytic_gamma_hat, analytic_auxiliary_shift_b,
      analytic_d_conformal_metric, analytic_d_conformal_factor,
      analytic_d_lapse, analytic_d_shift, f);

  // Replace the 4 incoming CRPBC modes
  if (use_analytic_for_all_) {
    // Debug mode: prescribe ALL incoming modes from analytic solution
    // (same behavior as DirichletCharacteristics)
    u_tnsr_minus = get<UTensorMinus<DataVector, Dim, Frame::Inertial>>(
        analytic_char_fields);
    u_vector2_minus_field =
        get<UVector2Minus<DataVector, Dim, Frame::Inertial>>(
            analytic_char_fields);
    u_scalar2_minus_field =
        get<UScalar2Minus<DataVector>>(analytic_char_fields);
    u_scalar3_minus_field = get<UScalar3Plus<DataVector>>(char_fields);
  } else {
    // Normal CRPBC mode: use time-integrated boundary-mode evolved variables
    u_tnsr_minus = get<UTensorMinus<DataVector, Dim, Frame::Inertial>>(
        analytic_char_fields);
    u_vector2_minus_field = interior_u_vector2_minus;
    u_scalar2_minus_field = interior_u_scalar2_minus;
    u_scalar3_minus_field = interior_u_scalar3_minus;
  }

  // Set gauge modes from analytic
  u_vector3_minus_field = get<UVector3Minus<DataVector, Dim, Frame::Inertial>>(
      analytic_char_fields);
  u_scalar4_minus_field = get<UScalar4Minus<DataVector>>(analytic_char_fields);
  u_scalar5_minus_field = get<UScalar5Minus<DataVector>>(analytic_char_fields);

  // Zero-speed modes: set from analytic if incoming
  const auto& analytic_u_vector1_zero =
      get<UVector1Zero<DataVector, Dim, Frame::Inertial>>(analytic_char_fields);
  const auto& analytic_u_scalar1_zero =
      get<UScalar1Zero<DataVector>>(analytic_char_fields);

  for (size_t s = 0; s < num_pts; ++s) {
    if (char_speeds[2][s] < 0.0) {
      for (size_t i = 0; i < Dim; ++i) {
        u_vector1_zero.get(i)[s] = analytic_u_vector1_zero.get(i)[s];
      }
    }
    if (char_speeds[7][s] < 0.0) {
      get(u_scalar1_zero)[s] = get(analytic_u_scalar1_zero)[s];
    }
  }

  // ============================================================
  // Phase F: Inverse characteristic transform
  // ============================================================
  const auto evolved_space =
      ::Ccz4::fd::evolved_space_from_characteristic_fields(
          u_tnsr_plus, u_tnsr_minus, u_vector1_zero,
          get<UVector2Plus<DataVector, Dim, Frame::Inertial>>(char_fields),
          u_vector2_minus_field,
          get<UVector3Plus<DataVector, Dim, Frame::Inertial>>(char_fields),
          u_vector3_minus_field, u_scalar1_zero,
          get<UScalar2Plus<DataVector>>(char_fields), u_scalar2_minus_field,
          get<UScalar3Plus<DataVector>>(char_fields), u_scalar3_minus_field,
          get<UScalar4Plus<DataVector>>(char_fields), u_scalar4_minus_field,
          get<UScalar5Plus<DataVector>>(char_fields), u_scalar5_minus_field,
          unit_normal_one_form, interior_conformal_metric,
          interior_conformal_factor, interior_lapse, interior_shift, f);

  // ============================================================
  // Phase G: Write exterior state
  // ============================================================
  using DnCM =
      ::Ccz4::fd::Tags::DnConformalMetric<DataVector, Dim, Frame::Inertial>;
  using DnL = ::Ccz4::fd::Tags::DnLapse<DataVector>;
  using DnS = ::Ccz4::fd::Tags::DnShift<DataVector, Dim, Frame::Inertial>;
  using DnCF = ::Ccz4::fd::Tags::DnConformalFactor<DataVector>;

  *a_tilde = get<::Ccz4::Tags::ATilde<DataVector, Dim, Frame::Inertial>>(
      evolved_space);
  *trace_extrinsic_curvature =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_space);
  *theta = get<::Ccz4::Tags::Theta<DataVector>>(evolved_space);
  *gamma_hat = get<::Ccz4::Tags::GammaHat<DataVector, Dim, Frame::Inertial>>(
      evolved_space);
  *auxiliary_shift_b =
      get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim, Frame::Inertial>>(
          evolved_space);

  const auto& dn_conformal_metric = get<DnCM>(evolved_space);
  const auto& dn_lapse = get<DnL>(evolved_space);
  const auto& dn_shift = get<DnS>(evolved_space);
  const auto& dn_conformal_factor = get<DnCF>(evolved_space);

  // Reconstruct auxiliary fields from normal derivatives
  Scalar<DataVector> n_dot_d_lapse{};
  ::tenex::evaluate(make_not_null(&n_dot_d_lapse),
                    unit_normal_vector(ti::I) * d_lapse(ti::i));
  Scalar<DataVector> n_dot_d_cf{};
  ::tenex::evaluate(make_not_null(&n_dot_d_cf),
                    unit_normal_vector(ti::I) * d_conformal_factor(ti::i));

  ::tenex::evaluate<ti::i>(
      field_a, (d_lapse(ti::i) - unit_normal_one_form(ti::i) * n_dot_d_lapse() +
                unit_normal_one_form(ti::i) * dn_lapse()) /
                   interior_lapse());

  ::tenex::evaluate<ti::i>(
      field_p,
      (d_conformal_factor(ti::i) - unit_normal_one_form(ti::i) * n_dot_d_cf() +
       unit_normal_one_form(ti::i) * dn_conformal_factor()) /
          interior_conformal_factor());

  tnsr::ii<DataVector, Dim, Frame::Inertial> n_dot_d_cm{};
  ::tenex::evaluate<ti::j, ti::k>(
      make_not_null(&n_dot_d_cm),
      unit_normal_vector(ti::M) * d_conformal_metric(ti::m, ti::j, ti::k));

  ::tenex::evaluate<ti::i, ti::j, ti::k>(
      field_d,
      0.5 * (d_conformal_metric(ti::i, ti::j, ti::k) -
             unit_normal_one_form(ti::i) * n_dot_d_cm(ti::j, ti::k) +
             unit_normal_one_form(ti::i) * dn_conformal_metric(ti::j, ti::k)));

  tnsr::I<DataVector, Dim, Frame::Inertial> n_dot_d_shift{};
  ::tenex::evaluate<ti::J>(make_not_null(&n_dot_d_shift),
                           unit_normal_vector(ti::M) * d_shift(ti::m, ti::J));

  ::tenex::evaluate<ti::i, ti::J>(
      field_b, d_shift(ti::i, ti::J) -
                   unit_normal_one_form(ti::i) * n_dot_d_shift(ti::J) +
                   unit_normal_one_form(ti::i) * dn_shift(ti::J));

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
