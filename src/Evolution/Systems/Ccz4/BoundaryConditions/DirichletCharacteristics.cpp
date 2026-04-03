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
    const gsl::not_null<Scalar<DataVector>*> bm_u_scalar3_minus,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        bm_u_vector2_minus,
    const gsl::not_null<Scalar<DataVector>*> bm_u_scalar2_minus,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        bm_u_tensor_minus,
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
    const Scalar<DataVector>& /*interior_u_scalar3_minus*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*interior_u_vector2_minus*/,
    const Scalar<DataVector>& /*interior_u_scalar2_minus*/,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& /*interior_u_tensor_minus*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
    [[maybe_unused]] const double time,
    [[maybe_unused]] const bool evolve_lapse_and_shift) const {
  static constexpr size_t Dim = 3;
  static constexpr double f = ::Ccz4::fd::System::f;

  ASSERT(evolve_lapse_and_shift,
         "DirichletCharacteristics BC is not implemented"
         " when lapse and shift are not evolved, as the SoCcz4 system"
         " is only weakly hyperbolic in that case");

  const size_t num_pts = get(interior_conformal_factor).size();

  // ============================================================
  // Phase A: Copy interior metric/gauge to exterior
  // ============================================================
  *conformal_metric = interior_conformal_metric;
  *conformal_factor = interior_conformal_factor;
  *lapse = interior_lapse;
  *shift = interior_shift;

  // ============================================================
  // Phase B: Reconstruct spatial derivatives from interior auxiliary fields
  // ============================================================
  // d_conformal_metric_ijk = 2 * field_d_ijk
  tnsr::ijj<DataVector, Dim, Frame::Inertial> d_conformal_metric{};
  ::tenex::evaluate<ti::k, ti::i, ti::j>(
      make_not_null(&d_conformal_metric),
      2.0 * interior_field_d(ti::k, ti::i, ti::j));

  // d_conformal_factor_i = conformal_factor * field_p_i
  tnsr::i<DataVector, Dim, Frame::Inertial> d_conformal_factor{};
  ::tenex::evaluate<ti::i>(
      make_not_null(&d_conformal_factor),
      interior_conformal_factor() * interior_field_p(ti::i));

  // d_lapse_i = lapse * field_a_i
  tnsr::i<DataVector, Dim, Frame::Inertial> d_lapse{};
  ::tenex::evaluate<ti::i>(make_not_null(&d_lapse),
                           interior_lapse() * interior_field_a(ti::i));

  // d_shift_iJ = field_b_iJ
  tnsr::iJ<DataVector, Dim, Frame::Inertial> d_shift{};
  ::tenex::evaluate<ti::i, ti::J>(make_not_null(&d_shift),
                                  interior_field_b(ti::i, ti::J));

  // ============================================================
  // Phase C: Compute unit normal one-form
  // ============================================================
  const auto [det_conformal_metric, inv_conformal_metric] =
      determinant_and_inverse(interior_conformal_metric);

  // inverse physical spatial metric: gamma^ij = phi^2 * tilde_gamma^ij
  tnsr::II<DataVector, Dim, Frame::Inertial> inv_spatial_metric{};
  ::tenex::evaluate<ti::I, ti::J>(make_not_null(&inv_spatial_metric),
                                  interior_conformal_factor() *
                                      interior_conformal_factor() *
                                      inv_conformal_metric(ti::I, ti::J));

  // magnitude^2 = gamma^ij * normal_covector_i * normal_covector_j
  const Scalar<DataVector> magnitude_sq =
      dot_product(normal_covector, normal_covector, inv_spatial_metric);
  const DataVector inv_magnitude = 1.0 / sqrt(get(magnitude_sq));

  // unit_normal_one_form_i = normal_covector_i / sqrt(magnitude^2)
  tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal_one_form(num_pts);
  for (size_t i = 0; i < Dim; ++i) {
    unit_normal_one_form.get(i) = normal_covector.get(i) * inv_magnitude;
  }

  // unit_normal_vector^i = gamma^ij * unit_normal_one_form_j
  tnsr::I<DataVector, Dim, Frame::Inertial> unit_normal_vector{};
  ::tenex::evaluate<ti::I>(
      make_not_null(&unit_normal_vector),
      inv_spatial_metric(ti::I, ti::J) * unit_normal_one_form(ti::j));

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
  // For asymptotically Minkowski spacetimes, certain speeds must have
  // definite signs. Check at each face point.
  for (size_t s = 0; s < num_pts; ++s) {
    if (char_speeds[0][s] < 0.0 || char_speeds[1][s] >= 0.0 ||
        char_speeds[5][s] < 0.0 || char_speeds[6][s] >= 0.0 ||
        char_speeds[12][s] < 0.0 || char_speeds[13][s] >= 0.0 ||
        char_speeds[14][s] < 0.0 || char_speeds[15][s] >= 0.0) {
      ERROR(
          "DirichletCharacteristics dg_ghost: characteristic speed sign "
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
  // Phase E: Replace incoming characteristic modes with analytic values
  // ============================================================

  // E1: Evaluate analytic solution at (coords, time) to get all 13 fields
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

  // E2: Reconstruct analytic spatial derivatives from analytic auxiliary fields
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

  // E3: Compute analytic characteristic fields using same unit normal
  const auto analytic_char_fields = ::Ccz4::fd::characteristic_fields(
      unit_normal_one_form, analytic_conformal_metric,
      analytic_conformal_factor, analytic_lapse, analytic_shift,
      analytic_trace_extrinsic_curvature, analytic_a_tilde, analytic_theta,
      analytic_gamma_hat, analytic_auxiliary_shift_b,
      analytic_d_conformal_metric, analytic_d_conformal_factor,
      analytic_d_lapse, analytic_d_shift, f);

  // E4: Replace incoming (or outgoing, if prescribe_outgoing_) modes with
  // analytic values.
  using namespace ::Ccz4::fd::Tags;

  // Access interior characteristic fields (mutable)
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

  // Access analytic characteristic fields (const)
  const auto& analytic_u_tnsr_plus =
      get<UTensorPlus<DataVector, Dim, Frame::Inertial>>(analytic_char_fields);
  const auto& analytic_u_tnsr_minus =
      get<UTensorMinus<DataVector, Dim, Frame::Inertial>>(analytic_char_fields);
  const auto& analytic_u_vector1_zero =
      get<UVector1Zero<DataVector, Dim, Frame::Inertial>>(analytic_char_fields);
  const auto& analytic_u_vector2_plus =
      get<UVector2Plus<DataVector, Dim, Frame::Inertial>>(analytic_char_fields);
  const auto& analytic_u_vector2_minus =
      get<UVector2Minus<DataVector, Dim, Frame::Inertial>>(
          analytic_char_fields);
  const auto& analytic_u_vector3_plus =
      get<UVector3Plus<DataVector, Dim, Frame::Inertial>>(analytic_char_fields);
  const auto& analytic_u_vector3_minus =
      get<UVector3Minus<DataVector, Dim, Frame::Inertial>>(
          analytic_char_fields);
  const auto& analytic_u_scalar1_zero =
      get<UScalar1Zero<DataVector>>(analytic_char_fields);
  const auto& analytic_u_scalar2_plus =
      get<UScalar2Plus<DataVector>>(analytic_char_fields);
  const auto& analytic_u_scalar2_minus =
      get<UScalar2Minus<DataVector>>(analytic_char_fields);
  const auto& analytic_u_scalar3_plus =
      get<UScalar3Plus<DataVector>>(analytic_char_fields);
  const auto& analytic_u_scalar3_minus =
      get<UScalar3Minus<DataVector>>(analytic_char_fields);
  const auto& analytic_u_scalar4_plus =
      get<UScalar4Plus<DataVector>>(analytic_char_fields);
  const auto& analytic_u_scalar4_minus =
      get<UScalar4Minus<DataVector>>(analytic_char_fields);
  const auto& analytic_u_scalar5_plus =
      get<UScalar5Plus<DataVector>>(analytic_char_fields);
  const auto& analytic_u_scalar5_minus =
      get<UScalar5Minus<DataVector>>(analytic_char_fields);

  if (not prescribe_outgoing_) {
    // Normal mode: replace INCOMING modes with analytic values

    // UTensorMinus (speed[1], always incoming for outward boundaries)
    u_tnsr_minus = analytic_u_tnsr_minus;

    // UVector1Zero (speed[2] = -beta^n, conditionally incoming)
    for (size_t s = 0; s < num_pts; ++s) {
      if (char_speeds[2][s] < 0.0) {
        for (size_t i = 0; i < Dim; ++i) {
          u_vector1_zero.get(i)[s] = analytic_u_vector1_zero.get(i)[s];
        }
      }
    }

    // UVector2Minus (speed[4], always incoming)
    u_vector2_minus = analytic_u_vector2_minus;

    // UVector3Minus (speed[6], always incoming)
    u_vector3_minus = analytic_u_vector3_minus;

    // UScalar1Zero (speed[7] = -beta^n, conditionally incoming)
    for (size_t s = 0; s < num_pts; ++s) {
      if (char_speeds[7][s] < 0.0) {
        get(u_scalar1_zero)[s] = get(analytic_u_scalar1_zero)[s];
      }
    }

    // UScalar2Minus (speed[9], always incoming)
    u_scalar2_minus = analytic_u_scalar2_minus;

    // UScalar3Minus (speed[11], always incoming)
    u_scalar3_minus = analytic_u_scalar3_minus;

    // UScalar4Minus (speed[13], always incoming)
    u_scalar4_minus = analytic_u_scalar4_minus;

    // UScalar5Minus (speed[15], always incoming)
    u_scalar5_minus = analytic_u_scalar5_minus;
  } else {
    // Kill-switch mode: replace OUTGOING modes with analytic values

    // UTensorPlus (speed[0], always outgoing for outward boundaries)
    u_tnsr_plus = analytic_u_tnsr_plus;

    // UVector1Zero (speed[2] = -beta^n, outgoing when speed >= 0)
    for (size_t s = 0; s < num_pts; ++s) {
      if (char_speeds[2][s] >= 0.0) {
        for (size_t i = 0; i < Dim; ++i) {
          u_vector1_zero.get(i)[s] = analytic_u_vector1_zero.get(i)[s];
        }
      }
    }

    // UVector2Plus (speed[3], always outgoing)
    u_vector2_plus = analytic_u_vector2_plus;

    // UVector3Plus (speed[5], always outgoing)
    u_vector3_plus = analytic_u_vector3_plus;

    // UScalar1Zero (speed[7] = -beta^n, outgoing when speed >= 0)
    for (size_t s = 0; s < num_pts; ++s) {
      if (char_speeds[7][s] >= 0.0) {
        get(u_scalar1_zero)[s] = get(analytic_u_scalar1_zero)[s];
      }
    }

    // UScalar2Plus (speed[8], always outgoing)
    u_scalar2_plus = analytic_u_scalar2_plus;

    // UScalar3Plus (speed[10], always outgoing)
    u_scalar3_plus = analytic_u_scalar3_plus;

    // UScalar4Plus (speed[12], always outgoing)
    u_scalar4_plus = analytic_u_scalar4_plus;

    // UScalar5Plus (speed[14], always outgoing)
    u_scalar5_plus = analytic_u_scalar5_plus;
  }

  // ============================================================
  // Phase F: Inverse characteristic transform
  // ============================================================
  const auto evolved_space =
      ::Ccz4::fd::evolved_space_from_characteristic_fields(
          u_tnsr_plus, u_tnsr_minus, u_vector1_zero, u_vector2_plus,
          u_vector2_minus, u_vector3_plus, u_vector3_minus, u_scalar1_zero,
          u_scalar2_plus, u_scalar2_minus, u_scalar3_plus, u_scalar3_minus,
          u_scalar4_plus, u_scalar4_minus, u_scalar5_plus, u_scalar5_minus,
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

  // Directly from inverse transform (evolved variables with independent BC):
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

  // Normal derivatives from inverse transform
  const auto& dn_conformal_metric = get<DnCM>(evolved_space);
  const auto& dn_lapse = get<DnL>(evolved_space);
  const auto& dn_shift = get<DnS>(evolved_space);
  const auto& dn_conformal_factor = get<DnCF>(evolved_space);

  // Reconstruct auxiliary fields from normal derivatives.
  // The full spatial derivative on the exterior is:
  //   d_i u_ext = q^k_i d_k u_int + n_i * dn_u_ext
  // Equivalently:
  //   d_i u_ext = d_i u_int - n_i*(n^k d_k u_int) + n_i * dn_u_ext

  // n^k d_lapse_k (scalar contraction)
  Scalar<DataVector> n_dot_d_lapse{};
  ::tenex::evaluate(make_not_null(&n_dot_d_lapse),
                    unit_normal_vector(ti::I) * d_lapse(ti::i));

  // n^k d_cf_k (scalar contraction)
  Scalar<DataVector> n_dot_d_cf{};
  ::tenex::evaluate(make_not_null(&n_dot_d_cf),
                    unit_normal_vector(ti::I) * d_conformal_factor(ti::i));

  // field_a_i_ext = (d_lapse_i - n_i*n^k*d_lapse_k + n_i*dn_lapse) / lapse
  ::tenex::evaluate<ti::i>(
      field_a, (d_lapse(ti::i) - unit_normal_one_form(ti::i) * n_dot_d_lapse() +
                unit_normal_one_form(ti::i) * dn_lapse()) /
                   interior_lapse());

  // field_p_i_ext = (d_cf_i - n_i*n^k*d_cf_k + n_i*dn_cf) / phi
  ::tenex::evaluate<ti::i>(
      field_p,
      (d_conformal_factor(ti::i) - unit_normal_one_form(ti::i) * n_dot_d_cf() +
       unit_normal_one_form(ti::i) * dn_conformal_factor()) /
          interior_conformal_factor());

  // n^m d_conformal_metric_mjk
  tnsr::ii<DataVector, Dim, Frame::Inertial> n_dot_d_cm{};
  ::tenex::evaluate<ti::j, ti::k>(
      make_not_null(&n_dot_d_cm),
      unit_normal_vector(ti::M) * d_conformal_metric(ti::m, ti::j, ti::k));

  // field_d_ijk_ext = 0.5*(d_cm_ijk - n_i*n^m*d_cm_mjk + n_i*dn_cm_jk)
  ::tenex::evaluate<ti::i, ti::j, ti::k>(
      field_d,
      0.5 * (d_conformal_metric(ti::i, ti::j, ti::k) -
             unit_normal_one_form(ti::i) * n_dot_d_cm(ti::j, ti::k) +
             unit_normal_one_form(ti::i) * dn_conformal_metric(ti::j, ti::k)));

  // n^m d_shift_mJ
  tnsr::I<DataVector, Dim, Frame::Inertial> n_dot_d_shift{};
  ::tenex::evaluate<ti::J>(make_not_null(&n_dot_d_shift),
                           unit_normal_vector(ti::M) * d_shift(ti::m, ti::J));

  // field_b_iJ_ext = d_shift_iJ - n_i*n^m*d_shift_mJ + n_i*dn_shift_J
  ::tenex::evaluate<ti::i, ti::J>(
      field_b, d_shift(ti::i, ti::J) -
                   unit_normal_one_form(ti::i) * n_dot_d_shift(ti::J) +
                   unit_normal_one_form(ti::i) * dn_shift(ti::J));

  // Boundary mode exterior values: zero (corrections are zero for these tags)
  *bm_u_scalar3_minus =
      make_with_value<Scalar<DataVector>>(get(interior_conformal_factor), 0.0);
  for (auto& component : *bm_u_vector2_minus) {
    component = 0.0;
  }
  *bm_u_scalar2_minus =
      make_with_value<Scalar<DataVector>>(get(interior_conformal_factor), 0.0);
  for (auto& component : *bm_u_tensor_minus) {
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
