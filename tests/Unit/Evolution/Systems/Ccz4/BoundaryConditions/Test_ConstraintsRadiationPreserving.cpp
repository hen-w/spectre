// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/ConstraintsRadiationPreserving.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Ccz4/Ccz4WrappedGr.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Characteristics.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/Z4Constraint.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            Ccz4::BoundaryConditions::BoundaryCondition,
            tmpl::list<Ccz4::BoundaryConditions::ConstraintsRadiationPreserving>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   Ccz4::Solutions::all_solutions>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>>;
  };
};

void test_creation_and_serialization() {
  register_factory_classes_with_charm<Metavariables>();
  const auto boundary_condition = TestHelpers::test_creation<
      std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>,
      Metavariables>(
      "ConstraintsRadiationPreserving:\n"
      "  AnalyticPrescription:\n"
      "    Ccz4(Minkowski):\n"
      "  UseAnalyticForAll: false\n"
      "  PenaltyMultiplier: 1.0\n"
      "  ZeroBoundaryThetaAndZ: false\n");

  CHECK(boundary_condition->get_clone() != nullptr);

  const auto serialized = serialize_and_deserialize(
      *dynamic_cast<
          Ccz4::BoundaryConditions::ConstraintsRadiationPreserving*>(
          boundary_condition.get()));
  CHECK(serialized.get_clone() != nullptr);
}

void test_bc_type() {
  CHECK(Ccz4::BoundaryConditions::ConstraintsRadiationPreserving::bc_type ==
        evolution::BoundaryConditions::Type::GhostAndTimeDerivative);
}

// Functional test on KerrSchild using UseAnalyticForAll=true, which degenerates
// the CRPBC to DirichletCharacteristics-like behavior (all incoming modes from
// the ghost-side analytic characteristic decomposition rather than
// time-integrated boundary modes). When interior data equals the analytic
// solution and boundary-integrated fields equal the interior, the ghost state
// should reproduce the analytic solution.
void test_kerrschild_use_analytic_for_all() {
  register_factory_classes_with_charm<Metavariables>();

  const auto bc_ptr = TestHelpers::test_creation<
      std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>,
      Metavariables>(
      "ConstraintsRadiationPreserving:\n"
      "  AnalyticPrescription:\n"
      "    Ccz4(KerrSchild):\n"
      "      Mass: 2.0\n"
      "      Spin: [0.2, 0.4, 0.8]\n"
      "      Center: [0.2, 0.5, 0.1]\n"
      "      Velocity: [0.0, 0.0, 0.0]\n"
      "  UseAnalyticForAll: true\n"
      "  PenaltyMultiplier: 1.0\n"
      "  ZeroBoundaryThetaAndZ: false\n");
  const auto& bc = dynamic_cast<
      const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving&>(*bc_ptr);

  static constexpr size_t Dim = 3;
  const size_t num_pts = 5;

  // Face coordinates well outside the horizon (r ~ 5 >> 2M = 4)
  tnsr::I<DataVector, Dim, Frame::Inertial> coords(num_pts, 0.0);
  for (size_t i = 0; i < num_pts; ++i) {
    coords.get(0)[i] = 5.0 + 0.1 * static_cast<double>(i);
    coords.get(1)[i] = 0.1 * static_cast<double>(i);
    coords.get(2)[i] = -0.1 * static_cast<double>(i);
  }

  // Evaluate KerrSchild via Ccz4WrappedGr to get data that the BC would
  // internally evaluate from its analytic_prescription.
  const double time = 0.0;
  const gr::Solutions::KerrSchild kerr_schild(
      2.0, std::array<double, 3>{{0.2, 0.4, 0.8}},
      std::array<double, 3>{{0.2, 0.5, 0.1}});
  const Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::KerrSchild>
      wrapped_solution(kerr_schild);

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
  const auto analytic_values =
      wrapped_solution.variables(coords, time, all_tags{});

  const auto& interior_conformal_metric =
      get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(analytic_values);
  const auto& interior_conformal_factor =
      get<Ccz4::Tags::ConformalFactor<DataVector>>(analytic_values);
  const auto& interior_a_tilde =
      get<Ccz4::Tags::ATilde<DataVector, 3>>(analytic_values);
  const auto& interior_trace_K =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(analytic_values);
  const auto& interior_theta =
      get<Ccz4::Tags::Theta<DataVector>>(analytic_values);
  const auto& interior_gamma_hat =
      get<Ccz4::Tags::GammaHat<DataVector, 3>>(analytic_values);
  const auto& interior_lapse =
      get<gr::Tags::Lapse<DataVector>>(analytic_values);
  const auto& interior_shift =
      get<gr::Tags::Shift<DataVector, 3>>(analytic_values);
  const auto& interior_auxiliary_shift_b =
      get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(analytic_values);
  const auto& interior_field_a =
      get<Ccz4::Tags::FieldA<DataVector, 3>>(analytic_values);
  const auto& interior_field_b =
      get<Ccz4::Tags::FieldB<DataVector, 3>>(analytic_values);
  const auto& interior_field_d =
      get<Ccz4::Tags::FieldD<DataVector, 3>>(analytic_values);
  const auto& interior_field_p =
      get<Ccz4::Tags::FieldP<DataVector, 3>>(analytic_values);

  // Time-integrated boundary mode values: zero (constraints satisfied).
  // Ignored by dg_ghost when UseAnalyticForAll=true, so their value
  // doesn't affect this test.
  const auto interior_boundary_u_tensor_minus =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(num_pts, 0.0);

  // Boundary-integrated second-order fields equal the interior (consistent
  // initial data), so coeff = interior = analytic.
  const auto& interior_boundary_conformal_metric = interior_conformal_metric;
  const auto& interior_boundary_conformal_factor = interior_conformal_factor;
  const auto& interior_boundary_lapse = interior_lapse;
  const auto& interior_boundary_shift = interior_shift;
  const auto interior_boundary_theta =
      make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  const auto interior_boundary_z =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(num_pts, 0.0);

  auto normal_covector =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(num_pts, 0.0);
  normal_covector.get(0) = 1.0;

  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
      face_mesh_velocity{};
  const bool evolve_lapse_and_shift = true;

  // ---- dg_ghost ----
  auto ghost_conformal_metric =
      make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_conformal_factor =
      make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_a_tilde = make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_trace_K = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_theta = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_gamma_hat =
      make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_lapse = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_shift = make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_auxiliary_shift_b =
      make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_field_a = make_with_value<tnsr::i<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_field_b = make_with_value<tnsr::iJ<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_field_d =
      make_with_value<tnsr::ijj<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_field_p = make_with_value<tnsr::i<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_u_tensor_minus =
      make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_boundary_lapse = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_boundary_shift =
      make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_boundary_theta =
      make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_boundary_z =
      make_with_value<tnsr::i<DataVector, Dim>>(num_pts, 0.0);

  const auto ghost_result = bc.dg_ghost(
      make_not_null(&ghost_conformal_metric),
      make_not_null(&ghost_conformal_factor), make_not_null(&ghost_a_tilde),
      make_not_null(&ghost_trace_K), make_not_null(&ghost_theta),
      make_not_null(&ghost_gamma_hat), make_not_null(&ghost_lapse),
      make_not_null(&ghost_shift), make_not_null(&ghost_auxiliary_shift_b),
      make_not_null(&ghost_field_a), make_not_null(&ghost_field_b),
      make_not_null(&ghost_field_d), make_not_null(&ghost_field_p),
      make_not_null(&ghost_u_tensor_minus),
      make_not_null(&ghost_boundary_conformal_metric),
      make_not_null(&ghost_boundary_conformal_factor),
      make_not_null(&ghost_boundary_lapse),
      make_not_null(&ghost_boundary_shift),
      make_not_null(&ghost_boundary_theta),
      make_not_null(&ghost_boundary_z), face_mesh_velocity, normal_covector,
      interior_conformal_metric, interior_conformal_factor, interior_a_tilde,
      interior_trace_K, interior_theta, interior_gamma_hat, interior_lapse,
      interior_shift, interior_auxiliary_shift_b, interior_field_a,
      interior_field_b, interior_field_d, interior_field_p,
      interior_boundary_u_tensor_minus,
      interior_boundary_conformal_metric, interior_boundary_conformal_factor,
      interior_boundary_lapse, interior_boundary_shift,
      interior_boundary_theta, interior_boundary_z,
      coords, time, evolve_lapse_and_shift);

  CHECK_FALSE(ghost_result.has_value());

  // With interior == analytic == coeff and UseAnalyticForAll=true, the ghost
  // state should reproduce the analytic KerrSchild solution.
  const Approx custom_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_conformal_metric,
                               interior_conformal_metric, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_conformal_factor,
                               interior_conformal_factor, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_a_tilde, interior_a_tilde, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_trace_K, interior_trace_K, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_theta, interior_theta, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_gamma_hat, interior_gamma_hat,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_lapse, interior_lapse, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_shift, interior_shift, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_field_a, interior_field_a, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_field_b, interior_field_b, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_field_d, interior_field_d, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_field_p, interior_field_p, custom_approx);

  // ---- dg_time_derivative ----
  auto dt_cm = make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto dt_cf = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto dt_a_tilde = make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto dt_K = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto dt_theta = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto dt_gamma_hat = make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto dt_lapse = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto dt_shift = make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto dt_b = make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto dt_field_a = make_with_value<tnsr::i<DataVector, Dim>>(num_pts, 0.0);
  auto dt_field_b = make_with_value<tnsr::iJ<DataVector, Dim>>(num_pts, 0.0);
  auto dt_field_d = make_with_value<tnsr::ijj<DataVector, Dim>>(num_pts, 0.0);
  auto dt_field_p = make_with_value<tnsr::i<DataVector, Dim>>(num_pts, 0.0);
  auto dt_u_tm = make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto dt_bcm = make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto dt_bcf = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto dt_blapse = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto dt_bshift = make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto dt_btheta = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto dt_bz = make_with_value<tnsr::i<DataVector, Dim>>(num_pts, 0.0);

  const auto dt_result = bc.dg_time_derivative(
      make_not_null(&dt_cm), make_not_null(&dt_cf), make_not_null(&dt_a_tilde),
      make_not_null(&dt_K), make_not_null(&dt_theta),
      make_not_null(&dt_gamma_hat), make_not_null(&dt_lapse),
      make_not_null(&dt_shift), make_not_null(&dt_b),
      make_not_null(&dt_field_a), make_not_null(&dt_field_b),
      make_not_null(&dt_field_d), make_not_null(&dt_field_p),
      make_not_null(&dt_u_tm), make_not_null(&dt_bcm),
      make_not_null(&dt_bcf), make_not_null(&dt_blapse),
      make_not_null(&dt_bshift), make_not_null(&dt_btheta),
      make_not_null(&dt_bz), face_mesh_velocity, normal_covector,
      interior_conformal_metric, interior_conformal_factor, interior_a_tilde,
      interior_trace_K, interior_theta, interior_gamma_hat, interior_lapse,
      interior_shift, interior_auxiliary_shift_b, interior_field_a,
      interior_field_b, interior_field_d, interior_field_p,
      interior_boundary_u_tensor_minus,
      interior_boundary_conformal_metric, interior_boundary_conformal_factor,
      interior_boundary_lapse, interior_boundary_shift,
      interior_boundary_theta, interior_boundary_z,
      coords, time, evolve_lapse_and_shift);

  CHECK_FALSE(dt_result.has_value());

  // CRPBC zeros out all 17 evolved/auxiliary dt corrections and the 4
  // characteristic-mode dt corrections. The 4 boundary second-order field dt
  // corrections equal CCZ4 eq 12a applied to the ghost state (with K_0=0
  // per SO-CCZ4); on KerrSchild this is NOT zero, so we compare against an
  // independently computed reference below.
  const DataVector zero(num_pts, 0.0);
  for (auto& component : dt_cm) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  CHECK_ITERABLE_APPROX(get(dt_cf), zero);
  for (auto& component : dt_a_tilde) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  CHECK_ITERABLE_APPROX(get(dt_K), zero);
  CHECK_ITERABLE_APPROX(get(dt_theta), zero);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(dt_gamma_hat.get(i), zero);
  }
  CHECK_ITERABLE_APPROX(get(dt_lapse), zero);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(dt_shift.get(i), zero);
  }
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(dt_b.get(i), zero);
  }
  for (auto& component : dt_field_a) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_field_b) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_field_d) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_field_p) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  // The boundary characteristic-mode dt corrections are zero'd by CRPBC
  // (the actual time derivatives of these come from ComputeCrpbcBoundaryModeDt
  // in the LDG pipeline, not from dg_time_derivative).
  for (auto& component : dt_u_tm) {
    CHECK_ITERABLE_APPROX(component, zero);
  }

  // Verify dt_bcm against an independent reference computation of CCZ4 eq 12a.
  // When interior == analytic == boundary-integrated and UseAnalyticForAll=true,
  // the char-mixed state equals the analytic state, so we can compute the
  // expected dt_boundary_conformal_metric from the analytic fields directly.
  //
  // eq 12a: dt γ̃_{ij} = 2β^k D_{k,i,j} + γ̃_{ki}B_j^k + γ̃_{kj}B_i^k
  //                      - (2/3) γ̃_{ij} B_k^k - 2α Ã_{ij}
  //
  // This catches the tnsr::ii vs tnsr::ij bug: if conformal_metric_times_field_b
  // were stored as tnsr::ii (symmetric), the M_{ij} + M_{ji} sum would
  // incorrectly become 2*M_{ij}, losing the antisymmetric part.
  tnsr::ii<DataVector, Dim, Frame::Inertial> expected_dt_bcm(num_pts, 0.0);
  for (size_t s = 0; s < num_pts; ++s) {
    double cfb = 0.0;
    for (size_t k = 0; k < Dim; ++k) {
      cfb += interior_field_b.get(k, k)[s];
    }
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        double val = 0.0;
        for (size_t k = 0; k < Dim; ++k) {
          val += 2.0 * interior_shift.get(k)[s] *
                 interior_field_d.get(k, i, j)[s];
        }
        // M_{ij} = γ̃_{ki} B_j^k  (NOT symmetric in i,j)
        for (size_t k = 0; k < Dim; ++k) {
          val += interior_conformal_metric.get(k, i)[s] *
                 interior_field_b.get(j, k)[s];
        }
        // M_{ji} = γ̃_{kj} B_i^k
        for (size_t k = 0; k < Dim; ++k) {
          val += interior_conformal_metric.get(k, j)[s] *
                 interior_field_b.get(i, k)[s];
        }
        val -= (2.0 / 3.0) * interior_conformal_metric.get(i, j)[s] * cfb;
        val -= 2.0 * get(interior_lapse)[s] * interior_a_tilde.get(i, j)[s];
        expected_dt_bcm.get(i, j)[s] = val;
      }
    }
  }
  CHECK_ITERABLE_CUSTOM_APPROX(dt_bcm, expected_dt_bcm, custom_approx);
}

// Functional test on Minkowski: all fields trivial (lapse=1, metric=delta,
// K=0, theta=0, auxiliary fields=0, aux_shift_b=0), so CCZ4 eq 12a produces
// zero for the 4 boundary second-order dt corrections, and ghost == interior.
void test_minkowski() {
  register_factory_classes_with_charm<Metavariables>();

  const auto bc_ptr = TestHelpers::test_creation<
      std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>,
      Metavariables>(
      "ConstraintsRadiationPreserving:\n"
      "  AnalyticPrescription:\n"
      "    Ccz4(Minkowski):\n"
      "  UseAnalyticForAll: false\n"
      "  PenaltyMultiplier: 1.0\n"
      "  ZeroBoundaryThetaAndZ: false\n");
  const auto& bc = dynamic_cast<
      const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving&>(*bc_ptr);

  static constexpr size_t Dim = 3;
  const size_t num_pts = 5;

  auto make_scalar = [&](const double val) {
    return Scalar<DataVector>{DataVector(num_pts, val)};
  };
  auto make_vector = [&](const double val) {
    return tnsr::I<DataVector, Dim, Frame::Inertial>(num_pts, val);
  };
  auto make_covector = [&](const double val) {
    return tnsr::i<DataVector, Dim, Frame::Inertial>(num_pts, val);
  };

  tnsr::ii<DataVector, Dim, Frame::Inertial> interior_conformal_metric(num_pts,
                                                                       0.0);
  for (size_t i = 0; i < Dim; ++i) {
    interior_conformal_metric.get(i, i) = 1.0;
  }
  const auto interior_conformal_factor = make_scalar(1.0);
  tnsr::ii<DataVector, Dim, Frame::Inertial> interior_a_tilde(num_pts, 0.0);
  const auto interior_trace_K = make_scalar(0.0);
  const auto interior_theta = make_scalar(0.0);
  const auto interior_gamma_hat = make_vector(0.0);
  const auto interior_lapse = make_scalar(1.0);
  const auto interior_shift = make_vector(0.0);
  const auto interior_auxiliary_shift_b = make_vector(0.0);
  const auto interior_field_a = make_covector(0.0);
  tnsr::iJ<DataVector, Dim, Frame::Inertial> interior_field_b(num_pts, 0.0);
  tnsr::ijj<DataVector, Dim, Frame::Inertial> interior_field_d(num_pts, 0.0);
  const auto interior_field_p = make_covector(0.0);

  tnsr::ii<DataVector, Dim, Frame::Inertial> interior_boundary_u_tensor_minus(
      num_pts, 0.0);

  auto interior_boundary_conformal_metric = interior_conformal_metric;
  const auto interior_boundary_conformal_factor = make_scalar(1.0);
  const auto interior_boundary_lapse = make_scalar(1.0);
  const auto interior_boundary_shift = make_vector(0.0);
  const auto interior_boundary_theta = make_scalar(0.0);
  const auto interior_boundary_z = make_covector(0.0);

  tnsr::I<DataVector, Dim, Frame::Inertial> coords(num_pts, 0.0);
  for (size_t i = 0; i < num_pts; ++i) {
    coords.get(0)[i] = 5.0 + 0.1 * static_cast<double>(i);
    coords.get(1)[i] = 0.1 * static_cast<double>(i);
    coords.get(2)[i] = -0.1 * static_cast<double>(i);
  }

  auto normal_covector = make_covector(0.0);
  normal_covector.get(0) = 1.0;

  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
      face_mesh_velocity{};
  const double time = 0.0;
  const bool evolve_lapse_and_shift = true;

  // ---- dg_ghost ----
  tnsr::ii<DataVector, Dim, Frame::Inertial> ghost_conformal_metric(num_pts,
                                                                    0.0);
  auto ghost_conformal_factor = make_scalar(0.0);
  tnsr::ii<DataVector, Dim, Frame::Inertial> ghost_a_tilde(num_pts, 0.0);
  auto ghost_trace_K = make_scalar(0.0);
  auto ghost_theta = make_scalar(0.0);
  auto ghost_gamma_hat = make_vector(0.0);
  auto ghost_lapse = make_scalar(0.0);
  auto ghost_shift = make_vector(0.0);
  auto ghost_auxiliary_shift_b = make_vector(0.0);
  auto ghost_field_a = make_covector(0.0);
  tnsr::iJ<DataVector, Dim, Frame::Inertial> ghost_field_b(num_pts, 0.0);
  tnsr::ijj<DataVector, Dim, Frame::Inertial> ghost_field_d(num_pts, 0.0);
  auto ghost_field_p = make_covector(0.0);
  tnsr::ii<DataVector, Dim, Frame::Inertial> ghost_u_tensor_minus(num_pts, 0.0);
  tnsr::ii<DataVector, Dim, Frame::Inertial> ghost_boundary_conformal_metric(
      num_pts, 0.0);
  auto ghost_boundary_conformal_factor = make_scalar(0.0);
  auto ghost_boundary_lapse = make_scalar(0.0);
  auto ghost_boundary_shift = make_vector(0.0);
  auto ghost_boundary_theta = make_scalar(0.0);
  auto ghost_boundary_z = make_covector(0.0);

  const auto ghost_result = bc.dg_ghost(
      make_not_null(&ghost_conformal_metric),
      make_not_null(&ghost_conformal_factor), make_not_null(&ghost_a_tilde),
      make_not_null(&ghost_trace_K), make_not_null(&ghost_theta),
      make_not_null(&ghost_gamma_hat), make_not_null(&ghost_lapse),
      make_not_null(&ghost_shift), make_not_null(&ghost_auxiliary_shift_b),
      make_not_null(&ghost_field_a), make_not_null(&ghost_field_b),
      make_not_null(&ghost_field_d), make_not_null(&ghost_field_p),
      make_not_null(&ghost_u_tensor_minus),
      make_not_null(&ghost_boundary_conformal_metric),
      make_not_null(&ghost_boundary_conformal_factor),
      make_not_null(&ghost_boundary_lapse),
      make_not_null(&ghost_boundary_shift),
      make_not_null(&ghost_boundary_theta),
      make_not_null(&ghost_boundary_z), face_mesh_velocity, normal_covector,
      interior_conformal_metric, interior_conformal_factor, interior_a_tilde,
      interior_trace_K, interior_theta, interior_gamma_hat, interior_lapse,
      interior_shift, interior_auxiliary_shift_b, interior_field_a,
      interior_field_b, interior_field_d, interior_field_p,
      interior_boundary_u_tensor_minus,
      interior_boundary_conformal_metric, interior_boundary_conformal_factor,
      interior_boundary_lapse, interior_boundary_shift,
      interior_boundary_theta, interior_boundary_z,
      coords, time, evolve_lapse_and_shift);

  CHECK_FALSE(ghost_result.has_value());

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      const double expected = (i == j) ? 1.0 : 0.0;
      CHECK_ITERABLE_APPROX(ghost_conformal_metric.get(i, j),
                            DataVector(num_pts, expected));
    }
  }
  CHECK_ITERABLE_APPROX(get(ghost_conformal_factor), DataVector(num_pts, 1.0));
  CHECK_ITERABLE_APPROX(get(ghost_lapse), DataVector(num_pts, 1.0));
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(ghost_shift.get(i), DataVector(num_pts, 0.0));
  }
  for (auto& component : ghost_a_tilde) {
    CHECK_ITERABLE_APPROX(component, DataVector(num_pts, 0.0));
  }
  CHECK_ITERABLE_APPROX(get(ghost_trace_K), DataVector(num_pts, 0.0));
  CHECK_ITERABLE_APPROX(get(ghost_theta), DataVector(num_pts, 0.0));

  // ---- dg_time_derivative ----
  tnsr::ii<DataVector, Dim, Frame::Inertial> dt_cm(num_pts, 0.0);
  auto dt_cf = make_scalar(0.0);
  tnsr::ii<DataVector, Dim, Frame::Inertial> dt_a_tilde(num_pts, 0.0);
  auto dt_K = make_scalar(0.0);
  auto dt_theta = make_scalar(0.0);
  auto dt_gamma_hat = make_vector(0.0);
  auto dt_lapse = make_scalar(0.0);
  auto dt_shift = make_vector(0.0);
  auto dt_b = make_vector(0.0);
  auto dt_field_a = make_covector(0.0);
  tnsr::iJ<DataVector, Dim, Frame::Inertial> dt_field_b(num_pts, 0.0);
  tnsr::ijj<DataVector, Dim, Frame::Inertial> dt_field_d(num_pts, 0.0);
  auto dt_field_p = make_covector(0.0);
  tnsr::ii<DataVector, Dim, Frame::Inertial> dt_u_tm(num_pts, 0.0);
  tnsr::ii<DataVector, Dim, Frame::Inertial> dt_bcm(num_pts, 0.0);
  auto dt_bcf = make_scalar(0.0);
  auto dt_blapse = make_scalar(0.0);
  auto dt_bshift = make_vector(0.0);
  auto dt_btheta = make_scalar(0.0);
  auto dt_bz = make_covector(0.0);

  const auto dt_result = bc.dg_time_derivative(
      make_not_null(&dt_cm), make_not_null(&dt_cf), make_not_null(&dt_a_tilde),
      make_not_null(&dt_K), make_not_null(&dt_theta),
      make_not_null(&dt_gamma_hat), make_not_null(&dt_lapse),
      make_not_null(&dt_shift), make_not_null(&dt_b),
      make_not_null(&dt_field_a), make_not_null(&dt_field_b),
      make_not_null(&dt_field_d), make_not_null(&dt_field_p),
      make_not_null(&dt_u_tm), make_not_null(&dt_bcm),
      make_not_null(&dt_bcf), make_not_null(&dt_blapse),
      make_not_null(&dt_bshift), make_not_null(&dt_btheta),
      make_not_null(&dt_bz), face_mesh_velocity, normal_covector,
      interior_conformal_metric, interior_conformal_factor, interior_a_tilde,
      interior_trace_K, interior_theta, interior_gamma_hat, interior_lapse,
      interior_shift, interior_auxiliary_shift_b, interior_field_a,
      interior_field_b, interior_field_d, interior_field_p,
      interior_boundary_u_tensor_minus,
      interior_boundary_conformal_metric, interior_boundary_conformal_factor,
      interior_boundary_lapse, interior_boundary_shift,
      interior_boundary_theta, interior_boundary_z,
      coords, time, evolve_lapse_and_shift);

  CHECK_FALSE(dt_result.has_value());

  // On Minkowski every input is trivial, so CCZ4 eq 12a produces zero for
  // the 4 boundary second-order dt corrections; the other 17 + 4 are zero
  // by construction in CRPBC.
  const DataVector zero(num_pts, 0.0);
  for (auto& component : dt_cm) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  CHECK_ITERABLE_APPROX(get(dt_cf), zero);
  for (auto& component : dt_a_tilde) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  CHECK_ITERABLE_APPROX(get(dt_K), zero);
  CHECK_ITERABLE_APPROX(get(dt_theta), zero);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(dt_gamma_hat.get(i), zero);
  }
  CHECK_ITERABLE_APPROX(get(dt_lapse), zero);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(dt_shift.get(i), zero);
  }
  for (auto& component : dt_bcm) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  CHECK_ITERABLE_APPROX(get(dt_bcf), zero);
  CHECK_ITERABLE_APPROX(get(dt_blapse), zero);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(dt_bshift.get(i), zero);
  }
}

// Roundtrip test that specifically exercises the T^perp_i reconstruction
// path in crpbc_characteristic_pipeline (UseAnalyticForAll=false).
//
// Rationale: the T^perp_i term is sensitive to conformal vs physical metric
// (paper: T^perp_i = q_{ij} T^j with physical q_{ij} = gamma_{ij} - n_i n_j;
// the code lowers with conformal metric and divides by phi^2). A past bug
// omitted the 1/phi^2 and would only manifest when phi != 1 and the
// transverse derivative of gamma-tilde is non-zero. KerrSchild has
// phi != 1 AND non-trivial d_gamma-tilde, which hits both conditions.
//
// On an exact constraint-satisfying analytic solution (KerrSchild in
// CCZ4, Theta = 0, Z_i = 0, and U^- consistent with U^+ and T^perp via
// the paper's inverse relations), setting interior == boundary-integrated
// == analytic and interior_boundary_u_tensor_minus to the analytic U^-
// should make the reconstructed ghost state reproduce the analytic
// state. Under the bug, ghost_a_tilde in particular would pick up a
// phi^2-scaled error in its (perp,n) part because UVector2Minus_rec
// enters a_tilde_perp_n in evolved_space_from_characteristic_fields.
void test_kerrschild_roundtrip_t_perp() {
  register_factory_classes_with_charm<Metavariables>();

  const auto bc_ptr = TestHelpers::test_creation<
      std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>,
      Metavariables>(
      "ConstraintsRadiationPreserving:\n"
      "  AnalyticPrescription:\n"
      "    Ccz4(KerrSchild):\n"
      "      Mass: 2.0\n"
      "      Spin: [0.2, 0.4, 0.8]\n"
      "      Center: [0.2, 0.5, 0.1]\n"
      "      Velocity: [0.0, 0.0, 0.0]\n"
      "  UseAnalyticForAll: false\n"
      "  PenaltyMultiplier: 1.0\n"
      "  ZeroBoundaryThetaAndZ: false\n");
  const auto& bc = dynamic_cast<
      const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving&>(*bc_ptr);

  static constexpr size_t Dim = 3;
  const size_t num_pts = 5;

  // Face coordinates well outside the horizon (r ~ 5 >> 2M = 4)
  tnsr::I<DataVector, Dim, Frame::Inertial> coords(num_pts, 0.0);
  for (size_t i = 0; i < num_pts; ++i) {
    coords.get(0)[i] = 5.0 + 0.1 * static_cast<double>(i);
    coords.get(1)[i] = 0.1 * static_cast<double>(i);
    coords.get(2)[i] = -0.1 * static_cast<double>(i);
  }

  const double time = 0.0;
  const gr::Solutions::KerrSchild kerr_schild(
      2.0, std::array<double, 3>{{0.2, 0.4, 0.8}},
      std::array<double, 3>{{0.2, 0.5, 0.1}});
  const Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::KerrSchild>
      wrapped_solution(kerr_schild);

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
  const auto analytic_values =
      wrapped_solution.variables(coords, time, all_tags{});

  const auto& interior_conformal_metric =
      get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(analytic_values);
  const auto& interior_conformal_factor =
      get<Ccz4::Tags::ConformalFactor<DataVector>>(analytic_values);
  const auto& interior_a_tilde =
      get<Ccz4::Tags::ATilde<DataVector, 3>>(analytic_values);
  const auto& interior_trace_K =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(analytic_values);
  const auto& interior_theta =
      get<Ccz4::Tags::Theta<DataVector>>(analytic_values);
  const auto& interior_gamma_hat =
      get<Ccz4::Tags::GammaHat<DataVector, 3>>(analytic_values);
  const auto& interior_lapse =
      get<gr::Tags::Lapse<DataVector>>(analytic_values);
  const auto& interior_shift =
      get<gr::Tags::Shift<DataVector, 3>>(analytic_values);
  const auto& interior_auxiliary_shift_b =
      get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(analytic_values);
  const auto& interior_field_a =
      get<Ccz4::Tags::FieldA<DataVector, 3>>(analytic_values);
  const auto& interior_field_b =
      get<Ccz4::Tags::FieldB<DataVector, 3>>(analytic_values);
  const auto& interior_field_d =
      get<Ccz4::Tags::FieldD<DataVector, 3>>(analytic_values);
  const auto& interior_field_p =
      get<Ccz4::Tags::FieldP<DataVector, 3>>(analytic_values);

  auto normal_covector =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(num_pts, 0.0);
  normal_covector.get(0) = 1.0;

  // Mirror CRPBC's ghost-side unit normal construction so we can call
  // characteristic_fields() with identical inputs.
  const auto [det_cm, inv_cm] =
      determinant_and_inverse(interior_conformal_metric);
  (void)det_cm;
  tnsr::II<DataVector, Dim, Frame::Inertial> inv_spatial_metric{};
  ::tenex::evaluate<ti::I, ti::J>(
      make_not_null(&inv_spatial_metric),
      interior_conformal_factor() * interior_conformal_factor() *
          inv_cm(ti::I, ti::J));
  const Scalar<DataVector> mag_sq =
      dot_product(normal_covector, normal_covector, inv_spatial_metric);
  const DataVector inv_mag = 1.0 / sqrt(get(mag_sq));
  tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal_one_form(num_pts);
  for (size_t i = 0; i < Dim; ++i) {
    unit_normal_one_form.get(i) = normal_covector.get(i) * inv_mag;
  }

  // First-order derivatives from the analytic auxiliary fields, matching
  // CRPBC's internal reconstruction (see ConstraintsRadiationPreserving.cpp).
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

  // Analytic characteristic fields → extract U^-_{ij} (UTensorMinus).
  static constexpr double f = Ccz4::fd::System::f;
  const auto analytic_char_fields = ::Ccz4::fd::characteristic_fields(
      unit_normal_one_form, interior_conformal_metric,
      interior_conformal_factor, interior_lapse, interior_shift,
      interior_trace_K, interior_a_tilde, interior_theta, interior_gamma_hat,
      interior_auxiliary_shift_b, d_conformal_metric, d_conformal_factor,
      d_lapse, d_shift, f);
  const auto& analytic_u_tensor_minus = get<::Ccz4::fd::Tags::UTensorMinus<
      DataVector, Dim, Frame::Inertial>>(analytic_char_fields);

  // Constraint-satisfying boundary-integrated state:
  // - coeff_four_fields = analytic (so ghost normal == interior normal),
  // - U^-_{ij} = analytic value,
  // - Theta = 0, Z_i = 0 exactly.
  const auto interior_boundary_u_tensor_minus = analytic_u_tensor_minus;
  const auto& interior_boundary_conformal_metric = interior_conformal_metric;
  const auto& interior_boundary_conformal_factor = interior_conformal_factor;
  const auto& interior_boundary_lapse = interior_lapse;
  const auto& interior_boundary_shift = interior_shift;
  const auto interior_boundary_theta =
      make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  const auto interior_boundary_z =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(num_pts, 0.0);

  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
      face_mesh_velocity{};
  const bool evolve_lapse_and_shift = true;

  // ---- dg_ghost ----
  auto ghost_conformal_metric =
      make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_conformal_factor =
      make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_a_tilde = make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_trace_K = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_theta = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_gamma_hat =
      make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_lapse = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_shift = make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_auxiliary_shift_b =
      make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_field_a = make_with_value<tnsr::i<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_field_b = make_with_value<tnsr::iJ<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_field_d =
      make_with_value<tnsr::ijj<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_field_p = make_with_value<tnsr::i<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_u_tensor_minus =
      make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_boundary_lapse = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_boundary_shift =
      make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_boundary_theta =
      make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_boundary_z =
      make_with_value<tnsr::i<DataVector, Dim>>(num_pts, 0.0);

  const auto ghost_result = bc.dg_ghost(
      make_not_null(&ghost_conformal_metric),
      make_not_null(&ghost_conformal_factor), make_not_null(&ghost_a_tilde),
      make_not_null(&ghost_trace_K), make_not_null(&ghost_theta),
      make_not_null(&ghost_gamma_hat), make_not_null(&ghost_lapse),
      make_not_null(&ghost_shift), make_not_null(&ghost_auxiliary_shift_b),
      make_not_null(&ghost_field_a), make_not_null(&ghost_field_b),
      make_not_null(&ghost_field_d), make_not_null(&ghost_field_p),
      make_not_null(&ghost_u_tensor_minus),
      make_not_null(&ghost_boundary_conformal_metric),
      make_not_null(&ghost_boundary_conformal_factor),
      make_not_null(&ghost_boundary_lapse),
      make_not_null(&ghost_boundary_shift),
      make_not_null(&ghost_boundary_theta),
      make_not_null(&ghost_boundary_z), face_mesh_velocity, normal_covector,
      interior_conformal_metric, interior_conformal_factor, interior_a_tilde,
      interior_trace_K, interior_theta, interior_gamma_hat, interior_lapse,
      interior_shift, interior_auxiliary_shift_b, interior_field_a,
      interior_field_b, interior_field_d, interior_field_p,
      interior_boundary_u_tensor_minus,
      interior_boundary_conformal_metric, interior_boundary_conformal_factor,
      interior_boundary_lapse, interior_boundary_shift,
      interior_boundary_theta, interior_boundary_z,
      coords, time, evolve_lapse_and_shift);

  CHECK_FALSE(ghost_result.has_value());

  // Fields carried through unchanged (not reconstructed via char modes in
  // the T^perp_i path) — sanity check that the test setup is consistent.
  const Approx custom_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_conformal_metric,
                               interior_conformal_metric, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_conformal_factor,
                               interior_conformal_factor, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_lapse, interior_lapse, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_shift, interior_shift, custom_approx);

  // Primary assertion: the reconstructed ghost a_tilde matches the analytic
  // value. a_tilde_perp_n carries the T^perp_i term through the inverse
  // characteristic transform (see evolved_space_from_characteristic_fields
  // in Characteristics.cpp, a_tilde_perp_n = -phi^4/4 * (U^+ - U^-)),
  // and U^-_vec2 reconstruction contains 2*T^perp_i. Under the phi^2
  // bug this component would be off by a factor of phi^2.
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_a_tilde, interior_a_tilde, custom_approx);
}

// Test that the CRPBC pipeline roundtrips boundary theta and Z_i:
// after prescribing non-zero boundary_theta and boundary_z, constructing
// incoming characteristic modes, mixing with outgoing ones, and inverse-
// transforming, the resulting ghost state should reproduce the prescribed
// theta and Z_i (computed from the ghost gamma_hat and Christoffel symbols).
void test_theta_and_z_roundtrip() {
  register_factory_classes_with_charm<Metavariables>();

  const auto bc_ptr = TestHelpers::test_creation<
      std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>,
      Metavariables>(
      "ConstraintsRadiationPreserving:\n"
      "  AnalyticPrescription:\n"
      "    Ccz4(KerrSchild):\n"
      "      Mass: 2.0\n"
      "      Spin: [0.2, 0.4, 0.8]\n"
      "      Center: [0.2, 0.5, 0.1]\n"
      "      Velocity: [0.0, 0.0, 0.0]\n"
      "  UseAnalyticForAll: false\n"
      "  PenaltyMultiplier: 1.0\n"
      "  ZeroBoundaryThetaAndZ: false\n");
  const auto& bc = dynamic_cast<
      const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving&>(*bc_ptr);

  static constexpr size_t Dim = 3;
  const size_t num_pts = 5;

  // Face coordinates well outside the horizon (r ~ 5 >> 2M = 4)
  tnsr::I<DataVector, Dim, Frame::Inertial> coords(num_pts, 0.0);
  for (size_t i = 0; i < num_pts; ++i) {
    coords.get(0)[i] = 5.0 + 0.1 * static_cast<double>(i);
    coords.get(1)[i] = 0.1 * static_cast<double>(i);
    coords.get(2)[i] = -0.1 * static_cast<double>(i);
  }

  const double time = 0.0;
  const gr::Solutions::KerrSchild kerr_schild(
      2.0, std::array<double, 3>{{0.2, 0.4, 0.8}},
      std::array<double, 3>{{0.2, 0.5, 0.1}});
  const Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::KerrSchild>
      wrapped_solution(kerr_schild);

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
  const auto analytic_values =
      wrapped_solution.variables(coords, time, all_tags{});

  const auto& interior_conformal_metric =
      get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(analytic_values);
  const auto& interior_conformal_factor =
      get<Ccz4::Tags::ConformalFactor<DataVector>>(analytic_values);
  const auto& interior_a_tilde =
      get<Ccz4::Tags::ATilde<DataVector, 3>>(analytic_values);
  const auto& interior_trace_K =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(analytic_values);
  const auto& interior_theta =
      get<Ccz4::Tags::Theta<DataVector>>(analytic_values);
  const auto& interior_gamma_hat =
      get<Ccz4::Tags::GammaHat<DataVector, 3>>(analytic_values);
  const auto& interior_lapse =
      get<gr::Tags::Lapse<DataVector>>(analytic_values);
  const auto& interior_shift =
      get<gr::Tags::Shift<DataVector, 3>>(analytic_values);
  const auto& interior_auxiliary_shift_b =
      get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(analytic_values);
  const auto& interior_field_a =
      get<Ccz4::Tags::FieldA<DataVector, 3>>(analytic_values);
  const auto& interior_field_b =
      get<Ccz4::Tags::FieldB<DataVector, 3>>(analytic_values);
  const auto& interior_field_d =
      get<Ccz4::Tags::FieldD<DataVector, 3>>(analytic_values);
  const auto& interior_field_p =
      get<Ccz4::Tags::FieldP<DataVector, 3>>(analytic_values);

  // Non-axis-aligned normal covector for generality
  auto normal_covector =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(num_pts, 0.0);
  normal_covector.get(0) = 0.9;
  normal_covector.get(1) = 0.3;
  normal_covector.get(2) = -0.2;

  // Boundary-integrated (coeff) four fields: perturbed from interior by 1-2%
  // to exercise the case where they differ.  The CRPBC pipeline uses coeff
  // fields consistently for both the forward and inverse char transforms, so
  // theta and Z_i must still roundtrip exactly.
  auto interior_boundary_conformal_metric = interior_conformal_metric;
  for (size_t i = 0; i < interior_boundary_conformal_metric.size(); ++i) {
    interior_boundary_conformal_metric[i] *= 1.0 + 0.01 * (1.0 + 0.3 * static_cast<double>(i));
  }
  auto interior_boundary_conformal_factor = interior_conformal_factor;
  get(interior_boundary_conformal_factor) *= 1.02;
  auto interior_boundary_lapse = interior_lapse;
  get(interior_boundary_lapse) *= 0.98;
  auto interior_boundary_shift = interior_shift;
  for (size_t i = 0; i < Dim; ++i) {
    interior_boundary_shift.get(i) *= 1.0 + 0.015 * static_cast<double>(i + 1);
  }

  // Non-zero boundary theta and Z_i (small perturbations from the constraint-
  // satisfying values of zero)
  Scalar<DataVector> interior_boundary_theta(num_pts);
  for (size_t s = 0; s < num_pts; ++s) {
    get(interior_boundary_theta)[s] =
        0.01 * (1.0 + 0.1 * static_cast<double>(s));
  }

  tnsr::i<DataVector, Dim, Frame::Inertial> interior_boundary_z(num_pts);
  for (size_t s = 0; s < num_pts; ++s) {
    interior_boundary_z.get(0)[s] =
        0.005 * (1.0 + 0.2 * static_cast<double>(s));
    interior_boundary_z.get(1)[s] =
        -0.003 * (1.0 - 0.1 * static_cast<double>(s));
    interior_boundary_z.get(2)[s] =
        0.002 * (0.5 + 0.3 * static_cast<double>(s));
  }

  // U^- tensor minus: not used in UseAnalyticForAll=false path (overridden by
  // ghost analytic char fields), so set to zero.
  const auto interior_boundary_u_tensor_minus =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(num_pts, 0.0);

  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
      face_mesh_velocity{};
  const bool evolve_lapse_and_shift = true;

  // ---- dg_ghost ----
  auto ghost_conformal_metric =
      make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_conformal_factor =
      make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_a_tilde = make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_trace_K = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_theta = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_gamma_hat =
      make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_lapse = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_shift = make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_auxiliary_shift_b =
      make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_field_a = make_with_value<tnsr::i<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_field_b = make_with_value<tnsr::iJ<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_field_d =
      make_with_value<tnsr::ijj<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_field_p = make_with_value<tnsr::i<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_u_tensor_minus =
      make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_boundary_lapse = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_boundary_shift =
      make_with_value<tnsr::I<DataVector, Dim>>(num_pts, 0.0);
  auto ghost_boundary_theta =
      make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  auto ghost_boundary_z =
      make_with_value<tnsr::i<DataVector, Dim>>(num_pts, 0.0);

  const auto ghost_result = bc.dg_ghost(
      make_not_null(&ghost_conformal_metric),
      make_not_null(&ghost_conformal_factor), make_not_null(&ghost_a_tilde),
      make_not_null(&ghost_trace_K), make_not_null(&ghost_theta),
      make_not_null(&ghost_gamma_hat), make_not_null(&ghost_lapse),
      make_not_null(&ghost_shift), make_not_null(&ghost_auxiliary_shift_b),
      make_not_null(&ghost_field_a), make_not_null(&ghost_field_b),
      make_not_null(&ghost_field_d), make_not_null(&ghost_field_p),
      make_not_null(&ghost_u_tensor_minus),
      make_not_null(&ghost_boundary_conformal_metric),
      make_not_null(&ghost_boundary_conformal_factor),
      make_not_null(&ghost_boundary_lapse),
      make_not_null(&ghost_boundary_shift),
      make_not_null(&ghost_boundary_theta),
      make_not_null(&ghost_boundary_z), face_mesh_velocity, normal_covector,
      interior_conformal_metric, interior_conformal_factor, interior_a_tilde,
      interior_trace_K, interior_theta, interior_gamma_hat, interior_lapse,
      interior_shift, interior_auxiliary_shift_b, interior_field_a,
      interior_field_b, interior_field_d, interior_field_p,
      interior_boundary_u_tensor_minus,
      interior_boundary_conformal_metric, interior_boundary_conformal_factor,
      interior_boundary_lapse, interior_boundary_shift,
      interior_boundary_theta, interior_boundary_z,
      coords, time, evolve_lapse_and_shift);

  CHECK_FALSE(ghost_result.has_value());

  const Approx custom_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);

  // Check 1: ghost theta should match the prescribed boundary theta.
  // By the inverse char transform: theta = -phi^2/4 * (U3+ - U3-), and the
  // CRPBC sets U3-_rec = U3+ + 4*theta_bdry/phi^2, giving theta = theta_bdry.
  CHECK_ITERABLE_CUSTOM_APPROX(ghost_theta, interior_boundary_theta,
                               custom_approx);

  // Check 2: Compute Z_i from the ghost state and verify it matches the
  // prescribed boundary Z_i.
  //   Z_i = (1/2) gamma_tilde_{ij} (gamma_hat^j - Gamma_tilde^j)
  // where Gamma_tilde^j is the contracted conformal Christoffel computed from
  // the ghost conformal metric and ghost field_d.
  const auto [det_ghost_cm, inv_ghost_cm] =
      determinant_and_inverse(ghost_conformal_metric);
  (void)det_ghost_cm;

  const auto conformal_christoffel =
      Ccz4::conformal_christoffel_second_kind(inv_ghost_cm, ghost_field_d);
  const auto contracted_christoffel =
      Ccz4::contracted_conformal_christoffel_second_kind(inv_ghost_cm,
                                                         conformal_christoffel);

  tnsr::I<DataVector, Dim, Frame::Inertial> gamma_hat_minus_christoffel{};
  ::tenex::evaluate<ti::I>(make_not_null(&gamma_hat_minus_christoffel),
                           ghost_gamma_hat(ti::I) -
                               contracted_christoffel(ti::I));

  const auto computed_z = Ccz4::spatial_z4_constraint(
      ghost_conformal_metric, gamma_hat_minus_christoffel);

  CHECK_ITERABLE_CUSTOM_APPROX(computed_z, interior_boundary_z, custom_approx);
}

SPECTRE_TEST_CASE("Unit.Ccz4.BoundaryConditions.ConstraintsRadiationPreserving",
                  "[Unit][Evolution]") {
  test_creation_and_serialization();
  test_bc_type();
  test_minkowski();
  test_kerrschild_use_analytic_for_all();
  test_kerrschild_roundtrip_t_perp();
  test_theta_and_z_roundtrip();
}
}  // namespace
