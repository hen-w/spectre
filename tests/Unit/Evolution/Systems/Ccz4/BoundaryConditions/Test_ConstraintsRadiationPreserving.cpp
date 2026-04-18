// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/ConstraintsRadiationPreserving.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Ccz4/Ccz4WrappedGr.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
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
            tmpl::list<
                Ccz4::BoundaryConditions::ConstraintsRadiationPreserving>>,
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
      "  PrescribeOutgoing: false\n"
      "  InitialTime: 0.0\n");

  CHECK(boundary_condition->get_clone() != nullptr);

  const auto serialized = serialize_and_deserialize(
      *dynamic_cast<
          Ccz4::BoundaryConditions::ConstraintsRadiationPreserving*>(
          boundary_condition.get()));
  CHECK(serialized.get_clone() != nullptr);
}

void test_bc_type() {
  CHECK(
      Ccz4::BoundaryConditions::ConstraintsRadiationPreserving::bc_type ==
      evolution::BoundaryConditions::Type::GhostAndTimeDerivative);
}

// Functional test on Minkowski: all fields are trivial, so the ghost state
// should match the interior, and all dt corrections should be zero.
void test_minkowski() {
  register_factory_classes_with_charm<Metavariables>();

  // Create BC with Minkowski prescription
  const auto bc_ptr = TestHelpers::test_creation<
      std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>,
      Metavariables>(
      "ConstraintsRadiationPreserving:\n"
      "  AnalyticPrescription:\n"
      "    Ccz4(Minkowski):\n"
      "  PrescribeOutgoing: false\n"
      "  InitialTime: 0.0\n");
  const auto& bc = dynamic_cast<
      const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving&>(
      *bc_ptr);

  static constexpr size_t Dim = 3;
  const size_t num_pts = 5;

  // Construct Minkowski interior data
  auto make_scalar = [&](const double val) {
    return Scalar<DataVector>{DataVector(num_pts, val)};
  };
  auto make_vector = [&](const double val) {
    auto v = tnsr::I<DataVector, Dim, Frame::Inertial>(num_pts, val);
    return v;
  };
  auto make_covector = [&](const double val) {
    auto v = tnsr::i<DataVector, Dim, Frame::Inertial>(num_pts, val);
    return v;
  };

  // conformal_metric = delta_ij
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

  // Boundary mode interior values = 0
  tnsr::ii<DataVector, Dim, Frame::Inertial> interior_u_tensor_minus(num_pts,
                                                                     0.0);

  // Boundary second-order field interior values = Minkowski
  auto interior_boundary_conformal_metric = interior_conformal_metric;
  const auto interior_boundary_conformal_factor = make_scalar(1.0);
  const auto interior_boundary_lapse = make_scalar(1.0);
  const auto interior_boundary_shift = make_vector(0.0);
  const auto interior_boundary_theta = make_scalar(0.0);
  const auto interior_boundary_z = make_covector(0.0);

  // Coordinates (arbitrary, Minkowski is spatially homogeneous)
  tnsr::I<DataVector, Dim, Frame::Inertial> coords(num_pts, 0.0);
  for (size_t i = 0; i < num_pts; ++i) {
    coords.get(0)[i] = 5.0 + 0.1 * i;
    coords.get(1)[i] = 0.1 * i;
    coords.get(2)[i] = -0.1 * i;
  }

  // Normal covector (pointing in x-direction, unnormalized for DG)
  auto normal_covector = make_covector(0.0);
  normal_covector.get(0) = 1.0;

  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
      face_mesh_velocity{};
  const double time = 0.0;
  const bool evolve_lapse_and_shift = true;

  // ---- Test dg_ghost ----
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
      interior_u_tensor_minus,
      interior_boundary_conformal_metric, interior_boundary_conformal_factor,
      interior_boundary_lapse, interior_boundary_shift,
      interior_boundary_theta, interior_boundary_z, coords, time,
      evolve_lapse_and_shift);

  CHECK_FALSE(ghost_result.has_value());

  // On Minkowski, ghost conformal metric should be delta_ij
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
  // a_tilde, K, theta should all be zero
  for (auto& component : ghost_a_tilde) {
    CHECK_ITERABLE_APPROX(component, DataVector(num_pts, 0.0));
  }
  CHECK_ITERABLE_APPROX(get(ghost_trace_K), DataVector(num_pts, 0.0));
  CHECK_ITERABLE_APPROX(get(ghost_theta), DataVector(num_pts, 0.0));

  // ---- Test dg_time_derivative ----
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
      make_not_null(&dt_bshift),
      make_not_null(&dt_btheta), make_not_null(&dt_bz),
      face_mesh_velocity, normal_covector,
      interior_conformal_metric, interior_conformal_factor, interior_a_tilde,
      interior_trace_K, interior_theta, interior_gamma_hat, interior_lapse,
      interior_shift, interior_auxiliary_shift_b, interior_field_a,
      interior_field_b, interior_field_d, interior_field_p,
      interior_u_tensor_minus,
      interior_boundary_conformal_metric, interior_boundary_conformal_factor,
      interior_boundary_lapse, interior_boundary_shift,
      interior_boundary_theta, interior_boundary_z, coords, time,
      evolve_lapse_and_shift);

  CHECK_FALSE(dt_result.has_value());

  // On Minkowski, all dt corrections should be zero (stationary spacetime).
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
  // Boundary field dt corrections
  for (auto& component : dt_bcm) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  CHECK_ITERABLE_APPROX(get(dt_bcf), zero);
  CHECK_ITERABLE_APPROX(get(dt_blapse), zero);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(dt_bshift.get(i), zero);
  }
}

// Functional test on KerrSchild: nontrivial spacetime exercises the full
// characteristic decomposition + mode mixing code path. With interior data ==
// analytic, the ghost state should reproduce the interior and all dt
// corrections = 0.
void test_kerrschild() {
  register_factory_classes_with_charm<Metavariables>();

  // Create BC with KerrSchild prescription
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
      "  PrescribeOutgoing: false\n"
      "  InitialTime: 0.0\n");
  const auto& bc = dynamic_cast<
      const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving&>(
      *bc_ptr);

  static constexpr size_t Dim = 3;
  const size_t num_pts = 5;

  // Face coordinates well outside the horizon (r ~ 5 >> 2M = 4)
  tnsr::I<DataVector, Dim, Frame::Inertial> coords(num_pts, 0.0);
  for (size_t i = 0; i < num_pts; ++i) {
    coords.get(0)[i] = 5.0 + 0.1 * static_cast<double>(i);
    coords.get(1)[i] = 0.1 * static_cast<double>(i);
    coords.get(2)[i] = -0.1 * static_cast<double>(i);
  }

  // Evaluate KerrSchild solution via Ccz4WrappedGr to get interior data
  // that is EXACTLY what the BC evaluates internally.
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

  // Extract interior data
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

  // Boundary mode interior values = 0 (constraints satisfied)
  auto interior_u_tensor_minus =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(num_pts, 0.0);

  // Boundary second-order fields = interior (consistent initial data)
  const auto& interior_boundary_conformal_metric = interior_conformal_metric;
  const auto& interior_boundary_conformal_factor = interior_conformal_factor;
  const auto& interior_boundary_lapse = interior_lapse;
  const auto& interior_boundary_shift = interior_shift;
  const auto interior_boundary_theta =
      make_with_value<Scalar<DataVector>>(num_pts, 0.0);
  const auto interior_boundary_z =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(num_pts, 0.0);

  // Normal covector (pointing in x-direction)
  auto normal_covector =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(num_pts, 0.0);
  normal_covector.get(0) = 1.0;

  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
      face_mesh_velocity{};
  const bool evolve_lapse_and_shift = true;

  // ---- Test dg_ghost ----
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
  auto ghost_boundary_theta = make_with_value<Scalar<DataVector>>(num_pts, 0.0);
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
      interior_u_tensor_minus,
      interior_boundary_conformal_metric, interior_boundary_conformal_factor,
      interior_boundary_lapse, interior_boundary_shift,
      interior_boundary_theta, interior_boundary_z, coords, time,
      evolve_lapse_and_shift);

  CHECK_FALSE(ghost_result.has_value());

  // When interior == analytic, the characteristic mode mixing (incoming from
  // analytic, outgoing from interior) should reproduce the KerrSchild solution.
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

  // ---- Test dg_time_derivative ----
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
      make_not_null(&dt_bshift),
      make_not_null(&dt_btheta), make_not_null(&dt_bz),
      face_mesh_velocity, normal_covector,
      interior_conformal_metric, interior_conformal_factor, interior_a_tilde,
      interior_trace_K, interior_theta, interior_gamma_hat, interior_lapse,
      interior_shift, interior_auxiliary_shift_b, interior_field_a,
      interior_field_b, interior_field_d, interior_field_p,
      interior_u_tensor_minus,
      interior_boundary_conformal_metric, interior_boundary_conformal_factor,
      interior_boundary_lapse, interior_boundary_shift,
      interior_boundary_theta, interior_boundary_z, coords, time,
      evolve_lapse_and_shift);

  CHECK_FALSE(dt_result.has_value());

  // The original 14 evolved-field dt corrections are explicitly zeroed by
  // dg_time_derivative (they are not modified by the boundary condition).
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
  // Boundary dt corrections (dt_bcm, dt_bcf, dt_blapse, dt_bshift) are
  // nonzero for KerrSchild because the CCZ4 gauge condition (1+log slicing)
  // does not match the Kerr-Schild gauge. These are tested in detail
  // by test_kerrschild_dt_boundary_conformal_metric.
}

// Verify that dg_time_derivative computes the correct dt_boundary_conformal_
// metric (boundary-integrated evolution path). This test catches the tnsr::ii
// vs tnsr::ij symmetry bug
// in compute_dt_second_order_fields: conformal_metric_times_field_b_{ij} =
// γ̃_{ki} B_j^k is NOT symmetric, so it must be stored as tnsr::ij. With
// tnsr::ii, the sum M_{ij} + M_{ji} in eq 12a silently becomes 2*M_{ij},
// losing the antisymmetric part.
void test_kerrschild_dt_boundary_conformal_metric() {
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
      "  PrescribeOutgoing: false\n"
      "  InitialTime: 0.0\n");
  const auto& bc = dynamic_cast<
      const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving&>(
      *bc_ptr);

  static constexpr size_t Dim = 3;
  const size_t num_pts = 5;

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

  auto interior_u_tensor_minus =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(num_pts, 0.0);

  // Boundary-integrated = analytic
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
      make_not_null(&dt_bshift),
      make_not_null(&dt_btheta), make_not_null(&dt_bz),
      face_mesh_velocity, normal_covector,
      interior_conformal_metric, interior_conformal_factor, interior_a_tilde,
      interior_trace_K, interior_theta, interior_gamma_hat, interior_lapse,
      interior_shift, interior_auxiliary_shift_b, interior_field_a,
      interior_field_b, interior_field_d, interior_field_p,
      interior_u_tensor_minus,
      interior_boundary_conformal_metric, interior_boundary_conformal_factor,
      interior_boundary_lapse, interior_boundary_shift,
      interior_boundary_theta, interior_boundary_z, coords, time,
      evolve_lapse_and_shift);

  CHECK_FALSE(dt_result.has_value());

  // Reference eq 12a computed with explicit component loops.
  // When interior == analytic == boundary-integrated, the char-mixed state
  // equals the analytic state, so we use analytic fields for the reference.
  //
  // eq 12a: dt γ̃_{ij} = 2β^k D_{k,i,j} + γ̃_{ki}B_j^k + γ̃_{kj}B_i^k
  //                      - (2/3) γ̃_{ij} B_k^k - 2α Ã_{ij}
  const Approx custom_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
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

// TODO: Add a test that verifies CRPBC evaluates the analytic solution at
// initial_time_ (NOT the runtime time). Currently all available CCZ4 analytic
// solutions (Minkowski, KerrSchild) are stationary, so passing different time
// values produces identical results and cannot distinguish CRPBC from
// DirichletCharacteristics. Once a time-dependent CCZ4 solution is available,
// add a test that:
//   1. Creates CRPBC with InitialTime: 0.0
//   2. Calls dg_ghost with time = 5.0
//   3. Verifies the ghost state matches the t=0 analytic solution (not t=5)

SPECTRE_TEST_CASE(
    "Unit.Ccz4.BoundaryConditions.ConstraintsRadiationPreserving",
    "[Unit][Evolution]") {
  test_creation_and_serialization();
  test_bc_type();
  test_minkowski();
  test_kerrschild();
  test_kerrschild_dt_boundary_conformal_metric();
}
}  // namespace
