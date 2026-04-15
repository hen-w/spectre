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
#include "Domain/Structure/Direction.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
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

// Helper: create a volume mesh, diagonal inverse Jacobian, and face
// coordinates for tests.  The face is at the upper side of dim=0.
// The affine map x^i = offset_i + scale_i * xi^i places the face
// well outside any BH horizon.
struct TestMeshData {
  Mesh<3> volume_mesh;
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      volume_inv_jac;
  tnsr::I<DataVector, 3, Frame::Inertial> face_coords;
};

TestMeshData make_test_mesh_and_inv_jac(const size_t ny, const size_t nz) {
  // Volume mesh with face at dim=0 having ny*nz grid points.
  // Both face dimensions use GaussLobatto so spectral derivatives work.
  Mesh<3> volume_mesh(
      {{2, ny, nz}},
      Spectral::Basis::Legendre,
      Spectral::Quadrature::GaussLobatto);

  // Affine map: x^i = offset_i + scale_i * xi^i
  // Places the face at x ~ 5, y in [4.5, 5.5], z in [-0.4, 0.6]
  const std::array<double, 3> offset{{4.5, 5.0, 0.1}};
  const std::array<double, 3> scale{{0.5, 0.5, 0.5}};

  const size_t num_vol_pts = volume_mesh.number_of_grid_points();
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      volume_inv_jac(num_vol_pts, 0.0);
  for (size_t d = 0; d < 3; ++d) {
    volume_inv_jac.get(d, d) = 1.0 / gsl::at(scale, d);
  }

  // Face coordinates at the upper side of dim=0 (xi^0 = 1).
  const size_t num_face_pts = ny * nz;
  const Mesh<2> face_mesh = volume_mesh.slice_away(0);
  const auto face_logical = logical_coordinates(face_mesh);
  tnsr::I<DataVector, 3, Frame::Inertial> face_coords(num_face_pts);
  // x = offset[0] + scale[0] * 1.0  (upper face)
  get<0>(face_coords) = offset[0] + scale[0];
  // y = offset[1] + scale[1] * xi^{face_0}  (face dim 0 = volume dim 1)
  get<1>(face_coords) = offset[1] + scale[1] * get<0>(face_logical);
  // z = offset[2] + scale[2] * xi^{face_1}  (face dim 1 = volume dim 2)
  get<2>(face_coords) = offset[2] + scale[2] * get<1>(face_logical);

  return {std::move(volume_mesh), std::move(volume_inv_jac),
          std::move(face_coords)};
}

void test_creation_and_serialization() {
  register_factory_classes_with_charm<Metavariables>();
  const auto boundary_condition = TestHelpers::test_creation<
      std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>,
      Metavariables>(
      "ConstraintsRadiationPreserving:\n"
      "  AnalyticPrescription:\n"
      "    Ccz4(Minkowski):\n"
      "  UseAnalyticForAll: false\n"
      "  ZeroAllIncomingModes: false\n"
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
      "  ZeroAllIncomingModes: false\n"
      "  PenaltyMultiplier: 1.0\n"
      "  ZeroBoundaryThetaAndZ: false\n");
  const auto& bc = dynamic_cast<
      const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving&>(*bc_ptr);

  static constexpr size_t Dim = 3;
  const size_t ny = 12;
  const size_t nz = 12;
  const size_t num_pts = ny * nz;
  const auto mesh_data = make_test_mesh_and_inv_jac(ny, nz);
  const auto& volume_mesh = mesh_data.volume_mesh;
  const auto& volume_inv_jac = mesh_data.volume_inv_jac;
  const auto& coords = mesh_data.face_coords;

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
      coords, time, evolve_lapse_and_shift, volume_mesh, volume_inv_jac);

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
      coords, time, evolve_lapse_and_shift, volume_mesh, volume_inv_jac);

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
      "  ZeroAllIncomingModes: false\n"
      "  PenaltyMultiplier: 1.0\n"
      "  ZeroBoundaryThetaAndZ: false\n");
  const auto& bc = dynamic_cast<
      const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving&>(*bc_ptr);

  static constexpr size_t Dim = 3;
  const size_t ny = 3;
  const size_t nz = 3;
  const size_t num_pts = ny * nz;
  const auto mesh_data = make_test_mesh_and_inv_jac(ny, nz);
  const auto& volume_mesh = mesh_data.volume_mesh;
  const auto& volume_inv_jac = mesh_data.volume_inv_jac;

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

  const auto& coords = mesh_data.face_coords;

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
      coords, time, evolve_lapse_and_shift, volume_mesh, volume_inv_jac);

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
      coords, time, evolve_lapse_and_shift, volume_mesh, volume_inv_jac);

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
      "  ZeroAllIncomingModes: false\n"
      "  PenaltyMultiplier: 1.0\n"
      "  ZeroBoundaryThetaAndZ: false\n");
  const auto& bc = dynamic_cast<
      const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving&>(*bc_ptr);

  static constexpr size_t Dim = 3;
  const size_t ny = 12;
  const size_t nz = 12;
  const size_t num_pts = ny * nz;
  const auto mesh_data = make_test_mesh_and_inv_jac(ny, nz);
  const auto& volume_mesh = mesh_data.volume_mesh;
  const auto& volume_inv_jac = mesh_data.volume_inv_jac;
  const auto& coords = mesh_data.face_coords;

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
      coords, time, evolve_lapse_and_shift, volume_mesh, volume_inv_jac);

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
      "  ZeroAllIncomingModes: false\n"
      "  PenaltyMultiplier: 1.0\n"
      "  ZeroBoundaryThetaAndZ: false\n");
  const auto& bc = dynamic_cast<
      const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving&>(*bc_ptr);

  static constexpr size_t Dim = 3;
  const size_t ny = 12;
  const size_t nz = 12;
  const size_t num_pts = ny * nz;
  const auto mesh_data = make_test_mesh_and_inv_jac(ny, nz);
  const auto& volume_mesh = mesh_data.volume_mesh;
  const auto& volume_inv_jac = mesh_data.volume_inv_jac;
  const auto& coords = mesh_data.face_coords;

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

  // Normal covector consistent with the mesh face at dim=0 (unnormalized
  // normal = J^{-1}_{0,:}).  With our diagonal Jacobian, this is (1/scale, 0, 0).
  auto normal_covector =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(num_pts, 0.0);
  normal_covector.get(0) = volume_inv_jac.get(0, 0)[0];

  // Boundary-integrated (coeff) four fields: perturb all of them.
  // The conformal metric must keep det = 1 (CCZ4 requirement).
  auto interior_boundary_conformal_metric = interior_conformal_metric;
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      interior_boundary_conformal_metric.get(i, j) *=
          1.0 + 0.01 * static_cast<double>(i + j + 1);
    }
  }
  // Rescale to enforce det(coeff_cm) = 1
  const auto det_coeff_cm =
      get(determinant(interior_boundary_conformal_metric));
  const auto scale = 1.0 / cbrt(det_coeff_cm);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      interior_boundary_conformal_metric.get(i, j) *= scale;
    }
  }
  auto interior_boundary_conformal_factor = interior_conformal_factor;
  get(interior_boundary_conformal_factor) *= 1.02;
  auto interior_boundary_lapse = interior_lapse;
  get(interior_boundary_lapse) *= 0.97;
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
      coords, time, evolve_lapse_and_shift, volume_mesh, volume_inv_jac);

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

// Test that face-tangential spectral derivatives + transverse-projected
// inverse Jacobian give the same result as projecting the volume transverse
// derivative to the face.  Uses a non-trivial polynomial conformal metric
// on a non-orthogonal affine map so the test is sensitive to all terms.
void test_face_transverse_derivatives() {
  static constexpr size_t Dim = 3;

  // Volume mesh (GaussLobatto so face projection = slicing)
  const Mesh<Dim> volume_mesh({{5, 6, 7}}, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto);

  // Constant non-orthogonal Jacobian: x^i = J^i_a ξ^a + offset
  // J is non-diagonal and non-symmetric for generality.
  const std::array<std::array<double, Dim>, Dim> J_matrix{{
      {{1.0, 0.15, -0.1}},
      {{0.1, 0.9, 0.12}},
      {{-0.05, 0.08, 1.1}}}};

  // Compute inverse Jacobian (J^{-1})_{a,i} = ∂ξ^a/∂x^i
  // For a 3x3 matrix, invert manually via cofactors.
  auto cofactor = [&](size_t r, size_t c) -> double {
    const size_t r1 = (r + 1) % 3;
    const size_t r2 = (r + 2) % 3;
    const size_t c1 = (c + 1) % 3;
    const size_t c2 = (c + 2) % 3;
    return gsl::at(gsl::at(J_matrix, r1), c1) *
               gsl::at(gsl::at(J_matrix, r2), c2) -
           gsl::at(gsl::at(J_matrix, r1), c2) *
               gsl::at(gsl::at(J_matrix, r2), c1);
  };
  double det_J = 0.0;
  for (size_t c = 0; c < Dim; ++c) {
    det_J += gsl::at(gsl::at(J_matrix, 0), c) * cofactor(0, c);
  }
  REQUIRE(det_J > 0.0);

  // inv_J(a, i) = cofactor(i, a) / det_J  (transpose of cofactor / det)
  std::array<std::array<double, Dim>, Dim> inv_J_matrix{};
  for (size_t a = 0; a < Dim; ++a) {
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(gsl::at(inv_J_matrix, a), i) = cofactor(i, a) / det_J;
    }
  }

  // Logical coordinates on volume mesh
  const auto logical_coords = logical_coordinates(volume_mesh);
  const size_t num_vol_pts = volume_mesh.number_of_grid_points();

  // Inertial coordinates: x^i = J^i_a ξ^a  (no offset for simplicity)
  tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords(num_vol_pts, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t a = 0; a < Dim; ++a) {
      inertial_coords.get(i) +=
          gsl::at(gsl::at(J_matrix, i), a) * logical_coords.get(a);
    }
  }
  const auto& x = inertial_coords.get(0);
  const auto& y = inertial_coords.get(1);
  const auto& z = inertial_coords.get(2);

  // Non-trivial polynomial conformal metric on the volume.
  // Must be low-degree enough that the spectral derivative is exact.
  tnsr::ii<DataVector, Dim, Frame::Inertial> gamma_tilde(num_vol_pts);
  gamma_tilde.get(0, 0) = 1.0 + 0.1 * x + 0.05 * y + 0.02 * z;
  gamma_tilde.get(0, 1) = 0.03 * x + 0.02 * y + 0.01 * z;
  gamma_tilde.get(0, 2) = 0.02 * x - 0.01 * y + 0.015 * z;
  gamma_tilde.get(1, 1) = 1.0 + 0.08 * y + 0.03 * z;
  gamma_tilde.get(1, 2) = 0.01 * x + 0.025 * y - 0.015 * z;
  gamma_tilde.get(2, 2) = 1.0 + 0.06 * z + 0.04 * x;

  // Volume inverse Jacobian (constant, broadcast to volume points)
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      volume_inv_jac(num_vol_pts);
  for (size_t a = 0; a < Dim; ++a) {
    for (size_t i = 0; i < Dim; ++i) {
      volume_inv_jac.get(a, i) = gsl::at(gsl::at(inv_J_matrix, a), i);
    }
  }

  // Volume logical derivative of gamma_tilde: d_logical(a, i, j) on volume
  const auto vol_logical_deriv =
      logical_partial_derivative(gamma_tilde, volume_mesh);

  // Volume spatial derivative: d_spatial(k, i, j) = inv_J(a,k) * d_log(a,i,j)
  tnsr::ijj<DataVector, Dim, Frame::Inertial> vol_d_gamma(num_vol_pts, 0.0);
  for (size_t k = 0; k < Dim; ++k) {
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        for (size_t a = 0; a < Dim; ++a) {
          vol_d_gamma.get(k, i, j) +=
              volume_inv_jac.get(a, k) * vol_logical_deriv.get(a, i, j);
        }
      }
    }
  }

  // Conformal factor for the spatial metric g_{ij} = gamma_tilde_{ij}/phi^2
  const double phi_val = 1.2;  // constant for simplicity

  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);

  // Test all 6 face directions
  for (const auto& direction : Direction<Dim>::all_directions()) {
    CAPTURE(direction);
    const size_t face_dim = direction.dimension();
    const Mesh<Dim - 1> face_mesh = volume_mesh.slice_away(face_dim);
    const size_t num_face_pts = face_mesh.number_of_grid_points();

    // Project gamma_tilde to face
    tnsr::ii<DataVector, Dim, Frame::Inertial> face_gamma(num_face_pts);
    ::dg::project_tensor_to_boundary(make_not_null(&face_gamma), gamma_tilde,
                                     volume_mesh, direction);

    // Face logical derivative of gamma_tilde, computed via spectral
    // differentiation on the face mesh.  We differentiate component by
    // component (as Scalars) because logical_partial_derivative is not
    // instantiated for tnsr::ii<DV,3> on Mesh<2>.
    // face_log_deriv[b].get(i,j) = ∂γ̃_{ij}/∂ξ^{face_b}
    std::array<tnsr::ii<DataVector, Dim, Frame::Inertial>, Dim - 1>
        face_log_deriv;
    for (size_t b = 0; b < Dim - 1; ++b) {
      gsl::at(face_log_deriv, b) =
          tnsr::ii<DataVector, Dim, Frame::Inertial>(num_face_pts, 0.0);
    }
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        const Scalar<DataVector> comp(face_gamma.get(i, j));
        const auto dcomp = logical_partial_derivative(comp, face_mesh);
        for (size_t b = 0; b < Dim - 1; ++b) {
          gsl::at(face_log_deriv, b).get(i, j) = dcomp.get(b);
        }
      }
    }

    // Map face logical dim b -> volume logical dim
    auto volume_dim_of_face = [&face_dim](const size_t b) -> size_t {
      return b < face_dim ? b : b + 1;
    };

    // Compute unit normal from the face-normal row of the inverse Jacobian.
    // Unnormalized normal covector: n_i^unnorm = J^{-1}_{d,i}
    tnsr::i<DataVector, Dim, Frame::Inertial> n_cov(num_face_pts);
    for (size_t i = 0; i < Dim; ++i) {
      n_cov.get(i) = gsl::at(gsl::at(inv_J_matrix, face_dim), i);
    }

    // Inverse spatial metric g^{ij} = phi^2 * gamma_tilde^{ij}
    // For normalizing, we need g^{ij} on the face.
    const auto [det_face_gamma, inv_face_gamma] =
        determinant_and_inverse(face_gamma);
    tnsr::II<DataVector, Dim, Frame::Inertial> inv_spatial_metric(
        num_face_pts);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        inv_spatial_metric.get(i, j) =
            phi_val * phi_val * inv_face_gamma.get(i, j);
      }
    }

    // Normalize: n_i = n_i^unnorm / sqrt(g^{jk} n_j n_k)
    const Scalar<DataVector> mag_sq =
        dot_product(n_cov, n_cov, inv_spatial_metric);
    const DataVector inv_mag = 1.0 / sqrt(get(mag_sq));
    for (size_t i = 0; i < Dim; ++i) {
      n_cov.get(i) *= inv_mag;
    }

    // Normal vector: n^i = g^{ij} n_j
    tnsr::I<DataVector, Dim, Frame::Inertial> n_vec(num_face_pts, 0.0);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        n_vec.get(i) += inv_spatial_metric.get(i, j) * n_cov.get(j);
      }
    }

    // --- Method 1: Face spectral derivative + transverse-projected Jacobian ---
    // q^j_i d_j gamma = sum_{b in face} (J^{-1}_{vol(b),i} - n_i n^j
    //     J^{-1}_{vol(b),j}) * d_face_log(b, k, l)
    tnsr::ijj<DataVector, Dim, Frame::Inertial> transverse_deriv_face(
        num_face_pts, 0.0);
    for (size_t b = 0; b < Dim - 1; ++b) {
      const size_t vol_a = volume_dim_of_face(b);
      // Compute projected Jacobian row:
      // P_i = J^{-1}_{vol_a, i} - n_i (n^j J^{-1}_{vol_a, j})
      DataVector n_dot_jac(num_face_pts, 0.0);
      for (size_t j = 0; j < Dim; ++j) {
        n_dot_jac += n_vec.get(j) *
                     gsl::at(gsl::at(inv_J_matrix, vol_a), j);
      }
      for (size_t i = 0; i < Dim; ++i) {
        const DataVector proj_jac_i =
            gsl::at(gsl::at(inv_J_matrix, vol_a), i) -
            n_cov.get(i) * n_dot_jac;
        for (size_t k = 0; k < Dim; ++k) {
          for (size_t l = k; l < Dim; ++l) {
            transverse_deriv_face.get(i, k, l) +=
                proj_jac_i * gsl::at(face_log_deriv, b).get(k, l);
          }
        }
      }
    }

    // --- Method 2: Project volume transverse derivative to face ---
    // First project vol_d_gamma to face, then apply q^j_i.
    tnsr::ijj<DataVector, Dim, Frame::Inertial> face_vol_d_gamma(num_face_pts);
    ::dg::project_tensor_to_boundary(make_not_null(&face_vol_d_gamma),
                                     vol_d_gamma, volume_mesh, direction);

    tnsr::ijj<DataVector, Dim, Frame::Inertial> transverse_deriv_vol(
        num_face_pts, 0.0);
    for (size_t i = 0; i < Dim; ++i) {
      // n_i * n^j
      for (size_t k = 0; k < Dim; ++k) {
        for (size_t l = k; l < Dim; ++l) {
          transverse_deriv_vol.get(i, k, l) = face_vol_d_gamma.get(i, k, l);
          for (size_t j = 0; j < Dim; ++j) {
            transverse_deriv_vol.get(i, k, l) -=
                n_cov.get(i) * n_vec.get(j) * face_vol_d_gamma.get(j, k, l);
          }
        }
      }
    }

    // Compare
    CHECK_ITERABLE_CUSTOM_APPROX(transverse_deriv_face, transverse_deriv_vol,
                                 custom_approx);
  }
}

// Same as test_face_transverse_derivatives but on a SphericalHarmonic mesh
// (as used by Shell/Sphere domain creators).  The outer radial face (dim=0)
// has a Mesh<2> with SphericalHarmonic basis, so the face derivative must
// use ylm::Spherepack::gradient instead of the standard differentiation
// matrix.
void test_face_transverse_derivatives_spherical() {
  static constexpr size_t Dim = 3;

  // Volume mesh: radial (Legendre Gauss) x angular (SphericalHarmonic).
  // Use l_max = 5 so that low-order Ylm are exactly representable.
  const size_t n_r = 4;
  const size_t l_max = 5;
  const Mesh<Dim> volume_mesh{
      {{n_r, l_max + 1, 2 * l_max + 1}},
      {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss,
        Spectral::Quadrature::Equiangular}}};

  // Logical coordinates: xi^0 = r_logical in [-1,1], xi^1 = theta, xi^2 = phi
  const auto logical_coords = logical_coordinates(volume_mesh);
  const size_t num_vol_pts = volume_mesh.number_of_grid_points();

  // Affine map for the radial direction: r = r0 + dr * xi^0
  // Angular directions: x = r*sin(theta)*cos(phi), etc.
  // For this test, we use a diagonal Jacobian where
  //   J^i_0 = dx^i/dr_logical (depends on angles)
  //   J^i_1 = dx^i/dtheta     (depends on r and angles)
  //   J^i_2 = dx^i/dphi       (depends on r and angles)
  // This is position-dependent, so we use the full volume Jacobian.
  const double r0 = 10.0;  // inner radius (in logical coords)
  const double dr = 2.0;   // half-width
  const DataVector r = r0 + dr * logical_coords.get(0);
  const DataVector& theta = logical_coords.get(1);
  const DataVector& phi = logical_coords.get(2);

  const DataVector sin_theta = sin(theta);
  const DataVector cos_theta = cos(theta);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_phi = cos(phi);

  // Inverse Jacobian for spherical coordinates:
  //   (J^{-1})_{0,i} = (1/dr) * (sin_theta cos_phi, sin_theta sin_phi,
  //                               cos_theta)   [radial]
  //   (J^{-1})_{1,i} = (1/r) * (cos_theta cos_phi, cos_theta sin_phi,
  //                              -sin_theta)   [theta]
  //   (J^{-1})_{2,i} = (1/(r sin_theta)) * (-sin_phi, cos_phi, 0)   [phi]
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      volume_inv_jac(num_vol_pts, 0.0);
  // radial row (d xi^0 / d x^i)
  volume_inv_jac.get(0, 0) = sin_theta * cos_phi / dr;
  volume_inv_jac.get(0, 1) = sin_theta * sin_phi / dr;
  volume_inv_jac.get(0, 2) = cos_theta / dr;
  // theta row (d theta / d x^i)
  volume_inv_jac.get(1, 0) = cos_theta * cos_phi / r;
  volume_inv_jac.get(1, 1) = cos_theta * sin_phi / r;
  volume_inv_jac.get(1, 2) = -sin_theta / r;
  // phi row (d phi / d x^i) — note: Spherepack returns csc(theta)*d/dphi,
  // so logical dim 2 already has the 1/sin(theta) absorbed.
  volume_inv_jac.get(2, 0) = -sin_phi / (r * sin_theta);
  volume_inv_jac.get(2, 1) = cos_phi / (r * sin_theta);
  volume_inv_jac.get(2, 2) = 0.0;

  // Construct conformal metric using low-order Ylm so angular derivatives
  // are exact.  Use Y_1^0 ~ cos(theta) and Y_1^1 ~ sin(theta)*cos(phi).
  tnsr::ii<DataVector, Dim, Frame::Inertial> gamma_tilde(num_vol_pts);
  gamma_tilde.get(0, 0) = 1.0 + 0.05 * cos_theta;
  gamma_tilde.get(0, 1) = 0.02 * sin_theta * cos_phi;
  gamma_tilde.get(0, 2) = 0.01 * sin_theta * sin_phi;
  gamma_tilde.get(1, 1) = 1.0 + 0.03 * cos_theta;
  gamma_tilde.get(1, 2) = 0.015 * cos_theta;
  gamma_tilde.get(2, 2) = 1.0 - 0.02 * cos_theta;

  // Volume logical derivative (handled by Spherepack internally for dims 1,2)
  const auto vol_logical_deriv =
      logical_partial_derivative(gamma_tilde, volume_mesh);

  // Volume spatial derivative: d_spatial(k,i,j) = sum_a inv_J(a,k) *
  // d_logical(a,i,j)
  tnsr::ijj<DataVector, Dim, Frame::Inertial> vol_d_gamma(num_vol_pts, 0.0);
  for (size_t k = 0; k < Dim; ++k) {
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        for (size_t a = 0; a < Dim; ++a) {
          vol_d_gamma.get(k, i, j) +=
              volume_inv_jac.get(a, k) * vol_logical_deriv.get(a, i, j);
        }
      }
    }
  }

  // Test the outer radial face (upper side of dim 0).
  const size_t face_dim = 0;
  const auto direction = Direction<Dim>(face_dim, Side::Upper);
  const Mesh<Dim - 1> face_mesh = volume_mesh.slice_away(face_dim);
  const size_t num_face_pts = face_mesh.number_of_grid_points();

  // Conformal factor for constructing the spatial metric (constant)
  const double phi_val = 1.3;

  // Project gamma_tilde to face
  tnsr::ii<DataVector, Dim, Frame::Inertial> face_gamma(num_face_pts);
  ::dg::project_tensor_to_boundary(make_not_null(&face_gamma), gamma_tilde,
                                   volume_mesh, direction);

  // Project inverse Jacobian to face
  auto face_inv_jac = ::dg::project_tensor_to_boundary(
      volume_inv_jac, volume_mesh, direction);

  // Compute unit normal from the radial row of the inverse Jacobian
  tnsr::i<DataVector, Dim, Frame::Inertial> n_cov(num_face_pts);
  for (size_t i = 0; i < Dim; ++i) {
    n_cov.get(i) = face_inv_jac.get(face_dim, i);
  }
  // Inverse spatial metric: g^{ij} = phi^2 * gamma_tilde^{ij}
  const auto [det_face_gamma, inv_face_gamma] =
      determinant_and_inverse(face_gamma);
  tnsr::II<DataVector, Dim, Frame::Inertial> inv_spatial_metric(num_face_pts);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      inv_spatial_metric.get(i, j) =
          phi_val * phi_val * inv_face_gamma.get(i, j);
    }
  }
  const Scalar<DataVector> mag_sq =
      dot_product(n_cov, n_cov, inv_spatial_metric);
  const DataVector inv_mag = 1.0 / sqrt(get(mag_sq));
  for (size_t i = 0; i < Dim; ++i) {
    n_cov.get(i) *= inv_mag;
  }
  tnsr::I<DataVector, Dim, Frame::Inertial> n_vec(num_face_pts, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      n_vec.get(i) += inv_spatial_metric.get(i, j) * n_cov.get(j);
    }
  }

  // --- Method 1: Face spectral derivative (Spherepack) + projected Jacobian ---
  // face_dim is 0, so volume_dim_of_face(b) = b + 1
  auto volume_dim_of_face = [](const size_t b) -> size_t {
    return b + 1;
  };

  const auto& ylm = ylm::get_spherepack_cache(face_mesh.extents(0) - 1);

  std::array<tnsr::ii<DataVector, Dim, Frame::Inertial>, Dim - 1>
      face_log_deriv;
  for (size_t b = 0; b < Dim - 1; ++b) {
    gsl::at(face_log_deriv, b) =
        tnsr::ii<DataVector, Dim, Frame::Inertial>(num_face_pts, 0.0);
  }
  for (size_t ii = 0; ii < Dim; ++ii) {
    for (size_t jj = ii; jj < Dim; ++jj) {
      const auto grad = ylm.gradient(face_gamma.get(ii, jj));
      gsl::at(face_log_deriv, 0).get(ii, jj) = grad.get(0);
      gsl::at(face_log_deriv, 1).get(ii, jj) = grad.get(1);
    }
  }

  tnsr::ijj<DataVector, Dim, Frame::Inertial> transverse_deriv_face(
      num_face_pts, 0.0);
  for (size_t b = 0; b < Dim - 1; ++b) {
    const size_t vol_a = volume_dim_of_face(b);
    DataVector n_dot_jac(num_face_pts, 0.0);
    for (size_t j = 0; j < Dim; ++j) {
      n_dot_jac += n_vec.get(j) * face_inv_jac.get(vol_a, j);
    }
    for (size_t ii = 0; ii < Dim; ++ii) {
      for (size_t jj = ii; jj < Dim; ++jj) {
        const DataVector& dcomp_b = gsl::at(face_log_deriv, b).get(ii, jj);
        for (size_t i = 0; i < Dim; ++i) {
          const DataVector proj_jac_i =
              face_inv_jac.get(vol_a, i) - n_cov.get(i) * n_dot_jac;
          transverse_deriv_face.get(i, ii, jj) += proj_jac_i * dcomp_b;
        }
      }
    }
  }

  // --- Method 2: Project volume transverse derivative to face ---
  tnsr::ijj<DataVector, Dim, Frame::Inertial> face_vol_d_gamma(num_face_pts);
  ::dg::project_tensor_to_boundary(make_not_null(&face_vol_d_gamma),
                                   vol_d_gamma, volume_mesh, direction);

  tnsr::ijj<DataVector, Dim, Frame::Inertial> transverse_deriv_vol(
      num_face_pts, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t k = 0; k < Dim; ++k) {
      for (size_t l = k; l < Dim; ++l) {
        transverse_deriv_vol.get(i, k, l) = face_vol_d_gamma.get(i, k, l);
        for (size_t j = 0; j < Dim; ++j) {
          transverse_deriv_vol.get(i, k, l) -=
              n_cov.get(i) * n_vec.get(j) * face_vol_d_gamma.get(j, k, l);
        }
      }
    }
  }

  // Compare — SphericalHarmonic is spectral, so expect high accuracy
  const Approx custom_approx = Approx::custom().epsilon(1.0e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(transverse_deriv_face, transverse_deriv_vol,
                                custom_approx);
}

SPECTRE_TEST_CASE("Unit.Ccz4.BoundaryConditions.ConstraintsRadiationPreserving",
                  "[Unit][Evolution]") {
  test_creation_and_serialization();
  test_bc_type();
  test_minkowski();
  test_kerrschild_use_analytic_for_all();
  test_kerrschild_roundtrip_t_perp();
  test_theta_and_z_roundtrip();
  test_face_transverse_derivatives();
  test_face_transverse_derivatives_spherical();
}
}  // namespace
