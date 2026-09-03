// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/DirichletCharacteristics.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/System.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SecondOrderWrapper.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

// `DirichletCharacteristics` is a `Type::Ghost` boundary condition whose
// `dg_ghost` takes the current per-face boundary-evolved value as an extra
// argument spliced in after the normal covector, and which also supplies a
// `boundary_field_time_derivatives` method for the boundary-evolved Psi. The
// generic
// `TestHelpers::evolution::dg::test_boundary_condition_with_python` helper
// knows only the fixed `dg_ghost`/`dg_time_derivative` calling conventions, not
// the extra boundary-value argument nor the `boundary_field_time_derivatives`
// method, so it cannot drive either method here. Instead this test calls both
// methods directly and compares against a python reference, mirroring the
// structure of the frozen SoScalarWave test.
//
// Adaptations relative to that frozen test:
//   * The second-order system has THREE characteristic fields (Psi has no
//     characteristic field): v^0_i (speed lambda^0), v^+ (speed lambda^+),
//     v^- (speed lambda^-). On a moving mesh each speed shifts by -n_i v^i, so
//     the per-point incoming/outgoing classification varies across a face. The
//     python reference implements the ghost formulas directly from the raw
//     inputs.
//   * Moving-mesh coverage: every pypp cross-check runs twice (with a null and
//     with an engaged face mesh velocity whose per-point components span
//     [-2, 2], giving both signs of n.v and superluminal points), a
//     zero-velocity engaged optional is checked to reduce exactly to the null
//     case, and a hand-computed 2D pin fixes the mask logic, the superluminal
//     v^+ flip, and the advection term independently of python.
//   * The analytic prescription is
//     `SecondOrderScalarWave::Solutions::SecondOrderWrapper<
//         ScalarWave::Solutions::PlaneWave<Dim>>`.

namespace {
template <size_t Dim>
using Solution = SecondOrderScalarWave::Solutions::SecondOrderWrapper<
    ScalarWave::Solutions::PlaneWave<Dim>>;

template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            SecondOrderScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
            tmpl::list<SecondOrderScalarWave::BoundaryConditions::
                           DirichletCharacteristics<Dim>,
                       SecondOrderScalarWave::BoundaryConditions::
                           DirichletAnalytic<Dim>>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   SecondOrderScalarWave::Solutions::all_solutions<Dim>>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>>;
  };
};

// Builds the plane-wave parameters shared by the C++ construction and the
// python analytic evaluation (matching the YAML below and the numbers in the
// python reference).
template <size_t Dim>
std::array<double, Dim> plane_wave_wave_vector() {
  std::array<double, Dim> wave_vector{};
  gsl::at(wave_vector, 0) = 0.1;
  if constexpr (Dim > 1) {
    gsl::at(wave_vector, 1) = 1.1;
  }
  if constexpr (Dim > 2) {
    gsl::at(wave_vector, 2) = 2.1;
  }
  return wave_vector;
}

template <size_t Dim>
std::array<double, Dim> plane_wave_center() {
  std::array<double, Dim> center{};
  gsl::at(center, 0) = 1.1;
  if constexpr (Dim > 1) {
    gsl::at(center, 1) = 0.1;
  }
  if constexpr (Dim > 2) {
    gsl::at(center, 2) = -0.9;
  }
  return center;
}

// Converts the analytic solution passed to `dg_ghost` /
// `boundary_field_time_derivatives` into the trailing `dim` python argument,
// matching the python reference's signature (the reference reconstructs the
// solution from `dim` alone since all its parameters are fixed).
template <size_t Dim>
struct ConvertPlaneWave {
  using unpacked_container = int;
  using packed_container = Solution<Dim>;
  using packed_type = double;

  static packed_container create_container() {
    return packed_container{
        plane_wave_wave_vector<Dim>(), plane_wave_center<Dim>(),
        std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(0.9, 0.6,
                                                                      0.0)};
  }

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    return Dim;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container /*unpacked*/,
                          const size_t /*grid_point_index*/) {
    *packed = create_container();
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

template <size_t Dim>
std::string yaml_string(const bool zero_incoming) {
  return "DirichletCharacteristics:\n"
         "  AnalyticPrescription:\n"
         "    SecondOrderPlaneWave:\n"
         "      WaveVector: [0.1" +
         (Dim > 1 ? std::string{", 1.1"} : std::string{}) +
         (Dim > 2 ? std::string{", 2.1"} : std::string{}) +
         "]\n"
         "      Center: [1.1" +
         (Dim > 1 ? std::string{", 0.1"} : std::string{}) +
         (Dim > 2 ? std::string{", -0.9"} : std::string{}) +
         "]\n"
         "      Profile:\n"
         "        Gaussian:\n"
         "          Amplitude: 0.9\n"
         "          Width: 0.6\n"
         "          Center: 0.0\n"
         "  ZeroIncomingMode: " +
         (zero_incoming ? "true\n" : "false\n");
}

// A random, unit-normalized covector plus the interior/boundary inputs,
// matching the setup the generic helper builds.
template <size_t Dim>
struct FaceData {
  Scalar<DataVector> interior_pi;
  tnsr::i<DataVector, Dim, Frame::Inertial> interior_phi;
  Scalar<DataVector> boundary_psi_value;
  tnsr::i<DataVector, Dim, Frame::Inertial> normal_covector;
  tnsr::I<DataVector, Dim, Frame::Inertial> coords;
};

template <size_t Dim>
FaceData<Dim> make_face_data(const gsl::not_null<std::mt19937*> gen,
                             const size_t num_pts) {
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  FaceData<Dim> data{};
  data.interior_pi = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  data.interior_phi =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          gen, make_not_null(&dist), num_pts);
  data.boundary_psi_value = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
  data.coords =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          gen, make_not_null(&dist), num_pts);
  data.normal_covector =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          gen, make_not_null(&dist), num_pts);
  const Scalar<DataVector> normal_magnitude = magnitude(data.normal_covector);
  for (DataVector& component : data.normal_covector) {
    component /= get(normal_magnitude);
  }
  return data;
}

// Builds an engaged face mesh velocity with per-point components drawn
// uniformly from [-2, 2], so a face carries both signs of n.v and points with
// |n.v| > 1 (superluminal mesh, flipping the sign of lambda^+).
template <size_t Dim>
std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> make_mesh_velocity(
    const gsl::not_null<std::mt19937*> gen, const size_t num_pts) {
  std::uniform_real_distribution<> dist(-2.0, 2.0);
  return make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
      gen, make_not_null(&dist), num_pts);
}

template <size_t Dim>
void test_dg_ghost(const gsl::not_null<std::mt19937*> gen,
                   const bool zero_incoming, const std::string& suffix) {
  CAPTURE(Dim);
  CAPTURE(zero_incoming);
  const size_t num_pts = 5;

  // Upcast to the domain boundary-condition base so serialization dispatches
  // through the registered PUPable interface (as the generic helper does).
  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition = TestHelpers::test_creation<
          std::unique_ptr<SecondOrderScalarWave::BoundaryConditions::
                              BoundaryCondition<Dim>>,
          Metavariables<Dim>>(yaml_string<Dim>(zero_incoming));

  const auto data = make_face_data<Dim>(gen, num_pts);
  const auto solution = ConvertPlaneWave<Dim>::create_container();
  const double time = 0.5;
  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> no_mesh_vel{};
  const auto engaged_mesh_vel = make_mesh_velocity<Dim>(gen, num_pts);

  const auto check = [&data, &solution, &suffix, &no_mesh_vel,
                      &engaged_mesh_vel, num_pts,
                      time](const SecondOrderScalarWave::BoundaryConditions::
                                DirichletCharacteristics<Dim>& concrete) {
    // Run the full pypp cross-check for both a null (static) and an engaged
    // (moving) mesh velocity. The moving pass exercises the per-point mode
    // selection at both signs of n.v and at superluminal points.
    for (const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
             mesh_velocity : {no_mesh_vel, engaged_mesh_vel}) {
      Scalar<DataVector> psi{num_pts};
      Scalar<DataVector> pi{num_pts};
      tnsr::i<DataVector, Dim, Frame::Inertial> phi{num_pts};
      const auto error = concrete.dg_ghost(
          make_not_null(&psi), make_not_null(&pi), make_not_null(&phi),
          mesh_velocity, data.normal_covector, data.boundary_psi_value,
          data.interior_pi, data.interior_phi, data.coords, time);
      CHECK_FALSE(error.has_value());

      const auto expected_psi =
          pypp::call<Scalar<DataVector>, tmpl::list<ConvertPlaneWave<Dim>>>(
              "DirichletCharacteristics", "psi" + suffix, mesh_velocity,
              data.normal_covector, data.interior_pi, data.interior_phi,
              data.boundary_psi_value, data.coords, time, solution);
      const auto expected_pi =
          pypp::call<Scalar<DataVector>, tmpl::list<ConvertPlaneWave<Dim>>>(
              "DirichletCharacteristics", "pi" + suffix, mesh_velocity,
              data.normal_covector, data.interior_pi, data.interior_phi,
              data.boundary_psi_value, data.coords, time, solution);
      const auto expected_phi =
          pypp::call<tnsr::i<DataVector, Dim, Frame::Inertial>,
                     tmpl::list<ConvertPlaneWave<Dim>>>(
              "DirichletCharacteristics", "phi" + suffix, mesh_velocity,
              data.normal_covector, data.interior_pi, data.interior_phi,
              data.boundary_psi_value, data.coords, time, solution);
      CHECK_ITERABLE_APPROX(psi, expected_psi);
      CHECK_ITERABLE_APPROX(pi, expected_pi);
      CHECK_ITERABLE_APPROX(phi, expected_phi);
    }

    // Zero-velocity reduction: an engaged optional with v = 0 everywhere must
    // reproduce the null-velocity result bit-for-bit within tolerance.
    Scalar<DataVector> psi_null{num_pts};
    Scalar<DataVector> pi_null{num_pts};
    tnsr::i<DataVector, Dim, Frame::Inertial> phi_null{num_pts};
    concrete.dg_ghost(make_not_null(&psi_null), make_not_null(&pi_null),
                      make_not_null(&phi_null), no_mesh_vel,
                      data.normal_covector, data.boundary_psi_value,
                      data.interior_pi, data.interior_phi, data.coords, time);
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
        zero_mesh_vel{tnsr::I<DataVector, Dim, Frame::Inertial>{num_pts, 0.0}};
    Scalar<DataVector> psi_zero{num_pts};
    Scalar<DataVector> pi_zero{num_pts};
    tnsr::i<DataVector, Dim, Frame::Inertial> phi_zero{num_pts};
    concrete.dg_ghost(make_not_null(&psi_zero), make_not_null(&pi_zero),
                      make_not_null(&phi_zero), zero_mesh_vel,
                      data.normal_covector, data.boundary_psi_value,
                      data.interior_pi, data.interior_phi, data.coords, time);
    CHECK_ITERABLE_APPROX(psi_zero, psi_null);
    CHECK_ITERABLE_APPROX(pi_zero, pi_null);
    CHECK_ITERABLE_APPROX(phi_zero, phi_null);
  };

  // Run against both the freshly-created and a serialized/deserialized
  // boundary condition (mirrors the generic helper's serialization check).
  check(dynamic_cast<const SecondOrderScalarWave::BoundaryConditions::
                         DirichletCharacteristics<Dim>&>(*boundary_condition));
  const auto deserialized = serialize_and_deserialize(boundary_condition);
  check(dynamic_cast<const SecondOrderScalarWave::BoundaryConditions::
                         DirichletCharacteristics<Dim>&>(*deserialized));
}

template <size_t Dim>
void test_boundary_field_time_derivatives(
    const gsl::not_null<std::mt19937*> gen, const bool zero_incoming,
    const std::string& suffix) {
  CAPTURE(Dim);
  CAPTURE(zero_incoming);
  const size_t num_pts = 5;

  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition = TestHelpers::test_creation<
          std::unique_ptr<SecondOrderScalarWave::BoundaryConditions::
                              BoundaryCondition<Dim>>,
          Metavariables<Dim>>(yaml_string<Dim>(zero_incoming));

  const auto data = make_face_data<Dim>(gen, num_pts);
  const auto solution = ConvertPlaneWave<Dim>::create_container();
  const double time = 0.5;
  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> no_mesh_vel{};
  const auto engaged_mesh_vel = make_mesh_velocity<Dim>(gen, num_pts);

  const auto check = [&data, &solution, &suffix, &no_mesh_vel,
                      &engaged_mesh_vel, num_pts,
                      time](const SecondOrderScalarWave::BoundaryConditions::
                                DirichletCharacteristics<Dim>& concrete) {
    // dt = -Pi_b (+ v.Phi_b with a mesh velocity), with (Pi_b, Phi_b) the same
    // mixed-mode ghost state as `dg_ghost`. Cross-checked against python for
    // both the null and the engaged (moving) mesh velocity.
    for (const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
             mesh_velocity : {no_mesh_vel, engaged_mesh_vel}) {
      Scalar<DataVector> dt_boundary_psi{num_pts};
      const auto error = concrete.boundary_field_time_derivatives(
          make_not_null(&dt_boundary_psi), mesh_velocity, data.normal_covector,
          data.boundary_psi_value, data.interior_pi, data.interior_phi,
          data.coords, time);
      CHECK_FALSE(error.has_value());

      const auto expected_dt =
          pypp::call<Scalar<DataVector>, tmpl::list<ConvertPlaneWave<Dim>>>(
              "DirichletCharacteristics", "dt_boundary_psi" + suffix,
              mesh_velocity, data.normal_covector, data.interior_pi,
              data.interior_phi, data.boundary_psi_value, data.coords, time,
              solution);
      CHECK_ITERABLE_APPROX(dt_boundary_psi, expected_dt);
    }

    // Zero-velocity reduction: an engaged optional with v = 0 everywhere must
    // reproduce the null-velocity result within tolerance.
    Scalar<DataVector> dt_null{num_pts};
    concrete.boundary_field_time_derivatives(
        make_not_null(&dt_null), no_mesh_vel, data.normal_covector,
        data.boundary_psi_value, data.interior_pi, data.interior_phi,
        data.coords, time);
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
        zero_mesh_vel{tnsr::I<DataVector, Dim, Frame::Inertial>{num_pts, 0.0}};
    Scalar<DataVector> dt_zero{num_pts};
    concrete.boundary_field_time_derivatives(
        make_not_null(&dt_zero), zero_mesh_vel, data.normal_covector,
        data.boundary_psi_value, data.interior_pi, data.interior_phi,
        data.coords, time);
    CHECK_ITERABLE_APPROX(dt_zero, dt_null);
  };

  check(dynamic_cast<const SecondOrderScalarWave::BoundaryConditions::
                         DirichletCharacteristics<Dim>&>(*boundary_condition));
  const auto deserialized = serialize_and_deserialize(boundary_condition);
  check(dynamic_cast<const SecondOrderScalarWave::BoundaryConditions::
                         DirichletCharacteristics<Dim>&>(*deserialized));
}

// A closed-form 2D pin (ZeroIncomingMode, so data = 0 and no analytic
// evaluation enters) that fixes the mask logic, the superluminal v^+ flip, and
// the advection term independently of the python reference. All three points
// share n = (1, 0), interior_pi = 2, interior_phi = (3, 4), so n.phi = 3 and
// the interior modes are v^+ = 5, v^- = -1, v^0 = (0, 4). The mesh velocity
// differs per point:
//   p0 v=(0.5,0):  lambda=(-0.5, 0.5,-1.5) -> v^0 data, v^+ int, v^- data
//                  -> Pi = 2.5, Phi = (2.5, 0),  dt = -2.5 + 0.5*2.5 = -1.25
//   p1 v=(-0.5,0): lambda=( 0.5, 1.5,-0.5) -> v^0 int,  v^+ int, v^- data
//                  -> Pi = 2.5, Phi = (2.5, 4),  dt = -2.5 - 0.5*2.5 = -3.75
//   p2 v=(1.5,0):  lambda=(-1.5,-0.5,-2.5) -> all modes from data (0)
//                  -> Pi = 0,   Phi = (0, 0),    dt = 0
// The ghost Psi is the passed boundary value at every point.
void test_hand_computed_pin() {
  constexpr size_t Dim = 2;
  const size_t num_pts = 3;
  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition = TestHelpers::test_creation<
          std::unique_ptr<SecondOrderScalarWave::BoundaryConditions::
                              BoundaryCondition<Dim>>,
          Metavariables<Dim>>(yaml_string<Dim>(true));
  const auto& concrete =
      dynamic_cast<const SecondOrderScalarWave::BoundaryConditions::
                       DirichletCharacteristics<Dim>&>(*boundary_condition);

  tnsr::i<DataVector, Dim, Frame::Inertial> normal_covector{
      {{DataVector{1.0, 1.0, 1.0}, DataVector{0.0, 0.0, 0.0}}}};
  const Scalar<DataVector> interior_pi{DataVector{2.0, 2.0, 2.0}};
  const tnsr::i<DataVector, Dim, Frame::Inertial> interior_phi{
      {{DataVector{3.0, 3.0, 3.0}, DataVector{4.0, 4.0, 4.0}}}};
  const Scalar<DataVector> boundary_psi_value{DataVector{7.0, 7.0, 7.0}};
  const tnsr::I<DataVector, Dim, Frame::Inertial> coords{
      {{DataVector{0.0, 0.0, 0.0}, DataVector{0.0, 0.0, 0.0}}}};
  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> mesh_velocity{
      tnsr::I<DataVector, Dim, Frame::Inertial>{
          {{DataVector{0.5, -0.5, 1.5}, DataVector{0.0, 0.0, 0.0}}}}};
  const double time = 0.5;

  Scalar<DataVector> psi{num_pts};
  Scalar<DataVector> pi{num_pts};
  tnsr::i<DataVector, Dim, Frame::Inertial> phi{num_pts};
  const auto error = concrete.dg_ghost(make_not_null(&psi), make_not_null(&pi),
                                       make_not_null(&phi), mesh_velocity,
                                       normal_covector, boundary_psi_value,
                                       interior_pi, interior_phi, coords, time);
  CHECK_FALSE(error.has_value());

  const Scalar<DataVector> expected_psi{DataVector{7.0, 7.0, 7.0}};
  const Scalar<DataVector> expected_pi{DataVector{2.5, 2.5, 0.0}};
  const tnsr::i<DataVector, Dim, Frame::Inertial> expected_phi{
      {{DataVector{2.5, 2.5, 0.0}, DataVector{0.0, 4.0, 0.0}}}};
  CHECK_ITERABLE_APPROX(psi, expected_psi);
  CHECK_ITERABLE_APPROX(pi, expected_pi);
  CHECK_ITERABLE_APPROX(phi, expected_phi);

  Scalar<DataVector> dt_boundary_psi{num_pts};
  const auto dt_error = concrete.boundary_field_time_derivatives(
      make_not_null(&dt_boundary_psi), mesh_velocity, normal_covector,
      boundary_psi_value, interior_pi, interior_phi, coords, time);
  CHECK_FALSE(dt_error.has_value());
  const Scalar<DataVector> expected_dt{DataVector{-1.25, -3.75, 0.0}};
  CHECK_ITERABLE_APPROX(dt_boundary_psi, expected_dt);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.SecondOrderScalarWave.BoundaryConditions.DirichletCharacteristics",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/"};
  MAKE_GENERATOR(gen);

  // Register the boundary condition and the analytic-prescription classes it
  // owns so the serialize/deserialize round-trip dispatches correctly.
  PUPable_reg(SINGLE_ARG(
      SecondOrderScalarWave::BoundaryConditions::DirichletCharacteristics<1>));
  PUPable_reg(SINGLE_ARG(
      SecondOrderScalarWave::BoundaryConditions::DirichletCharacteristics<2>));
  PUPable_reg(SINGLE_ARG(
      SecondOrderScalarWave::BoundaryConditions::DirichletCharacteristics<3>));
  register_classes_with_charm(
      SecondOrderScalarWave::Solutions::all_solutions<1>{});
  register_classes_with_charm(
      SecondOrderScalarWave::Solutions::all_solutions<2>{});
  register_classes_with_charm(
      SecondOrderScalarWave::Solutions::all_solutions<3>{});
  register_classes_with_charm(
      MathFunctions::all_math_functions<1, Frame::Inertial>{});

  const auto test_all_dims = [&gen](const bool zero_incoming,
                                    const std::string& suffix) {
    test_dg_ghost<1>(make_not_null(&gen), zero_incoming, suffix);
    test_dg_ghost<2>(make_not_null(&gen), zero_incoming, suffix);
    test_dg_ghost<3>(make_not_null(&gen), zero_incoming, suffix);
    test_boundary_field_time_derivatives<1>(make_not_null(&gen), zero_incoming,
                                            suffix);
    test_boundary_field_time_derivatives<2>(make_not_null(&gen), zero_incoming,
                                            suffix);
    test_boundary_field_time_derivatives<3>(make_not_null(&gen), zero_incoming,
                                            suffix);
  };

  // ZeroIncomingMode = false (incoming v^- from the analytic data)
  test_all_dims(false, "_keep_zero");
  // ZeroIncomingMode = true (incoming v^- set to zero)
  test_all_dims(true, "_zero_incoming");

  // Closed-form 2D pin of the moving-mesh mask logic and advection term.
  test_hand_computed_pin();
}
