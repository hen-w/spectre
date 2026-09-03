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
//     characteristic field): v^0_i (speed 0), v^+ (speed +1, always outgoing),
//     v^- (speed -1, always incoming). The python reference implements the
//     ghost formulas directly from the raw inputs.
//   * Moving meshes are no longer supported: both methods now return an error
//     string when a face mesh velocity is supplied. The pypp cross-check runs
//     only for the non-moving case; a separate check confirms the error is
//     returned when a mesh velocity is engaged.
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

constexpr char moving_mesh_error[] =
    "DirichletCharacteristics does not support moving meshes: the "
    "characteristic speeds are defined without a mesh velocity.";

template <size_t Dim>
void test_dg_ghost(const gsl::not_null<std::mt19937*> gen,
                   const bool zero_incoming, const std::string& suffix) {
  CAPTURE(Dim);
  CAPTURE(zero_incoming);
  const size_t num_pts = Dim == 1 ? 1 : 5;

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

  const auto check = [&data, &solution, &suffix, &no_mesh_vel, num_pts,
                      time](const SecondOrderScalarWave::BoundaryConditions::
                                DirichletCharacteristics<Dim>& concrete) {
    Scalar<DataVector> psi{num_pts};
    Scalar<DataVector> pi{num_pts};
    tnsr::i<DataVector, Dim, Frame::Inertial> phi{num_pts};
    const auto error = concrete.dg_ghost(
        make_not_null(&psi), make_not_null(&pi), make_not_null(&phi),
        no_mesh_vel, data.normal_covector, data.boundary_psi_value,
        data.interior_pi, data.interior_phi, data.coords, time);
    CHECK_FALSE(error.has_value());

    const auto expected_psi =
        pypp::call<Scalar<DataVector>, tmpl::list<ConvertPlaneWave<Dim>>>(
            "DirichletCharacteristics", "psi" + suffix, no_mesh_vel,
            data.normal_covector, data.interior_pi, data.interior_phi,
            data.boundary_psi_value, data.coords, time, solution);
    const auto expected_pi =
        pypp::call<Scalar<DataVector>, tmpl::list<ConvertPlaneWave<Dim>>>(
            "DirichletCharacteristics", "pi" + suffix, no_mesh_vel,
            data.normal_covector, data.interior_pi, data.interior_phi,
            data.boundary_psi_value, data.coords, time, solution);
    const auto expected_phi =
        pypp::call<tnsr::i<DataVector, Dim, Frame::Inertial>,
                   tmpl::list<ConvertPlaneWave<Dim>>>(
            "DirichletCharacteristics", "phi" + suffix, no_mesh_vel,
            data.normal_covector, data.interior_pi, data.interior_phi,
            data.boundary_psi_value, data.coords, time, solution);
    CHECK_ITERABLE_APPROX(psi, expected_psi);
    CHECK_ITERABLE_APPROX(pi, expected_pi);
    CHECK_ITERABLE_APPROX(phi, expected_phi);

    // A moving mesh is not supported: the method must return the error.
    std::uniform_real_distribution<> dist(-2.0, 2.0);
    std::mt19937 local_gen{};
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
        mesh_velocity =
            make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
                make_not_null(&local_gen), make_not_null(&dist), num_pts);
    const auto moving_error = concrete.dg_ghost(
        make_not_null(&psi), make_not_null(&pi), make_not_null(&phi),
        mesh_velocity, data.normal_covector, data.boundary_psi_value,
        data.interior_pi, data.interior_phi, data.coords, time);
    REQUIRE(moving_error.has_value());
    CHECK(moving_error.value() == moving_mesh_error);
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
  const size_t num_pts = Dim == 1 ? 1 : 5;

  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition = TestHelpers::test_creation<
          std::unique_ptr<SecondOrderScalarWave::BoundaryConditions::
                              BoundaryCondition<Dim>>,
          Metavariables<Dim>>(yaml_string<Dim>(zero_incoming));

  const auto data = make_face_data<Dim>(gen, num_pts);
  const auto solution = ConvertPlaneWave<Dim>::create_container();
  const double time = 0.5;
  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> no_mesh_vel{};

  const auto check = [&data, &solution, &suffix, &no_mesh_vel, num_pts,
                      time](const SecondOrderScalarWave::BoundaryConditions::
                                DirichletCharacteristics<Dim>& concrete) {
    Scalar<DataVector> dt_boundary_psi{num_pts};
    const auto error = concrete.boundary_field_time_derivatives(
        make_not_null(&dt_boundary_psi), no_mesh_vel, data.normal_covector,
        data.boundary_psi_value, data.interior_pi, data.interior_phi,
        data.coords, time);
    CHECK_FALSE(error.has_value());

    // dt = -0.5 (interior v^+ + analytic v^- [or 0 with ZeroIncomingMode]).
    const auto expected_dt =
        pypp::call<Scalar<DataVector>, tmpl::list<ConvertPlaneWave<Dim>>>(
            "DirichletCharacteristics", "dt_boundary_psi" + suffix, no_mesh_vel,
            data.normal_covector, data.interior_pi, data.interior_phi,
            data.boundary_psi_value, data.coords, time, solution);
    CHECK_ITERABLE_APPROX(dt_boundary_psi, expected_dt);

    // A moving mesh is not supported: the method must return the error.
    std::uniform_real_distribution<> dist(-2.0, 2.0);
    std::mt19937 local_gen{};
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
        mesh_velocity =
            make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
                make_not_null(&local_gen), make_not_null(&dist), num_pts);
    const auto moving_error = concrete.boundary_field_time_derivatives(
        make_not_null(&dt_boundary_psi), mesh_velocity, data.normal_covector,
        data.boundary_psi_value, data.interior_pi, data.interior_phi,
        data.coords, time);
    REQUIRE(moving_error.has_value());
    CHECK(moving_error.value() == moving_mesh_error);
  };

  check(dynamic_cast<const SecondOrderScalarWave::BoundaryConditions::
                         DirichletCharacteristics<Dim>&>(*boundary_condition));
  const auto deserialized = serialize_and_deserialize(boundary_condition);
  check(dynamic_cast<const SecondOrderScalarWave::BoundaryConditions::
                         DirichletCharacteristics<Dim>&>(*deserialized));
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
}
