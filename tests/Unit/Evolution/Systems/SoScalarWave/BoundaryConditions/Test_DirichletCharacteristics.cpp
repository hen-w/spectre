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
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/DirichletCharacteristics.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SoPlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

// `DirichletCharacteristics` is now a `Type::Ghost` boundary condition (it no
// longer supplies a volume `dg_time_derivative`), and its `dg_ghost` takes the
// current per-face boundary-evolved value as an extra argument spliced in
// after the normal covector. The generic
// `TestHelpers::evolution::dg::test_boundary_condition_with_python` helper
// knows only the fixed `dg_ghost`/`dg_time_derivative` calling conventions, not
// the extra boundary-value argument nor the `boundary_field_time_derivatives`
// method, so it cannot drive either method here. Instead this test calls both
// methods directly and compares against the same python reference the helper
// would have used.
//
// Coverage accounting vs. the old (helper-driven) test:
//   * The old `dg_ghost` cross-check (psi/pi/phi outputs vs python, both
//     moving and non-moving mesh, plus serialize/deserialize) is reproduced
//     directly below in `test_dg_ghost`, with the ghost-Psi selection now
//     sourced from the passed boundary value in the default branch.
//   * The old `dg_time_derivative` cross-check (dt of Psi/Pi/BoundaryPsi vs
//     python) is replaced by `test_boundary_field_time_derivatives`, a direct
//     comparison of `dt = -0.5 (v^+ + v^-) = -Pi_boundary` (or 0 when Psi is
//     copied from the interior) against the `dt_boundary_psi_*` python
//     references. The old dt path also asserted dt_psi = dt_pi = 0 (no volume
//     correction); the boundary-evolved facility removes that path entirely
//     (the boundary field is never lifted into the volume dt -- statically
//     asserted in `Test_BoundaryConditions.cpp`), so there is nothing
//     analogous to check here.
//   * The delivery of the stored boundary value into `dg_ghost`, and the write
//     of the produced dt into the per-face dt-stash, are exercised by the
//     facility plumbing tests in `Test_BoundaryConditions.cpp` at the
//     `apply_boundary_conditions_on_all_external_faces` level.

namespace {
template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            SoScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
            tmpl::list<
                SoScalarWave::BoundaryConditions::DirichletCharacteristics<Dim>,
                SoScalarWave::BoundaryConditions::DirichletAnalytic<Dim>>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   SoScalarWave::Solutions::all_solutions<Dim>>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>>;
  };
};

// Converts the `SoPlaneWave` analytic solution passed to `dg_ghost` /
// `boundary_field_time_derivatives` into the trailing `dim` python argument,
// matching the python reference's signature (the reference reconstructs the
// solution from `dim` alone since all its parameters are fixed).
template <size_t Dim>
struct ConvertPlaneWave {
  using unpacked_container = int;
  using packed_container = SoScalarWave::Solutions::SoPlaneWave<Dim>;
  using packed_type = double;

  static packed_container create_container() {
    std::array<double, Dim> wave_vector{};
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(wave_vector, i) = 0.1 + i;
    }
    std::array<double, Dim> center{};
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(center, i) = 1.1 - i;
    }
    return {wave_vector, center,
            std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                0.9, 0.6, 0.0)};
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
std::string yaml_string(const bool prescribe_zero, const bool copy_psi,
                        const bool zero_incoming) {
  return "DirichletCharacteristics:\n"
         "  AnalyticPrescription:\n"
         "    SoPlaneWave:\n"
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
         "  PrescribeZeroSpeedModes: " +
         (prescribe_zero ? "true\n" : "false\n") +
         "  CopyPsiFromInterior: " + (copy_psi ? "true\n" : "false\n") +
         "  ZeroIncomingMode: " + (zero_incoming ? "true\n" : "false\n");
}

// A random, unit-normalized covector and optional mesh velocity, matching the
// setup the generic helper builds.
template <size_t Dim>
struct FaceData {
  Scalar<DataVector> interior_psi;
  Scalar<DataVector> interior_pi;
  tnsr::i<DataVector, Dim, Frame::Inertial> interior_phi;
  Scalar<DataVector> boundary_psi_value;
  tnsr::i<DataVector, Dim, Frame::Inertial> normal_covector;
  tnsr::I<DataVector, Dim, Frame::Inertial> coords;
  std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> face_mesh_velocity;
};

template <size_t Dim>
FaceData<Dim> make_face_data(const gsl::not_null<std::mt19937*> gen,
                             const size_t num_pts, const bool use_moving_mesh) {
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  FaceData<Dim> data{};
  data.interior_psi = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&dist), num_pts);
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
  if (use_moving_mesh) {
    std::uniform_real_distribution<> local_dist(-2.0, 2.0);
    data.face_mesh_velocity =
        make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
            gen, make_not_null(&local_dist), num_pts);
  }
  return data;
}

template <size_t Dim>
void test_dg_ghost(const gsl::not_null<std::mt19937*> gen,
                   const bool prescribe_zero, const bool copy_psi,
                   const bool zero_incoming, const std::string& suffix) {
  CAPTURE(Dim);
  CAPTURE(prescribe_zero);
  CAPTURE(copy_psi);
  CAPTURE(zero_incoming);
  const size_t num_pts = Dim == 1 ? 1 : 5;

  // Upcast to the domain boundary-condition base so serialization dispatches
  // through the registered PUPable interface (as the generic helper does).
  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition = TestHelpers::test_creation<
          std::unique_ptr<
              SoScalarWave::BoundaryConditions::BoundaryCondition<Dim>>,
          Metavariables<Dim>>(
          yaml_string<Dim>(prescribe_zero, copy_psi, zero_incoming));
  const auto solution = ConvertPlaneWave<Dim>::create_container();

  for (const bool use_moving_mesh : {false, true}) {
    CAPTURE(use_moving_mesh);
    const auto data = make_face_data<Dim>(gen, num_pts, use_moving_mesh);
    const double time = 0.5;

    const auto check =
        [&data, &solution, &suffix, num_pts,
         time](const SoScalarWave::BoundaryConditions::DirichletCharacteristics<
               Dim>& concrete) {
          Scalar<DataVector> psi{num_pts};
          Scalar<DataVector> pi{num_pts};
          tnsr::i<DataVector, Dim, Frame::Inertial> phi{num_pts};
          const auto error = concrete.dg_ghost(
              make_not_null(&psi), make_not_null(&pi), make_not_null(&phi),
              data.face_mesh_velocity, data.normal_covector,
              data.boundary_psi_value, data.interior_psi, data.interior_pi,
              data.interior_phi, data.coords, time);
          CHECK_FALSE(error.has_value());

          const auto expected_psi =
              pypp::call<Scalar<DataVector>, tmpl::list<ConvertPlaneWave<Dim>>>(
                  "DirichletCharacteristics", "psi" + suffix,
                  data.face_mesh_velocity, data.normal_covector,
                  data.interior_psi, data.interior_pi, data.interior_phi,
                  data.boundary_psi_value, data.coords, time, solution);
          const auto expected_pi =
              pypp::call<Scalar<DataVector>, tmpl::list<ConvertPlaneWave<Dim>>>(
                  "DirichletCharacteristics", "pi" + suffix,
                  data.face_mesh_velocity, data.normal_covector,
                  data.interior_psi, data.interior_pi, data.interior_phi,
                  data.boundary_psi_value, data.coords, time, solution);
          const auto expected_phi =
              pypp::call<tnsr::i<DataVector, Dim, Frame::Inertial>,
                         tmpl::list<ConvertPlaneWave<Dim>>>(
                  "DirichletCharacteristics", "phi" + suffix,
                  data.face_mesh_velocity, data.normal_covector,
                  data.interior_psi, data.interior_pi, data.interior_phi,
                  data.boundary_psi_value, data.coords, time, solution);
          CHECK_ITERABLE_APPROX(psi, expected_psi);
          CHECK_ITERABLE_APPROX(pi, expected_pi);
          CHECK_ITERABLE_APPROX(phi, expected_phi);
        };

    // Run against both the freshly-created and a serialized/deserialized
    // boundary condition (mirrors the generic helper's serialization check).
    check(
        dynamic_cast<const SoScalarWave::BoundaryConditions::
                         DirichletCharacteristics<Dim>&>(*boundary_condition));
    const auto deserialized = serialize_and_deserialize(boundary_condition);
    check(dynamic_cast<const SoScalarWave::BoundaryConditions::
                           DirichletCharacteristics<Dim>&>(*deserialized));
  }
}

template <size_t Dim>
void test_boundary_field_time_derivatives(
    const gsl::not_null<std::mt19937*> gen, const bool prescribe_zero,
    const bool copy_psi, const bool zero_incoming, const std::string& suffix) {
  CAPTURE(Dim);
  CAPTURE(prescribe_zero);
  CAPTURE(copy_psi);
  CAPTURE(zero_incoming);
  const size_t num_pts = Dim == 1 ? 1 : 5;

  // Upcast to the domain boundary-condition base so serialization dispatches
  // through the registered PUPable interface (as the generic helper does).
  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition = TestHelpers::test_creation<
          std::unique_ptr<
              SoScalarWave::BoundaryConditions::BoundaryCondition<Dim>>,
          Metavariables<Dim>>(
          yaml_string<Dim>(prescribe_zero, copy_psi, zero_incoming));
  const auto solution = ConvertPlaneWave<Dim>::create_container();

  for (const bool use_moving_mesh : {false, true}) {
    CAPTURE(use_moving_mesh);
    const auto data = make_face_data<Dim>(gen, num_pts, use_moving_mesh);
    const double time = 0.5;

    const auto check =
        [&data, &solution, &suffix, num_pts,
         time](const SoScalarWave::BoundaryConditions::DirichletCharacteristics<
               Dim>& concrete) {
          Scalar<DataVector> dt_boundary_psi{num_pts};
          const auto error = concrete.boundary_field_time_derivatives(
              make_not_null(&dt_boundary_psi), data.face_mesh_velocity,
              data.normal_covector, data.boundary_psi_value, data.interior_psi,
              data.interior_pi, data.interior_phi, data.coords, time);
          CHECK_FALSE(error.has_value());

          // dt = -0.5 (v^+ + v^-) = -Pi_boundary (or 0 when Psi is copied from
          // the interior).
          const auto expected_dt =
              pypp::call<Scalar<DataVector>, tmpl::list<ConvertPlaneWave<Dim>>>(
                  "DirichletCharacteristics", "dt_boundary_psi" + suffix,
                  data.face_mesh_velocity, data.normal_covector,
                  data.interior_psi, data.interior_pi, data.interior_phi,
                  data.boundary_psi_value, data.coords, time, solution);
          CHECK_ITERABLE_APPROX(dt_boundary_psi, expected_dt);
        };

    check(
        dynamic_cast<const SoScalarWave::BoundaryConditions::
                         DirichletCharacteristics<Dim>&>(*boundary_condition));
    const auto deserialized = serialize_and_deserialize(boundary_condition);
    check(dynamic_cast<const SoScalarWave::BoundaryConditions::
                           DirichletCharacteristics<Dim>&>(*deserialized));
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.SoScalarWave.BoundaryConditions.DirichletCharacteristics",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/SoScalarWave/BoundaryConditions/"};
  MAKE_GENERATOR(gen);

  // Register the boundary condition and the analytic-prescription classes it
  // owns so the serialize/deserialize round-trip dispatches correctly.
  PUPable_reg(SINGLE_ARG(
      SoScalarWave::BoundaryConditions::DirichletCharacteristics<1>));
  PUPable_reg(SINGLE_ARG(
      SoScalarWave::BoundaryConditions::DirichletCharacteristics<2>));
  PUPable_reg(SINGLE_ARG(
      SoScalarWave::BoundaryConditions::DirichletCharacteristics<3>));
  register_classes_with_charm(SoScalarWave::Solutions::all_solutions<1>{});
  register_classes_with_charm(SoScalarWave::Solutions::all_solutions<2>{});
  register_classes_with_charm(SoScalarWave::Solutions::all_solutions<3>{});
  register_classes_with_charm(
      MathFunctions::all_math_functions<1, Frame::Inertial>{});

  const auto test_all_dims = [&gen](const bool prescribe_zero,
                                    const bool copy_psi,
                                    const bool zero_incoming,
                                    const std::string& suffix) {
    test_dg_ghost<1>(make_not_null(&gen), prescribe_zero, copy_psi,
                     zero_incoming, suffix);
    test_dg_ghost<2>(make_not_null(&gen), prescribe_zero, copy_psi,
                     zero_incoming, suffix);
    test_dg_ghost<3>(make_not_null(&gen), prescribe_zero, copy_psi,
                     zero_incoming, suffix);
    test_boundary_field_time_derivatives<1>(make_not_null(&gen), prescribe_zero,
                                            copy_psi, zero_incoming, suffix);
    test_boundary_field_time_derivatives<2>(make_not_null(&gen), prescribe_zero,
                                            copy_psi, zero_incoming, suffix);
    test_boundary_field_time_derivatives<3>(make_not_null(&gen), prescribe_zero,
                                            copy_psi, zero_incoming, suffix);
  };

  // PrescribeZeroSpeedModes = true
  test_all_dims(true, false, false, "_prescribe_zero");
  // PrescribeZeroSpeedModes = false (ghost Psi = the boundary-evolved value)
  test_all_dims(false, false, false, "_keep_zero");
  // CopyPsiFromInterior = true (ghost Psi = interior Psi, dt = 0)
  test_all_dims(false, true, false, "_copy_interior");
  // ZeroIncomingMode = true
  test_all_dims(false, false, true, "_zero_incoming");
}
