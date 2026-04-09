// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/DirichletCharacteristics.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryCorrections/LaxFriedrichs.hpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SoPlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<SoScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
                   tmpl::list<SoScalarWave::BoundaryConditions::
                                  DirichletCharacteristics<Dim>,
                              SoScalarWave::BoundaryConditions::
                                  DirichletAnalytic<Dim>>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   SoScalarWave::Solutions::all_solutions<Dim>>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>>;
  };
};

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
std::string yaml_string() {
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
         "          Center: 0.0\n";
}

template <size_t Dim>
void test_prescribe_zero() {
  register_classes_with_charm(SoScalarWave::Solutions::all_solutions<Dim>{});
  register_classes_with_charm(
      MathFunctions::all_math_functions<1, Frame::Inertial>{});
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time,
      Tags::AnalyticSolution<SoScalarWave::Solutions::SoPlaneWave<Dim>>>>(
      0.5, ConvertPlaneWave<Dim>::create_container());

  helpers::test_boundary_condition_with_python<
      SoScalarWave::BoundaryConditions::DirichletCharacteristics<Dim>,
      SoScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
      SoScalarWave::System<Dim>,
      tmpl::list<SoScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>>,
      tmpl::list<ConvertPlaneWave<Dim>>,
      tmpl::list<
          Tags::AnalyticSolution<SoScalarWave::Solutions::SoPlaneWave<Dim>>>,
      Metavariables<Dim>>(
      make_not_null(&gen), "DirichletCharacteristics",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<SoScalarWave::Tags::Psi>,
          helpers::Tags::PythonFunctionName<SoScalarWave::Tags::Pi>,
          helpers::Tags::PythonFunctionName<SoScalarWave::Tags::Phi<Dim>>>{
          "error", "psi_prescribe_zero", "pi_prescribe_zero",
          "phi_prescribe_zero"},
      yaml_string<Dim>() + "  PrescribeZeroSpeedModes: true\n",
      Index<Dim - 1>{Dim == 1 ? 1 : 5}, box_analytic_soln,
      tuples::TaggedTuple<>{});
}

template <size_t Dim>
void test_keep_zero() {
  register_classes_with_charm(SoScalarWave::Solutions::all_solutions<Dim>{});
  register_classes_with_charm(
      MathFunctions::all_math_functions<1, Frame::Inertial>{});
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time,
      Tags::AnalyticSolution<SoScalarWave::Solutions::SoPlaneWave<Dim>>>>(
      0.5, ConvertPlaneWave<Dim>::create_container());

  helpers::test_boundary_condition_with_python<
      SoScalarWave::BoundaryConditions::DirichletCharacteristics<Dim>,
      SoScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
      SoScalarWave::System<Dim>,
      tmpl::list<SoScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>>,
      tmpl::list<ConvertPlaneWave<Dim>>,
      tmpl::list<
          Tags::AnalyticSolution<SoScalarWave::Solutions::SoPlaneWave<Dim>>>,
      Metavariables<Dim>>(
      make_not_null(&gen), "DirichletCharacteristics",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<SoScalarWave::Tags::Psi>,
          helpers::Tags::PythonFunctionName<SoScalarWave::Tags::Pi>,
          helpers::Tags::PythonFunctionName<SoScalarWave::Tags::Phi<Dim>>>{
          "error", "psi_keep_zero", "pi_keep_zero", "phi_keep_zero"},
      yaml_string<Dim>() + "  PrescribeZeroSpeedModes: false\n",
      Index<Dim - 1>{Dim == 1 ? 1 : 5}, box_analytic_soln,
      tuples::TaggedTuple<>{});
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.SoScalarWave.BoundaryConditions.DirichletCharacteristics",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/SoScalarWave/BoundaryConditions/"};
  test_prescribe_zero<1>();
  test_prescribe_zero<2>();
  test_prescribe_zero<3>();
  test_keep_zero<1>();
  test_keep_zero<2>();
  test_keep_zero<3>();
}
