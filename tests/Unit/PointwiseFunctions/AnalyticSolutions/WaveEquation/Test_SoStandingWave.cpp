// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SoStandingWave.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<evolution::initial_data::InitialData,
                   tmpl::list<SoScalarWave::Solutions::SoStandingWave<Dim>>>>;
  };
};

// Evaluate u = k . (x - x0)
template <size_t Dim>
DataVector coordinate_phase(const tnsr::I<DataVector, Dim>& x,
                            const std::array<double, Dim>& wave_vector,
                            const std::array<double, Dim>& center) {
  auto u = make_with_value<DataVector>(get<0>(x), 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    u += gsl::at(wave_vector, d) * (x.get(d) - gsl::at(center, d));
  }
  return u;
}

template <size_t Dim>
void check_solution(const SoScalarWave::Solutions::SoStandingWave<Dim>& sw,
                    const tnsr::I<DataVector, Dim>& x,
                    const std::array<double, Dim>& wave_vector,
                    const std::array<double, Dim>& center,
                    const double amplitude, const double t) {
  CHECK_FALSE(sw != sw);
  test_copy_semantics(sw);
  auto sw_for_move = sw;
  test_move_semantics(std::move(sw_for_move), sw);

  const double omega = magnitude(wave_vector);
  const DataVector u = coordinate_phase(x, wave_vector, center);
  const DataVector sin_u = sin(u);
  const DataVector cos_u = cos(u);
  const double cos_wt = cos(omega * t);
  const double sin_wt = sin(omega * t);

  // Expected evolution variables:
  //   Psi   =  A sin(u) cos(wt)
  //   Pi    = -dt Psi = A w sin(u) sin(wt)
  //   Phi_i =  di Psi = A k_i cos(u) cos(wt)
  const DataVector expected_psi = amplitude * sin_u * cos_wt;
  const DataVector expected_pi = amplitude * omega * sin_u * sin_wt;

  const auto vars =
      sw.variables(x, t,
                   tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                              SoScalarWave::Tags::Phi<Dim>>{});
  CHECK_ITERABLE_APPROX(get(get<SoScalarWave::Tags::Psi>(vars)), expected_psi);
  CHECK_ITERABLE_APPROX(get(get<SoScalarWave::Tags::Pi>(vars)), expected_pi);
  const auto& phi = get<SoScalarWave::Tags::Phi<Dim>>(vars);
  for (size_t d = 0; d < Dim; ++d) {
    CHECK_ITERABLE_APPROX(phi.get(d),
                          amplitude * gsl::at(wave_vector, d) * cos_u * cos_wt);
  }

  // The LDG evolved variables {Psi, Pi} must match the {Psi, Pi, Phi} overload
  const auto ldg_vars = sw.variables(
      x, t, tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi>{});
  CHECK_ITERABLE_APPROX(get(get<SoScalarWave::Tags::Psi>(ldg_vars)),
                        get(get<SoScalarWave::Tags::Psi>(vars)));
  CHECK_ITERABLE_APPROX(get(get<SoScalarWave::Tags::Pi>(ldg_vars)),
                        get(get<SoScalarWave::Tags::Pi>(vars)));

  // Expected time derivatives:
  //   dt Psi   = -A w sin(u) sin(wt)
  //   dt Pi    =  A w^2 sin(u) cos(wt)
  //   dt Phi_i = -A w k_i cos(u) sin(wt)
  const auto dt_vars =
      sw.variables(x, t,
                   tmpl::list<Tags::dt<SoScalarWave::Tags::Psi>,
                              Tags::dt<SoScalarWave::Tags::Pi>,
                              Tags::dt<SoScalarWave::Tags::Phi<Dim>>>{});
  CHECK_ITERABLE_APPROX(get(get<Tags::dt<SoScalarWave::Tags::Psi>>(dt_vars)),
                        -amplitude * omega * sin_u * sin_wt);
  CHECK_ITERABLE_APPROX(get(get<Tags::dt<SoScalarWave::Tags::Pi>>(dt_vars)),
                        amplitude * square(omega) * sin_u * cos_wt);
  const auto& dt_phi = get<Tags::dt<SoScalarWave::Tags::Phi<Dim>>>(dt_vars);
  for (size_t d = 0; d < Dim; ++d) {
    CHECK_ITERABLE_APPROX(
        dt_phi.get(d),
        -amplitude * omega * gsl::at(wave_vector, d) * cos_u * sin_wt);
  }

  // dt Psi must be the negative of Pi (Pi = -dt Psi by convention)
  CHECK_ITERABLE_APPROX(get(get<Tags::dt<SoScalarWave::Tags::Psi>>(dt_vars)),
                        -get(get<SoScalarWave::Tags::Pi>(vars)));
}

// At t = 0 the standing wave has Pi = 0 everywhere.
template <size_t Dim>
void check_pi_zero_at_t0(const SoScalarWave::Solutions::SoStandingWave<Dim>& sw,
                         const tnsr::I<DataVector, Dim>& x) {
  const auto vars =
      sw.variables(x, 0.0,
                   tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                              SoScalarWave::Tags::Phi<Dim>>{});
  const auto expected_pi = make_with_value<Scalar<DataVector>>(get<0>(x), 0.0);
  CHECK_ITERABLE_APPROX(get(get<SoScalarWave::Tags::Pi>(vars)),
                        get(expected_pi));
}

template <size_t Dim>
void test_serialization_and_creation(
    const SoScalarWave::Solutions::SoStandingWave<Dim>& sw,
    const tnsr::I<DataVector, Dim>& x, const std::string& options_string,
    const double t) {
  register_factory_classes_with_charm<Metavariables<Dim>>();
  const auto deserialized_sw = serialize_and_deserialize(sw);
  CHECK(deserialized_sw == sw);

  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag<
          evolution::initial_data::OptionTags::InitialData, Metavariables<Dim>>(
          options_string)
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& created_solution =
      dynamic_cast<const SoScalarWave::Solutions::SoStandingWave<Dim>&>(
          *deserialized_option_solution);

  CHECK(created_solution.variables(
            x, t,
            tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                       SoScalarWave::Tags::Phi<Dim>>{}) ==
        sw.variables(x, t,
                     tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                                SoScalarWave::Tags::Phi<Dim>>{}));
}

void test_1d() {
  const std::array<double, 1> wave_vector{{-1.5}};
  const std::array<double, 1> center{{2.4}};
  const double amplitude = 0.7;
  const double t = 3.1;
  const tnsr::I<DataVector, 1> x(DataVector({-0.2, 8.7}));
  const SoScalarWave::Solutions::SoStandingWave<1> sw(wave_vector, center,
                                                      amplitude);
  check_solution<1>(sw, x, wave_vector, center, amplitude, t);
  check_pi_zero_at_t0<1>(sw, x);
  test_serialization_and_creation<1>(sw, x,
                                     "SoStandingWave:\n"
                                     "  WaveVector: [-1.5]\n"
                                     "  Center: [2.4]\n"
                                     "  Amplitude: 0.7",
                                     t);
}

void test_2d() {
  const std::array<double, 2> wave_vector{{1.5, -7.2}};
  const std::array<double, 2> center{{2.4, -4.8}};
  const double amplitude = 0.7;
  const double t = 3.1;
  const tnsr::I<DataVector, 2> x{std::array<DataVector, 2>{
      {DataVector({-10.2, 8.7}), DataVector({-1.98, 48.27})}}};
  const SoScalarWave::Solutions::SoStandingWave<2> sw(wave_vector, center,
                                                      amplitude);
  check_solution<2>(sw, x, wave_vector, center, amplitude, t);
  check_pi_zero_at_t0<2>(sw, x);
  test_serialization_and_creation<2>(sw, x,
                                     "SoStandingWave:\n"
                                     "  WaveVector: [1.5, -7.2]\n"
                                     "  Center: [2.4, -4.8]\n"
                                     "  Amplitude: 0.7",
                                     t);
}

void test_3d() {
  const std::array<double, 3> wave_vector{{1.5, -7.2, 2.7}};
  const std::array<double, 3> center{{2.4, -4.8, 8.4}};
  const double amplitude = 0.7;
  const double t = 3.1;
  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({-10.2, 8.7}), DataVector({-1.98, 48.27}),
       DataVector({2.2, 1.1})}}};
  const SoScalarWave::Solutions::SoStandingWave<3> sw(wave_vector, center,
                                                      amplitude);
  check_solution<3>(sw, x, wave_vector, center, amplitude, t);
  check_pi_zero_at_t0<3>(sw, x);
  test_serialization_and_creation<3>(sw, x,
                                     "SoStandingWave:\n"
                                     "  WaveVector: [1.5, -7.2, 2.7]\n"
                                     "  Center: [2.4, -4.8, 8.4]\n"
                                     "  Amplitude: 0.7",
                                     t);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.WaveEquation.SoStandingWave",
    "[PointwiseFunctions][Unit]") {
  test_1d();
  test_2d();
  test_3d();
}
