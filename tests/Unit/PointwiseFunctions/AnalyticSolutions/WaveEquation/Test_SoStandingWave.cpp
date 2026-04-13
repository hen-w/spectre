// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SoStandingWave.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
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

template <size_t Dim>
void test() {
  // Parameters
  std::array<double, Dim> wave_vector{};
  std::array<double, Dim> center{};
  double k_sq = 0.0;
  // Use values that are exactly representable so YAML round-trip is exact
  gsl::at(wave_vector, 0) = 1.5;
  gsl::at(center, 0) = 0.5;
  if constexpr (Dim >= 2) {
    gsl::at(wave_vector, 1) = -2.25;
    gsl::at(center, 1) = 1.75;
  }
  if constexpr (Dim >= 3) {
    gsl::at(wave_vector, 2) = 0.75;
    gsl::at(center, 2) = -0.5;
  }
  for (size_t d = 0; d < Dim; ++d) {
    k_sq += square(gsl::at(wave_vector, d));
  }
  const double amplitude = 2.5;
  const double omega = std::sqrt(k_sq);
  const double t = 1.7;

  const SoScalarWave::Solutions::SoStandingWave<Dim> sw(wave_vector, center,
                                                        amplitude);

  // Test copy/move semantics and equality
  CHECK_FALSE(sw != sw);
  test_copy_semantics(sw);
  auto sw_for_move = sw;
  test_move_semantics(std::move(sw_for_move), sw);

  // Test points
  tnsr::I<DataVector, Dim> x{};
  if constexpr (Dim == 1) {
    get<0>(x) = DataVector({-0.5, 0.3, 2.1});
  } else if constexpr (Dim == 2) {
    get<0>(x) = DataVector({-0.5, 0.3, 2.1});
    get<1>(x) = DataVector({1.0, -0.7, 0.4});
  } else {
    get<0>(x) = DataVector({-0.5, 0.3, 2.1});
    get<1>(x) = DataVector({1.0, -0.7, 0.4});
    get<2>(x) = DataVector({0.2, 1.5, -1.0});
  }

  // Compute u = k . (x - x0)
  DataVector u(get<0>(x).size(), 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    u += gsl::at(wave_vector, d) * (x.get(d) - gsl::at(center, d));
  }

  const DataVector sin_u = sin(u);
  const DataVector cos_u = cos(u);
  const double cos_wt = cos(omega * t);
  const double sin_wt = sin(omega * t);

  // Expected values
  const DataVector expected_psi = amplitude * sin_u * cos_wt;
  const DataVector expected_pi = amplitude * omega * sin_u * sin_wt;
  const DataVector expected_dt_psi = -amplitude * omega * sin_u * sin_wt;
  const DataVector expected_dt_pi =
      amplitude * omega * omega * sin_u * cos_wt;

  // Check variables (Psi, Pi, Phi)
  const auto vars = sw.variables(
      x, t,
      tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                 SoScalarWave::Tags::Phi<Dim>>{});
  CHECK_ITERABLE_APPROX(get(get<SoScalarWave::Tags::Psi>(vars)),
                        expected_psi);
  CHECK_ITERABLE_APPROX(get(get<SoScalarWave::Tags::Pi>(vars)), expected_pi);
  for (size_t d = 0; d < Dim; ++d) {
    const DataVector expected_phi_d =
        amplitude * gsl::at(wave_vector, d) * cos_u * cos_wt;
    CHECK_ITERABLE_APPROX(
        get<SoScalarWave::Tags::Phi<Dim>>(vars).get(d), expected_phi_d);
  }

  // Check variables with BoundaryPsi
  const auto vars_with_bpsi = sw.variables(
      x, t,
      tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                 SoScalarWave::Tags::Phi<Dim>,
                 SoScalarWave::Tags::BoundaryPsi>{});
  CHECK_ITERABLE_APPROX(
      get(get<SoScalarWave::Tags::BoundaryPsi>(vars_with_bpsi)),
      expected_psi);

  // Check dt variables
  const auto dt_vars = sw.variables(
      x, t,
      tmpl::list<::Tags::dt<SoScalarWave::Tags::Psi>,
                 ::Tags::dt<SoScalarWave::Tags::Pi>,
                 ::Tags::dt<SoScalarWave::Tags::Phi<Dim>>>{});
  CHECK_ITERABLE_APPROX(
      get(get<::Tags::dt<SoScalarWave::Tags::Psi>>(dt_vars)),
      expected_dt_psi);
  CHECK_ITERABLE_APPROX(
      get(get<::Tags::dt<SoScalarWave::Tags::Pi>>(dt_vars)), expected_dt_pi);
  for (size_t d = 0; d < Dim; ++d) {
    const DataVector expected_dt_phi_d =
        -amplitude * omega * gsl::at(wave_vector, d) * cos_u * sin_wt;
    CHECK_ITERABLE_APPROX(
        get<::Tags::dt<SoScalarWave::Tags::Phi<Dim>>>(dt_vars).get(d),
        expected_dt_phi_d);
  }

  // Check dt variables with dt<BoundaryPsi>
  const auto dt_vars_with_bpsi = sw.variables(
      x, t,
      tmpl::list<::Tags::dt<SoScalarWave::Tags::Psi>,
                 ::Tags::dt<SoScalarWave::Tags::Pi>,
                 ::Tags::dt<SoScalarWave::Tags::Phi<Dim>>,
                 ::Tags::dt<SoScalarWave::Tags::BoundaryPsi>>{});
  CHECK_ITERABLE_APPROX(
      get(get<::Tags::dt<SoScalarWave::Tags::BoundaryPsi>>(
          dt_vars_with_bpsi)),
      expected_dt_psi);

  // Check Pi = 0 at t = 0 (standing wave property)
  const auto vars_t0 = sw.variables(
      x, 0.0,
      tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                 SoScalarWave::Tags::Phi<Dim>>{});
  const DataVector zero(get<0>(x).size(), 0.0);
  CHECK_ITERABLE_APPROX(get(get<SoScalarWave::Tags::Pi>(vars_t0)), zero);

  // Serialization round-trip
  register_factory_classes_with_charm<Metavariables<Dim>>();
  const auto deserialized = serialize_and_deserialize(sw);
  const auto vars_de = deserialized.variables(
      x, t,
      tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                 SoScalarWave::Tags::Phi<Dim>>{});
  CHECK(vars_de == vars);

  // Option creation — values must match those set above exactly
  std::string yaml = "SoStandingWave:\n  WaveVector: [1.5";
  if constexpr (Dim >= 2) {
    yaml += ", -2.25";
  }
  if constexpr (Dim >= 3) {
    yaml += ", 0.75";
  }
  yaml += "]\n  Center: [0.5";
  if constexpr (Dim >= 2) {
    yaml += ", 1.75";
  }
  if constexpr (Dim >= 3) {
    yaml += ", -0.5";
  }
  yaml += "]\n  Amplitude: 2.5";

  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag<
          evolution::initial_data::OptionTags::InitialData,
          Metavariables<Dim>>(yaml)
          ->get_clone();
  const auto deserialized_option = serialize_and_deserialize(option_solution);
  const auto& created =
      dynamic_cast<const SoScalarWave::Solutions::SoStandingWave<Dim>&>(
          *deserialized_option);
  CHECK(created == sw);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.WaveEquation.SoStandingWave",
    "[PointwiseFunctions][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
