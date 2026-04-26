// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/ConstraintDamping/Constant.hpp"
#include "PointwiseFunctions/ConstraintDamping/DampingFunction.hpp"
#include "PointwiseFunctions/ConstraintDamping/GaussianPlusConstant.hpp"
#include "Time/Tags/Time.hpp"

namespace {
struct ArbitraryFrame;
}  // namespace

template <typename DataType, size_t Dim, typename Frame>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<Ccz4::Tags::ConformalFactor<DataType>>(
      "ConformalFactor");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ConformalFactorSquared<DataType>>("ConformalFactorSquared");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ConformalMetric<DataType, Dim, Frame>>(
      "Conformal(SpatialMetric)");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::InverseConformalMetric<DataType, Dim, Frame>>(
      "Conformal(InverseSpatialMetric)");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::ATilde<DataType, Dim, Frame>>(
      "ATilde");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::TraceATilde<DataType>>(
      "TraceATilde");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::LogLapse<DataType>>("LogLapse");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldA<DataType, Dim, Frame>>(
      "FieldA");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldB<DataType, Dim, Frame>>(
      "FieldB");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldD<DataType, Dim, Frame>>(
      "FieldD");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::LogConformalFactor<DataType>>(
      "LogConformalFactor");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldP<DataType, Dim, Frame>>(
      "FieldP");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldDUp<DataType, Dim, Frame>>(
      "FieldDUp");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ConformalChristoffelSecondKind<DataType, Dim, Frame>>(
      "ConformalChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::DerivConformalChristoffelSecondKind<DataType, Dim, Frame>>(
      "DerivConformalChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ChristoffelSecondKind<DataType, Dim, Frame>>(
      "ChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::Ricci<DataType, Dim, Frame>>(
      "Ricci");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::GradGradLapse<DataType, Dim, Frame>>("GradGradLapse");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::DivergenceLapse<DataType>>(
      "DivergenceLapse");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ContractedConformalChristoffelSecondKind<DataType, Dim,
                                                           Frame>>(
      "ContractedConformalChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::DerivContractedConformalChristoffelSecondKind<DataType, Dim,
                                                                Frame>>(
      "DerivContractedConformalChristoffelSecondKind");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::GammaHat<DataType, Dim, Frame>>(
      "GammaHat");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::SpatialZ4Constraint<DataType, Dim, Frame>>(
      "SpatialZ4Constraint");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::SpatialZ4ConstraintUp<DataType, Dim, Frame>>(
      "SpatialZ4ConstraintUp");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::GradSpatialZ4Constraint<DataType, Dim, Frame>>(
      "GradSpatialZ4Constraint");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::RicciScalarPlusDivergenceZ4Constraint<DataType>>(
      "RicciScalarPlusDivergenceZ4Constraint");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::Theta<DataType>>("Theta");
  TestHelpers::db::test_simple_tag<
    Ccz4::Tags::AuxiliaryShiftB<DataType, Dim, Frame>>(
        "AuxiliaryShiftB");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::GammaDriverParam>(
      "GammaDriverParam");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::Kappa1>("Kappa1");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::Kappa2>("Kappa2");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::DampingFunctionKappa1>(
      "DampingFunctionKappa1");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::DampingFunctionKappa2>(
      "DampingFunctionKappa2");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::Kappa3>("Kappa3");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::K0<DataType>>("K0");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::Eta<DataType>>("Eta");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Tags", "[Unit][Evolution]") {
  test_simple_tags<double, 1,
                   ArbitraryFrame>();
  test_simple_tags<DataVector, 1, ArbitraryFrame>();
  test_simple_tags<double, 2, ArbitraryFrame>();
  test_simple_tags<DataVector, 2, ArbitraryFrame>();
  test_simple_tags<double, 3, ArbitraryFrame>();
  test_simple_tags<DataVector, 3, ArbitraryFrame>();
}

namespace {
void test_kappa_compute_tags() {
  const size_t num_pts = 5;
  const double time = 1.3;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  tnsr::I<DataVector, 3, Frame::Grid> coords(num_pts);
  get<0>(coords) = DataVector{1.0, 2.0, 3.0, 4.0, 5.0};
  get<1>(coords) = DataVector{0.5, 1.5, 2.5, 3.5, 4.5};
  get<2>(coords) = DataVector{0.1, 0.2, 0.3, 0.4, 0.5};

  // Test with Constant damping function
  {
    const double constant_value = 0.42;
    auto box = db::create<
        db::AddSimpleTags<Ccz4::Tags::DampingFunctionKappa1,
                          domain::Tags::Coordinates<3, Frame::Grid>,
                          ::Tags::Time, domain::Tags::FunctionsOfTime>,
        db::AddComputeTags<Ccz4::Tags::Kappa1Compute>>(
        std::unique_ptr<ConstraintDamping::DampingFunction<3, Frame::Grid>>(
            std::make_unique<ConstraintDamping::Constant<3, Frame::Grid>>(
                constant_value)),
        coords, time, std::move(functions_of_time));

    const auto& kappa1 = db::get<Ccz4::Tags::Kappa1>(box);
    CHECK(get(kappa1).size() == num_pts);
    for (size_t i = 0; i < num_pts; ++i) {
      CHECK(get(kappa1)[i] == approx(constant_value));
    }
  }

  // Test with GaussianPlusConstant damping function
  {
    const double constant = 1.0;
    const double amplitude = 5.0;
    const double width = 2.0;
    const std::array<double, 3> center{{0.0, 0.0, 0.0}};

    functions_of_time.clear();

    auto box = db::create<
        db::AddSimpleTags<Ccz4::Tags::DampingFunctionKappa2,
                          domain::Tags::Coordinates<3, Frame::Grid>,
                          ::Tags::Time, domain::Tags::FunctionsOfTime>,
        db::AddComputeTags<Ccz4::Tags::Kappa2Compute>>(
        std::unique_ptr<ConstraintDamping::DampingFunction<3, Frame::Grid>>(
            std::make_unique<
                ConstraintDamping::GaussianPlusConstant<3, Frame::Grid>>(
                constant, amplitude, width, center)),
        coords, time, std::move(functions_of_time));

    const auto& kappa2 = db::get<Ccz4::Tags::Kappa2>(box);
    CHECK(get(kappa2).size() == num_pts);

    // Verify against expected Gaussian + Constant values
    for (size_t i = 0; i < num_pts; ++i) {
      const double r_sq = square(get<0>(coords)[i] - center[0]) +
                          square(get<1>(coords)[i] - center[1]) +
                          square(get<2>(coords)[i] - center[2]);
      const double expected =
          constant + amplitude * exp(-r_sq / square(width));
      CHECK(get(kappa2)[i] == approx(expected));
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.KappaComputeTags",
                  "[Unit][Evolution]") {
  test_kappa_compute_tags();
}
