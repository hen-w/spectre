// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
using Condition = gp::BoundaryConditions::DirichletAnalytic;
using KerrSchild = Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::KerrSchild>;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Ccz4::BoundaryConditions::BoundaryCondition,
                             tmpl::list<Condition>>,
                  tmpl::pair<evolution::initial_data::InitialData,
                             Ccz4::Solutions::all_solutions>>;
  };
};

using physical_tags = tmpl::list<
    Ccz4::Tags::ConformalMetric<DataVector, 3>,
    Ccz4::Tags::ConformalFactor<DataVector>, Ccz4::Tags::ATilde<DataVector, 3>,
    gr::Tags::TraceExtrinsicCurvature<DataVector>,
    Ccz4::Tags::Theta<DataVector>, Ccz4::Tags::GammaHat<DataVector, 3>,
    gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
    Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>;

// The chart potentials of the analytic solution at a single point.
gp::PotentialVector potentials(const KerrSchild& solution,
                               const std::array<double, 3>& x,
                               const double time) {
  const tnsr::I<DataVector, 3> coords{
      {{DataVector{x[0]}, DataVector{x[1]}, DataVector{x[2]}}}};
  const auto values = solution.variables(coords, time, physical_tags{});
  gp::PhysicalPointState state{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      state.conformal_metric(i, j) =
          get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(values).get(i, j)[0];
      state.a_tilde(i, j) =
          get<Ccz4::Tags::ATilde<DataVector, 3>>(values).get(i, j)[0];
    }
    gsl::at(state.shift, i) =
        get<gr::Tags::Shift<DataVector, 3>>(values).get(i)[0];
  }
  state.conformal_factor =
      get(get<Ccz4::Tags::ConformalFactor<DataVector>>(values))[0];
  state.lapse = get(get<gr::Tags::Lapse<DataVector>>(values))[0];
  gp::PotentialVector u{};
  gp::VelocityVector w{};
  gp::chart_from_physical(make_not_null(&u), make_not_null(&w), state);
  return u;
}

void test_analytic(const Condition& condition, const KerrSchild& solution) {
  const double time = 0.7;
  const DataVector x{2.1, -3.0, 4.2};
  const DataVector y{1.4, 2.5, -0.3};
  const DataVector z{-2.2, 1.9, 3.3};
  const tnsr::I<DataVector, 3> coords{{{x, y, z}}};
  const size_t n = x.size();
  gp::Tags::PotentialGradients::type q{n};
  tnsr::ii<DataVector, 3> g{n};
  Scalar<DataVector> phi{n};
  tnsr::ii<DataVector, 3> a_tilde{n};
  Scalar<DataVector> k{n};
  Scalar<DataVector> theta{n};
  tnsr::I<DataVector, 3> gamma_hat{n};
  Scalar<DataVector> lapse{n};
  tnsr::I<DataVector, 3> shift{n};
  tnsr::I<DataVector, 3> b{n};
  const tnsr::i<DataVector, 3> normal{n, 0.0};
  const auto result = condition.dg_ghost(
      make_not_null(&g), make_not_null(&phi), make_not_null(&a_tilde),
      make_not_null(&k), make_not_null(&theta), make_not_null(&gamma_hat),
      make_not_null(&lapse), make_not_null(&shift), make_not_null(&b),
      make_not_null(&q), std::nullopt, normal, coords, time);
  CHECK(not result.has_value());

  // The physical exterior fields are the analytic fields.
  const auto values = solution.variables(coords, time, physical_tags{});
  CHECK(g == get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(values));
  CHECK(phi == get<Ccz4::Tags::ConformalFactor<DataVector>>(values));
  CHECK(a_tilde == get<Ccz4::Tags::ATilde<DataVector, 3>>(values));
  CHECK(k == get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(values));
  CHECK(theta == get<Ccz4::Tags::Theta<DataVector>>(values));
  CHECK(gamma_hat == get<Ccz4::Tags::GammaHat<DataVector, 3>>(values));
  CHECK(lapse == get<gr::Tags::Lapse<DataVector>>(values));
  CHECK(shift == get<gr::Tags::Shift<DataVector, 3>>(values));
  CHECK(b == get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(values));

  for (size_t s = 0; s < n; ++s) {
    const std::array<double, 3> point{{x[s], y[s], z[s]}};
    // The chart gradients against fourth-order finite differences of the
    // chart potentials (H, phi, alpha, beta) of the analytic solution, in
    // particular q_phi = phi FieldP.
    const double h = 1.0e-3;
    for (size_t i = 0; i < 3; ++i) {
      const auto shifted = [&](const double offset) {
        std::array<double, 3> moved = point;
        gsl::at(moved, i) += offset;
        return potentials(solution, moved, time);
      };
      const auto plus_2 = shifted(2.0 * h);
      const auto plus_1 = shifted(h);
      const auto minus_1 = shifted(-h);
      const auto minus_2 = shifted(-2.0 * h);
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        const double derivative =
            (-gsl::at(plus_2, a) + 8.0 * gsl::at(plus_1, a) -
             8.0 * gsl::at(minus_1, a) + gsl::at(minus_2, a)) /
            (12.0 * h);
        CAPTURE(i);
        CAPTURE(a);
        CHECK(q.get(i, a)[s] == approx(derivative).epsilon(1e-8).scale(1.0));
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection.DirichletAnalytic",
    "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables>();
  const auto created = TestHelpers::test_creation<
      std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>,
      Metavariables>(
      "DirichletAnalytic:\n"
      "  AnalyticPrescription:\n"
      "      Ccz4(KerrSchild):\n"
      "        Mass: 1.0\n"
      "        Spin: [0.5, -0.2, 0.0]\n"
      "        Center: [-0.2, 0.0, 0.3]\n"
      "        Velocity: [0.0, 0.0, 0.0]\n");
  const auto* const typed = dynamic_cast<const Condition*>(created.get());
  REQUIRE(typed != nullptr);
  const KerrSchild solution{
      1.0, {{0.5, -0.2, 0.0}}, {{-0.2, 0.0, 0.3}}, {{0.0, 0.0, 0.0}}};
  test_analytic(serialize_and_deserialize(*typed), solution);
  const auto cloned = typed->get_clone();
  test_analytic(dynamic_cast<const Condition&>(*cloned), solution);
}
