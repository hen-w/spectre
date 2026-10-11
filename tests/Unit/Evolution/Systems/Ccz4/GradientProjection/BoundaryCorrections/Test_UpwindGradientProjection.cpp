// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryCorrections/UpwindGradientProjection.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/NormalOperator.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;
using Correction = gp::BoundaryCorrections::UpwindGradientProjection;
using helpers::to_data_vector;
constexpr const char* module = "GradientProjectionReference";
constexpr size_t n43 = gp::number_of_target_components;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<evolution::BoundaryCorrection, tmpl::list<Correction>>>;
  };
};

// Packaged data of one side of a face.
struct Side {
  gp::Tags::PackagedVelocities::type w;
  gp::Tags::PackagedGradients::type q;
  gp::Tags::PackagedPotentials::type u;
  tnsr::i<DataVector, 3> normal;
};

// The face values of one side: physical fields and stored gradients.
struct RawSide {
  helpers::ChartData chart;
  gp::Tags::PotentialGradients::type q;
  tnsr::i<DataVector, 3> normal;
};

struct Output {
  gp::Tags::VelocityTarget::type w;
  gp::Tags::GradientTarget::type q;
  gp::Tags::PotentialRateTarget::type u;
};

Side package(const Correction& correction, const RawSide& raw,
             const gsl::not_null<double*> largest_speed) {
  const size_t n = get(raw.chart.lapse).size();
  Side packaged{gp::Tags::PackagedVelocities::type{n},
                gp::Tags::PackagedGradients::type{n},
                gp::Tags::PackagedPotentials::type{n},
                tnsr::i<DataVector, 3>{n}};
  const auto& chart = raw.chart;
  *largest_speed = correction.dg_package_data(
      make_not_null(&packaged.w), make_not_null(&packaged.q),
      make_not_null(&packaged.u), make_not_null(&packaged.normal),
      chart.conformal_metric, chart.conformal_factor, chart.a_tilde,
      chart.trace_extrinsic_curvature, chart.theta, chart.gamma_hat,
      chart.lapse, chart.shift, chart.b, raw.q, raw.normal, std::nullopt,
      std::nullopt, Direction<3>::upper_xi());
  return packaged;
}

template <bool External>
Output boundary_terms(
    const Correction& correction, const Side& interior, const Side& exterior,
    const dg::Formulation formulation = dg::Formulation::StrongInertial) {
  const size_t n = get<0>(interior.u).size();
  Output result{gp::Tags::VelocityTarget::type{n},
                gp::Tags::GradientTarget::type{n},
                gp::Tags::PotentialRateTarget::type{n}};
  correction.dg_boundary_terms<External>(
      make_not_null(&result.w), make_not_null(&result.q),
      make_not_null(&result.u), interior.w, interior.q, interior.u,
      interior.normal, exterior.w, exterior.q, exterior.u, exterior.normal,
      formulation);
  return result;
}

// Random admissible face states with the given normals (times `sign`), and
// their chart fields.
RawSide random_raw_side(
    const gsl::not_null<std::mt19937*> gen, const size_t n,
    const std::vector<std::array<double, 3>>& normals, const double sign,
    const gsl::not_null<std::vector<gp::PotentialVector>*> chart_potentials,
    const gsl::not_null<std::vector<gp::VelocityVector>*> chart_velocities) {
  RawSide side{helpers::ChartData{n}, gp::Tags::PotentialGradients::type{n},
               tnsr::i<DataVector, 3>{n}};
  chart_potentials->clear();
  chart_velocities->clear();
  for (size_t s = 0; s < n; ++s) {
    const auto state = helpers::random_state(gen, 0.3, 0.25, 0.3);
    side.chart.set(s, state.u, state.w);
    gp::set_gradients_at(make_not_null(&side.q), s, state.q);
    chart_potentials->push_back(state.u);
    chart_velocities->push_back(state.w);
    for (size_t i = 0; i < 3; ++i) {
      side.normal.get(i)[s] = sign * gsl::at(normals[s], i);
    }
  }
  return side;
}

Side random_side(const gsl::not_null<std::mt19937*> gen, const size_t n,
                 const std::vector<std::array<double, 3>>& normals,
                 const double sign) {
  Side side{gp::Tags::PackagedVelocities::type{n},
            gp::Tags::PackagedGradients::type{n},
            gp::Tags::PackagedPotentials::type{n}, tnsr::i<DataVector, 3>{n}};
  for (size_t s = 0; s < n; ++s) {
    const auto state = helpers::random_state(gen, 0.3, 0.25, 0.3);
    gp::set_potentials_at(make_not_null(&side.u), s, state.u);
    gp::set_velocities_at(make_not_null(&side.w), s, state.w);
    gp::set_gradients_at(make_not_null(&side.q), s, state.q);
    for (size_t i = 0; i < 3; ++i) {
      side.normal.get(i)[s] = sign * gsl::at(normals[s], i);
    }
  }
  return side;
}

template <typename WType, typename QType>
std::array<double, n43> target_at(const WType& w, const QType& q,
                                  const size_t s) {
  std::array<double, n43> result{};
  for (size_t a = 0; a < gp::number_of_velocities; ++a) {
    gsl::at(result, a) = w.get(a)[s];
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      gsl::at(result, 13 + 10 * i + a) = q.get(i, a)[s];
    }
  }
  return result;
}

// Expected -1/2 (A_n + factor (|A_n| + sigma I)) [Y] from the independently assembled
// normal matrix and its eigenvector absolute value at the coefficient state
// u_average (by default the arithmetic mean u_* = (u^- + u^+)/2), for the
// configured Gamma-driver parameter Ccz4::fd::System::f.
std::array<double, n43> expected_correction(
    const Side& interior, const Side& exterior, const size_t s,
    const double factor,
    const std::optional<DataVector>& coefficient_state = std::nullopt,
    const double scalar_penalty = 1.0) {
  DataVector u_average{gp::number_of_potentials};
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    u_average[a] = 0.5 * (interior.u.get(a)[s] + exterior.u.get(a)[s]);
  }
  if (coefficient_state.has_value()) {
    u_average = *coefficient_state;
  }
  const DataVector normal{get<0>(interior.normal)[s],
                          get<1>(interior.normal)[s],
                          get<2>(interior.normal)[s]};
  const DataVector a = pypp::call<DataVector>(
      module, "normal_matrix", u_average, normal, ::Ccz4::fd::System::f);
  const DataVector abs_a =
      pypp::call<DataVector>(module, "absolute_normal_matrix_eig", u_average,
                             normal, ::Ccz4::fd::System::f);
  const auto y_int = target_at(interior.w, interior.q, s);
  const auto y_ext = target_at(exterior.w, exterior.q, s);
  std::array<double, n43> result{};
  for (size_t row = 0; row < n43; ++row) {
    double value = factor * scalar_penalty *
                   (gsl::at(y_ext, row) - gsl::at(y_int, row));
    for (size_t column = 0; column < n43; ++column) {
      const double jump = gsl::at(y_ext, column) - gsl::at(y_int, column);
      value +=
          (a[row * n43 + column] + factor * abs_a[row * n43 + column]) * jump;
    }
    gsl::at(result, row) = -0.5 * value;
  }
  return result;
}

void test_against_reference(const gsl::not_null<std::mt19937*> gen) {
  const size_t n = 5;
  std::vector<std::array<double, 3>> normals{};
  for (size_t s = 0; s < n; ++s) {
    normals.push_back(s == 0 ? std::array<double, 3>{{0.0, 1.0, 0.0}}
                             : helpers::random_unit_vector(gen));
  }
  std::vector<gp::PotentialVector> interior_u{};
  std::vector<gp::VelocityVector> interior_w{};
  std::vector<gp::PotentialVector> exterior_u{};
  std::vector<gp::VelocityVector> exterior_w{};
  const RawSide interior_raw =
      random_raw_side(gen, n, normals, 1.0, make_not_null(&interior_u),
                      make_not_null(&interior_w));
  const RawSide exterior_raw =
      random_raw_side(gen, n, normals, -1.0, make_not_null(&exterior_u),
                      make_not_null(&exterior_w));
  for (const double factor : {0.0, 1.0, 2.0}) {
    const Correction correction{factor};
    double largest_speed = 0.0;
    const Side interior =
        package(correction, interior_raw, make_not_null(&largest_speed));
    // The packaged chart fields of the physical face values; H and a are
    // recomputed and agree with the chart state to rounding.
    for (size_t s = 0; s < n; ++s) {
      const auto packaged_u = gp::potentials_at(interior.u, s);
      const auto packaged_w = gp::velocities_at(interior.w, s);
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        CHECK(gsl::at(packaged_u, a) ==
              approx(gsl::at(interior_u[s], a)).epsilon(1e-12).scale(1.0));
      }
      for (size_t a = 0; a < gp::number_of_velocities; ++a) {
        CHECK(gsl::at(packaged_w, a) ==
              approx(gsl::at(interior_w[s], a)).epsilon(1e-12).scale(1.0));
      }
      // The conformal factor and the other physical potentials are copied.
      CHECK(packaged_u[gp::PotentialIndex::conformal_factor] ==
            get(interior_raw.chart.conformal_factor)[s]);
      CHECK(packaged_u[gp::PotentialIndex::lapse] ==
            get(interior_raw.chart.lapse)[s]);
    }
    CHECK(interior.q == interior_raw.q);
    CHECK(interior.normal == interior_raw.normal);
    double expected_largest = 0.0;
    for (size_t s = 0; s < n; ++s) {
      const auto u = gp::potentials_at(interior.u, s);
      const gp::LogMetricChart chart{
          gp::stf_decode(&u[gp::PotentialIndex::log_metric])};
      const gp::NormalCharacteristics characteristics{chart, u, normals[s],
                                                      ::Ccz4::fd::System::f};
      for (const double speed : characteristics.speeds()) {
        expected_largest = std::max(expected_largest, std::abs(speed));
      }
    }
    CHECK(largest_speed == approx(expected_largest));
    const Side exterior =
        package(correction, exterior_raw, make_not_null(&largest_speed));

    // Internal faces always use unit dissipation and carry the value jump.
    const Output internal =
        boundary_terms<false>(correction, interior, exterior);
    // External faces scale the dissipative part and carry no value jump.
    const Output external =
        boundary_terms<true>(correction, interior, exterior);
    for (size_t s = 0; s < n; ++s) {
      const auto internal_expected =
          expected_correction(interior, exterior, s, 1.0);
      const auto external_expected =
          expected_correction(interior, exterior, s, factor, std::nullopt, 0.0);
      const auto internal_actual = target_at(internal.w, internal.q, s);
      const auto external_actual = target_at(external.w, external.q, s);
      double scale = 1.0;
      for (const double x : internal_expected) {
        scale = std::max(scale, std::abs(x));
      }
      for (size_t row = 0; row < n43; ++row) {
        CHECK(gsl::at(internal_actual, row) ==
              approx(gsl::at(internal_expected, row))
                  .scale(scale)
                  .epsilon(1e-11));
        CHECK(gsl::at(external_actual, row) ==
              approx(gsl::at(external_expected, row))
                  .scale(scale)
                  .epsilon(1e-11));
      }
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        CHECK(internal.u.get(a)[s] ==
              approx(-0.5 * (exterior.u.get(a)[s] - interior.u.get(a)[s])));
        CHECK(external.u.get(a)[s] == 0.0);
      }
    }

    // The neighbor evaluates the same u_* with the reversed normal and jump:
    // the sum of the two sides' corrections is -A_n [Y] and the difference
    // is -(|A_n| + I) [Y]; the value-jump inputs are opposite.
    const Output neighbor =
        boundary_terms<false>(correction, exterior, interior);
    for (size_t s = 0; s < n; ++s) {
      const auto mine = target_at(internal.w, internal.q, s);
      const auto theirs = target_at(neighbor.w, neighbor.q, s);
      const auto central = expected_correction(interior, exterior, s, 0.0);
      for (size_t row = 0; row < n43; ++row) {
        CHECK(gsl::at(mine, row) + gsl::at(theirs, row) ==
              approx(2.0 * gsl::at(central, row)).scale(10.0));
      }
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        CHECK(internal.u.get(a)[s] == approx(-neighbor.u.get(a)[s]));
      }
    }

    // The neighbor normal is ignored: the correction uses only the interior
    // normal and is unchanged by an arbitrary exterior normal.
    {
      Side other_normal = exterior;
      for (size_t i = 0; i < 3; ++i) {
        other_normal.normal.get(i) =
            2.0 + static_cast<double>(i) - 0.3 * exterior.normal.get(i);
      }
      const Output ignored =
          boundary_terms<false>(correction, interior, other_normal);
      CHECK(ignored.w == internal.w);
      CHECK(ignored.q == internal.q);
      CHECK(ignored.u == internal.u);
    }

    // Equal traces give no correction.
    const Output equal = boundary_terms<false>(correction, interior, [&]() {
      Side same = interior;
      for (auto& component : same.normal) {
        component *= -1.0;
      }
      return same;
    }());
    for (const auto& component : equal.w) {
      CHECK(max(abs(component)) == 0.0);
    }
    for (const auto& component : equal.q) {
      CHECK(max(abs(component)) == 0.0);
    }
    for (const auto& component : equal.u) {
      CHECK(max(abs(component)) == 0.0);
    }
  }
}

// Upwind sign: a jump along a single characteristic mode with speed lambda
// (A_n r = lambda r for Y_t = A^i d_i Y) gets -1/2 (lambda + |lambda| + 1) r,
// which after the strong-form lift (which subtracts L_f times the returned
// value) adds L_f (lambda + 1/2) r for an incoming (lambda > 0) mode and only
// the scalar penalty L_f r / 2 for an outgoing one.
// At external faces the scalar penalty is absent, so outgoing modes receive
// no correction while incoming modes receive the ordinary upwind correction.
void test_upwind_sign() {
  gp::PotentialVector u{};
  u[gp::PotentialIndex::lapse] = 0.8;
  u[gp::PotentialIndex::conformal_factor] = 0.9;
  u[gp::PotentialIndex::shift] = 0.1;
  const std::array<double, 3> normal{{0.6, 0.0, 0.8}};
  const gp::LogMetricChart chart{gp::Matrix3{}};
  const gp::NormalCharacteristics characteristics{chart, u, normal,
                                                  ::Ccz4::fd::System::f};
  const Correction correction{1.0};
  for (size_t k = 0; k < gp::number_of_normal_components; ++k) {
    if (gp::NormalCharacteristics::is_kernel_field(k)) {
      continue;
    }
    const double speed = gsl::at(characteristics.speeds(), k);
    gp::NormalSubspaceVector unit{};
    gsl::at(unit, k) = 1.0;
    gp::VelocityVector mode_w{};
    gp::GradientVector mode_q{};
    gp::embed_normal_subspace_vector(
        make_not_null(&mode_w), make_not_null(&mode_q), normal,
        characteristics.normal_fields_from_characteristic_fields(unit));
    Side interior{gp::Tags::PackagedVelocities::type{1_st, 0.0},
                  gp::Tags::PackagedGradients::type{1_st, 0.0},
                  gp::Tags::PackagedPotentials::type{1_st, 0.0},
                  tnsr::i<DataVector, 3>{1_st}};
    gp::set_potentials_at(make_not_null(&interior.u), 0, u);
    for (size_t i = 0; i < 3; ++i) {
      interior.normal.get(i) = gsl::at(normal, i);
    }
    Side exterior = interior;
    gp::set_velocities_at(make_not_null(&exterior.w), 0, mode_w);
    gp::set_gradients_at(make_not_null(&exterior.q), 0, mode_q);
    for (auto& component : exterior.normal) {
      component *= -1.0;
    }
    const Output output = boundary_terms<false>(correction, interior, exterior);
    const Output external = boundary_terms<true>(correction, interior, exterior);
    const auto actual = target_at(output.w, output.q, 0);
    const auto external_actual = target_at(external.w, external.q, 0);
    const auto mode = target_at(exterior.w, exterior.q, 0);
    const double expected_factor = -0.5 * (speed + std::abs(speed) + 1.0);
    const double external_factor = -0.5 * (speed + std::abs(speed));
    double scale = 1.0;
    for (const double x : mode) {
      scale = std::max(scale, std::abs(x));
    }
    CAPTURE(k);
    CAPTURE(speed);
    for (size_t row = 0; row < n43; ++row) {
      CHECK(gsl::at(actual, row) == approx(expected_factor * gsl::at(mode, row))
                                        .scale(scale)
                                        .epsilon(1e-12));
      CHECK(gsl::at(external_actual, row) ==
            approx(external_factor * gsl::at(mode, row))
                .scale(scale)
                .epsilon(1e-12));
    }
  }
}

// The correction is pointwise in global components: a permutation of the
// face points (as from an orientation map) permutes the result.
void test_permutation(const gsl::not_null<std::mt19937*> gen) {
  const size_t n = 4;
  std::vector<std::array<double, 3>> normals(n, {{0.0, 0.0, 1.0}});
  const Side interior = random_side(gen, n, normals, 1.0);
  const Side exterior = random_side(gen, n, normals, -1.0);
  const Correction correction{1.0};
  const Output output = boundary_terms<false>(correction, interior, exterior);
  const std::array<size_t, 4> permutation{{2, 0, 3, 1}};
  const auto permute = [&permutation](auto tensor) {
    auto result = tensor;
    for (size_t c = 0; c < tensor.size(); ++c) {
      for (size_t s = 0; s < 4; ++s) {
        result[c][s] = tensor[c][gsl::at(permutation, s)];
      }
    }
    return result;
  };
  const Side permuted_interior{permute(interior.w), permute(interior.q),
                               permute(interior.u), permute(interior.normal)};
  const Side permuted_exterior{permute(exterior.w), permute(exterior.q),
                               permute(exterior.u), permute(exterior.normal)};
  const Output permuted =
      boundary_terms<false>(correction, permuted_interior, permuted_exterior);
  CHECK_ITERABLE_APPROX(permuted.w, permute(output.w));
  CHECK_ITERABLE_APPROX(permuted.q, permute(output.q));
  CHECK_ITERABLE_APPROX(permuted.u, permute(output.u));
}

// The coefficient state of the correction is the arithmetic mean of the
// potentials, so for unequal conformal factors on the two sides it uses the
// arithmetic mean of phi (not the geometric mean of the previous log(phi)
// potential). The geometric-mean control distinguishes the two.
void test_unequal_conformal_factors(const gsl::not_null<std::mt19937*> gen) {
  const std::vector<std::array<double, 3>> normals(
      2, std::array<double, 3>{{0.6, 0.0, 0.8}});
  Side interior = random_side(gen, 2, normals, 1.0);
  Side exterior = random_side(gen, 2, normals, -1.0);
  // Equal potentials apart from phi, chosen away from the characteristic
  // speed collisions (where the eigenvector oracle is unusable).
  const std::array<double, 5> h{{0.1, -0.05, 0.08, 0.0, -0.03}};
  for (size_t a = 0; a < 5; ++a) {
    interior.u.get(gp::PotentialIndex::log_metric + a) = gsl::at(h, a);
  }
  get<gp::PotentialIndex::lapse>(interior.u) = 0.9;
  for (size_t i = 0; i < 3; ++i) {
    interior.u.get(gp::PotentialIndex::shift + i) =
        0.04 * static_cast<double>(i + 1);
  }
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    exterior.u.get(a) = interior.u.get(a);
  }
  get<gp::PotentialIndex::conformal_factor>(interior.u) = DataVector{0.5, 0.3};
  get<gp::PotentialIndex::conformal_factor>(exterior.u) = DataVector{1.5, 1.2};
  const Correction correction{1.0};
  const Output output = boundary_terms<false>(correction, interior, exterior);
  for (size_t s = 0; s < 2; ++s) {
    const auto expected = expected_correction(interior, exterior, s, 1.0);
    DataVector geometric{gp::number_of_potentials};
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      geometric[a] = interior.u.get(a)[s];
    }
    geometric[gp::PotentialIndex::conformal_factor] =
        std::sqrt(get<gp::PotentialIndex::conformal_factor>(interior.u)[s] *
                  get<gp::PotentialIndex::conformal_factor>(exterior.u)[s]);
    const auto control =
        expected_correction(interior, exterior, s, 1.0, geometric);
    const auto actual = target_at(output.w, output.q, s);
    double scale = 1.0;
    double control_difference = 0.0;
    for (size_t row = 0; row < n43; ++row) {
      scale = std::max(scale, std::abs(gsl::at(expected, row)));
      control_difference =
          std::max(control_difference,
                   std::abs(gsl::at(control, row) - gsl::at(expected, row)));
    }
    for (size_t row = 0; row < n43; ++row) {
      CHECK(gsl::at(actual, row) ==
            approx(gsl::at(expected, row)).scale(scale).epsilon(1e-11));
    }
    CHECK(control_difference > 1.0e-3 * scale);
    // The value-jump input is the jump of phi itself.
    CHECK(output.u.get(gp::PotentialIndex::conformal_factor)[s] ==
          approx(-0.5 *
                 (exterior.u.get(gp::PotentialIndex::conformal_factor)[s] -
                  interior.u.get(gp::PotentialIndex::conformal_factor)[s])));
  }
}

// Packaging rejects face states off the algebraic-constraint surface and
// non-positive conformal factors instead of silently projecting or clipping.
void test_package_errors(const gsl::not_null<std::mt19937*> gen) {
  const std::vector<std::array<double, 3>> normals(
      3, std::array<double, 3>{{1.0, 0.0, 0.0}});
  std::vector<gp::PotentialVector> u{};
  std::vector<gp::VelocityVector> w{};
  const RawSide raw = random_raw_side(gen, 3, normals, 1.0, make_not_null(&u),
                                      make_not_null(&w));
  const Correction correction{1.0};
  double largest_speed = 0.0;
  RawSide off_constraint = raw;
  for (auto& component : off_constraint.chart.conformal_metric) {
    component *= 1.0 + 1.0e-6;
  }
  CHECK_THROWS_WITH(
      package(correction, off_constraint, make_not_null(&largest_speed)),
      Catch::Matchers::ContainsSubstring(
          "off the algebraic-constraint surface"));
  RawSide traceful = raw;
  for (size_t i = 0; i < 3; ++i) {
    traceful.chart.a_tilde.get(i, i) +=
        1.0e-6 * raw.chart.conformal_metric.get(i, i);
  }
  CHECK_THROWS_WITH(
      package(correction, traceful, make_not_null(&largest_speed)),
      Catch::Matchers::ContainsSubstring(
          "off the algebraic-constraint surface"));
  RawSide negative = raw;
  get(negative.chart.conformal_factor)[1] = -0.2;
  CHECK_THROWS_WITH(
      package(correction, negative, make_not_null(&largest_speed)),
      Catch::Matchers::ContainsSubstring(
          "The conformal factor must be positive"));
}

void test_options_and_errors(const gsl::not_null<std::mt19937*> gen) {
  register_factory_classes_with_charm<Metavariables>();
  const auto created =
      TestHelpers::test_creation<std::unique_ptr<evolution::BoundaryCorrection>,
                                 Metavariables>(
          "UpwindGradientProjection:\n"
          "  ExternalDissipationFactor: 2.0\n");
  const auto* const typed = dynamic_cast<const Correction*>(created.get());
  REQUIRE(typed != nullptr);
  CHECK(typed->external_dissipation_factor() == 2.0);
  const auto serialized = serialize_and_deserialize(*typed);
  CHECK(serialized.external_dissipation_factor() == 2.0);
  CHECK(dynamic_cast<const Correction&>(*typed->get_clone())
            .external_dissipation_factor() == 2.0);

  std::vector<std::array<double, 3>> normals(1, {{1.0, 0.0, 0.0}});
  const Side interior = random_side(gen, 1, normals, 1.0);
  const Side exterior = random_side(gen, 1, normals, -1.0);
  CHECK_THROWS_WITH(boundary_terms<false>(*typed, interior, exterior,
                                          dg::Formulation::WeakInertial),
                    Catch::Matchers::ContainsSubstring(
                        "supports only the strong formulation"));
}
}  // namespace

// [[TimeOut, 30]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection.UpwindGradientProjection",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Evolution/Systems/Ccz4/GradientProjection/");
  MAKE_GENERATOR(gen);
  test_against_reference(make_not_null(&gen));
  test_upwind_sign();
  test_permutation(make_not_null(&gen));
  test_unequal_conformal_factors(make_not_null(&gen));
  test_package_errors(make_not_null(&gen));
  test_options_and_errors(make_not_null(&gen));
}
