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
  gp::Tags::ChartW::type w;
  gp::Tags::ChartQ::type q;
  gp::Tags::ChartU::type u;
  tnsr::i<DataVector, 3> normal;
};

struct Output {
  gp::Tags::ChartW::type w;
  gp::Tags::ChartQ::type q;
  gp::Tags::ChartU::type u;
  tnsr::ii<DataVector, 3> g;
  Scalar<DataVector> phi;
  tnsr::ii<DataVector, 3> a_tilde;
  Scalar<DataVector> k;
  Scalar<DataVector> theta;
  tnsr::I<DataVector, 3> gamma_hat;
  Scalar<DataVector> lapse;
  tnsr::I<DataVector, 3> shift;
  tnsr::I<DataVector, 3> b;
};

Side package(const Correction& correction, const Side& raw,
             const gsl::not_null<double*> largest_speed) {
  const size_t n = get<0>(raw.u).size();
  Side packaged{gp::Tags::ChartW::type{n}, gp::Tags::ChartQ::type{n},
                gp::Tags::ChartU::type{n}, tnsr::i<DataVector, 3>{n}};
  const tnsr::ii<DataVector, 3> unused_ii{n, 0.0};
  const Scalar<DataVector> unused_scalar{n, 0.0};
  const tnsr::I<DataVector, 3> unused_vector{n, 0.0};
  *largest_speed = correction.dg_package_data(
      make_not_null(&packaged.w), make_not_null(&packaged.q),
      make_not_null(&packaged.u), make_not_null(&packaged.normal), raw.w, raw.q,
      raw.u, unused_ii, unused_scalar, unused_ii, unused_scalar, unused_scalar,
      unused_vector, unused_scalar, unused_vector, unused_vector, raw.normal,
      std::nullopt, std::nullopt, Direction<3>::upper_xi());
  return packaged;
}

template <bool External>
Output boundary_terms(
    const Correction& correction, const Side& interior, const Side& exterior,
    const dg::Formulation formulation = dg::Formulation::StrongInertial) {
  const size_t n = get<0>(interior.u).size();
  Output result{gp::Tags::ChartW::type{n}, gp::Tags::ChartQ::type{n},
                gp::Tags::ChartU::type{n}, tnsr::ii<DataVector, 3>{n},
                Scalar<DataVector>{n},     tnsr::ii<DataVector, 3>{n},
                Scalar<DataVector>{n},     Scalar<DataVector>{n},
                tnsr::I<DataVector, 3>{n}, Scalar<DataVector>{n},
                tnsr::I<DataVector, 3>{n}, tnsr::I<DataVector, 3>{n}};
  correction.dg_boundary_terms<External>(
      make_not_null(&result.w), make_not_null(&result.q),
      make_not_null(&result.u), make_not_null(&result.g),
      make_not_null(&result.phi), make_not_null(&result.a_tilde),
      make_not_null(&result.k), make_not_null(&result.theta),
      make_not_null(&result.gamma_hat), make_not_null(&result.lapse),
      make_not_null(&result.shift), make_not_null(&result.b), interior.w,
      interior.q, interior.u, interior.normal, exterior.w, exterior.q,
      exterior.u, exterior.normal, formulation);
  return result;
}

Side random_side(const gsl::not_null<std::mt19937*> gen, const size_t n,
                 const std::vector<std::array<double, 3>>& normals,
                 const double sign) {
  Side side{gp::Tags::ChartW::type{n}, gp::Tags::ChartQ::type{n},
            gp::Tags::ChartU::type{n}, tnsr::i<DataVector, 3>{n}};
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

std::array<double, n43> target_at(const gp::Tags::ChartW::type& w,
                                  const gp::Tags::ChartQ::type& q,
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

// Expected -1/2 (A_n + factor (|A_n| + I)) [Y] from the independently assembled
// normal matrix and its eigenvector absolute value at u_* = (u^- + u^+)/2, for
// the configured Gamma-driver parameter Ccz4::fd::System::f.
std::array<double, n43> expected_correction(const Side& interior,
                                            const Side& exterior,
                                            const size_t s,
                                            const double factor) {
  DataVector u_average{gp::number_of_potentials};
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    u_average[a] = 0.5 * (interior.u.get(a)[s] + exterior.u.get(a)[s]);
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
    double value = factor * (gsl::at(y_ext, row) - gsl::at(y_int, row));
    for (size_t column = 0; column < n43; ++column) {
      const double jump = gsl::at(y_ext, column) - gsl::at(y_int, column);
      value +=
          (a[row * n43 + column] + factor * abs_a[row * n43 + column]) * jump;
    }
    gsl::at(result, row) = -0.5 * value;
  }
  return result;
}

void check_physical_zero(const Output& output) {
  for (const auto& component : output.g) {
    CHECK(max(abs(component)) == 0.0);
  }
  CHECK(max(abs(get(output.phi))) == 0.0);
  for (const auto& component : output.a_tilde) {
    CHECK(max(abs(component)) == 0.0);
  }
  CHECK(max(abs(get(output.k))) == 0.0);
  CHECK(max(abs(get(output.theta))) == 0.0);
  CHECK(max(abs(get(output.lapse))) == 0.0);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(max(abs(output.gamma_hat.get(i))) == 0.0);
    CHECK(max(abs(output.shift.get(i))) == 0.0);
    CHECK(max(abs(output.b.get(i))) == 0.0);
  }
}

void test_against_reference(const gsl::not_null<std::mt19937*> gen) {
  const size_t n = 5;
  std::vector<std::array<double, 3>> normals{};
  for (size_t s = 0; s < n; ++s) {
    normals.push_back(s == 0 ? std::array<double, 3>{{0.0, 1.0, 0.0}}
                             : helpers::random_unit_vector(gen));
  }
  const Side interior_raw = random_side(gen, n, normals, 1.0);
  const Side exterior_raw = random_side(gen, n, normals, -1.0);
  for (const double factor : {0.0, 1.0, 2.0}) {
    const Correction correction{factor};
    double largest_speed = 0.0;
    const Side interior =
        package(correction, interior_raw, make_not_null(&largest_speed));
    CHECK(interior.w == interior_raw.w);
    CHECK(interior.q == interior_raw.q);
    CHECK(interior.u == interior_raw.u);
    CHECK(interior.normal == interior_raw.normal);
    double expected_largest = 0.0;
    for (size_t s = 0; s < n; ++s) {
      const auto u = gp::potentials_at(interior_raw.u, s);
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
          expected_correction(interior, exterior, s, factor);
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
    check_physical_zero(internal);
    check_physical_zero(external);

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
void test_upwind_sign() {
  gp::PotentialVector u{};
  u[gp::PotentialIndex::lapse] = 0.8;
  u[gp::PotentialIndex::log_conformal_factor] = std::log(0.9);
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
    Side interior{
        gp::Tags::ChartW::type{1_st, 0.0}, gp::Tags::ChartQ::type{1_st, 0.0},
        gp::Tags::ChartU::type{1_st, 0.0}, tnsr::i<DataVector, 3>{1_st}};
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
    const auto actual = target_at(output.w, output.q, 0);
    const auto mode = target_at(exterior.w, exterior.q, 0);
    const double expected_factor = -0.5 * (speed + std::abs(speed) + 1.0);
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
  test_options_and_errors(make_not_null(&gen));
}
