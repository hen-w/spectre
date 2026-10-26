// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
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
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;
using Correction = gp::BoundaryCorrections::UpwindGradientProjection;
using gp::BoundaryCorrections::ExternalCorrection;
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

std::array<double, n43> flatten_target(const gp::VelocityVector& w,
                                       const gp::GradientVector& q) {
  std::array<double, n43> result{};
  for (size_t a = 0; a < gp::number_of_velocities; ++a) {
    gsl::at(result, a) = gsl::at(w, a);
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      gsl::at(result, 13 + 10 * i + a) = gsl::at(gsl::at(q, i), a);
    }
  }
  return result;
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

// Expected -1/2 (A_n + factor (|A_n| + sigma I)) [Y] from the independently
// assembled normal matrix and its eigenvector absolute value at the coefficient
// state u_average (by default the arithmetic mean u_* = (u^- + u^+)/2), for
// the configured Gamma-driver parameter Ccz4::fd::System::f, with sigma =
// scalar_penalty.
std::array<double, n43> expected_correction(
    const Side& interior, const Side& exterior, const size_t s,
    const double factor, const double scalar_penalty,
    const std::optional<DataVector>& coefficient_state = std::nullopt) {
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
  for (const auto external_correction :
       {ExternalCorrection::Central, ExternalCorrection::Upwind,
        ExternalCorrection::DoubledDissipation}) {
    const double factor =
        gp::BoundaryCorrections::external_upwind_factor(external_correction);
    for (const double sigma : {0.0, 1.0, 2.5}) {
      CAPTURE(factor);
      CAPTURE(sigma);
      // Equal penalties on all three characteristic groups reproduce the
      // previous componentwise scalar penalty.
      const Correction correction{external_correction, sigma, sigma, sigma,
                                  std::nullopt};
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

      // Internal faces always use unit dissipation plus the scalar jump
      // penalty sigma, and carry the value jump.
      const Output internal =
          boundary_terms<false>(correction, interior, exterior);
      // External faces scale the dissipative part and carry neither the
      // scalar jump penalty nor the value jump.
      const Output external =
          boundary_terms<true>(correction, interior, exterior);
      for (size_t s = 0; s < n; ++s) {
        const auto internal_expected =
            expected_correction(interior, exterior, s, 1.0, sigma);
        const auto external_expected =
            expected_correction(interior, exterior, s, factor, 0.0);
        const auto internal_actual = target_at(internal.w, internal.q, s);
        const auto external_actual = target_at(external.w, external.q, s);
        double scale = 1.0;
        for (const double x : internal_expected) {
          scale = std::max(scale, std::abs(x));
        }
        // The analytic transform and the dense reference differ by roundoff
        // amplified by the conditioning of the characteristic decomposition
        // at the averaged state, for which the speed spread is a proxy.
        gp::PotentialVector u_average{};
        for (size_t a = 0; a < gp::number_of_potentials; ++a) {
          gsl::at(u_average, a) =
              0.5 * (interior.u.get(a)[s] + exterior.u.get(a)[s]);
        }
        const gp::LogMetricChart average_chart{
            gp::stf_decode(&u_average[gp::PotentialIndex::log_metric])};
        const gp::NormalCharacteristics average_characteristics{
            average_chart, u_average, normals[s], ::Ccz4::fd::System::f};
        double largest_point_speed = 0.0;
        double smallest_point_speed = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < gp::number_of_normal_components; ++k) {
          const double speed =
              std::abs(gsl::at(average_characteristics.speeds(), k));
          largest_point_speed = std::max(largest_point_speed, speed);
          if (not gp::NormalCharacteristics::is_kernel_field(k)) {
            smallest_point_speed = std::min(smallest_point_speed, speed);
          }
        }
        const double epsilon =
            1.0e-12 *
            std::max(10.0, 10.0 * largest_point_speed / smallest_point_speed);
        CAPTURE(epsilon);
        for (size_t row = 0; row < n43; ++row) {
          CHECK(gsl::at(internal_actual, row) ==
                approx(gsl::at(internal_expected, row))
                    .scale(scale)
                    .epsilon(epsilon));
          CHECK(gsl::at(external_actual, row) ==
                approx(gsl::at(external_expected, row))
                    .scale(scale)
                    .epsilon(epsilon));
        }
        for (size_t a = 0; a < gp::number_of_potentials; ++a) {
          CHECK(internal.u.get(a)[s] ==
                approx(-0.5 * (exterior.u.get(a)[s] - interior.u.get(a)[s])));
          CHECK(external.u.get(a)[s] == 0.0);
        }
      }

      // The neighbor evaluates the same u_* with the reversed normal and
      // jump: the sum of the two sides' corrections is -A_n [Y] and the
      // difference is -(|A_n| + sigma I) [Y]; the value-jump inputs are
      // opposite.
      const Output neighbor =
          boundary_terms<false>(correction, exterior, interior);
      for (size_t s = 0; s < n; ++s) {
        const auto mine = target_at(internal.w, internal.q, s);
        const auto theirs = target_at(neighbor.w, neighbor.q, s);
        const auto central =
            expected_correction(interior, exterior, s, 0.0, 0.0);
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
}

// Upwind sign: a jump along a single characteristic mode with speed lambda
// (A_n r = lambda r for Y_t = A^i d_i Y) gets
// -1/2 (lambda + |lambda| + sigma) r, which after the strong-form lift (which
// subtracts L_f times the returned value) adds L_f (lambda + sigma / 2) r for
// an incoming (lambda > 0) mode and only the scalar penalty sigma L_f r / 2
// for an outgoing one (nothing for sigma = 0).
// At external faces the scalar penalty is absent, so outgoing modes receive
// no correction while incoming modes receive the ordinary upwind correction.
void test_upwind_sign(const double sigma) {
  CAPTURE(sigma);
  gp::PotentialVector u{};
  u[gp::PotentialIndex::lapse] = 0.8;
  u[gp::PotentialIndex::conformal_factor] = 0.9;
  u[gp::PotentialIndex::shift] = 0.1;
  const std::array<double, 3> normal{{0.6, 0.0, 0.8}};
  const gp::LogMetricChart chart{gp::Matrix3{}};
  const gp::NormalCharacteristics characteristics{chart, u, normal,
                                                  ::Ccz4::fd::System::f};
  const Correction correction{ExternalCorrection::Upwind, sigma, sigma, sigma,
                              std::nullopt};
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
    const Output external =
        boundary_terms<true>(correction, interior, exterior);
    const auto actual = target_at(output.w, output.q, 0);
    const auto external_actual = target_at(external.w, external.q, 0);
    const auto mode = target_at(exterior.w, exterior.q, 0);
    const double expected_factor = -0.5 * (speed + std::abs(speed) + sigma);
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
  // The accepted candidate penalties, exercising the grouped-penalty path.
  const Correction correction{ExternalCorrection::Upwind, 1.0, 1.0, 0.0,
                              std::nullopt};
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
  const Correction correction{ExternalCorrection::Upwind, 1.0, 1.0, 1.0,
                              std::nullopt};
  const Output output = boundary_terms<false>(correction, interior, exterior);
  for (size_t s = 0; s < 2; ++s) {
    const auto expected = expected_correction(interior, exterior, s, 1.0, 1.0);
    DataVector geometric{gp::number_of_potentials};
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      geometric[a] = interior.u.get(a)[s];
    }
    geometric[gp::PotentialIndex::conformal_factor] =
        std::sqrt(get<gp::PotentialIndex::conformal_factor>(interior.u)[s] *
                  get<gp::PotentialIndex::conformal_factor>(exterior.u)[s]);
    const auto control =
        expected_correction(interior, exterior, s, 1.0, 1.0, geometric);
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
  const Correction correction{ExternalCorrection::Upwind, 1.0, 1.0, 1.0,
                              std::nullopt};
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

// The grouped penalties at the accepted candidate (1,1,0): the correction is
// the uniform-penalty one plus half the GammaHat-b characteristic projection
// of the jump (the dissipation loses that projection and enters with -1/2),
// and jumps with equal GammaHat and b jumps receive the uniform correction.
void test_group_penalties(const gsl::not_null<std::mt19937*> gen) {
  const size_t n = 4;
  std::vector<std::array<double, 3>> normals{};
  for (size_t s = 0; s < n; ++s) {
    normals.push_back(s == 0 ? std::array<double, 3>{{0.0, 0.0, 1.0}}
                             : helpers::random_unit_vector(gen));
  }
  const Side interior = random_side(gen, n, normals, 1.0);
  Side exterior = random_side(gen, n, normals, -1.0);
  const Correction candidate{ExternalCorrection::Upwind, 1.0, 1.0, 0.0,
                             std::nullopt};
  const Correction uniform{ExternalCorrection::Upwind, 1.0, 1.0, 1.0,
                           std::nullopt};
  CHECK(candidate.propagating_scalar_penalty() == 1.0);
  CHECK(candidate.stationary_gradient_scalar_penalty() == 1.0);
  CHECK(candidate.gamma_hat_minus_b_scalar_penalty() == 0.0);
  const Output candidate_output =
      boundary_terms<false>(candidate, interior, exterior);
  const Output uniform_output =
      boundary_terms<false>(uniform, interior, exterior);
  for (size_t s = 0; s < n; ++s) {
    gp::PotentialVector u_average{};
    gp::VelocityVector jump_w{};
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      gsl::at(u_average, a) =
          0.5 * (interior.u.get(a)[s] + exterior.u.get(a)[s]);
    }
    for (size_t a = 0; a < gp::number_of_velocities; ++a) {
      gsl::at(jump_w, a) = exterior.w.get(a)[s] - interior.w.get(a)[s];
    }
    const gp::LogMetricChart chart{
        gp::stf_decode(&u_average[gp::PotentialIndex::log_metric])};
    const gp::NormalCharacteristics characteristics{
        chart, u_average, normals[s], ::Ccz4::fd::System::f};
    // P_g [Y] through the unchanged primitives: transform the velocity jump,
    // keep the kernel entries, invert and embed.
    gp::NormalSubspaceVector z{};
    for (size_t a = 0; a < gp::number_of_velocities; ++a) {
      gsl::at(z, a) = gsl::at(jump_w, a);
    }
    auto fields = characteristics.characteristic_fields(z);
    for (size_t k = 0; k < gp::number_of_normal_components; ++k) {
      if (not gp::NormalCharacteristics::is_kernel_field(k)) {
        gsl::at(fields, k) = 0.0;
      }
    }
    gp::VelocityVector projected_w{};
    gp::GradientVector projected_q{};
    gp::embed_normal_subspace_vector(
        make_not_null(&projected_w), make_not_null(&projected_q), normals[s],
        characteristics.normal_fields_from_characteristic_fields(fields));
    const auto projected = flatten_target(projected_w, projected_q);
    const auto candidate_actual =
        target_at(candidate_output.w, candidate_output.q, s);
    const auto uniform_actual =
        target_at(uniform_output.w, uniform_output.q, s);
    double scale = 1.0;
    for (size_t row = 0; row < n43; ++row) {
      scale = std::max(scale, std::abs(gsl::at(uniform_actual, row)));
    }
    for (size_t row = 0; row < n43; ++row) {
      CHECK(gsl::at(candidate_actual, row) ==
            approx(gsl::at(uniform_actual, row) + 0.5 * gsl::at(projected, row))
                .scale(scale)
                .epsilon(1e-11));
    }
    // The value-jump input does not depend on the penalties.
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      CHECK(candidate_output.u.get(a)[s] == uniform_output.u.get(a)[s]);
    }
  }
  // Equal GammaHat and b jumps: the candidate and uniform corrections agree.
  for (size_t s = 0; s < n; ++s) {
    for (size_t i = 0; i < 3; ++i) {
      exterior.w.get(gp::VelocityIndex::b + i)[s] =
          interior.w.get(gp::VelocityIndex::b + i)[s] +
          exterior.w.get(gp::VelocityIndex::gamma_hat + i)[s] -
          interior.w.get(gp::VelocityIndex::gamma_hat + i)[s];
    }
  }
  const Output candidate_equal =
      boundary_terms<false>(candidate, interior, exterior);
  const Output uniform_equal =
      boundary_terms<false>(uniform, interior, exterior);
  for (size_t s = 0; s < n; ++s) {
    const auto candidate_actual =
        target_at(candidate_equal.w, candidate_equal.q, s);
    const auto uniform_actual = target_at(uniform_equal.w, uniform_equal.q, s);
    double scale = 1.0;
    for (size_t row = 0; row < n43; ++row) {
      scale = std::max(scale, std::abs(gsl::at(uniform_actual, row)));
    }
    for (size_t row = 0; row < n43; ++row) {
      CHECK(gsl::at(candidate_actual, row) ==
            approx(gsl::at(uniform_actual, row)).scale(scale).epsilon(1e-11));
    }
  }
}

using Targets = gp::Tags::RhsTargets::type;

Mesh<2> shell_face_mesh(const size_t l_max) {
  return {{{l_max + 1, 2 * l_max + 1}},
          {{Spectral::Basis::SphericalHarmonic,
            Spectral::Basis::SphericalHarmonic}},
          {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}}};
}

// The tensor kinds of the filtered blocks: scalar, vector, symmetric rank two
// in the trace-free coordinates, general rank two and rank three symmetric in
// the last pair in its trace-free coordinates.
enum class Kind : size_t { Scalar, Vector, Symmetric, General, RankThree };

// The filtered blocks of the targets: the kind and the components in the
// packed order of the reference (the chart index fastest).
std::vector<std::pair<Kind, std::vector<DataVector*>>> filtered_blocks(
    const gsl::not_null<Targets*> targets) {
  auto& w = get<gp::Tags::VelocityTarget>(*targets);
  auto& q = get<gp::Tags::GradientTarget>(*targets);
  std::vector<std::pair<Kind, std::vector<DataVector*>>> blocks{};
  const auto velocities = [&w](const size_t first, const size_t size) {
    std::vector<DataVector*> result{};
    for (size_t a = first; a < first + size; ++a) {
      result.push_back(&w.get(a));
    }
    return result;
  };
  const auto gradients = [&q](const size_t first, const size_t size) {
    std::vector<DataVector*> result{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = first; a < first + size; ++a) {
        result.push_back(&q.get(i, a));
      }
    }
    return result;
  };
  blocks.emplace_back(Kind::Symmetric, velocities(gp::VelocityIndex::a, 5));
  blocks.emplace_back(
      Kind::Scalar,
      velocities(gp::VelocityIndex::trace_extrinsic_curvature, 1));
  blocks.emplace_back(Kind::Scalar, velocities(gp::VelocityIndex::theta, 1));
  blocks.emplace_back(Kind::Vector,
                      velocities(gp::VelocityIndex::gamma_hat, 3));
  blocks.emplace_back(Kind::Vector, velocities(gp::VelocityIndex::b, 3));
  blocks.emplace_back(Kind::RankThree,
                      gradients(gp::PotentialIndex::log_metric, 5));
  blocks.emplace_back(Kind::Vector,
                      gradients(gp::PotentialIndex::conformal_factor, 1));
  blocks.emplace_back(Kind::Vector, gradients(gp::PotentialIndex::lapse, 1));
  blocks.emplace_back(Kind::General, gradients(gp::PotentialIndex::shift, 3));
  return blocks;
}

DataVector packed(const std::vector<DataVector*>& components) {
  const size_t n = components.front()->size();
  DataVector result(components.size() * n);
  for (size_t c = 0; c < components.size(); ++c) {
    for (size_t j = 0; j < n; ++j) {
      result[j + n * c] = (*components[c])[j];
    }
  }
  return result;
}

// The dense native nodal filter I + Delta of `Structure` on its storage
// components (row and column j + n_Omega c), built from fill_filter and the
// Spherepack transforms of every unit nodal vector.
template <typename Structure>
Matrix storage_filter(const size_t l_max, const size_t modes_to_kill) {
  const auto& ylm = ylm::get_spherepack_cache(l_max);
  const size_t angular_size = ylm.physical_size();
  const size_t spectral_size = ylm.spectral_size();
  const size_t components = Structure::size();
  SimpleSparseMatrix delta{};
  ylm::TensorYlm::fill_filter<Structure>(
      make_not_null(&delta), l_max, modes_to_kill, std::nullopt,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);
  const size_t size = components * angular_size;
  Matrix result(size, size);
  std::vector<double> spectral(components * spectral_size, 0.0);
  for (size_t column = 0; column < size; ++column) {
    const size_t source = column / angular_size;
    DataVector values{angular_size, 0.0};
    values[column % angular_size] = 1.0;
    const DataVector coefficients = ylm.phys_to_spec(values);
    std::fill(spectral.begin(), spectral.end(), 0.0);
    for (size_t k = 0; k < spectral_size; ++k) {
      spectral[k + spectral_size * source] = coefficients[k];
    }
    std::vector<double> filtered = spectral;
    delta.increment_multiply_on_right(make_not_null(&filtered), 0, 1, spectral,
                                      0, 1);
    for (size_t c = 0; c < components; ++c) {
      DataVector filtered_coefficients{spectral_size};
      for (size_t k = 0; k < spectral_size; ++k) {
        filtered_coefficients[k] = filtered[k + spectral_size * c];
      }
      const DataVector filtered_values =
          ylm.spec_to_phys(filtered_coefficients);
      for (size_t j = 0; j < angular_size; ++j) {
        result(j + angular_size * c, column) = filtered_values[j];
      }
    }
  }
  return result;
}

// The decoding from the packed components to the tensor storage and the
// encoding back, from stf_decode and stf_encode on unit tensors.
template <typename Structure>
std::pair<Matrix, Matrix> packing(const Kind kind) {
  constexpr auto indices = Structure::storage_to_tensor_index();
  const size_t storage = Structure::size();
  const size_t packed_size =
      kind == Kind::Symmetric ? 5 : (kind == Kind::RankThree ? 15 : storage);
  Matrix decode(storage, packed_size, 0.0);
  Matrix encode(packed_size, storage, 0.0);
  for (size_t c = 0; c < storage; ++c) {
    if (kind == Kind::Scalar or kind == Kind::Vector) {
      decode(c, c) = 1.0;
      encode(c, c) = 1.0;
    } else if (kind == Kind::General) {
      decode(c, 3 * indices[c][0] + indices[c][1]) = 1.0;
      encode(3 * indices[c][0] + indices[c][1], c) = 1.0;
    } else {
      // The symmetric pair is the last two indices.
      const size_t first = kind == Kind::RankThree ? indices[c][0] : 0;
      const size_t j = indices[c][kind == Kind::RankThree ? 1 : 0];
      const size_t k = indices[c][kind == Kind::RankThree ? 2 : 1];
      gp::Matrix3 unit_tensor(0.0);
      unit_tensor(j, k) = 1.0;
      unit_tensor(k, j) = 1.0;
      const auto encoded = gp::stf_encode(unit_tensor);
      for (size_t a = 0; a < 5; ++a) {
        std::array<double, 5> unit{};
        gsl::at(unit, a) = 1.0;
        decode(c, 5 * first + a) = gp::stf_decode(unit.data())(j, k);
        encode(5 * first + a, c) = gsl::at(encoded, a);
      }
    }
  }
  return {decode, encode};
}

// E S D in the nodal components, (E S D)_{j + n a, j' + n b}.
Matrix packed_filter(const Matrix& storage, const Matrix& decode,
                     const Matrix& encode, const size_t angular_size) {
  const size_t size = decode.columns() * angular_size;
  Matrix decoded(storage.rows(), size, 0.0);
  for (size_t b = 0; b < decode.columns(); ++b) {
    for (size_t s = 0; s < decode.rows(); ++s) {
      if (decode(s, b) != 0.0) {
        blaze::submatrix(decoded, 0, b * angular_size, storage.rows(),
                         angular_size) +=
            decode(s, b) * blaze::submatrix(storage, 0, s * angular_size,
                                            storage.rows(), angular_size);
      }
    }
  }
  Matrix result(size, size, 0.0);
  for (size_t a = 0; a < encode.rows(); ++a) {
    for (size_t s = 0; s < encode.columns(); ++s) {
      if (encode(a, s) != 0.0) {
        blaze::submatrix(result, a * angular_size, 0, angular_size, size) +=
            encode(a, s) *
            blaze::submatrix(decoded, s * angular_size, 0, angular_size, size);
      }
    }
  }
  return result;
}

// The reference operators of the five kinds: the packed nodal filter, and the
// storage filter with its packing for the trace-free checks.
struct ReferenceFilters {
  std::array<Matrix, 5> packed{};
  std::array<Matrix, 5> storage{};
  std::array<std::pair<Matrix, Matrix>, 5> packings{};
};

ReferenceFilters reference_filters(const size_t l_max,
                                   const size_t modes_to_kill) {
  const size_t n = ylm::get_spherepack_cache(l_max).physical_size();
  ReferenceFilters result{};
  const auto set = [&]<typename Structure>(const Kind kind) {
    const auto k = static_cast<size_t>(kind);
    gsl::at(result.storage, k) =
        storage_filter<Structure>(l_max, modes_to_kill);
    gsl::at(result.packings, k) = packing<Structure>(kind);
    gsl::at(result.packed, k) = packed_filter(
        gsl::at(result.storage, k), gsl::at(result.packings, k).first,
        gsl::at(result.packings, k).second, n);
  };
  set.operator()<Scalar<DataVector>::structure>(Kind::Scalar);
  set.operator()<tnsr::i<DataVector, 3>::structure>(Kind::Vector);
  set.operator()<tnsr::ii<DataVector, 3>::structure>(Kind::Symmetric);
  set.operator()<tnsr::ij<DataVector, 3>::structure>(Kind::General);
  set.operator()<tnsr::ijj<DataVector, 3>::structure>(Kind::RankThree);
  return result;
}

Targets random_targets(const gsl::not_null<std::mt19937*> gen, const size_t n) {
  std::normal_distribution<double> normal(0.0, 1.0);
  Targets result{n};
  for (size_t i = 0; i < result.size(); ++i) {
    result.data()[i] = normal(*gen);
  }
  return result;
}

// Targets whose components are all band limited (degree at most L).
Targets band_limited_targets(const gsl::not_null<std::mt19937*> gen,
                             const size_t l_max) {
  const auto& ylm = ylm::get_spherepack_cache(l_max);
  std::normal_distribution<double> normal(0.0, 1.0);
  Targets result{ylm.physical_size()};
  for (size_t c = 0; c < Targets::number_of_independent_components; ++c) {
    DataVector coefficients{ylm.spectral_size(), 0.0};
    for (ylm::SpherepackIterator iterator(l_max, l_max); iterator; ++iterator) {
      coefficients[iterator()] = normal(*gen);
    }
    const DataVector values = ylm.spec_to_phys(coefficients);
    std::copy(values.begin(), values.end(),
              std::next(result.data(),
                        static_cast<std::ptrdiff_t>(c * values.size())));
  }
  return result;
}

DataVector matrix_times(const Matrix& matrix, const DataVector& vector) {
  DataVector result{matrix.rows(), 0.0};
  for (size_t column = 0; column < matrix.columns(); ++column) {
    for (size_t row = 0; row < matrix.rows(); ++row) {
      result[row] += matrix(row, column) * vector[column];
    }
  }
  return result;
}

double weighted_inner_product(const DataVector& lhs, const DataVector& rhs,
                              const std::vector<double>& weights) {
  double result = 0.0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    result += weights[i % weights.size()] * lhs[i] * rhs[i];
  }
  return result;
}

Targets filtered(const Correction& correction, Targets targets,
                 const Mesh<2>& face_mesh, const bool nonconforming = true) {
  correction.dg_filter_boundary_correction(make_not_null(&targets), face_mesh,
                                           nonconforming);
  return targets;
}

// The hook on random spherical-harmonic faces against independently built
// dense native filters of the five tensor kinds, with the trace-free,
// idempotency and self-adjointness checks of each block.
void test_shell_boundary_filter(const gsl::not_null<std::mt19937*> gen) {
  for (const auto& [l_max, modes_to_kill] :
       std::array<std::pair<size_t, size_t>, 5>{
           {{2, 0}, {2, 2}, {4, 0}, {4, 4}, {5, 3}}}) {
    CAPTURE(l_max);
    CAPTURE(modes_to_kill);
    const Mesh<2> face_mesh = shell_face_mesh(l_max);
    const auto& ylm = ylm::get_spherepack_cache(l_max);
    const std::vector<double>& weights = ylm.integration_weights();
    const size_t n = face_mesh.number_of_grid_points();
    REQUIRE(n == ylm.physical_size());
    const Correction correction{ExternalCorrection::Upwind, 1.0, 1.0, 0.0,
                                modes_to_kill};
    const ReferenceFilters reference = reference_filters(l_max, modes_to_kill);
    Targets input = random_targets(gen, n);
    const Targets output = filtered(correction, input, face_mesh);
    Targets second_input = random_targets(gen, n);
    const Targets second_output = filtered(correction, second_input, face_mesh);
    Targets twice = filtered(correction, output, face_mesh);
    Targets output_copy = output;
    Targets second_output_copy = second_output;

    // The provisional potential-rate correction is never filtered.
    CHECK(get<gp::Tags::PotentialRateTarget>(output) ==
          get<gp::Tags::PotentialRateTarget>(input));

    const auto input_blocks = filtered_blocks(make_not_null(&input));
    const auto output_blocks = filtered_blocks(make_not_null(&output_copy));
    const auto second_input_blocks =
        filtered_blocks(make_not_null(&second_input));
    const auto second_output_blocks =
        filtered_blocks(make_not_null(&second_output_copy));
    const auto twice_blocks = filtered_blocks(make_not_null(&twice));
    size_t filtered_components = 0;
    for (size_t b = 0; b < input_blocks.size(); ++b) {
      const Kind kind = input_blocks[b].first;
      CAPTURE(b);
      const auto k = static_cast<size_t>(kind);
      filtered_components += input_blocks[b].second.size();
      const DataVector x = packed(input_blocks[b].second);
      const DataVector y = packed(output_blocks[b].second);
      const Matrix& packed_matrix = gsl::at(reference.packed, k);
      const DataVector expected = matrix_times(packed_matrix, x);
      // Relative to the input: the filtered data can be much smaller.
      const auto custom = Approx::custom().epsilon(1.0e-12).scale(
          std::max(max(abs(x)), max(abs(expected))));
      CHECK_ITERABLE_CUSTOM_APPROX(y, expected, custom);
      // Idempotent.
      CHECK_ITERABLE_CUSTOM_APPROX(packed(twice_blocks[b].second), y, custom);
      // Self-adjoint in the angular weights: an orthogonal projector.
      const DataVector x2 = packed(second_input_blocks[b].second);
      const DataVector y2 = packed(second_output_blocks[b].second);
      CHECK(weighted_inner_product(y, x2, weights) ==
            Approx::custom().epsilon(1.0e-12).scale(
                std::sqrt(weighted_inner_product(x, x, weights) *
                          weighted_inner_product(x2, x2, weights)))(
                weighted_inner_product(x, y2, weights)));
      // The filter changes the random data.
      CHECK(max(abs(y - x)) > 1.0e-3);
      if (kind == Kind::Symmetric or kind == Kind::RankThree) {
        // The native filter of the decoded symmetric trace-free tensors
        // stays trace free and agrees with the packed-slot path.
        const auto& [decode, encode] = gsl::at(reference.packings, k);
        const Matrix& storage = gsl::at(reference.storage, k);
        DataVector decoded(decode.rows() * n, 0.0);
        for (size_t s = 0; s < decode.rows(); ++s) {
          for (size_t a = 0; a < decode.columns(); ++a) {
            for (size_t j = 0; j < n; ++j) {
              decoded[j + n * s] += decode(s, a) * x[j + n * a];
            }
          }
        }
        const DataVector native = matrix_times(storage, decoded);
        DataVector redecoded(decode.rows() * n, 0.0);
        for (size_t s = 0; s < decode.rows(); ++s) {
          for (size_t a = 0; a < decode.columns(); ++a) {
            for (size_t j = 0; j < n; ++j) {
              redecoded[j + n * s] += decode(s, a) * y[j + n * a];
            }
          }
        }
        CHECK_ITERABLE_CUSTOM_APPROX(redecoded, native, custom);
        const size_t pairs = kind == Kind::Symmetric ? 1 : 3;
        for (size_t i = 0; i < pairs; ++i) {
          DataVector trace{n, 0.0};
          for (size_t d = 0; d < 3; ++d) {
            const size_t s =
                kind == Kind::Symmetric
                    ? tnsr::ii<DataVector, 3>::structure::get_storage_index(d,
                                                                            d)
                    : tnsr::ijj<DataVector, 3>::structure::get_storage_index(
                          i, d, d);
            for (size_t j = 0; j < n; ++j) {
              trace[j] += native[j + n * s];
            }
          }
          CHECK(max(abs(trace)) < 1.0e-12 * max(abs(decoded)));
        }
      }
    }
    CHECK(filtered_components == 43);

    // The native zero-band filter is not a bypass: on band-limited data the
    // scalars are unchanged while the tensor content of total degree above L
    // is removed from every other block.
    if (modes_to_kill == 0) {
      Targets limited = band_limited_targets(gen, l_max);
      Targets limited_output = filtered(correction, limited, face_mesh);
      const auto limited_blocks = filtered_blocks(make_not_null(&limited));
      const auto limited_output_blocks =
          filtered_blocks(make_not_null(&limited_output));
      for (size_t b = 0; b < limited_blocks.size(); ++b) {
        CAPTURE(b);
        const DataVector x = packed(limited_blocks[b].second);
        const DataVector y = packed(limited_output_blocks[b].second);
        if (limited_blocks[b].first == Kind::Scalar) {
          CHECK_ITERABLE_CUSTOM_APPROX(
              y, x, Approx::custom().epsilon(1.0e-12).scale(max(abs(x))));
        } else {
          CHECK(max(abs(y - x)) > 1.0e-2 * max(abs(x)));
        }
      }
    }
  }
}

// The hook does nothing, bitwise, without a cutoff, on a conforming
// connection or on a face that is not spherical-harmonic, and rejects a
// cutoff above L and inconsistent face data.
void test_shell_boundary_filter_activation(
    const gsl::not_null<std::mt19937*> gen) {
  const size_t l_max = 4;
  const Mesh<2> face_mesh = shell_face_mesh(l_max);
  const size_t n = face_mesh.number_of_grid_points();
  const Targets input = random_targets(gen, n);
  const Correction disabled{ExternalCorrection::Upwind, 1.0, 1.0, 0.0,
                            std::nullopt};
  const Correction enabled{ExternalCorrection::Upwind, 1.0, 1.0, 0.0,
                           size_t{4}};
  const Correction zero_bands{ExternalCorrection::Upwind, 1.0, 1.0, 0.0,
                              size_t{0}};
  CHECK(filtered(disabled, input, face_mesh) == input);
  CHECK(filtered(enabled, input, face_mesh, false) == input);
  CHECK(filtered(zero_bands, input, face_mesh, false) == input);
  CHECK(filtered(enabled, input, face_mesh) != input);
  CHECK(filtered(zero_bands, input, face_mesh) != input);
  const Mesh<2> legendre_face{3, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
  const Targets legendre_input = random_targets(gen, 9);
  CHECK(filtered(enabled, legendre_input, legendre_face) == legendre_input);
  // The cutoff is checked only where the filter acts.
  const Correction too_many{ExternalCorrection::Upwind, 1.0, 1.0, 0.0,
                            size_t{5}};
  CHECK(filtered(too_many, input, face_mesh, false) == input);
  CHECK(filtered(too_many, legendre_input, legendre_face) == legendre_input);
  CHECK_THROWS_WITH(
      filtered(too_many, input, face_mesh),
      Catch::Matchers::ContainsSubstring(
          "ShellBoundaryModesToKill = 5 of the shell boundary-correction "
          "filter exceeds L = 4 of the face mesh"));
  CHECK_THROWS_WITH(filtered(enabled, random_targets(gen, n + 1), face_mesh),
                    Catch::Matchers::ContainsSubstring(
                        "and " + std::to_string(n + 1) + " correction points"));
  const Mesh<2> wrong_extents{
      {{l_max + 1, 2 * l_max}},
      {{Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}}};
  CHECK_THROWS_WITH(
      filtered(enabled, random_targets(gen, (l_max + 1) * 2 * l_max),
               wrong_extents),
      Catch::Matchers::ContainsSubstring(
          "requires a spherical-harmonic face mesh (L + 1, 2 L + 1)"));
}

// The time of one application of the hook to all 43 components of a face
// with L = 10 and four removed bands.
void test_shell_boundary_filter_timing(const gsl::not_null<std::mt19937*> gen) {
  const size_t l_max = 10;
  const Mesh<2> face_mesh = shell_face_mesh(l_max);
  const Correction correction{ExternalCorrection::Upwind, 1.0, 1.0, 0.0,
                              size_t{4}};
  Targets targets = random_targets(gen, face_mesh.number_of_grid_points());
  // Build the cached filters.
  correction.dg_filter_boundary_correction(make_not_null(&targets), face_mesh,
                                           true);
#ifdef SPECTRE_DEBUG
  const size_t repetitions = 10;
#else
  const size_t repetitions = 1000;
#endif
  const auto start = std::chrono::steady_clock::now();
  for (size_t r = 0; r < repetitions; ++r) {
    correction.dg_filter_boundary_correction(make_not_null(&targets), face_mesh,
                                             true);
  }
  const auto stop = std::chrono::steady_clock::now();
  const double ns_per_application =
      static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start)
              .count()) /
      static_cast<double>(repetitions);
  Parallel::printf(
      "UpwindGradientProjection shell boundary filter (L = 10, 4 removed "
      "bands, all 43 components): %.0f ns per application (checksum %.6e)\n",
      ns_per_application, targets.data()[0]);
}

void test_options_and_errors(const gsl::not_null<std::mt19937*> gen) {
  register_factory_classes_with_charm<Metavariables>();
  const auto created =
      TestHelpers::test_creation<std::unique_ptr<evolution::BoundaryCorrection>,
                                 Metavariables>(
          "UpwindGradientProjection:\n"
          "  ExternalCorrection: DoubledDissipation\n"
          "  PropagatingScalarPenalty: 0.5\n"
          "  StationaryGradientScalarPenalty: 0.25\n"
          "  GammaHatMinusBScalarPenalty: 0.75\n"
          "  ShellBoundaryModesToKill: 4\n");
  const auto* const typed = dynamic_cast<const Correction*>(created.get());
  REQUIRE(typed != nullptr);
  CHECK(typed->external_correction() == ExternalCorrection::DoubledDissipation);
  CHECK(typed->propagating_scalar_penalty() == 0.5);
  CHECK(typed->stationary_gradient_scalar_penalty() == 0.25);
  CHECK(typed->gamma_hat_minus_b_scalar_penalty() == 0.75);
  CHECK(typed->shell_boundary_modes_to_kill() == std::optional<size_t>{4});
  const auto serialized = serialize_and_deserialize(*typed);
  CHECK(serialized.external_correction() ==
        ExternalCorrection::DoubledDissipation);
  CHECK(serialized.propagating_scalar_penalty() == 0.5);
  CHECK(serialized.stationary_gradient_scalar_penalty() == 0.25);
  CHECK(serialized.gamma_hat_minus_b_scalar_penalty() == 0.75);
  CHECK(serialized.shell_boundary_modes_to_kill() == std::optional<size_t>{4});
  const auto clone = typed->get_clone();
  const auto& typed_clone = dynamic_cast<const Correction&>(*clone);
  CHECK(typed_clone.external_correction() ==
        ExternalCorrection::DoubledDissipation);
  CHECK(typed_clone.propagating_scalar_penalty() == 0.5);
  CHECK(typed_clone.stationary_gradient_scalar_penalty() == 0.25);
  CHECK(typed_clone.gamma_hat_minus_b_scalar_penalty() == 0.75);
  CHECK(typed_clone.shell_boundary_modes_to_kill() == std::optional<size_t>{4});
  // The serialized and cloned cutoffs filter like the original.
  {
    const Mesh<2> face_mesh = shell_face_mesh(4);
    const Targets input =
        random_targets(gen, face_mesh.number_of_grid_points());
    const Targets expected = filtered(*typed, input, face_mesh);
    CHECK(expected != input);
    CHECK(filtered(serialized, input, face_mesh) == expected);
    CHECK(filtered(typed_clone, input, face_mesh) == expected);
  }
  // None and zero removed bands parse and are distinct; None survives
  // serialization.
  for (const auto& [value, expected] :
       std::array<std::pair<std::string, std::optional<size_t>>, 2>{
           {{"None", std::nullopt}, {"0", size_t{0}}}}) {
    CAPTURE(value);
    const auto parsed = TestHelpers::test_creation<
        std::unique_ptr<evolution::BoundaryCorrection>, Metavariables>(
        "UpwindGradientProjection:\n"
        "  ExternalCorrection: Upwind\n"
        "  PropagatingScalarPenalty: 1.0\n"
        "  StationaryGradientScalarPenalty: 1.0\n"
        "  GammaHatMinusBScalarPenalty: 0.0\n"
        "  ShellBoundaryModesToKill: " +
        value + "\n");
    const auto& parsed_typed = dynamic_cast<const Correction&>(*parsed);
    CHECK(parsed_typed.shell_boundary_modes_to_kill() == expected);
    CHECK(serialize_and_deserialize(parsed_typed)
              .shell_boundary_modes_to_kill() == expected);
  }
  // The option is required: an input without it fails to parse.
  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<
          std::unique_ptr<evolution::BoundaryCorrection>, Metavariables>(
          "UpwindGradientProjection:\n"
          "  ExternalCorrection: Upwind\n"
          "  PropagatingScalarPenalty: 1.0\n"
          "  StationaryGradientScalarPenalty: 1.0\n"
          "  GammaHatMinusBScalarPenalty: 0.0\n")),
      Catch::Matchers::ContainsSubstring("ShellBoundaryModesToKill"));
  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<
          std::unique_ptr<evolution::BoundaryCorrection>, Metavariables>(
          "UpwindGradientProjection:\n"
          "  ExternalCorrection: Upwind\n"
          "  PropagatingScalarPenalty: 1.0\n"
          "  StationaryGradientScalarPenalty: 1.0\n"
          "  GammaHatMinusBScalarPenalty: 0.0\n"
          "  ShellBoundaryModesToKill: -1\n")),
      Catch::Matchers::ContainsSubstring("ShellBoundaryModesToKill"));

  // The remaining enum values parse; every option is required and every
  // penalty has a zero lower bound.
  {
    const auto central = TestHelpers::test_creation<
        std::unique_ptr<evolution::BoundaryCorrection>, Metavariables>(
        "UpwindGradientProjection:\n"
        "  ExternalCorrection: Central\n"
        "  PropagatingScalarPenalty: 0.5\n"
        "  StationaryGradientScalarPenalty: 0.5\n"
        "  GammaHatMinusBScalarPenalty: 0.0\n"
        "  ShellBoundaryModesToKill: None\n");
    const auto& central_typed = dynamic_cast<const Correction&>(*central);
    CHECK(central_typed.external_correction() == ExternalCorrection::Central);
    CHECK(central_typed.stationary_gradient_scalar_penalty() == 0.5);
  }
  CHECK(dynamic_cast<const Correction&>(
            *TestHelpers::test_creation<
                std::unique_ptr<evolution::BoundaryCorrection>, Metavariables>(
                "UpwindGradientProjection:\n"
                "  ExternalCorrection: Upwind\n"
                "  PropagatingScalarPenalty: 1.0\n"
                "  StationaryGradientScalarPenalty: 1.0\n"
                "  GammaHatMinusBScalarPenalty: 0.0\n"
                "  ShellBoundaryModesToKill: None\n"))
            .external_correction() == ExternalCorrection::Upwind);

  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<
          std::unique_ptr<evolution::BoundaryCorrection>, Metavariables>(
          "UpwindGradientProjection:\n"
          "  ExternalCorrection: Upwind\n"
          "  PropagatingScalarPenalty: -0.5\n"
          "  StationaryGradientScalarPenalty: 1.0\n"
          "  GammaHatMinusBScalarPenalty: 0.0\n"
          "  ShellBoundaryModesToKill: None\n")),
      Catch::Matchers::ContainsSubstring(
          "Value -0.5 is below the lower bound of 0"));
  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<
          std::unique_ptr<evolution::BoundaryCorrection>, Metavariables>(
          "UpwindGradientProjection:\n"
          "  ExternalCorrection: Upwind\n"
          "  PropagatingScalarPenalty: 1.0\n"
          "  StationaryGradientScalarPenalty: -0.75\n"
          "  GammaHatMinusBScalarPenalty: 0.0\n"
          "  ShellBoundaryModesToKill: None\n")),
      Catch::Matchers::ContainsSubstring(
          "Value -0.75 is below the lower bound of 0"));
  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<
          std::unique_ptr<evolution::BoundaryCorrection>, Metavariables>(
          "UpwindGradientProjection:\n"
          "  ExternalCorrection: Upwind\n"
          "  PropagatingScalarPenalty: 1.0\n"
          "  StationaryGradientScalarPenalty: 1.0\n"
          "  GammaHatMinusBScalarPenalty: -0.25\n"
          "  ShellBoundaryModesToKill: None\n")),
      Catch::Matchers::ContainsSubstring(
          "Value -0.25 is below the lower bound of 0"));
  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<
          std::unique_ptr<evolution::BoundaryCorrection>, Metavariables>(
          "UpwindGradientProjection:\n"
          "  ExternalCorrection: OneSided\n"
          "  PropagatingScalarPenalty: 1.0\n"
          "  StationaryGradientScalarPenalty: 1.0\n"
          "  GammaHatMinusBScalarPenalty: 0.0\n"
          "  ShellBoundaryModesToKill: None\n")),
      Catch::Matchers::ContainsSubstring(
          "Failed to convert \"OneSided\" to ExternalCorrection"));
  CHECK_THROWS_WITH(
      (Correction{ExternalCorrection::Upwind, 1.0, -0.5, 0.0, std::nullopt}),
      Catch::Matchers::ContainsSubstring(
          "The scalar jump penalties must be finite and nonnegative"));

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
  for (const double sigma : {0.0, 1.0, 2.5}) {
    test_upwind_sign(sigma);
  }
  test_permutation(make_not_null(&gen));
  test_group_penalties(make_not_null(&gen));
  test_unequal_conformal_factors(make_not_null(&gen));
  test_package_errors(make_not_null(&gen));
  test_shell_boundary_filter(make_not_null(&gen));
  test_shell_boundary_filter_activation(make_not_null(&gen));
  test_shell_boundary_filter_timing(make_not_null(&gen));
  test_options_and_errors(make_not_null(&gen));
}
