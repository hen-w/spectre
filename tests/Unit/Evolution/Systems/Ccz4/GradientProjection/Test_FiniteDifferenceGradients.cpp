// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/FiniteDifferenceGradients.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;
using helpers::flatten;
constexpr const char* module = "GradientProjectionReference";

gp::Matrix3 metric_at(const tnsr::ii<DataVector, 3>& metric,
                      const size_t point) {
  gp::Matrix3 result{};
  for (size_t j = 0; j < 3; ++j) {
    for (size_t k = 0; k < 3; ++k) {
      result(j, k) = metric.get(j, k)[point];
    }
  }
  return result;
}

gp::Matrix3 derivative_at(const tnsr::ijj<DataVector, 3>& d_metric,
                          const size_t i, const size_t point) {
  gp::Matrix3 result{};
  for (size_t j = 0; j < 3; ++j) {
    for (size_t k = 0; k < 3; ++k) {
      result(j, k) = d_metric.get(i, j, k)[point];
    }
  }
  return result;
}

gp::Matrix3 log_metric_gradient_at(const gp::Tags::PotentialGradients::type& q,
                                   const size_t i, const size_t point) {
  std::array<double, 5> components{};
  for (size_t c = 0; c < 5; ++c) {
    gsl::at(components, c) =
        q.get(i, gp::PotentialIndex::log_metric + c)[point];
  }
  return gp::stf_decode(components.data());
}

// Pointwise: the transmitted metric gradient is the STF part of
// T_H^{-1} d(gtilde), i.e. T_H q_H = d(gtilde) - gtilde tr(gtilde^{-1}
// d(gtilde)) / 3 (checked with SciPy's matrix logarithm and Frechet
// derivative of the exponential); for a tangent d(gtilde) = T_H X with STF X,
// q_H = X; the other gradients are the raw derivatives, bitwise.
void test_pointwise(const gsl::not_null<std::mt19937*> gen) {
  const std::array<std::array<double, 3>, 5> spectra{{{{0.0, 0.0, 0.0}},
                                                      {{0.2, -0.1, -0.1}},
                                                      {{0.5, 0.1, -0.6}},
                                                      {{1.2, -0.3, -0.9}},
                                                      {{0.8, 0.8, -1.6}}}};
  const size_t number_of_points = spectra.size();
  std::normal_distribution<double> normal(0.0, 1.0);
  tnsr::ii<DataVector, 3> metric{number_of_points};
  tnsr::ijj<DataVector, 3> d_metric{number_of_points};
  tnsr::ijj<DataVector, 3> d_metric_tangent{number_of_points};
  std::array<std::array<gp::Matrix3, 3>, 5> tangent_directions{};
  for (size_t s = 0; s < number_of_points; ++s) {
    const gp::LogMetricChart chart{
        helpers::log_metric_with_spectrum(gen, gsl::at(spectra, s))};
    const gp::Matrix3 g = chart.metric();
    for (size_t i = 0; i < 3; ++i) {
      // A generic (not tangent) derivative, including a trace component.
      const gp::Matrix3 dg = helpers::random_symmetric(gen, 1.0);
      // A tangent derivative T_H X with STF X (from SciPy).
      gp::Matrix3 x = helpers::random_symmetric(gen, 1.0);
      x -= ((x(0, 0) + x(1, 1) + x(2, 2)) / 3.0) *
           gp::Matrix3{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
      gsl::at(gsl::at(tangent_directions, s), i) = x;
      const DataVector tangent = pypp::call<DataVector>(
          module, "tangent", flatten(chart.log_metric()), flatten(x));
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = j; k < 3; ++k) {
          d_metric.get(i, j, k)[s] = dg(j, k);
          d_metric_tangent.get(i, j, k)[s] = tangent[3 * j + k];
        }
      }
    }
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = j; k < 3; ++k) {
        metric.get(j, k)[s] = g(j, k);
      }
    }
  }
  tnsr::i<DataVector, 3> d_conformal_factor{number_of_points};
  tnsr::i<DataVector, 3> d_lapse{number_of_points};
  tnsr::iJ<DataVector, 3> d_shift{number_of_points};
  for (auto* const tensor : {&d_conformal_factor, &d_lapse}) {
    for (auto& component : *tensor) {
      for (double& value : component) {
        value = normal(*gen);
      }
    }
  }
  for (auto& component : d_shift) {
    for (double& value : component) {
      value = normal(*gen);
    }
  }

  for (const bool tangent_input : {false, true}) {
    CAPTURE(tangent_input);
    const auto& input = tangent_input ? d_metric_tangent : d_metric;
    gp::Tags::PotentialGradients::type q{number_of_points, 3.0e3};
    gp::finite_difference_potential_gradients(
        make_not_null(&q), metric, input, d_conformal_factor, d_lapse, d_shift);
    for (size_t i = 0; i < 3; ++i) {
      CHECK(q.get(i, gp::PotentialIndex::conformal_factor) ==
            d_conformal_factor.get(i));
      CHECK(q.get(i, gp::PotentialIndex::lapse) == d_lapse.get(i));
      for (size_t k = 0; k < 3; ++k) {
        CHECK(q.get(i, gp::PotentialIndex::shift + k) == d_shift.get(i, k));
      }
    }
    for (size_t s = 0; s < number_of_points; ++s) {
      const gp::Matrix3 g = metric_at(metric, s);
      const DataVector h_scipy =
          pypp::call<DataVector>(module, "log_metric", flatten(g));
      const gp::Matrix3 inverse_g = blaze::inv(g);
      for (size_t i = 0; i < 3; ++i) {
        CAPTURE(s);
        CAPTURE(i);
        const gp::Matrix3 dg = derivative_at(input, i, s);
        const gp::Matrix3 q_h = log_metric_gradient_at(q, i, s);
        const gp::Matrix3 inverse_g_dg = inverse_g * dg;
        const double trace =
            inverse_g_dg(0, 0) + inverse_g_dg(1, 1) + inverse_g_dg(2, 2);
        const gp::Matrix3 projected = dg - (trace / 3.0) * g;
        const DataVector pushed_forward =
            pypp::call<DataVector>(module, "tangent", h_scipy, flatten(q_h));
        CHECK(helpers::max_abs_difference(projected, pushed_forward) <
              1e-12 * (1.0 + helpers::max_abs(dg)));
        if (tangent_input) {
          CHECK(helpers::max_abs(
                    q_h - gsl::at(gsl::at(tangent_directions, s), i)) < 1e-12);
        }
      }
    }
  }

  // Off the unit-determinant surface beyond the chart tolerance.
  tnsr::ii<DataVector, 3> scaled_metric = metric;
  for (auto& component : scaled_metric) {
    component *= 1.0 + 1.0e-6;
  }
  gp::Tags::PotentialGradients::type q{number_of_points};
  CHECK_THROWS_WITH(gp::finite_difference_potential_gradients(
                        make_not_null(&q), scaled_metric, d_metric,
                        d_conformal_factor, d_lapse, d_shift),
                    Catch::Matchers::ContainsSubstring(
                        "is off the unit-determinant surface"));
  // Within the tolerance it is accepted (the trace-free chart is used).
  scaled_metric = metric;
  for (auto& component : scaled_metric) {
    component *= 1.0 + 1.0e-12;
  }
  CHECK_NOTHROW(gp::finite_difference_potential_gradients(
      make_not_null(&q), scaled_metric, d_metric, d_conformal_factor, d_lapse,
      d_shift));
}

struct Metavariables {
  using component_list = tmpl::list<>;
};
struct Component {};

constexpr double gradient_poison = 3.0e3;

// Smooth fields with linear potentials and a log-metric linear in the
// coordinates, so the conformal metric is exactly on the unit-determinant
// surface; the stored gradients are poisoned.
Variables<Ccz4::fd::System::variables_tag_list> smooth_solution(
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& coords) {
  const size_t number_of_points = get<0>(coords).size();
  Variables<Ccz4::fd::System::variables_tag_list> vars{number_of_points, 0.0};
  auto& metric = get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(vars);
  for (size_t s = 0; s < number_of_points; ++s) {
    const double x = get<0>(coords)[s];
    const double y = get<1>(coords)[s];
    const double z = get<2>(coords)[s];
    const gp::Matrix3 h{{0.05 * x - 0.02 * z, 0.03 * y, -0.01 * x},
                        {0.03 * y, -0.04 * x + 0.01 * y, 0.02 * z},
                        {-0.01 * x, 0.02 * z, -0.01 * x - 0.01 * y + 0.02 * z}};
    const gp::Matrix3 g = gp::LogMetricChart{h}.metric();
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = j; k < 3; ++k) {
        metric.get(j, k)[s] = g(j, k);
      }
    }
  }
  get(get<Ccz4::Tags::ConformalFactor<DataVector>>(vars)) =
      0.9 + 0.02 * get<0>(coords) - 0.01 * get<1>(coords) +
      0.015 * get<2>(coords);
  get(get<gr::Tags::Lapse<DataVector>>(vars)) = 0.8 - 0.01 * get<0>(coords) +
                                                0.02 * get<1>(coords) -
                                                0.005 * get<2>(coords);
  for (size_t k = 0; k < 3; ++k) {
    get<gr::Tags::Shift<DataVector, 3>>(vars).get(k) =
        0.01 * static_cast<double>(k + 1) * coords.get(k) +
        0.003 * coords.get((k + 1) % 3);
    get(get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars)) +=
        0.1 * coords.get(k);
  }
  for (auto& component : get<Ccz4::fd::Tags::PotentialGradients>(vars)) {
    component = gradient_poison;
  }
  return vars;
}

// The action overwrites only the stored gradients, with
// finite_difference_potential_gradients of the fourth-order derivatives that
// use the ghost data in the DataBox.
void test_action() {
  const Mesh<3> mesh{5, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered};
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(mesh);
  const Element<3> element = TestHelpers::Ccz4::fd::detail::set_element();
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian{mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    inverse_jacobian.get(i, i) = 1.0;
  }
  const auto ghost_data = TestHelpers::Ccz4::fd::detail::compute_ghost_data(
      mesh, logical_coords, element.neighbors(), 3, smooth_solution);
  const auto evolved_vars = smooth_solution(logical_coords);

  for (const bool evolve_lapse_and_shift : {true, false}) {
    CAPTURE(evolve_lapse_and_shift);
    auto box = db::create<db::AddSimpleTags<
        Ccz4::fd::Tags::EvolveLapseAndShift,
        evolution::dg::subcell::Tags::Mesh<3>, Ccz4::fd::System::variables_tag,
        evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
        evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<3>>>(
        evolve_lapse_and_shift, mesh, evolved_vars, ghost_data,
        inverse_jacobian);
    tuples::TaggedTuple<> inboxes{};
    const Parallel::GlobalCache<Metavariables> cache{};
    const auto apply = [&box, &inboxes, &cache]() {
      return gp::Actions::ComputeFiniteDifferenceGradients::apply(
          box, inboxes, cache, 0, tmpl::list<>{},
          static_cast<const Component*>(nullptr));
    };
    if (not evolve_lapse_and_shift) {
      CHECK_THROWS_WITH(apply(), Catch::Matchers::ContainsSubstring(
                                     "must too (EvolveLapseAndShift)"));
      continue;
    }
    apply();

    using deriv_tags =
        db::wrap_tags_in<::Tags::deriv, Ccz4::fd::System::gradients_tags,
                         tmpl::size_t<3>, Frame::Inertial>;
    Variables<deriv_tags> derivatives{mesh.number_of_grid_points()};
    Ccz4::fd::spacetime_derivatives(make_not_null(&derivatives), evolved_vars,
                                    ghost_data, 4, mesh, inverse_jacobian);
    gp::Tags::PotentialGradients::type expected{mesh.number_of_grid_points()};
    gp::finite_difference_potential_gradients(
        make_not_null(&expected),
        get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(evolved_vars),
        get<::Tags::deriv<Ccz4::Tags::ConformalMetric<DataVector, 3>,
                          tmpl::size_t<3>, Frame::Inertial>>(derivatives),
        get<::Tags::deriv<Ccz4::Tags::ConformalFactor<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>>(derivatives),
        get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                          Frame::Inertial>>(derivatives),
        get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                          Frame::Inertial>>(derivatives));
    const auto& result = db::get<Ccz4::fd::System::variables_tag>(box);
    CHECK(get<gp::Tags::PotentialGradients>(result) == expected);
    // The physical fields are unchanged.
    tmpl::for_each<Ccz4::fd::System::original_evolved_variables_tags>(
        [&result, &evolved_vars]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          CHECK(get<Tag>(result) == get<Tag>(evolved_vars));
        });

    // Independent check of the values: the potentials are linear, so their
    // fourth-order derivatives are exact; the metric gradient approximates
    // the analytic dH to the fourth-order truncation error
    // ~ dx^4 |2 dH|^5 e^{2|H|} / 30 ~ 1e-8 (dx = 0.4, |2 dH| ~ 0.1).
    const auto& q = get<gp::Tags::PotentialGradients>(result);
    const std::array<double, 3> d_phi{{0.02, -0.01, 0.015}};
    const std::array<double, 3> d_lapse{{-0.01, 0.02, -0.005}};
    const std::array<gp::Matrix3, 3> d_h{
        {gp::Matrix3{
             {0.05, 0.0, -0.01}, {0.0, -0.04, 0.0}, {-0.01, 0.0, -0.01}},
         gp::Matrix3{{0.0, 0.03, 0.0}, {0.03, 0.01, 0.0}, {0.0, 0.0, -0.01}},
         gp::Matrix3{{-0.02, 0.0, 0.0}, {0.0, 0.0, 0.02}, {0.0, 0.02, 0.02}}}};
    for (size_t s = 0; s < mesh.number_of_grid_points(); ++s) {
      for (size_t i = 0; i < 3; ++i) {
        CHECK(q.get(i, gp::PotentialIndex::conformal_factor)[s] ==
              approx(gsl::at(d_phi, i)));
        CHECK(q.get(i, gp::PotentialIndex::lapse)[s] ==
              approx(gsl::at(d_lapse, i)));
        for (size_t k = 0; k < 3; ++k) {
          const double expected_d_shift =
              (i == k ? 0.01 * static_cast<double>(k + 1) : 0.0) +
              (i == (k + 1) % 3 ? 0.003 : 0.0);
          CHECK(q.get(i, gp::PotentialIndex::shift + k)[s] ==
                approx(expected_d_shift));
        }
        CHECK(helpers::max_abs(log_metric_gradient_at(q, i, s) -
                               gsl::at(d_h, i)) < 5e-8);
      }
    }
  }
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection.FiniteDifferenceGradients",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Evolution/Systems/Ccz4/GradientProjection/");
  MAKE_GENERATOR(gen);
  test_pointwise(make_not_null(&gen));
  test_action();
}
}  // namespace
