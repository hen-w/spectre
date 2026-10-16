// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UnlimitedDeg4Prim.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TrumpetSchwarzschild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
using Condition = gp::BoundaryConditions::DirichletAnalytic;
using KerrSchild = Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::KerrSchild>;
using Trumpet =
    Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::TrumpetSchwarzschild>;

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
  gp::LogMetricChart chart{};
  gp::chart_from_physical(make_not_null(&u), make_not_null(&w),
                          make_not_null(&chart), state);
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

// The finite-difference ghost cells of the inner and outer faces of the six
// equiangular wedges of an excised sphere (one element per wedge, 9 cells per
// dimension) hold the analytic fields at their own points. The expected points
// are built here without the ghost-zone coordinate helper: three layers of
// cells continuing the uniform cell-centered radial grid outside the wedge,
// ordered with the radial logical coordinate increasing like the volume
// cells, at the angular logical coordinates of the cells. The wedges have
// spherical surfaces and a linear radial distribution, so the radius is
// linear in the radial logical coordinate also outside the wedge: the m-th
// ghost layer from the face (m = 0, 1, 2) has the radius
// inner_radius - (m + 1/2) dr or outer_radius + (m + 1/2) dr, with
// dr = (outer_radius - inner_radius) / 9.
template <typename Solution>
void test_fd_ghost(const Condition& condition, const Solution& solution,
                   const double inner_radius, const double outer_radius) {
  const double time = 0.7;
  const size_t dg_points = 5;
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(
      Mesh<3>{dg_points, Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto});
  const size_t n = subcell_mesh.extents(0);
  REQUIRE(n == 9);
  const Ccz4::fd::UnlimitedDeg4Prim reconstructor{};
  const size_t layers = reconstructor.ghost_zone_size();
  REQUIRE(layers == 3);
  const double spacing = 2.0 / static_cast<double>(n);
  const double radial_spacing =
      (outer_radius - inner_radius) / static_cast<double>(n);
  const size_t number_of_ghost_points = layers * n * n;

  const domain::creators::Sphere creator(inner_radius, outer_radius,
                                         domain::creators::Sphere::Excision{},
                                         0_st, dg_points, true);
  const auto domain = creator.create_domain();
  REQUIRE(domain.blocks().size() == 6);
  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});
  const domain::FunctionsOfTimeMap functions_of_time{};

  double smallest_radius = std::numeric_limits<double>::max();
  double largest_radius = 0.0;
  for (const auto& block : domain.blocks()) {
    const ElementMap<3, Frame::Grid> element_map{ElementId<3>{block.id()},
                                                 block};
    for (const auto& direction :
         {Direction<3>::lower_zeta(), Direction<3>::upper_zeta()}) {
      CAPTURE(block.id());
      CAPTURE(direction);
      REQUIRE(block.external_boundaries().contains(direction));
      const bool upper = direction.side() == Side::Upper;
      tnsr::I<DataVector, 3, Frame::ElementLogical> logical_coords{
          number_of_ghost_points};
      DataVector expected_radius{number_of_ghost_points};
      for (size_t k = 0; k < layers; ++k) {
        // Distance from the face in cells.
        const double depth = upper ? static_cast<double>(k) + 0.5
                                   : static_cast<double>(layers - k) - 0.5;
        for (size_t j = 0; j < n; ++j) {
          for (size_t i = 0; i < n; ++i) {
            const size_t s = i + n * (j + n * k);
            get<0>(logical_coords)[s] =
                -1.0 + (static_cast<double>(i) + 0.5) * spacing;
            get<1>(logical_coords)[s] =
                -1.0 + (static_cast<double>(j) + 0.5) * spacing;
            get<2>(logical_coords)[s] =
                upper ? 1.0 + depth * spacing : -1.0 - depth * spacing;
            expected_radius[s] = upper ? outer_radius + depth * radial_spacing
                                       : inner_radius - depth * radial_spacing;
          }
        }
      }
      const auto coords = (*grid_to_inertial_map)(element_map(logical_coords),
                                                  time, functions_of_time);
      const DataVector radius = get(magnitude(coords));
      CHECK_ITERABLE_APPROX(radius, expected_radius);
      smallest_radius = std::min(smallest_radius, min(radius));
      largest_radius = std::max(largest_radius, max(radius));

      Variables<physical_tags> ghost_vars{number_of_ghost_points};
      condition.fd_ghost(
          make_not_null(
              &get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(ghost_vars)),
          make_not_null(
              &get<Ccz4::Tags::ConformalFactor<DataVector>>(ghost_vars)),
          make_not_null(&get<Ccz4::Tags::ATilde<DataVector, 3>>(ghost_vars)),
          make_not_null(
              &get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(ghost_vars)),
          make_not_null(&get<Ccz4::Tags::Theta<DataVector>>(ghost_vars)),
          make_not_null(&get<Ccz4::Tags::GammaHat<DataVector, 3>>(ghost_vars)),
          make_not_null(&get<gr::Tags::Lapse<DataVector>>(ghost_vars)),
          make_not_null(&get<gr::Tags::Shift<DataVector, 3>>(ghost_vars)),
          make_not_null(
              &get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(ghost_vars)),
          direction, subcell_mesh, time, functions_of_time, element_map,
          *grid_to_inertial_map, reconstructor);
      const auto expected = solution.variables(coords, time, physical_tags{});
      tmpl::for_each<physical_tags>(
          [&ghost_vars, &expected]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
            CAPTURE(db::tag_name<Tag>());
            CHECK_ITERABLE_APPROX(get<Tag>(ghost_vars), get<Tag>(expected));
          });
    }
  }
  CHECK(smallest_radius == approx(inner_radius - 2.5 * radial_spacing));
  CHECK(largest_radius == approx(outer_radius + 2.5 * radial_spacing));
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
  test_fd_ghost(serialize_and_deserialize(*typed), solution, 4.0, 8.0);
  test_fd_ghost(dynamic_cast<const Condition&>(*cloned), solution, 4.0, 8.0);
  // The stationary trumpet on the shell 15M < r < 30M; the innermost ghost
  // radius is 15M - 2.5 (15M / 9).
  const Trumpet trumpet{gr::Solutions::TrumpetSchwarzschild{1.0, 2.0}};
  test_fd_ghost(Condition{std::make_unique<Trumpet>(trumpet)}, trumpet, 15.0,
                30.0);
}
