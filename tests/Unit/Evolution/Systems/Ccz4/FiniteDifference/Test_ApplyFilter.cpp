// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ApplyFilter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Filter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/Slab.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
void set_profile(
    const gsl::not_null<std::vector<DataVector>*> vars_ptr,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& local_logical_coords,
    const size_t degree, const double wave_number = 0.0) {
  for (auto& var : *vars_ptr) {
    var = 0.0;
    for (size_t i = 0; i < Dim; ++i) {
      if (wave_number == 0.0) {
        var += pow(local_logical_coords.get(i), degree);
      } else {
        var += sin(wave_number * local_logical_coords.get(i));
      }
    }
  }
}

template <typename System>
void set_solution(
    const gsl::not_null<Variables<typename System::variables_tag::tags_list>*>
        volume_vars,
    const gsl::not_null<DirectionalIdMap<3, evolution::dg::subcell::GhostData>*>
        neighbor_data,
    const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& logical_coords,
    const size_t deriv_order, const size_t degree,
    const double wave_number = 0.0) {
  const auto set_data_vectors = [](const gsl::not_null<std::vector<DataVector>*>
                                       local_dvs,
                                   const auto local_vars) {
    for (size_t i = 0; i < 6; ++i) {
      (*local_dvs)[i].set_data_ref(make_not_null(
          &get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 1; ++i) {
      (*local_dvs)[i + 10].set_data_ref(make_not_null(
          &get<::Ccz4::Tags::ConformalFactor<DataVector>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 6; ++i) {
      (*local_dvs)[i + 11].set_data_ref(make_not_null(
          &get<::Ccz4::Tags::ATilde<DataVector, 3>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 1; ++i) {
      (*local_dvs)[i + 17].set_data_ref(make_not_null(
          &get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 1; ++i) {
      (*local_dvs)[i + 18].set_data_ref(
          make_not_null(&get<::Ccz4::Tags::Theta<DataVector>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 3; ++i) {
      (*local_dvs)[i + 19].set_data_ref(make_not_null(
          &get<::Ccz4::Tags::GammaHat<DataVector, 3>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 1; ++i) {
      (*local_dvs)[i + 6].set_data_ref(
          make_not_null(&get<gr::Tags::Lapse<DataVector>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 3; ++i) {
      (*local_dvs)[i + 7].set_data_ref(
          make_not_null(&get<gr::Tags::Shift<DataVector, 3>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 3; ++i) {
      (*local_dvs)[i + 22].set_data_ref(make_not_null(
          &get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(*local_vars)[i]));
    }
  };

  std::vector<DataVector> vars(25);
  set_data_vectors(make_not_null(&vars), volume_vars);
  set_profile(&vars, logical_coords, degree, wave_number);

  for (const auto& direction : Direction<3>::all_directions()) {
    auto neighbor_logical_coords = logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    std::vector<DataVector> neighbor_dvs(25);
    Variables<::Ccz4::fd::System::variables_tag_list> neighbor_vars{
        mesh.number_of_grid_points(), 7.0};

    set_data_vectors(make_not_null(&neighbor_dvs),
                     make_not_null(&neighbor_vars));
    set_profile(&neighbor_dvs, neighbor_logical_coords, degree, wave_number);

    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars), mesh.extents(), deriv_order / 2 + 1,
        std::unordered_set{direction.opposite()}, 0, {});
    CAPTURE(deriv_order / 2 + 1);
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    const auto key = DirectionalId<3>{direction, ElementId<3>{0}};
    (*neighbor_data)[key] = evolution::dg::subcell::GhostData{1};
    (*neighbor_data)[key].neighbor_ghost_data_for_reconstruction() =
        sliced_data.at(direction.opposite());
  }
}

// A sixth difference annihilates degrees <=5 and gives 6! h^6 on
// x^6. These expectations do not call the production filter.
void test(const bool evolve_lapse_and_shift, const bool forward,
          const bool self_start, const double epsilon, const size_t degree) {
  CAPTURE(evolve_lapse_and_shift, forward, self_start, epsilon, degree);
  using System = Ccz4::fd::System;
  using Dissipation = Ccz4::fd::AddKreissOligerDissipation;
  using DtTag = Dissipation::dt_variables_tag;
  const Mesh<3> mesh{8, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered};
  Variables<System::variables_tag_list> vars{mesh.number_of_grid_points(), 7.0};
  DirectionalIdMap<3, evolution::dg::subcell::GhostData> ghosts{};
  set_solution<System>(
      &vars, &ghosts, mesh,
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(mesh), 6, degree);
  const auto original_vars = vars;
  const auto original_ghosts = ghosts;
  const Slab slab{0.0, 1.0};
  const auto full_step = (forward ? 1 : -1) * slab.duration() / 100;
  const TimeStepId id{forward, self_start ? -1 : 0,
                      forward ? slab.start() : slab.end()};
  auto box = db::create<db::AddSimpleTags<
      DtTag, System::variables_tag, evolution::dg::subcell::Tags::Mesh<3>,
      Ccz4::fd::Tags::EvolveLapseAndShift, Ccz4::fd::Tags::KreissOligerEpsilon,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
      ::Tags::TimeStep, ::Tags::TimeStepId,
      SelfStart::Tags::InitialValue<::Tags::TimeStep>>>(
      DtTag::type{mesh.number_of_grid_points(), 0.125}, vars, mesh,
      evolve_lapse_and_shift, epsilon, ghosts,
      self_start ? full_step / 3 : full_step, id,
      std::tuple{self_start ? full_step : TimeDelta{}});
  db::mutate_apply<Dissipation>(make_not_null(&box));
  const auto& result = get<DtTag>(box);
  const double correction = degree == 6 ? 3.0 * 720.0 * pow(0.25, 6) * epsilon /
                                              (64.0 * full_step.value())
                                        : 0.0;
  tmpl::for_each<
      System::original_evolved_variables_tags>([&result, evolve_lapse_and_shift,
                                                correction]<typename Tag>(
                                                   tmpl::type_<Tag> /*meta*/) {
    constexpr bool is_gauge = tmpl::list_contains_v<
        tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                   Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>,
        Tag>;
    const double expected =
        0.125 + ((evolve_lapse_and_shift or not is_gauge) ? correction : 0.0);
    CAPTURE(db::tag_name<Tag>());
    for (const auto& component : get<::Tags::dt<Tag>>(result)) {
      for (const double value : component) {
        CHECK(value == approx(expected));
      }
    }
  });
  for (const auto& component :
       get<::Tags::dt<Ccz4::fd::Tags::PotentialGradients>>(result)) {
    CHECK(component == DataVector(mesh.number_of_grid_points(), 0.125));
  }
  CHECK(get<System::variables_tag>(box) == original_vars);
  const auto& remaining_ghosts =
      get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box);
  for (const auto& [key, data] : original_ghosts) {
    CHECK(remaining_ghosts.at(key).neighbor_ghost_data_for_reconstruction() ==
          data.neighbor_ghost_data_for_reconstruction());
  }
}

// Fourier modes have the exact sixth-order KO eigenvalue
// -epsilon * sin(k h / 2)^6. Check modes through the grid cutoff and two
// actual step sizes, including points whose stencil crosses each face.
void test_fourier_modes() {
  using System = Ccz4::fd::System;
  using Dissipation = Ccz4::fd::AddKreissOligerDissipation;
  const Mesh<3> mesh{8, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered};
  const auto x = TestHelpers::Ccz4::fd::detail::set_logical_coordinates(mesh);
  const Slab slab{0.0, 1.0};
  for (const double wave_number : {M_PI, 2.0 * M_PI, 4.0 * M_PI}) {
    System::variables_tag::type vars{mesh.number_of_grid_points(), 7.0};
    DirectionalIdMap<3, evolution::dg::subcell::GhostData> ghosts{};
    set_solution<System>(&vars, &ghosts, mesh, x, 6, 0, wave_number);
    for (const int denominator : {50, 100}) {
      const auto step = slab.duration() / denominator;
      Dissipation::dt_variables_tag::type dt{mesh.number_of_grid_points(), 0.0};
      Dissipation::apply(make_not_null(&dt), vars, mesh, true, 0.1, ghosts,
                         step, TimeStepId{true, 0, slab.start()}, {});
      const DataVector expected = -0.1 / step.value() *
                                  pow(sin(wave_number * 0.25 / 2.0), 6) *
                                  get(get<gr::Tags::Lapse<DataVector>>(vars));
      CHECK_ITERABLE_APPROX(
          get(get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(dt)), expected);
    }
  }
}

void test_invalid_epsilon() {
  using Dissipation = Ccz4::fd::AddKreissOligerDissipation;
  const Mesh<3> mesh{8, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered};
  const Ccz4::fd::System::variables_tag::type vars{mesh.number_of_grid_points(),
                                                   1.0};
  Dissipation::dt_variables_tag::type dt{mesh.number_of_grid_points(), 0.125};
  const Slab slab{0.0, 1.0};
  for (const double epsilon : {-0.1, 1.1}) {
    CHECK_THROWS_WITH(
        Dissipation::apply(make_not_null(&dt), vars, mesh, true, epsilon, {},
                           slab.duration() / 100,
                           TimeStepId{true, 0, slab.start()}, {}),
        Catch::Matchers::ContainsSubstring(
            "Kreiss-Oliger epsilon should be in the interval [0, 1]"));
  }
  // Disabled dissipation requires neither ghosts nor a valid step.
  Dissipation::apply(make_not_null(&dt), vars, mesh, true, 0.0, {}, TimeDelta{},
                     TimeStepId{}, {});
  CHECK(dt == Dissipation::dt_variables_tag::type(mesh.number_of_grid_points(),
                                                  0.125));
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.ApplyFilter",
                  "[Unit][Evolution]") {
  for (const bool gauge : {false, true}) {
    for (const bool forward : {false, true}) {
      for (const bool self_start : {false, true}) {
        for (const double epsilon : {0.0, 0.1, 1.0}) {
          for (const size_t degree : {size_t{0}, size_t{5}, size_t{6}}) {
            test(gauge, forward, self_start, epsilon, degree);
          }
        }
      }
    }
  }
  test_fourier_modes();
  test_invalid_epsilon();
}
}  // namespace
