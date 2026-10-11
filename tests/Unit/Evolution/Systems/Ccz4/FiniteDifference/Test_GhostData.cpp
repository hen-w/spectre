// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/GhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using System = ::Ccz4::fd::System;
using AllVariables = Variables<System::variables_tag_list>;
using PhysicalVariables = Variables<System::original_evolved_variables_tags>;
using GaugeVariables = Variables<
    tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
               ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>;

// The wire layout assumed by the finite-difference derivatives, filter and
// reconstruction: 25 physical components first, the gauge fields (7
// components) last among them, then the 30 gradient components.
static_assert(PhysicalVariables::number_of_independent_components == 25);
static_assert(GaugeVariables::number_of_independent_components == 7);
static_assert(AllVariables::number_of_independent_components == 55);
static_assert(std::is_same_v<tmpl::back<System::variables_tag_list>,
                             ::Ccz4::fd::Tags::PotentialGradients>);
static_assert(
    std::is_same_v<
        tmpl::list<tmpl::at_c<System::original_evolved_variables_tags, 6>,
                   tmpl::at_c<System::original_evolved_variables_tags, 7>,
                   tmpl::at_c<System::original_evolved_variables_tags, 8>>,
        GaugeVariables::tags_list>);

template <typename GhostMutator, bool GradientsSent>
void test_ghost_variables() {
  CAPTURE(GradientsSent);
  MAKE_GENERATOR(gen);
  const std::uniform_real_distribution<> dist(-1.0, 1.0);

  const Mesh<3> subcell_mesh{5, Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  const size_t number_of_points = subcell_mesh.number_of_grid_points();
  const auto evolved_vars = make_with_random_values<AllVariables>(
      make_not_null(&gen), dist, number_of_points);
  auto box = db::create<db::AddSimpleTags<System::variables_tag>>(evolved_vars);

  for (const size_t rdmp_size : {0_st, 2_st}) {
    CAPTURE(rdmp_size);
    const DataVector buffer =
        db::mutate_apply<GhostMutator>(make_not_null(&box), rdmp_size);
    REQUIRE(buffer.size() ==
            AllVariables::number_of_independent_components * number_of_points +
                rdmp_size);
    // The mutator does not modify the evolved variables.
    CHECK(db::get<System::variables_tag>(box) == evolved_vars);

    // The physical fields are copied exactly, as the contiguous prefix.
    const size_t physical_size =
        PhysicalVariables::number_of_independent_components * number_of_points;
    for (size_t i = 0; i < physical_size; ++i) {
      CHECK(buffer[i] == evolved_vars.data()[i]);
    }
    // The gradients are copied exactly or zero.
    const size_t gradients_size =
        AllVariables::number_of_independent_components * number_of_points -
        physical_size;
    for (size_t i = physical_size; i < physical_size + gradients_size; ++i) {
      CHECK(buffer[i] == (GradientsSent ? evolved_vars.data()[i] : 0.0));
    }
    // The space reserved for the RDMP data is zero (filled by the caller).
    for (size_t i = physical_size + gradients_size; i < buffer.size(); ++i) {
      CHECK(buffer[i] == 0.0);
    }

    // Tag-wise view of the same buffer.
    const AllVariables sent{const_cast<double*>(buffer.data()),
                            buffer.size() - rdmp_size};
    tmpl::for_each<System::original_evolved_variables_tags>(
        [&evolved_vars, &sent]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          CHECK(get<Tag>(sent) == get<Tag>(evolved_vars));
        });
    if constexpr (GradientsSent) {
      CHECK(get<::Ccz4::fd::Tags::PotentialGradients>(sent) ==
            get<::Ccz4::fd::Tags::PotentialGradients>(evolved_vars));
    } else {
      for (const auto& component :
           get<::Ccz4::fd::Tags::PotentialGradients>(sent)) {
        CHECK(component == DataVector(number_of_points, 0.0));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.GhostData",
                  "[Unit][Evolution]") {
  test_ghost_variables<::Ccz4::fd::GhostVariables, false>();
  test_ghost_variables<::Ccz4::fd::GhostVariablesPhysical, true>();
}
}  // namespace
