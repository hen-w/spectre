// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/GhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.GhostData",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  const std::uniform_real_distribution<> dist(-1.0, 1.0);

  const size_t points_per_dimension = 5;
  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  const auto random_vars_subcell = make_with_random_values<
      Variables<::Ccz4::fd::System::variables_tag_list>>(
      make_not_null(&gen), dist, subcell_mesh.number_of_grid_points());
  auto box_subcell =
      db::create<db::AddSimpleTags<::Ccz4::fd::System::variables_tag>>(
          random_vars_subcell);

  DataVector retrieved_vars_subcell =
      db::mutate_apply<::Ccz4::fd::GhostVariables>(make_not_null(&box_subcell),
                                                   2_st);

  const Variables<::Ccz4::fd::System::variables_tag_list> retrieved_vars{
      retrieved_vars_subcell.data(), retrieved_vars_subcell.size() - 2};
  tmpl::for_each<::Ccz4::fd::System::original_evolved_variables_tags>(
      [&random_vars_subcell, &retrieved_vars](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<tag>(random_vars_subcell),
                              get<tag>(retrieved_vars));
      });
  // Verify auxiliary and boundary fields are zero in ghost data
  const DataVector zero_dv(retrieved_vars.number_of_grid_points(), 0.0);
  tmpl::for_each<tmpl::append<::Ccz4::fd::System::auxiliary_variables_tags,
                              ::Ccz4::fd::System::boundary_second_order_tags>>(
      [&retrieved_vars, &zero_dv](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        for (const auto& component : get<tag>(retrieved_vars)) {
          CHECK_ITERABLE_APPROX(component, zero_dv);
        }
      });
}
}  // namespace
