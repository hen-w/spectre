// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Hessian.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Autodiff/Autodiff.hpp"

namespace domain {
namespace {
template <size_t Dim>
void test_hessian(const ElementMap<Dim, Frame::Inertial>& element_map,
                  const size_t num_pts,
                  const gsl::not_null<std::mt19937*> generator) {
  DataVector source_coords_per_dim(num_pts);
  auto value_distribution = std::uniform_real_distribution(
      std::numeric_limits<double>::min(), std::numeric_limits<double>::max());
  source_coords_per_dim = make_with_random_values<DataVector>(
      generator, distribution, source_coords_per_dim);

  tnsr::I<DataVector, Dim, Frame::ElementLogical> source_coords;
  for (size_t i = 0; i < Dim; ++i) {
    source_coords.get(i) = source_coords_per_dim;
  }

  const auto inverse_jac = element_map.inv_jacobian(source_coords);
  const auto inv_hessian =
      domain::Hessian::inv_hessian(element_map, inverse_jac, source_coords);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Hessian", "[Unit][Domain]") {}
}  // namespace domain
