// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/Systems/Ccz4/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::Ccz4::fd {
namespace detail {
using GhostData = evolution::dg::subcell::GhostData;
template <typename F>
DirectionalIdMap<3, GhostData> compute_ghost_data(
    const Mesh<3>& subcell_mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& volume_logical_coords,
    const DirectionMap<3, Neighbors<3>>& neighbors,
    const size_t ghost_zone_size, const F& compute_variables_of_neighbor_data) {
  DirectionalIdMap<3, GhostData> ghost_data{};
  for (const auto& [direction, neighbors_in_direction] : neighbors) {
    REQUIRE(neighbors_in_direction.size() == 1);
    const ElementId<3>& neighbor_id = *neighbors_in_direction.begin();
    auto neighbor_logical_coords = volume_logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    const auto neighbor_vars_for_reconstruction =
        compute_variables_of_neighbor_data(neighbor_logical_coords);

    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars_for_reconstruction.data(),
                       neighbor_vars_for_reconstruction.size()),
        subcell_mesh.extents(), ghost_zone_size,
        std::unordered_set{direction.opposite()}, 0, {});
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    ghost_data[DirectionalId<3>{direction, neighbor_id}] = GhostData{1};
    ghost_data.at(DirectionalId<3>{direction, neighbor_id})
        .neighbor_ghost_data_for_reconstruction() =
        sliced_data.at(direction.opposite());
  }
  return ghost_data;
}

inline Variables<
    ::Ccz4::Tags::primitive_grmhd_and_spacetime_reconstruction_tags>
compute_prim_solution(
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& coords) {
  /* initialize all evolved variables for testing derivs*/
  using ConformalMetric = ::Ccz4::Tags::ConformalMetric<DataVector, 3>;
  using ATilde = ::Ccz4::Tags::ATilde<DataVector, 3>;
  using ConformalFactor = ::Ccz4::Tags::ConformalFactor<DataVector>;
  using TraceExtrinsicCurvature = gr::Tags::TraceExtrinsicCurvature<DataVector>;
  using Theta = ::Ccz4::Tags::Theta<DataVector>;
  using GammaHat = ::Ccz4::Tags::GammaHat<DataVector, 3>;
  using Lapse = gr::Tags::Lapse<DataVector>;
  using Shift = gr::Tags::Shift<DataVector, 3>;
  using Fieldb = ::Ccz4::Tags::Fieldb<DataVector, 3>;

  Variables<::Ccz4::Tags::primitive_grmhd_and_spacetime_reconstruction_tags>
      vars{get<0>(coords).size(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    get(get<ConformalFactor>(vars)) += coords.get(i);
    get(get<TraceExtrinsicCurvature>(vars)) += coords.get(i);
    get(get<Theta>(vars)) += coords.get(i);
    get(get<Lapse>(vars)) += coords.get(i);
    for (size_t j = 0; j < 3; ++j) {
      get<GammaHat>(vars).get(j) += coords.get(i);
      get<Shift>(vars).get(j) += coords.get(i);
      get<Fieldb>(vars).get(j) += coords.get(i);
    }
  }
  get(get<ConformalFactor>(vars)) += 2.0;
  get(get<TraceExtrinsicCurvature>(vars)) += 15.0;
  get(get<Theta>(vars)) += 30.0;
  get(get<Lapse>(vars)) += 50.0;
  for (size_t j = 0; j < 3; ++j) {
    get<GammaHat>(vars).get(j) += 1.0e-2 * (j + 2.0) + 10.0;
    get<Shift>(vars).get(j) += 1.0e-2 * (j + 2.0) + 60.0;
    get<Fieldb>(vars).get(j) += 1.0e-2 * (j + 2.0) + 110.0;
  }

  auto& conformal_metric = get<ConformalMetric>(vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      conformal_metric.get(i, j) = (10 * i + 50 * j + 1) * coords.get(i);
    }
  }
  auto& atilde = get<ATilde>(vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      atilde.get(i, j) = (1000 * i + 5000 * j + 1) * coords.get(i);
    }
  }
  return vars;
}

inline Element<3> set_element(const bool skip_last = false) {
  /* I don't know what this is doing; it seems to set up */
  /* some element adjacency relations */
  DirectionMap<3, Neighbors<3>> neighbors{};
  for (size_t i = 0; i < 6; ++i) {
    if (skip_last and i == 5) {
      break;
    }
    neighbors[gsl::at(Direction<3>::all_directions(), i)] = Neighbors<3>{
        {ElementId<3>{i + 1, {}}}, OrientationMap<3>::create_aligned()};
  }
  return Element<3>{ElementId<3>{0, {}}, neighbors};
}

inline tnsr::I<DataVector, 3, Frame::ElementLogical> set_logical_coordinates(
    const Mesh<3>& subcell_mesh) {
  /* this computes the positions of the grid points in [-1,1]^3 */
  auto logical_coords = logical_coordinates(subcell_mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < 3; ++i) {
    /* this seems to shift all grid points in [-1,1]^3 by some number */
    logical_coords.get(i) += 4.0 * i;
  }
  return logical_coords;
}
}  // namespace detail
}  // namespace TestHelpers::Ccz4::fd
