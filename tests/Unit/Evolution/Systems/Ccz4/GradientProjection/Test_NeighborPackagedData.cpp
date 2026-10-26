// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataBox/AsAccess.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/MeshForGhostData.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UnlimitedDeg4Prim.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryCorrections/UpwindGradientProjection.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/NeighborPackagedData.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;
using System = gp::System;
using AllVariables = Variables<System::variables_tag_list>;
using Correction = gp::BoundaryCorrections::UpwindGradientProjection;
using ConformalMetric = Ccz4::Tags::ConformalMetric<DataVector, 3>;
using ATilde = Ccz4::Tags::ATilde<DataVector, 3>;

// The package of the face state as the DG elements compute it (no
// projection).
gp::PackagedFaceData dg_package(const AllVariables& vars,
                                const tnsr::i<DataVector, 3>& normal,
                                const Direction<3>& direction) {
  gp::PackagedFaceData packaged{vars.number_of_grid_points()};
  evolution::dg::Actions::detail::dg_package_data<System>(
      make_not_null(&packaged),
      Correction{gp::BoundaryCorrections::ExternalCorrection::Upwind, 1.0, 1.0,
                 1.0},
      vars, normal, {std::nullopt}, direction, System::variables_tag_list{});
  return packaged;
}

// On-surface physical face values from random chart states (anisotropic H,
// non-unit conformal factor and lapse), with random gradients.
AllVariables random_face_state(const gsl::not_null<std::mt19937*> gen,
                               const size_t number_of_points) {
  helpers::ChartData chart(number_of_points);
  AllVariables vars{number_of_points};
  auto& q = get<gp::Tags::PotentialGradients>(vars);
  for (size_t s = 0; s < number_of_points; ++s) {
    const auto state = helpers::random_state(gen, 0.3, 0.1, 0.5);
    chart.set(s, state.u, state.w);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        q.get(i, a)[s] = gsl::at(gsl::at(state.q, i), a);
      }
    }
  }
  get<ConformalMetric>(vars) = chart.conformal_metric;
  get<Ccz4::Tags::ConformalFactor<DataVector>>(vars) = chart.conformal_factor;
  get<ATilde>(vars) = chart.a_tilde;
  get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars) =
      chart.trace_extrinsic_curvature;
  get<Ccz4::Tags::Theta<DataVector>>(vars) = chart.theta;
  get<Ccz4::Tags::GammaHat<DataVector, 3>>(vars) = chart.gamma_hat;
  get<gr::Tags::Lapse<DataVector>>(vars) = chart.lapse;
  get<gr::Tags::Shift<DataVector, 3>>(vars) = chart.shift;
  get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(vars) = chart.b;
  return vars;
}

tnsr::i<DataVector, 3> random_unit_normal(
    const gsl::not_null<std::mt19937*> gen, const size_t number_of_points) {
  tnsr::i<DataVector, 3> normal{number_of_points};
  for (size_t s = 0; s < number_of_points; ++s) {
    const auto n = helpers::random_unit_vector(gen);
    for (size_t i = 0; i < 3; ++i) {
      normal.get(i)[s] = gsl::at(n, i);
    }
  }
  return normal;
}

double max_abs_difference(const gp::PackagedFaceData& a,
                          const gp::PackagedFaceData& b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result = std::max(result, std::abs(a.data()[i] - b.data()[i]));
  }
  return result;
}

template <typename TensorType>
double max_abs_tensor_difference(const TensorType& a, const TensorType& b) {
  double result = 0.0;
  for (size_t c = 0; c < a.size(); ++c) {
    result = std::max(result, max(abs(a[c] - b[c])));
  }
  return result;
}

// D1: the reconstructed face metric and Atilde are projected with the volume
// enforce_algebraic_constraints formula before packaging. For gtilde -> s
// gtilde and Atilde -> Atilde + c gtilde (on-surface gtilde, gtilde-trace-free
// Atilde) that projection restores the original state exactly, whereas
// packaging without it fails the surface check and the alternative "silent"
// chart encoding (which also rescales Atilde by det^(-1/3)) differs.
void test_face_projection(const gsl::not_null<std::mt19937*> gen) {
  const size_t number_of_points = 7;
  const Direction<3> direction = Direction<3>::lower_eta();
  const AllVariables on_surface = random_face_state(gen, number_of_points);
  const auto normal = random_unit_normal(gen, number_of_points);
  const Correction correction{
      gp::BoundaryCorrections::ExternalCorrection::Upwind, 1.0, 1.0, 1.0};
  const auto reference = dg_package(on_surface, normal, direction);

  {
    INFO("On the surface the projection changes the package at roundoff");
    const auto packaged = gp::package_finite_difference_face(
        on_surface, normal, correction, direction);
    CHECK(max_abs_difference(packaged, reference) < 1e-13);
  }

  const double scale = 1.03;
  const double trace_shift = 0.2;
  AllVariables off_surface = on_surface;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      get<ConformalMetric>(off_surface).get(i, j) *= scale;
      get<ATilde>(off_surface).get(i, j) +=
          trace_shift * get<ConformalMetric>(on_surface).get(i, j);
    }
  }
  {
    INFO("D1 restores the on-surface package");
    const auto packaged = gp::package_finite_difference_face(
        off_surface, normal, correction, direction);
    CHECK(max_abs_difference(packaged, reference) < 1e-13);
    // The gradients are packaged unprojected, as reconstructed.
    CHECK(get<gp::Tags::PackagedGradients>(packaged) ==
          get<gp::Tags::PotentialGradients>(off_surface));
  }
  {
    INFO("Without the projection the off-surface face state is rejected");
    CHECK_THROWS_WITH(dg_package(off_surface, normal, direction),
                      Catch::Matchers::ContainsSubstring(
                          "off the algebraic-constraint surface"));
  }
  {
    INFO("The silent chart encoding is a different projection");
    // Normalize the metric and drop the trace, then also rescale Atilde by
    // det^(-1/3) = 1/scale.
    AllVariables silent = on_surface;
    for (auto& component : get<ATilde>(silent)) {
      component /= scale;
    }
    const auto silent_package = dg_package(silent, normal, direction);
    const auto packaged = gp::package_finite_difference_face(
        off_surface, normal, correction, direction);
    CHECK(max_abs_tensor_difference(
              get<gp::Tags::PackagedVelocities>(packaged),
              get<gp::Tags::PackagedVelocities>(silent_package)) > 1e-3);
  }
  {
    INFO("Inadmissible face states are errors");
    AllVariables bad = on_surface;
    get(get<Ccz4::Tags::ConformalFactor<DataVector>>(bad))[3] = -0.1;
    CHECK_THROWS_WITH(
        gp::package_finite_difference_face(bad, normal, correction, direction),
        Catch::Matchers::ContainsSubstring(
            "The conformal factor must be positive"));
    bad = on_surface;
    get(get<gr::Tags::Lapse<DataVector>>(bad))[2] = 0.0;
    CHECK_THROWS_WITH(
        gp::package_finite_difference_face(bad, normal, correction, direction),
        Catch::Matchers::ContainsSubstring("The lapse must be positive"));
    bad = on_surface;
    // det = -(g_11 g_22 - g_12^2) < 0.
    get<0, 0>(get<ConformalMetric>(bad))[1] = -1.0;
    get<0, 1>(get<ConformalMetric>(bad))[1] = 0.0;
    get<0, 2>(get<ConformalMetric>(bad))[1] = 0.0;
    CHECK_THROWS_WITH(
        gp::package_finite_difference_face(bad, normal, correction, direction),
        Catch::Matchers::ContainsSubstring(
            "The determinant of the conformal spatial metric is non-positive"));
  }
}

double max_abs_difference(const gp::PackagedFaceData& a, const DataVector& b) {
  REQUIRE(a.size() == b.size());
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result = std::max(result, std::abs(a.data()[i] - b[i]));
  }
  return result;
}

// A state that is linear in the logical coordinates with H = 0 (so every
// packaged field is linear), or constant with an anisotropic H.
struct Solution {
  bool linear;
  AllVariables operator()(
      const tnsr::I<DataVector, 3, Frame::ElementLogical>& coords) const {
    const size_t number_of_points = get<0>(coords).size();
    const double slope = linear ? 1.0 : 0.0;
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    const auto& z = get<2>(coords);
    AllVariables vars{number_of_points, 0.0};
    auto& metric = get<ConformalMetric>(vars);
    auto& a_tilde = get<ATilde>(vars);
    if (linear) {
      for (size_t i = 0; i < 3; ++i) {
        metric.get(i, i) = 1.0;
      }
      // Trace-free with respect to the flat metric.
      get<0, 0>(a_tilde) = 0.1 * x - 0.05 * z;
      get<1, 1>(a_tilde) = -0.1 * x + 0.02 * y;
      get<2, 2>(a_tilde) = 0.05 * z - 0.02 * y;
      get<0, 1>(a_tilde) = 0.03 * y + 0.01;
      get<1, 2>(a_tilde) = -0.04 * x;
    } else {
      const gp::LogMetricChart chart{gp::Matrix3{
          {0.3, 0.1, -0.05}, {0.1, -0.2, 0.15}, {-0.05, 0.15, -0.1}}};
      const gp::Matrix3 g = chart.metric();
      const gp::Matrix3 a{
          {0.2, -0.1, 0.05}, {-0.1, 0.1, 0.3}, {0.05, 0.3, -0.3}};
      const gp::Matrix3 at = 0.5 * chart.tangent(a);
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          metric.get(i, j) = g(i, j);
          a_tilde.get(i, j) = at(i, j);
        }
      }
    }
    get(get<Ccz4::Tags::ConformalFactor<DataVector>>(vars)) =
        0.9 + slope * (0.02 * x - 0.01 * y + 0.015 * z);
    get(get<gr::Tags::Lapse<DataVector>>(vars)) =
        0.8 + slope * (-0.01 * x + 0.02 * y - 0.005 * z);
    get(get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars)) =
        0.1 + slope * 0.03 * y;
    get(get<Ccz4::Tags::Theta<DataVector>>(vars)) = -0.05 + slope * 0.02 * z;
    for (size_t k = 0; k < 3; ++k) {
      const double kd = static_cast<double>(k + 1);
      get<gr::Tags::Shift<DataVector, 3>>(vars).get(k) =
          0.02 * kd + slope * 0.01 * kd * coords.get(k);
      get<Ccz4::Tags::GammaHat<DataVector, 3>>(vars).get(k) =
          -0.01 * kd + slope * 0.02 * coords.get((k + 1) % 3);
      get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(vars).get(k) =
          0.03 * kd + slope * 0.01 * coords.get((k + 2) % 3);
    }
    auto& q = get<gp::Tags::PotentialGradients>(vars);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        q.get(i, a) = 0.01 * static_cast<double>(a + 3 * i) - 0.1 +
                      slope * 0.005 * static_cast<double>(a + 1) *
                          coords.get((a + i) % 3);
      }
    }
    return vars;
  }
};

using NormalVariables =
    Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                         evolution::dg::Tags::NormalCovector<3>>>;
using Normals = DirectionMap<3, std::optional<NormalVariables>>;
using GhostDataMap = DirectionalIdMap<3, evolution::dg::subcell::GhostData>;

// The unit normal covectors of an element with the identity map.
Normals unit_normals(const Mesh<3>& dg_mesh) {
  Normals normals{};
  for (const auto& direction : Direction<3>::all_directions()) {
    NormalVariables face{
        dg_mesh.slice_away(direction.dimension()).number_of_grid_points(), 0.0};
    get(get<evolution::dg::Tags::MagnitudeOfNormal>(face)) = 1.0;
    get<evolution::dg::Tags::NormalCovector<3>>(face).get(
        direction.dimension()) = static_cast<double>(direction.sign());
    normals[direction] = std::move(face);
  }
  return normals;
}

// The DataBox items read by the adapter.
struct AdapterInputs {
  Element<3> element;
  Mesh<3> dg_mesh;
  AllVariables dg_vars;
  GhostDataMap ghost_data;
  DirectionalIdMap<3, Mesh<3>> mesh_for_ghost_data;
  DirectionalIdMap<3, Mesh<2>> mortar_meshes;
  DirectionalIdMap<3, Mesh<3>> neighbor_meshes;
  bool moving = false;
};

auto make_adapter_box(const AdapterInputs& inputs) {
  return db::create<db::AddSimpleTags<
      domain::Tags::Element<3>, domain::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Mesh<3>, System::variables_tag,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
      evolution::dg::subcell::Tags::MeshForGhostData<3>,
      evolution::dg::Tags::MortarMesh<3>, domain::Tags::NeighborMesh<3>,
      Ccz4::fd::Tags::Reconstructor, evolution::Tags::BoundaryCorrection,
      domain::Tags::MeshVelocity<3>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<3>,
      evolution::dg::subcell::Tags::SubcellOptions<3>>>(
      inputs.element, inputs.dg_mesh,
      evolution::dg::subcell::fd::mesh(inputs.dg_mesh), inputs.dg_vars,
      inputs.ghost_data, inputs.mesh_for_ghost_data, inputs.mortar_meshes,
      inputs.neighbor_meshes,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<Ccz4::fd::UnlimitedDeg4Prim>()},
      std::unique_ptr<evolution::BoundaryCorrection>{
          std::make_unique<Correction>(
              gp::BoundaryCorrections::ExternalCorrection::Upwind, 1.0, 1.0,
              1.0)},
      inputs.moving ? std::optional{tnsr::I<DataVector, 3>{
                          inputs.dg_mesh.number_of_grid_points(), 0.0}}
                    : std::optional<tnsr::I<DataVector, 3>>{},
      unit_normals(inputs.dg_mesh),
      evolution::dg::subcell::SubcellOptions{
          4.0, 1_st, 1.0e-3, 1.0e-4, false, false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
          std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1});
}

// Six neighbors (blocks 1-6), aligned except for the given orientations.
Element<3> element_with_neighbors(
    const DirectionMap<3, OrientationMap<3>>& orientations = {}) {
  DirectionMap<3, Neighbors<3>> neighbors{};
  for (size_t i = 0; i < 6; ++i) {
    const auto& direction = gsl::at(Direction<3>::all_directions(), i);
    neighbors[direction] = Neighbors<3>{
        {ElementId<3>{i + 1, {}}},
        orientations.contains(direction) ? orientations.at(direction)
                                         : OrientationMap<3>::create_aligned()};
  }
  return Element<3>{ElementId<3>{0, {}}, neighbors};
}

// Conforming equal-order mortar metadata for all neighbors.
void set_conforming_meshes(const gsl::not_null<AdapterInputs*> inputs) {
  const Mesh<3> subcell_mesh =
      evolution::dg::subcell::fd::mesh(inputs->dg_mesh);
  for (const auto& [direction, neighbors_in_direction] :
       inputs->element.neighbors()) {
    const DirectionalId<3> mortar_id{direction,
                                     *neighbors_in_direction.begin()};
    inputs->mesh_for_ghost_data[mortar_id] = subcell_mesh;
    inputs->mortar_meshes[mortar_id] =
        inputs->dg_mesh.slice_away(direction.dimension());
    inputs->neighbor_meshes[mortar_id] = inputs->dg_mesh;
  }
}

std::vector<DirectionalId<3>> mortars_of(const Element<3>& element) {
  std::vector<DirectionalId<3>> mortars{};
  for (const auto& [direction, neighbors_in_direction] : element.neighbors()) {
    mortars.emplace_back(direction, *neighbors_in_direction.begin());
  }
  return mortars;
}

// The adapter on an element with six finite-difference neighbors: linear
// data with H = 0 is reproduced exactly on the DG faces; a constant
// anisotropic state gives the DG element's own package with the reversed
// normal and therefore no boundary correction. Unsupported inputs are errors.
void test_adapter(const bool linear) {
  CAPTURE(linear);
  AdapterInputs inputs{};
  inputs.element = element_with_neighbors();
  inputs.dg_mesh =
      Mesh<3>{5, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const Mesh<3>& dg_mesh = inputs.dg_mesh;
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const Solution solution{linear};
  const auto dg_coords = logical_coordinates(dg_mesh);
  const auto subcell_coords = logical_coordinates(subcell_mesh);
  const Ccz4::fd::UnlimitedDeg4Prim reconstructor{};
  inputs.dg_vars = solution(dg_coords);
  inputs.ghost_data = TestHelpers::Ccz4::fd::detail::compute_ghost_data(
      subcell_mesh, subcell_coords, inputs.element.neighbors(),
      reconstructor.ghost_zone_size(), solution);
  set_conforming_meshes(make_not_null(&inputs));
  const auto mortars = mortars_of(inputs.element);
  const Normals normals = unit_normals(dg_mesh);

  auto box = make_adapter_box(inputs);
  const auto packages =
      gp::NeighborPackagedData::apply(db::as_access(box), mortars);
  REQUIRE(packages.size() == 6);
  for (const auto& mortar_id : mortars) {
    CAPTURE(mortar_id);
    const Direction<3>& direction = mortar_id.direction();
    const size_t dim = direction.dimension();
    const Mesh<2> face_mesh = dg_mesh.slice_away(dim);
    const size_t index =
        direction.side() == Side::Lower ? 0 : dg_mesh.extents(dim) - 1;
    const auto face_coords =
        data_on_slice(dg_coords, dg_mesh.extents(), dim, index);
    const auto& dg_normal =
        get<evolution::dg::Tags::NormalCovector<3>>(*normals.at(direction));
    tnsr::i<DataVector, 3> reversed_normal = dg_normal;
    for (auto& component : reversed_normal) {
      component *= -1.0;
    }
    const auto expected =
        dg_package(solution(face_coords), reversed_normal, direction);
    const DataVector& received = packages.at(mortar_id);
    CHECK(max_abs_difference(expected, received) < 1e-12);

    if (not linear) {
      // The boundary correction between the DG element's own package and the
      // neighbor's vanishes for a constant state.
      const auto interior =
          dg_package(solution(face_coords), dg_normal, direction);
      gp::PackagedFaceData exterior{face_mesh.number_of_grid_points()};
      std::copy(received.begin(), received.end(), exterior.data());
      const size_t n = face_mesh.number_of_grid_points();
      gp::Tags::VelocityTarget::type w_correction{n};
      gp::Tags::GradientTarget::type q_correction{n};
      gp::Tags::PotentialRateTarget::type u_correction{n};
      Correction{gp::BoundaryCorrections::ExternalCorrection::Upwind, 1.0, 1.0,
                 1.0}
          .dg_boundary_terms<false>(
              make_not_null(&w_correction), make_not_null(&q_correction),
              make_not_null(&u_correction),
              get<gp::Tags::PackagedVelocities>(interior),
              get<gp::Tags::PackagedGradients>(interior),
              get<gp::Tags::PackagedPotentials>(interior),
              get<Ccz4::Tags::NormalCovector<DataVector, 3>>(interior),
              get<gp::Tags::PackagedVelocities>(exterior),
              get<gp::Tags::PackagedGradients>(exterior),
              get<gp::Tags::PackagedPotentials>(exterior),
              get<Ccz4::Tags::NormalCovector<DataVector, 3>>(exterior),
              dg::Formulation::StrongInertial);
      for (const auto& component : w_correction) {
        CHECK(max(abs(component)) < 1e-12);
      }
      for (const auto& component : q_correction) {
        CHECK(max(abs(component)) < 1e-12);
      }
      for (const auto& component : u_correction) {
        CHECK(max(abs(component)) < 1e-12);
      }
    }
  }

  const auto check_error = [&mortars](const AdapterInputs& bad_inputs,
                                      const std::string& message) {
    auto bad_box = make_adapter_box(bad_inputs);
    CHECK_THROWS_WITH(
        gp::NeighborPackagedData::apply(db::as_access(bad_box), mortars),
        Catch::Matchers::ContainsSubstring(message));
  };
  // Ghost data that were not sent by a finite-difference element (e.g. by a
  // DG element, which zeroes the gradients) are rejected.
  {
    AdapterInputs bad = inputs;
    bad.mesh_for_ghost_data[mortars.front()] = dg_mesh;
    check_error(bad, "ghost data are not finite-difference data");
    bad.mesh_for_ghost_data.erase(mortars.front());
    check_error(bad, "ghost data are not finite-difference data");
  }
  // Neighbors that are not conforming of equal order: a higher-order
  // neighbor (larger mortar mesh and ghost data on a larger subcell mesh) and
  // a lower-order neighbor (mortar mesh equal to the local face mesh but
  // ghost data on a smaller subcell mesh). Both would otherwise be lifted with
  // mismatched sizes.
  {
    const std::string message =
        "supports only conforming neighbors of equal order";
    const Mesh<3> higher_order{6, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto};
    const Mesh<3> lower_order{4, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
    const auto& mortar_id = mortars.back();
    const size_t dim = mortar_id.direction().dimension();
    AdapterInputs bad = inputs;
    bad.mortar_meshes[mortar_id] = higher_order.slice_away(dim);
    bad.mesh_for_ghost_data[mortar_id] =
        evolution::dg::subcell::fd::mesh(higher_order);
    bad.neighbor_meshes[mortar_id] = higher_order;
    check_error(bad, message);
    bad = inputs;
    bad.mortar_meshes[mortar_id] = higher_order.slice_away(dim);
    check_error(bad, message);
    bad = inputs;
    bad.mesh_for_ghost_data[mortar_id] =
        evolution::dg::subcell::fd::mesh(lower_order);
    bad.neighbor_meshes[mortar_id] = lower_order;
    check_error(bad, message);
  }
  {
    AdapterInputs moving = inputs;
    moving.moving = true;
    check_error(moving, "Moving meshes are not supported");
  }
  CHECK(gp::NeighborPackagedData::apply(db::as_access(box), {}).empty());
}

// Smooth admissible fields (anisotropic H, non-unit conformal factor and
// lapse, gradients) given by low-order polynomials of the host-frame logical
// coordinates. The `outside` fields, sent by the finite-difference
// neighbors, have different values, slopes and curvature from the `inside`
// fields of the DG element, so the two sides have different traces on every
// face.
struct Fields {
  bool outside;
  AllVariables operator()(
      const tnsr::I<DataVector, 3, Frame::ElementLogical>& x) const {
    const size_t number_of_points = get<0>(x).size();
    // A distinct polynomial for each chart component c.
    const auto value = [this](const size_t c, const std::array<double, 3>& p) {
      const double k = static_cast<double>(c);
      double result = 0.3 * std::sin(1.3 * k + 0.2);
      for (size_t i = 0; i < 3; ++i) {
        result += 0.2 * std::cos(0.7 * k + 1.1 * static_cast<double>(i)) *
                  gsl::at(p, i);
      }
      if (outside) {
        result += 0.15 * std::cos(2.1 * k + 0.4);
        for (size_t i = 0; i < 3; ++i) {
          result += 0.1 * std::sin(0.9 * k + 2.0 * static_cast<double>(i)) *
                        gsl::at(p, i) +
                    0.04 * std::cos(1.7 * k + static_cast<double>(i)) *
                        square(gsl::at(p, i)) +
                    0.01 * std::sin(1.1 * k + static_cast<double>(i)) *
                        cube(gsl::at(p, i));
        }
      }
      return result;
    };
    helpers::ChartData chart(number_of_points);
    AllVariables vars{number_of_points};
    auto& q = get<gp::Tags::PotentialGradients>(vars);
    for (size_t s = 0; s < number_of_points; ++s) {
      const std::array<double, 3> p{{get<0>(x)[s], get<1>(x)[s], get<2>(x)[s]}};
      gp::PotentialVector u{};
      gp::VelocityVector w{};
      for (size_t c = 0; c < 5; ++c) {
        gsl::at(u, gp::PotentialIndex::log_metric + c) = 0.2 * value(c, p);
      }
      u[gp::PotentialIndex::conformal_factor] = 1.0 + 0.2 * value(5, p);
      u[gp::PotentialIndex::lapse] = 0.9 + 0.2 * value(6, p);
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(u, gp::PotentialIndex::shift + i) = 0.1 * value(7 + i, p);
      }
      for (size_t c = 0; c < gp::number_of_velocities; ++c) {
        gsl::at(w, c) = 0.5 * value(10 + c, p);
      }
      chart.set(s, u, w);
      for (size_t i = 0; i < 3; ++i) {
        for (size_t c = 0; c < gp::number_of_potentials; ++c) {
          q.get(i, c)[s] = value(23 + 10 * i + c, p);
        }
      }
    }
    get<ConformalMetric>(vars) = chart.conformal_metric;
    get<Ccz4::Tags::ConformalFactor<DataVector>>(vars) = chart.conformal_factor;
    get<ATilde>(vars) = chart.a_tilde;
    get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars) =
        chart.trace_extrinsic_curvature;
    get<Ccz4::Tags::Theta<DataVector>>(vars) = chart.theta;
    get<Ccz4::Tags::GammaHat<DataVector, 3>>(vars) = chart.gamma_hat;
    get<gr::Tags::Lapse<DataVector>>(vars) = chart.lapse;
    get<gr::Tags::Shift<DataVector, 3>>(vars) = chart.shift;
    get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(vars) = chart.b;
    return vars;
  }
};

// The ghost data a finite-difference neighbor in `direction`, with the
// orientation `orientation` (from the host to the neighbor), sends to the
// host: its `outside` fields on its own subcell grid and in its own logical
// frame (tensor components are inertial and not rotated), sliced toward the
// host and oriented to the host's data order. For coinciding grid points this
// is what the production interpolation to the host's ghost points gives.
DataVector neighbor_ghost_data(const Mesh<3>& subcell_mesh,
                               const Direction<3>& direction,
                               const OrientationMap<3>& orientation,
                               const size_t ghost_zone_size) {
  const auto neighbor_logical = logical_coordinates(subcell_mesh);
  // The host-frame logical coordinates of the neighbor's points: the host
  // axis i is the neighbor axis orientation(i), with its sign.
  tnsr::I<DataVector, 3, Frame::ElementLogical> host_coords{
      subcell_mesh.number_of_grid_points()};
  for (size_t i = 0; i < 3; ++i) {
    const Direction<3> neighbor_axis =
        orientation(Direction<3>{i, Side::Upper});
    host_coords.get(i) = static_cast<double>(neighbor_axis.sign()) *
                         neighbor_logical.get(neighbor_axis.dimension());
  }
  host_coords.get(direction.dimension()) +=
      2.0 * static_cast<double>(direction.sign());
  const AllVariables neighbor_vars = Fields{true}(host_coords);
  const Direction<3> direction_from_neighbor =
      orientation(direction.opposite());
  const auto sliced = evolution::dg::subcell::detail::slice_data_impl(
      gsl::make_span(neighbor_vars.data(), neighbor_vars.size()),
      subcell_mesh.extents(), ghost_zone_size,
      std::unordered_set{direction_from_neighbor}, 0, {});
  Index<3> slice_extents = subcell_mesh.extents();
  slice_extents[direction_from_neighbor.dimension()] = ghost_zone_size;
  return orient_variables(sliced.at(direction_from_neighbor), slice_extents,
                          orientation.inverse_map());
}

// The value of the neighbor's (finite-difference) side of the face in
// `direction` at the subcell face points, from the degree-4 interpolant
// through the host's two cells next to the face (the subcell projection of
// the DG volume) and the neighbor's three cells (the `outside` fields at the
// host-frame ghost points), centered on the neighbor's cell next to the face.
// With `wrong_side` the interpolant is instead centered on the host's cell
// next to the face (three host cells and two neighbor cells), i.e. the host's
// side of the face.
AllVariables predicted_face_values(const AllVariables& dg_vars,
                                   const Mesh<3>& dg_mesh,
                                   const Direction<3>& direction,
                                   const bool wrong_side) {
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t dim = direction.dimension();
  const size_t n = subcell_mesh.extents(dim);
  const double delta = 2.0 / static_cast<double>(n);
  const double sign = static_cast<double>(direction.sign());
  const AllVariables projected = evolution::dg::subcell::fd::project(
      dg_vars, dg_mesh, subcell_mesh.extents());
  const auto face_logical = data_on_slice(logical_coordinates(subcell_mesh),
                                          subcell_mesh.extents(), dim, 0);
  // Cell k = 0, 1, ... away from the face on the host side, and k = -1, -2,
  // ... on the neighbor side; the host cell k has the logical coordinate
  // sign * (1 - (k + 1/2) delta) along the normal.
  const auto layer = [&](const int k) {
    if (k >= 0) {
      const size_t index = direction.side() == Side::Upper
                               ? n - 1 - static_cast<size_t>(k)
                               : static_cast<size_t>(k);
      return data_on_slice(projected, subcell_mesh.extents(), dim, index);
    }
    auto coords = face_logical;
    coords.get(dim) = sign * (1.0 - (static_cast<double>(k) + 0.5) * delta);
    return Fields{true}(coords);
  };
  const std::array<int, 5> cells = wrong_side
                                       ? std::array<int, 5>{{2, 1, 0, -1, -2}}
                                       : std::array<int, 5>{{1, 0, -1, -2, -3}};
  std::array<double, 5> nodes{};
  for (size_t m = 0; m < 5; ++m) {
    gsl::at(nodes, m) =
        1.0 - (static_cast<double>(gsl::at(cells, m)) + 0.5) * delta;
  }
  const auto weights = helpers::lagrange_weights(nodes, 1.0);
  AllVariables result{get<0>(face_logical).size(), 0.0};
  for (size_t m = 0; m < 5; ++m) {
    result += gsl::at(weights, m) * layer(gsl::at(cells, m));
  }
  return result;
}

double max_abs_difference(const DataVector& a, const DataVector& b) {
  REQUIRE(a.size() == b.size());
  return max(abs(a - b));
}

// Genuinely different DG volume and finite-difference ghost traces and slopes
// on all six faces, one neighbor across a rotated interface: the adapter's
// package equals the package (with the D1 projection) of the independently
// predicted neighbor-side face values, reconstructed onto the DG face, and
// differs from the prediction for the host's side of the faces.
void test_adapter_distinct_traces() {
  // A rotation by 90 degrees about the host's eta axis for the neighbor in
  // the lower-eta direction.
  const OrientationMap<3> rotated{std::array{Direction<3>::lower_zeta(),
                                             Direction<3>::upper_eta(),
                                             Direction<3>::upper_xi()}};
  DirectionMap<3, OrientationMap<3>> orientations{};
  orientations[Direction<3>::lower_eta()] = rotated;
  AdapterInputs inputs{};
  inputs.element = element_with_neighbors(orientations);
  inputs.dg_mesh =
      Mesh<3>{5, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const Mesh<3>& dg_mesh = inputs.dg_mesh;
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t ghost_zone_size =
      Ccz4::fd::UnlimitedDeg4Prim{}.ghost_zone_size();
  inputs.dg_vars = Fields{false}(logical_coordinates(dg_mesh));
  for (const auto& [direction, neighbors_in_direction] :
       inputs.element.neighbors()) {
    const auto& neighbor = *neighbors_in_direction.begin();
    const DirectionalId<3> mortar_id{direction, neighbor};
    inputs.ghost_data[mortar_id] = evolution::dg::subcell::GhostData{1};
    inputs.ghost_data[mortar_id].neighbor_ghost_data_for_reconstruction() =
        neighbor_ghost_data(subcell_mesh, direction,
                            neighbors_in_direction.orientation(neighbor),
                            ghost_zone_size);
  }
  set_conforming_meshes(make_not_null(&inputs));
  const auto mortars = mortars_of(inputs.element);
  auto box = make_adapter_box(inputs);
  const auto packages =
      gp::NeighborPackagedData::apply(db::as_access(box), mortars);
  REQUIRE(packages.size() == 6);
  const Correction correction{
      gp::BoundaryCorrections::ExternalCorrection::Upwind, 1.0, 1.0, 1.0};
  for (const auto& mortar_id : mortars) {
    CAPTURE(mortar_id);
    const Direction<3>& direction = mortar_id.direction();
    const size_t dim = direction.dimension();
    const auto expected_package = [&](const bool wrong_side) {
      AllVariables face_values =
          predicted_face_values(inputs.dg_vars, dg_mesh, direction, wrong_side);
      tnsr::i<DataVector, 3> normal{face_values.number_of_grid_points(), 0.0};
      normal.get(dim) = -static_cast<double>(direction.sign());
      const auto fd_package = gp::package_finite_difference_face(
          std::move(face_values), normal, correction, direction);
      const auto dg_face_package = evolution::dg::subcell::fd::reconstruct(
          fd_package, dg_mesh.slice_away(dim),
          subcell_mesh.extents().slice_away(dim),
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);
      DataVector result(dg_face_package.size());
      std::copy(dg_face_package.data(),
                dg_face_package.data() + dg_face_package.size(),
                result.begin());
      return result;
    };
    const DataVector& received = packages.at(mortar_id);
    const DataVector expected = expected_package(false);
    CHECK(max_abs_difference(received, expected) < 1e-11);
    // The test distinguishes the two sides of the face.
    CHECK(max_abs_difference(received, expected_package(true)) > 1e-3);
  }
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection.NeighborPackagedData",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  test_face_projection(make_not_null(&gen));
  test_adapter(true);
  test_adapter(false);
  test_adapter_distinct_traces();
}
}  // namespace
