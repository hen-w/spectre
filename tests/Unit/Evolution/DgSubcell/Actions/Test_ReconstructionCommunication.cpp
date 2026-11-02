// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <deque>
#include <iterator>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/CombineVolumeGhostData.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SetInterpolators.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Interpolators.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/MeshForGhostData.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxBoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
class DummyReconstructor {
 public:
  static size_t ghost_zone_size() { return 2; }
  void pup(PUP::er& /*p*/) {}
};

namespace Tags {
struct Reconstructor : db::SimpleTag {
  using type = std::unique_ptr<DummyReconstructor>;
};
}  // namespace Tags

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = ::Tags::Variables<tmpl::list<Var1>>;
  using flux_variables = tmpl::list<Var1>;
};

template <size_t Dim, typename Metavariables, bool UseNodegroupDgElements,
          bool ExtraTesting = false, bool IsAuxiliary = false>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;

  using initial_tags = tmpl::flatten<tmpl::list<
      Tags::Reconstructor, ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
      domain::Tags::Mesh<Dim>, evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::ActiveGrid, domain::Tags::Element<Dim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
      evolution::dg::subcell::Tags::DataForRdmpTci,
      evolution::dg::subcell::Tags::TciDecision,
      evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>,
      ::Tags::Variables<tmpl::list<Var1>>,
      ::Tags::HistoryEvolvedVariables<::Tags::Variables<tmpl::list<Var1>>>,
      evolution::dg::Tags::MortarInfo<Dim>,
      evolution::dg::Tags::MortarMesh<Dim>,
      evolution::dg::Tags::MortarData<Dim>,
      evolution::dg::Tags::MortarNextTemporalId<Dim>,
      domain::Tags::NeighborMesh<Dim>,
      evolution::dg::subcell::Tags::MeshForGhostData<Dim>,
      evolution::dg::subcell::Tags::CellCenteredFlux<tmpl::list<Var1>, Dim>,
      evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<Dim>,
      evolution::dg::subcell::Tags::InterpolatorsFromNeighborDgToFd<Dim>,
      tmpl::conditional_t<
          ExtraTesting,
          tmpl::list<evolution::dg::subcell::Tags::
                         InterpolatorsFromDgToNeighborFd<Dim>,
                     ::domain::Tags::ElementMap<Dim, Frame::Grid>,
                     evolution::dg::subcell::Tags::NeighborActiveGrids<Dim>>,
          tmpl::list<>>,
      evolution::dg::subcell::Tags::ExtensionDirections<Dim>,
      evolution::dg::subcell::Tags::SubcellOptions<Dim>>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::flatten<tmpl::list<
          tmpl::list<ActionTesting::InitializeDataBox<initial_tags>>,
          tmpl::conditional_t<ExtraTesting,
                              tmpl::list<Actions::MutateApply<
                                  evolution::dg::subcell::SetInterpolators<
                                      Dim, Tags::Reconstructor>>>,
                              tmpl::list<>>,
          tmpl::list<evolution::dg::subcell::Actions::SendDataForReconstruction<
                         Dim, typename Metavariables::GhostDataMutator,
                         UseNodegroupDgElements, IsAuxiliary>,
                     evolution::dg::subcell::Actions::
                         ReceiveAndSendDataForReconstruction<
                             Dim, typename Metavariables::GhostDataMutator,
                             UseNodegroupDgElements, IsAuxiliary>,
                     evolution::dg::subcell::Actions::
                         ReceiveDataForReconstruction<Dim,
                                                      IsAuxiliary>>>>>>;
};

template <size_t Dim, bool UseNodegroupDgElements, bool ExtraTesting = false,
          bool IsAuxiliary = false, bool FixedActiveGrids = false>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  static constexpr bool is_auxiliary = IsAuxiliary;
  using component_list = tmpl::list<component<Dim, Metavariables,
                                              UseNodegroupDgElements,
                                              ExtraTesting, IsAuxiliary>>;
  using system = System<Dim>;
  using const_global_cache_tags = tmpl::flatten<tmpl::list<tmpl::conditional_t<
      ExtraTesting, ::domain::Tags::Domain<Dim>, tmpl::list<>>>>;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool ghost_zone_size_invoked;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool ghost_data_mutator_invoked;

  struct SubcellOptions {
    static constexpr bool active_grids_fixed_at_initialization =
        FixedActiveGrids;

    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) {
      return db::get<Tags::Reconstructor>(box).ghost_zone_size();
    }
  };

  struct GhostDataMutator {
    using return_tags = tmpl::list<>;
    using argument_tags = tmpl::list<::Tags::Variables<tmpl::list<Var1>>>;
    static DataVector apply(const Variables<tmpl::list<Var1>>& vars,
                            const size_t rdmp_size) {
      CAPTURE(rdmp_size);
      CAPTURE(Dim);
      CAPTURE(vars.number_of_grid_points());
      CHECK(
          (rdmp_size == 0 or rdmp_size == Dim * vars.number_of_grid_points()));
      DataVector buffer{vars.size() + rdmp_size};
      ghost_data_mutator_invoked = true;
      // make some trivial but testable modification
      DataVector view{buffer.data(), vars.size()};
      view = 2.0 * get(get<Var1>(vars));
      return buffer;
    }
  };
};

template <size_t Dim, bool UseNodegroupDgElements, bool ExtraTesting,
          bool IsAuxiliary, bool FixedActiveGrids>
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim, UseNodegroupDgElements, ExtraTesting, IsAuxiliary,
                   FixedActiveGrids>::ghost_zone_size_invoked = false;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
template <size_t Dim, bool UseNodegroupDgElements, bool ExtraTesting,
          bool IsAuxiliary, bool FixedActiveGrids>
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim, UseNodegroupDgElements, ExtraTesting, IsAuxiliary,
                   FixedActiveGrids>::ghost_data_mutator_invoked = false;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

template <size_t Dim, bool UseNodegroupDgElements>
void test(const bool use_cell_centered_flux) {
  CAPTURE(Dim);
  CAPTURE(use_cell_centered_flux);
  using Interps = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;
  using ExtensionDirs =
      DirectionMap<Dim, interpolators_detail::ExtensionDirection<Dim>>;
  using SubcellOptions = evolution::dg::subcell::SubcellOptions;

  using metavars = Metavariables<Dim, UseNodegroupDgElements>;
  metavars::ghost_zone_size_invoked = false;
  metavars::ghost_data_mutator_invoked = false;
  using comp = component<Dim, metavars, UseNodegroupDgElements>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{}};

  const TimeStepId time_step_id{true, 1, Time{Slab{1.0, 2.0}, {0, 10}}};
  const TimeStepId next_time_step_id{true, 1, Time{Slab{1.0, 2.0}, {1, 10}}};
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const evolution::dg::subcell::ActiveGrid active_grid =
      evolution::dg::subcell::ActiveGrid::Subcell;

  // ^ eta
  // +-+-+> xi
  // |X| |
  // +-+-+
  // | | |
  // +-+-+
  //
  // The "self_id" for the element that we are considering is marked by an X in
  // the diagram. We consider a configuration with one neighbor in the +xi
  // direction (east_id), and (in 2d and 3d) one in the -eta (south_id)
  // direction.
  //
  // In 1d there aren't any projections to test, and in 3d we only have 1
  // element in the z-direction.
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  ElementId<Dim> self_id{};
  ElementId<Dim> east_id{};
  ElementId<Dim> south_id{};  // not used in 1d
  OrientationMap<Dim> orientation = OrientationMap<Dim>::create_aligned();
  // Note: in 2d and 3d it is the neighbor in the lower eta direction that has a
  // non-trivial orientation.

  if constexpr (Dim == 1) {
    self_id = ElementId<Dim>{0, {{{1, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}}}};
    neighbors[Direction<Dim>::upper_xi()] =
        Neighbors<Dim>{{east_id}, OrientationMap<Dim>::create_aligned()};
  } else if constexpr (Dim == 2) {
    orientation = OrientationMap<Dim>{
        std::array{Direction<Dim>::lower_xi(), Direction<Dim>::lower_eta()}};
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{0, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] =
        Neighbors<Dim>{{east_id}, OrientationMap<Dim>::create_aligned()};
    neighbors[Direction<Dim>::lower_eta()] =
        Neighbors<Dim>{{south_id}, orientation};
  } else {
    static_assert(Dim == 3, "Only implemented tests in 1, 2, and 3d");
    orientation = OrientationMap<Dim>{std::array{Direction<Dim>::lower_xi(),
                                                 Direction<Dim>::lower_eta(),
                                                 Direction<Dim>::upper_zeta()}};
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{0, 0}, {0, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] =
        Neighbors<Dim>{{east_id}, OrientationMap<Dim>::create_aligned()};
    neighbors[Direction<Dim>::lower_eta()] =
        Neighbors<Dim>{{south_id}, orientation};
  }
  const Element<Dim> element{self_id, neighbors};

  using NeighborDataMap =
      DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>;
  NeighborDataMap neighbor_data{};
  const DirectionalId<Dim> east_neighbor_id{Direction<Dim>::upper_xi(),
                                            east_id};
  // insert data from one of the neighbors to make sure the send actions clears
  // it.
  neighbor_data[east_neighbor_id] = {};

  using evolved_vars_tags = tmpl::list<Var1>;
  Variables<evolved_vars_tags> evolved_vars{
      subcell_mesh.number_of_grid_points()};
  // Set Var1 to the logical coords, just need some data
  get(get<Var1>(evolved_vars)) = get<0>(logical_coordinates(subcell_mesh));
  TimeSteppers::History<Variables<evolved_vars_tags>> time_stepper_history{};

  using CellCenteredFluxTag =
      evolution::dg::subcell::Tags::CellCenteredFlux<tmpl::list<Var1>, Dim>;
  typename CellCenteredFluxTag::type cell_centered_flux{};
  if (use_cell_centered_flux) {
    cell_centered_flux = typename CellCenteredFluxTag::type{
        subcell_mesh.number_of_grid_points()};
    const auto logical_coords = logical_coordinates(subcell_mesh);
    for (size_t i = 0; i < Dim; ++i) {
      get<::Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>>(
          cell_centered_flux.value())
          .get(i) = logical_coords.get(i) + (static_cast<double>(i) + 1.0);
    }
  }

  using MortarInfo = typename evolution::dg::Tags::MortarInfo<Dim>::type;
  using MortarData = typename evolution::dg::Tags::MortarData<Dim>::type;
  using MortarMesh = typename evolution::dg::Tags::MortarMesh<Dim>::type;
  using MortarNextId =
      typename evolution::dg::Tags::MortarNextTemporalId<Dim>::type;
  MortarInfo mortar_info{};
  MortarData mortar_data{};
  MortarMesh mortar_mesh{};
  MortarNextId mortar_next_id{};
  mortar_info[east_neighbor_id] = evolution::dg::MortarInfo<Dim>{
      {.time_stepping_policy = evolution::dg::TimeSteppingPolicy::EqualRate}};
  mortar_data[east_neighbor_id] = {};
  mortar_mesh[east_neighbor_id] = {dg_mesh.slice_away(0)};
  mortar_next_id[east_neighbor_id] = {};
  if constexpr (Dim > 1) {
    const DirectionalId<Dim> south_neighbor_id{Direction<Dim>::lower_eta(),
                                               south_id};
    mortar_info[south_neighbor_id] = evolution::dg::MortarInfo<Dim>{
        {.time_stepping_policy = evolution::dg::TimeSteppingPolicy::EqualRate}};
    mortar_data[south_neighbor_id] = {};
    mortar_mesh[south_neighbor_id] = {dg_mesh.slice_away(1)};
    mortar_next_id[south_neighbor_id] = {};
  }

  size_t neighbor_tci_decision = 0;
  typename evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>::type
      neighbor_decision{};
  for (const auto& [direction, neighbor_ids] : neighbors) {
    (void)direction;
    for (const auto& neighbor_id : neighbor_ids) {
      neighbor_decision.insert(
          std::pair{DirectionalId<Dim>{direction, neighbor_id}, 0});
      // Initialize neighbors with garbage data. We won't ever run any actions
      // on them, we just need to insert them to make sure things are sent to
      // the right places. We'll check their inboxes directly.
      ActionTesting::emplace_array_component_and_initialize<comp>(
          &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
          neighbor_id,
          {std::make_unique<DummyReconstructor>(),
           time_step_id,
           next_time_step_id,
           Mesh<Dim>{},
           Mesh<Dim>{},
           active_grid,
           Element<Dim>{},
           NeighborDataMap{},
           evolution::dg::subcell::RdmpTciData{},
           neighbor_tci_decision,
           typename evolution::dg::subcell::Tags::NeighborTciDecisions<
               Dim>::type{},
           Variables<evolved_vars_tags>{},
           time_stepper_history,
           MortarInfo{},
           MortarMesh{},
           MortarData{},
           MortarNextId{},
           typename domain::Tags::NeighborMesh<Dim>::type{},
           typename evolution::dg::subcell::Tags::MeshForGhostData<Dim>::type{},
           cell_centered_flux,
           Interps{},
           Interps{},
           ExtensionDirs{},
           SubcellOptions{}});
      ++neighbor_tci_decision;
    }
  }
  Interps fd_to_neighbor_fd_interpolants{};
  Interps neighbor_dg_to_fd_interpolants{};
  ExtensionDirs extension_directions{};
  SubcellOptions subcell_options(
      4.0, 1, 2.0e-3, 2.0e-4, false, false,
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
      std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1);
  const int self_tci_decision = 100;
  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, self_id,
      {std::make_unique<DummyReconstructor>(), time_step_id, next_time_step_id,
       dg_mesh, subcell_mesh, active_grid, element, neighbor_data,
       // Explicitly set RDMP data since this would be set previously by the TCI
       evolution::dg::subcell::RdmpTciData{{max(get(get<Var1>(evolved_vars)))},
                                           {min(get(get<Var1>(evolved_vars)))}},
       self_tci_decision, neighbor_decision, evolved_vars, time_stepper_history,
       mortar_info, mortar_mesh, mortar_data, mortar_next_id,
       typename domain::Tags::NeighborMesh<Dim>::type{},
       typename evolution::dg::subcell::Tags::MeshForGhostData<Dim>::type{},
       cell_centered_flux, fd_to_neighbor_fd_interpolants,
       neighbor_dg_to_fd_interpolants, extension_directions, subcell_options});

  using ghost_data_tag =
      evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>;
  using rdmp_tci_data_tag = evolution::dg::subcell::Tags::DataForRdmpTci;
  using ActionTesting::get_databox_tag;
  CHECK(get_databox_tag<comp, ghost_data_tag>(runner, self_id).size() == 1);
  CHECK(get_databox_tag<comp, ghost_data_tag>(runner, self_id)
            .count(east_neighbor_id) == 1);

  // Run the SendDataForReconstruction action on self_id
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  CHECK(get_databox_tag<comp, ghost_data_tag>(runner, self_id).empty());

  // Check local RDMP data
  const evolution::dg::subcell::RdmpTciData& rdmp_tci_data =
      get_databox_tag<comp, rdmp_tci_data_tag>(runner, self_id);
  REQUIRE(rdmp_tci_data.max_variables_values.size() == 1);
  CHECK(rdmp_tci_data.max_variables_values[0] ==
        max(get(get<Var1>(evolved_vars))));
  REQUIRE(rdmp_tci_data.min_variables_values.size() == 1);
  CHECK(rdmp_tci_data.min_variables_values[0] ==
        min(get(get<Var1>(evolved_vars))));
  // Check data sent to neighbors
  const auto& directions_to_slice = element.internal_boundaries();
  const size_t ghost_zone_size = 2;
  const size_t rdmp_size = rdmp_tci_data.max_variables_values.size() +
                           rdmp_tci_data.min_variables_values.size();
  const DirectionMap<Dim, DataVector> all_sliced_data =
      [&evolved_vars, &subcell_mesh, ghost_zone_size, &directions_to_slice,
       &cell_centered_flux, &fd_to_neighbor_fd_interpolants]() {
        (void)ghost_zone_size;
        if (cell_centered_flux.has_value()) {
          DataVector buffer{evolved_vars.size() +
                            cell_centered_flux.value().size()};
          std::copy(evolved_vars.data(),
                    std::next(evolved_vars.data(),
                              static_cast<std::ptrdiff_t>(evolved_vars.size())),
                    buffer.data());
          std::copy(cell_centered_flux.value().data(),
                    std::next(cell_centered_flux.value().data(),
                              static_cast<std::ptrdiff_t>(
                                  cell_centered_flux.value().size())),
                    std::next(buffer.data(), static_cast<std::ptrdiff_t>(
                                                 evolved_vars.size())));
          return evolution::dg::subcell::slice_data(
              buffer, subcell_mesh.extents(), ghost_zone_size,
              directions_to_slice, 0, fd_to_neighbor_fd_interpolants);
        } else {
          return evolution::dg::subcell::slice_data(
              evolved_vars, subcell_mesh.extents(), ghost_zone_size,
              directions_to_slice, 0, fd_to_neighbor_fd_interpolants);
        }
      }();
  {
    const auto& east_sliced_neighbor_data =
        all_sliced_data.at(east_neighbor_id.direction());
    DataVector expected_east_data{east_sliced_neighbor_data.size() + rdmp_size};
    std::copy(east_sliced_neighbor_data.begin(),
              east_sliced_neighbor_data.end(), expected_east_data.begin());
    // We only multiple Var1 by 2, not the fluxes.
    const size_t bound_for_vars_to_multiply =
        east_sliced_neighbor_data.size() /
        (cell_centered_flux.has_value() ? (1 + Dim) : 1_st);
    for (size_t i = 0; i < bound_for_vars_to_multiply; i++) {
      expected_east_data[i] *= 2.0;
    }
    std::copy(rdmp_tci_data.max_variables_values.cbegin(),
              rdmp_tci_data.max_variables_values.cend(),
              std::prev(expected_east_data.end(), static_cast<int>(rdmp_size)));
    std::copy(
        rdmp_tci_data.min_variables_values.cbegin(),
        rdmp_tci_data.min_variables_values.cend(),
        std::prev(expected_east_data.end(),
                  static_cast<int>(rdmp_tci_data.min_variables_values.size())));

    const auto& east_data =
        ActionTesting::get_inbox_tag<
            comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                      Dim, UseNodegroupDgElements>>(runner, east_id)
            .messages;
    const auto east_message =
        alg::find_if(east_data.at(time_step_id), [&](const auto& entry) {
          return entry.first ==
                 DirectionalId<Dim>{Direction<Dim>::lower_xi(), self_id};
        });
    REQUIRE(east_message != east_data.at(time_step_id).end());
    CHECK_ITERABLE_APPROX(expected_east_data,
                          east_message->second.ghost_cell_data.value());
    CHECK(east_message->second.tci_status == self_tci_decision);
  }
  if constexpr (Dim > 1) {
    const auto direction = Direction<Dim>::lower_eta();

    std::array<size_t, Dim> slice_extents{};
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(slice_extents, d) = subcell_mesh.extents(d);
    }
    gsl::at(slice_extents, direction.dimension()) = ghost_zone_size;

    const auto& south_sliced_neighbor_data = all_sliced_data.at(direction);

    DataVector expected_south_data{south_sliced_neighbor_data.size() +
                                   rdmp_size};
    DataVector expected_south_data_view{expected_south_data.data(),
                                        south_sliced_neighbor_data.size()};
    // Note: We do not orient the variables because that's currently handled
    // by the interpolation code.
    //
    // orient_variables(make_not_null(&expected_south_data_view),
    //                  all_sliced_data.at(direction),
    //                  Index<Dim>{slice_extents},
    //                  orientation);
    expected_south_data_view = all_sliced_data.at(direction);
    const size_t bound_for_vars_to_multiply =
        expected_south_data_view.size() /
        (cell_centered_flux.has_value() ? (1 + Dim) : 1_st);
    for (size_t i = 0; i < bound_for_vars_to_multiply; i++) {
      expected_south_data_view[i] *= 2.0;
    }
    std::copy(
        rdmp_tci_data.max_variables_values.cbegin(),
        rdmp_tci_data.max_variables_values.cend(),
        std::prev(expected_south_data.end(), static_cast<int>(rdmp_size)));
    std::copy(
        rdmp_tci_data.min_variables_values.cbegin(),
        rdmp_tci_data.min_variables_values.cend(),
        std::prev(expected_south_data.end(),
                  static_cast<int>(rdmp_tci_data.min_variables_values.size())));

    const auto& south_data =
        ActionTesting::get_inbox_tag<
            comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                      Dim, UseNodegroupDgElements>>(runner, south_id)
            .messages;
    const auto south_message =
        alg::find_if(south_data.at(time_step_id), [&](const auto& entry) {
          return entry.first ==
                 DirectionalId<Dim>{orientation(direction.opposite()), self_id};
        });
    REQUIRE(south_message != south_data.at(time_step_id).end());
    CHECK(expected_south_data == south_message->second.ghost_cell_data.value());
    CHECK(south_message->second.tci_status == self_tci_decision);
  }

  // Set the inbox data on self_id and then check that it gets processed
  // correctly. We need to check both if a neighbor is doing DG or if a neighbor
  // is doing subcell.
  auto& self_inbox = ActionTesting::get_inbox_tag<
      comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                Dim, UseNodegroupDgElements>>(make_not_null(&runner), self_id);

  // Send data from east neighbor
  DataVector east_ghost_cells_and_rdmp{};
  {
    const auto face_mesh = dg_mesh.slice_away(0);
    east_ghost_cells_and_rdmp = DataVector{
        subcell_mesh.slice_away(0).number_of_grid_points() * ghost_zone_size +
        rdmp_size};
    alg::iota(east_ghost_cells_and_rdmp, 0.0);
    DataVector boundary_data{face_mesh.number_of_grid_points() * (2 + Dim)};
    alg::iota(boundary_data, 1000.0);
    const DirectionalId<Dim> east_dir_id{Direction<Dim>::upper_xi(), east_id};
    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
        Dim, UseNodegroupDgElements>::
        insert_into_inbox(
            make_not_null(&self_inbox), time_step_id,
            std::pair{east_dir_id,
                      evolution::dg::BoundaryData<Dim>{
                          dg_mesh,
                          // subcell_mesh because we are sending the
                          // projected data right now.
                          subcell_mesh, mortar_mesh.at(east_dir_id),
                          east_ghost_cells_and_rdmp, boundary_data,
                          next_time_step_id, -10}});
  }
  [[maybe_unused]] DataVector south_ghost_cells_and_rdmp{};
  if constexpr (Dim > 1) {
    south_ghost_cells_and_rdmp = DataVector{
        subcell_mesh.slice_away(0).number_of_grid_points() * ghost_zone_size +
        rdmp_size};
    alg::iota(south_ghost_cells_and_rdmp, 10000.0);
    *std::prev(south_ghost_cells_and_rdmp.end()) = -10.0;
    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
        Dim, UseNodegroupDgElements>::
        insert_into_inbox(
            make_not_null(&self_inbox), time_step_id,
            std::pair{
                DirectionalId<Dim>{Direction<Dim>::lower_eta(), south_id},
                evolution::dg::BoundaryData<Dim>{
                    dg_mesh,
                    // subcell_mesh because we are sending the projected
                    // data right now.
                    subcell_mesh, std::nullopt, south_ghost_cells_and_rdmp,
                    std::nullopt, next_time_step_id, -15}});
  }

  // Run the ReceiveAndSendDataForReconstruction action on self_id
  // As for this test, we are using EnableExtensionDirections = false,
  // This action will not do anything.
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  // Run the ReceiveDataForReconstruction action on self_id
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  // Check the received data was stored correctly
  const auto& ghost_data_from_box =
      get_databox_tag<comp, ghost_data_tag>(runner, self_id);
  CHECK(rdmp_tci_data.max_variables_values.size() == 1);
  CHECK(approx(rdmp_tci_data.max_variables_values[0]) ==
        (Dim > 1 ? static_cast<double>(
                       ghost_zone_size *
                       subcell_mesh.slice_away(1).number_of_grid_points()) +
                       10000.0
                 : static_cast<double>(ghost_zone_size)));
  CHECK(rdmp_tci_data.min_variables_values.size() == 1);
  CHECK(approx(rdmp_tci_data.min_variables_values[0]) ==
        (Dim > 1 ? -10.0 : min(get<0>(logical_coordinates(subcell_mesh)))));

  REQUIRE(ghost_data_from_box.find(east_neighbor_id) !=
          ghost_data_from_box.end());
  const DataVector east_ghost_cells_and_rdmp_view{
      east_ghost_cells_and_rdmp.data(),
      east_ghost_cells_and_rdmp.size() - rdmp_size};
  CHECK(ghost_data_from_box.find(east_neighbor_id)
            ->second.neighbor_ghost_data_for_reconstruction() ==
        east_ghost_cells_and_rdmp_view);
  CHECK(
      get_databox_tag<comp,
                      evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>(
          runner, self_id)
          .at(east_neighbor_id) == -10);
  if constexpr (Dim > 1) {
    const DirectionalId<Dim> south_neighbor_id{Direction<Dim>::lower_eta(),
                                               south_id};
    REQUIRE(ghost_data_from_box.find(south_neighbor_id) !=
            ghost_data_from_box.end());
    const DataVector south_ghost_cells_and_rdmp_view{
        south_ghost_cells_and_rdmp.data(),
        south_ghost_cells_and_rdmp.size() - rdmp_size};
    CHECK(ghost_data_from_box.find(south_neighbor_id)
              ->second.neighbor_ghost_data_for_reconstruction() ==
          south_ghost_cells_and_rdmp_view);
    CHECK(get_databox_tag<
              comp, evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>(
              runner, self_id)
              .at(south_neighbor_id) == -15);
  }

  // Check that we got a neighbor mesh from all neighbors.
  size_t total_neighbors = 0;
  const auto& neighbor_meshes =
      get_databox_tag<comp, ::domain::Tags::NeighborMesh<Dim>>(
          runner, self_id);
  const auto& meshes_for_ghost_data = get_databox_tag<
      comp, evolution::dg::subcell::Tags::MeshForGhostData<Dim>>(runner,
                                                                 self_id);
  for (const auto& [direction, neighbors_in_direction] : element.neighbors()) {
    for (const auto& neighbor : neighbors_in_direction) {
      const auto it =
          neighbor_meshes.find(DirectionalId<Dim>{direction, neighbor});
      REQUIRE(it != neighbor_meshes.end());
      CHECK(it->second == dg_mesh);
      const auto ghost_it =
          meshes_for_ghost_data.find(DirectionalId<Dim>{direction, neighbor});
      REQUIRE(ghost_it != meshes_for_ghost_data.end());
      CHECK(ghost_it->second == subcell_mesh);
      ++total_neighbors;
    }
  }
  CHECK(neighbor_meshes.size() == total_neighbors);
}

template <size_t Dim>
Element<Dim> create_element(const std::vector<Block<Dim>>& blocks,
                            const ElementId<Dim>& self_id,
                            const std::vector<Direction<Dim>>& directions,
                            const std::vector<ElementId<Dim>>& neighbor_ids) {
  ASSERT(neighbor_ids.size() == directions.size(),
         "The number of neighbor IDs must match the number of directions.");
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  const size_t self_block_id = self_id.block_id();
  for (size_t i = 0; i < directions.size(); ++i) {
    const auto& direction = directions[i];
    const auto& neighbor_id = neighbor_ids[i];
    neighbors[direction] = Neighbors<Dim>{
        {neighbor_id},
        self_block_id == neighbor_id.block_id()
            ? OrientationMap<Dim>::create_aligned()
            : blocks[self_block_id].neighbors().at(direction).orientation(
                  neighbor_id.block_id())};
  }
  return Element<Dim>{self_id, std::move(neighbors)};
}

void add_rdmp_data_to_datavector(
    gsl::not_null<DataVector*> data_to_fill, const size_t original_data_size,
    const evolution::dg::subcell::RdmpTciData& rdmp_tci_data) {
  const size_t rdmp_max_values_size = rdmp_tci_data.max_variables_values.size();

  // Copy max_variables_values after the original data
  std::copy(
      rdmp_tci_data.max_variables_values.cbegin(),
      rdmp_tci_data.max_variables_values.cend(),
      data_to_fill->begin() + static_cast<std::ptrdiff_t>(original_data_size));

  // Copy min_variables_values after max_variables_values
  std::copy(
      rdmp_tci_data.min_variables_values.cbegin(),
      rdmp_tci_data.min_variables_values.cend(),
      data_to_fill->begin() + static_cast<std::ptrdiff_t>(
                                  original_data_size + rdmp_max_values_size));
}

template <size_t Dim, bool UseNodegroupDgElements, typename MetaVars>
DataVector compute_expected_ghost_data(
    const ActionTesting::MockRuntimeSystem<MetaVars>& runner,
    const Element<Dim>& element, const Direction<Dim>& direction,
    const ::Mesh<3> subcell_mesh, const bool enable_extension,
    const bool use_cell_centered_flux,
    const Variables<tmpl::list<Var1>>& evolved_vars,
    const evolution::dg::subcell::RdmpTciData& rdmp_tci_data,
    const TimeStepId time_step_id, const size_t ghost_zone_size) {
  using comp = tmpl::front<typename MetaVars::component_list>;
  using CellCenteredFluxTag =
      evolution::dg::subcell::Tags::CellCenteredFlux<tmpl::list<Var1>, Dim>;
  using ActionTesting::get_databox_tag;
  const auto& element_id = element.id();
  const auto& fd_to_neighbor_fd_interpolants = get_databox_tag<
      comp, evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<Dim>>(
      runner, element_id);
  const auto& extension_directions = get_databox_tag<
      comp, evolution::dg::subcell::Tags::ExtensionDirections<Dim>>(runner,
                                                                    element_id);
  const size_t rdmp_size = rdmp_tci_data.max_variables_values.size() +
                           rdmp_tci_data.min_variables_values.size();

  if (enable_extension and extension_directions.contains(direction)) {
    const auto& receiver_id = element.neighbors().at(direction).ids().begin();
    const auto& extender_direction =
        extension_directions.at(direction).direction_to_extend;
    const auto& extender_id =
        element.neighbors().at(extender_direction).ids().begin();
    const auto& data_from_extender =
        alg::find_if(
            ActionTesting::get_inbox_tag<
                comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                          Dim, UseNodegroupDgElements, MetaVars::is_auxiliary>>(
                runner, element_id)
                .messages.at(time_step_id),
            [&](const auto& entry) {
              return entry.first ==
                     DirectionalId<Dim>{extender_direction, *extender_id};
            })
            ->second.ghost_cell_data.value();

    const DataVector relevant_ghost_data;
    const size_t ghost_data_size = data_from_extender.size() - rdmp_size;
    make_const_view(make_not_null(&relevant_ghost_data), data_from_extender, 0,
                    ghost_data_size);

    DataVector volume_data_to_slice = [&evolved_vars, use_cell_centered_flux,
                                       &runner, &element_id]() {
      if (use_cell_centered_flux) {
        const auto& cell_centered_flux =
            get_databox_tag<comp, CellCenteredFluxTag>(
                runner, element_id);
        DataVector buffer{evolved_vars.size() +
                          cell_centered_flux.value().size()};
        std::copy(evolved_vars.data(),
                  std::next(evolved_vars.data(),
                            static_cast<std::ptrdiff_t>(evolved_vars.size())),
                  buffer.data());
        std::copy(cell_centered_flux.value().data(),
                  std::next(cell_centered_flux.value().data(),
                            static_cast<std::ptrdiff_t>(
                                cell_centered_flux.value().size())),
                  std::next(buffer.data(),
                            static_cast<std::ptrdiff_t>(evolved_vars.size())));
        return buffer;
      } else {
        return get(get<Var1>(evolved_vars));
      }
    }();
    for (size_t i = 0; i < evolved_vars.size(); ++i) {
      volume_data_to_slice[i] *= 2.0;
    }
    const DataVector combined_data =
        evolution::dg::subcell::combine_volume_ghost_data(
            volume_data_to_slice, relevant_ghost_data, subcell_mesh.extents(),
            ghost_zone_size, extender_direction);
    DataVector exepected_ghost_data{ghost_data_size + rdmp_size};
    auto result_span =
        gsl::make_span(exepected_ghost_data.data(), ghost_data_size);
    const auto& interpolant =
        fd_to_neighbor_fd_interpolants
            .at(DirectionalId<Dim>{direction, *receiver_id})
            .value();
    interpolant.interpolate(
        make_not_null(&result_span),
        gsl::make_span(combined_data.data(), combined_data.size()));
    add_rdmp_data_to_datavector(make_not_null(&exepected_ghost_data),
                                ghost_data_size, rdmp_tci_data);
    return exepected_ghost_data;
  } else {
    std::unordered_set<Direction<Dim>> directions_to_slice{direction};
    const DirectionMap<Dim, DataVector> sliced_volume_data =
        [&evolved_vars, &subcell_mesh, ghost_zone_size, &directions_to_slice,
         &fd_to_neighbor_fd_interpolants, &runner, use_cell_centered_flux,
         &element_id]() {
          if (use_cell_centered_flux) {
            const auto& cell_centered_flux =
                get_databox_tag<comp, CellCenteredFluxTag>(
                    runner, element_id);
            DataVector buffer{evolved_vars.size() +
                              cell_centered_flux.value().size()};
            std::copy(
                evolved_vars.data(),
                std::next(evolved_vars.data(),
                          static_cast<std::ptrdiff_t>(evolved_vars.size())),
                buffer.data());
            std::copy(cell_centered_flux.value().data(),
                      std::next(cell_centered_flux.value().data(),
                                static_cast<std::ptrdiff_t>(
                                    cell_centered_flux.value().size())),
                      std::next(buffer.data(), static_cast<std::ptrdiff_t>(
                                                   evolved_vars.size())));
            return evolution::dg::subcell::slice_data(
                buffer, subcell_mesh.extents(), ghost_zone_size,
                directions_to_slice, 0, fd_to_neighbor_fd_interpolants);
          } else {
            return evolution::dg::subcell::slice_data(
                evolved_vars, subcell_mesh.extents(), ghost_zone_size,
                directions_to_slice, 0, fd_to_neighbor_fd_interpolants);
          }
        }();
    const DataVector& sliced_data_for_direction =
        sliced_volume_data.at(direction);
    const size_t sliced_data_size = sliced_data_for_direction.size();
    const size_t total_size = sliced_data_size + rdmp_size;
    DataVector expected_ghost_data(total_size);

    // Copy the sliced data into the beginning of the result
    std::copy(sliced_data_for_direction.begin(),
              sliced_data_for_direction.end(), expected_ghost_data.begin());
    // We only multiple Var1 by 2, not the fluxes.
    const size_t bound_for_vars_to_multiply =
        sliced_data_size / (use_cell_centered_flux ? (1 + Dim) : 1_st);

    for (size_t i = 0; i < bound_for_vars_to_multiply; i++) {
      expected_ghost_data[i] *= 2.0;
    }
    add_rdmp_data_to_datavector(make_not_null(&expected_ghost_data),
                                sliced_data_size, rdmp_tci_data);

    return expected_ghost_data;
  }
}

// Test ReceiveAndSendDataForReconstruction action
// for both EnableExtensionDirections options.
//
// With `FixedActiveGrids` (in the auxiliary exchange), the sender treats the
// receiver as an element using DG: the receiver gets a message without ghost
// data from the sender, directly or (with extension directions) after the
// sender has received the data of its neighbors, and the sender computes no
// ghost data for its forwarding. All other messages are unchanged.
template <size_t Dim, bool UseNodegroupDgElements, bool IsAuxiliary = false,
          bool FixedActiveGrids = false>
void test_receive_and_send_data(const bool enable_extension,
                                const bool use_cell_centered_flux) {
  CAPTURE(Dim);
  CAPTURE(enable_extension);
  CAPTURE(use_cell_centered_flux);
  CAPTURE(IsAuxiliary);
  CAPTURE(FixedActiveGrids);
  using Inbox = evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
      Dim, UseNodegroupDgElements, IsAuxiliary>;
  // create Sphere domain, and 4 releveant elements.
  domain::creators::register_derived_with_charm();

  using Sphere = domain::creators::Sphere;
  using Interior = std::variant<domain::creators::Sphere::Excision,
                                domain::creators::Sphere::InnerCube>;
  size_t refinement_level = 2;
  const Sphere domain_creator{1.,
                              5.,
                              Interior{Sphere::Excision{}},
                              refinement_level,
                              6_st,
                              true,
                              std::nullopt,
                              {},
                              domain::CoordinateMaps::Distribution::Linear,
                              ShellWedges::All,
                              std::nullopt};
  auto domain = domain_creator.create_domain();

  using metavars = Metavariables<Dim, UseNodegroupDgElements, true, IsAuxiliary,
                                 FixedActiveGrids>;
  metavars::ghost_zone_size_invoked = false;
  metavars::ghost_data_mutator_invoked = false;
  using comp =
      component<Dim, metavars, UseNodegroupDgElements, true, IsAuxiliary>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{std::move(domain_creator.create_domain())}};

  const std::vector<Block<3>>& blocks = domain.blocks();
  // Create the elements that we will use in the test.
  // We creat a sender, an extender, a receiver, and a receiver extender.
  // sender and recevier are in different blocks.
  // extender and receiver extender are in the same block as sender and
  // receiver, respectively. they provide the data that is sent to the
  // sender and receiver such that the sender and receiver can
  // perform the new action when enable_extension is true.
  const auto sender =
      create_element<3>(blocks, ElementId<3>{0, {{{2, 0}, {2, 0}, {2, 0}}}},
                        {Direction<3>::upper_eta(), Direction<3>::lower_xi()},
                        {ElementId<3>{0, {{{2, 0}, {2, 1}, {2, 0}}}},
                         ElementId<3>{5, {{{2, 3}, {2, 3}, {2, 0}}}}});
  const auto extender =
      create_element<3>(blocks, ElementId<3>{0, {{{2, 0}, {2, 1}, {2, 0}}}},
                        {Direction<3>::lower_eta()},
                        {ElementId<3>{0, {{{2, 0}, {2, 0}, {2, 0}}}}});
  const auto receiver =
      create_element<3>(blocks, ElementId<3>{5, {{{2, 3}, {2, 3}, {2, 0}}}},
                        {Direction<3>::upper_eta(), Direction<3>::lower_xi()},
                        {ElementId<3>{0, {{{2, 0}, {2, 0}, {2, 0}}}},
                         ElementId<3>{5, {{{2, 2}, {2, 3}, {2, 0}}}}});
  const auto receiver_extender =
      create_element<3>(blocks, ElementId<3>{5, {{{2, 2}, {2, 3}, {2, 0}}}},
                        {Direction<3>::upper_xi()},
                        {ElementId<3>{5, {{{2, 3}, {2, 3}, {2, 0}}}}});

  const std::array<const Element<Dim>*, 4> elements{
      &sender, &extender, &receiver, &receiver_extender};

  // Create a map from ElementId to Element* for easy access
  std::unordered_map<ElementId<Dim>, const Element<Dim>*> id_to_element;
  for (const auto* element : elements) {
    id_to_element[element->id()] = element;
  }

  using Interps = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;
  using ExtensionDirs =
      DirectionMap<Dim, interpolators_detail::ExtensionDirection<Dim>>;
  using SubcellOptions = evolution::dg::subcell::SubcellOptions;
  const SubcellOptions subcell_options(
      4.0, 1, 2.0e-3, 2.0e-4, enable_extension, enable_extension,
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
      std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1);
  const ::Mesh<3> dg_mesh{6, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const ::Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  const TimeStepId time_step_id{true, 1, Time{Slab{1.0, 2.0}, {0, 10}}};
  const TimeStepId next_time_step_id{true, 1, Time{Slab{1.0, 2.0}, {1, 10}}};
  const evolution::dg::subcell::ActiveGrid active_grid =
      evolution::dg::subcell::ActiveGrid::Subcell;
  using MortarInfo = typename evolution::dg::Tags::MortarInfo<Dim>::type;
  using MortarData = typename evolution::dg::Tags::MortarData<Dim>::type;
  using MortarMesh = typename evolution::dg::Tags::MortarMesh<Dim>::type;
  using MortarNextId =
      typename evolution::dg::Tags::MortarNextTemporalId<Dim>::type;
  using ActionTesting::get_databox_tag;
  using evolved_vars_tags = tmpl::list<Var1>;
  Variables<evolved_vars_tags> evolved_vars{
      subcell_mesh.number_of_grid_points()};
  // Set Var1 to the logical coords, just need some data
  get(get<Var1>(evolved_vars)) = get<0>(logical_coordinates(subcell_mesh));
  TimeSteppers::History<Variables<evolved_vars_tags>> time_stepper_history{};

  using CellCenteredFluxTag =
      evolution::dg::subcell::Tags::CellCenteredFlux<tmpl::list<Var1>, Dim>;
  typename CellCenteredFluxTag::type cell_centered_flux{};
  if (use_cell_centered_flux) {
    cell_centered_flux = typename CellCenteredFluxTag::type{
        subcell_mesh.number_of_grid_points()};
    const auto logical_coords = logical_coordinates(subcell_mesh);
    for (size_t i = 0; i < Dim; ++i) {
      get<::Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>>(
          cell_centered_flux.value())
          .get(i) = logical_coords.get(i) + (static_cast<double>(i) + 1.0);
    }
  }
  Interps fd_to_neighbor_fd_interpolants{};
  Interps neighbor_dg_to_fd_interpolants{};
  Interps dg_to_neighbor_fd_interpolants{};
  ExtensionDirs extension_directions{};

  const int self_tci_decision = 100;

  for (const auto* element : elements) {
    const auto& element_id = element->id();
    DirectionalIdMap<Dim, evolution::dg::subcell::GhostData> neighbor_data{};
    typename evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>::type
        neighbor_decision{};

    MortarInfo mortar_info{};
    MortarData mortar_data{};
    MortarMesh mortar_mesh{};
    MortarNextId mortar_next_id{};
    typename evolution::dg::subcell::Tags::NeighborActiveGrids<Dim>::type
        neighbor_active_grids{};

    for (const auto& [direction, neighbors] : element->neighbors()) {
      for (const auto& neighbor_id : neighbors.ids()) {
        const DirectionalId<Dim> mortar_id{direction, neighbor_id};
        neighbor_data[mortar_id] = evolution::dg::subcell::GhostData{};
        neighbor_decision.insert(std::pair{mortar_id, 100});
        neighbor_active_grids.emplace(
            mortar_id,
            (element_id == sender.id() and neighbor_id == receiver.id())
                ? evolution::dg::subcell::ActiveGrid::Dg
                : evolution::dg::subcell::ActiveGrid::Subcell);

        mortar_info[mortar_id] = evolution::dg::MortarInfo<Dim>{
            {.time_stepping_policy =
                 evolution::dg::TimeSteppingPolicy::EqualRate}};
        mortar_data[mortar_id] = {};
        mortar_mesh[mortar_id] = dg_mesh.slice_away(direction.dimension());
        mortar_next_id[mortar_id] = {};
      }
    }

    ActionTesting::emplace_array_component_and_initialize<comp>(
        &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
        element_id,
        {std::make_unique<DummyReconstructor>(), time_step_id,
         next_time_step_id, dg_mesh, subcell_mesh, active_grid, *element,
         neighbor_data,
         // Explicitly set RDMP data since this would be set previously by the
         // TCI
         evolution::dg::subcell::RdmpTciData{
             {max(get(get<Var1>(evolved_vars)))},
             {min(get(get<Var1>(evolved_vars)))}},
         self_tci_decision, neighbor_decision, evolved_vars,
         time_stepper_history, mortar_info, mortar_mesh, mortar_data,
         mortar_next_id, typename domain::Tags::NeighborMesh<Dim>::type{},
         typename evolution::dg::subcell::Tags::MeshForGhostData<Dim>::type{},
         cell_centered_flux, fd_to_neighbor_fd_interpolants,
         neighbor_dg_to_fd_interpolants, dg_to_neighbor_fd_interpolants,
         ElementMap<3, Frame::Grid>{element_id, blocks[element_id.block_id()]},
         neighbor_active_grids, extension_directions, subcell_options});
  }

  // SetInterpolator
  for (const auto* element : elements) {
    const auto& element_id = element->id();
    ActionTesting::next_action<comp>(make_not_null(&runner), element_id);
    if (not enable_extension) {
      const size_t extension_directions_size =
          get_databox_tag<
              comp, evolution::dg::subcell::Tags::ExtensionDirections<Dim>>(
              runner, element_id)
              .size();
      REQUIRE(extension_directions_size == 0);
    }
  }

  // Let use generate expected sender_ghost_data_non_block_boundary
  // and sender_ghost_data_block_boundary.
  const size_t ghost_zone_size = 2;
  using rdmp_tci_data_tag = evolution::dg::subcell::Tags::DataForRdmpTci;
  const evolution::dg::subcell::RdmpTciData& rdmp_tci_data =
      get_databox_tag<comp, rdmp_tci_data_tag>(runner, sender.id());

  // SendDataForReconstruction
  for (const auto* element : elements) {
    const auto& element_id = element->id();
    ActionTesting::next_action<comp>(make_not_null(&runner), element_id);
  }
  // Check that the inbox has the expected data size depending on the
  // enable_extension option.
  for (const auto* element : elements) {
    const auto& element_id = element->id();
    const size_t neighbor_size = element->neighbors().size();
    const size_t extension_directions_size =
        get_databox_tag<comp,
                        evolution::dg::subcell::Tags::ExtensionDirections<Dim>>(
            runner, element_id)
            .size();
    const size_t received_data_size =
        ActionTesting::get_inbox_tag<comp, Inbox>(runner, element_id)
            .messages.at(time_step_id)
            .size();
    REQUIRE(received_data_size == (neighbor_size - extension_directions_size));
  }

  // ReceiveAndSendDataForReconstruction: the ghost data are computed by the
  // elements with extension directions, except by the sender if its only
  // problematic neighbor (the receiver) is treated as using DG.
  for (const auto* element : elements) {
    const auto& element_id = element->id();
    metavars::ghost_data_mutator_invoked = false;
    ActionTesting::next_action<comp>(make_not_null(&runner), element_id);
    const bool forwards =
        not get_databox_tag<
                comp, evolution::dg::subcell::Tags::ExtensionDirections<Dim>>(
                runner, element_id)
                .empty();
    CHECK(metavars::ghost_data_mutator_invoked ==
          (forwards and not(FixedActiveGrids and element_id == sender.id())));
  }
  // The inbox size should match the neighbor size at this point for
  // all elements, regardless of the enable_extension option.
  // Compare the contents as well between the data in the inbox
  // versus the expected data.
  for (const auto* element : elements) {
    const auto& element_id = element->id();
    const size_t neighbor_size = element->neighbors().size();
    const size_t received_data_size =
        ActionTesting::get_inbox_tag<comp, Inbox>(runner, element_id)
            .messages.at(time_step_id)
            .size();
    REQUIRE(received_data_size == neighbor_size);

    for (const auto& [direction, neighbors] : element->neighbors()) {
      // only one neighbor in each direction
      const auto& neighbor_id = neighbors.ids().begin();
      const DirectionalId<Dim> directional_id{direction, *neighbor_id};
      const auto& orientation = neighbors.orientation(*neighbor_id);
      const Direction<Dim> direction_from_neighbor =
          orientation(direction.opposite());
      const auto& neighbor_element = *(id_to_element.at(*neighbor_id));
      const auto& received_data =
          alg::find_if(
              ActionTesting::get_inbox_tag<comp, Inbox>(runner, element_id)
                  .messages.at(time_step_id),
              [&](const auto& entry) { return entry.first == directional_id; })
              ->second;
      CHECK(received_data.volume_mesh == dg_mesh);
      CHECK(received_data.validity_range == next_time_step_id);
      CHECK(received_data.tci_status == self_tci_decision);
      if (FixedActiveGrids and element_id == receiver.id() and
          *neighbor_id == sender.id()) {
        CHECK_FALSE(received_data.ghost_cell_data.has_value());
        CHECK_FALSE(received_data.volume_mesh_ghost_cell_data.has_value());
        continue;
      }
      CHECK(received_data.volume_mesh_ghost_cell_data == subcell_mesh);
      const DataVector& ghost_data = received_data.ghost_cell_data.value();
      const DataVector expected_ghost_data =
          compute_expected_ghost_data<Dim, UseNodegroupDgElements>(
              runner, neighbor_element, direction_from_neighbor, subcell_mesh,
              enable_extension, use_cell_centered_flux, evolved_vars,
              rdmp_tci_data, time_step_id, ghost_zone_size);
      CHECK_ITERABLE_APPROX(expected_ghost_data, ghost_data);
    }
  }

  // ReceiveDataForReconstruction (not by the receiver treated as using DG,
  // which would discard the data)
  for (const auto* element : elements) {
    const auto& element_id = element->id();
    if (FixedActiveGrids and element_id == receiver.id()) {
      continue;
    }
    ActionTesting::next_action<comp>(make_not_null(&runner), element_id);
  }
}

// Test that IsAuxiliary template parameter routes data to the correct inbox:
// IsAuxiliary=true  -> BoundaryCorrectionAndGhostCellsInbox<..., true>
// IsAuxiliary=false -> BoundaryCorrectionAndGhostCellsInbox<..., false>
template <size_t Dim, bool UseNodegroupDgElements, bool IsAuxiliary>
void test_inbox_routing() {
  CAPTURE(Dim);
  CAPTURE(IsAuxiliary);
  using Interps = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;
  using ExtensionDirs =
      DirectionMap<Dim, interpolators_detail::ExtensionDirection<Dim>>;
  using SubcellOptions = evolution::dg::subcell::SubcellOptions;

  using metavars =
      Metavariables<Dim, UseNodegroupDgElements, false, IsAuxiliary>;
  metavars::ghost_zone_size_invoked = false;
  metavars::ghost_data_mutator_invoked = false;
  using comp =
      component<Dim, metavars, UseNodegroupDgElements, false, IsAuxiliary>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{}};

  const TimeStepId time_step_id{true, 1, Time{Slab{1.0, 2.0}, {0, 10}}};
  const TimeStepId next_time_step_id{true, 1, Time{Slab{1.0, 2.0}, {1, 10}}};
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const evolution::dg::subcell::ActiveGrid active_grid =
      evolution::dg::subcell::ActiveGrid::Subcell;

  // 1D: one element with one neighbor in +xi
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  ElementId<Dim> self_id{};
  ElementId<Dim> east_id{};

  if constexpr (Dim == 1) {
    self_id = ElementId<Dim>{0, {{{1, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}}}};
    neighbors[Direction<Dim>::upper_xi()] =
        Neighbors<Dim>{{east_id}, OrientationMap<Dim>::create_aligned()};
  } else {
    // Only test 1D for this auxiliary inbox check
    static_assert(Dim == 1, "test_inbox_routing only implemented for 1D");
  }

  const Element<Dim> element{self_id, neighbors};

  using NeighborDataMap =
      DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>;
  NeighborDataMap neighbor_data{};

  using evolved_vars_tags = tmpl::list<Var1>;
  Variables<evolved_vars_tags> evolved_vars{
      subcell_mesh.number_of_grid_points()};
  get(get<Var1>(evolved_vars)) = get<0>(logical_coordinates(subcell_mesh));
  TimeSteppers::History<Variables<evolved_vars_tags>> time_stepper_history{};

  using CellCenteredFluxTag =
      evolution::dg::subcell::Tags::CellCenteredFlux<tmpl::list<Var1>, Dim>;
  typename CellCenteredFluxTag::type cell_centered_flux{};

  using MortarInfo = typename evolution::dg::Tags::MortarInfo<Dim>::type;
  using MortarData = typename evolution::dg::Tags::MortarData<Dim>::type;
  using MortarMesh = typename evolution::dg::Tags::MortarMesh<Dim>::type;
  using MortarNextId =
      typename evolution::dg::Tags::MortarNextTemporalId<Dim>::type;
  const DirectionalId<Dim> east_neighbor_id{Direction<Dim>::upper_xi(),
                                            east_id};
  MortarInfo mortar_info{};
  MortarData mortar_data{};
  MortarMesh mortar_mesh{};
  MortarNextId mortar_next_id{};
  mortar_info[east_neighbor_id] = evolution::dg::MortarInfo<Dim>{
      {.time_stepping_policy = evolution::dg::TimeSteppingPolicy::EqualRate}};
  mortar_data[east_neighbor_id] = {};
  mortar_mesh[east_neighbor_id] = {dg_mesh.slice_away(0)};
  mortar_next_id[east_neighbor_id] = {};

  typename evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>::type
      neighbor_decision{};
  neighbor_decision.insert(
      std::pair{east_neighbor_id, 0});
  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
      east_id,
      {std::make_unique<DummyReconstructor>(),
       time_step_id, next_time_step_id, Mesh<Dim>{}, Mesh<Dim>{},
       active_grid, Element<Dim>{}, NeighborDataMap{},
       evolution::dg::subcell::RdmpTciData{}, 0,
       typename evolution::dg::subcell::Tags::NeighborTciDecisions<
           Dim>::type{},
       Variables<evolved_vars_tags>{}, time_stepper_history,
       MortarInfo{}, MortarMesh{}, MortarData{}, MortarNextId{},
       typename domain::Tags::NeighborMesh<Dim>::type{},
       typename evolution::dg::subcell::Tags::MeshForGhostData<Dim>::type{},
       cell_centered_flux, Interps{}, Interps{}, ExtensionDirs{},
       SubcellOptions{}});

  Interps fd_to_neighbor_fd_interpolants{};
  Interps neighbor_dg_to_fd_interpolants{};
  ExtensionDirs extension_directions{};
  SubcellOptions subcell_options(
      4.0, 1, 2.0e-3, 2.0e-4, false, false,
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
      std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1);
  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
      self_id,
      {std::make_unique<DummyReconstructor>(), time_step_id, next_time_step_id,
       dg_mesh, subcell_mesh, active_grid, element, neighbor_data,
       evolution::dg::subcell::RdmpTciData{{max(get(get<Var1>(evolved_vars)))},
                                           {min(get(get<Var1>(evolved_vars)))}},
       100, neighbor_decision, evolved_vars, time_stepper_history,
       mortar_info, mortar_mesh, mortar_data, mortar_next_id,
       typename domain::Tags::NeighborMesh<Dim>::type{},
       typename evolution::dg::subcell::Tags::MeshForGhostData<Dim>::type{},
       cell_centered_flux, fd_to_neighbor_fd_interpolants,
       neighbor_dg_to_fd_interpolants, extension_directions, subcell_options});

  // Run SendDataForReconstruction with IsAuxiliary template parameter
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  // Data should be in the inbox matching IsAuxiliary
  const auto& inbox =
      ActionTesting::get_inbox_tag<
          comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                    Dim, UseNodegroupDgElements, IsAuxiliary>>(runner, east_id)
          .messages;
  CHECK(inbox.count(time_step_id) == 1);
  CHECK(inbox.at(time_step_id).size() == 1);
}

// The tensors of the ghost data of the two-exchange tests below: `Var1` and
// `Var2`, which the neighbors using finite differences use, followed by
// `Gradient`, which changes between the two exchanges (as the gradients of the
// CCZ4 gradient-projection scheme).
struct Var2 : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};

struct Gradient : db::SimpleTag {
  using type = tnsr::i<DataVector, 3>;
};

using TwoExchangeGhostVars = Variables<tmpl::list<Var1, Var2, Gradient>>;
constexpr size_t number_of_used_ghost_components =
    Variables<tmpl::list<Var1, Var2>>::number_of_independent_components;

struct SystemWithoutFluxes {
  using variables_tag = ::Tags::Variables<tmpl::list<Var1>>;
  using flux_variables = tmpl::list<>;
};

// Whether the ghost-data mutator of the first ([0]) and second ([1]) exchange
// was called.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::array<bool, 2> two_exchange_mutator_called{};

// The ghost data from `Var1`. Reusing the auxiliary ghost data requires zero
// gradients in the first exchange, but they are not zero here, so that the
// tests show that elements reusing the auxiliary ghost data neither slice nor
// interpolate them but send zero.
template <bool IsAuxiliary>
struct TwoExchangeGhostDataMutator {
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::Variables<tmpl::list<Var1>>>;
  static DataVector apply(const Variables<tmpl::list<Var1>>& vars,
                          const size_t cell_centered_flux_size) {
    CHECK(cell_centered_flux_size == 0);
    gsl::at(two_exchange_mutator_called, IsAuxiliary ? 0 : 1) = true;
    DataVector buffer{vars.number_of_grid_points() *
                      TwoExchangeGhostVars::number_of_independent_components};
    TwoExchangeGhostVars ghost_vars{buffer.data(), buffer.size()};
    const DataVector& var1 = get(get<Var1>(vars));
    get(get<Var1>(ghost_vars)) = 2.0 * var1;
    for (size_t i = 0; i < 3; ++i) {
      const auto offset = static_cast<double>(i);
      get<Var2>(ghost_vars).get(i) = (offset + 2.0) * var1 + offset;
      if constexpr (IsAuxiliary) {
        get<Gradient>(ghost_vars).get(i) = 1.0e3 + offset;
      } else {
        get<Gradient>(ghost_vars).get(i) = (offset + 5.0) * var1 * var1 + 0.5;
      }
    }
    return buffer;
  }
};

using two_exchange_initial_tags = tmpl::list<
    Tags::Reconstructor, ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
    domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>,
    evolution::dg::subcell::Tags::ActiveGrid, domain::Tags::Element<3>,
    evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
    evolution::dg::subcell::Tags::DataForRdmpTci,
    evolution::dg::subcell::Tags::TciDecision,
    evolution::dg::subcell::Tags::NeighborTciDecisions<3>,
    ::Tags::Variables<tmpl::list<Var1>>,
    ::Tags::HistoryEvolvedVariables<::Tags::Variables<tmpl::list<Var1>>>,
    evolution::dg::Tags::MortarInfo<3>, evolution::dg::Tags::MortarMesh<3>,
    evolution::dg::Tags::MortarData<3>,
    evolution::dg::Tags::MortarNextTemporalId<3>, domain::Tags::NeighborMesh<3>,
    evolution::dg::subcell::Tags::MeshForGhostData<3>,
    evolution::dg::subcell::Tags::CellCenteredFlux<tmpl::list<>, 3>,
    evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<3>,
    evolution::dg::subcell::Tags::InterpolatorsFromNeighborDgToFd<3>,
    evolution::dg::subcell::Tags::InterpolatorsFromDgToNeighborFd<3>,
    ::domain::Tags::ElementMap<3, Frame::Grid>,
    evolution::dg::subcell::Tags::NeighborActiveGrids<3>,
    evolution::dg::subcell::Tags::ExtensionDirections<3>,
    evolution::dg::subcell::Tags::SubcellOptions<3>>;

template <bool IsAuxiliary>
using exchange_actions = tmpl::list<
    evolution::dg::subcell::Actions::SendDataForReconstruction<
        3, TwoExchangeGhostDataMutator<IsAuxiliary>, false, IsAuxiliary>,
    evolution::dg::subcell::Actions::ReceiveAndSendDataForReconstruction<
        3, TwoExchangeGhostDataMutator<IsAuxiliary>, false, IsAuxiliary>,
    evolution::dg::subcell::Actions::ReceiveDataForReconstruction<3,
                                                                  IsAuxiliary>>;

// The two exchanges of an element using finite differences, as in
// `Ccz4::GradientProjection::subcell_step_actions` without the actions
// between and after them.
using two_exchange_actions = tmpl::flatten<
    tmpl::list<ActionTesting::InitializeDataBox<two_exchange_initial_tags>,
               Actions::MutateApply<evolution::dg::subcell::SetInterpolators<
                   3, Tags::Reconstructor>>,
               exchange_actions<true>, exchange_actions<false>>>;

template <typename Action>
constexpr size_t two_exchange_index =
    tmpl::index_of<two_exchange_actions, Action>::value;

using FirstSend = tmpl::front<exchange_actions<true>>;
using FirstReceive = tmpl::back<exchange_actions<true>>;
using SecondSend = tmpl::front<exchange_actions<false>>;
using SecondReceive = tmpl::back<exchange_actions<false>>;
// After its last action, an element is at the first action of the list.
using TwoExchangesDone = tmpl::front<two_exchange_actions>;

template <typename Metavariables>
struct two_exchange_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        two_exchange_actions>>;
};

// With fixed active grids and, if `ReuseAuxiliaryGhostData`, reused auxiliary
// finite-difference ghost data.
template <bool ReuseAuxiliaryGhostData>
struct TwoExchangeMetavariables {
  static constexpr size_t volume_dim = 3;
  using component_list =
      tmpl::list<two_exchange_component<TwoExchangeMetavariables>>;
  using system = SystemWithoutFluxes;
  using const_global_cache_tags = tmpl::list<::domain::Tags::Domain<3>>;

  struct SubcellOptions {
    static constexpr bool active_grids_fixed_at_initialization = true;
    static constexpr bool reuse_auxiliary_finite_difference_ghost_data =
        ReuseAuxiliaryGhostData;
    using finite_difference_neighbor_ghost_tags = tmpl::list<Var1, Var2>;

    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) {
      return db::get<Tags::Reconstructor>(box).ghost_zone_size();
    }
  };
};

using GhostDataMap = DirectionalIdMap<3, evolution::dg::subcell::GhostData>;
using InboxMessages = evolution::dg::InboxBoundaryData<3>::mapped_type;
// The messages of the neighbor using DG: receiver, key at the receiver, data.
using DgMessages = std::vector<
    std::tuple<ElementId<3>, DirectionalId<3>, evolution::dg::BoundaryData<3>>>;

bool bitwise_equal(const DataVector& lhs, const DataVector& rhs) {
  return lhs.size() == rhs.size() and
         (lhs.size() == 0 or std::memcmp(lhs.data(), rhs.data(),
                                         lhs.size() * sizeof(double)) == 0);
}

void check_bitwise_equal(const GhostDataMap& lhs, const GhostDataMap& rhs) {
  REQUIRE(lhs.size() == rhs.size());
  for (const auto& [key, entry] : lhs) {
    CAPTURE(key);
    REQUIRE(rhs.contains(key));
    CHECK(bitwise_equal(entry.neighbor_ghost_data_for_reconstruction(),
                        rhs.at(key).neighbor_ghost_data_for_reconstruction()));
  }
}

void check_bitwise_equal(const InboxMessages& lhs, const InboxMessages& rhs) {
  REQUIRE(lhs.size() == rhs.size());
  for (const auto& [key, data] : lhs) {
    CAPTURE(key);
    const auto other = alg::find_if(
        rhs, [&key](const auto& message) { return message.first == key; });
    REQUIRE(other != rhs.end());
    CHECK(data == other->second);
    CHECK(data.ghost_cell_data.has_value() ==
          other->second.ghost_cell_data.has_value());
    if (data.ghost_cell_data.has_value()) {
      CHECK(
          bitwise_equal(*data.ghost_cell_data, *other->second.ghost_cell_data));
    }
  }
}

// Checks that the ghost data `data` (of `number_of_ghost_points` points,
// possibly followed by RDMP data) of a neighbor using finite differences hold
// the used components of `legacy_data` and zero gradients instead of its
// gradients.
void check_used_components(const DataVector& data,
                           const DataVector& legacy_data,
                           const size_t number_of_ghost_points) {
  const size_t used_size =
      number_of_used_ghost_components * number_of_ghost_points;
  const size_t ghost_size =
      TwoExchangeGhostVars::number_of_independent_components *
      number_of_ghost_points;
  REQUIRE(legacy_data.size() == data.size());
  REQUIRE(data.size() >= ghost_size);
  CHECK(std::memcmp(data.data(), legacy_data.data(),
                    used_size * sizeof(double)) == 0);
  bool gradients_differ = false;
  for (size_t i = used_size; i < ghost_size; ++i) {
    CHECK((data[i] == 0.0 and not std::signbit(data[i])));
    gradients_differ = gradients_differ or legacy_data[i] != 0.0;
  }
  CHECK(gradients_differ);
  CHECK(std::memcmp(data.data() + ghost_size, legacy_data.data() + ghost_size,
                    (data.size() - ghost_size) * sizeof(double)) == 0);
}

// What a round of the two exchanges recorded.
struct TwoExchangeRecord {
  // Whether the ghost-data mutator of each exchange was called
  std::unordered_map<ElementId<3>, std::array<bool, 2>> mutator_called{};
  // The messages in the inboxes: in the first exchange after the sends, the
  // messages of the neighbor using DG and the forwards; in the second exchange
  // after the sends
  std::array<std::unordered_map<ElementId<3>, InboxMessages>, 2> messages{};
  std::unordered_map<ElementId<3>, GhostDataMap> ghost_data_after_first{};
  std::unordered_map<ElementId<3>, evolution::dg::subcell::RdmpTciData>
      rdmp_after_first{};
  std::unordered_map<ElementId<3>, GhostDataMap> ghost_data_after_second_send{};
  // The elements that completed the second exchange before the messages of
  // the neighbor using DG of the second exchange
  std::unordered_set<ElementId<3>> done_before_dg_messages{};
};

template <bool IsAuxiliary, typename Metavars>
void insert_dg_messages(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavars>*> runner,
    const DgMessages& dg_messages, const TimeStepId& time_step_id) {
  using Inbox =
      evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<3, false,
                                                                IsAuxiliary>;
  for (const auto& [receiver, key, data] : dg_messages) {
    Inbox::insert_into_inbox(
        make_not_null(
            &ActionTesting::get_inbox_tag<two_exchange_component<Metavars>,
                                          Inbox>(runner, receiver)),
        time_step_id, std::pair{key, data});
  }
}

template <bool IsAuxiliary, typename Metavars>
std::unordered_map<ElementId<3>, InboxMessages> inbox_messages(
    const ActionTesting::MockRuntimeSystem<Metavars>& runner,
    const std::vector<ElementId<3>>& ids, const TimeStepId& time_step_id) {
  std::unordered_map<ElementId<3>, InboxMessages> result{};
  for (const auto& id : ids) {
    const auto& messages =
        ActionTesting::get_inbox_tag<
            two_exchange_component<Metavars>,
            evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                3, false, IsAuxiliary>>(runner, id)
            .messages;
    if (const auto found = messages.find(time_step_id);
        found != messages.end()) {
      result.emplace(id, found->second);
    }
  }
  return result;
}

// Runs each element while it is ready and its next action is not `Stop`.
template <typename Stop, typename Metavars>
void run_while_ready(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavars>*> runner,
    const std::vector<ElementId<3>>& ids) {
  using component = two_exchange_component<Metavars>;
  for (const auto& id : ids) {
    while (ActionTesting::get_next_action_index<component>(*runner, id) !=
               two_exchange_index<Stop> and
           ActionTesting::next_action_if_ready<component>(runner, id)) {
    }
  }
}

// Runs the two exchanges on the elements using finite differences `fd_ids`,
// starting from the first send, with the messages `dg_messages` of the neighbor
// `dg_id` using DG. Before the second exchange, an external-boundary entry is
// added to the ghost data of each element (as by
// `Ccz4::fd::Actions::SetBoundaryConditionGhostData`).
template <typename Metavars>
TwoExchangeRecord run_two_exchanges(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavars>*> runner,
    const std::vector<ElementId<3>>& fd_ids, const ElementId<3>& dg_id,
    const std::array<DgMessages, 2>& dg_messages,
    const TimeStepId& time_step_id) {
  using component = two_exchange_component<Metavars>;
  using ghost_data_tag =
      evolution::dg::subcell::Tags::GhostDataForReconstruction<3>;
  std::vector<ElementId<3>> all_ids = fd_ids;
  all_ids.push_back(dg_id);
  TwoExchangeRecord record{};

  // First exchange
  for (const auto& id : fd_ids) {
    REQUIRE(ActionTesting::get_next_action_index<component>(*runner, id) ==
            two_exchange_index<FirstSend>);
    two_exchange_mutator_called = {};
    REQUIRE(ActionTesting::next_action_if_ready<component>(runner, id));
    record.mutator_called[id][0] = two_exchange_mutator_called[0];
  }
  insert_dg_messages<true>(runner, dg_messages[0], time_step_id);
  run_while_ready<FirstReceive>(runner, fd_ids);
  record.messages[0] = inbox_messages<true>(*runner, all_ids, time_step_id);
  run_while_ready<SecondSend>(runner, fd_ids);
  for (const auto& id : fd_ids) {
    REQUIRE(ActionTesting::get_next_action_index<component>(*runner, id) ==
            two_exchange_index<SecondSend>);
    record.ghost_data_after_first[id] =
        ActionTesting::get_databox_tag<component, ghost_data_tag>(*runner, id);
    record.rdmp_after_first[id] = ActionTesting::get_databox_tag<
        component, evolution::dg::subcell::Tags::DataForRdmpTci>(*runner, id);
    db::mutate<ghost_data_tag>(
        [](const auto ghost_data) {
          evolution::dg::subcell::GhostData external{1};
          external.neighbor_ghost_data_for_reconstruction() =
              DataVector{3, -7.0};
          ghost_data->emplace(
              DirectionalId<3>{Direction<3>::lower_zeta(),
                               ElementId<3>::external_boundary_id()},
              std::move(external));
        },
        make_not_null(&ActionTesting::get_databox<component>(runner, id)));
  }

  // Second exchange
  for (const auto& id : fd_ids) {
    two_exchange_mutator_called = {};
    REQUIRE(ActionTesting::next_action_if_ready<component>(runner, id));
    record.mutator_called[id][1] = two_exchange_mutator_called[1];
    record.ghost_data_after_second_send[id] =
        ActionTesting::get_databox_tag<component, ghost_data_tag>(*runner, id);
  }
  record.messages[1] = inbox_messages<false>(*runner, all_ids, time_step_id);
  run_while_ready<TwoExchangesDone>(runner, fd_ids);
  for (const auto& id : fd_ids) {
    if (ActionTesting::get_next_action_index<component>(*runner, id) ==
        two_exchange_index<TwoExchangesDone>) {
      record.done_before_dg_messages.insert(id);
    }
  }
  insert_dg_messages<false>(runner, dg_messages[1], time_step_id);
  run_while_ready<TwoExchangesDone>(runner, fd_ids);
  for (const auto& id : fd_ids) {
    REQUIRE(ActionTesting::get_next_action_index<component>(*runner, id) ==
            two_exchange_index<TwoExchangesDone>);
  }
  return record;
}

// The messages of the neighbor `dg_element` using DG to its neighbors in the
// two exchanges, with the RDMP data `rdmp_data` of each exchange. The ghost
// data of the two exchanges differ, as the DG elements of the CCZ4
// gradient-projection scheme filter between them.
std::array<DgMessages, 2> make_dg_messages(
    const Element<3>& dg_element, const Mesh<3>& dg_mesh,
    const TimeStepId& next_time_step_id, const double offset,
    const std::array<std::array<double, 2>, 2>& rdmp_data) {
  std::array<DgMessages, 2> result{};
  for (size_t exchange = 0; exchange < 2; ++exchange) {
    DataVector ghost_data{
        dg_mesh.number_of_grid_points() *
            TwoExchangeGhostVars::number_of_independent_components +
        2};
    for (size_t i = 0; i < ghost_data.size() - 2; ++i) {
      ghost_data[i] = offset + 0.25 * static_cast<double>(exchange) +
                      1.0e-3 * static_cast<double>(i);
    }
    ghost_data[ghost_data.size() - 2] = gsl::at(rdmp_data, exchange)[0];
    ghost_data[ghost_data.size() - 1] = gsl::at(rdmp_data, exchange)[1];
    for (const auto& [direction, neighbors] : dg_element.neighbors()) {
      for (const auto& neighbor_id : neighbors) {
        const auto direction_from_neighbor =
            neighbors.orientation(neighbor_id)(direction.opposite());
        const Mesh<2> face_mesh =
            dg_mesh.slice_away(direction_from_neighbor.dimension());
        DataVector boundary_correction_data{face_mesh.number_of_grid_points()};
        alg::iota(boundary_correction_data, 1000.0 + offset);
        gsl::at(result, exchange)
            .emplace_back(
                neighbor_id,
                DirectionalId<3>{direction_from_neighbor, dg_element.id()},
                evolution::dg::BoundaryData<3>{
                    dg_mesh, dg_mesh,
                    exchange == 0 ? std::optional<Mesh<2>>{} : face_mesh,
                    ghost_data,
                    exchange == 0 ? std::optional<DataVector>{}
                                  : boundary_correction_data,
                    next_time_step_id, -20});
      }
    }
  }
  return result;
}

// Two rounds (at the same time step id, with different states) of the two
// exchanges of elements using finite differences with fixed active grids, with
// reused auxiliary ghost data compared to without. The elements, on the Sphere
// domain of `test_receive_and_send_data`, use finite differences except one
// (D) using DG:
// - the sender of `test_receive_and_send_data` (S), with neighbors using
//   finite differences in its block (sliced ghost data) and in another block
//   with a different orientation (interpolated ghost data), and D;
// - the other elements of `test_receive_and_send_data`, which with extension
//   directions forward ghost data;
// - an element (F) whose only neighbor is D, and an element without neighbors.
// Without extension directions, with reused auxiliary ghost data:
// - in the first exchange the elements using finite differences send each
//   other the used components on all ghost points (sliced in the block,
//   interpolated across blocks) and zero gradients, followed by the RDMP data,
//   and their other messages are unchanged;
// - in the second exchange the elements using finite differences send each
//   other nothing and do not wait for each other, keep the ghost data and
//   metadata of their neighbors using finite differences from the first
//   exchange, replace the ghost data of D and of the external boundary, and
//   send byte-identical messages to D;
// - only the elements with a neighbor using DG compute the ghost data of the
//   second exchange.
// With extension directions, opting into reuse must fail explicitly.
void test_reuse_auxiliary_ghost_data(const bool enable_extension) {
  CAPTURE(enable_extension);
  domain::creators::register_derived_with_charm();
  using Sphere = domain::creators::Sphere;
  using Interior = std::variant<domain::creators::Sphere::Excision,
                                domain::creators::Sphere::InnerCube>;
  const Sphere domain_creator{1.,
                              5.,
                              Interior{Sphere::Excision{}},
                              2_st,
                              6_st,
                              true,
                              std::nullopt,
                              {},
                              domain::CoordinateMaps::Distribution::Linear,
                              ShellWedges::All,
                              std::nullopt};
  const auto domain = domain_creator.create_domain();
  const std::vector<Block<3>>& blocks = domain.blocks();
  using LegacyMetavars = TwoExchangeMetavariables<false>;
  using ReuseMetavars = TwoExchangeMetavariables<true>;
  ActionTesting::MockRuntimeSystem<LegacyMetavars> legacy_runner{
      {domain_creator.create_domain()}};
  ActionTesting::MockRuntimeSystem<ReuseMetavars> reuse_runner{
      {domain_creator.create_domain()}};
  const bool reuses = not enable_extension;

  const ElementId<3> sender_id{0, {{{2, 0}, {2, 0}, {2, 0}}}};
  const ElementId<3> extender_id{0, {{{2, 0}, {2, 1}, {2, 0}}}};
  const ElementId<3> receiver_id{5, {{{2, 3}, {2, 3}, {2, 0}}}};
  const ElementId<3> receiver_extender_id{5, {{{2, 2}, {2, 3}, {2, 0}}}};
  const ElementId<3> dg_id{0, {{{2, 1}, {2, 0}, {2, 0}}}};
  const ElementId<3> next_to_dg_id{0, {{{2, 2}, {2, 0}, {2, 0}}}};
  const ElementId<3> isolated_id{1, {{{2, 0}, {2, 0}, {2, 0}}}};
  const std::vector<Element<3>> fd_elements{
      create_element<3>(blocks, sender_id,
                        {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
                         Direction<3>::upper_xi()},
                        {extender_id, receiver_id, dg_id}),
      create_element<3>(blocks, extender_id, {Direction<3>::lower_eta()},
                        {sender_id}),
      create_element<3>(blocks, receiver_id,
                        {Direction<3>::upper_eta(), Direction<3>::lower_xi()},
                        {sender_id, receiver_extender_id}),
      create_element<3>(blocks, receiver_extender_id,
                        {Direction<3>::upper_xi()}, {receiver_id}),
      create_element<3>(blocks, next_to_dg_id, {Direction<3>::lower_xi()},
                        {dg_id}),
      create_element<3>(blocks, isolated_id, {}, {})};
  const Element<3> dg_element = create_element<3>(
      blocks, dg_id, {Direction<3>::lower_xi(), Direction<3>::upper_xi()},
      {sender_id, next_to_dg_id});
  std::vector<ElementId<3>> fd_ids{};
  for (const auto& element : fd_elements) {
    fd_ids.push_back(element.id());
  }

  const evolution::dg::subcell::SubcellOptions subcell_options(
      4.0, 1, 2.0e-3, 2.0e-4, enable_extension, enable_extension,
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim, false,
      std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1);
  const Mesh<3> dg_mesh{6, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const TimeStepId time_step_id{true, 1, Time{Slab{1.0, 2.0}, {0, 10}}};
  const TimeStepId next_time_step_id{true, 1, Time{Slab{1.0, 2.0}, {1, 10}}};
  const size_t ghost_zone_size = DummyReconstructor::ghost_zone_size();
  const auto logical_coords = logical_coordinates(subcell_mesh);
  // The RDMP data of the elements, set by the TCI (here at each round).
  const auto initial_rdmp = [](const size_t index) {
    const auto offset = static_cast<double>(index);
    return evolution::dg::subcell::RdmpTciData{{1.0 + 0.1 * offset},
                                               {-1.0 - 0.1 * offset}};
  };
  const auto initial_vars = [&logical_coords](const size_t index) {
    Variables<tmpl::list<Var1>> vars{get<0>(logical_coords).size()};
    get(get<Var1>(vars)) =
        get<0>(logical_coords) + 2.0 * get<1>(logical_coords) +
        0.5 * get<2>(logical_coords) + 0.1 * static_cast<double>(index);
    return vars;
  };

  const auto initialize = [&]<typename Metavars>(
                              const gsl::not_null<
                                  ActionTesting::MockRuntimeSystem<Metavars>*>
                                  runner) {
    using component = two_exchange_component<Metavars>;
    using Interps = DirectionalIdMap<3, std::optional<intrp::Irregular<3>>>;
    for (size_t index = 0; index <= fd_elements.size(); ++index) {
      const bool uses_dg = index == fd_elements.size();
      const Element<3>& element = uses_dg ? dg_element : fd_elements[index];
      typename evolution::dg::subcell::Tags::NeighborTciDecisions<3>::type
          neighbor_decisions{};
      typename evolution::dg::Tags::MortarInfo<3>::type mortar_info{};
      typename evolution::dg::Tags::MortarData<3>::type mortar_data{};
      typename evolution::dg::Tags::MortarMesh<3>::type mortar_mesh{};
      typename evolution::dg::Tags::MortarNextTemporalId<3>::type
          mortar_next_id{};
      typename evolution::dg::subcell::Tags::NeighborActiveGrids<3>::type
          neighbor_active_grids{};
      for (const auto& [direction, neighbors] : element.neighbors()) {
        for (const auto& neighbor_id : neighbors) {
          const DirectionalId<3> mortar_id{direction, neighbor_id};
          neighbor_decisions.emplace(mortar_id, 0);
          mortar_info[mortar_id] = evolution::dg::MortarInfo<3>{
              {.time_stepping_policy =
                   evolution::dg::TimeSteppingPolicy::EqualRate}};
          mortar_data[mortar_id] = {};
          mortar_mesh[mortar_id] = dg_mesh.slice_away(direction.dimension());
          mortar_next_id[mortar_id] = {};
          neighbor_active_grids.emplace(
              mortar_id, neighbor_id == dg_id
                             ? evolution::dg::subcell::ActiveGrid::Dg
                             : evolution::dg::subcell::ActiveGrid::Subcell);
        }
      }
      ActionTesting::emplace_array_component_and_initialize<component>(
          runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
          element.id(),
          {std::make_unique<DummyReconstructor>(),
           time_step_id,
           next_time_step_id,
           dg_mesh,
           subcell_mesh,
           uses_dg ? evolution::dg::subcell::ActiveGrid::Dg
                   : evolution::dg::subcell::ActiveGrid::Subcell,
           element,
           GhostDataMap{},
           initial_rdmp(index),
           100 + static_cast<int>(index),
           neighbor_decisions,
           initial_vars(index),
           TimeSteppers::History<Variables<tmpl::list<Var1>>>{},
           mortar_info,
           mortar_mesh,
           mortar_data,
           mortar_next_id,
           typename domain::Tags::NeighborMesh<3>::type{},
           typename evolution::dg::subcell::Tags::MeshForGhostData<3>::type{},
           typename evolution::dg::subcell::Tags::CellCenteredFlux<tmpl::list<>,
                                                                   3>::type{},
           Interps{},
           Interps{},
           Interps{},
           ElementMap<3, Frame::Grid>{element.id(),
                                      blocks[element.id().block_id()]},
           neighbor_active_grids,
           DirectionMap<3, interpolators_detail::ExtensionDirection<3>>{},
           subcell_options});
      if (not uses_dg) {
        ActionTesting::next_action<component>(runner, element.id());
      }
    }
  };
  initialize(make_not_null(&legacy_runner));
  initialize(make_not_null(&reuse_runner));
  if (enable_extension) {
    // ERROR must reject the unsupported combination in both Debug and Release,
    // before sending messages or changing the reconstruction buffers.
    using component = two_exchange_component<ReuseMetavars>;
    CHECK_FALSE(ActionTesting::next_action_if_ready<component>(
        make_not_null(&reuse_runner), sender_id,
        Catch::Matchers::ContainsSubstring(
            "Reusing auxiliary finite-difference ghost data is incompatible "
            "with EnableExtensionDirections")));
    return;
  }
  CHECK(ActionTesting::get_databox_tag<
            two_exchange_component<ReuseMetavars>,
            evolution::dg::subcell::Tags::ExtensionDirections<3>>(reuse_runner,
                                                                  sender_id)
            .empty() == not enable_extension);

  const auto neighbor_uses_dg = [&dg_id](const DirectionalId<3>& key) {
    return key.id() == dg_id;
  };
  const auto is_fd_neighbor = [&dg_id](const DirectionalId<3>& key) {
    return key.id() != dg_id and
           key.id() != ElementId<3>::external_boundary_id();
  };
  // The number of ghost points of the neighbor `key`
  const auto number_of_ghost_points =
      [&ghost_zone_size, &subcell_mesh](const DirectionalId<3>& key) {
        return ghost_zone_size * subcell_mesh.extents()
                                     .slice_away(key.direction().dimension())
                                     .product();
      };
  std::optional<std::unordered_map<ElementId<3>, GhostDataMap>>
      previous_ghost_data{};
  for (size_t round = 0; round < 2; ++round) {
    CAPTURE(round);
    // Second round: a new state at the same time step id.
    if (round == 1) {
      const auto restart =
          [&dg_id, &fd_ids, &initial_rdmp]<typename Metavars>(
              const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavars>*>
                  runner) {
            using component = two_exchange_component<Metavars>;
            for (size_t index = 0; index < fd_ids.size(); ++index) {
              db::mutate<::Tags::Variables<tmpl::list<Var1>>,
                         evolution::dg::subcell::Tags::DataForRdmpTci>(
                  [&initial_rdmp, &index](const auto vars, const auto rdmp) {
                    get(get<Var1>(*vars)) = 1.5 * get(get<Var1>(*vars)) + 0.3;
                    *rdmp = initial_rdmp(index);
                  },
                  make_not_null(&ActionTesting::get_databox<component>(
                      runner, fd_ids[index])));
              runner->template force_next_action_to_be<component, FirstSend>(
                  fd_ids[index]);
            }
            ActionTesting::get_inbox_tag<
                component,
                evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                    3, false, true>>(runner, dg_id)
                .messages.clear();
            ActionTesting::get_inbox_tag<
                component,
                evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                    3, false, false>>(runner, dg_id)
                .messages.clear();
          };
      restart(make_not_null(&legacy_runner));
      restart(make_not_null(&reuse_runner));
    }
    const std::array<std::array<double, 2>, 2> dg_rdmp_data{
        {{{0.5, -0.5}}, {{9.0 + static_cast<double>(round), -0.25}}}};
    const auto dg_messages =
        make_dg_messages(dg_element, dg_mesh, next_time_step_id,
                         static_cast<double>(round), dg_rdmp_data);
    const TwoExchangeRecord legacy =
        run_two_exchanges(make_not_null(&legacy_runner), fd_ids, dg_id,
                          dg_messages, time_step_id);
    const TwoExchangeRecord reuse = run_two_exchanges(
        make_not_null(&reuse_runner), fd_ids, dg_id, dg_messages, time_step_id);

    for (const auto& element : fd_elements) {
      const ElementId<3>& id = element.id();
      CAPTURE(id);
      const bool has_fd_neighbor = alg::any_of(
          element.neighbors(), [&dg_id](const auto& direction_and_neighbors) {
            return not direction_and_neighbors.second.ids().contains(dg_id);
          });
      const bool has_dg_neighbor = alg::any_of(
          element.neighbors(), [&dg_id](const auto& direction_and_neighbors) {
            return direction_and_neighbors.second.ids().contains(dg_id);
          });
      // With the active grids fixed at initialization, the ghost data of the
      // first exchange are computed only by the elements with a neighbor using
      // finite differences; with reused auxiliary ghost data, those of the
      // second exchange only by the elements with a neighbor using DG.
      CHECK(legacy.mutator_called.at(id) ==
            std::array<bool, 2>{{has_fd_neighbor, true}});
      CHECK(reuse.mutator_called.at(id) ==
            std::array<bool, 2>{
                {has_fd_neighbor, reuses ? has_dg_neighbor : true}});

      // The first exchange
      REQUIRE(reuse.ghost_data_after_first.at(id).size() ==
              legacy.ghost_data_after_first.at(id).size());
      for (const auto& [key, entry] : reuse.ghost_data_after_first.at(id)) {
        CAPTURE(key);
        const DataVector& data = entry.neighbor_ghost_data_for_reconstruction();
        const DataVector& legacy_data =
            legacy.ghost_data_after_first.at(id)
                .at(key)
                .neighbor_ghost_data_for_reconstruction();
        if (reuses and is_fd_neighbor(key)) {
          check_used_components(data, legacy_data, number_of_ghost_points(key));
        } else {
          CHECK(bitwise_equal(data, legacy_data));
        }
      }
      CHECK(reuse.rdmp_after_first.at(id) == legacy.rdmp_after_first.at(id));

      // The second send keeps (only) the ghost data of the neighbors using
      // finite differences.
      CHECK(legacy.ghost_data_after_second_send.at(id).empty());
      GhostDataMap expected_kept{};
      if (reuses) {
        for (const auto& [key, entry] : reuse.ghost_data_after_first.at(id)) {
          if (is_fd_neighbor(key)) {
            expected_kept.emplace(key, entry);
          }
        }
      }
      check_bitwise_equal(reuse.ghost_data_after_second_send.at(id),
                          expected_kept);

      // The final ghost data and metadata: in the second exchange, the ghost
      // data of the neighbors using finite differences are those of the first
      // exchange (with zero gradients) instead of the second.
      const auto get_tags = [&id]<typename Metavars>(
                                const ActionTesting::MockRuntimeSystem<
                                    Metavars>& runner) {
        using component = two_exchange_component<Metavars>;
        return std::tuple{
            ActionTesting::get_databox_tag<
                component,
                evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
                runner, id),
            ActionTesting::get_databox_tag<
                component, evolution::dg::subcell::Tags::DataForRdmpTci>(runner,
                                                                         id),
            ActionTesting::get_databox_tag<component,
                                           domain::Tags::NeighborMesh<3>>(
                runner, id),
            ActionTesting::get_databox_tag<
                component, evolution::dg::subcell::Tags::MeshForGhostData<3>>(
                runner, id),
            ActionTesting::get_databox_tag<
                component, evolution::dg::Tags::MortarNextTemporalId<3>>(runner,
                                                                         id),
            ActionTesting::get_databox_tag<
                component,
                evolution::dg::subcell::Tags::NeighborTciDecisions<3>>(runner,
                                                                       id),
            ActionTesting::get_databox_tag<component,
                                           evolution::dg::Tags::MortarData<3>>(
                runner, id)};
      };
      const auto [legacy_ghost_data, legacy_rdmp, legacy_neighbor_meshes,
                  legacy_ghost_meshes, legacy_next_ids, legacy_decisions,
                  legacy_mortar_data] = get_tags(legacy_runner);
      const auto [ghost_data, rdmp, neighbor_meshes, ghost_meshes, next_ids,
                  decisions, mortar_data] = get_tags(reuse_runner);
      CHECK(neighbor_meshes == legacy_neighbor_meshes);
      CHECK(ghost_meshes == legacy_ghost_meshes);
      CHECK(next_ids == legacy_next_ids);
      CHECK(decisions == legacy_decisions);
      CHECK(mortar_data == legacy_mortar_data);
      CHECK(neighbor_meshes.size() == element.number_of_neighbors());
      REQUIRE(ghost_data.size() == element.number_of_neighbors());
      REQUIRE(legacy_ghost_data.size() == element.number_of_neighbors());
      for (const auto& [key, entry] : ghost_data) {
        CAPTURE(key);
        REQUIRE(legacy_ghost_data.contains(key));
        const DataVector& data = entry.neighbor_ghost_data_for_reconstruction();
        const DataVector& legacy_data =
            legacy_ghost_data.at(key).neighbor_ghost_data_for_reconstruction();
        if (neighbor_uses_dg(key)) {
          // The data of the second exchange
          CHECK(bitwise_equal(data, legacy_data));
          CHECK_FALSE(bitwise_equal(
              data, reuse.ghost_data_after_first.at(id)
                        .at(key)
                        .neighbor_ghost_data_for_reconstruction()));
        } else if (not reuses) {
          CHECK(bitwise_equal(data, legacy_data));
        } else {
          // The ghost data of the first exchange, whose used components are
          // those of the second exchange
          CHECK(bitwise_equal(data,
                              reuse.ghost_data_after_first.at(id)
                                  .at(key)
                                  .neighbor_ghost_data_for_reconstruction()));
          REQUIRE(data.size() ==
                  TwoExchangeGhostVars::number_of_independent_components *
                      number_of_ghost_points(key));
          check_used_components(data, legacy_data, number_of_ghost_points(key));
        }
      }
      if (reuses) {
        // The RDMP data of the neighbors using finite differences are those
        // of the first exchange.
        evolution::dg::subcell::RdmpTciData expected_rdmp =
            reuse.rdmp_after_first.at(id);
        if (has_dg_neighbor) {
          expected_rdmp.max_variables_values[0] = std::max(
              expected_rdmp.max_variables_values[0], dg_rdmp_data[1][0]);
          expected_rdmp.min_variables_values[0] = std::min(
              expected_rdmp.min_variables_values[0], dg_rdmp_data[1][1]);
        }
        CHECK(rdmp == expected_rdmp);
      } else {
        CHECK(rdmp == legacy_rdmp);
      }
      // The first exchange of the second round replaced the kept ghost data.
      if (previous_ghost_data.has_value() and has_fd_neighbor) {
        for (const auto& [key, entry] : ghost_data) {
          CAPTURE(key);
          if (is_fd_neighbor(key)) {
            CHECK_FALSE(
                bitwise_equal(entry.neighbor_ghost_data_for_reconstruction(),
                              previous_ghost_data->at(id)
                                  .at(key)
                                  .neighbor_ghost_data_for_reconstruction()));
          }
        }
      }
    }

    // The messages of the first exchange: from the elements using finite
    // differences to each other (with reused auxiliary ghost data, the used
    // components and zero gradients, followed by the RDMP data), from them to
    // D (no ghost data), and from D. In the second exchange the elements using
    // finite differences send each other nothing and finish without waiting
    // for each other.
    REQUIRE(reuse.messages[0].size() == legacy.messages[0].size());
    for (const auto& [receiver, messages] : reuse.messages[0]) {
      CAPTURE(receiver);
      const auto& legacy_messages = legacy.messages[0].at(receiver);
      if (receiver == dg_id or not reuses) {
        check_bitwise_equal(messages, legacy_messages);
        continue;
      }
      REQUIRE(messages.size() == legacy_messages.size());
      for (const auto& [key, data] : messages) {
        CAPTURE(key);
        const auto legacy_message = alg::find_if(
            legacy_messages,
            [&key](const auto& message) { return message.first == key; });
        REQUIRE(legacy_message != legacy_messages.end());
        if (key.id() == dg_id) {
          CHECK(data == legacy_message->second);
          continue;
        }
        CHECK(data.volume_mesh == legacy_message->second.volume_mesh);
        CHECK(data.volume_mesh_ghost_cell_data ==
              legacy_message->second.volume_mesh_ghost_cell_data);
        CHECK(data.validity_range == legacy_message->second.validity_range);
        CHECK(data.tci_status == legacy_message->second.tci_status);
        CHECK(data.integration_order ==
              legacy_message->second.integration_order);
        CHECK_FALSE(data.boundary_correction_data.has_value());
        check_used_components(data.ghost_cell_data.value(),
                              legacy_message->second.ghost_cell_data.value(),
                              number_of_ghost_points(key));
      }
    }
    if (reuses) {
      REQUIRE(reuse.messages[1].size() == 1);
      check_bitwise_equal(reuse.messages[1].at(dg_id),
                          legacy.messages[1].at(dg_id));
    } else {
      REQUIRE(reuse.messages[1].size() == legacy.messages[1].size());
      for (const auto& [receiver, messages] : legacy.messages[1]) {
        CAPTURE(receiver);
        check_bitwise_equal(reuse.messages[1].at(receiver), messages);
      }
    }
    if (reuses) {
      CHECK(reuse.done_before_dg_messages ==
            std::unordered_set<ElementId<3>>{
                extender_id, receiver_id, receiver_extender_id, isolated_id});
    }

    std::unordered_map<ElementId<3>, GhostDataMap> ghost_data{};
    for (const auto& id : fd_ids) {
      ghost_data[id] = ActionTesting::get_databox_tag<
          two_exchange_component<ReuseMetavars>,
          evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
          reuse_runner, id);
    }
    previous_ghost_data = std::move(ghost_data);
  }

#ifdef SPECTRE_DEBUG
  if (reuses) {
    // The second receive requires the ghost data of exactly the neighbors
    // using finite differences: an element without messages to receive
    // misses one, and an element with messages to receive still holds the
    // data of its neighbor using DG.
    using component = two_exchange_component<ReuseMetavars>;
    reuse_runner.force_next_action_to_be<component, SecondReceive>(extender_id);
    db::mutate<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
        [&sender_id](const auto ghost_data) {
          ghost_data->erase(
              DirectionalId<3>{Direction<3>::lower_eta(), sender_id});
        },
        make_not_null(&ActionTesting::get_databox<component>(
            make_not_null(&reuse_runner), extender_id)));
    CHECK_FALSE(ActionTesting::next_action_if_ready<component>(
        make_not_null(&reuse_runner), extender_id,
        Catch::Matchers::ContainsSubstring(
            "should hold only the ghost data of its neighbors using finite "
            "differences")));
    reuse_runner.force_next_action_to_be<component, SecondReceive>(sender_id);
    insert_dg_messages<false>(
        make_not_null(&reuse_runner),
        make_dg_messages(dg_element, dg_mesh, next_time_step_id, 0.0,
                         {{{{0.5, -0.5}}, {{0.5, -0.5}}}})[1],
        time_step_id);
    CHECK_FALSE(ActionTesting::next_action_if_ready<component>(
        make_not_null(&reuse_runner), sender_id,
        Catch::Matchers::ContainsSubstring(
            "except those of the neighbors using finite differences kept")));
  }
#endif  // SPECTRE_DEBUG
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Actions.ReconstructionCommunication",
                  "[Evolution][Unit]") {
  for (const bool use_cell_centered_flux : {false, true}) {
    test<1, false>(use_cell_centered_flux);
    test<2, false>(use_cell_centered_flux);
    test<3, false>(use_cell_centered_flux);
    test_receive_and_send_data<3, false>(false, use_cell_centered_flux);
    test_receive_and_send_data<3, false>(true, use_cell_centered_flux);
  }
  for (const bool enable_extension : {false, true}) {
    test_receive_and_send_data<3, false, true, true>(enable_extension, false);
  }
  test_inbox_routing<1, false, true>();
  test_inbox_routing<1, false, false>();
  for (const bool enable_extension : {false, true}) {
    test_reuse_auxiliary_ghost_data(enable_extension);
  }
}
}  // namespace
