// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/GetTciDecision.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Interpolators.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/MeshForGhostData.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/AuxiliaryGhostData.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/InterfaceDataPolicy.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr size_t Dim = 3;

// A "physical" field and a "gradient" field which, as for the CCZ4
// gradient-projection DG-FD scheme, is zeroed in the first finite-difference
// exchange and in all ghost data of DG elements, and sent by
// finite-difference elements in the second exchange.
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Var2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};
using evolved_tags = tmpl::list<Var1, Var2>;
using VariablesTag = ::Tags::Variables<evolved_tags>;

struct System {
  using variables_tag = VariablesTag;
  using flux_variables = tmpl::list<>;
};

class DummyReconstructor {
 public:
  static size_t ghost_zone_size() { return 2; }
  void pup(PUP::er& /*p*/) {}
};

namespace TestTags {
struct Reconstructor : db::SimpleTag {
  using type = std::unique_ptr<DummyReconstructor>;
};
// Whether a DG element sends its second (physical-inbox) data before its
// auxiliary ghost data, emulating out-of-order delivery at the receivers.
struct SendPhysicalFirst : db::SimpleTag {
  using type = bool;
};
struct StepsCompleted : db::SimpleTag {
  using type = size_t;
};
// {step, pass, neighbor segment index, max |Var2| of the neighbor's data,
// whether the message carried boundary-correction data}
struct Records : db::SimpleTag {
  using type = std::vector<std::array<double, 5>>;
};
}  // namespace TestTags

template <bool SendGradients>
struct GhostVariablesImpl {
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<VariablesTag>;
  static DataVector apply(const Variables<evolved_tags>& vars,
                          const size_t rdmp_size) {
    DataVector buffer{vars.size() + rdmp_size, 0.0};
    Variables<evolved_tags> view{buffer.data(), vars.size()};
    get<Var1>(view) = get<Var1>(vars);
    if constexpr (SendGradients) {
      get<Var2>(view) = get<Var2>(vars);
    }
    return buffer;
  }
};

using AuxInbox =
    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim, false, true>;
using PhysicalInbox =
    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim, false,
                                                              false>;

TimeStepId time_step_id(const size_t step) {
  return TimeStepId{true, 0,
                    Time{Slab{1.0, 2.0}, {static_cast<int>(step), 10}}};
}

double marker(const size_t step, const size_t segment) {
  return 100.0 * static_cast<double>(step + 1) + static_cast<double>(segment);
}

// Max |Var2| of a ghost-data buffer [Var1, Var2(, RDMP data)].
double max_gradient(const DataVector& data, const size_t rdmp_size) {
  const size_t half = (data.size() - rdmp_size) / 2;
  double result = 0.0;
  for (size_t i = half; i < 2 * half; ++i) {
    result = std::max(result, std::abs(data[i]));
  }
  return result;
}

// The send of ComputeTimeDerivative (boundary-correction data and ghost data
// to the physical inbox), on one side of the auxiliary send.
template <bool PhysicalFirst>
struct SendPhysicalIf {
  using inbox_tags = tmpl::list<PhysicalInbox>;
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (db::get<TestTags::SendPhysicalFirst>(box) == PhysicalFirst) {
      const Variables<db::wrap_tags_in<::Tags::Flux, tmpl::list<>,
                                       tmpl::size_t<Dim>, Frame::Inertial>>
          no_fluxes{};
      evolution::dg::Actions::detail::send_boundary_and_ghost_data<
          Dim, System, false, false, true, ParallelComponent>(
          make_not_null(&cache), make_not_null(&box), no_fluxes);
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// Stand-in for the receive of ApplyBoundaryCorrections (tested elsewhere):
// wait for one physical message per neighbor, record and consume it.
struct ReceivePhysical {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t expected =
        db::get<domain::Tags::Element<Dim>>(box).number_of_neighbors();
    auto& inbox = tuples::get<PhysicalInbox>(inboxes);
    inbox.collect_messages();
    const auto received = inbox.messages.find(db::get<::Tags::TimeStepId>(box));
    if (received == inbox.messages.end() or
        received->second.size() < expected) {
      inbox.set_missing_messages(expected - (received == inbox.messages.end()
                                                 ? 0
                                                 : received->second.size()));
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    REQUIRE(received->second.size() == expected);
    const size_t step = db::get<TestTags::StepsCompleted>(box);
    db::mutate<TestTags::Records>(
        [&received, &step](const auto records) {
          for (const auto& [sender, data] : received->second) {
            REQUIRE(data.ghost_cell_data.has_value());
            records->push_back(
                {{static_cast<double>(step), 2.0,
                  static_cast<double>(sender.id().segment_id(0).index()),
                  max_gradient(*data.ghost_cell_data, 2),
                  data.boundary_correction_data.has_value() ? 1.0 : 0.0}});
          }
        },
        make_not_null(&box));
    inbox.messages.erase(received);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// Record the received finite-difference ghost data; after the first exchange
// also set the "gradient" (stand-in for computing it from the ghost data).
template <size_t Pass>
struct RecordFdPass {
  template <typename DbTags, typename... InboxTags, typename ActionList,
            typename ParallelComponent, typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t step = db::get<TestTags::StepsCompleted>(box);
    const auto& ghost_data =
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>>(
            box);
    REQUIRE(ghost_data.size() ==
            db::get<domain::Tags::Element<Dim>>(box).number_of_neighbors());
    db::mutate<TestTags::Records>(
        [&ghost_data, &step](const auto records) {
          for (const auto& [id, data] : ghost_data) {
            records->push_back(
                {{static_cast<double>(step), static_cast<double>(Pass),
                  static_cast<double>(id.id().segment_id(0).index()),
                  max_gradient(data.neighbor_ghost_data_for_reconstruction(),
                               0),
                  0.0}});
          }
        },
        make_not_null(&box));
    if constexpr (Pass == 1) {
      db::mutate<VariablesTag>(
          [&step, &element_id](const auto vars) {
            get(get<Var2>(*vars)) =
                marker(step, element_id.segment_id(0).index());
          },
          make_not_null(&box));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct AdvanceStep {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
               TestTags::StepsCompleted>(
        [](const auto id, const auto next_id, const auto steps) {
          ++*steps;
          *id = time_step_id(*steps);
          *next_id = time_step_id(*steps + 1);
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

using initial_tags = tmpl::list<
    TestTags::Reconstructor, ::Tags::TimeStepId,
    ::Tags::Next<::Tags::TimeStepId>, domain::Tags::Mesh<Dim>,
    evolution::dg::subcell::Tags::Mesh<Dim>,
    evolution::dg::subcell::Tags::ActiveGrid, domain::Tags::Element<Dim>,
    evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
    evolution::dg::subcell::Tags::DataForRdmpTci,
    evolution::dg::subcell::Tags::TciDecision,
    evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>, VariablesTag,
    ::Tags::HistoryEvolvedVariables<VariablesTag>,
    evolution::dg::Tags::MortarInfo<Dim>, evolution::dg::Tags::MortarMesh<Dim>,
    evolution::dg::Tags::MortarData<Dim>,
    evolution::dg::Tags::MortarNextTemporalId<Dim>,
    domain::Tags::NeighborMesh<Dim>,
    evolution::dg::subcell::Tags::MeshForGhostData<Dim>,
    evolution::dg::subcell::Tags::CellCenteredFlux<tmpl::list<>, Dim>,
    evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<Dim>,
    evolution::dg::subcell::Tags::InterpolatorsFromDgToNeighborFd<Dim>,
    evolution::dg::subcell::Tags::InterpolatorsFromNeighborDgToFd<Dim>,
    evolution::dg::subcell::Tags::ExtensionDirections<Dim>,
    evolution::dg::subcell::Tags::SubcellOptions<Dim>,
    TestTags::SendPhysicalFirst, TestTags::StepsCompleted, TestTags::Records>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<initial_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             typename Metavariables::testing_actions>>;
};

namespace Labels = evolution::dg::subcell::Actions::Labels;

// The DG and finite-difference branches of the DG-subcell step, as in the
// CCZ4 gradient-projection executable, with stand-ins for the right-hand
// sides.
using chain_actions = tmpl::list<
    evolution::dg::subcell::Actions::SelectNumericalMethod,
    ::Actions::Label<Labels::BeginDg>, SendPhysicalIf<true>,
    evolution::dg::Actions::SendAuxiliaryGhostData<Dim, System, false>,
    SendPhysicalIf<false>, ReceivePhysical,
    evolution::dg::Actions::DrainAuxiliaryGhostData<Dim, false>,
    ::Actions::Goto<Labels::EndOfSolvers>,
    ::Actions::Label<Labels::BeginSubcell>,
    evolution::dg::subcell::Actions::SendDataForReconstruction<
        Dim, GhostVariablesImpl<false>, false, true>,
    evolution::dg::subcell::Actions::ReceiveAndSendDataForReconstruction<
        Dim, GhostVariablesImpl<false>, false, true>,
    evolution::dg::subcell::Actions::ReceiveDataForReconstruction<Dim, true>,
    RecordFdPass<1>,
    evolution::dg::subcell::Actions::SendDataForReconstruction<
        Dim, GhostVariablesImpl<true>, false>,
    evolution::dg::subcell::Actions::ReceiveAndSendDataForReconstruction<
        Dim, GhostVariablesImpl<true>, false>,
    evolution::dg::subcell::Actions::ReceiveDataForReconstruction<Dim>,
    RecordFdPass<2>, ::Actions::Label<Labels::EndOfSolvers>, AdvanceStep>;

// Gives the component a physical inbox.
struct DeclarePhysicalInbox {
  using inbox_tags = tmpl::list<PhysicalInbox>;
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& /*box*/,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

using send_and_drain_actions = tmpl::list<
    evolution::dg::Actions::SendAuxiliaryGhostData<Dim, System, false>,
    evolution::dg::Actions::DrainAuxiliaryGhostData<Dim, false>,
    DeclarePhysicalInbox>;

template <typename TestingActions>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using testing_actions = TestingActions;
  using component_list = tmpl::list<Component<Metavariables>>;
  using system = System;
  using const_global_cache_tags = tmpl::list<>;

  struct SubcellOptions {
    static constexpr bool subcell_enabled = true;
    static constexpr bool subcell_enabled_at_external_boundary = false;
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) {
      return db::get<TestTags::Reconstructor>(box).ghost_zone_size();
    }
    using GhostVariables = GhostVariablesImpl<false>;
  };
};

evolution::dg::subcell::SubcellOptions subcell_options() {
  return evolution::dg::subcell::SubcellOptions{
      4.0,
      1,
      2.0e-3,
      2.0e-4,
      false,
      false,
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
      false,
      std::nullopt,
      ::fd::DerivativeOrder::Two,
      1,
      1,
      1};
}

// Emplace an element with the given neighbors (and mortars on its DG faces).
template <typename Comp, typename Runner>
void emplace_element(const gsl::not_null<Runner*> runner,
                     const ElementId<Dim>& id,
                     const DirectionMap<Dim, Neighbors<Dim>>& neighbors,
                     const Mesh<Dim>& dg_mesh,
                     const evolution::dg::subcell::ActiveGrid active_grid,
                     const bool send_physical_first, const double var1,
                     const double var2, const int tci_decision) {
  const Element<Dim> element{id, neighbors};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t number_of_points =
      active_grid == evolution::dg::subcell::ActiveGrid::Dg
          ? dg_mesh.number_of_grid_points()
          : subcell_mesh.number_of_grid_points();
  Variables<evolved_tags> vars{number_of_points};
  get(get<Var1>(vars)) = var1;
  get(get<Var2>(vars)) = var2;

  typename evolution::dg::Tags::MortarInfo<Dim>::type mortar_info{};
  typename evolution::dg::Tags::MortarMesh<Dim>::type mortar_mesh{};
  typename evolution::dg::Tags::MortarData<Dim>::type mortar_data{};
  typename evolution::dg::Tags::MortarNextTemporalId<Dim>::type mortar_next{};
  typename evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>::type
      neighbor_decisions{};
  for (const auto& [direction, neighbors_in_direction] : neighbors) {
    for (const auto& neighbor : neighbors_in_direction) {
      const DirectionalId<Dim> mortar_id{direction, neighbor};
      mortar_info[mortar_id] = evolution::dg::MortarInfo<Dim>{
          {.interface_data_policy =
               neighbors_in_direction.orientation(neighbor).is_aligned()
                   ? evolution::dg::InterfaceDataPolicy::CopyProject
                   : evolution::dg::InterfaceDataPolicy::OrientCopyProject,
           .time_stepping_policy =
               evolution::dg::TimeSteppingPolicy::EqualRate}};
      mortar_mesh[mortar_id] = dg_mesh.slice_away(direction.dimension());
      mortar_data[mortar_id] = {};
      mortar_data[mortar_id].local().mortar_data =
          DataVector(mortar_mesh[mortar_id].number_of_grid_points(),
                     var1 + static_cast<double>(direction.dimension()));
      mortar_next[mortar_id] = time_step_id(0);
      neighbor_decisions.insert(std::pair{mortar_id, 0});
    }
  }
  ActionTesting::emplace_array_component_and_initialize<Comp>(
      runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, id,
      {std::make_unique<DummyReconstructor>(),
       time_step_id(0),
       time_step_id(1),
       dg_mesh,
       subcell_mesh,
       active_grid,
       element,
       typename evolution::dg::subcell::Tags::GhostDataForReconstruction<
           Dim>::type{},
       evolution::dg::subcell::RdmpTciData{{var1}, {var1 - 1.0}},
       tci_decision,
       neighbor_decisions,
       vars,
       TimeSteppers::History<Variables<evolved_tags>>{3},
       mortar_info,
       mortar_mesh,
       mortar_data,
       mortar_next,
       typename domain::Tags::NeighborMesh<Dim>::type{},
       typename evolution::dg::subcell::Tags::MeshForGhostData<Dim>::type{},
       typename evolution::dg::subcell::Tags::CellCenteredFlux<tmpl::list<>,
                                                               Dim>::type{},
       typename evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<
           Dim>::type{},
       typename evolution::dg::subcell::Tags::InterpolatorsFromDgToNeighborFd<
           Dim>::type{},
       typename evolution::dg::subcell::Tags::InterpolatorsFromNeighborDgToFd<
           Dim>::type{},
       typename evolution::dg::subcell::Tags::ExtensionDirections<Dim>::type{},
       subcell_options(),
       send_physical_first,
       0_st,
       typename TestTags::Records::type{}});
}

// The ghost-only send: content and destination, including a neighbor across
// a rotated block interface; then the drain: it waits (without consuming)
// until all neighbors' auxiliary data at the current time step id have
// arrived, also when the physical data arrive first, consumes only that
// entry, leaves the next step's entries and all other state alone, and
// rejects surplus messages.
void test_send_and_drain() {
  using metavars = Metavariables<send_and_drain_actions>;
  using comp = Component<metavars>;
  ActionTesting::MockRuntimeSystem<metavars> runner{{}};

  const Mesh<Dim> dg_mesh{{{4, 5, 6}},
                          Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const ElementId<Dim> self_id{0, {{{1, 0}, {0, 0}, {0, 0}}}};
  const ElementId<Dim> east_id{0, {{{1, 1}, {0, 0}, {0, 0}}}};
  const ElementId<Dim> south_id{1, {{{0, 0}, {0, 0}, {0, 0}}}};
  const ElementId<Dim> lonely_id{2, {{{0, 0}, {0, 0}, {0, 0}}}};
  // A rotation by 90 degrees about zeta, which permutes the xi and eta
  // extents of the oriented mesh.
  const OrientationMap<Dim> rotated{std::array{Direction<Dim>::upper_eta(),
                                               Direction<Dim>::lower_xi(),
                                               Direction<Dim>::upper_zeta()}};
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  neighbors[Direction<Dim>::upper_xi()] =
      Neighbors<Dim>{{east_id}, OrientationMap<Dim>::create_aligned()};
  neighbors[Direction<Dim>::lower_eta()] = Neighbors<Dim>{{south_id}, rotated};
  const double var1 = 1.5;
  const int tci_decision = 17;
  emplace_element<comp>(make_not_null(&runner), self_id, neighbors, dg_mesh,
                        evolution::dg::subcell::ActiveGrid::Dg, false, var1,
                        7.0, tci_decision);
  for (const auto& id : {east_id, south_id}) {
    emplace_element<comp>(make_not_null(&runner), id, {}, dg_mesh,
                          evolution::dg::subcell::ActiveGrid::Dg, false, 0.0,
                          0.0, 0);
  }
  emplace_element<comp>(make_not_null(&runner), lonely_id, {}, dg_mesh,
                        evolution::dg::subcell::ActiveGrid::Dg, false, 0.0, 0.0,
                        0);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  using ActionTesting::get_databox_tag;
  const auto mortar_data_before =
      get_databox_tag<comp, evolution::dg::Tags::MortarData<Dim>>(runner,
                                                                  self_id);
  const auto mortar_next_before =
      get_databox_tag<comp, evolution::dg::Tags::MortarNextTemporalId<Dim>>(
          runner, self_id);
  const auto vars_before = get_databox_tag<comp, VariablesTag>(runner, self_id);

  // Send.
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);
  CHECK(get_databox_tag<comp, evolution::dg::Tags::MortarData<Dim>>(
            runner, self_id) == mortar_data_before);
  CHECK(get_databox_tag<comp, VariablesTag>(runner, self_id) == vars_before);

  // The ghost data: Var1, zero Var2, then the RDMP data (all neighbors use
  // DG, so the unprojected volume data are sent).
  DataVector expected_ghost_data{2 * dg_mesh.number_of_grid_points() + 2, 0.0};
  for (size_t i = 0; i < dg_mesh.number_of_grid_points(); ++i) {
    expected_ghost_data[i] = var1;
  }
  expected_ghost_data[expected_ghost_data.size() - 2] = var1;
  expected_ghost_data[expected_ghost_data.size() - 1] = var1 - 1.0;
  for (const auto& [direction, neighbors_in_direction] : neighbors) {
    for (const auto& neighbor : neighbors_in_direction) {
      CAPTURE(neighbor);
      const auto& orientation = neighbors_in_direction.orientation(neighbor);
      CHECK(ActionTesting::get_inbox_tag<comp, PhysicalInbox>(runner, neighbor)
                .messages.empty());
      const auto& messages =
          ActionTesting::get_inbox_tag<comp, AuxInbox>(runner, neighbor)
              .messages;
      REQUIRE(messages.size() == 1);
      REQUIRE(messages.contains(time_step_id(0)));
      REQUIRE(messages.at(time_step_id(0)).size() == 1);
      const auto& [key, data] = messages.at(time_step_id(0)).front();
      CHECK(key ==
            DirectionalId<Dim>{orientation(direction.opposite()), self_id});
      CHECK(data.volume_mesh ==
            (orientation.is_aligned() ? dg_mesh : orientation(dg_mesh)));
      CHECK(data.volume_mesh_ghost_cell_data == std::optional{dg_mesh});
      CHECK_FALSE(data.boundary_correction_mesh.has_value());
      CHECK_FALSE(data.boundary_correction_data.has_value());
      CHECK(data.ghost_cell_data == std::optional{expected_ghost_data});
      CHECK(data.validity_range == time_step_id(1));
      CHECK(data.tci_status == tci_decision);
      CHECK(data.integration_order == 3);
      CHECK_FALSE(data.interpolated_boundary_data.has_value());
    }
  }
  // The rotated neighbor sees the oriented extents.
  CHECK(rotated(dg_mesh).extents() == Index<Dim>{5, 4, 6});

  // Drain.
  const DirectionalId<Dim> from_east{Direction<Dim>::upper_xi(), east_id};
  const DirectionalId<Dim> from_south{Direction<Dim>::lower_eta(), south_id};
  auto& aux_inbox = ActionTesting::get_inbox_tag<comp, AuxInbox>(
      make_not_null(&runner), self_id);
  auto& physical_inbox = ActionTesting::get_inbox_tag<comp, PhysicalInbox>(
      make_not_null(&runner), self_id);
  // The physical data arrive first.
  PhysicalInbox::insert_into_inbox(
      make_not_null(&physical_inbox), time_step_id(0),
      std::pair{from_east, evolution::dg::BoundaryData<Dim>{}});
  CHECK_FALSE(ActionTesting::next_action_if_ready<comp>(make_not_null(&runner),
                                                        self_id));
  // Partial data, and data of the next step.
  AuxInbox::insert_into_inbox(
      make_not_null(&aux_inbox), time_step_id(0),
      std::pair{from_east, evolution::dg::BoundaryData<Dim>{}});
  AuxInbox::insert_into_inbox(
      make_not_null(&aux_inbox), time_step_id(1),
      std::pair{from_east, evolution::dg::BoundaryData<Dim>{}});
  CHECK_FALSE(ActionTesting::next_action_if_ready<comp>(make_not_null(&runner),
                                                        self_id));
  REQUIRE(aux_inbox.messages.at(time_step_id(0)).size() == 1);
  REQUIRE(aux_inbox.messages.at(time_step_id(1)).size() == 1);
  AuxInbox::insert_into_inbox(
      make_not_null(&aux_inbox), time_step_id(0),
      std::pair{from_south, evolution::dg::BoundaryData<Dim>{}});
  CHECK(ActionTesting::next_action_if_ready<comp>(make_not_null(&runner),
                                                  self_id));
  CHECK_FALSE(aux_inbox.messages.contains(time_step_id(0)));
  CHECK(aux_inbox.messages.at(time_step_id(1)).size() == 1);
  CHECK(physical_inbox.messages.at(time_step_id(0)).size() == 1);
  CHECK(get_databox_tag<comp, evolution::dg::Tags::MortarData<Dim>>(
            runner, self_id) == mortar_data_before);
  CHECK(get_databox_tag<comp, evolution::dg::Tags::MortarNextTemporalId<Dim>>(
            runner, self_id) == mortar_next_before);
  CHECK(get_databox_tag<comp, domain::Tags::NeighborMesh<Dim>>(runner, self_id)
            .empty());
  CHECK(
      get_databox_tag<
          comp, evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>>(
          runner, self_id)
          .empty());
  CHECK(get_databox_tag<comp,
                        evolution::dg::subcell::Tags::MeshForGhostData<Dim>>(
            runner, self_id)
            .empty());

  // Surplus messages are an error. (The no-op and the send are run first.)
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);
  for (const auto& id : {from_east, from_south, from_east}) {
    AuxInbox::insert_into_inbox(
        make_not_null(&aux_inbox), time_step_id(0),
        std::pair{id, evolution::dg::BoundaryData<Dim>{}});
  }
  CHECK_FALSE(ActionTesting::next_action_if_ready<comp>(
      make_not_null(&runner), self_id,
      Catch::Matchers::ContainsSubstring(
          "received 3 auxiliary ghost-data messages")));
#ifdef SPECTRE_DEBUG
  aux_inbox.messages.erase(time_step_id(0));
  for (const auto& id : {from_east, from_east}) {
    AuxInbox::insert_into_inbox(
        make_not_null(&aux_inbox), time_step_id(0),
        std::pair{id, evolution::dg::BoundaryData<Dim>{}});
  }
  CHECK_FALSE(ActionTesting::next_action_if_ready<comp>(
      make_not_null(&runner), self_id,
      Catch::Matchers::ContainsSubstring(
          "received auxiliary ghost data twice")));
  aux_inbox.messages.erase(time_step_id(0));
  for (const auto& id :
       {from_east, DirectionalId<Dim>{Direction<Dim>::lower_xi(), lonely_id}}) {
    AuxInbox::insert_into_inbox(
        make_not_null(&aux_inbox), time_step_id(0),
        std::pair{id, evolution::dg::BoundaryData<Dim>{}});
  }
  CHECK_FALSE(ActionTesting::next_action_if_ready<comp>(
      make_not_null(&runner), self_id,
      Catch::Matchers::ContainsSubstring("which is not a neighbor")));
#endif  // SPECTRE_DEBUG

  // An element without neighbors sends nothing and does not wait.
  ActionTesting::next_action<comp>(make_not_null(&runner), lonely_id);
  CHECK(ActionTesting::next_action_if_ready<comp>(make_not_null(&runner),
                                                  lonely_id));
}

// A chain of elements DG - DG - FD - FD - DG along xi, for two time steps,
// with the elements scheduled in different orders. DG elements send to both
// inboxes without waiting (the first one physical-inbox data first); FD
// elements exchange twice. Checks that nothing deadlocks, that the gradients
// are zero in all first-exchange data and in all data sent by DG elements and
// are the actual FD gradients in the second exchange, and that all inboxes
// are empty at the end.
void test_chain(const size_t schedule) {
  CAPTURE(schedule);
  using metavars = Metavariables<chain_actions>;
  using comp = Component<metavars>;
  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  const Mesh<Dim> dg_mesh{4, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const size_t number_of_elements = 5;
  const std::array<bool, 5> is_fd{{false, false, true, true, false}};
  std::vector<ElementId<Dim>> ids{};
  for (size_t i = 0; i < number_of_elements; ++i) {
    ids.emplace_back(
        0, std::array{SegmentId{3, i}, SegmentId{0, 0}, SegmentId{0, 0}});
  }
  for (size_t i = 0; i < number_of_elements; ++i) {
    DirectionMap<Dim, Neighbors<Dim>> neighbors{};
    if (i > 0) {
      neighbors[Direction<Dim>::lower_xi()] =
          Neighbors<Dim>{{ids[i - 1]}, OrientationMap<Dim>::create_aligned()};
    }
    if (i + 1 < number_of_elements) {
      neighbors[Direction<Dim>::upper_xi()] =
          Neighbors<Dim>{{ids[i + 1]}, OrientationMap<Dim>::create_aligned()};
    }
    emplace_element<comp>(make_not_null(&runner), ids[i], neighbors, dg_mesh,
                          gsl::at(is_fd, i)
                              ? evolution::dg::subcell::ActiveGrid::Subcell
                              : evolution::dg::subcell::ActiveGrid::Dg,
                          i == 0, 1.0 + static_cast<double>(i),
                          gsl::at(is_fd, i) ? -5.0 : 7.0, 0);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const size_t number_of_steps = 2;
  const auto steps = [&runner](const ElementId<Dim>& id) {
    return ActionTesting::get_databox_tag<comp, TestTags::StepsCompleted>(
        runner, id);
  };
  std::vector<size_t> order(number_of_elements);
  for (size_t i = 0; i < number_of_elements; ++i) {
    order[i] = schedule == 1 ? number_of_elements - 1 - i : i;
  }
  bool done = false;
  for (size_t sweep = 0; sweep < 1000 and not done; ++sweep) {
    bool progress = false;
    for (const size_t i : order) {
      // Schedules 0 and 1 run each element until it blocks; schedule 2 runs
      // one action per element and sweep.
      while (steps(ids[i]) < number_of_steps and
             ActionTesting::next_action_if_ready<comp>(make_not_null(&runner),
                                                       ids[i])) {
        progress = true;
        if (schedule == 2) {
          break;
        }
      }
    }
    done = true;
    for (const auto& id : ids) {
      done = done and steps(id) == number_of_steps;
    }
    if (not done) {
      REQUIRE(progress);  // otherwise the protocol deadlocked
    }
  }
  REQUIRE(done);

  for (size_t i = 0; i < number_of_elements; ++i) {
    CAPTURE(i);
    CHECK(ActionTesting::get_inbox_tag<comp, AuxInbox>(runner, ids[i])
              .messages.empty());
    CHECK(ActionTesting::get_inbox_tag<comp, PhysicalInbox>(runner, ids[i])
              .messages.empty());
    const auto& records =
        ActionTesting::get_databox_tag<comp, TestTags::Records>(runner, ids[i]);
    const size_t number_of_neighbors =
        (i == 0 or i + 1 == number_of_elements) ? 1 : 2;
    CHECK(records.size() ==
          number_of_steps * number_of_neighbors * (gsl::at(is_fd, i) ? 2 : 1));
    for (const auto& [step, pass, neighbor, max_var2, had_correction] :
         records) {
      CAPTURE(step);
      CAPTURE(pass);
      CAPTURE(neighbor);
      const auto neighbor_index = static_cast<size_t>(neighbor);
      const bool neighbor_is_fd = gsl::at(is_fd, neighbor_index);
      if (pass == 1.0) {
        // First exchange: no gradients.
        CHECK(max_var2 == 0.0);
      } else {
        // Second exchange (FD receive, or DG physical receive): the FD
        // neighbor's gradients of this step, or zero from DG.
        CHECK(max_var2 == (neighbor_is_fd ? marker(static_cast<size_t>(step),
                                                   neighbor_index)
                                          : 0.0));
      }
      if (not gsl::at(is_fd, i)) {
        // DG elements receive boundary-correction data only from DG.
        CHECK(had_correction == (neighbor_is_fd ? 0.0 : 1.0));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.Actions.AuxiliaryGhostData",
                  "[Unit][Evolution][Actions]") {
  test_send_and_drain();
  for (size_t schedule = 0; schedule < 3; ++schedule) {
    test_chain(schedule);
  }
}
}  // namespace
