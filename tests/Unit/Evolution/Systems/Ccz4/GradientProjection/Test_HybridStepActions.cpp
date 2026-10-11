// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/Actions/TakeTimeStep.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/GetTciDecision.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SetInterpolators.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/AuxiliaryGhostData.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/Ccz4WrappedGr.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ResizeTimeDerivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SetInitialEta.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SetK0.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UnlimitedDeg4Prim.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryCorrections/UpwindGradientProjection.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Filter.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Initialize.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/NeighborPackagedData.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/SubcellStepActions.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TestHelpers.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/ConstraintDamping/Constant.hpp"
#include "PointwiseFunctions/ConstraintDamping/DampingFunction.hpp"
#include "PointwiseFunctions/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
namespace helpers = TestHelpers::Ccz4GradientProjection;
using system = gp::System;
using variables_tag = system::variables_tag;
using variables_tags = variables_tag::tags_list;
using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
using history_tag = ::Tags::HistoryEvolvedVariables<variables_tag>;
using Vars = variables_tag::type;
using DtVars = dt_variables_tag::type;
using physical_tags = system::physical_tags;
using filter_tag = Filters::runtime::Tags::SpectralFilter<3, variables_tags>;
using AuxInbox =
    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<3, false, true>;
using PhysicalInbox =
    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<3, false, false>;
using Minkowski = Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::Minkowski<3>>;
using ActiveGrid = evolution::dg::subcell::ActiveGrid;

// The periodic brick: four elements along x, of which the first two use DG
// and the last two finite differences, so that the x-interfaces are DG-DG,
// DG-FD, FD-FD and (periodically) FD-DG. Each element is its own neighbor
// along y and z.
constexpr size_t number_of_elements = 4;
constexpr size_t dg_points = 4;
const std::array<double, 3> element_lengths{{1.0, 1.2, 0.9}};

bool uses_dg(const ElementId<3>& id) { return id.segment_id(0).index() < 2; }

ElementId<3> element_id(const size_t index_x) {
  return {0, {{SegmentId{2, index_x}, SegmentId{0, 0}, SegmentId{0, 0}}}};
}

domain::creators::Brick brick() {
  return {{{0.0, 0.0, 0.0}},
          {{number_of_elements * element_lengths[0], element_lengths[1],
            element_lengths[2]}},
          {{2, 0, 0}},
          {{dg_points, dg_points, dg_points}},
          {{true, true, true}}};
}

// Fixes the numerical method of each element (in a periodic domain
// `SetSubcellGrid` selects finite differences everywhere) and sizes its
// evolved variables for it, before the initial data are set.
struct SetActiveGridForTest {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<3>& id, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t number_of_points =
        uses_dg(id)
            ? db::get<domain::Tags::Mesh<3>>(box).number_of_grid_points()
            : db::get<evolution::dg::subcell::Tags::Mesh<3>>(box)
                  .number_of_grid_points();
    db::mutate<evolution::dg::subcell::Tags::ActiveGrid, variables_tag>(
        [&id, &number_of_points](const auto active_grid, const auto vars) {
          *active_grid = uses_dg(id) ? ActiveGrid::Dg : ActiveGrid::Subcell;
          vars->initialize(number_of_points);
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// As in EvolveCcz4 (with the active grid fixed after SetSubcellGrid, and the
// time-stepping, filter and dense-trigger items given directly).
template <typename Metavariables>
using initialization_actions = tmpl::list<
    Initialization::Actions::InitializeItems<
        evolution::dg::Initialization::Domain<Metavariables>>,
    evolution::dg::subcell::Actions::SetSubcellGrid<3, system, true>,
    SetActiveGridForTest, gp::Actions::CheckFiniteDifferenceOptions,
    ::Actions::MutateApply<evolution::dg::subcell::SetInterpolators<
        3, Ccz4::fd::Tags::Reconstructor>>,
    ::Actions::MutateApply<Ccz4::fd::ResizeTimeDerivatives>,
    gp::Actions::SetInitialData,
    ::Actions::MutateApply<gp::EnforceAlgebraicConstraints>,
    evolution::dg::Initialization::Mortars<3>,
    Initialization::Actions::AddSimpleTags<Ccz4::fd::SetInitialEta,
                                           Ccz4::fd::SetK0>,
    Initialization::Actions::InitializeItems<gp::InitializeLocalOperators>>;

// The production DG-subcell step and the time advance.
using evolve_actions = tmpl::push_back<gp::subcell_step_actions,
                                       ::Actions::MutateApply<AdvanceTime<>>>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using simple_tags = tmpl::list<
      domain::Tags::InitialExtents<3>, domain::Tags::InitialRefinementLevels<3>,
      evolution::dg::Tags::Quadrature, ::Tags::TimeStepId,
      ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep, ::Tags::Time,
      ::Tags::StepNumberWithinSlab, ::Tags::AdaptiveSteppingDiagnostics,
      variables_tag, dt_variables_tag, history_tag, filter_tag,
      ::Tags::EventsAndDenseTriggers, ::Tags::PreviousTriggerTime>;
  using compute_tags =
      tmpl::push_back<time_stepper_ref_tags<TimeStepper>,
                      Ccz4::Tags::Kappa1Compute, Ccz4::Tags::Kappa2Compute>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::push_front<
              initialization_actions<Metavariables>,
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::InitializeTimeStepperHistory,
          SelfStart::self_start_procedure<gp::subcell_step_actions, system>>,
      Parallel::PhaseActions<Parallel::Phase::Evolve, evolve_actions>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 3;
  static constexpr bool local_time_stepping = false;
  using system = gp::System;
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tags = tmpl::list<
      ::Tags::ConcreteTimeStepper<TimeStepper>, Ccz4::Tags::Kappa3,
      evolution::initial_data::Tags::InitialData, Ccz4::fd::Tags::Reconstructor,
      Ccz4::fd::Tags::KreissOligerEpsilon, Ccz4::fd::Tags::EvolveLapseAndShift,
      Ccz4::fd::Tags::ConstrainedEvolution, Ccz4::fd::Tags::EtaConstant,
      Ccz4::Tags::DampingFunctionKappa1, Ccz4::Tags::DampingFunctionKappa2,
      ::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>;
  struct SubcellOptions {
    static constexpr bool subcell_enabled = true;
    static constexpr bool subcell_enabled_at_external_boundary = false;
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) {
      return db::get<Ccz4::fd::Tags::Reconstructor>(box).ghost_zone_size();
    }
    using GhostVariables = Ccz4::fd::GhostVariables;
    using DgComputeSubcellNeighborPackagedData = gp::NeighborPackagedData;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            evolution::BoundaryCorrection,
            tmpl::list<gp::BoundaryCorrections::UpwindGradientProjection>>,
        tmpl::pair<
            Ccz4::BoundaryConditions::BoundaryCondition,
            tmpl::list<gp::BoundaryConditions::DirichletAnalytic,
                       domain::BoundaryConditions::Periodic<
                           Ccz4::BoundaryConditions::BoundaryCondition>>>,
        tmpl::pair<evolution::initial_data::InitialData, tmpl::list<Minkowski>>,
        tmpl::pair<DenseTrigger, tmpl::list<>>,
        tmpl::pair<Event, tmpl::list<>>>;
  };
};

using component = Component<Metavariables>;
using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;

template <typename Tag>
const auto& get_tag(const MockRuntimeSystem& runner, const ElementId<3>& id) {
  return ActionTesting::get_databox_tag<component, Tag>(runner, id);
}

constexpr size_t time_stepper_order = 2;
constexpr double kappa_1 = 0.001;
constexpr double eta = 1.0;
const Slab slab{0.0, 0.002};

tuples::tagged_tuple_from_typelist<
    Parallel::get_const_global_cache_tags<Metavariables>>
make_cache() {
  const auto creator = brick();
  tuples::tagged_tuple_from_typelist<
      Parallel::get_const_global_cache_tags<Metavariables>>
      cache{};
  get<domain::Tags::Domain<3>>(cache) = creator.create_domain();
  get<::Tags::ConcreteTimeStepper<TimeStepper>>(cache) =
      std::make_unique<TimeSteppers::AdamsBashforth>(time_stepper_order);
  get<Ccz4::Tags::Kappa3>(cache) = 0.5;
  get<evolution::Tags::BoundaryCorrection>(cache) =
      std::make_unique<gp::BoundaryCorrections::UpwindGradientProjection>(1.0);
  get<::dg::Tags::Formulation>(cache) = ::dg::Formulation::StrongInertial;
  get<domain::Tags::ExternalBoundaryConditions<3>>(cache) =
      creator.external_boundary_conditions();
  get<evolution::initial_data::Tags::InitialData>(cache) =
      std::make_unique<Minkowski>();
  get<evolution::dg::subcell::Tags::SubcellOptions<3>>(cache) =
      evolution::dg::subcell::SubcellOptions{
          4.0,
          1_st,
          1.0e-3,
          1.0e-4,
          false,
          false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
          false,
          std::nullopt,
          ::fd::DerivativeOrder::Two,
          1,
          1,
          1};
  get<Ccz4::fd::Tags::Reconstructor>(cache) =
      std::make_unique<Ccz4::fd::UnlimitedDeg4Prim>();
  get<Ccz4::fd::Tags::KreissOligerEpsilon>(cache) = 0.1;
  get<Ccz4::fd::Tags::EvolveLapseAndShift>(cache) = true;
  get<Ccz4::fd::Tags::ConstrainedEvolution>(cache) = true;
  get<Ccz4::fd::Tags::EtaConstant>(cache) = eta;
  get<Ccz4::Tags::DampingFunctionKappa1>(cache) =
      std::make_unique<ConstraintDamping::Constant<3, Frame::Grid>>(kappa_1);
  get<Ccz4::Tags::DampingFunctionKappa2>(cache) =
      std::make_unique<ConstraintDamping::Constant<3, Frame::Grid>>(0.0);
  return cache;
}

// Emplaces and initializes the elements, and enters the evolution phase.
void initialize_elements(const gsl::not_null<MockRuntimeSystem*> runner) {
  const auto creator = brick();
  const TimeDelta step = slab.duration();
  for (size_t i = 0; i < number_of_elements; ++i) {
    const ElementId<3> id = element_id(i);
    ActionTesting::emplace_array_component_and_initialize<component>(
        runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, id,
        {creator.initial_extents(), creator.initial_refinement_levels(),
         Spectral::Quadrature::GaussLobatto, TimeStepId{},
         TimeStepId{true, 1 - static_cast<int64_t>(time_stepper_order),
                    slab.start()},
         step, slab.start().value(), uint64_t{0},
         ::Tags::AdaptiveSteppingDiagnostics::type{}, Vars{1, 777.0},
         DtVars{1, 0.0}, history_tag::type{1},
         std::unique_ptr<Filters::runtime::Filter<3, variables_tags>>{
             std::make_unique<Filters::runtime::None<3, variables_tags>>(
                 std::nullopt)},
         EventsAndDenseTriggers(EventsAndDenseTriggers::ConstructionType{}),
         std::optional<double>{}});
    for (size_t a = 0;
         a < tmpl::size<initialization_actions<Metavariables>>::value; ++a) {
      ActionTesting::next_action<component>(runner, id);
    }
    CHECK(get_tag<evolution::dg::subcell::Tags::ActiveGrid>(*runner, id) ==
          (uses_dg(id) ? ActiveGrid::Dg : ActiveGrid::Subcell));
  }
}

std::vector<ElementId<3>> all_ids() {
  std::vector<ElementId<3>> ids{};
  for (size_t i = 0; i < number_of_elements; ++i) {
    ids.push_back(element_id(i));
  }
  return ids;
}

size_t next_action(const MockRuntimeSystem& runner, const ElementId<3>& id) {
  return ActionTesting::get_next_action_index<component>(runner, id);
}

template <typename Action>
constexpr size_t index_of = tmpl::index_of<evolve_actions, Action>::value;

// Runs the elements, in the order of `schedule` (0: forward, 1: reverse,
// each element until it blocks; 2: forward, one action per element and
// sweep), until `stop(id)` holds for all of them. Fails on a deadlock.
void run_until(const gsl::not_null<MockRuntimeSystem*> runner,
               const size_t schedule,
               const std::function<bool(const ElementId<3>&)>& stop) {
  std::vector<ElementId<3>> order = all_ids();
  if (schedule == 1) {
    std::reverse(order.begin(), order.end());
  }
  for (size_t sweep = 0; sweep < 100000; ++sweep) {
    bool progress = false;
    bool done = true;
    for (const auto& id : order) {
      while (not stop(id) and
             ActionTesting::next_action_if_ready<component>(runner, id)) {
        progress = true;
        if (schedule == 2) {
          break;
        }
      }
      done = done and stop(id);
    }
    if (done) {
      return;
    }
    REQUIRE(progress);  // otherwise the protocol deadlocked
  }
  REQUIRE(false);
}

// Whether the element has completed `steps` steps (and not started another).
// Each step is a slab.
std::function<bool(const ElementId<3>&)> completed(
    const MockRuntimeSystem& runner, const size_t steps) {
  const double target = static_cast<double>(steps) * slab.duration().value();
  return [&runner, target](const ElementId<3>& id) {
    return std::abs(get_tag<::Tags::Time>(runner, id) - target) < 1.0e-14 and
           next_action(runner, id) == 0;
  };
}

void check_inboxes_empty(const MockRuntimeSystem& runner) {
  for (const auto& id : all_ids()) {
    CAPTURE(id);
    CHECK(ActionTesting::get_inbox_tag<component, AuxInbox>(runner, id)
              .messages.empty());
    CHECK(ActionTesting::get_inbox_tag<component, PhysicalInbox>(runner, id)
              .messages.empty());
  }
}

const tnsr::I<DataVector, 3>& active_coordinates(
    const MockRuntimeSystem& runner, const ElementId<3>& id) {
  return uses_dg(id) ? get_tag<domain::Tags::Coordinates<3, Frame::Inertial>>(
                           runner, id)
                     : get_tag<evolution::dg::subcell::Tags::Coordinates<
                           3, Frame::Inertial>>(runner, id);
}

// Runs the production self-start procedure (the step actions on all elements)
// and enters the evolution phase at the start of the slab.
void self_start(const gsl::not_null<MockRuntimeSystem*> runner,
                const size_t schedule) {
  ActionTesting::set_phase(runner,
                           Parallel::Phase::InitializeTimeStepperHistory);
  run_until(runner, schedule, [&runner](const ElementId<3>& id) {
    return ActionTesting::get_terminate<component>(*runner, id);
  });
  for (const auto& id : all_ids()) {
    CHECK(get_tag<::Tags::TimeStepId>(*runner, id) ==
          TimeStepId(true, 0, slab.start()));
    CHECK(get_tag<history_tag>(*runner, id).integration_order() ==
          time_stepper_order);
  }
  check_inboxes_empty(*runner);
  ActionTesting::set_phase(runner, Parallel::Phase::Evolve);
}

constexpr double gradient_poison = 3.0e3;

void poison_gradients(const gsl::not_null<MockRuntimeSystem*> runner) {
  for (const auto& id : all_ids()) {
    auto& box = ActionTesting::get_databox<component>(runner, id);
    db::mutate<variables_tag>(
        [](const auto vars) {
          for (auto& component : get<gp::Tags::PotentialGradients>(*vars)) {
            component = gradient_poison;
          }
        },
        make_not_null(&box));
  }
}

// Minkowski data stay Minkowski, to roundoff, through two steps of the hybrid
// scheme (Gamma-driver damping eta = 1, kappa_1 = 0.001), and every exchange
// is consumed.
void test_minkowski(const size_t schedule) {
  CAPTURE(schedule);
  MockRuntimeSystem runner{
      make_cache(),
      tuples::TaggedTuple<domain::Tags::FunctionsOfTimeInitialize>{}};
  initialize_elements(make_not_null(&runner));
  self_start(make_not_null(&runner), schedule);
  poison_gradients(make_not_null(&runner));
  for (size_t steps = 1; steps <= 2; ++steps) {
    run_until(make_not_null(&runner), schedule, completed(runner, steps));
    check_inboxes_empty(runner);
  }
  for (const auto& id : all_ids()) {
    CAPTURE(id);
    const auto& vars = get_tag<variables_tag>(runner, id);
    const auto exact = Minkowski{}.variables(active_coordinates(runner, id),
                                             0.0, physical_tags{});
    tmpl::for_each<physical_tags>(
        [&vars, &exact]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          CAPTURE(db::tag_name<Tag>());
          for (size_t c = 0; c < get<Tag>(vars).size(); ++c) {
            CHECK(max(abs(get<Tag>(vars)[c] - get<Tag>(exact)[c])) < 1.0e-12);
          }
        });
  }
}

// A smooth admissible state, periodic on the brick: anisotropic H, non-unit
// conformal factor and lapse, nonzero shift and velocities.
Vars smooth_state(const tnsr::I<DataVector, 3>& x) {
  const size_t number_of_points = get<0>(x).size();
  const std::array<double, 3> wave_numbers{
      {2.0 * M_PI / (number_of_elements * element_lengths[0]),
       2.0 * M_PI / element_lengths[1], 2.0 * M_PI / element_lengths[2]}};
  helpers::ChartData chart(number_of_points);
  for (size_t s = 0; s < number_of_points; ++s) {
    std::array<double, 3> phase{};
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(phase, i) = gsl::at(wave_numbers, i) * x.get(i)[s];
    }
    const auto mode = [&phase](const size_t c) {
      const double k = static_cast<double>(c);
      return std::sin(phase[0] + 0.7 * k) + 0.5 * std::cos(phase[1] + 1.3 * k) +
             0.3 * std::sin(phase[2] - 0.4 * k);
    };
    gp::PotentialVector u{};
    gp::VelocityVector w{};
    for (size_t c = 0; c < 5; ++c) {
      gsl::at(u, gp::PotentialIndex::log_metric + c) = 0.05 * mode(c);
    }
    u[gp::PotentialIndex::conformal_factor] = 1.0 + 0.05 * mode(5);
    u[gp::PotentialIndex::lapse] = 1.0 + 0.05 * mode(6);
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(u, gp::PotentialIndex::shift + i) = 0.02 * mode(7 + i);
    }
    for (size_t c = 0; c < gp::number_of_velocities; ++c) {
      gsl::at(w, c) = 0.02 * mode(10 + c);
    }
    chart.set(s, u, w);
  }
  Vars vars{number_of_points, gradient_poison};
  get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(vars) =
      chart.conformal_metric;
  get<Ccz4::Tags::ConformalFactor<DataVector>>(vars) = chart.conformal_factor;
  get<Ccz4::Tags::ATilde<DataVector, 3>>(vars) = chart.a_tilde;
  get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars) =
      chart.trace_extrinsic_curvature;
  get<Ccz4::Tags::Theta<DataVector>>(vars) = chart.theta;
  get<Ccz4::Tags::GammaHat<DataVector, 3>>(vars) = chart.gamma_hat;
  get<gr::Tags::Lapse<DataVector>>(vars) = chart.lapse;
  get<gr::Tags::Shift<DataVector, 3>>(vars) = chart.shift;
  get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(vars) = chart.b;
  return vars;
}

// On a DG element stopped after ApplyBoundaryCorrectionsToTimeDerivative
// (before DrainAuxiliaryGhostData and FinalizeTimeDerivative) whose
// neighbor in `direction` uses finite differences and is stopped before its
// TakeTimeStep:
// - the physical (and gradient) time derivatives are still zero;
// - the DG element's stored gradients are those of the current state (not the
//   poisoned old ones);
// - the neighbor's package stored on the mortar is the package of the
//   neighbor-side face values predicted, with independent degree-4 Lagrange
//   weights, from the subcell projection of the DG element's volume and the
//   neighbor's actual cells (its current physical fields and the gradients
//   it computed this step, not its poisoned old ones), reconstructed onto
//   the DG face.
void check_dg_element(const MockRuntimeSystem& runner, const ElementId<3>& id,
                      const Direction<3>& direction,
                      const ElementId<3>& neighbor_id) {
  CAPTURE(id);
  CAPTURE(direction);
  REQUIRE(next_action(runner, id) ==
          index_of<evolution::dg::Actions::DrainAuxiliaryGhostData<3, false>>);
  REQUIRE(next_action(runner, neighbor_id) ==
          index_of<evolution::dg::subcell::fd::Actions::TakeTimeStep<
              Ccz4::fd::SoTimeDerivative>>);
  const auto& dt_vars = get_tag<dt_variables_tag>(runner, id);
  for (size_t i = 0; i < dt_vars.size(); ++i) {
    CHECK(dt_vars.data()[i] == 0.0);
  }

  const Vars& vars = get_tag<variables_tag>(runner, id);
  {
    Vars recomputed = vars;
    for (auto& component : get<gp::Tags::PotentialGradients>(recomputed)) {
      component = gradient_poison;
    }
    DtVars dt_recomputed{vars.number_of_grid_points()};
    gp::Tags::RhsTargets::type targets{};
    gp::ComputeVolumeTargets::apply(
        make_not_null(&recomputed), make_not_null(&dt_recomputed),
        make_not_null(&targets), get_tag<domain::Tags::Mesh<3>>(runner, id),
        get_tag<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                              Frame::Inertial>>(runner, id),
        get_tag<Ccz4::Tags::Kappa1>(runner, id),
        get_tag<Ccz4::Tags::Kappa2>(runner, id),
        get_tag<Ccz4::Tags::Kappa3>(runner, id),
        get_tag<Ccz4::Tags::Eta<DataVector>>(runner, id),
        get_tag<Ccz4::Tags::K0<DataVector>>(runner, id));
    CHECK(get<gp::Tags::PotentialGradients>(vars) ==
          get<gp::Tags::PotentialGradients>(recomputed));
  }

  const Mesh<3>& dg_mesh = get_tag<domain::Tags::Mesh<3>>(runner, id);
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  REQUIRE(get_tag<evolution::dg::subcell::Tags::Mesh<3>>(runner, neighbor_id) ==
          subcell_mesh);
  const Vars& neighbor_vars = get_tag<variables_tag>(runner, neighbor_id);
  REQUIRE(neighbor_vars.number_of_grid_points() ==
          subcell_mesh.number_of_grid_points());
  CHECK(max(abs(get<0, 0>(get<gp::Tags::PotentialGradients>(neighbor_vars)))) <
        1.0);

  const size_t dim = direction.dimension();
  const size_t n = subcell_mesh.extents(dim);
  const double delta = 2.0 / static_cast<double>(n);
  const Vars projected = evolution::dg::subcell::fd::project(
      vars, dg_mesh, subcell_mesh.extents());
  // Host cells k = 0, 1 and neighbor cells k = -1, -2, -3 counted from the
  // face; the cells are aligned (same block).
  const auto layer = [&](const int k) {
    if (k >= 0) {
      const size_t index = direction.side() == Side::Upper
                               ? n - 1 - static_cast<size_t>(k)
                               : static_cast<size_t>(k);
      return data_on_slice(projected, subcell_mesh.extents(), dim, index);
    }
    const auto j = static_cast<size_t>(-k - 1);
    const size_t index = direction.side() == Side::Upper ? j : n - 1 - j;
    return data_on_slice(neighbor_vars, subcell_mesh.extents(), dim, index);
  };
  const std::array<int, 5> cells{{1, 0, -1, -2, -3}};
  std::array<double, 5> nodes{};
  for (size_t m = 0; m < 5; ++m) {
    gsl::at(nodes, m) =
        1.0 - (static_cast<double>(gsl::at(cells, m)) + 0.5) * delta;
  }
  const auto weights = helpers::lagrange_weights(nodes, 1.0);
  Vars face_values{subcell_mesh.extents().slice_away(dim).product(), 0.0};
  for (size_t m = 0; m < 5; ++m) {
    face_values += gsl::at(weights, m) * layer(gsl::at(cells, m));
  }
  tnsr::i<DataVector, 3> reversed_normal{face_values.number_of_grid_points(),
                                         0.0};
  reversed_normal.get(dim) = -static_cast<double>(direction.sign());
  const auto fd_package = gp::package_finite_difference_face(
      std::move(face_values), reversed_normal,
      gp::BoundaryCorrections::UpwindGradientProjection{1.0}, direction);
  const auto expected = evolution::dg::subcell::fd::reconstruct(
      fd_package, dg_mesh.slice_away(dim),
      subcell_mesh.extents().slice_away(dim),
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);

  const auto& neighbor_data =
      get_tag<evolution::dg::Tags::MortarData<3>>(runner, id)
          .at(DirectionalId<3>{direction, neighbor_id})
          .neighbor();
  REQUIRE(neighbor_data.mortar_data.has_value());
  CHECK(neighbor_data.mortar_mesh == std::optional{dg_mesh.slice_away(dim)});
  const DataVector& stored = *neighbor_data.mortar_data;
  REQUIRE(stored.size() == expected.size());
  double largest_difference = 0.0;
  for (size_t i = 0; i < stored.size(); ++i) {
    largest_difference =
        std::max(largest_difference, std::abs(stored[i] - expected.data()[i]));
  }
  CAPTURE(largest_difference);
  CHECK(largest_difference < 1.0e-11);
}

// A smooth nonconstant state with poisoned stored gradients, over two steps
// with different schedules: at each step the DG elements next to the
// finite-difference elements are checked between their boundary
// corrections and their finalization.
void test_smooth_state() {
  MockRuntimeSystem runner{
      make_cache(),
      tuples::TaggedTuple<domain::Tags::FunctionsOfTimeInitialize>{}};
  initialize_elements(make_not_null(&runner));
  for (const auto& id : all_ids()) {
    auto& box =
        ActionTesting::get_databox<component>(make_not_null(&runner), id);
    const Vars state = smooth_state(active_coordinates(runner, id));
    db::mutate<variables_tag>([&state](const auto vars) { *vars = state; },
                              make_not_null(&box));
  }
  self_start(make_not_null(&runner), 2);
  const auto& ids = all_ids();
  const auto stop_for_check = [&runner](const ElementId<3>& id) {
    return next_action(runner, id) ==
           (uses_dg(id)
                ? index_of<
                      evolution::dg::Actions::DrainAuxiliaryGhostData<3, false>>
                : index_of<evolution::dg::subcell::fd::Actions::TakeTimeStep<
                      Ccz4::fd::SoTimeDerivative>>);
  };
  for (size_t steps = 1; steps <= 2; ++steps) {
    CAPTURE(steps);
    poison_gradients(make_not_null(&runner));
    // Forward greedy for the first step, round robin for the second.
    run_until(make_not_null(&runner), steps == 1 ? 0 : 2, stop_for_check);
    check_dg_element(runner, ids[1], Direction<3>::upper_xi(), ids[2]);
    check_dg_element(runner, ids[0], Direction<3>::lower_xi(), ids[3]);
    run_until(make_not_null(&runner), steps == 1 ? 1 : 0,
              completed(runner, steps));
    check_inboxes_empty(runner);
    for (const auto& id : ids) {
      CHECK(get_tag<::Tags::Time>(runner, id) ==
            approx(static_cast<double>(steps) * slab.duration().value()));
    }
  }
}

// [[TimeOut, 10]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.GradientProjection.HybridStepActions",
    "[Unit][Evolution]") {
  domain::creators::register_derived_with_charm();
  register_factory_classes_with_charm<Metavariables>();
  register_classes_with_charm<TimeSteppers::AdamsBashforth>();
  register_classes_with_charm<Filters::runtime::None<3, variables_tags>>();
  Ccz4::fd::register_derived_with_charm();
  ConstraintDamping::register_derived_with_charm();
  test_minkowski(1);
  test_smooth_state();
}
}  // namespace
