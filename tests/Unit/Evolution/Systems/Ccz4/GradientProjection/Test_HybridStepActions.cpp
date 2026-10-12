// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/AlignedLattice.hpp"
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
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
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
#include "Evolution/DgSubcell/GhostData.hpp"
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
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
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
#include "Evolution/Systems/Ccz4/FiniteDifference/ApplyFilter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
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
#include "Evolution/Systems/Ccz4/GradientProjection/FiniteDifferenceGradients.hpp"
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
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
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
using KerrSchild = Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::KerrSchild>;
using ActiveGrid = evolution::dg::subcell::ActiveGrid;
using GhostDataMap = DirectionalIdMap<3, evolution::dg::subcell::GhostData>;

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

std::vector<ElementId<3>> all_ids() {
  std::vector<ElementId<3>> ids{};
  for (size_t i = 0; i < number_of_elements; ++i) {
    ids.push_back(element_id(i));
  }
  return ids;
}

// The lattice with analytic (Kerr-Schild) external boundaries: four blocks
// of one element each along x, of which the last is DG-only. With finite
// differences allowed at external boundaries (as in EvolveCcz4),
// `SetSubcellGrid` selects finite differences for the first two blocks and DG
// for the last two (DG-only, or bordering a DG-only block), so that the
// x-interfaces are FD-FD, FD-DG and DG-DG. Every element is at the external y
// and z boundaries, the first also at the lower x boundary and the last at
// the upper x boundary.
constexpr size_t number_of_lattice_blocks = 4;
const std::array<double, 3> lattice_lower_corner{{3.0, -0.6, -0.45}};

std::vector<ElementId<3>> lattice_ids() {
  std::vector<ElementId<3>> ids{};
  for (size_t block = 0; block < number_of_lattice_blocks; ++block) {
    ids.emplace_back(block);
  }
  return ids;
}

bool uses_dg_in_lattice(const ElementId<3>& id) { return id.block_id() >= 2; }

KerrSchild kerr_schild() {
  return KerrSchild{gr::Solutions::KerrSchild{
      1.0, {{0.5, -0.2, 0.0}}, {{-0.2, 0.0, 0.3}}, {{0.0, 0.0, 0.0}}}};
}

domain::creators::AlignedLattice<3> lattice() {
  std::array<std::vector<double>, 3> block_bounds{};
  for (size_t block = 0; block <= number_of_lattice_blocks; ++block) {
    block_bounds[0].push_back(lattice_lower_corner[0] +
                              static_cast<double>(block) * element_lengths[0]);
  }
  for (size_t d = 1; d < 3; ++d) {
    gsl::at(block_bounds, d) = {
        gsl::at(lattice_lower_corner, d),
        gsl::at(lattice_lower_corner, d) + gsl::at(element_lengths, d)};
  }
  std::array<
      std::array<std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>,
                 2>,
      3>
      boundary_conditions{};
  for (auto& lower_and_upper : boundary_conditions) {
    for (auto& boundary_condition : lower_and_upper) {
      boundary_condition =
          std::make_unique<gp::BoundaryConditions::DirichletAnalytic>(
              std::make_unique<KerrSchild>(kerr_schild()));
    }
  }
  return domain::creators::AlignedLattice<3>(
      block_bounds, {{0, 0, 0}}, {{dg_points, dg_points, dg_points}}, {}, {},
      {}, std::move(boundary_conditions));
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

// As in EvolveCcz4 (with the active grid of the periodic brick fixed after
// SetSubcellGrid, and the time-stepping, filter and dense-trigger items given
// directly).
template <typename Metavariables>
using initialization_actions = tmpl::flatten<tmpl::list<
    Initialization::Actions::InitializeItems<
        evolution::dg::Initialization::Domain<Metavariables>>,
    evolution::dg::subcell::Actions::SetSubcellGrid<3, system, true>,
    typename Metavariables::set_active_grid,
    gp::Actions::CheckFiniteDifferenceOptions,
    ::Actions::MutateApply<evolution::dg::subcell::SetInterpolators<
        3, Ccz4::fd::Tags::Reconstructor>>,
    ::Actions::MutateApply<Ccz4::fd::ResizeTimeDerivatives>,
    gp::Actions::SetInitialData,
    ::Actions::MutateApply<gp::EnforceAlgebraicConstraints>,
    evolution::dg::Initialization::Mortars<3>,
    Initialization::Actions::AddSimpleTags<Ccz4::fd::SetInitialEta,
                                           Ccz4::fd::SetK0>,
    Initialization::Actions::InitializeItems<gp::InitializeLocalOperators>>>;

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

// The periodic brick, or (`OnLattice`) the lattice with external boundaries.
template <bool OnLattice>
struct MetavariablesImpl {
  static constexpr size_t volume_dim = 3;
  static constexpr bool local_time_stepping = false;
  using system = gp::System;
  using component_list = tmpl::list<Component<MetavariablesImpl>>;
  using set_active_grid =
      tmpl::conditional_t<OnLattice, tmpl::list<>, SetActiveGridForTest>;
  static std::vector<ElementId<3>> element_ids() {
    return OnLattice ? lattice_ids() : all_ids();
  }
  using const_global_cache_tags = tmpl::list<
      ::Tags::ConcreteTimeStepper<TimeStepper>, Ccz4::Tags::Kappa3,
      evolution::initial_data::Tags::InitialData, Ccz4::fd::Tags::Reconstructor,
      Ccz4::fd::Tags::KreissOligerEpsilon, Ccz4::fd::Tags::EvolveLapseAndShift,
      Ccz4::fd::Tags::ConstrainedEvolution, Ccz4::fd::Tags::EtaConstant,
      Ccz4::Tags::DampingFunctionKappa1, Ccz4::Tags::DampingFunctionKappa2,
      ::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>;
  struct SubcellOptions {
    static constexpr bool subcell_enabled = true;
    static constexpr bool subcell_enabled_at_external_boundary = OnLattice;
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
        tmpl::pair<evolution::initial_data::InitialData,
                   tmpl::list<Minkowski, KerrSchild>>,
        tmpl::pair<DenseTrigger, tmpl::list<>>,
        tmpl::pair<Event, tmpl::list<>>>;
  };
};

using Metavariables = MetavariablesImpl<false>;
using LatticeMetavariables = MetavariablesImpl<true>;
using component = Component<Metavariables>;
using lattice_component = Component<LatticeMetavariables>;
using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
using LatticeRuntimeSystem =
    ActionTesting::MockRuntimeSystem<LatticeMetavariables>;

template <typename Tag, typename Metavars>
const auto& get_tag(const ActionTesting::MockRuntimeSystem<Metavars>& runner,
                    const ElementId<3>& id) {
  return ActionTesting::get_databox_tag<Component<Metavars>, Tag>(runner, id);
}

constexpr size_t time_stepper_order = 2;
constexpr double kappa_1 = 0.001;
constexpr double eta = 1.0;
const Slab slab{0.0, 0.002};

template <typename Metavars>
tuples::tagged_tuple_from_typelist<
    Parallel::get_const_global_cache_tags<Metavars>>
make_cache(const DomainCreator<3>& creator,
           std::unique_ptr<evolution::initial_data::InitialData> initial_data,
           std::optional<std::vector<std::string>> only_dg_block_names) {
  tuples::tagged_tuple_from_typelist<
      Parallel::get_const_global_cache_tags<Metavars>>
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
      std::move(initial_data);
  const evolution::dg::subcell::SubcellOptions subcell_options_with_names{
      4.0,
      1_st,
      1.0e-3,
      1.0e-4,
      false,
      false,
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
      false,
      std::move(only_dg_block_names),
      ::fd::DerivativeOrder::Two,
      1,
      1,
      1};
  get<evolution::dg::subcell::Tags::SubcellOptions<3>>(cache) =
      evolution::dg::subcell::SubcellOptions{subcell_options_with_names,
                                             creator};
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

// Emplaces and initializes the elements, and checks their numerical
// methods.
template <typename Metavars>
void initialize_elements(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavars>*> runner,
    const DomainCreator<3>& creator,
    const std::function<bool(const ElementId<3>&)>& expect_dg) {
  const TimeDelta step = slab.duration();
  for (const auto& id : Metavars::element_ids()) {
    ActionTesting::emplace_array_component_and_initialize<Component<Metavars>>(
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
    for (size_t a = 0; a < tmpl::size<initialization_actions<Metavars>>::value;
         ++a) {
      ActionTesting::next_action<Component<Metavars>>(runner, id);
    }
    CAPTURE(id);
    CHECK(get_tag<evolution::dg::subcell::Tags::ActiveGrid>(*runner, id) ==
          (expect_dg(id) ? ActiveGrid::Dg : ActiveGrid::Subcell));
  }
}

template <typename Metavars>
size_t next_action(const ActionTesting::MockRuntimeSystem<Metavars>& runner,
                   const ElementId<3>& id) {
  return ActionTesting::get_next_action_index<Component<Metavars>>(runner, id);
}

template <typename Action>
constexpr size_t index_of = tmpl::index_of<evolve_actions, Action>::value;

// Runs the elements, in the order of `schedule` (0: forward, 1: reverse,
// each element until it blocks; 2: forward, one action per element and
// sweep), until `stop(id)` holds for all of them. Fails on a deadlock.
template <typename Metavars>
void run_until(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavars>*> runner,
    const size_t schedule,
    const std::function<bool(const ElementId<3>&)>& stop) {
  std::vector<ElementId<3>> order = Metavars::element_ids();
  if (schedule == 1) {
    std::reverse(order.begin(), order.end());
  }
  for (size_t sweep = 0; sweep < 100000; ++sweep) {
    bool progress = false;
    bool done = true;
    for (const auto& id : order) {
      while (not stop(id) and
             ActionTesting::next_action_if_ready<Component<Metavars>>(runner,
                                                                      id)) {
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
template <typename Metavars>
std::function<bool(const ElementId<3>&)> completed(
    const ActionTesting::MockRuntimeSystem<Metavars>& runner,
    const size_t steps) {
  const double target = static_cast<double>(steps) * slab.duration().value();
  return [&runner, target](const ElementId<3>& id) {
    return std::abs(get_tag<::Tags::Time>(runner, id) - target) < 1.0e-14 and
           next_action(runner, id) == 0;
  };
}

template <typename Metavars>
void check_inboxes_empty(
    const ActionTesting::MockRuntimeSystem<Metavars>& runner) {
  for (const auto& id : Metavars::element_ids()) {
    CAPTURE(id);
    CHECK(
        ActionTesting::get_inbox_tag<Component<Metavars>, AuxInbox>(runner, id)
            .messages.empty());
    CHECK(ActionTesting::get_inbox_tag<Component<Metavars>, PhysicalInbox>(
              runner, id)
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
template <typename Metavars>
void self_start(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavars>*> runner,
    const size_t schedule) {
  ActionTesting::set_phase(runner,
                           Parallel::Phase::InitializeTimeStepperHistory);
  run_until(runner, schedule, [&runner](const ElementId<3>& id) {
    return ActionTesting::get_terminate<Component<Metavars>>(*runner, id);
  });
  for (const auto& id : Metavars::element_ids()) {
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
      make_cache<Metavariables>(brick(), std::make_unique<Minkowski>(),
                                std::nullopt),
      tuples::TaggedTuple<domain::Tags::FunctionsOfTimeInitialize>{}};
  initialize_elements(make_not_null(&runner), brick(), uses_dg);
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
      make_cache<Metavariables>(brick(), std::make_unique<Minkowski>(),
                                std::nullopt),
      tuples::TaggedTuple<domain::Tags::FunctionsOfTimeInitialize>{}};
  initialize_elements(make_not_null(&runner), brick(), uses_dg);
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

// The logical coordinates of the ghost cells of the face in `direction`,
// without the ghost-zone coordinate helper: `ghost_zone_size` cells continuing
// the uniform cell-centered grid outside the face, ordered like the volume
// cells, at the tangential coordinates of the cells.
tnsr::I<DataVector, 3, Frame::ElementLogical> ghost_logical_coordinates(
    const Mesh<3>& mesh, const size_t ghost_zone_size,
    const Direction<3>& direction) {
  const size_t dim = direction.dimension();
  Index<3> extents = mesh.extents();
  extents[dim] = ghost_zone_size;
  tnsr::I<DataVector, 3, Frame::ElementLogical> coords{extents.product()};
  for (IndexIterator<3> index(extents); index; ++index) {
    for (size_t d = 0; d < 3; ++d) {
      const double spacing = 2.0 / static_cast<double>(mesh.extents(d));
      const auto i = static_cast<double>(index()[d]);
      if (d == dim) {
        const double depth =
            direction.side() == Side::Upper
                ? i + 0.5
                : static_cast<double>(ghost_zone_size) - i - 0.5;
        coords.get(d)[index.collapsed_index()] =
            direction.sign() * (1.0 + depth * spacing);
      } else {
        coords.get(d)[index.collapsed_index()] = -1.0 + (i + 0.5) * spacing;
      }
    }
  }
  return coords;
}

// Checks the ghost data of a finite-difference element of the lattice and
// returns the entries of its neighbors. With `external_entries`, there is an
// entry in each external direction with the full ghost-data layout: the
// Kerr-Schild fields at the ghost points (the elements are affine) at the
// current time, and default-initialized gradients (signaling NaN with
// SPECTRE_NAN_INIT, compared bitwise; otherwise uninitialized and not read).
// Without, there are no external entries.
GhostDataMap check_lattice_ghost_data(const LatticeRuntimeSystem& runner,
                                      const ElementId<3>& id,
                                      const bool external_entries) {
  CAPTURE(id);
  CAPTURE(external_entries);
  const auto& element = get_tag<domain::Tags::Element<3>>(runner, id);
  const auto& ghost_data =
      get_tag<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
          runner, id);
  GhostDataMap neighbor_entries{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    for (const auto& neighbor_id : neighbors.ids()) {
      const DirectionalId<3> key{direction, neighbor_id};
      REQUIRE(ghost_data.contains(key));
      neighbor_entries.insert(std::pair{key, ghost_data.at(key)});
    }
  }
  CHECK(ghost_data.size() ==
        neighbor_entries.size() +
            (external_entries ? element.external_boundaries().size() : 0));
  const Mesh<3>& subcell_mesh =
      get_tag<evolution::dg::subcell::Tags::Mesh<3>>(runner, id);
  const double time = get_tag<::Tags::Time>(runner, id);
  const size_t ghost_zone_size =
      Ccz4::fd::UnlimitedDeg4Prim{}.ghost_zone_size();
  for (const auto& direction : element.external_boundaries()) {
    CAPTURE(direction);
    const DirectionalId<3> key{direction, ElementId<3>::external_boundary_id()};
    if (not external_entries) {
      CHECK(not ghost_data.contains(key));
      continue;
    }
    REQUIRE(ghost_data.contains(key));
    const DataVector& data =
        ghost_data.at(key).neighbor_ghost_data_for_reconstruction();
    const size_t number_of_points =
        ghost_zone_size *
        subcell_mesh.extents().slice_away(direction.dimension()).product();
    REQUIRE(data.size() ==
            Vars::number_of_independent_components * number_of_points);
    const auto logical_coords =
        ghost_logical_coordinates(subcell_mesh, ghost_zone_size, direction);
    tnsr::I<DataVector, 3> coords{number_of_points};
    for (size_t d = 0; d < 3; ++d) {
      const double lower_bound =
          gsl::at(lattice_lower_corner, d) +
          (d == 0 ? static_cast<double>(id.block_id()) * element_lengths[0]
                  : 0.0);
      coords.get(d) = lower_bound + 0.5 * (logical_coords.get(d) + 1.0) *
                                        gsl::at(element_lengths, d);
    }
    const auto expected =
        kerr_schild().variables(coords, time, physical_tags{});
    const Vars ghost_vars{
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<double*>(data.data()), data.size()};
    tmpl::for_each<physical_tags>(
        [&ghost_vars, &expected]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          CAPTURE(db::tag_name<Tag>());
          CHECK_ITERABLE_APPROX(get<Tag>(ghost_vars), get<Tag>(expected));
        });
#ifdef SPECTRE_NAN_INIT
    const size_t physical_size =
        Variables<physical_tags>::number_of_independent_components *
        number_of_points;
    std::vector<std::uint64_t> gradient_bits(data.size() - physical_size);
    std::memcpy(gradient_bits.data(), data.data() + physical_size,
                gradient_bits.size() * sizeof(double));
    constexpr auto signaling_nan_bits = std::bit_cast<std::uint64_t>(
        std::numeric_limits<double>::signaling_NaN());
    CHECK(std::all_of(
        gradient_bits.begin(), gradient_bits.end(),
        [](const std::uint64_t bits) { return bits == signaling_nan_bits; }));
#endif  // SPECTRE_NAN_INIT
  }
  return neighbor_entries;
}

void check_same_entries(const GhostDataMap& expected,
                        const GhostDataMap& entries) {
  REQUIRE(entries.size() == expected.size());
  for (const auto& [key, entry] : expected) {
    CAPTURE(key);
    CHECK(entries.at(key) == entry);
  }
}

// The stored gradients of a finite-difference element are those of its
// physical fields and its current ghost data.
void check_finite_difference_gradients(const LatticeRuntimeSystem& runner,
                                       const ElementId<3>& id) {
  CAPTURE(id);
  const auto& vars = get_tag<variables_tag>(runner, id);
  const Mesh<3>& mesh =
      get_tag<evolution::dg::subcell::Tags::Mesh<3>>(runner, id);
  using deriv_tags =
      db::wrap_tags_in<::Tags::deriv, Ccz4::fd::System::gradients_tags,
                       tmpl::size_t<3>, Frame::Inertial>;
  Variables<deriv_tags> derivatives{mesh.number_of_grid_points()};
  Ccz4::fd::spacetime_derivatives(
      make_not_null(&derivatives), vars,
      get_tag<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
          runner, id),
      4, mesh,
      get_tag<evolution::dg::subcell::fd::Tags::
                  InverseJacobianLogicalToInertial<3>>(runner, id));
  gp::Tags::PotentialGradients::type expected{mesh.number_of_grid_points()};
  gp::finite_difference_potential_gradients(
      make_not_null(&expected),
      get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(vars),
      get<::Tags::deriv<Ccz4::Tags::ConformalMetric<DataVector, 3>,
                        tmpl::size_t<3>, Frame::Inertial>>(derivatives),
      get<::Tags::deriv<Ccz4::Tags::ConformalFactor<DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>(derivatives),
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>>(derivatives),
      get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(derivatives));
  CHECK(get<gp::Tags::PotentialGradients>(vars) == expected);
}

// On the lattice, through one step: the numerical methods selected at
// initialization, and the ghost data of the finite-difference elements at
// the external boundaries. Each exchange replaces all ghost data with the
// neighbors' entries; the external entries are then added from the boundary
// conditions, before the gradients for the DG neighbors are computed and
// before the finite-difference right-hand side and dissipation.
void test_external_boundaries() {
  using SetGhostData = Ccz4::fd::Actions::SetBoundaryConditionGhostData;
  constexpr size_t first_receive = index_of<
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<3, true>>;
  constexpr size_t second_receive = index_of<
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<3>>;
  CHECK(std::is_same_v<tmpl::at_c<evolve_actions, first_receive + 1>,
                       SetGhostData>);
  CHECK(std::is_same_v<tmpl::at_c<evolve_actions, first_receive + 2>,
                       gp::Actions::ComputeFiniteDifferenceGradients>);
  CHECK(std::is_same_v<tmpl::at_c<evolve_actions, second_receive + 1>,
                       SetGhostData>);
  constexpr size_t dg_boundary_corrections =
      index_of<evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          3, false>>;
  constexpr size_t dg_drain =
      index_of<evolution::dg::Actions::DrainAuxiliaryGhostData<3, false>>;
  constexpr size_t fd_dissipation =
      index_of<::Actions::MutateApply<Ccz4::fd::AddKreissOligerDissipation>>;

  LatticeRuntimeSystem runner{
      make_cache<LatticeMetavariables>(
          lattice(), std::make_unique<KerrSchild>(kerr_schild()),
          std::vector<std::string>{lattice().block_names()[3]}),
      tuples::TaggedTuple<domain::Tags::FunctionsOfTimeInitialize>{}};
  initialize_elements(make_not_null(&runner), lattice(), uses_dg_in_lattice);
  std::vector<ElementId<3>> fd_ids{};
  for (const auto& id : lattice_ids()) {
    CHECK(get_tag<domain::Tags::Element<3>>(runner, id)
              .external_boundaries()
              .size() == (id.block_id() == 0 or id.block_id() == 3 ? 5 : 4));
    if (not uses_dg_in_lattice(id)) {
      fd_ids.push_back(id);
    }
  }
  self_start(make_not_null(&runner), 0);

  const auto stop_at = [&runner](const size_t fd_index, const size_t dg_index) {
    return [&runner, fd_index, dg_index](const ElementId<3>& id) {
      return next_action(runner, id) ==
             (uses_dg_in_lattice(id) ? dg_index : fd_index);
    };
  };
  // The first exchange, the external ghost data and the gradients for the DG
  // neighbors.
  run_until(make_not_null(&runner), 0,
            stop_at(first_receive + 1, dg_boundary_corrections));
  for (const auto& id : fd_ids) {
    const GhostDataMap received = check_lattice_ghost_data(runner, id, false);
    ActionTesting::next_action<lattice_component>(make_not_null(&runner), id);
    check_same_entries(received, check_lattice_ghost_data(runner, id, true));
    ActionTesting::next_action<lattice_component>(make_not_null(&runner), id);
    check_finite_difference_gradients(runner, id);
  }
  // The second exchange and the external ghost data.
  run_until(make_not_null(&runner), 1, stop_at(second_receive + 1, dg_drain));
  for (const auto& id : fd_ids) {
    const GhostDataMap received = check_lattice_ghost_data(runner, id, false);
    ActionTesting::next_action<lattice_component>(make_not_null(&runner), id);
    check_same_entries(received, check_lattice_ghost_data(runner, id, true));
  }
  // The right-hand side leaves the ghost data for the dissipation.
  run_until(make_not_null(&runner), 2, stop_at(fd_dissipation, dg_drain));
  for (const auto& id : fd_ids) {
    check_lattice_ghost_data(runner, id, true);
  }
  run_until(make_not_null(&runner), 0, completed(runner, 1));
  check_inboxes_empty(runner);
  for (const auto& id : lattice_ids()) {
    CHECK(get_tag<::Tags::Time>(runner, id) == approx(slab.duration().value()));
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
  test_external_boundaries();
}
}  // namespace
