// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/Actions/TakeTimeStep.hpp"
#include "Evolution/DgSubcell/GetTciDecision.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/SetInterpolators.hpp"
#include "Evolution/DgSubcell/Tags/MethodOrder.hpp"
#include "Evolution/DgSubcell/Tags/ObserverCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/ObserverMesh.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ApplyFilter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/DetConformalSpatialMetricCompute.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceConstrainedEvolution.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceTracelessDtConformalMetric.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/GhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ResizeTimeDerivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SetInitialEta.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SetK0.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SoTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/TraceATildeCompute.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UnlimitedDeg4Prim.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryCorrections/UpwindGradientProjection.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Filter.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/FinalizeTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/FiniteDifferenceGradients.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Initialize.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/NeighborPackagedData.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/StepActions.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/SubcellStepActions.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "Evolution/Systems/Ccz4/SpectralFilter.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/LocalizedPerturbation.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/RandomizeVariables.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsOnFailure.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/LimitIncrease.hpp"
#include "Time/StepChoosers/Maximum.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepChoosers/StepToTimes.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Time/UpdateU.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond

struct EvolutionMetavars {
  // With DG-subcell, elements may use finite differences (fixed at
  // initialization by the subcell options; there is no troubled-cell
  // indicator) and every right-hand side exchanges ghost data twice. Without
  // it, only DG is used, with the single DG exchange.
  static constexpr bool use_dg_subcell = true;
  static constexpr size_t volume_dim = 3;
  using initial_data_list = Ccz4::Solutions::all_solutions;
  using initial_data_tag = evolution::initial_data::Tags::InitialData;
  using system = Ccz4::GradientProjection::System;
  using temporal_id = Tags::TimeStepId;
  using TimeStepperBase = TimeStepper;

  // For labeling the yaml option for RandomizeVariables
  struct RandomizeInitialData {};
  // For labeling the yaml option for LocalizedPerturbation
  struct PerturbInitialData {};

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;
  static constexpr bool use_dg_element_collection = false;

  using analytic_variables_tags = typename system::physical_tags;

  using analytic_compute = evolution::Tags::AnalyticSolutionsCompute<
      volume_dim, analytic_variables_tags, use_dg_subcell, initial_data_list>;
  using error_compute = Tags::ErrorsCompute<analytic_variables_tags>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_variables_tags>;
  // Only the physical fields are observable: the stored gradients are
  // scratch storage recomputed before every use.
  using observe_fields = tmpl::push_back<
      tmpl::append<
          typename system::physical_tags, error_tags,
          tmpl::list<::Ccz4::Tags::DetConformalSpatialMetric<DataVector>,
                     ::Ccz4::Tags::TraceATilde<DataVector>>,
          db::wrap_tags_in<::Tags::dt, typename system::physical_tags>,
          tmpl::conditional_t<
              use_dg_subcell,
              tmpl::list<
                  evolution::dg::subcell::Tags::TciStatusCompute<volume_dim>,
                  evolution::dg::subcell::Tags::MethodOrderCompute<volume_dim>>,
              tmpl::list<>>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
              volume_dim, Frame::ElementLogical>,
          ::Events::Tags::ObserverCoordinatesCompute<volume_dim,
                                                     Frame::ElementLogical>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<volume_dim,
                                                                   Frame::Grid>,
          ::Events::Tags::ObserverCoordinatesCompute<volume_dim, Frame::Grid>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
              volume_dim, Frame::Inertial>,
          ::Events::Tags::ObserverCoordinatesCompute<volume_dim,
                                                     Frame::Inertial>>>;
  using non_tensor_compute_tags = tmpl::list<
      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Tags::ObserverMeshCompute<volume_dim>,
              evolution::dg::subcell::Tags::ObserverInverseJacobianCompute<
                  volume_dim, Frame::ElementLogical, Frame::Inertial>,
              evolution::dg::subcell::Tags::
                  ObserverJacobianAndDetInvJacobianCompute<
                      volume_dim, Frame::ElementLogical, Frame::Inertial>>,
          tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                     ::Events::Tags::ObserverInverseJacobianCompute<
                         volume_dim, Frame::ElementLogical, Frame::Inertial>,
                     ::Events::Tags::ObserverJacobianCompute<
                         volume_dim, Frame::ElementLogical, Frame::Inertial>,
                     ::Events::Tags::ObserverDetInvJacobianCompute<
                         Frame::ElementLogical, Frame::Inertial>>>,
      analytic_compute, error_compute,
      ::Ccz4::fd::DetConformalSpatialMetricCompute,
      ::Ccz4::fd::TraceATildeCompute>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       Events::Completion,
                       dg::Events::field_observations<
                           volume_dim, observe_fields, non_tensor_compute_tags>,
                       Events::time_events<system>>>>,
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>,
        tmpl::pair<evolution::BoundaryCorrection,
                   tmpl::list<Ccz4::GradientProjection::BoundaryCorrections::
                                  UpwindGradientProjection>>,
        tmpl::pair<
            Ccz4::BoundaryConditions::BoundaryCondition,
            tmpl::list<
                Ccz4::GradientProjection::BoundaryConditions::DirichletAnalytic,
                domain::BoundaryConditions::Periodic<
                    Ccz4::BoundaryConditions::BoundaryCondition>>>,
        tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
        // Step choosers that need characteristic-speed compute tags (CFL)
        // are not offered. The slab-size error control measures the evolved
        // Variables, including the scratch gradients. Their time derivatives
        // are zero, so the Adams-Bashforth and Adams-Moulton
        // predictor-corrector error estimates (linear combinations of the
        // recorded time derivatives) vanish on them.
        tmpl::pair<
            StepChooser<StepChooserUse::LtsStep>,
            tmpl::list<StepChoosers::Constant, StepChoosers::LimitIncrease,
                       StepChoosers::Maximum>>,
        tmpl::pair<
            StepChooser<StepChooserUse::Slab>,
            tmpl::list<
                StepChoosers::Constant,
                StepChoosers::ErrorControl<StepChooserUse::Slab,
                                           typename system::variables_tag>,
                StepChoosers::LimitIncrease, StepChoosers::Maximum,
                StepChoosers::StepToTimes>>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>,
        // ApplyBoundaryCorrections applies a runtime boundary-correction
        // filter; this scheme has none, so only the no-op filter exists.
        tmpl::pair<
            Filters::runtime::Filter<volume_dim,
                                     typename system::variables_tag::tags_list>,
            tmpl::list<Filters::runtime::None<
                volume_dim, typename system::variables_tag::tags_list>>>>;
  };

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::at<typename factory_creation::factory_classes, Event>>;

  struct SubcellOptions {
    using evolved_vars_tags = typename system::variables_tag::tags_list;

    static constexpr bool subcell_enabled = use_dg_subcell;
    // Finite-difference elements at external boundaries take the ghost data
    // of these boundaries from the boundary conditions
    // (Ccz4::fd::BoundaryConditionGhostData).
    static constexpr bool subcell_enabled_at_external_boundary = true;

    // We send `ghost_zone_size` cell-centered grid points for variable
    // reconstruction, of which we need `ghost_zone_size-1` for reconstruction
    // to the internal side of the element face, and `ghost_zone_size` for
    // reconstruction to the external side of the element face.
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) {
      return db::get<Ccz4::fd::Tags::Reconstructor>(box).ghost_zone_size();
    }

    // The ghost data sent by DG elements: the physical fields with zero
    // gradients, as in the first exchange of the finite-difference elements
    // (see Ccz4::GradientProjection::subcell_step_actions).
    using GhostVariables = Ccz4::fd::GhostVariables;

    using DgComputeSubcellNeighborPackagedData =
        Ccz4::GradientProjection::NeighborPackagedData;
  };

  static_assert(not local_time_stepping and not use_dg_element_collection,
                "The CCZ4 gradient-projection step actions support only "
                "global time stepping on DG element arrays.");

  using step_actions =
      tmpl::conditional_t<use_dg_subcell,
                          Ccz4::GradientProjection::subcell_step_actions,
                          Ccz4::GradientProjection::step_actions>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::flatten<tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, TimeStepperBase>,
          evolution::dg::Initialization::Domain<EvolutionMetavars>,
          Initialization::TimeStepperHistory<EvolutionMetavars>>,
      Initialization::Actions::NonconservativeSystem<system>,
      // The subcell grid is set without initial data, which
      // SetInitialData sets on the active grid.
      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Actions::SetSubcellGrid<volume_dim,
                                                              system, true>,
              Ccz4::GradientProjection::Actions::CheckFiniteDifferenceOptions,
              Actions::MutateApply<evolution::dg::subcell::SetInterpolators<
                  volume_dim, Ccz4::fd::Tags::Reconstructor>>,
              Actions::MutateApply<Ccz4::fd::ResizeTimeDerivatives>>,
          tmpl::list<>>,
      Ccz4::GradientProjection::Actions::SetInitialData,
      ::Actions::RandomizeVariables<typename system::variables_tag,
                                    RandomizeInitialData>,
      ::Actions::LocalizedPerturbation<typename system::variables_tag,
                                       PerturbInitialData>,
      Actions::MutateApply<
          Ccz4::GradientProjection::EnforceAlgebraicConstraints>,
      Initialization::Actions::AddComputeTags<tmpl::push_back<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars,
                                                  local_time_stepping>,
          Ccz4::Tags::Kappa1Compute, Ccz4::Tags::Kappa2Compute>>,
      ::evolution::dg::Initialization::Mortars<volume_dim>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Initialization::Actions::AddSimpleTags<::Ccz4::fd::SetInitialEta,
                                             ::Ccz4::fd::SetK0>,
      Initialization::Actions::InitializeItems<
          evolution::dg::Initialization::SpectralFilters<
              volume_dim, typename system::variables_tag::tags_list>,
          Ccz4::GradientProjection::InitializeLocalOperators>,
      Parallel::Actions::TerminatePhase>>;

  using dg_element_array_component = DgElementArray<
      EvolutionMetavars,
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        initialization_actions>,

                 Parallel::PhaseActions<
                     Parallel::Phase::InitializeTimeStepperHistory,
                     SelfStart::self_start_procedure<step_actions, system>>,

                 Parallel::PhaseActions<
                     Parallel::Phase::Register,
                     tmpl::push_back<dg_registration_list,
                                     Parallel::Actions::TerminatePhase>>,

                 Parallel::PhaseActions<
                     Parallel::Phase::Restart,
                     tmpl::push_back<dg_registration_list,
                                     Parallel::Actions::TerminatePhase>>,

                 Parallel::PhaseActions<
                     Parallel::Phase::WriteCheckpoint,
                     tmpl::list<evolution::Actions::RunEventsAndTriggers<
                                    Triggers::WhenToCheck::AtCheckpoints>,
                                Parallel::Actions::TerminatePhase>>,

                 Parallel::PhaseActions<
                     Parallel::Phase::Evolve,
                     tmpl::list<evolution::Actions::RunEventsAndTriggers<
                                    Triggers::WhenToCheck::AtSlabs>,
                                Actions::ChangeSlabSize, step_actions,
                                Actions::MutateApply<AdvanceTime<>>,
                                PhaseControl::Actions::ExecutePhaseChange>>,
                 Parallel::PhaseActions<
                     Parallel::Phase::PostFailureCleanup,
                     tmpl::list<Actions::RunEventsOnFailure<Tags::Time>,
                                Parallel::Actions::TerminatePhase>>>>;

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array_component, dg_registration_list>>;
  };

  using component_list =
      tmpl::list<observers::Observer<EvolutionMetavars>,
                 observers::ObserverWriter<EvolutionMetavars>,
                 dg_element_array_component>;

  using const_global_cache_tags = tmpl::push_back<
      tmpl::conditional_t<use_dg_subcell,
                          tmpl::list<Ccz4::fd::Tags::Reconstructor,
                                     Ccz4::fd::Tags::KreissOligerEpsilon,
                                     Ccz4::fd::Tags::EvolveLapseAndShift,
                                     Ccz4::fd::Tags::ConstrainedEvolution>,
                          tmpl::list<>>,
      Ccz4::fd::Tags::EtaConstant, Ccz4::Tags::DampingFunctionKappa1,
      Ccz4::Tags::DampingFunctionKappa2, Ccz4::Tags::Kappa3, initial_data_tag,
      domain::Tags::ExternalBoundaryConditions<3>>;

  static constexpr Options::String help{
      "Evolve the second-order CCZ4 formulation of the Einstein equations "
      "with the discontinuous Galerkin upwind gradient-projection scheme, "
      "optionally with finite-difference subcells.\n\n"};

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
