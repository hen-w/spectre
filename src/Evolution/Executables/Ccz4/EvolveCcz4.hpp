// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/GetTciDecision.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/Actions/TakeTimeStep.hpp"
#include "Evolution/DgSubcell/SetInterpolators.hpp"
#include "Evolution/DgSubcell/Tags/MethodOrder.hpp"
#include "Evolution/DgSubcell/Tags/ObserverCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/ObserverMesh.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Ccz4/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ConstraintEnergyCompute.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/HamiltonianConstraintCompute.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/MomentumConstraintCompute.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SpatialZ4ConstraintUpCompute.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ApplyFilter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/DetConformalSpatialMetricCompute.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UnlimitedDeg4Prim.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceConstrainedEvolution.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceTracelessDerivConformalMetric.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceTracelessDtConformalMetric.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/GhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/NeighborPackagedData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/LdgTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ResizeTimeDerivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SetInitialEta.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SetK0.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SoTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/TraceATildeCompute.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UpdateAuxiliaryVariables.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UpdateAuxiliaryVariablesFd.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "Evolution/Systems/Ccz4/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Tags/Filter.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/LinearOperators/CgFilter.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/FilterAction.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/LocalizedPerturbation.hpp"
#include "ParallelAlgorithms/Actions/RandomizeVariables.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsOnFailure.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
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
  static constexpr bool use_dg_subcell = true;
  static constexpr size_t volume_dim = ::Ccz4::fd::System::volume_dim;
  using initial_data_list = Ccz4::Solutions::all_solutions;
  using initial_data_tag = evolution::initial_data::Tags::InitialData;
  using system = Ccz4::fd::System;
  using temporal_id = Tags::TimeStepId;
  using TimeStepperBase = TimeStepper;

  // For labeling the yaml option for RandomizeVariables
  struct RandomizeInitialData {};
  // For labeling the yaml option for LocalizedPerturbation
  struct PerturbInitialData {};

  struct FilterEvolvedVariables {};

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;
  static constexpr bool use_dg_element_collection = false;

  using analytic_variables_tags =
      tmpl::append<::Ccz4::fd::System::original_evolved_variables_tags,
                   ::Ccz4::fd::System::auxiliary_variables_tags>;

  using analytic_compute = evolution::Tags::AnalyticSolutionsCompute<
      volume_dim, analytic_variables_tags, use_dg_subcell, initial_data_list>;
  using error_compute = Tags::ErrorsCompute<analytic_variables_tags>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_variables_tags>;
  using observe_fields = tmpl::push_back<
      tmpl::append<
          typename system::variables_tag::tags_list, error_tags,
          tmpl::list<::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>,
                     ::Ccz4::Tags::DetConformalSpatialMetric<DataVector>,
                     ::Ccz4::Tags::TraceATilde<DataVector>,
                     gr::Tags::HamiltonianConstraint<DataVector>,
                     gr::Tags::MomentumConstraint<DataVector, 3,
                                                  Frame::Inertial>,
                     ::Ccz4::Tags::ConstraintEnergy<DataVector>>,
          typename db::add_tag_prefix<::Tags::dt,
                                      system::variables_tag>::tags_list,
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
      ::Ccz4::fd::SpatialZ4ConstraintUpCompute,
      ::Ccz4::fd::DetConformalSpatialMetricCompute,
      ::Ccz4::fd::TraceATildeCompute,
      ::Ccz4::fd::HamiltonianConstraintCompute,
      ::Ccz4::fd::MomentumConstraintCompute,
      ::Ccz4::fd::ConstraintEnergyCompute>;

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
                   Ccz4::BoundaryCorrections::standard_boundary_corrections<
                       volume_dim>>,
        tmpl::pair<Ccz4::BoundaryConditions::BoundaryCondition,
                   Ccz4::BoundaryConditions::standard_boundary_conditions>,
        tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   StepChoosers::standard_step_choosers<system>>,
        tmpl::pair<
            StepChooser<StepChooserUse::Slab>,
            StepChoosers::standard_slab_choosers<system, local_time_stepping>>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>,
        tmpl::pair<Filters::Filter,
                   tmpl::list<Filters::Exponential<volume_dim>,
                              Filters::CgFilter<volume_dim>,
                              Ccz4::TensorYlmFilter>>>;
  };

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::at<typename factory_creation::factory_classes, Event>>;

  struct SubcellOptions {
    using evolved_vars_tags = typename system::variables_tag::tags_list;

    static constexpr bool subcell_enabled = use_dg_subcell;
    static constexpr bool subcell_enabled_at_external_boundary = true;

    // We send `ghost_zone_size` cell-centered grid points for variable
    // reconstruction, of which we need `ghost_zone_size-1` for reconstruction
    // to the internal side of the element face, and `ghost_zone_size` for
    // reconstruction to the external side of the element face.
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) {
      // probably should hard code this as a varibale in Ccz4::fd::System
      return db::get<Ccz4::fd::Tags::Reconstructor>(box).ghost_zone_size();
    }

    using GhostVariables = Ccz4::fd::GhostVariables;
    using GhostVariablesPhysical = Ccz4::fd::GhostVariablesPhysical;

    using DgComputeSubcellNeighborPackagedData =
        Ccz4::fd::NeighborPackagedData;
    using DgComputeSubcellNeighborAuxPackagedData =
        Ccz4::fd::NeighborPackagedDataImpl<true>;
  };

  using events_and_dense_triggers_subcell_postprocessors = tmpl::list<>;

  using dg_step_actions = tmpl::flatten<tmpl::list<
      Actions::MutateApply<::Ccz4::fd::EnforceConstrainedEvolution>,
      Actions::MutateApply<Ccz4::fd::UpdateAuxiliaryVariables>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping,
          use_dg_element_collection, true>,
      evolution::dg::Actions::ApplyAuxiliaryBoundaryCorrectionsToVariables<
          volume_dim, use_dg_element_collection>,
      Actions::MutateApply<::Ccz4::fd::EnforceTracelessDerivConformalMetric>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping,
          use_dg_element_collection>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          volume_dim, use_dg_element_collection>,
      Actions::MutateApply<::Ccz4::fd::EnforceTracelessDtConformalMetric>,
      Actions::MutateApply<RecordTimeStepperData<system>>,
      evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
      Actions::MutateApply<UpdateU<system, local_time_stepping>>,
      Actions::MutateApply<CleanHistory<system>>,
      dg::Actions::Filter<
          FilterEvolvedVariables,
          tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                     ::Ccz4::Tags::ConformalFactor<DataVector>,
                     ::Ccz4::Tags::ATilde<DataVector, 3>,
                     gr::Tags::TraceExtrinsicCurvature<DataVector>,
                     ::Ccz4::Tags::Theta<DataVector>,
                     ::Ccz4::Tags::GammaHat<DataVector, 3>,
                     gr::Tags::Lapse<DataVector>,
                     gr::Tags::Shift<DataVector, 3>,
                     ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>>>;

  using dg_subcell_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::subcell::Actions::SelectNumericalMethod,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
      Actions::MutateApply<::Ccz4::fd::EnforceConstrainedEvolution>,
      Actions::MutateApply<Ccz4::fd::UpdateAuxiliaryVariables>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping,
          use_dg_element_collection, true>,
      evolution::dg::Actions::ApplyAuxiliaryBoundaryCorrectionsToVariables<
          volume_dim, use_dg_element_collection>,
      Actions::MutateApply<::Ccz4::fd::EnforceTracelessDerivConformalMetric>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping,
          use_dg_element_collection>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          volume_dim, use_dg_element_collection>,
      Actions::MutateApply<::Ccz4::fd::EnforceTracelessDtConformalMetric>,
      Actions::MutateApply<RecordTimeStepperData<system>>,
      evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
      Actions::MutateApply<UpdateU<system, local_time_stepping>>,
      Actions::MutateApply<CleanHistory<system>>,
      dg::Actions::Filter<
          FilterEvolvedVariables,
          tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                     ::Ccz4::Tags::ConformalFactor<DataVector>,
                     ::Ccz4::Tags::ATilde<DataVector, 3>,
                     gr::Tags::TraceExtrinsicCurvature<DataVector>,
                     ::Ccz4::Tags::Theta<DataVector>,
                     ::Ccz4::Tags::GammaHat<DataVector, 3>,
                     gr::Tags::Lapse<DataVector>,
                     gr::Tags::Shift<DataVector, 3>,
                     ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>,
      Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
      Actions::MutateApply<::Ccz4::fd::EnforceConstrainedEvolution>,

      // -- Round 1: exchange evolved variables via Inbox<true> --
      evolution::dg::subcell::Actions::SendDataForReconstruction<
          volume_dim, SubcellOptions::GhostVariables,
          use_dg_element_collection, true>,
      evolution::dg::subcell::Actions::ReceiveAndSendDataForReconstruction<
          volume_dim, SubcellOptions::GhostVariables,
          use_dg_element_collection, true>,
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<
          volume_dim, true>,

      // -- Between rounds: compute FieldA/B/D/P from FD derivatives --
      Ccz4::fd::UpdateAuxiliaryVariablesFd,

      // -- Round 2: send evolved vars + FieldA/B/D/P via Inbox<false> --
      evolution::dg::subcell::Actions::SendDataForReconstruction<
          volume_dim, SubcellOptions::GhostVariablesPhysical,
          use_dg_element_collection>,
      evolution::dg::subcell::Actions::ReceiveAndSendDataForReconstruction<
          volume_dim, SubcellOptions::GhostVariablesPhysical,
          use_dg_element_collection>,
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<volume_dim>,

      Actions::Label<
          evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback>,

      // subcell actions
      evolution::dg::subcell::fd::Actions::TakeTimeStep<
          Ccz4::fd::SoTimeDerivative>,
      Actions::MutateApply<RecordTimeStepperData<system>>,
      evolution::Actions::RunEventsAndDenseTriggers<
          events_and_dense_triggers_subcell_postprocessors>,
      Actions::MutateApply<UpdateU<system, local_time_stepping>>,
      Actions::MutateApply<::Ccz4::fd::ApplyFilter>,
      Actions::MutateApply<CleanHistory<system>>,
      Actions::Label<evolution::dg::subcell::Actions::Labels::EndOfSolvers>>>;

  using step_actions =
      tmpl::conditional_t<use_dg_subcell, dg_subcell_step_actions,
                          dg_step_actions>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::flatten<tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, TimeStepperBase>,
          evolution::dg::Initialization::Domain<EvolutionMetavars>,
          dg::Actions::InitializeFilters<FilterEvolvedVariables>,
          Initialization::TimeStepperHistory<EvolutionMetavars>>,
      Initialization::Actions::NonconservativeSystem<system>,
      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Actions::SetSubcellGrid<volume_dim,
                                                              system, false>,
              Actions::MutateApply<evolution::dg::subcell::SetInterpolators<
                  volume_dim, Ccz4::fd::Tags::Reconstructor>>,
              Actions::MutateApply<Ccz4::fd::ResizeTimeDerivatives>>,
          tmpl::list<evolution::Initialization::Actions::SetVariables<
              domain::Tags::Coordinates<volume_dim, Frame::ElementLogical>>>>,
      ::Actions::RandomizeVariables<typename system::variables_tag,
                                    RandomizeInitialData>,
      ::Actions::LocalizedPerturbation<typename system::variables_tag,
                                       PerturbInitialData>,
      Initialization::Actions::AddComputeTags<tmpl::push_back<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars,
                                                  local_time_stepping>,
          Ccz4::Tags::Kappa1Compute, Ccz4::Tags::Kappa2Compute>>,
      ::evolution::dg::Initialization::Mortars<volume_dim>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Initialization::Actions::AddSimpleTags<
          ::Ccz4::fd::SetInitialEta, ::Ccz4::fd::SetK0>,
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
                                     Ccz4::fd::Tags::KreissOligerEpsilon>,
                          tmpl::list<>>,
      Ccz4::fd::Tags::EvolveLapseAndShift, Ccz4::fd::Tags::ConstrainedEvolution,
      Ccz4::fd::Tags::EtaConstant, Ccz4::Tags::DampingFunctionKappa1,
      Ccz4::Tags::DampingFunctionKappa2, Ccz4::Tags::Kappa3, initial_data_tag,
      domain::Tags::ExternalBoundaryConditions<3>>;

  static constexpr Options::String help{
      "Evolve the second-order Ccz4 formulation of the Einstein Field "
      "Equations\n\n"};

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
