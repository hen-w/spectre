// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/BoundaryConditions/Cartoon.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/SendAuxiliaryBoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedFields/Actions.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedFields/InitializeBoundaryEvolvedFields.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedFields/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/ProjectSpectralFilters.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "Evolution/Systems/SoScalarWave/UpdateAuxiliaryVariables.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Factory.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayCollection/DgElementCollection.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/RandomizeVariables.hpp"
#include "ParallelAlgorithms/Actions/SpectralFilter.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SoPlaneWave.hpp"
#include "PointwiseFunctions/InitialDataUtilities/NumericData.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/ChangeTimeStepperOrder.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/StepChoosers/ByBlock.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Time/UpdateU.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {

struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

template <size_t Dim>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = Dim;

  using initial_data_list = SoScalarWave::Solutions::all_solutions<Dim>;

  using system = SoScalarWave::System<Dim>;
  using temporal_id = Tags::TimeStepId;
  using TimeStepperBase = TimeStepper;

  // For labeling the yaml option for RandomizeVariables
  struct RandomizeInitialData {};

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;
  static constexpr bool use_dg_element_collection = false;

  // Fields compared against the analytic solution (so Error(...) is available
  // and observable). Phi is an auxiliary variable but is still compared. Listed
  // explicitly in the order the analytic solutions' `variables` overload
  // provides them ({Psi, Pi, Phi}); this is not `variables_tag`, which no
  // longer contains Phi.
  using analytic_solution_fields =
      tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                 SoScalarWave::Tags::Phi<volume_dim>>;
  using analytic_compute =
      evolution::Tags::AnalyticSolutionsCompute<Dim, analytic_solution_fields,
                                                false, initial_data_list>;
  using error_compute = Tags::ErrorsCompute<analytic_solution_fields>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_solution_fields>;

  using observe_fields = tmpl::push_back<
      tmpl::append<typename system::variables_tag::tags_list,
                   typename system::auxiliary_variables, error_tags>,
      domain::Tags::Coordinates<volume_dim, Frame::Grid>,
      domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;
  using non_tensor_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 ::Events::Tags::ObserverDetInvJacobianCompute<
                     Frame::ElementLogical, Frame::Inertial>,
                 analytic_compute, error_compute>;

  // The concrete boundary conditions the boundary-evolved-fields facility can
  // encounter on an external face, and the duplicate-free union of the
  // boundary-evolved field tags they declare, matching the resolution done in
  // `apply_boundary_conditions_on_all_external_faces`.
  using derived_boundary_conditions = tmpl::remove_if<
      SoScalarWave::BoundaryConditions::standard_boundary_conditions<
          volume_dim>,
      tmpl::or_<
          std::is_base_of<domain::BoundaryConditions::MarkAsCartoon, tmpl::_1>,
          std::is_base_of<domain::BoundaryConditions::MarkAsNone, tmpl::_1>,
          std::is_base_of<domain::BoundaryConditions::MarkAsPeriodic,
                          tmpl::_1>>>;
  using boundary_field_tags =
      evolution::dg::BoundaryEvolvedFields::boundary_evolved_field_tags<
          derived_boundary_conditions>;
  // The boundary-evolved field value map that must be snapshotted and reset at
  // each self-start order boundary. Built directly from `boundary_field_tags`
  // (the single source of truth) so it is type-identical to the tag the
  // facility's `InitializeBoundaryEvolvedFields` stores, and threaded into
  // `self_start_procedure` below -- no hand-maintained list on the System.
  using boundary_evolved_fields_self_start_vars =
      tmpl::list<evolution::dg::Tags::BoundaryEvolvedFieldsValues<
          volume_dim, boundary_field_tags>>;

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
        tmpl::pair<evolution::BoundaryCorrection,
                   SoScalarWave::BoundaryCorrections::
                       standard_boundary_corrections<volume_dim>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   tmpl::push_back<initial_data_list,
                                   evolution::initial_data::NumericData>>,
        tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>,
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
        tmpl::pair<
            SoScalarWave::BoundaryConditions::BoundaryCondition<volume_dim>,
            SoScalarWave::BoundaryConditions::standard_boundary_conditions<
                volume_dim>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   tmpl::push_back<StepChoosers::standard_step_choosers<system>,
                                   StepChoosers::ByBlock<volume_dim>>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::push_back<StepChoosers::standard_slab_choosers<
                                       system, local_time_stepping>,
                                   StepChoosers::ByBlock<volume_dim>>>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>,
        tmpl::pair<Filters::Filter<volume_dim,
                                   typename system::variables_tag::tags_list>,
                   Filters::all_filters<
                       volume_dim, typename system::variables_tag::tags_list>>>;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>>>>;

  using step_actions = tmpl::flatten<tmpl::list<
      Actions::MutateApply<SoScalarWave::UpdateAuxiliaryVariables<volume_dim>>,
      evolution::dg::Actions::SendAuxiliaryBoundaryData<
          volume_dim, system, local_time_stepping, use_dg_element_collection>,
      evolution::dg::Actions::ApplyAuxiliaryBoundaryCorrectionsToVariables<
          volume_dim, use_dg_element_collection>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping,
          use_dg_element_collection>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          volume_dim, use_dg_element_collection>,
      Actions::MutateApply<RecordTimeStepperData<system>>,
      evolution::dg::BoundaryEvolvedFields::RecordBoundaryEvolvedFields<
          volume_dim, boundary_field_tags>,
      evolution::Actions::RunEventsAndDenseTriggers<
          tmpl::list<::AlwaysReadyPostprocessor<
              evolution::dg::BoundaryEvolvedFields::
                  DenseOutputBoundaryEvolvedFields<volume_dim,
                                                   boundary_field_tags>>>>,
      Actions::MutateApply<UpdateU<system, local_time_stepping>>,
      evolution::dg::BoundaryEvolvedFields::UpdateBoundaryEvolvedFields<
          volume_dim, boundary_field_tags>,
      Actions::MutateApply<CleanHistory<system>>,
      evolution::dg::BoundaryEvolvedFields::CleanBoundaryEvolvedFieldsHistory<
          volume_dim, boundary_field_tags>,
      Actions::MutateApply<evolution::dg::CleanMortarHistory<volume_dim>>,
      dg::Actions::SpectralFilter>>;

  using const_global_cache_tags =
      tmpl::list<evolution::initial_data::Tags::InitialData>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, TimeStepperBase,
                                       false, local_time_stepping>,
          evolution::dg::Initialization::Domain<EvolutionMetavars>,
          Initialization::TimeStepperHistory<EvolutionMetavars>>,
      Initialization::Actions::NonconservativeSystem<system>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
      ::Actions::RandomizeVariables<typename system::variables_tag,
                                    RandomizeInitialData>,
      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars>>,
      ::evolution::dg::Initialization::Mortars<volume_dim>,
      evolution::dg::BoundaryEvolvedFields::InitializeBoundaryEvolvedFields<
          volume_dim, system, derived_boundary_conditions>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Initialization::Actions::InitializeItems<
          evolution::dg::Initialization::SpectralFilters<
              volume_dim, typename system::variables_tag::tags_list>>,
      Parallel::Actions::TerminatePhase>;

  using dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<Parallel::Phase::InitializeTimeStepperHistory,
                                 SelfStart::self_start_procedure<
                                     step_actions, system, std::type_identity_t,
                                     boundary_evolved_fields_self_start_vars>>,

          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<Parallel::Phase::Restart,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Parallel::Phase::WriteCheckpoint,
              tmpl::list<evolution::Actions::RunEventsAndTriggers<
                             Triggers::WhenToCheck::AtCheckpoints>,
                         Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::flatten<tmpl::list<
                  std::conditional_t<local_time_stepping,
                                     evolution::Actions::RunEventsAndTriggers<
                                         Triggers::WhenToCheck::AtSteps>,
                                     tmpl::list<>>,
                  evolution::Actions::RunEventsAndTriggers<
                      Triggers::WhenToCheck::AtSlabs>,
                  Actions::ChangeSlabSize,
                  evolution::dg::BoundaryEvolvedFields::
                      ChangeSlabSizeBoundaryEvolvedFields<volume_dim,
                                                          boundary_field_tags>,
                  step_actions, Actions::MutateApply<AdvanceTime<>>,
                  PhaseControl::Actions::ExecutePhaseChange>>>>>;

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array, dg_registration_list>>;
  };

  using component_list =
      tmpl::list<observers::Observer<EvolutionMetavars>,
                 observers::ObserverWriter<EvolutionMetavars>,
                 dg_element_array>;

  static constexpr Options::String help{
      "Evolve a second-order in space Scalar Wave in Dim spatial dimension "
      "using the local discontinuous Galerkin (LDG) method."};

  static constexpr auto default_phase_order = std::array<Parallel::Phase, 5>{
      Parallel::Phase::Initialization,
      Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
      Parallel::Phase::Evolve, Parallel::Phase::Exit};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
