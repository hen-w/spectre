// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

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

template <typename InterpolationTargetTags>
struct EvolutionMetavars;

template <typename... InterpolationTargetTags>
struct EvolutionMetavars<tmpl::list<InterpolationTargetTags...>> {
  static constexpr bool use_dg_subcell = true;
  static constexpr size_t volume_dim = 3;
  using initial_data_list = Ccz4::fd::AllSolutions;
  using initial_data_tag = evolution::initial_data::Tags::InitialData;
  using system = Ccz4::fd::System;
  using temporal_id = Tags::TimeStepId;
  using TimeStepperBase = TimeStepper;

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;
  static constexpr bool use_dg_element_collection = false;

  using analytic_variables_tags = typename system::variables_tag::tags_list;

  using interpolator_source_vars =
      tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
          typename InterpolationTargetTags::vars_to_interpolate_to_target...>>>;

  using interpolation_target_tags = tmpl::list<InterpolationTargetTags...>;

  using analytic_compute = evolution::Tags::AnalyticSolutionsCompute<
      volume_dim, analytic_variables_tags, use_dg_subcell, initial_data_list>;
  using error_compute = Tags::ErrorsCompute<analytic_variables_tags>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_variables_tags>;
  using observe_fields = tmpl::push_back<tmpl::append<
      typename system::variables_tag::tags_list, error_tags,
      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Tags::TciStatusCompute<volume_dim>,
              evolution::dg::subcell::Tags::MethodOrderCompute<volume_dim>>,
          tmpl::list<>>>>;
  using non_tensor_compute_tags = tmpl::list<analytic_compute, error_compute>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                Events::Completion,
                dg::Events::field_observations<volume_dim, observe_fields,
                                               non_tensor_compute_tags>,
                Events::time_events<system>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, InterpolationTargetTags, interpolator_source_vars>...>>>,
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>,
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
                                         Triggers::time_triggers>>>;
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
  };

  using events_and_dense_triggers_subcell_postprocessors = tmpl::list<>;

  using dg_step_actions = tmpl::flatten<tmpl::list<>>;

  using dg_subcell_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::subcell::Actions::SelectNumericalMethod,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
      Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
      evolution::dg::subcell::Actions::SendDataForReconstruction<
          volume_dim, GhostVariables, local_time_stepping,
          use_dg_element_collection>,
      evolution::dg::subcell::Actions::ReceiveAndSendDataForReconstruction<
          volume_dim, GhostVariables, local_time_stepping,
          use_dg_element_collection>,
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<volume_dim>,
      // the following action should never happen cuz we are only doing FD
      Actions::Label<
          evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback>,
      evolution::dg::subcell::fd::Actions::TakeTimeStep<
          Ccz4::fd::SoTimeDerivative>,
      Actions::RecordTimeStepperData<system>,
      evolution::Actions::RunEventsAndDenseTriggers<
          events_and_dense_triggers_subcell_postprocessors>,
      Actions::UpdateU<system>,
      Actions::CleanHistory<system, local_time_stepping>,
      Actions::Label<evolution::dg::subcell::Actions::Labels::EndOfSolvers>>>;

  using step_actions =
      tmpl::conditional_t<use_dg_subcell, dg_subcell_step_actions,
                          dg_step_actions>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions =
      tmpl::flatten <
      tmpl::list<
          Initialization::Actions::InitializeItems<
              Initialization::TimeStepping<EvolutionMetavars, TimeStepperBase>,
              evolution::dg::Initialization::Domain<EvolutionMetavars>,
              Initialization::TimeStepperHistory<EvolutionMetavars>>,
          Initialization::Actions::AddSimpleTags<::Ccz4::Tags::Eta<DataVector>,
                                                 ::Ccz4::Tags::K0<DataVector>>,
          Actions::MutateApply<::Ccz4::fd::SetInitialEta>,
          Actions::MutateApply<::Ccz4::fd::SetInitialK0>,
          Initialization::Actions::NonconservativeSystem<system>,

          tmpl::conditional_t<
              use_dg_subcell,
              tmpl::list<
                  evolution::dg::subcell::Actions::SetSubcellGrid<
                      volume_dim, system, false>,
                  Actions::MutateApply<evolution::dg::subcell::SetInterpolators<
                      volume_dim, Ccz4::fd::Tags::Reconstructor>>>,

              Initialization::Actions::AddComputeTags<
                  StepChoosers::step_chooser_compute_tags<EvolutionMetavars,
                                                          local_time_stepping>>,
              ::evolution::dg::Initialization::Mortars<volume_dim, system>,
              evolution::Actions::InitializeRunEventsAndDenseTriggers,
              intrp::Actions::ElementInitInterpPoints<
                  volume_dim, interpolation_target_tags>,
              Parallel::Actions::TerminatePhase>>;

  using dg_element_array_component = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
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
              tmpl::list<
                  evolution::Actions::RunEventsAndTriggers<local_time_stepping>,
                  Actions::ChangeSlabSize, step_actions, Actions::AdvanceTime,
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

  using component_list = tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      intrp::InterpolationTarget<EvolutionMetavars, InterpolationTargetTags>...,
      dg_element_array_component>;

  using const_global_cache_tags = tmpl::push_back<
      tmpl::conditional_t<use_dg_subcell,
                          tmpl::list<Ccz4::fd::Tags::Reconstructor>,
                          tmpl::list<>>,
      initial_data_tag,
      /* how is this specified? */
      Ccz4::fd::Tags::ConstraintDampingParameter>;

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
