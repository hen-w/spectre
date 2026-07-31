// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedFields/Actions.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedFields/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
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
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/UpdateU.hpp"
#include "Time/UpdateU.tpp"
#include "Time/VariableOrderAlgorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

class TimeStepper;

namespace {
// The interior source field whose boundary twin we evolve.
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using evolution::dg::Tags::BoundaryValue;

enum class Rate { Const, Linear };

// A dummy evolved var that drives the self-start machinery and supplies the
// integration order the boundary field facility syncs onto the face histories.
// Its dt is fixed to zero and no check reads its value.
struct VolumeVar : db::SimpleTag {
  using type = double;
};

using field_tags = tmpl::list<BoundaryValue<Psi>>;
template <size_t Dim>
using values_tag =
    evolution::dg::Tags::BoundaryEvolvedFieldsValues<Dim, field_tags>;
template <size_t Dim>
using dt_stash_tag =
    evolution::dg::Tags::BoundaryEvolvedFieldsDtStash<Dim, field_tags>;
template <size_t Dim>
using history_tag =
    evolution::dg::Tags::BoundaryEvolvedFieldsHistory<Dim, field_tags>;

using volume_history_tag = ::Tags::HistoryEvolvedVariables<VolumeVar>;

// The base rate lambda of the per-face-node ODE the mock derivative action
// implements: d(value)/dt = -lambda * multiplier (Rate::Const) or
// lambda * multiplier * value (Rate::Linear).
constexpr double rate_constant = 0.75;

// A per-face multiplier making each external face's boundary field time
// derivative distinct. The time step is the same for every face as we assume
// global time stepping.
template <size_t Dim>
double face_rate_multiplier(const Direction<Dim>& direction) {
  return 1.0 + 0.5 * static_cast<double>(direction.dimension()) +
         (direction.side() == Side::Upper ? 0.25 : 0.0);
}

// Writes the per-face dt-stash from the current per-face values, acting as
// the boundary condition's supplied boundary field time derivative
// in production.
template <size_t Dim, Rate TheRate>
struct MockComputeBoundaryDt {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<Tags::dt<VolumeVar>>(
        [](const gsl::not_null<double*> dt_volume_var) { *dt_volume_var = 0.; },
        make_not_null(&box));
    const auto& values = db::get<values_tag<Dim>>(box);
    db::mutate<dt_stash_tag<Dim>>(
        [&values](
            const gsl::not_null<typename dt_stash_tag<Dim>::type*> dt_stash) {
          for (auto& [direction, dt_variables] : *dt_stash) {
            const auto& value =
                get(get<BoundaryValue<Psi>>(values.at(direction)));
            auto& dt_value =
                get(get<Tags::dt<BoundaryValue<Psi>>>(dt_variables));
            const double multiplier = face_rate_multiplier(direction);
            if constexpr (TheRate == Rate::Const) {
              dt_value = -rate_constant * multiplier;
            } else {
              dt_value = rate_constant * multiplier * value;
            }
          }
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <size_t Dim>
struct MockSystem {
  static constexpr bool has_primitive_and_conservative_vars = false;
  using variables_tag = VolumeVar;
};

template <typename Metavariables>
struct Component;

template <size_t Dim, Rate TheRate>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  static constexpr bool local_time_stepping = false;
  static constexpr Rate rate = TheRate;
  using system = MockSystem<Dim>;
  using component_list = tmpl::list<Component<Metavariables>>;

  struct TemporalId {
    template <typename Tag>
    using step_prefix = Tags::dt<Tag>;
  };
  using temporal_id = TemporalId;
};

template <typename Metavariables>
struct Component {
  static constexpr size_t Dim = Metavariables::volume_dim;
  static constexpr Rate TheRate = Metavariables::rate;
  using metavariables = Metavariables;
  using system = MockSystem<Dim>;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags =
      tmpl::list<Tags::ConcreteTimeStepper<TimeStepper>,
                 Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>;
  using simple_tags =
      tmpl::list<VolumeVar, Tags::dt<VolumeVar>, volume_history_tag,
                 values_tag<Dim>, dt_stash_tag<Dim>, history_tag<Dim>,
                 Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
                 Tags::Time, Tags::StepNumberWithinSlab,
                 Tags::AdaptiveSteppingDiagnostics>;
  using compute_tags = time_stepper_ref_tags<TimeStepper>;

  using step_actions = tmpl::list<
      MockComputeBoundaryDt<Dim, TheRate>,
      Actions::MutateApply<RecordTimeStepperData<system>>,
      evolution::dg::BoundaryEvolvedFields::RecordBoundaryEvolvedFields<
          Dim, field_tags>,
      Actions::MutateApply<UpdateU<system>>,
      evolution::dg::BoundaryEvolvedFields::UpdateBoundaryEvolvedFields<
          Dim, field_tags>,
      Actions::MutateApply<CleanHistory<system>>,
      evolution::dg::BoundaryEvolvedFields::CleanBoundaryEvolvedFieldsHistory<
          Dim, field_tags>>;

  using self_start_actions =
      tmpl::flatten<tmpl::list<SelfStart::self_start_procedure<
          step_actions, system, std::type_identity_t,
          tmpl::list<values_tag<Dim>>>>>;
  using testing_actions = tmpl::flatten<
      tmpl::list<step_actions, Actions::MutateApply<::AdvanceTime<>>>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::InitializeTimeStepperHistory,
                             self_start_actions>,
      Parallel::PhaseActions<Parallel::Phase::Testing, testing_actions>>;
};

template <typename T>
struct is_initialize : std::false_type {};
template <typename System, template <typename> typename CacheTagPrefix,
          typename AdditionalVarsToSave>
struct is_initialize<SelfStart::Actions::Initialize<System, CacheTagPrefix,
                                                    AdditionalVarsToSave>>
    : std::true_type {};

// Run actions until one matching the `Stop` metalambda has executed; return
// whether that action jumped the algorithm somewhere other than the next
// action in the list.
template <typename Metavariables, typename ActionList, typename Stop>
bool run_past(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavariables>*>
        runner) {
  using component = Component<Metavariables>;
  for (;;) {
    bool done = false;
    const size_t current_action =
        ActionTesting::get_next_action_index<component>(*runner, 0);
    size_t action_to_check = current_action;
    tmpl::for_each<ActionList>([&action_to_check, &done](const auto action) {
      using Action = tmpl::type_from<decltype(action)>;
      if (action_to_check-- == 0) {
        done = tmpl::apply<Stop, Action>::value;
      }
    });
    ActionTesting::next_action<component>(runner, 0);
    if (done) {
      return current_action + 1 !=
             ActionTesting::get_next_action_index<component>(*runner, 0);
    }
  }
}

// Run the complete self-start phase (through its TerminatePhase) and switch
// to the stepping phase, as the phase machinery does in a real executable.
template <size_t Dim, Rate TheRate>
void run_self_start(
    const gsl::not_null<
        ActionTesting::MockRuntimeSystem<Metavariables<Dim, TheRate>>*>
        runner) {
  using component = Component<Metavariables<Dim, TheRate>>;
  ActionTesting::set_phase(runner,
                           Parallel::Phase::InitializeTimeStepperHistory);
  while (not ActionTesting::get_terminate<component>(*runner, 0)) {
    ActionTesting::next_action<component>(runner, 0);
  }
  ActionTesting::set_phase(runner, Parallel::Phase::Testing);
}

// Build a value map holding `node_values` on each given face.
template <size_t Dim>
typename values_tag<Dim>::type make_value_map(
    const std::vector<Direction<Dim>>& directions,
    const DataVector& node_values) {
  typename values_tag<Dim>::type value_map{};
  for (const auto& direction : directions) {
    Variables<field_tags> face_value{node_values.size()};
    get(get<BoundaryValue<Psi>>(face_value)) = node_values;
    value_map.insert({direction, std::move(face_value)});
  }
  return value_map;
}

// Allocate one face-sized dt-stash entry per face of `value_map`.
template <size_t Dim>
typename dt_stash_tag<Dim>::type make_dt_stash_map(
    const typename values_tag<Dim>::type& value_map) {
  typename dt_stash_tag<Dim>::type dt_stash{};
  for (const auto& [direction, face_values] : value_map) {
    dt_stash.insert({direction, typename dt_stash_tag<Dim>::type::mapped_type{
                                    face_values.number_of_grid_points()}});
  }
  return dt_stash;
}

template <size_t Dim>
typename history_tag<Dim>::type make_history_map(
    const std::vector<Direction<Dim>>& directions, const size_t order) {
  typename history_tag<Dim>::type histories{};
  for (const auto& direction : directions) {
    histories.insert(
        {direction, typename history_tag<Dim>::type::mapped_type{order}});
  }
  return histories;
}

// The exact solution of the mock ODE at `time`.
template <Rate TheRate>
double analytic_time_integration(const double initial_value,
                                 const double multiplier, const double time) {
  if constexpr (TheRate == Rate::Const) {
    return initial_value - rate_constant * multiplier * time;
  } else {
    return initial_value * exp(rate_constant * multiplier * time);
  }
}

// Emplace the mock component with the given per-face storage and a stepper
// needing `number_of_past_steps` pre-slabs, at slab number
// `-number_of_past_steps` (so self-start builds up to the stepper's order, as
// in Test_SelfStartActions). The volume history starts at
// `initial_integration_order` = stepper order - number_of_past_steps,
// mirroring `Initialization::TimeStepping` (1 for Adams-Bashforth, 2 for
// Adams-Moulton predictor-corrector).
template <size_t Dim, Rate TheRate>
void emplace(const gsl::not_null<
                 ActionTesting::MockRuntimeSystem<Metavariables<Dim, TheRate>>*>
                 runner,
             const Time& initial_time, const TimeDelta& time_step,
             const size_t number_of_past_steps,
             const size_t initial_integration_order,
             typename values_tag<Dim>::type values,
             typename dt_stash_tag<Dim>::type dt_stash,
             typename history_tag<Dim>::type histories) {
  ActionTesting::emplace_component_and_initialize<
      Component<Metavariables<Dim, TheRate>>>(
      runner, 0,
      // The leading `0., 0.` initialize VolumeVar and Tags::dt<VolumeVar>,
      // which are dummy values just to drive the time stepper so that the
      // boundary field facility can run in this test. Their values are not
      // monitored.
      {0., 0., typename volume_history_tag::type{initial_integration_order},
       std::move(values), std::move(dt_stash), std::move(histories),
       TimeStepId{},
       TimeStepId(time_step.is_positive(),
                  -static_cast<int64_t>(number_of_past_steps), initial_time),
       time_step, initial_time.value(), uint64_t{0},
       Tags::AdaptiveSteppingDiagnostics::type{}});
}

// Drive self-start plus one real step and require every face node to equal
// the analytic solution exactly (a constant derivative is integrated exactly
// at any order). The important test coverage is the Dim = 2, 3
// instantiations and the multi-node faces with random per-node data.
template <size_t Dim, Rate TheRate>
void test_analytic_time_integration(
    const gsl::not_null<std::mt19937*> generator, const size_t order) {
  CAPTURE(order);
  const double step = 0.1;
  const auto slab = Slab::with_duration_from_start(1., step);
  const TimeDelta time_step = slab.duration();
  const Time initial_time = slab.start();

  const size_t number_of_nodes = 3;
  const auto direction = Direction<Dim>::lower_xi();
  // Random per-node initial values
  std::uniform_real_distribution<double> distribution{0.1, 1.0};
  DataVector node_values{number_of_nodes};
  for (size_t i = 0; i < number_of_nodes; ++i) {
    node_values[i] = distribution(*generator);
  }
  const std::vector<Direction<Dim>> directions{direction};

  ActionTesting::MockRuntimeSystem<Metavariables<Dim, TheRate>> runner{
      {std::make_unique<TimeSteppers::AdamsBashforth>(order),
       EventsAndTriggers{},
       std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>{},
       VariableOrderAlgorithm{order}}};
  auto value_map = make_value_map<Dim>(directions, node_values);
  auto dt_stash_map = make_dt_stash_map<Dim>(value_map);
  emplace<Dim, TheRate>(make_not_null(&runner), initial_time, time_step,
                        order - 1, 1, std::move(value_map),
                        std::move(dt_stash_map),
                        make_history_map<Dim>(directions, 1));
  using component = Component<Metavariables<Dim, TheRate>>;
  // Run through the whole self-start phase...
  run_self_start<Dim, TheRate>(make_not_null(&runner));
  // ...and then one real step, up to and including the facility Update.
  run_past<
      Metavariables<Dim, TheRate>, typename component::testing_actions,
      std::is_same<tmpl::pin<evolution::dg::BoundaryEvolvedFields::
                                 UpdateBoundaryEvolvedFields<Dim, field_tags>>,
                   tmpl::_1>>(make_not_null(&runner));

  const double multiplier = face_rate_multiplier(direction);
  const auto& final_values =
      ActionTesting::get_databox_tag<component, values_tag<Dim>>(runner, 0);
  const auto& final_field =
      get(get<BoundaryValue<Psi>>(final_values.at(direction)));
  DataVector expected{number_of_nodes};
  for (size_t i = 0; i < number_of_nodes; ++i) {
    expected[i] =
        analytic_time_integration<TheRate>(node_values[i], multiplier, step);
  }
  CHECK_ITERABLE_APPROX(final_field, expected);
}

// A genuine multistep integration through self-start. Asserts (a) the initial
// snapshot equals the seeded map; (b) the value is restored to the snapshot at
// an order boundary after being integrated away from it mid-self-start;
// (c) the face-history order equals the volume-history order after self-start
// (order-sync); (d) history size grows (accumulate, not reset).
void test_multistep_self_start(const size_t order) {
  CAPTURE(order);
  constexpr size_t Dim = 1;
  constexpr Rate TheRate = Rate::Linear;
  using metavariables = Metavariables<Dim, TheRate>;
  using component = Component<metavariables>;

  const double step = 0.1;
  const auto slab = Slab::with_duration_from_start(1., step);
  const TimeDelta time_step = slab.duration();
  const Time initial_time = slab.start();
  const size_t number_of_nodes = 2;
  const auto direction = Direction<Dim>::lower_xi();
  const std::vector<Direction<Dim>> directions{direction};
  DataVector node_values{number_of_nodes};
  node_values[0] = 0.4;
  node_values[1] = 0.7;

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {std::make_unique<TimeSteppers::AdamsBashforth>(order),
       EventsAndTriggers{},
       std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>{},
       VariableOrderAlgorithm{order}}};
  auto value_map = make_value_map<Dim>(directions, node_values);
  auto dt_stash_map = make_dt_stash_map<Dim>(value_map);
  emplace<Dim, TheRate>(make_not_null(&runner), initial_time, time_step,
                        order - 1, 1, std::move(value_map),
                        std::move(dt_stash_map),
                        make_history_map<Dim>(directions, 1));
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::InitializeTimeStepperHistory);

  const auto value_at = [&runner, &direction]() -> DataVector {
    return get(get<BoundaryValue<Psi>>(
        ActionTesting::get_databox_tag<component, values_tag<Dim>>(runner, 0)
            .at(direction)));
  };
  const auto face_order = [&runner, &direction]() {
    return ActionTesting::get_databox_tag<component, history_tag<Dim>>(runner,
                                                                       0)
        .at(direction)
        .integration_order();
  };
  const auto volume_order = [&runner]() {
    return ActionTesting::get_databox_tag<component, volume_history_tag>(runner,
                                                                         0)
        .integration_order();
  };
  const auto face_history_size = [&runner, &direction]() {
    return ActionTesting::get_databox_tag<component, history_tag<Dim>>(runner,
                                                                       0)
        .at(direction)
        .size();
  };
  const auto volume_history_size = [&runner]() {
    return ActionTesting::get_databox_tag<component, volume_history_tag>(runner,
                                                                         0)
        .size();
  };

  {
    INFO("Initialize: the snapshot equals the seeded value map.");
    run_past<metavariables, typename component::self_start_actions,
             is_initialize<tmpl::_1>>(make_not_null(&runner));
    const auto& snapshot =
        get<0>(ActionTesting::get_databox_tag<
               component, SelfStart::Tags::InitialValue<values_tag<Dim>>>(
            runner, 0));
    // (a) the boundary value map is snapshotted (it is in vars_to_save).
    CHECK(get(get<BoundaryValue<Psi>>(snapshot.at(direction))) == node_values);
    // The history is not snapshotted; it must accumulate below.
    CHECK(face_history_size() == 0);
  }

  {
    INFO("First order boundary: the reset copy preserves the snapshot.");
    // Advance to the first CheckForCompletion. The value has not been evolved
    // yet, so this confirms the reset copy leaves the (still-initial) value
    // intact.
    run_past<metavariables, typename component::self_start_actions,
             tt::is_a<SelfStart::Actions::CheckForCompletion, tmpl::_1>>(
        make_not_null(&runner));
    // (b) the value still equals the snapshot after the reset copy.
    CHECK(value_at() == node_values);
  }

  {
    INFO("Mid-self-start: the value is integrated away from the snapshot.");
    // Take one self-start step so the boundary value is genuinely evolved away
    // from its initial condition (the linear rate gives a nonzero dt).
    run_past<metavariables, typename component::self_start_actions,
             std::is_same<
                 tmpl::pin<evolution::dg::BoundaryEvolvedFields::
                               UpdateBoundaryEvolvedFields<Dim, field_tags>>,
                 tmpl::_1>>(make_not_null(&runner));
    // The value has drifted: it is no longer the snapshot.
    CHECK(value_at()[0] != approx(node_values[0]));
  }

  // Run the remainder of the self-start procedure to completion.
  run_past<metavariables, typename component::self_start_actions,
           std::is_same<SelfStart::Actions::Cleanup, tmpl::_1>>(
      make_not_null(&runner));

  // (b) Having drifted mid-self-start, the value is restored to the
  // snapshot by the terminal order-boundary reset: the real evolution begins
  // from the initial condition, not the drifted self-start value.
  CHECK(value_at() == node_values);

  // (c) order-sync end state: the boundary history order tracks the volume
  // history order, both at the requested order after self-start.
  CHECK(face_order() == volume_order());
  CHECK(face_order() == order);
  // (d) the boundary history accumulated in lockstep with the volume history
  // (it was NOT reset at the order boundaries, unlike the value).
  CHECK(face_history_size() > 0);
  CHECK(face_history_size() == volume_history_size());
}

// The sensitive end-to-end accuracy check: measure the empirical convergence
// rate of the per-face integration through a complete self-start plus one
// real step (all substeps), by comparing the error at step h against h/2.
// The one-step error of an order-k stepper is O(h^(k+1)), so the measured
// rate must be order + 1. A wrong order-sync or value reset anywhere in
// self-start pollutes the step with lower-order error and degrades the rate.
// Runs forward and backward in time.
template <typename StepperBuilder>
void test_time_stepper_convergence(const std::string& stepper_label,
                                   const StepperBuilder& make_stepper,
                                   const size_t initial_integration_order,
                                   const size_t order, const double rate_margin,
                                   const bool forward_in_time) {
  CAPTURE(stepper_label);
  CAPTURE(order);
  CAPTURE(forward_in_time);
  constexpr size_t Dim = 1;
  constexpr Rate TheRate = Rate::Linear;
  // A signed `step` selects the time direction, mirroring the
  // Test_SelfStartActions convergence test.
  const auto error = [&make_stepper, initial_integration_order,
                      order](const double step) {
    const bool forward = step > 0.;
    const auto slab = forward ? Slab::with_duration_from_start(1., step)
                              : Slab::with_duration_to_end(1., -step);
    const TimeDelta time_step = (forward ? 1 : -1) * slab.duration();
    const Time initial_time = forward ? slab.start() : slab.end();
    const size_t number_of_nodes = 1;
    const auto direction = Direction<Dim>::lower_xi();
    const std::vector<Direction<Dim>> directions{direction};
    const double initial_value = 0.5;
    DataVector node_values{number_of_nodes, initial_value};

    auto stepper = make_stepper();
    const size_t number_of_past_steps = stepper->number_of_past_steps();
    const uint64_t number_of_substeps = stepper->number_of_substeps();
    ActionTesting::MockRuntimeSystem<Metavariables<Dim, TheRate>> runner{
        {std::move(stepper), EventsAndTriggers{},
         std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>{},
         VariableOrderAlgorithm{order}}};
    auto value_map = make_value_map<Dim>(directions, node_values);
    auto dt_stash_map = make_dt_stash_map<Dim>(value_map);
    emplace<Dim, TheRate>(
        make_not_null(&runner), initial_time, time_step, number_of_past_steps,
        initial_integration_order, std::move(value_map),
        std::move(dt_stash_map),
        make_history_map<Dim>(directions, initial_integration_order));
    run_self_start<Dim, TheRate>(make_not_null(&runner));
    // One real step is one facility Update per substep: a substep stepper
    // (e.g. Adams-Moulton predictor-corrector) passes through the
    // Record-Update-Clean actions once per substep, and only the final
    // substep's update yields the step's value. After every update, the face
    // and volume histories must have matching step and substep structure.
    for (uint64_t substep = 0; substep < number_of_substeps; ++substep) {
      run_past<Metavariables<Dim, TheRate>,
               typename Component<Metavariables<Dim, TheRate>>::testing_actions,
               std::is_same<
                   tmpl::pin<evolution::dg::BoundaryEvolvedFields::
                                 UpdateBoundaryEvolvedFields<Dim, field_tags>>,
                   tmpl::_1>>(make_not_null(&runner));

      const auto& face_history =
          ActionTesting::get_databox_tag<Component<Metavariables<Dim, TheRate>>,
                                         history_tag<Dim>>(runner, 0)
              .at(direction);
      const auto& volume_history =
          ActionTesting::get_databox_tag<Component<Metavariables<Dim, TheRate>>,
                                         volume_history_tag>(runner, 0);
      CHECK(face_history.integration_order() ==
            volume_history.integration_order());
      REQUIRE(face_history.size() == volume_history.size());
      for (size_t i = 0; i < volume_history.size(); ++i) {
        CHECK(face_history[i].time_step_id == volume_history[i].time_step_id);
        CHECK(face_history[i].value.has_value() ==
              volume_history[i].value.has_value());
      }

      const auto face_substeps = face_history.substeps();
      const auto volume_substeps = volume_history.substeps();
      REQUIRE(face_substeps.size() == volume_substeps.size());
      for (size_t i = 0; i < volume_substeps.size(); ++i) {
        CHECK(face_substeps[i].time_step_id == volume_substeps[i].time_step_id);
        CHECK(face_substeps[i].value.has_value() ==
              volume_substeps[i].value.has_value());
      }
    }

    const double multiplier = face_rate_multiplier(direction);
    const double value = get(get<BoundaryValue<Psi>>(
        ActionTesting::get_databox_tag<Component<Metavariables<Dim, TheRate>>,
                                       values_tag<Dim>>(runner, 0)
            .at(direction)))[0];
    return value -
           analytic_time_integration<TheRate>(initial_value, multiplier, step);
  };
  const double step = forward_in_time ? 0.1 : -0.1;
  const double convergence_rate =
      (log(fabs(error(step))) - log(fabs(error(0.5 * step)))) / log(2.);
  CHECK(convergence_rate == approx(order + 1).margin(rate_margin));
}

// A node shared by two external faces (a corner), whose faces have distinct
// time derivatives under the same global time step, must end up with two
// distinct values — one per face.
void test_per_face_corner() {
  INFO("Per-face corner: distinct value per face at a shared node.");
  constexpr size_t Dim = 2;
  constexpr Rate TheRate = Rate::Const;
  using metavariables = Metavariables<Dim, TheRate>;
  using component = Component<metavariables>;

  const double step = 0.1;
  const auto slab = Slab::with_duration_from_start(1., step);
  const TimeDelta time_step = slab.duration();
  const Time initial_time = slab.start();
  const size_t number_of_nodes = 3;
  const auto face_x = Direction<Dim>::lower_xi();
  const auto face_y = Direction<Dim>::lower_eta();
  const std::vector<Direction<Dim>> external_directions{face_x, face_y};
  const double initial_value = 0.5;
  DataVector node_values{number_of_nodes, initial_value};

  // The per-face rate multipliers must be distinct so the same initial value
  // evolves to two different values.
  REQUIRE(face_rate_multiplier(face_x) != face_rate_multiplier(face_y));

  const size_t order = 1;
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {std::make_unique<TimeSteppers::AdamsBashforth>(order),
       EventsAndTriggers{},
       std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>{},
       VariableOrderAlgorithm{order}}};
  auto value_map = make_value_map<Dim>(external_directions, node_values);
  auto dt_stash_map = make_dt_stash_map<Dim>(value_map);
  emplace<Dim, TheRate>(make_not_null(&runner), initial_time, time_step,
                        order - 1, 1, std::move(value_map),
                        std::move(dt_stash_map),
                        make_history_map<Dim>(external_directions, 1));

  // Order 1 has no self-start build-up; the self-start phase completes
  // immediately (priming the time-step id). Then one real step through the
  // facility Update.
  run_self_start<Dim, TheRate>(make_not_null(&runner));
  run_past<
      metavariables, typename component::testing_actions,
      std::is_same<tmpl::pin<evolution::dg::BoundaryEvolvedFields::
                                 UpdateBoundaryEvolvedFields<Dim, field_tags>>,
                   tmpl::_1>>(make_not_null(&runner));

  const auto& values =
      ActionTesting::get_databox_tag<component, values_tag<Dim>>(runner, 0);
  const double value_x = get(get<BoundaryValue<Psi>>(values.at(face_x)))[0];
  const double value_y = get(get<BoundaryValue<Psi>>(values.at(face_y)))[0];
  const double expected_x = analytic_time_integration<TheRate>(
      initial_value, face_rate_multiplier(face_x), step);
  const double expected_y = analytic_time_integration<TheRate>(
      initial_value, face_rate_multiplier(face_y), step);
  CHECK(value_x == approx(expected_x));
  CHECK(value_y == approx(expected_y));
  // There should be distinct values at the shared corner node.
  CHECK(value_x != approx(value_y));
}

// An element with no opting external faces (any interior element) carries
// empty per-face maps; the public Record/Update/Clean actions must pass
// through as exact no-ops.
void test_empty_maps_no_op() {
  INFO("Empty maps: the public actions no-op on a non-opting element");
  constexpr size_t Dim = 1;
  constexpr Rate TheRate = Rate::Const;
  using metavariables = Metavariables<Dim, TheRate>;
  using component = Component<metavariables>;

  const double step = 0.1;
  const auto slab = Slab::with_duration_from_start(1., step);
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {std::make_unique<TimeSteppers::AdamsBashforth>(1), EventsAndTriggers{},
       std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>{},
       VariableOrderAlgorithm{size_t{1}}}};
  emplace<Dim, TheRate>(make_not_null(&runner), slab.start(), slab.duration(),
                        0, 1, {}, {}, {});
  run_self_start<Dim, TheRate>(make_not_null(&runner));

  // One full pass through the facility Record, Update, and Clean actions.
  run_past<
      metavariables, typename component::testing_actions,
      std::is_same<
          tmpl::pin<evolution::dg::BoundaryEvolvedFields::
                        CleanBoundaryEvolvedFieldsHistory<Dim, field_tags>>,
          tmpl::_1>>(make_not_null(&runner));

  CHECK(ActionTesting::get_databox_tag<component, values_tag<Dim>>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<component, dt_stash_tag<Dim>>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<component, history_tag<Dim>>(runner, 0)
            .empty());
}

// Dense output: feed the face history self-consistent polynomial records
// of the stepper's order; the dense update to a requested time must return
// expected value to machine precision, mirroring the volume dense update.
void test_dense_output() {
  INFO("Dense output reproduces order-matched polynomial data exactly");
  constexpr size_t Dim = 1;
  const size_t order = 3;
  const size_t number_of_nodes = 2;
  const TimeSteppers::AdamsBashforth stepper(order);
  const auto direction = Direction<Dim>::lower_xi();

  // A degree-3 polynomial and its derivative; AdamsBashforth of order 3
  // reproduces it exactly (as in the AdamsBashforth reversal test).
  const auto p = [](const double t) {
    return 1. + t * (2. + t * (3. + t * 4.));
  };
  const auto dp = [](const double t) { return 2. + t * (6. + t * 12.); };

  typename values_tag<Dim>::type values{};
  values.insert({direction, Variables<field_tags>{number_of_nodes}});
  typename history_tag<Dim>::type histories{};
  histories.insert(
      {direction, typename history_tag<Dim>::type::mapped_type{order}});
  auto& history = histories.at(direction);

  const Slab slab(0., 1.);
  const auto add = [&history, &p, &dp](const Time& time) {
    Variables<field_tags> value{number_of_nodes};
    get(get<BoundaryValue<Psi>>(value)) =
        DataVector{number_of_nodes, p(time.value())};
    typename history_tag<Dim>::type::mapped_type::DerivVars deriv{
        number_of_nodes};
    get(get<Tags::dt<BoundaryValue<Psi>>>(deriv)) =
        DataVector{number_of_nodes, dp(time.value())};
    history.insert(TimeStepId(true, 0, time), std::move(value),
                   std::move(deriv));
  };
  add(slab.start());
  add(slab.start() + slab.duration() / 3);
  add(slab.start() + slab.duration() * 2 / 3);

  for (const double fraction : {5.0 / 6.0, 7.0 / 8.0}) {
    const double dense_time = slab.start().value() + fraction;
    auto dense_values = values;
    evolution::dg::BoundaryEvolvedFields::DenseOutputBoundaryEvolvedFields<
        Dim, field_tags>::apply(make_not_null(&dense_values), histories,
                                stepper, dense_time);
    CHECK_ITERABLE_APPROX(
        get(get<BoundaryValue<Psi>>(dense_values.at(direction))),
        (DataVector{number_of_nodes, p(dense_time)}));
  }
}

// ChangeSlabSize: after a slab-size change the stock machinery updates the
// volume history's latest time_step_id but not the boundary history (which is
// invisible to `get_all_history_tags`). The facility action must re-sync each
// per-face history's latest id to the volume history's.
template <typename Metavariables>
struct SlabComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags = tmpl::list<volume_history_tag, history_tag<1>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, tmpl::list<>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<evolution::dg::BoundaryEvolvedFields::
                         ChangeSlabSizeBoundaryEvolvedFields<1, field_tags>>>>;
};
struct SlabMetavars {
  static constexpr size_t volume_dim = 1;
  static constexpr bool local_time_stepping = false;
  using component_list = tmpl::list<SlabComponent<SlabMetavars>>;
};

void test_change_slab_size() {
  INFO(
      "ChangeSlabSize re-syncs the boundary history's latest id to the volume");
  using component = SlabComponent<SlabMetavars>;
  const auto direction = Direction<1>::lower_xi();
  const size_t number_of_nodes = 2;
  const size_t order = 2;
  const Slab old_slab(0., 1.);
  const Slab new_slab(0., 0.5);
  const TimeStepId old_id(true, 3, old_slab.start());
  const TimeStepId new_id(true, 3, new_slab.start());

  const auto run = [&](const TimeStepId& volume_back_id,
                       const TimeStepId& boundary_back_id) {
    typename volume_history_tag::type volume_history{order};
    volume_history.insert(volume_back_id, 0., 0.);
    typename history_tag<1>::type::mapped_type face_history{order};
    Variables<field_tags> value{number_of_nodes};
    get(get<BoundaryValue<Psi>>(value)) = DataVector{number_of_nodes, 1.};
    typename dt_stash_tag<1>::type::mapped_type deriv{number_of_nodes};
    get(get<Tags::dt<BoundaryValue<Psi>>>(deriv)) =
        DataVector{number_of_nodes, 0.};
    face_history.insert(boundary_back_id, std::move(value), std::move(deriv));
    typename history_tag<1>::type histories{};
    histories.insert({direction, std::move(face_history)});

    ActionTesting::MockRuntimeSystem<SlabMetavars> runner{{}};
    ActionTesting::emplace_component_and_initialize<component>(
        make_not_null(&runner), 0,
        {std::move(volume_history), std::move(histories)});
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
    return ActionTesting::get_databox_tag<component, history_tag<1>>(runner, 0)
        .at(direction)
        .back()
        .time_step_id;
  };

  // Slab changed: the boundary history's stale old-slab id is synced to the
  // new.
  CHECK(run(new_id, old_id) == new_id);
  // No change (ids already equal): the record is left untouched.
  CHECK(run(new_id, new_id) == new_id);

  const auto check_mismatched_histories_error =
      [&](const bool volume_history_is_empty) {
        typename volume_history_tag::type volume_history{order};
        if (not volume_history_is_empty) {
          volume_history.insert(new_id, 0., 0.);
        }

        typename history_tag<1>::type::mapped_type face_history{order};
        if (volume_history_is_empty) {
          Variables<field_tags> value{number_of_nodes};
          get(get<BoundaryValue<Psi>>(value)) = DataVector{number_of_nodes, 1.};
          typename dt_stash_tag<1>::type::mapped_type deriv{number_of_nodes};
          get(get<Tags::dt<BoundaryValue<Psi>>>(deriv)) =
              DataVector{number_of_nodes, 0.};
          face_history.insert(old_id, std::move(value), std::move(deriv));
        }
        typename history_tag<1>::type histories{};
        histories.insert({direction, std::move(face_history)});

        ActionTesting::MockRuntimeSystem<SlabMetavars> runner{{}};
        ActionTesting::emplace_component_and_initialize<component>(
            make_not_null(&runner), 0,
            {std::move(volume_history), std::move(histories)});
        ActionTesting::set_phase(make_not_null(&runner),
                                 Parallel::Phase::Testing);
        CHECK_THROWS_WITH(
            ActionTesting::next_action<component>(make_not_null(&runner), 0),
            Catch::Matchers::ContainsSubstring(
                volume_history_is_empty
                    ? "volume time-stepper history is empty"
                    : "volume time-stepper history is not empty"));
      };

  // Histories that are recorded and cleaned out of lockstep fail loudly in
  // both possible mismatched-emptiness states.
  check_mismatched_histories_error(true);
  check_mismatched_histories_error(false);
}

// The subcell guard, tested at both levels: directly on the
// `check_not_on_subcell` helper, and through a public production action
// (Record; Update and Clean carry the identical guard call).
template <typename Metavariables>
struct SubcellRecordComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags =
      tmpl::list<evolution::dg::subcell::Tags::ActiveGrid, Tags::TimeStepId,
                 values_tag<1>, dt_stash_tag<1>, history_tag<1>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, tmpl::list<>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<evolution::dg::BoundaryEvolvedFields::
                         RecordBoundaryEvolvedFields<1, field_tags>>>>;
};
struct SubcellMetavars {
  static constexpr size_t volume_dim = 1;
  static constexpr bool local_time_stepping = false;
  struct SubcellOptions {
    static constexpr bool subcell_enabled = true;
  };
  using component_list = tmpl::list<SubcellRecordComponent<SubcellMetavars>>;
};
struct NonSubcellMetavars {};

void test_subcell_guard() {
  INFO("Subcell guard");
  static_assert(evolution::dg::using_subcell_v<SubcellMetavars>);
  static_assert(not evolution::dg::using_subcell_v<NonSubcellMetavars>);

  using history_map_type = typename history_tag<1>::type;
  history_map_type empty_map{};
  history_map_type nonempty_map{};
  nonempty_map.insert(
      {Direction<1>::lower_xi(), typename history_map_type::mapped_type{1}});

  const auto make_box = [](const evolution::dg::subcell::ActiveGrid grid) {
    return db::create<
        db::AddSimpleTags<evolution::dg::subcell::Tags::ActiveGrid>>(grid);
  };
  const auto subcell_active_box =
      make_box(evolution::dg::subcell::ActiveGrid::Subcell);
  const auto dg_active_box = make_box(evolution::dg::subcell::ActiveGrid::Dg);

  // Fires: opted-in element (non-empty map) actually on the subcell grid.
  CHECK_THROWS_WITH(
      (evolution::dg::BoundaryEvolvedFields::detail::check_not_on_subcell<
          SubcellMetavars>(nonempty_map, subcell_active_box)),
      Catch::Matchers::ContainsSubstring("subcell"));
  // Does NOT fire: subcell grid but no opted-in boundary field (empty map).
  CHECK_NOTHROW(
      evolution::dg::BoundaryEvolvedFields::detail::check_not_on_subcell<
          SubcellMetavars>(empty_map, subcell_active_box));
  // Does NOT fire: opted-in element but on the DG grid.
  CHECK_NOTHROW(
      evolution::dg::BoundaryEvolvedFields::detail::check_not_on_subcell<
          SubcellMetavars>(nonempty_map, dg_active_box));
  // Compiles away (and never fires) for an executable without subcell, even
  // with a non-empty map. There is no ActiveGrid tag in this box.
  const auto pure_dg_box = db::create<db::AddSimpleTags<>>();
  CHECK_NOTHROW(
      evolution::dg::BoundaryEvolvedFields::detail::check_not_on_subcell<
          NonSubcellMetavars>(nonempty_map, pure_dg_box));
}

void test_subcell_guard_through_action() {
  INFO("The subcell guard fires when a public facility action runs on subcell");
  using component = SubcellRecordComponent<SubcellMetavars>;
  const auto direction = Direction<1>::lower_xi();
  const size_t number_of_nodes = 2;

  // A non-empty per-face history == an opted-in boundary element.
  typename history_tag<1>::type histories{};
  histories.insert(
      {direction, typename history_tag<1>::type::mapped_type{size_t{1}}});
  typename values_tag<1>::type values{};
  values.insert({direction, Variables<field_tags>{number_of_nodes}});
  typename dt_stash_tag<1>::type dt_stash{};
  dt_stash.insert({direction, typename dt_stash_tag<1>::type::mapped_type{
                                  number_of_nodes}});

  ActionTesting::MockRuntimeSystem<SubcellMetavars> runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(
      make_not_null(&runner), 0,
      {evolution::dg::subcell::ActiveGrid::Subcell, TimeStepId{},
       std::move(values), std::move(dt_stash), std::move(histories)});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  CHECK_THROWS_WITH(
      ActionTesting::next_action<component>(make_not_null(&runner), 0),
      Catch::Matchers::ContainsSubstring("subcell"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Dg.BoundaryEvolvedFields.Actions",
                  "[Unit][Evolution]") {
  register_classes_with_charm<TimeSteppers::AdamsBashforth,
                              TimeSteppers::AdamsMoultonPc<false>,
                              TimeSteppers::Rk3HesthavenSsp>();

  test_subcell_guard();
  test_subcell_guard_through_action();
  test_dense_output();
  test_change_slab_size();

  MAKE_GENERATOR(generator);

  test_analytic_time_integration<2, Rate::Const>(make_not_null(&generator), 2);
  test_analytic_time_integration<3, Rate::Const>(make_not_null(&generator), 4);
  // Order 1 (Euler) has no multistep self-start build-up, so it is checked only
  // for convergence, not in test_multistep_self_start (which needs order >= 2
  // to exercise the restore mechanism across order boundaries). Convergence is
  // checked both forward and backward in time.
  for (size_t order = 1; order < 5; ++order) {
    if (order >= 2) {
      test_multistep_self_start(order);
    }
    for (const bool forward_in_time : {true, false}) {
      test_time_stepper_convergence(
          "AdamsBashforth",
          [order]() {
            return std::make_unique<TimeSteppers::AdamsBashforth>(order);
          },
          1, order, 0.1, forward_in_time);
    }
  }
  // Test with a predictor-corrector stepper
  for (const size_t order : {size_t{2}, size_t{4}}) {
    for (const bool forward_in_time : {true, false}) {
      test_time_stepper_convergence(
          "AdamsMoultonPc",
          [order]() {
            return std::make_unique<TimeSteppers::AdamsMoultonPc<false>>(order);
          },
          2, order, 0.25, forward_in_time);
    }
  }

  // Test with a Runge-Kutta stepper
  for (const bool forward_in_time : {true, false}) {
    test_time_stepper_convergence(
        "Rk3HesthavenSsp",
        []() { return std::make_unique<TimeSteppers::Rk3HesthavenSsp>(); }, 3,
        3, 0.1, forward_in_time);
  }

  test_per_face_corner();
  test_empty_maps_no_op();
}
