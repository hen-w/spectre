// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BoundaryVariables.hpp"
#include "Domain/BoundaryVariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedVariables.hpp"
#include "Evolution/Initialization/Evolution.hpp"
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
#include "Time/History.hpp"
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
// This is the first test anywhere that time-steps a `BoundaryVariables`
// container through the STOCK time-stepper machinery (RecordTimeStepperData,
// UpdateU, CleanHistory, self_start_procedure), driving it entirely as a
// list-valued `variables_tag` entry alongside a volume variable.  The mock
// `FillTimeDerivatives` action plays the role of the boundary-condition
// coupling layer: it fills the volume dt and every per-face boundary dt from
// closed-form functions of the substep time.  No boundary conditions, domain,
// or DG machinery are required.

constexpr size_t Dim = 2;

// The interior source field.  Its boundary twin is what we time-integrate.
struct VolumeVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

namespace Tags_ns = ::evolution::dg::Tags;

// The volume-variables entry and the boundary-variables entry of the mock
// split system's list-valued `variables_tag`.
using volume_vars = tmpl::list<VolumeVar>;
using boundary_vars = tmpl::list<Tags_ns::BoundaryValue<VolumeVar>>;
using volume_tag = ::Tags::Variables<volume_vars>;
using boundary_tag = ::Tags::BoundaryVariables<Dim, boundary_vars>;

using volume_history_tag = ::Tags::HistoryEvolvedVariables<volume_tag>;
using boundary_history_tag = ::Tags::HistoryEvolvedVariables<boundary_tag>;

// The dt-prefixed variables tags wrap each contained tag in `Tags::dt`, so the
// dt of a `Tags::Variables`/`Tags::BoundaryVariables` entry is another
// `Tags::Variables`/`Tags::BoundaryVariables` over the dt-wrapped tags.
using dt_volume_tag = db::add_tag_prefix<::Tags::dt, volume_tag>;
using dt_boundary_tag = db::add_tag_prefix<::Tags::dt, boundary_tag>;

// The two opting faces, deliberately of different node counts.
const Direction<Dim>& face_xi() {
  static const Direction<Dim> direction = Direction<Dim>::lower_xi();
  return direction;
}
const Direction<Dim>& face_eta() {
  static const Direction<Dim> direction = Direction<Dim>::lower_eta();
  return direction;
}
constexpr size_t face_xi_points = 3;
constexpr size_t face_eta_points = 2;

// Genuine linear ODEs with closed-form solutions.  Each component satisfies
// y' = lambda * y, which integrates to y(t) = y0 * exp(lambda * (t - t0)).  A
// value-dependent right-hand side (rather than a pure function of t, which
// would let a Runge-Kutta stepper superconverge on the resulting quadrature)
// gives the textbook global convergence order for every stepper family.  The
// volume and boundary rates differ, and the boundary rate is
// direction-dependent so the two faces follow different closed forms.
constexpr double volume_rate = 0.8;

double volume_solution(const double initial_value, const double initial_time,
                       const double time) {
  return initial_value * exp(volume_rate * (time - initial_time));
}

double face_rate(const Direction<Dim>& direction) {
  return 0.6 + 0.35 * static_cast<double>(direction.dimension()) +
         (direction.side() == Side::Upper ? 0.2 : 0.0);
}
double boundary_solution(const Direction<Dim>& direction,
                         const double initial_value, const double initial_time,
                         const double time) {
  return initial_value * exp(face_rate(direction) * (time - initial_time));
}

// Mock coupling action: fills the volume dt and every per-face boundary dt from
// the linear-ODE right-hand side using the CURRENT values (standing in for the
// boundary condition's supplied boundary field time derivative in production).
struct FillTimeDerivatives {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& volume = db::get<volume_tag>(box);
    const auto& boundary = db::get<boundary_tag>(box);
    db::mutate<dt_volume_tag, dt_boundary_tag>(
        [&volume, &boundary](
            const gsl::not_null<typename dt_volume_tag::type*> dt_volume,
            const gsl::not_null<typename dt_boundary_tag::type*> dt_boundary) {
          get(get<::Tags::dt<VolumeVar>>(*dt_volume)) =
              volume_rate * get(get<VolumeVar>(volume));
          for (auto& [direction, dt_variables] : dt_boundary->variables()) {
            get(get<::Tags::dt<Tags_ns::BoundaryValue<VolumeVar>>>(
                dt_variables)) =
                face_rate(direction) *
                get(get<Tags_ns::BoundaryValue<VolumeVar>>(
                    boundary.variables().at(direction)));
          }
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// The mock split system: a list-valued `variables_tag` with a volume entry and
// a boundary entry, exactly as SecondOrderScalarWave declares.
struct MockSystem {
  static constexpr bool has_primitive_and_conservative_vars = false;
  using variables_tag = tmpl::list<volume_tag, boundary_tag>;
};

// Simple action wrapping the production `TimeStepperHistory` initialization
// mutator, so it can be applied once through the ActionTesting harness (it
// reads `ConcreteTimeStepper` from the cache and each variables entry from the
// box, and sets each history's starting integration order per stepper family).
struct InitializeTimeStepperHistory {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    db::mutate_apply<Initialization::TimeStepperHistory<
        typename Metavariables::system>>(make_not_null(&box));
  }
};

template <typename Metavariables>
struct Component;

struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using system = MockSystem;
  using component_list = tmpl::list<Component<Metavariables>>;

  struct TemporalId {
    template <typename Tag>
    using step_prefix = ::Tags::dt<Tag>;
  };
  using temporal_id = TemporalId;
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using system = MockSystem;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags =
      tmpl::list<::Tags::ConcreteTimeStepper<TimeStepper>,
                 ::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>;
  using simple_tags =
      tmpl::list<volume_tag, boundary_tag, dt_volume_tag, dt_boundary_tag,
                 volume_history_tag, boundary_history_tag, ::Tags::TimeStepId,
                 ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
                 ::Tags::Time, ::Tags::StepNumberWithinSlab,
                 ::Tags::AdaptiveSteppingDiagnostics>;
  using compute_tags = time_stepper_ref_tags<TimeStepper>;

  using step_actions =
      tmpl::list<FillTimeDerivatives,
                 ::Actions::MutateApply<RecordTimeStepperData<system>>,
                 ::Actions::MutateApply<UpdateU<system>>,
                 ::Actions::MutateApply<CleanHistory<system>>>;

  using self_start_actions = tmpl::flatten<
      tmpl::list<SelfStart::self_start_procedure<step_actions, system>>>;
  using testing_actions = tmpl::flatten<
      tmpl::list<step_actions, ::Actions::MutateApply<::AdvanceTime<>>>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::InitializeTimeStepperHistory,
                             self_start_actions>,
      Parallel::PhaseActions<Parallel::Phase::Testing, testing_actions>>;
};

using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
using component = Component<Metavariables>;

// Build a boundary container over the given faces, seeding each face's value.
BoundaryVariables<Dim, boundary_vars> make_boundary_variables(
    const std::vector<Direction<Dim>>& directions, const double initial_value) {
  DirectionMap<Dim, size_t> points_per_direction{};
  for (const auto& direction : directions) {
    points_per_direction[direction] =
        direction == face_eta() ? face_eta_points : face_xi_points;
  }
  BoundaryVariables<Dim, boundary_vars> boundary_variables{
      std::move(points_per_direction), initial_value};
  return boundary_variables;
}

// Emplace and initialize the mock component.  `Initialization` sets up the
// dt vars and histories; `TimeStepperHistory` (invoked in `run`) then sets the
// correct starting integration order per stepper family.
void emplace(const gsl::not_null<MockRuntimeSystem*> runner,
             const bool forward_in_time, const Time& initial_time,
             const TimeDelta& time_step, const size_t number_of_past_steps,
             const double initial_volume_value,
             BoundaryVariables<Dim, boundary_vars> boundary_variables) {
  typename volume_tag::type volume{1};
  get(get<VolumeVar>(volume)) = DataVector{1, initial_volume_value};
  // The dt containers are correctly typed (dt-wrapped tags) and sized from the
  // value container's per-direction point counts; their contents are
  // overwritten before use by the TimeStepperHistory mutator and the mock
  // coupling action.
  typename dt_boundary_tag::type dt_boundary{
      boundary_variables.points_per_direction()};
  ActionTesting::emplace_component_and_initialize<component>(
      runner, 0,
      {std::move(volume), boundary_variables, typename dt_volume_tag::type{1},
       std::move(dt_boundary),
       // Histories are given a placeholder order; the TimeStepperHistory
       // mutator run below resets it to the correct per-stepper start.
       typename volume_history_tag::type{1},
       typename boundary_history_tag::type{1}, TimeStepId{},
       TimeStepId(forward_in_time,
                  -static_cast<int64_t>(number_of_past_steps), initial_time),
       time_step, initial_time.value(), uint64_t{0},
       Tags::AdaptiveSteppingDiagnostics::type{}});
  // Set the histories' starting integration order exactly as production does.
  ActionTesting::simple_action<component, InitializeTimeStepperHistory>(runner,
                                                                        0);
}

// Run the self-start phase to completion and switch to the stepping phase, as
// the phase machinery does in a real executable (procedure in its own phase;
// step actions + AdvanceTime in the run phase).
void run_self_start(const gsl::not_null<MockRuntimeSystem*> runner) {
  ActionTesting::set_phase(runner,
                           Parallel::Phase::InitializeTimeStepperHistory);
  while (not ActionTesting::get_terminate<component>(*runner, 0)) {
    ActionTesting::next_action<component>(runner, 0);
  }
  ActionTesting::set_phase(runner, Parallel::Phase::Testing);
}

// Run a whole number of real steps (each step is number_of_substeps passes
// through the step actions plus AdvanceTime).
void run_steps(const gsl::not_null<MockRuntimeSystem*> runner,
               const uint64_t number_of_substeps,
               const size_t number_of_steps) {
  const size_t actions_per_substep =
      tmpl::size<typename component::testing_actions>::value;
  for (size_t step = 0; step < number_of_steps; ++step) {
    for (uint64_t substep = 0; substep < number_of_substeps; ++substep) {
      for (size_t action = 0; action < actions_per_substep; ++action) {
        ActionTesting::next_action<component>(runner, 0);
      }
    }
  }
}

// Assert every history (volume + both faces) shares one integration order and
// one stored-record count.
void check_lockstep(const MockRuntimeSystem& runner,
                    const std::vector<Direction<Dim>>& directions) {
  const auto& volume_history =
      ActionTesting::get_databox_tag<component, volume_history_tag>(runner, 0);
  const auto& boundary_history =
      ActionTesting::get_databox_tag<component, boundary_history_tag>(runner,
                                                                      0);
  CHECK(boundary_history.integration_order() ==
        volume_history.integration_order());
  CHECK(boundary_history.size() == volume_history.size());
  // A single boundary history integrates every face in lockstep by
  // construction: all faces share one buffer, so equal size and order for the
  // one boundary history covers both faces.
  (void)directions;
}

// One convergence experiment: self-start then a fixed number of real steps,
// returning per-variable errors against the closed forms.
struct Errors {
  double volume{};
  DirectionMap<Dim, double> boundary{};
};

template <typename StepperBuilder>
Errors run_experiment(const StepperBuilder& make_stepper,
                      const std::vector<Direction<Dim>>& directions,
                      const bool forward_in_time, const double step_magnitude,
                      const size_t number_of_steps,
                      const double initial_volume_value,
                      const double initial_boundary_value,
                      const bool check_lockstep_invariant) {
  auto stepper = make_stepper();
  const size_t number_of_past_steps = stepper->number_of_past_steps();
  const uint64_t number_of_substeps = stepper->number_of_substeps();
  const auto slab =
      forward_in_time
          ? Slab::with_duration_from_start(1., step_magnitude)
          : Slab::with_duration_to_end(1., step_magnitude);
  const TimeDelta time_step =
      (forward_in_time ? 1 : -1) * slab.duration();
  const Time initial_time = forward_in_time ? slab.start() : slab.end();

  MockRuntimeSystem runner{
      {std::move(stepper), EventsAndTriggers{},
       std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>{},
       VariableOrderAlgorithm{}}};
  emplace(make_not_null(&runner), forward_in_time, initial_time, time_step,
          number_of_past_steps, initial_volume_value,
          make_boundary_variables(directions, initial_boundary_value));
  run_self_start(make_not_null(&runner));
  run_steps(make_not_null(&runner), number_of_substeps, number_of_steps);

  if (check_lockstep_invariant) {
    check_lockstep(runner, directions);
  }

  const double final_time =
      ActionTesting::get_databox_tag<component, ::Tags::TimeStepId>(runner, 0)
          .step_time()
          .value();
  const double initial_time_value = initial_time.value();

  Errors errors{};
  const double numerical_volume = get(get<VolumeVar>(
      ActionTesting::get_databox_tag<component, volume_tag>(runner, 0)))[0];
  errors.volume =
      numerical_volume - volume_solution(initial_volume_value,
                                         initial_time_value, final_time);
  const auto& boundary_variables =
      ActionTesting::get_databox_tag<component, boundary_tag>(runner, 0);
  for (const auto& direction : directions) {
    const double numerical_boundary =
        get(get<Tags_ns::BoundaryValue<VolumeVar>>(
            boundary_variables.variables().at(direction)))[0];
    errors.boundary[direction] =
        numerical_boundary -
        boundary_solution(direction, initial_boundary_value,
                          initial_time_value, final_time);
  }
  return errors;
}

double convergence_rate(const double coarse_error, const double fine_error) {
  return (log(fabs(coarse_error)) - log(fabs(fine_error))) / log(2.);
}

// Full convergence matrix cell: one stepper, one time direction.  Following the
// stock self-start convergence tests, we measure the one-step (local
// truncation) error after a complete self-start: for an order-k stepper this is
// O(h^(k+1)), so halving the step and measuring the error ratio must reproduce
// order + 1 for the volume and for both faces' boundary variable.  A wrong
// order-sync or value reset anywhere in self-start pollutes the step with
// lower-order error and degrades the measured rate.
template <typename StepperBuilder>
void test_convergence(const std::string& stepper_label,
                      const StepperBuilder& make_stepper, const size_t order,
                      const double rate_margin, const bool forward_in_time) {
  CAPTURE(stepper_label);
  CAPTURE(order);
  CAPTURE(forward_in_time);
  const std::vector<Direction<Dim>> directions{face_xi(), face_eta()};
  const double initial_volume_value = 0.5;
  const double initial_boundary_value = 0.3;
  // One real step at two halved step sizes isolates the local truncation error
  // (no accumulation constant), giving the clean rate order + 1.
  const size_t number_of_steps = 1;
  const double coarse_step = 0.1;
  const double fine_step = 0.05;

  const Errors coarse = run_experiment(
      make_stepper, directions, forward_in_time, coarse_step, number_of_steps,
      initial_volume_value, initial_boundary_value, true);
  const Errors fine = run_experiment(
      make_stepper, directions, forward_in_time, fine_step, number_of_steps,
      initial_volume_value, initial_boundary_value, true);

  {
    INFO("volume convergence");
    CHECK(convergence_rate(coarse.volume, fine.volume) ==
          approx(order + 1).margin(rate_margin));
  }
  for (const auto& direction : directions) {
    CAPTURE(direction);
    INFO("boundary convergence");
    CHECK(convergence_rate(coarse.boundary.at(direction),
                           fine.boundary.at(direction)) ==
          approx(order + 1).margin(rate_margin));
  }
}

// An element whose BoundaryVariables is empty (no directions) must run the same
// action list without error; its volume convergence must be unaffected and the
// boundary container must stay empty.
template <typename StepperBuilder>
void test_empty_no_op(const StepperBuilder& make_stepper, const size_t order,
                      const double rate_margin) {
  INFO("empty boundary container is a no-op");
  const std::vector<Direction<Dim>> no_directions{};
  const double initial_volume_value = 0.5;
  const Errors coarse =
      run_experiment(make_stepper, no_directions, true, 0.1, 1,
                     initial_volume_value, 0., false);
  const Errors fine = run_experiment(make_stepper, no_directions, true, 0.05, 1,
                                     initial_volume_value, 0., false);
  CHECK(coarse.boundary.empty());
  CHECK(fine.boundary.empty());
  // The empty boundary container does not perturb the volume: its one-step
  // convergence is the unaffected order + 1.
  CHECK(convergence_rate(coarse.volume, fine.volume) ==
        approx(order + 1).margin(rate_margin));
}

// Stepper-level dense-output check (no action machinery).  Build a boundary
// history by hand from the closed forms, advance it a few steps, then
// dense-update a copy of the boundary container at a mid-step time and compare
// to the closed form to the stepper's dense accuracy.
void test_dense_output() {
  INFO("boundary dense output");
  const std::vector<Direction<Dim>> directions{face_xi(), face_eta()};
  const double initial_boundary_value = 0.3;
  const size_t order = 4;
  const auto make_stepper = [order]() {
    return std::make_unique<TimeSteppers::AdamsBashforth>(order);
  };

  using Vars = BoundaryVariables<Dim, boundary_vars>;
  using DerivVars = db::prefix_variables<::Tags::dt, Vars>;

  const auto dense_error = [&](const size_t steps_per_unit) {
    const auto stepper = make_stepper();
    // One slab spans every seed step plus a couple more, so all step times stay
    // in range; the step size is 1/steps_per_unit, halved by doubling
    // steps_per_unit.
    const size_t number_of_seed_steps = stepper->number_of_past_steps();
    const size_t total_steps = number_of_seed_steps + 2;
    const auto slab = Slab::with_duration_from_start(
        1., static_cast<double>(total_steps) /
                static_cast<double>(steps_per_unit));
    const TimeDelta time_step = slab.duration() / static_cast<int>(total_steps);
    const double initial_time = slab.start().value();

    TimeSteppers::History<Vars> history{order};
    const DirectionMap<Dim, size_t> points_per_direction =
        make_boundary_variables(directions, 0.).points_per_direction();

    // Seed the history with the exact solution and derivative at enough past
    // steps for the multistep stepper to run at full order, then advance
    // through the requested number of steps by exact stepping (each step's
    // record seeded from the closed form).  The boundary container tracks the
    // value at the last recorded step start.
    Vars boundary{points_per_direction};
    Time last_step_time = slab.start();
    for (size_t step = 0; step < total_steps; ++step) {
      const Time step_time = slab.start() + static_cast<int>(step) * time_step;
      const double step_time_value = step_time.value();
      Vars value{points_per_direction};
      DerivVars derivative{points_per_direction};
      for (const auto& direction : directions) {
        const double solution = boundary_solution(
            direction, initial_boundary_value, initial_time, step_time_value);
        get(get<Tags_ns::BoundaryValue<VolumeVar>>(
            value.variables().at(direction))) = solution;
        get(get<::Tags::dt<Tags_ns::BoundaryValue<VolumeVar>>>(
            derivative.variables().at(direction))) =
            face_rate(direction) * solution;
      }
      history.insert(TimeStepId(true, 0, step_time), value, derivative);
      boundary = value;
      last_step_time = step_time;
    }

    // Dense-update the boundary container to a mid-step time within the step
    // that starts at the last recorded time.
    const double dense_time = last_step_time.value() + 0.4 * time_step.value();
    const bool did_update =
        stepper->dense_update_u(make_not_null(&boundary), history, dense_time);
    REQUIRE(did_update);

    double max_error = 0.;
    for (const auto& direction : directions) {
      const double numerical = get(get<Tags_ns::BoundaryValue<VolumeVar>>(
          boundary.variables().at(direction)))[0];
      const double exact = boundary_solution(direction, initial_boundary_value,
                                             initial_time, dense_time);
      max_error = std::max(max_error, fabs(numerical - exact));
    }
    return max_error;
  };

  // Dense output of Adams-Bashforth is accurate to the stepper order; check the
  // error shrinks at that rate across a step halving (loose margin tied to the
  // order).
  const double coarse_error = dense_error(20);
  const double fine_error = dense_error(40);
  // Dense output of an order-k Adams-Bashforth stepper is accurate to at least
  // order k; with an exactly seeded history the only error is the dense
  // interpolation, whose observed rate meets or exceeds the step order.
  CHECK(convergence_rate(coarse_error, fine_error) >= order - 0.5);
  CHECK(fine_error < 1.0e-3);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Dg.BoundaryEvolvedVariables.Stepping",
                  "[Unit][Evolution]") {
  register_classes_with_charm<TimeSteppers::AdamsBashforth,
                              TimeSteppers::AdamsMoultonPc<false>,
                              TimeSteppers::Rk3HesthavenSsp>();

  for (const bool forward_in_time : {true, false}) {
    for (size_t order = 1; order < 5; ++order) {
      const auto make_ab = [order]() {
        return std::make_unique<TimeSteppers::AdamsBashforth>(order);
      };
      test_convergence("AdamsBashforth", make_ab, order, 0.25,
                       forward_in_time);
    }
    for (const size_t order : {size_t{2}, size_t{4}}) {
      const auto make_ampc = [order]() {
        return std::make_unique<TimeSteppers::AdamsMoultonPc<false>>(order);
      };
      test_convergence("AdamsMoultonPc", make_ampc, order, 0.25,
                       forward_in_time);
    }
    {
      const auto make_rk3 = []() {
        return std::make_unique<TimeSteppers::Rk3HesthavenSsp>();
      };
      test_convergence("Rk3HesthavenSsp", make_rk3, 3, 0.1, forward_in_time);
    }
  }

  {
    const auto make_ab = []() {
      return std::make_unique<TimeSteppers::AdamsBashforth>(2);
    };
    test_empty_no_op(make_ab, 2, 0.25);
  }

  test_dense_output();
}
