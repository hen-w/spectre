// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepperErrorEstimatesEnabled.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/UpdateU.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
using variables_tag = gp::System::variables_tag;
using Vars = variables_tag::type;
using DtVars = db::add_tag_prefix<::Tags::dt, variables_tag>::type;
using physical_tags = gp::System::physical_tags;
using PhysicalVars = Variables<physical_tags>;
using PhysicalDtVars = Variables<db::wrap_tags_in<::Tags::dt, physical_tags>>;
using history_tag = ::Tags::HistoryEvolvedVariables<variables_tag>;
using errors_tag = ::Tags::StepperErrors<variables_tag>;
using ErrorControl =
    StepChoosers::ErrorControl<StepChooserUse::Slab, variables_tag>;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event, tmpl::list<Events::ChangeSlabSize>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>, tmpl::list<ErrorControl>>,
        tmpl::pair<Trigger, tmpl::list<Triggers::Always>>>;
  };
};

constexpr size_t number_of_points = 3;
constexpr double absolute_tolerance = 1.0e-6;
constexpr double relative_tolerance = 1.0e-6;
constexpr double max_factor = 10.0;
constexpr double min_factor = 0.01;
constexpr double safety_factor = 0.9;

// The right-hand side y' = f(t) of the physical fields, different for every
// component and grid point and not polynomial in t, so that the error
// estimates do not vanish.
PhysicalDtVars physical_rate(const double time) {
  PhysicalDtVars rate{number_of_points};
  DataVector flat_rate{rate.data(), rate.size()};
  for (size_t i = 0; i < flat_rate.size(); ++i) {
    flat_rate[i] = 0.1 * cos(time + 0.37 * static_cast<double>(i));
  }
  return rate;
}

// Stored gradients unrelated to the physical fields, much larger than them,
// and different for every `seed`.
gp::Tags::PotentialGradients::type gradients(const double seed) {
  gp::Tags::PotentialGradients::type result{number_of_points};
  for (size_t c = 0; c < result.size(); ++c) {
    for (size_t s = 0; s < number_of_points; ++s) {
      result[c][s] = 1.0e3 * (2.0 + sin(seed + static_cast<double>(c) +
                                        0.5 * static_cast<double>(s)));
    }
  }
  return result;
}

// Takes unequal steps of y' = f(t) for the physical fields, through `UpdateU`
// on the evolved variables (with the tolerances requested by an
// `ErrorControl` slab-size chooser of `Events::ChangeSlabSize`) and through
// the time stepper on the physical fields alone. The stored gradients have
// zero time derivative and are reset to unrelated values at every record,
// including the predicted state of a predictor-corrector step. The physical
// fields, the recorded errors and the step chosen by the chooser must be
// those of the physical fields alone, and the gradients those of the step
// start.
void test_error_control(std::unique_ptr<TimeStepper> time_stepper) {
  const size_t order = 3;
  const Slab slab{0.0, 2.0};
  // The step start times, as fractions of the slab.
  const std::array<Rational, 7> step_starts{
      {{0, 1}, {1, 16}, {3, 16}, {1, 4}, {7, 16}, {1, 2}, {5, 8}}};
  const StepperErrorTolerances tolerances{
      .estimates = StepperErrorTolerances::Estimates::StepperOrder,
      .absolute = absolute_tolerance,
      .relative = relative_tolerance};

  PhysicalVars physical{number_of_points, 1.0};
  TimeSteppers::History<PhysicalVars> physical_history{order};
  errors_tag::type physical_errors{};

  Vars initial_vars{number_of_points, 0.0};
  initial_vars.assign_subset(physical);
  auto box =
      db::create<db::AddSimpleTags<
                     Parallel::Tags::MetavariablesImpl<Metavariables>,
                     ::Tags::ConcreteTimeStepper<TimeStepper>,
                     ::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
                     ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                     ::Tags::TimeStep, variables_tag, history_tag, errors_tag>,
                 tmpl::append<time_stepper_ref_tags<TimeStepper>,
                              UpdateU<gp::System, false>::compute_tags>>(
          Metavariables{}, std::move(time_stepper),
          TestHelpers::test_creation<EventsAndTriggers, Metavariables>(
              "- Trigger: Always\n"
              "  Events:\n"
              "    - ChangeSlabSize:\n"
              "        DelayChange: 0\n"
              "        StepChoosers:\n"
              "          - ErrorControl:\n"
              "              AbsoluteTolerance: 1.0e-6\n"
              "              RelativeTolerance: 1.0e-6\n"
              "              MaxFactor: 10.0\n"
              "              MinFactor: 0.01\n"
              "              SafetyFactor: 0.9\n"),
          TimeStepId{}, TimeStepId{}, TimeDelta{}, std::move(initial_vars),
          history_tag::type{order}, errors_tag::type{});
  const TimeStepper& stepper = db::get<::Tags::TimeStepper<TimeStepper>>(box);
  CHECK(db::get<::Tags::StepperErrorEstimatesEnabled>(box));
  CHECK(db::get<::Tags::StepperErrorTolerances<variables_tag>>(box) ==
        tolerances);
  const StepChooser<StepChooserUse::Slab>* chooser = nullptr;
  db::get<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>(box)
      .for_each_event([&chooser](const Event& event) {
        dynamic_cast<const Events::ChangeSlabSize&>(event)
            .for_each_step_chooser(
                [&chooser](
                    const StepChooser<StepChooserUse::Slab>& step_chooser) {
                  chooser = &step_chooser;
                });
      });
  REQUIRE(chooser != nullptr);
  const ErrorControl reference_chooser{absolute_tolerance, relative_tolerance,
                                       max_factor, min_factor, safety_factor};

  double gradient_seed = 0.0;
  for (size_t step = 0; step + 1 < step_starts.size(); ++step) {
    const TimeDelta step_size =
        slab.duration() *
        (gsl::at(step_starts, step + 1) - gsl::at(step_starts, step));
    TimeStepId id{true, 0,
                  slab.start() + slab.duration() * gsl::at(step_starts, step)};
    const double step_start_gradient_seed = gradient_seed + 1.0;
    for (size_t substep = 0; substep < stepper.number_of_substeps_for_error();
         ++substep) {
      CAPTURE(id);
      gradient_seed += 1.0;
      const PhysicalDtVars rate = physical_rate(id.substep_time());
      physical_history.insert(id, physical, rate);
      db::mutate<::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                 ::Tags::TimeStep, variables_tag, history_tag>(
          [&id, &step_size, &stepper, &rate, gradient_seed](
              const gsl::not_null<TimeStepId*> time_step_id,
              const gsl::not_null<TimeStepId*> next_time_step_id,
              const gsl::not_null<TimeDelta*> time_step,
              const gsl::not_null<Vars*> vars,
              const gsl::not_null<history_tag::type*> history) {
            *time_step_id = id;
            *next_time_step_id = stepper.next_time_id_for_error(id, step_size);
            *time_step = step_size;
            get<gp::Tags::PotentialGradients>(*vars) = gradients(gradient_seed);
            DtVars full_rate{number_of_points, 0.0};
            full_rate.assign_subset(rate);
            history->insert(id, *vars, full_rate);
          },
          make_not_null(&box));
      if (step < stepper.number_of_past_steps()) {
        // Only the history before the first step.
        break;
      }

      db::mutate_apply<UpdateU<gp::System, false>>(make_not_null(&box));
      const auto physical_error = stepper.update_u(
          make_not_null(&physical), physical_history, step_size, tolerances);
      CHECK(physical_error.has_value() ==
            (substep + 1 == stepper.number_of_substeps_for_error()));
      if (physical_error.has_value()) {
        // Far above roundoff.
        CHECK(physical_error->step_error() > 1.0);
        if (physical_errors[1].has_value()) {
          physical_errors[0] = physical_errors[1];
        }
        physical_errors[1] = physical_error;
      }

      const Vars& vars = db::get<variables_tag>(box);
      CHECK_VARIABLES_APPROX(vars.extract_subset<physical_tags>(), physical);
      CHECK(get<gp::Tags::PotentialGradients>(vars) ==
            gradients(step_start_gradient_seed));
      const errors_tag::type& errors = db::get<errors_tag>(box);
      for (size_t i = 0; i < 2; ++i) {
        CAPTURE(i);
        const auto& error = gsl::at(errors, i);
        const auto& expected_error = gsl::at(physical_errors, i);
        REQUIRE(error.has_value() == expected_error.has_value());
        if (error.has_value()) {
          CHECK(error->step_time == expected_error->step_time);
          CHECK(error->step_size == expected_error->step_size);
          CHECK(error->order == expected_error->order);
          CHECK(error->step_error() == approx(expected_error->step_error()));
        }
      }
      if (physical_error.has_value()) {
        const TimeStepRequest request =
            chooser->desired_step(step_size.value(), box);
        const TimeStepRequest expected_request =
            reference_chooser(physical_errors, step_size.value());
        REQUIRE(request.size_goal.has_value());
        REQUIRE(expected_request.size_goal.has_value());
        CHECK(*request.size_goal == approx(*expected_request.size_goal));
        // Neither MinFactor nor MaxFactor limits the goal.
        CHECK(*request.size_goal > min_factor * step_size.value());
        CHECK(*request.size_goal < max_factor * step_size.value());
      }

      stepper.clean_history(make_not_null(&physical_history));
      db::mutate<history_tag>(
          [&stepper](const gsl::not_null<history_tag::type*> history) {
            stepper.clean_history(history);
          },
          make_not_null(&box));
      id = stepper.next_time_id_for_error(id, step_size);
    }
  }
  // At least two steps were checked, the last with the PI controller.
  CHECK(physical_errors[0].has_value());
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.GradientProjection.ErrorControl",
                  "[Unit][Evolution]") {
  {
    INFO("AdamsBashforth");
    test_error_control(
        std::make_unique<TimeSteppers::AdamsBashforth>(size_t{3}));
  }
  {
    INFO("AdamsMoultonPc");
    test_error_control(
        std::make_unique<TimeSteppers::AdamsMoultonPc<false>>(size_t{3}));
  }
}
}  // namespace
