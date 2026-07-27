// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedFields/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/SelfStart.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
template <typename Tag>
struct Next;
struct Time;
struct TimeStep;
struct TimeStepId;
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace evolution::dg::BoundaryEvolvedFields {
namespace detail {
/// Errors if an opted-in boundary element is actually being evolved on the FD
/// (subcell) grid; the subcell integration of boundary-evolved fields is not
/// yet supported. Compiles away when the executable does not use subcell.
template <typename Metavariables, typename HistoryMap, typename DbTags>
void check_not_on_subcell(const HistoryMap& history_map,
                          const db::DataBox<DbTags>& box) {
  if constexpr (evolution::dg::using_subcell_v<Metavariables>) {
    if (not history_map.empty() and
        db::get<evolution::dg::subcell::Tags::ActiveGrid>(box) ==
            evolution::dg::subcell::ActiveGrid::Subcell) {
      ERROR(
          "Boundary-evolved fields are not yet supported on a subcell "
          "element.");
    }
  } else {
    (void)history_map;
    (void)box;
  }
}
}  // namespace detail

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Records the current boundary-field values and their stashed time
/// derivatives into the per-face time-stepper history.
///
/// Iterates the per-face history map and inserts one step record per opting
/// external face. Runs right after the volume `RecordTimeStepperData`.
///
/// Uses:
/// - DataBox:
///   - `Tags::TimeStepId`
///   - `Tags::BoundaryEvolvedFieldsValues<Dim, FieldTagsList>`
///   - `Tags::BoundaryEvolvedFieldsDtStash<Dim, FieldTagsList>`
///
/// DataBox changes:
/// - Modifies: `Tags::BoundaryEvolvedFieldsHistory<Dim, FieldTagsList>`
template <size_t Dim, typename FieldTagsList>
struct RecordBoundaryEvolvedFields {
  using values_tag = Tags::BoundaryEvolvedFieldsValues<Dim, FieldTagsList>;
  using dt_stash_tag = Tags::BoundaryEvolvedFieldsDtStash<Dim, FieldTagsList>;
  using history_tag = Tags::BoundaryEvolvedFieldsHistory<Dim, FieldTagsList>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(not Metavariables::local_time_stepping,
                  "Boundary-evolved fields are not yet supported with local "
                  "time stepping: the facility integrates each face with the "
                  "generic (GTS) `update_u` and does not handle the LTS "
                  "substep / dense-output structure.");
    detail::check_not_on_subcell<Metavariables>(db::get<history_tag>(box), box);

    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    const auto& values = db::get<values_tag>(box);
    const auto& dt_stash = db::get<dt_stash_tag>(box);
    db::mutate<history_tag>(
        [&time_step_id, &values, &dt_stash](
            const gsl::not_null<typename history_tag::type*> history_map) {
          for (auto& [direction, history] : *history_map) {
            history.insert(time_step_id, values.at(direction),
                           dt_stash.at(direction));
          }
        },
        make_not_null(&box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Updates the boundary-field values from their per-face time-stepper
/// history, synchronizing the multistep integration order with the volume
/// history.
///
/// Runs right after the volume `UpdateU`. Because the boundary history is
/// invisible to `Tags::get_all_history_tags`, its integration order is not
/// bumped by `SelfStart::Actions::CheckForOrderIncrease`; this action copies
/// the volume history's current order into each face history before stepping
/// it. During self-start it runs after `CheckForOrderIncrease`, so the order it
/// reads is already the freshly-bumped value.
///
/// Uses:
/// - DataBox:
///   - `Tags::TimeStepId`
///   - `Tags::Next<Tags::TimeStepId>`
///   - `Tags::TimeStep`
///   - `Tags::TimeStepper<TimeStepper>`
///   - the first `Tags::HistoryEvolvedVariables` tag (for the order)
///
/// DataBox changes:
/// - Modifies:
///   - `Tags::BoundaryEvolvedFieldsValues<Dim, FieldTagsList>`
///   - `Tags::BoundaryEvolvedFieldsHistory<Dim, FieldTagsList>`
template <size_t Dim, typename FieldTagsList>
struct UpdateBoundaryEvolvedFields {
  using values_tag = Tags::BoundaryEvolvedFieldsValues<Dim, FieldTagsList>;
  using history_tag = Tags::BoundaryEvolvedFieldsHistory<Dim, FieldTagsList>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(not Metavariables::local_time_stepping,
                  "Boundary-evolved fields are not yet supported with local "
                  "time stepping: the facility integrates each face with the "
                  "generic (GTS) `update_u` and does not handle the LTS "
                  "substep / dense-output structure.");
    detail::check_not_on_subcell<Metavariables>(db::get<history_tag>(box), box);

    using volume_history_tags = ::Tags::get_all_history_tags<DbTags>;
    const size_t integration_order =
        db::get<tmpl::front<volume_history_tags>>(box).integration_order();
    // Copy the volume history's integration order onto each face history.
    // The face histories are invisible to `CheckForOrderIncrease`, so this
    // copy is the only place their order is ever set. It must happen even
    // when the step itself is unused during self-start (where only the
    // `update_u` call is skipped, like the volume `UpdateU`): the multistep
    // `clean_history` running later this step prunes records based on the
    // history's own integration order (keeping `order - 1` past records),
    // and `CheckForOrderIncrease` may have bumped the volume order on
    // exactly such an unused step. Cleaning a face history that still
    // carried the old, lower order would prune records that the next,
    // higher-order update still needs.
    const bool step_is_unused = ::SelfStart::step_unused(
        db::get<::Tags::TimeStepId>(box),
        db::get<::Tags::Next<::Tags::TimeStepId>>(box));
    const auto& time_stepper = db::get<::Tags::TimeStepper<TimeStepper>>(box);
    const auto& time_step = db::get<::Tags::TimeStep>(box);

    db::mutate<values_tag, history_tag>(
        [&integration_order, &time_stepper, &time_step, step_is_unused](
            const gsl::not_null<typename values_tag::type*> values,
            const gsl::not_null<typename history_tag::type*> history_map) {
          for (auto& [direction, history] : *history_map) {
            history.integration_order(integration_order);
            if (not step_is_unused) {
              time_stepper.update_u(make_not_null(&values->at(direction)),
                                    history, time_step);
            }
          }
        },
        make_not_null(&box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Cleans the per-face boundary-field time-stepper history after a
/// substep.
///
/// Runs alongside the volume `CleanHistory`.
///
/// Uses:
/// - DataBox: `Tags::TimeStepper<TimeStepper>`
///
/// DataBox changes:
/// - Modifies: `Tags::BoundaryEvolvedFieldsHistory<Dim, FieldTagsList>`
template <size_t Dim, typename FieldTagsList>
struct CleanBoundaryEvolvedFieldsHistory {
  using history_tag = Tags::BoundaryEvolvedFieldsHistory<Dim, FieldTagsList>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(not Metavariables::local_time_stepping,
                  "Boundary-evolved fields are not yet supported with local "
                  "time stepping: the facility integrates each face with the "
                  "generic (GTS) `update_u` and does not handle the LTS "
                  "substep / dense-output structure.");
    detail::check_not_on_subcell<Metavariables>(db::get<history_tag>(box), box);
    const auto& time_stepper = db::get<::Tags::TimeStepper<TimeStepper>>(box);
    db::mutate<history_tag>(
        [&time_stepper](
            const gsl::not_null<typename history_tag::type*> history_map) {
          for (auto& [direction, history] : *history_map) {
            time_stepper.clean_history(make_not_null(&history));
          }
        },
        make_not_null(&box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Dense-output postprocessor for the boundary-evolved fields.
///
/// Advances each opting face's boundary values to the dense-output time
/// `::Tags::Time` from its per-face history, mirroring how the volume
/// `variables_tag` is dense-updated in
/// `evolution::Actions::RunEventsAndDenseTriggers`. Wrap in
/// `evolution::Actions::AlwaysReadyPostprocessor` and add to that action's
/// postprocessor list.
///
/// The postprocessor's `return_tags` (the value map) are
/// saved and restored around the dense observation by the action's state
/// restorer.
///
/// This action must be invoked from that action's postprocessor list: the
/// action dense-updates the volume variables first and only invokes the
/// postprocessors when that succeeded (it returns early otherwise), so the
/// per-face dense update over the same time-step-id history is expected to
/// succeed as well, and any failure is reported as an error.
template <size_t Dim, typename FieldTagsList>
struct DenseOutputBoundaryEvolvedFields {
  using values_tag = Tags::BoundaryEvolvedFieldsValues<Dim, FieldTagsList>;
  using history_tag = Tags::BoundaryEvolvedFieldsHistory<Dim, FieldTagsList>;

  using return_tags = tmpl::list<values_tag>;
  using argument_tags =
      tmpl::list<history_tag, ::Tags::TimeStepper<TimeStepper>, ::Tags::Time>;

  static void apply(const gsl::not_null<typename values_tag::type*> values,
                    const typename history_tag::type& histories,
                    const TimeStepper& time_stepper, const double time) {
    for (auto& [direction, value] : *values) {
      const auto& history = histories.at(direction);
      value = *history.step_start(time).value;
      if (not time_stepper.dense_update_u(make_not_null(&value), history,
                                          time)) {
        // `direction` is a structured binding, which the ERROR macro's
        // internal lambda cannot capture in C++17; alias it first.
        const auto& failed_direction = direction;
        ERROR("Dense output of the boundary-evolved fields failed at time "
              << time << " for direction " << failed_direction
              << ". This postprocessor must be invoked from the postprocessor "
                 "list of `evolution::Actions::RunEventsAndDenseTriggers`. "
                 "If it is wired "
                 "up that way, this failure means the per-face history has "
                 "diverged from the volume history.");
      }
    }
  }
};

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Applies a slab-size change to the per-face boundary-field histories.
///
/// `Actions::ChangeSlabSize` rewrites the latest `time_step_id` of every
/// history in `Tags::get_all_history_tags` from the old slab to the new one,
/// but the boundary-evolved-fields history is deliberately invisible to that
/// list (see `Tags::BoundaryEvolvedFieldsHistory`), so it would keep the stale
/// old-slab id and the subsequent multistep update would compute a wrong step
/// size. The per-face histories record in lockstep with the volume history, so
/// this action re-establishes that invariant: it copies the volume history's
/// (freshly updated) latest `time_step_id` onto each per-face history's latest
/// record. Place it immediately after `Actions::ChangeSlabSize`. It is a no-op
/// on any step where the slab did not change (the ids already match) and on
/// interior/non-opting elements (empty maps).
template <size_t Dim, typename FieldTagsList>
struct ChangeSlabSizeBoundaryEvolvedFields {
  using history_tag = Tags::BoundaryEvolvedFieldsHistory<Dim, FieldTagsList>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(not Metavariables::local_time_stepping,
                  "Boundary-evolved fields are not yet supported with local "
                  "time stepping: the facility integrates each face with the "
                  "generic (GTS) `update_u` and does not handle the LTS "
                  "substep / dense-output structure.");
    detail::check_not_on_subcell<Metavariables>(db::get<history_tag>(box), box);

    using volume_history_tags = ::Tags::get_all_history_tags<DbTags>;
    const auto& volume_history = db::get<tmpl::front<volume_history_tags>>(box);
    if (volume_history.empty()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const auto volume_latest_id = volume_history.back().time_step_id;
    db::mutate<history_tag>(
        [&volume_latest_id](
            const gsl::not_null<typename history_tag::type*> history_map) {
          for (auto& [direction, history] : *history_map) {
            if (not history.empty() and
                history.back().time_step_id != volume_latest_id) {
              ASSERT(history.at_step_start(),
                     "Cannot apply a slab-size change to a boundary-evolved "
                     "field history that has substep data.");
              history.back().time_step_id = volume_latest_id;
            }
          }
        },
        make_not_null(&box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::dg::BoundaryEvolvedFields
