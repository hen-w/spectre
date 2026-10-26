// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::Actions {
/*!
 * \brief Send the DG-subcell ghost data of an element using DG to all its
 * neighbors through the auxiliary
 * `evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox`, without any
 * boundary-correction data.
 *
 * \details This is the first of two ghost exchanges per right-hand side for
 * systems whose finite-difference elements must receive the neighbor ghost
 * data, compute derived fields from it, and send a second set of ghost data
 * before their neighbors can compute boundary corrections. A
 * finite-difference element receives this message with
 * `evolution::dg::subcell::Actions::ReceiveDataForReconstruction<Dim, true>`;
 * an element using DG discards it with `DrainAuxiliaryGhostData`. The second
 * exchange is the usual send of `ComputeTimeDerivative`.
 *
 * The data sent are those of the `ComputeTimeDerivative` send (the ghost data
 * from `Metavariables::SubcellOptions::GhostVariables`, the oriented volume
 * mesh, the ghost-data mesh, the TCI decision, the integration order and the
 * validity range), with empty boundary-correction mesh and data. The system
 * must have no fluxes, and only global time stepping is supported.
 */
template <size_t Dim, typename EvolutionSystem, bool UseNodegroupDgElements>
struct SendAuxiliaryGhostData {
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          Dim, UseNodegroupDgElements, true>>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(UseNodegroupDgElements ==
                      Parallel::is_dg_element_collection_v<ParallelComponent>,
                  "The template parameter UseNodegroupDgElements of "
                  "SendAuxiliaryGhostData must match whether the "
                  "ParallelComponent is a DgElementCollection.");
    static_assert(using_subcell_v<Metavariables>,
                  "Auxiliary ghost data are only sent with DG-subcell.");
    using flux_variables = typename EvolutionSystem::flux_variables;
    static_assert(tmpl::size<flux_variables>::value == 0,
                  "SendAuxiliaryGhostData sends no fluxes, so the system must "
                  "have none.");
    ASSERT(alg::all_of(db::get<evolution::dg::Tags::MortarInfo<Dim>>(box),
                       [](const auto& mortar) {
                         return mortar.second.time_stepping_policy() ==
                                TimeSteppingPolicy::EqualRate;
                       }),
           "Cannot send auxiliary ghost data from "
               << db::get<::domain::Tags::Element<Dim>>(box).id()
               << " across an LTS mortar.");

    const Variables<db::wrap_tags_in<::Tags::Flux, flux_variables,
                                     tmpl::size_t<Dim>, Frame::Inertial>>
        no_fluxes{};
    detail::send_boundary_and_ghost_data<Dim, EvolutionSystem,
                                         UseNodegroupDgElements, true, false,
                                         ParallelComponent>(
        make_not_null(&cache), make_not_null(&box), no_fluxes);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Discard the auxiliary ghost data received by an element using DG.
 *
 * \details Waits until one message per neighbor has arrived in the auxiliary
 * `evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox` at the current
 * `Tags::TimeStepId`, then erases that entry. Messages for other time step
 * ids are left in the inbox, and no other state (mortar data, mortar time
 * step ids, neighbor meshes or ghost data) is changed. The auxiliary data may
 * arrive before or after the data of the second exchange. The number of
 * expected messages does not depend on the mortar time step ids, so the
 * action may run after `ApplyBoundaryCorrectionsToTimeDerivative` (which
 * advances them), where it does not delay the right-hand side. Elements using
 * DG need the auxiliary data of neither their DG nor their finite-difference
 * neighbors. Directions with several (nonconforming) neighbors expect one
 * message per neighbor, each under its own sender key.
 */
template <size_t Dim, bool UseNodegroupDgElements>
struct DrainAuxiliaryGhostData {
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          Dim, UseNodegroupDgElements, true>>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(UseNodegroupDgElements ==
                      Parallel::is_dg_element_collection_v<ParallelComponent>,
                  "The template parameter UseNodegroupDgElements of "
                  "DrainAuxiliaryGhostData must match whether the "
                  "ParallelComponent is a DgElementCollection.");
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);
    const size_t number_of_expected_messages = element.number_of_neighbors();
    if (number_of_expected_messages == 0) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    auto& inbox =
        tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
            Dim, UseNodegroupDgElements, true>>(inboxes);
    inbox.collect_messages();
    const auto received = inbox.messages.find(time_step_id);
    if (received == inbox.messages.end()) {
      inbox.set_missing_messages(number_of_expected_messages);
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    const size_t number_of_received_messages = received->second.size();
    if (number_of_received_messages < number_of_expected_messages) {
      inbox.set_missing_messages(number_of_expected_messages -
                                 number_of_received_messages);
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    if (number_of_received_messages > number_of_expected_messages) {
      ERROR("Element " << element.id() << " received "
                       << number_of_received_messages
                       << " auxiliary ghost-data messages at " << time_step_id
                       << " but has only " << number_of_expected_messages
                       << " neighbors.");
    }
#ifdef SPECTRE_DEBUG
    std::unordered_set<DirectionalId<Dim>> senders{};
    for (const auto& [sender, data] : received->second) {
      (void)data;
      ASSERT(element.neighbors().contains(sender.direction()) and
                 element.neighbors()
                     .at(sender.direction())
                     .ids()
                     .contains(sender.id()),
             "Element " << element.id()
                        << " received auxiliary ghost data from ("
                        << sender.direction() << ',' << sender.id()
                        << "), which is not a neighbor.");
      ASSERT(senders.insert(sender).second,
             "Element " << element.id()
                        << " received auxiliary ghost data twice from ("
                        << sender.direction() << ',' << sender.id() << ").");
    }
#endif  // SPECTRE_DEBUG
    inbox.messages.erase(received);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::dg::Actions
