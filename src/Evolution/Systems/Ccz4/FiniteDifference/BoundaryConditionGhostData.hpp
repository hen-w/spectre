// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace Ccz4::fd {
/*!
 * \brief Computes the finite-difference ghost data of the external boundaries
 * of an element from its boundary conditions.
 *
 * \details For each external boundary direction of the element, the ghost
 * data are computed by the `fd_ghost` function of the boundary condition and
 * stored in `evolution::dg::subcell::Tags::GhostDataForReconstruction` with
 * the key `{direction, ElementId<3>::external_boundary_id()}`, replacing any
 * previous external ghost data in that direction. Entries of neighboring
 * elements are not changed.
 *
 * The ghost data have the layout of the ghost data sent by neighbors: the
 * full `Variables<System::variables_tag_list>` (physical fields and
 * gradients) on `reconstructor.ghost_zone_size()` layers of the face points.
 * The boundary condition sets only the physical fields. The gradients are
 * left default-initialized (signaling NaN with `SPECTRE_NAN_INIT`): the
 * finite-difference derivatives and dissipation use only the physical fields
 * of the ghost data.
 *
 * It is an `ERROR` if the boundary condition does not support finite
 * differences.
 */
struct BoundaryConditionGhostData {
  template <typename DbTagsList>
  static void apply(gsl::not_null<db::DataBox<DbTagsList>*> box,
                    const Element<3>& element,
                    const Reconstructor& reconstructor);

 private:
  template <typename FdBoundaryConditionHelper, typename DbTagsList,
            typename... FdBoundaryConditionArgsTags>
  static void apply_subcell_boundary_condition_impl(
      FdBoundaryConditionHelper& fd_boundary_condition_helper,
      const gsl::not_null<db::DataBox<DbTagsList>*>& box,
      tmpl::list<FdBoundaryConditionArgsTags...> /*meta*/) {
    fd_boundary_condition_helper(db::get<FdBoundaryConditionArgsTags>(*box)...);
  }
};

template <typename DbTagsList>
void BoundaryConditionGhostData::apply(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const Element<3>& element, const Reconstructor& reconstructor) {
  if (element.external_boundaries().empty()) {
    return;
  }
  const auto& external_boundary_conditions =
      db::get<domain::Tags::ExternalBoundaryConditions<3>>(*box).at(
          element.id().block_id());
  const Mesh<3>& subcell_mesh =
      db::get<evolution::dg::subcell::Tags::Mesh<3>>(*box);
  const size_t ghost_zone_size = reconstructor.ghost_zone_size();

  using GhostDataVariables = Variables<System::variables_tag_list>;
  using factory_classes =
      typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
          *box))>::factory_creation::factory_classes;
  using derived_boundary_conditions = tmpl::remove_if<
      tmpl::at<factory_classes, System::boundary_conditions_base>,
      tmpl::or_<
          std::is_base_of<domain::BoundaryConditions::MarkAsPeriodic, tmpl::_1>,
          std::is_base_of<domain::BoundaryConditions::MarkAsNone, tmpl::_1>>>;

  for (const auto& direction : element.external_boundaries()) {
    const size_t number_of_ghost_points =
        ghost_zone_size *
        subcell_mesh.extents().slice_away(direction.dimension()).product();
    evolution::dg::subcell::GhostData ghost_data{1};
    // A newly allocated buffer, so the gradients that the boundary condition
    // does not set are default-initialized.
    ghost_data.neighbor_ghost_data_for_reconstruction() =
        DataVector{GhostDataVariables::number_of_independent_components *
                   number_of_ghost_points};
    GhostDataVariables ghost_vars{
        ghost_data.neighbor_ghost_data_for_reconstruction().data(),
        ghost_data.neighbor_ghost_data_for_reconstruction().size()};

    const auto& boundary_condition_at_direction =
        *external_boundary_conditions.at(direction);
    call_with_dynamic_type<void, derived_boundary_conditions>(
        &boundary_condition_at_direction,
        [&box, &direction, &ghost_vars](const auto* const boundary_condition) {
          using BoundaryCondition = std::decay_t<decltype(*boundary_condition)>;
          if constexpr (BoundaryCondition::bc_type ==
                        evolution::BoundaryConditions::Type::Ghost) {
            using fd_tags = tmpl::append<
                typename BoundaryCondition::fd_interior_evolved_variables_tags,
                typename BoundaryCondition::fd_interior_temporary_tags,
                typename BoundaryCondition::fd_gridless_tags>;
            const auto apply_fd_ghost = [&boundary_condition, &direction,
                                         &ghost_vars](
                                            const auto&... fd_ghost_args) {
              boundary_condition->fd_ghost(
                  make_not_null(
                      &get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(
                          ghost_vars)),
                  make_not_null(&get<::Ccz4::Tags::ConformalFactor<DataVector>>(
                      ghost_vars)),
                  make_not_null(
                      &get<::Ccz4::Tags::ATilde<DataVector, 3>>(ghost_vars)),
                  make_not_null(
                      &get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
                          ghost_vars)),
                  make_not_null(
                      &get<::Ccz4::Tags::Theta<DataVector>>(ghost_vars)),
                  make_not_null(
                      &get<::Ccz4::Tags::GammaHat<DataVector, 3>>(ghost_vars)),
                  make_not_null(&get<gr::Tags::Lapse<DataVector>>(ghost_vars)),
                  make_not_null(
                      &get<gr::Tags::Shift<DataVector, 3>>(ghost_vars)),
                  make_not_null(
                      &get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(
                          ghost_vars)),
                  direction, fd_ghost_args...);
            };
            apply_subcell_boundary_condition_impl(apply_fd_ghost, box,
                                                  fd_tags{});
          } else {
            ERROR("Unsupported boundary condition "
                  << pretty_type::short_name<BoundaryCondition>()
                  << " on an element using finite differences.");
          }
        });

    db::mutate<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
        [&direction, &ghost_data](const auto all_ghost_data) {
          all_ghost_data->insert_or_assign(
              DirectionalId<3>{direction, ElementId<3>::external_boundary_id()},
              std::move(ghost_data));
        },
        box);
  }
}

namespace Actions {
/*!
 * \brief On elements using finite differences, insert the ghost data of the
 * external boundaries (see `Ccz4::fd::BoundaryConditionGhostData`).
 *
 * \details `evolution::dg::subcell::Actions::SendDataForReconstruction` clears
 * all ghost data, so this action must follow each
 * `evolution::dg::subcell::Actions::ReceiveDataForReconstruction` whose ghost
 * data are used.
 */
struct SetBoundaryConditionGhostData {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    BoundaryConditionGhostData::apply(make_not_null(&box),
                                      db::get<domain::Tags::Element<3>>(box),
                                      db::get<Tags::Reconstructor>(box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Ccz4::fd
