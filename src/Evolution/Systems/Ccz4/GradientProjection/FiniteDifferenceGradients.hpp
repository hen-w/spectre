// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/InitialActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
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

namespace Ccz4::GradientProjection {
/*!
 * \brief The gradients \f$q_i\f$ sent by a finite-difference element to its
 * DG neighbors, from the finite-difference first derivatives of the physical
 * fields:
 * \f{align*}{
 * q_{H,i}&=\mathrm{STF}\left(\mathcal T_H^{-1}\partial_i\tilde\gamma\right),&
 * q_{\phi,i}&=\partial_i\phi,&
 * q_{\alpha,i}&=\partial_i\alpha,&
 * q_{\beta,i}{}^j&=\partial_i\beta^j,
 * \f}
 * with \f$H=\frac12\log\tilde\gamma\f$ of the local conformal metric.
 *
 * \details The trace-free projection makes the transmitted metric gradient
 * tangent to the unit-determinant surface,
 * \f$\mathcal T_Hq_{H,i}=\partial_i\tilde\gamma
 * -\tilde\gamma\,\mathrm{tr}(\tilde\gamma^{-1}\partial_i\tilde\gamma)/3\f$; a
 * discrete derivative of a unit-determinant metric need not be. The
 * finite-difference right-hand side does not use these gradients. It is an
 * `ERROR` if a local conformal metric is off the unit-determinant surface by
 * more than `chart_trace_tolerance` (the volume is projected at the start of
 * each finite-difference right-hand side with `ConstrainedEvolution`).
 */
void finite_difference_potential_gradients(
    gsl::not_null<Tags::PotentialGradients::type*> gradients,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::ijj<DataVector, 3>& d_conformal_metric,
    const tnsr::i<DataVector, 3>& d_conformal_factor,
    const tnsr::i<DataVector, 3>& d_lapse,
    const tnsr::iJ<DataVector, 3>& d_shift);

namespace Actions {
/*!
 * \brief On finite-difference elements, between the two ghost-data
 * exchanges: overwrite the stored gradients with
 * `finite_difference_potential_gradients` of the fourth-order derivatives of
 * the physical fields `Ccz4::fd::System::second_derivative_tags`
 * (`Ccz4::fd::spacetime_derivatives`, using the ghost data of the first
 * exchange and of the external boundaries), so that the second exchange sends
 * them to the DG neighbors.
 *
 * \details It is an `ERROR` if the lapse and shift are not evolved, since the
 * DG elements always evolve them.
 *
 * If the active grids are fixed at initialization
 * (`evolution::dg::active_grids_fixed_at_initialization`) and the auxiliary
 * finite-difference ghost data are reused
 * (`evolution::dg::reuse_auxiliary_finite_difference_ghost_data`), as in
 * `EvolveCcz4`, an element none of whose neighbors uses DG (according to
 * `evolution::dg::subcell::Tags::NeighborActiveGrids`) leaves the stored
 * gradients unchanged: only the DG neighbors use them, the finite-difference
 * neighbors using only the physical fields of the ghost data. The active grid
 * of every neighbor is checked, so it is an `ERROR` if a neighbor has none
 * (`evolution::dg::subcell::neighbor_uses_subcell`), even if another neighbor
 * uses DG. Reusing the auxiliary ghost data rules out donor extension
 * (`evolution::dg::subcell::SubcellOptions::enable_extension_directions`),
 * with which an element can interpolate the ghost data that it sends to a
 * neighbor from the ghost data of another neighbor, so that the gradients of
 * a finite-difference element could reach a DG element that is not its
 * neighbor. Otherwise the gradients are always computed.
 */
struct ComputeFiniteDifferenceGradients {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (not db::get<::Ccz4::fd::Tags::EvolveLapseAndShift>(box)) {
      ERROR(
          "The CCZ4 gradient-projection DG elements always evolve the lapse "
          "and shift, so the finite-difference elements must too "
          "(EvolveLapseAndShift).");
    }
    if constexpr (evolution::dg::active_grids_fixed_at_initialization_v<
                      Metavariables> and
                  evolution::dg::reuse_auxiliary_finite_difference_ghost_data_v<
                      Metavariables>) {
      // Every neighbor is checked (without stopping at the first one using
      // DG), so that a neighbor without an active grid is an error.
      bool has_dg_neighbor = false;
      for (const auto& [direction, neighbors_in_direction] :
           db::get<::domain::Tags::Element<3>>(box).neighbors()) {
        for (const ElementId<3>& neighbor : neighbors_in_direction) {
          if (not evolution::dg::subcell::neighbor_uses_subcell(
                  box, DirectionalId<3>{direction, neighbor})) {
            has_dg_neighbor = true;
          }
        }
      }
      if (not has_dg_neighbor) {
        return {Parallel::AlgorithmExecution::Continue, std::nullopt};
      }
    }
    using deriv_tags =
        db::wrap_tags_in<::Tags::deriv,
                         typename ::Ccz4::fd::System::second_derivative_tags,
                         tmpl::size_t<3>, Frame::Inertial>;
    constexpr size_t fd_order = 4;
    const Mesh<3>& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<3>>(box);
    const auto& evolved_vars =
        db::get<typename ::Ccz4::fd::System::variables_tag>(box);
    Variables<deriv_tags> derivatives{subcell_mesh.number_of_grid_points()};
    ::Ccz4::fd::spacetime_derivatives(
        make_not_null(&derivatives), evolved_vars,
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
            box),
        fd_order, subcell_mesh,
        db::get<evolution::dg::subcell::fd::Tags::
                    InverseJacobianLogicalToInertial<3>>(box));
    db::mutate<Tags::PotentialGradients>(
        [&derivatives, &evolved_vars](
            const gsl::not_null<Tags::PotentialGradients::type*> gradients) {
          finite_difference_potential_gradients(
              gradients,
              get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(evolved_vars),
              get<::Tags::deriv<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                                tmpl::size_t<3>, Frame::Inertial>>(derivatives),
              get<::Tags::deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                                tmpl::size_t<3>, Frame::Inertial>>(derivatives),
              get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                                Frame::Inertial>>(derivatives),
              get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                                Frame::Inertial>>(derivatives));
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Ccz4::GradientProjection
