// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Compute FieldA/B/D/P from FD derivatives of evolved variables and
 * store them into the evolved variables.
 *
 * \details This action is used between the two rounds of subcell ghost data
 * communication. After round 1 provides ghost data for the 9 original evolved
 * variables, this action computes:
 * - \f$A_i = \partial_i \alpha / \alpha\f$
 * - \f$B_{iJ} = \partial_i \beta^J\f$
 * - \f$D_{ijk} = \frac{1}{2}\partial_i \tilde\gamma_{jk}\f$
 * - \f$P_i = \partial_i \phi / \phi\f$
 *
 * These are then stored into the evolved variables so that round 2
 * communication can send them to neighbors.
 */
struct UpdateAuxiliaryVariablesFd {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    using evolved_vars_tag = typename System::variables_tag;
    using gradients_tags = typename System::gradients_tags;

    constexpr size_t fd_order = 4;
    const auto& evolved_vars = db::get<evolved_vars_tag>(box);
    const Mesh<3>& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<3>>(box);
    const size_t num_pts = subcell_mesh.number_of_grid_points();

    const auto& cell_centered_logical_to_inertial_inv_jacobian =
        db::get<evolution::dg::subcell::fd::Tags::
                    InverseJacobianLogicalToInertial<3>>(box);

    constexpr bool subcell_enabled_at_external_boundary =
        std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            box))>::SubcellOptions::subcell_enabled_at_external_boundary;

    const Element<3>& element = db::get<domain::Tags::Element<3>>(box);
    const Ccz4::fd::Reconstructor& recons =
        db::get<Ccz4::fd::Tags::Reconstructor>(box);

    if constexpr (subcell_enabled_at_external_boundary) {
      if (not element.external_boundaries().empty()) {
        fd::BoundaryConditionGhostData::apply(make_not_null(&box), element,
                                              recons);
      }
    }

    using deriv_var_tag = db::wrap_tags_in<::Tags::deriv, gradients_tags,
                                           tmpl::size_t<3>, Frame::Inertial>;
    Variables<deriv_var_tag> cell_centered_Ccz4_derivs{num_pts};

    ::Ccz4::fd::spacetime_derivatives(
        make_not_null(&cell_centered_Ccz4_derivs), evolved_vars,
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
            box),
        fd_order, subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);

    const auto& d_lapse =
        get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                          Frame::Inertial>>(cell_centered_Ccz4_derivs);
    const auto& d_shift =
        get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                          Frame::Inertial>>(cell_centered_Ccz4_derivs);
    const auto& d_spatial_conformal_metric =
        get<::Tags::deriv<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                          tmpl::size_t<3>, Frame::Inertial>>(
            cell_centered_Ccz4_derivs);
    const auto& d_conformal_factor =
        get<::Tags::deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>>(
            cell_centered_Ccz4_derivs);

    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(evolved_vars);
    auto field_a = ::tenex::evaluate<ti::i>(d_lapse(ti::i) / lapse());

    const auto& field_b = d_shift;

    tnsr::ijj<DataVector, 3> field_d;
    ::tenex::evaluate<ti::i, ti::j, ti::k>(
        make_not_null(&field_d),
        0.5 * d_spatial_conformal_metric(ti::i, ti::j, ti::k));

    const auto& conformal_factor =
        get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars);
    auto field_p = ::tenex::evaluate<ti::i>(d_conformal_factor(ti::i) /
                                            conformal_factor());

    db::mutate<::Ccz4::Tags::FieldA<DataVector, 3>,
               ::Ccz4::Tags::FieldB<DataVector, 3>,
               ::Ccz4::Tags::FieldD<DataVector, 3>,
               ::Ccz4::Tags::FieldP<DataVector, 3>>(
        [&field_a, &field_b, &field_d, &field_p](
            const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
                field_a_ptr,
            const gsl::not_null<tnsr::iJ<DataVector, 3, Frame::Inertial>*>
                field_b_ptr,
            const gsl::not_null<tnsr::ijj<DataVector, 3, Frame::Inertial>*>
                field_d_ptr,
            const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
                field_p_ptr) {
          *field_a_ptr = std::move(field_a);
          *field_b_ptr = field_b;
          *field_d_ptr = std::move(field_d);
          *field_p_ptr = std::move(field_p);
        },
        make_not_null(&box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Ccz4::fd
