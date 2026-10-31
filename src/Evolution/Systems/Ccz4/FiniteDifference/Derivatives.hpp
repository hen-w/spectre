// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/System.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
namespace evolution::dg::subcell {
class GhostData;
}  // namespace evolution::dg::subcell
/// \endcond

namespace Ccz4::fd {
/*!
 * \brief Compute first partial derivatives of all the evolved variables in the
 * second order Ccz4 system.
 *
 * The derivatives are computed using FD of order deriv_order using stencil
 * the same as `fd::partial_derivatives()`.
 *
 */
void spacetime_derivatives(
    gsl::not_null<Variables<
        db::wrap_tags_in<::Tags::deriv, typename System::gradients_tags,
                         tmpl::size_t<3>, Frame::Inertial>>*>
        result,
    const Variables<typename System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const size_t& deriv_order, const Mesh<3>& volume_mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_jacobian);

/*!
 * \brief Compute first partial derivatives of the evolved variables
 * `System::second_derivative_tags` only.
 *
 * \details The derivatives are the same as the corresponding ones of the
 * overload for all the evolved variables.
 */
void spacetime_derivatives(
    gsl::not_null<Variables<
        db::wrap_tags_in<::Tags::deriv, System::second_derivative_tags,
                         tmpl::size_t<3>, Frame::Inertial>>*>
        result,
    const Variables<typename System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const size_t& deriv_order, const Mesh<3>& volume_mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_jacobian);

/*!
 * \brief Compute second partial derivatives of all the evolved variables in the
 * second order Ccz4 system
 * \details
 * The derivatives are computed using FD of order deriv_order.
 *
 * \note Only 3D 4-th order second derivatives are implemented
 * in this stencil
 */
void second_spacetime_derivatives(
    gsl::not_null<
        Variables<db::wrap_tags_in<::Tags::second_deriv, System::gradients_tags,
                                   tmpl::size_t<3>, Frame::Inertial>>*>
        result,
    const Variables<typename System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const size_t& deriv_order, const Mesh<3>& volume_mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_jacobian,
    const InverseHessian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_hessian);

/*!
 * \brief Compute first partial derivatives of all the evolved variables and
 * second partial derivatives of the evolved variables
 * `System::second_derivative_tags` in the second order Ccz4 system.
 *
 * \details The derivatives are the same as those of `spacetime_derivatives()`
 * and the corresponding ones of `second_spacetime_derivatives()`, but the
 * logical first derivatives are computed once and reused in the
 * inverse-Hessian terms of the second derivatives.
 *
 * \note Only 3D 4-th order second derivatives are implemented
 * in this stencil
 */
void first_and_second_spacetime_derivatives(
    gsl::not_null<
        Variables<db::wrap_tags_in<::Tags::deriv, System::gradients_tags,
                                   tmpl::size_t<3>, Frame::Inertial>>*>
        first_derivatives,
    gsl::not_null<Variables<
        db::wrap_tags_in<::Tags::second_deriv, System::second_derivative_tags,
                         tmpl::size_t<3>, Frame::Inertial>>*>
        second_derivatives,
    const Variables<typename System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const size_t& deriv_order, const Mesh<3>& volume_mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_jacobian,
    const InverseHessian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_hessian);
}  // namespace Ccz4::fd
