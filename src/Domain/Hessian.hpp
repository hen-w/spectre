// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/ElementMap.hpp"

/*!
 * \brief Hessian for coordinate maps using autodiff forward mode with Duals
 * \details This function computes the inverse hessian by computing the hessian
 * of the map and multiply it by the inverse jacobians.
 */
namespace domain::Hessian {
template <size_t Dim, typename TargetFrame>
auto inv_hessian(
    const ElementMap<Dim, TargetFrame>& map,
    const ::InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            TargetFrame>& inverse_jac,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& source_coords)
    -> ::InverseHessian<DataVector, Dim, Frame::ElementLogical, TargetFrame>;

/*!
 * \brief Hessian for coordinate maps using autodiff forward mode with dual
 * numbers \details This function computes the inverse hessian by propagating
 * the dual numbers through the inverse map \note This function will not work
 * for maps using root finding in their inverse. Currently it is not clear which
 * inv_hessian is more efficient.
 */
template <size_t Dim, typename TargetFrame>
auto inv_hessian(
    const ElementMap<Dim, TargetFrame>& map,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& source_coords)
    -> ::InverseHessian<DataVector, Dim, Frame::ElementLogical, TargetFrame>;
}  // namespace domain::Hessian
