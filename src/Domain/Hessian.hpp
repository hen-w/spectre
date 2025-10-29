// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/ElementMap.hpp"

/*!
 * \brief Hessian for coordinate maps using autodiff
 */
namespace domain::Hessian {
template <size_t Dim, typename TargetFrame>
auto inv_hessian(
    const ElementMap<Dim, TargetFrame>& map,
    const ::InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            TargetFrame>& inverse_jac,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& source_coords)
    -> ::InverseHessian<DataVector, Dim, Frame::ElementLogical, TargetFrame>;
}  // namespace domain::Hessian
