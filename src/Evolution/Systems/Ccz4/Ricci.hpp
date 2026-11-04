// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
/// @{
/*!
 * \brief Computes the spatial Ricci tensor
 *
 * \details Computes the Ricci tensor of the spatial metric
 * \f$\gamma_{ij} = \phi^{-2} \tilde{\gamma}_{ij}\f$ in \f$n\f$ spatial
 * dimensions from the conformal decomposition
 *
 * \f{align}
 *     R_{ij} &= \tilde{R}_{ij} + (n - 2) (H_{ij} + P_i P_j) +
 *       \tilde{\gamma}_{ij} \tilde{\gamma}^{kl} (H_{kl} - (n - 2) P_k P_l),
 * \f}
 *
 * where
 *
 * \f{align}
 *     \tilde{R}_{ij} &=
 *       \partial_m \tilde{\Gamma}^m_{ij} - \partial_j \tilde{\Gamma}^m_{im} +
 *       \tilde{\Gamma}^l_{ij} \tilde{\Gamma}^m_{lm} -
 *       \tilde{\Gamma}^l_{im} \tilde{\Gamma}^m_{lj},\\
 *     H_{ij} &= \partial_{(i} P_{j)} - \tilde{\Gamma}^k_{ij} P_k,
 * \f}
 *
 * \f$\tilde{\Gamma}^k_{ij}\f$ is the conformal spatial christoffel symbols of
 * the second kind defined by `Ccz4::Tags::ConformalChristoffelSecondKind`,
 * \f$\tilde{\gamma}_{ij}\f$ is the conformal spatial metric defined by
 * `Ccz4::Tags::ConformalMetric`, \f$\tilde{\gamma}^{ij}\f$ is the inverse
 * conformal spatial metric defined by `Ccz4::Tags::InverseConformalMetric`,
 * \f$P_i\f$ is the CCZ4 auxiliary variable defined by `Ccz4::Tags::FieldP`,
 * and \f$\partial_j P_{i}\f$ is its spatial derivative. The argument
 * `contracted_d_conformal_christoffel_difference` corresponds to the
 * \f$\partial_m \tilde{\Gamma}^m_{ij} - \partial_j \tilde{\Gamma}^m_{im}\f$
 * term.
 *
 * The arguments must be consistent: the inverse conformal spatial metric is
 * the inverse of the conformal spatial metric, and
 * \f$\tilde{\Gamma}^k_{ij}\f$ is computed from \f$D_{kij}\f$ and the inverse
 * conformal spatial metric by `Ccz4::conformal_christoffel_second_kind`. The
 * result is then algebraically identical to the expansion
 *
 * \f{align}
 *     R_{ij} &=
 *       \partial_m \Gamma^m_{ij} - \partial_j \Gamma^m_{im} +
 *       \Gamma^l_{ij} \Gamma^m_{lm} - \Gamma^l_{im} \Gamma^m_{lj}
 * \f}
 *
 * in the spatial christoffel symbols of the second kind
 * \f$\Gamma^k_{ij}\f$ defined by `Ccz4::Tags::ChristoffelSecondKind`, with
 *
 * \f{align}
 *     \partial_k \Gamma^m_{ij} &=
 *       \partial_k \tilde{\Gamma}^m_{ij} +
 *       2 D_k{}^{ml} (\tilde{\gamma}_{jl} P_i + \tilde{\gamma}_{il} P_j -
 *                     \tilde{\gamma}_{ij} P_l)\nonumber\\
 *       & - 2 \tilde{\gamma}^{ml} (D_{kjl} P_i + D_{kil} P_j - D_{kij} P_l) -
 *       \tilde{\gamma}^{ml} (
 *         \tilde{\gamma}_{jl} \partial_{(k} P_{i)} +
 *         \tilde{\gamma}_{il} \partial_{(k} P_{j)} -
 *         \tilde{\gamma}_{ij} \partial_{(k} P_{l)}),
 * \f}
 *
 * where \f$D_{kij}\f$ is the CCZ4 auxiliary variable defined by
 * `Ccz4::Tags::FieldD` and \f$D_k{}^{ij}\f$ is the CCZ4 identity defined by
 * `Ccz4::Tags::FieldDUp`. The identity holds for any \f$D_{kij}\f$ symmetric
 * in its last two indices, \f$P_i\f$, \f$\partial_j P_i\f$, and
 * \f$\partial_m \tilde{\Gamma}^m_{ij} - \partial_j \tilde{\Gamma}^m_{im}\f$:
 * it does not assume that
 * \f$D_{kij} = \frac{1}{2} \partial_k \tilde{\gamma}_{ij}\f$, that the
 * conformal spatial metric has unit determinant, or any constraint.
 *
 * \note Only the symmetric part \f$\partial_{(i} P_{j)}\f$ of the derivative
 * of \f$P_i\f$ enters. In second-order Ccz4,
 * \f$\partial_i P_j = \partial_i \partial_j \ln \phi\f$ is symmetric in i and
 * j (`TensorType` is `tnsr::ii`), because partial derivatives commute and to
 * use `second_partial_derivatives()`. \f$P_i\f$ is evolved in the first-order
 * system so no such symmetry is imposed (`TensorType` is `tnsr::ij`), and
 * \f$\partial_j P_i\f$ is symmetrized.
 */
template <typename DataType, size_t Dim, typename Frame, typename TensorType>
void spatial_ricci_tensor(
    const gsl::not_null<tnsr::ii<DataType, Dim, Frame>*> result,
    const tnsr::Ijj<DataType, Dim, Frame>& conformal_christoffel_second_kind,
    const tnsr::ij<DataType, Dim, Frame>&
        contracted_d_conformal_christoffel_difference,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::i<DataType, Dim, Frame>& field_p, const TensorType& d_field_p);

template <typename DataType, size_t Dim, typename Frame, typename TensorType>
tnsr::ii<DataType, Dim, Frame> spatial_ricci_tensor(
    const tnsr::Ijj<DataType, Dim, Frame>& conformal_christoffel_second_kind,
    const tnsr::ij<DataType, Dim, Frame>&
        contracted_d_conformal_christoffel_difference,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::i<DataType, Dim, Frame>& field_p, const TensorType& d_field_p);
/// @}
}  // namespace Ccz4
