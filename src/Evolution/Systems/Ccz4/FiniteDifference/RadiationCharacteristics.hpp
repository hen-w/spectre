// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
/// \endcond

namespace Ccz4::fd {

/// @{
/*!
 * \brief Computes the incoming and outgoing gravitational radiation
 * characteristic speeds.
 *
 * We list radiation characteristic fields and speeds (superscripts) below.
 * See \ref radiation_characteristic_fields() for the definitions of the
 * characteristic fields. We define $n_i$ as a spatial unit normal
 * (w.r.t. the physical background metric) one-form, $\beta^n := \beta^i n_i$,
 * and q_ij = \gamma_{ij} - n_i n_j$ as the projector.
 *
 * char_speeds[0] :  $C_{ij}^{\alpha-\beta^n}$
 * char_speeds[1] :  $C_{ij}^{-\alpha-\beta^n}$
 */
template <typename Frame>
std::array<DataVector, 2> radiation_characteristic_speeds(
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& shift,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form);

template <typename Frame>
void radiation_characteristic_speeds(
    gsl::not_null<std::array<DataVector, 2>*> char_speeds,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& shift,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form);

template <typename Frame>
struct RadiationCharacteristicSpeedsCompute
    : Tags::RadiationCharacteristicSpeeds<DataVector>,
      db::ComputeTag {
  using base = Tags::RadiationCharacteristicSpeeds<DataVector>;
  using type = typename base::type;
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<DataVector, ::Ccz4::fd::System::volume_dim, Frame>,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
          ::Ccz4::fd::System::volume_dim, Frame>>>;
  using return_type = typename base::type;
  static void function(
      const gsl::not_null<return_type*> result, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& shift,
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          unit_normal_one_form) {
    radiation_characteristic_speeds(result, lapse, shift, unit_normal_one_form);
  };
};
/// @}

/// @{
/*!
 * \brief Compute the radiation characteristic variables for the SoCcz4 system.
 *
 * Define the projector $q_{ij}=\gamma_{ij}-n_i n_j$ where $n_i$ is a spatial
 * unit normal (w.r.t. the physical background metric) one-form. Then for a
 * vector $v^i$ \f[ v_i^\perp := q_{ij}v^j, \qquad v_n := v^i n_i \f]. Define
 * the TT projector as $\Pi_{ij}^{kl}=q_i^k q_j^l-\tfrac{1}{2}q_{ij}q^{kl}$.
 *
 * The radiation characteristic fields are:
 *
 * c_tensor_plus/c_tensor_minus:
 * \f[ C_{ij}^{\pm\alpha-\beta^n}=\Pi_{ij}^{kl}(R_{kl}+KK_{kl}-K^m_k K_{ml}
 * \mp\,n^m\nabla_m K_{kl} \pm\,n^m\nabla_{(k}K_{l)m}) \f]
 */
template <typename Frame>
typename Tags::RadiationCharacteristicFields<
    DataVector, ::Ccz4::fd::System::volume_dim, Frame>::type
radiation_characteristic_fields(
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_squared,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        conformal_spatial_metric,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        spatial_metric,
    const tnsr::II<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        inverse_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& a_tilde,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_conformal_factor,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_trace_extrinsic_curvature,
    const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_conformal_spatial_metric,
    const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_a_tilde,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        spatial_ricci_tensor,
    const tnsr::Ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        christoffel,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form);

template <typename Frame>
void radiation_characteristic_fields(
    const gsl::not_null<typename Tags::RadiationCharacteristicFields<
        DataVector, ::Ccz4::fd::System::volume_dim, Frame>::type*>
        radiation_char_fields,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_squared,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        conformal_spatial_metric,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        spatial_metric,
    const tnsr::II<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        inverse_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& a_tilde,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_conformal_factor,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_trace_extrinsic_curvature,
    const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_conformal_spatial_metric,
    const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_a_tilde,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        spatial_ricci_tensor,
    const tnsr::Ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        christoffel,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form);

template <typename Frame>
struct RadiationCharacteristicFieldsCompute
    : Tags::RadiationCharacteristicFields<
          DataVector, ::Ccz4::fd::System::volume_dim, Frame>,
      db::ComputeTag {
  using base = Tags::RadiationCharacteristicFields<
      DataVector, ::Ccz4::fd::System::volume_dim, Frame>;
  using type = typename base::type;
  using argument_tags = tmpl::list<
      ::Ccz4::Tags::ConformalFactor<DataVector>,
      ::Ccz4::Tags::ConformalFactorSquared<DataVector>,
      ::Ccz4::Tags::ConformalMetric<DataVector, ::Ccz4::fd::System::volume_dim,
                                    Frame>,
      ::Ccz4::Tags::InverseConformalMetric<
          DataVector, ::Ccz4::fd::System::volume_dim, Frame>,
      gr::Tags::SpatialMetric<DataVector, ::Ccz4::fd::System::volume_dim,
                              Frame>,
      gr::Tags::InverseSpatialMetric<DataVector, ::Ccz4::fd::System::volume_dim,
                                     Frame>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Ccz4::Tags::ATilde<DataVector, ::Ccz4::fd::System::volume_dim, Frame>,
      ::Ccz4::Tags::GammaHat<DataVector, ::Ccz4::fd::System::volume_dim, Frame>,
      ::Tags::deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                    tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame>,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                    tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame>,
      ::Tags::deriv<::Ccz4::Tags::ConformalMetric<
                        DataVector, ::Ccz4::fd::System::volume_dim, Frame>,
                    tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame>,
      ::Tags::deriv<::Ccz4::Tags::ATilde<DataVector,
                                         ::Ccz4::fd::System::volume_dim, Frame>,
                    tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame>,
      ::Ccz4::Tags::SpatialRicciTensor<DataVector,
                                       ::Ccz4::fd::System::volume_dim, Frame>,
      ::Ccz4::Tags::ChristoffelSecondKind<
          DataVector, ::Ccz4::fd::System::volume_dim, Frame>,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
          ::Ccz4::fd::System::volume_dim, Frame>>>;
  using return_type = typename base::type;
  static void function(
      const gsl::not_null<return_type*> result,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& conformal_factor_squared,
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          conformal_spatial_metric,
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          spatial_metric,
      const tnsr::II<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          inverse_spatial_metric,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          a_tilde,
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          d_conformal_factor,
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          d_trace_extrinsic_curvature,
      const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          d_conformal_spatial_metric,
      const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          d_a_tilde,
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          spatial_ricci_tensor,
      const tnsr::Ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          christoffel,
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          unit_normal_one_form) {
    radiation_characteristic_fields(
        result, conformal_factor, conformal_factor_squared,
        conformal_spatial_metric, spatial_metric, inverse_spatial_metric,
        trace_extrinsic_curvature, a_tilde, d_conformal_factor,
        d_trace_extrinsic_curvature, d_conformal_spatial_metric, d_a_tilde,
        spatial_ricci_tensor, christoffel, unit_normal_one_form);
  }
};
/// @}
}  // namespace Ccz4::fd
