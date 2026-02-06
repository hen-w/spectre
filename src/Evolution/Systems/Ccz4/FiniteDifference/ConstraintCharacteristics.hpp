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
 * \brief Compute the constraint characteristic speeds
 * for SoCcz4
 *
 * There are totally 4 constraint characteristic fields, 2*1 from
 * the vector sector and 2 from the scalar sector. In the following,
 * we only compute 1+2=3 characteristic speeds as the two transverse
 * characteristics in the vector sector have the same speed.
 *
 * We list constraint characteristic fields and speeds (superscripts) below.
 * See \ref constraint_characteristic_fields() for the definitions of the
 * characteristic fields. We define $\beta^n := \beta^i n_i$ where $n_i$ is a
 * spatial unit normal (w.r.t. the physical background metric) one-form.
 *
 * char_speeds[0] :  $C_i^{-\beta^n}$
 *
 * char_speeds[1] :  $C^{\alpha-\beta^n}$
 *
 * char_speeds[2] :  $C^{-\alpha-\beta^n}$
 */
template <typename Frame>
std::array<DataVector, 3> constraint_characteristic_speeds(
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& shift,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form);

template <typename Frame>
void constraint_characteristic_speeds(
    gsl::not_null<std::array<DataVector, 3>*> char_speeds,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& shift,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form);

template <typename Frame>
struct ConstraintCharacteristicSpeedsCompute
    : Tags::ConstraintCharacteristicSpeeds<DataVector>,
      db::ComputeTag {
  using base = Tags::ConstraintCharacteristicSpeeds<DataVector>;
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
    constraint_characteristic_speeds(result, lapse, shift,
                                     unit_normal_one_form);
  };
};
/// @}

/// @{
/*!
 * \brief Compute the constraint characteristic variables for the SoCcz4 system.
 *
 * Define the projector $q_{ij}=\gamma_{ij}-n_i n_j$ where $n_i$ is a spatial
 * unit normal (w.r.t. the physical background metric) one-form. Then for a
 * vector $v^i$ \f[ v_i^\perp := q_{ij}v^j, \qquad v_n := v^i n_i \f].
 *
 * The constraint characteristic fields are:
 *
 * c_vector_zero:
 * \f[
 * C_i^{-\beta^n}=Z_i^\perp
 * =\frac{\phi^2}{4}\Big(U_i^{+\alpha-\beta^n}+U_i^{-\alpha-\beta^n}\Big)
 * -\frac{\phi^2}{2}\,\mathcal T^\perp_i.
 * \f]
 * c_scalar_plus/minus:
 * \f[
 * C^{\pm\alpha-\beta^n} = \mp\Theta + Z_n
 * =\frac{\phi^2}{2}\,U_{(2)}^{\pm\alpha-\beta^n}
 * -\frac{1}{2\phi^2}
 * \Big(U_{(1)}^{+\alpha-\beta^n}-U_{(1)}^{-\alpha-\beta^n}\Big)
 * -\frac{\phi^2}{2}\,\mathcal T^{n}.
 * \f]
 * where
 * \f[
 * \partial^\perp_\ell := q_\ell{}^{m}\partial_m,
 * \qquad
 * \mathcal T^{i}:=\tilde\gamma^{ij}\tilde\gamma^{k\ell}\,\partial^\perp_\ell
 * \tilde\gamma_{jk}, \qquad \mathcal T^{n}:=n_i\mathcal T^{i}, \qquad \mathcal
 * T^\perp_i:=q_{ij}\mathcal T^{j}. \f] and $U_i^{\pm\alpha-\beta^n}$,
 * $U_{(1)}^{\pm\alpha-\beta^n}$, and $U_{(2)}^{\pm\alpha-\beta^n}$ are some of
 * the main characteristic fields of SoCcz4 evolution system. See \ref
 * characteristic_fields() for their definitions.
 */
template <typename Frame>
typename Tags::ConstraintCharacteristicFields<
    DataVector, ::Ccz4::fd::System::volume_dim, Frame>::type
constraint_characteristic_fields(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        conformal_spatial_metric,
    const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_conformal_spatial_metric,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        u_vector2_plus,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        u_vector2_minus,
    const Scalar<DataVector>& u_scalar2_plus,
    const Scalar<DataVector>& u_scalar2_minus,
    const Scalar<DataVector>& u_scalar3_plus,
    const Scalar<DataVector>& u_scalar3_minus,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form);

template <typename Frame>
void constraint_characteristic_fields(
    const gsl::not_null<typename Tags::ConstraintCharacteristicFields<
        DataVector, ::Ccz4::fd::System::volume_dim, Frame>::type*>
        constraint_char_fields,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        conformal_spatial_metric,
    const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_conformal_spatial_metric,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        u_vector2_plus,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        u_vector2_minus,
    const Scalar<DataVector>& u_scalar2_plus,
    const Scalar<DataVector>& u_scalar2_minus,
    const Scalar<DataVector>& u_scalar3_plus,
    const Scalar<DataVector>& u_scalar3_minus,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form);

template <typename Frame>
struct ConstraintCharacteristicFieldsCompute
    : Tags::ConstraintCharacteristicFields<
          DataVector, ::Ccz4::fd::System::volume_dim, Frame>,
      db::ComputeTag {
  using base = Tags::ConstraintCharacteristicFields<
      DataVector, ::Ccz4::fd::System::volume_dim, Frame>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      ::Ccz4::Tags::ConformalFactor<DataVector>,
      ::Ccz4::Tags::ConformalMetric<DataVector, ::Ccz4::fd::System::volume_dim>,
      ::Tags::deriv<Ccz4::Tags::ConformalMetric<
                        DataVector, ::Ccz4::fd::System::volume_dim, Frame>,
                    tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame>,
      ::Ccz4::fd::Tags::UVector2Plus<DataVector, ::Ccz4::fd::System::volume_dim,
                                     Frame>,
      ::Ccz4::fd::Tags::UVector2Minus<DataVector,
                                      ::Ccz4::fd::System::volume_dim, Frame>,
      ::Ccz4::fd::Tags::UScalar2Plus<DataVector>,
      ::Ccz4::fd::Tags::UScalar2Minus<DataVector>,
      ::Ccz4::fd::Tags::UScalar3Plus<DataVector>,
      ::Ccz4::fd::Tags::UScalar3Minus<DataVector>,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
          ::Ccz4::fd::System::volume_dim, Frame>>>;
  static void function(
      const gsl::not_null<return_type*> result,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          conformal_spatial_metric,
      const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          d_conformal_spatial_metric,
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          u_vector2_plus,
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          u_vector2_minus,
      const Scalar<DataVector>& u_scalar2_plus,
      const Scalar<DataVector>& u_scalar2_minus,
      const Scalar<DataVector>& u_scalar3_plus,
      const Scalar<DataVector>& u_scalar3_minus,
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
          unit_normal_one_form) {
    constraint_characteristic_fields(
        result, conformal_factor, conformal_spatial_metric,
        d_conformal_spatial_metric, u_vector2_plus, u_vector2_minus,
        u_scalar2_plus, u_scalar2_minus, u_scalar3_plus, u_scalar3_minus,
        unit_normal_one_form);
  };
};
/// @}
}  // namespace Ccz4::fd
