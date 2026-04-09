// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename>
class Variables;

namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
/// \endcond

namespace SoScalarWave {
/// @{
/*!
 * \brief Compute the characteristic speeds for the second-order scalar wave
 * system.
 *
 * The characteristic speeds are:
 * \f{align*}
 * \lambda_{\hat \psi} =& -v_g \cdot n \\
 * \lambda_{\hat 0} =& -v_g \cdot n \\
 * \lambda_{\hat \pm} =& \pm 1 - v_g \cdot n
 * \f}
 *
 * where \f$v_g\f$ is the mesh velocity (zero if not provided).
 */
template <size_t Dim>
std::array<DataVector, 4> characteristic_speeds(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity);

template <size_t Dim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity);

/// Overloads without mesh velocity (equivalent to mesh_velocity = nullopt).
template <size_t Dim>
std::array<DataVector, 4> characteristic_speeds(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form);

template <size_t Dim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form);

namespace Tags {
template <size_t Dim>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds<Dim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  static void function(
      gsl::not_null<return_type*> char_speeds,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
    characteristic_speeds(char_speeds, unit_normal_one_form);
  }
};
}  // namespace Tags
/// @}

/// @{
/*!
 * \brief Computes characteristic fields from evolved fields for the
 * second-order scalar wave system.
 *
 * The characteristic fields are:
 * \f{align*}
 * v^{\hat \psi} =& \psi \\
 * v^{\hat 0}_{i} =& (\delta^k_i - n_i n^k) \Phi_{k} \\
 * v^{\hat \pm} =& \Pi \pm n^i \Phi_{i}
 * \f}
 *
 * The inverse relations (evolved from characteristic fields) are:
 * \f{align*}
 * \psi =& v^{\hat \psi}, \\
 * \Pi =& \frac{1}{2}(v^{\hat +} + v^{\hat -}), \\
 * \Phi_{i} =& \frac{1}{2}(v^{\hat +} - v^{\hat -}) n_i + v^{\hat 0}_{i}.
 * \f}
 */
template <size_t Dim>
Variables<tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>
characteristic_fields(
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form);

template <size_t Dim>
void characteristic_fields(
    gsl::not_null<Variables<
        tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>*>
        char_fields,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form);

namespace Tags {
template <size_t Dim>
struct CharacteristicFieldsCompute : Tags::CharacteristicFields<Dim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicFields<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Psi, Pi, Phi<Dim>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  static void function(
      const gsl::not_null<return_type*> char_fields,
      const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
    characteristic_fields(char_fields, psi, pi, phi, unit_normal_one_form);
  };
};
}  // namespace Tags
/// @}

/// @{
/*!
 * \brief Compute evolved fields from characteristic fields.
 *
 * For expressions used here to compute evolved fields from characteristic ones,
 * see \ref Tags::CharacteristicFieldsCompute.
 */
template <size_t Dim>
Variables<tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>>
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form);

template <size_t Dim>
void evolved_fields_from_characteristic_fields(
    gsl::not_null<Variables<tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>>*>
        evolved_fields,
    const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form);

namespace Tags {
template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFieldsCompute
    : Tags::EvolvedFieldsFromCharacteristicFields<Dim>,
      db::ComputeTag {
  using base = Tags::EvolvedFieldsFromCharacteristicFields<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  static void function(
      const gsl::not_null<return_type*> evolved_fields,
      const Scalar<DataVector>& v_psi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
    evolved_fields_from_characteristic_fields(evolved_fields, v_psi, v_zero,
                                              v_plus, v_minus,
                                              unit_normal_one_form);
  };
};

struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};

/// Compute the maximum magnitude of the characteristic speeds.
struct ComputeLargestCharacteristicSpeed : LargestCharacteristicSpeed,
                                           db::ComputeTag {
  using argument_tags = tmpl::list<>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  SPECTRE_ALWAYS_INLINE static constexpr void function(
      const gsl::not_null<double*> speed) {
    *speed = 1.0;
  }
};
}  // namespace Tags
/// @}
}  // namespace SoScalarWave
