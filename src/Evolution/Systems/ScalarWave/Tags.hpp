// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for scalar wave system

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarWave/OptionTags.hpp"
#include "Evolution/Systems/ScalarWave/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;

namespace ScalarWave::Tags {
/*!
 * \brief The scalar field.
 */
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Auxiliary variable which is analytically the negative time derivative
 * of the scalar field.
 * \details If \f$\Psi\f$ is the scalar field then we define
 * \f$\Pi = -\partial_t \Psi\f$
 */
struct Pi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Auxiliary variable which is analytically the spatial derivative of the
 * scalar field.
 * \details If \f$\Psi\f$ is the scalar field then we define
 * \f$\Phi_{i} = \partial_i \Psi\f$
 */
template <size_t Dim>
struct Phi : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

struct ConstraintGamma2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Tag for the one-index constraint of the ScalarWave system
 *
 * For details on how this is defined and computed, see
 * `OneIndexConstraintCompute`.
 */
template <size_t Dim>
struct OneIndexConstraint : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

/// \brief Tag for gamma_2 as a double
struct Gamma2Value : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<ScalarWave::OptionTags::Gamma2>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double gamma2) { return gamma2; }
};

/// \brief Tag for mass as a double
struct MassValue : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<ScalarWave::OptionTags::Mass>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double mass) { return mass; }
};

/*!
 * \brief Tag for the two-index constraint of the ScalarWave system
 *
 * For details on how this is defined and computed, see
 * `TwoIndexConstraintCompute`.
 */
template <size_t Dim>
struct TwoIndexConstraint : db::SimpleTag {
  using type = tnsr::ij<DataVector, Dim, Frame::Inertial>;
};

/// @{
/// \brief Tags corresponding to the characteristic fields of the flat-spacetime
/// scalar-wave system.
///
/// \details For details on how these are defined and computed, \see
/// CharacteristicSpeedsCompute
struct VPsi : db::SimpleTag {
  using type = Scalar<DataVector>;
};
template <size_t Dim>
struct VZero : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};
struct VPlus : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct VMinus : db::SimpleTag {
  using type = Scalar<DataVector>;
};
/// @}

template <size_t Dim>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 4>;
};

template <size_t Dim>
struct CharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<VPsi, VZero<Dim>, VPlus, VMinus>>;
};

template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<Psi, Pi, Phi<Dim>>>;
};

/// The energy density of the scalar wave
template <size_t Dim>
struct EnergyDensity : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The mass potential of the scalar wave
struct MassPotential : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The momentum density of the scalar wave
template <size_t Dim>
struct MomentumDensity : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

}  // namespace ScalarWave::Tags
