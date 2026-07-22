// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for second-order scalar wave system

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/SoScalarWave/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;

namespace SoScalarWave::Tags {

/*!
 * \brief The scalar field.
 */
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The negative time derivative of the scalar field.
 * \details If \f$\Psi\f$ is the scalar field then
 * \f$\Pi = -\partial_t \Psi\f$.
 */
struct Pi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Auxiliary variable, the spatial derivative of the scalar field.
 * \details If \f$\Psi\f$ is the scalar field then
 * \f$\Phi_i = \partial_i \Psi\f$.
 */
template <size_t Dim>
struct Phi : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

/*!
 * \brief Boundary-evolved copy of Psi, integrated via
 * \f$\partial_t \text{BoundaryPsi} = -\Pi_{\text{boundary}}\f$.
 */
struct BoundaryPsi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct NormalDotPhi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct PsiTimesNormal : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

/// @{
/// \brief Tags corresponding to the characteristic fields of the second-order
/// scalar-wave system.
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
}  // namespace SoScalarWave::Tags
