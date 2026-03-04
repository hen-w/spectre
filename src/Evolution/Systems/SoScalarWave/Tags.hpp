// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for second-order scalar wave system

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarWave/TagsDeclarations.hpp"

class DataVector;

namespace SoScalarWave::Tags {
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

template <size_t Dim>
struct Phi : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

struct NormalDotPhi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct PsiTimesNormal : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};
}  // namespace SoScalarWave::Tags
