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
#include "Evolution/Systems/Ccz4/System.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
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
 * \brief Apply a Kreiss-Oliger filter to \f$g_{ab}\f$, \f$\Phi_{iab}\f$, and
 * \f$\Pi_{ab}\f$.
 */
/* do we filter all evolved variables in Ccz4?? */
void spacetime_kreiss_oliger_filter(
    gsl::not_null<Variables<typename Ccz4::System::variables_tag::tags_list>*>
        result,
    const Variables<typename Ccz4::System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const Mesh<3>& volume_mesh, size_t order, double epsilon);
}  // namespace Ccz4::fd
