// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/Ccz4/BoundaryCorrections/LaxFriedrichs.hpp"
#include "Evolution/Systems/Ccz4/BoundaryCorrections/ProposedFlux.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::BoundaryCorrections {
template <size_t Dim>
using standard_boundary_corrections =
    tmpl::list<LaxFriedrichs<Dim>, ProposedFlux<Dim>>;
}  // namespace Ccz4::BoundaryCorrections
