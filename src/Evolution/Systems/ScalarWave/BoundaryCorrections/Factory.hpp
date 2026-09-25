// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/ScalarWave/BoundaryCorrections/Central.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryCorrections/UpwindPenalty.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarWave::BoundaryCorrections {
template <size_t Dim>
using standard_boundary_corrections =
    tmpl::list<Central<Dim>, UpwindPenalty<Dim>>;
}  // namespace ScalarWave::BoundaryCorrections
