// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/SoScalarWave/BoundaryCorrections/CgCollocation.hpp"
#include "Utilities/TMPL.hpp"

namespace SoScalarWave::BoundaryCorrections {
template <size_t Dim>
using standard_boundary_corrections = tmpl::list<CgCollocation<Dim>>;
}  // namespace SoScalarWave::BoundaryCorrections
