// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/DirichletAnalytic.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::BoundaryConditions {
/// Typelist of standard BoundaryConditions
using standard_boundary_conditions = tmpl::list<DirichletAnalytic>;
}  // namespace Ccz4::BoundaryConditions
