// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/ConstraintsRadiationPreserving.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/CRPBCExp1.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/CRPBCExp2.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/CRPBCExp3.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/DirichletCharacteristics.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Sommerfeld.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::BoundaryConditions {
/// Typelist of standard BoundaryConditions
using standard_boundary_conditions =
    tmpl::list<DirichletAnalytic, DirichletCharacteristics,
               domain::BoundaryConditions::Periodic<BoundaryCondition>,
               Sommerfeld, ConstraintsRadiationPreserving, CRPBCExp1,
               CRPBCExp2, CRPBCExp3>;
}  // namespace Ccz4::BoundaryConditions
