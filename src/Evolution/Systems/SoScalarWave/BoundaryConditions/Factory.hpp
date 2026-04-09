// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/DirichletCharacteristics.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/TimeDerivativeDirichlet.hpp"
#include "Utilities/TMPL.hpp"

namespace SoScalarWave::BoundaryConditions {
/// Typelist of standard BoundaryConditions
template <size_t Dim>
using standard_boundary_conditions =
    tmpl::list<DirichletAnalytic<Dim>, DirichletCharacteristics<Dim>,
               TimeDerivativeDirichlet<Dim>,
               domain::BoundaryConditions::Periodic<BoundaryCondition<Dim>>>;
}  // namespace SoScalarWave::BoundaryConditions
