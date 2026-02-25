// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class SoScalarWaveSystem.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/SoScalarWave/TimeDerivative.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the second-order scalar wave equations.
 *
 * \f{aligned*}
 * \partial_t \Pi &= -\delta^{ij}\partial_i \partial_j \Psi \\
 * \partial_t \Psi &= -\Pi
 * \f}
 */
namespace SoScalarWave {

template <size_t Dim>
struct System {
  using boundary_conditions_base = BoundaryConditions::BoundaryCondition<Dim>;

  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;
  static constexpr bool use_cg_collocation_scheme = true;

  using variables_tag = ::Tags::Variables<tmpl::list<Tags::Psi, Tags::Pi>>;
  using flux_variables = tmpl::list<>;
  // We will compute the first and second derivatives of
  // Psi manually instead of using the DG infrastructure
  using gradient_variables = tmpl::list<>;

  using compute_volume_time_derivative_terms = TimeDerivative<Dim>;
};
}  // namespace SoScalarWave
