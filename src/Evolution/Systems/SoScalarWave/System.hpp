// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class SoScalarWaveSystem.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/SoScalarWave/Characteristics.hpp"
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

  using variables_tag =
      ::Tags::Variables<tmpl::list<Tags::Psi, Tags::Pi, Tags::BoundaryPsi>>;
  using flux_variables = tmpl::list<>;
  // Phi_i = d_i Psi is an auxiliary variable: it is recomputed from Psi each
  // step (LDG auxiliary pass) rather than time-evolved. Its first derivative is
  // still needed by the volume time derivative (dt_pi = -d_i Phi^i), so Phi
  // remains in `gradient_variables`, sourced from `variables_tag union
  // auxiliary_variables`.
  using auxiliary_variables = tmpl::list<Tags::Phi<Dim>>;
  using gradient_variables =
      tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>, Tags::BoundaryPsi>;

  using compute_volume_time_derivative_terms = TimeDerivative<Dim>;

  using compute_largest_characteristic_speed =
      Tags::ComputeLargestCharacteristicSpeed;
};
}  // namespace SoScalarWave
