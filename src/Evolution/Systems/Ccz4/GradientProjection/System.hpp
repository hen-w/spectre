// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \brief The pruned second-order CCZ4 discontinuous Galerkin scheme with an
 * upwind gradient projection.
 *
 * \details The physical CCZ4 fields are stored and time stepped. Each
 * right-hand side uses the chart potentials \f$u=(H,\phi,\alpha,\beta)\f$
 * with \f$H=\frac12\log\tilde\gamma\f$, the velocities
 * \f$w=(a,K,\Theta,\hat\Gamma,b)\f$ with \f$a=2\mathcal T_H^{-1}\tilde A\f$
 * (only \f$H\f$ and \f$a\f$ are constructed, in local temporaries; the other
 * components are the physical fields) and recomputes the element-local
 * ordinary gradients \f$q_i=D_iu\f$; evaluates the full production CCZ4
 * volume equations through the chart chain rule into temporary targets for
 * \f$(w,q_i)\f$; corrects these targets with the full upwind normal matrix and
 * a scalar jump penalty in a single DG exchange; fits \f$u_t\f$ by an
 * element-local constrained least-squares gradient fit; applies the
 * compatibility correction; and converts all rates to the physical fields.
 * After each step the physical \f$\tilde A,K,\Theta,\hat\Gamma,b\f$ are
 * filtered and the algebraic constraints are enforced.
 */
namespace Ccz4::GradientProjection {
/*!
 * \brief The evolution system.
 *
 * \details The evolved `Variables` hold the 25 physical fields followed by
 * the 30 stored gradients \f$q\f$ (`Tags::PotentialGradients`), the layout
 * shared with the finite-difference elements. Only the physical fields carry
 * nonzero time derivatives into the time stepper; the gradients are
 * overwritten before every use. The volume terms and lifted boundary
 * corrections go into the separate `Tags::RhsTargets` (the
 * `boundary_correction_target_tag`), from which `FinalizeTimeDerivative`
 * computes the physical rates. The face normal is the Euclidean unit covector
 * because the system has no inverse spatial metric tag. The Gamma-driver
 * parameter \f$f\f$ of \f$\partial_t\beta^i=fb^i\f$ is `Ccz4::fd::System::f`,
 * which must be positive.
 */
struct System {
  static constexpr size_t volume_dim = 3;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr bool is_in_flux_conservative_form = false;
  using boundary_conditions_base =
      ::Ccz4::BoundaryConditions::BoundaryCondition;
  using flux_variables = tmpl::list<>;

  using physical_tags = ::Ccz4::fd::System::original_evolved_variables_tags;
  using variables_tag = ComputeVolumeTargets::variables_tag;
  static_assert(
      std::is_same_v<variables_tag, ::Ccz4::fd::System::variables_tag>,
      "The DG and finite-difference elements share the evolved "
      "variables.");
  using variables_tag_list = typename variables_tag::tags_list;
  // The volume terms are evaluated by ComputeVolumeTargets before the generic
  // DG action, which therefore differentiates nothing.
  using gradient_variables = tmpl::list<>;
  using boundary_correction_target_tag = Tags::RhsTargets;

  using compute_volume_time_derivative_terms = ComputeVolumeTargets;
};
}  // namespace Ccz4::GradientProjection
