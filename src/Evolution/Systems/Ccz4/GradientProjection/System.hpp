// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
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
 * right-hand side reconstructs the chart coordinates
 * \f$u=(H,\log\phi,\alpha,\beta)\f$ with \f$H=\frac12\log\tilde\gamma\f$,
 * \f$w=(a,K,\Theta,\hat\Gamma,b)\f$ with \f$a=2\mathcal T_H^{-1}\tilde A\f$ and
 * the element-local ordinary gradients \f$q_i=D_iu\f$; evaluates the full
 * production CCZ4 volume equations through the chart chain rule into temporary
 * targets for \f$(w,q_i)\f$; corrects these targets with the full upwind
 * normal matrix and a unit scalar jump penalty in a single DG exchange; fits
 * \f$u_t\f$ by an element-local constrained least-squares gradient fit; applies
 * the compatibility correction; and converts all rates to the physical fields.
 * After each step the physical \f$\tilde A,K,\Theta,\hat\Gamma,b\f$ are
 * filtered and the algebraic constraints are enforced.
 */
namespace Ccz4::GradientProjection {
/*!
 * \brief The evolution system.
 *
 * \details The evolved `Variables` hold the chart scratch fields first (the
 * velocities and gradients are the differentiated prefix) followed by the
 * physical fields. Only the physical fields carry meaningful time
 * derivatives into the time stepper; the chart fields are overwritten before
 * every use (see `Tags::ChartU`). The face normal is the Euclidean unit
 * covector because the system has no inverse spatial metric tag. The
 * Gamma-driver parameter \f$f\f$ of \f$\partial_t\beta^i=fb^i\f$ is
 * `Ccz4::fd::System::f`, which must be positive.
 */
struct System {
  static constexpr size_t volume_dim = 3;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr bool is_in_flux_conservative_form = false;
  using boundary_conditions_base =
      ::Ccz4::BoundaryConditions::BoundaryCondition;
  using flux_variables = tmpl::list<>;

  using chart_tags = tmpl::list<Tags::ChartW, Tags::ChartQ, Tags::ChartU>;
  using physical_tags =
      tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                 ::Ccz4::Tags::ConformalFactor<DataVector>,
                 ::Ccz4::Tags::ATilde<DataVector, 3>,
                 gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 ::Ccz4::Tags::Theta<DataVector>,
                 ::Ccz4::Tags::GammaHat<DataVector, 3>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>;
  using gradient_variables = tmpl::list<Tags::ChartW, Tags::ChartQ>;
  using variables_tag =
      ::Tags::Variables<tmpl::append<chart_tags, physical_tags>>;
  using variables_tag_list = typename variables_tag::tags_list;

  using compute_volume_time_derivative_terms = TimeDerivative;
};
}  // namespace Ccz4::GradientProjection
