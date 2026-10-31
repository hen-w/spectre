// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
template <typename Frame>
struct ComputeLargestCharacteristicSpeed;

/*!
 * \brief The second-order CCZ4 system evolved with finite differences on the
 * subcells of the CCZ4 gradient-projection DG-FD scheme.
 *
 * \details The evolved `Variables` (shared with
 * `Ccz4::GradientProjection::System`) hold the 25 physical fields followed by
 * the 30 scratch gradients `Tags::PotentialGradients`. The finite-difference
 * right-hand side, derivatives, reconstruction and filter act on the physical
 * prefix; the gradients are only formed for the DG neighbors' face packages
 * and have zero time derivative.
 */
struct System {
  using flux_variables = tmpl::list<>;
  using boundary_conditions_base = BoundaryConditions::BoundaryCondition;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = 3;
  static constexpr bool is_in_flux_conservative_form = false;
  /* shifting_shift and f should be created from options in the future */
  // whether to add the advective terms in the Gamma-driver condition
  static constexpr bool shifting_shift = false;

  // The free parameter f in the Gamma-driver condition.
  // We assume in cpbc that f has no spatial dependence.
  static constexpr double f = 1.0;

  // Original 9 evolved variables (the order is important as it is assumed
  // in the filter)
  using original_evolved_variables_tags =
      tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                 ::Ccz4::Tags::ConformalFactor<DataVector>,
                 ::Ccz4::Tags::ATilde<DataVector, 3>,
                 gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 ::Ccz4::Tags::Theta<DataVector>,
                 ::Ccz4::Tags::GammaHat<DataVector, 3>,
                 // gauge variables
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>;

  // The physical fields followed by the scratch gradients of the CCZ4
  // gradient-projection scheme. The physical fields must stay first and the
  // gauge fields (lapse, shift, b) last among them: finite-difference
  // derivatives, reconstruction and filtering act on that prefix.
  using physical_and_gradient_tags =
      tmpl::push_back<original_evolved_variables_tags,
                      Tags::PotentialGradients>;

  using variables_tag = ::Tags::Variables<physical_and_gradient_tags>;
  using variables_tag_list = typename variables_tag::tags_list;

  // The variables differentiated by finite differences: the physical fields.
  using gradients_tags = original_evolved_variables_tags;

  // The physical fields whose first derivatives define the auxiliary fields
  // A_i, B_k^i, D_kij and P_i, so the right-hand side needs their second
  // derivatives. Their first derivatives also form the stored gradients
  // `Tags::PotentialGradients` sent to the DG neighbors.
  using second_derivative_tags =
      tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                 ::Ccz4::Tags::ConformalFactor<DataVector>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>>;

  using compute_largest_characteristic_speed =
      ComputeLargestCharacteristicSpeed<Frame::Inertial>;
};

namespace Tags {
/// \brief Tags sent for second-order Ccz4 evolution.
using spacetime_reconstruction_tags = System::variables_tag_list;
}  // namespace Tags

}  // namespace Ccz4::fd
