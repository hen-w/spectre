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
struct LdgTimeDerivative;
template <typename Frame>
struct ComputeLargestCharacteristicSpeed;

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

  // Auxiliary reduction variables for LDG
  using auxiliary_variables_tags = tmpl::list<
      ::Ccz4::Tags::FieldA<DataVector, 3>, ::Ccz4::Tags::FieldB<DataVector, 3>,
      ::Ccz4::Tags::FieldD<DataVector, 3>, ::Ccz4::Tags::FieldP<DataVector, 3>>;

  // Boundary mode tags: incoming characteristic modes evolved at CRPBC faces
  using boundary_mode_tags = tmpl::list<
      ::Ccz4::fd::Tags::UScalar3Minus<DataVector>,
      ::Ccz4::fd::Tags::UVector2Minus<DataVector, 3, Frame::Inertial>,
      ::Ccz4::fd::Tags::UScalar2Minus<DataVector>,
      ::Ccz4::fd::Tags::UTensorMinus<DataVector, 3, Frame::Inertial>>;

  // Boundary-integrated second-order fields for DirichletCharacteristics BC
  using boundary_second_order_tags =
      tmpl::list<::Ccz4::Tags::BoundaryConformalMetric<DataVector, 3>,
                 ::Ccz4::Tags::BoundaryConformalFactor<DataVector>,
                 ::Ccz4::Tags::BoundaryLapse<DataVector>,
                 ::Ccz4::Tags::BoundaryShift<DataVector, 3>>;

  // Full evolved variables = original 9 + 4 boundary modes + 4 boundary
  // second-order fields
  using evolved_variables_tags =
      tmpl::append<original_evolved_variables_tags, boundary_mode_tags,
                   boundary_second_order_tags>;

  // Variables whose spectral derivatives are computed by the DG infrastructure.
  // This includes all variables in variables_tag: original evolved + auxiliary
  // + boundary modes + boundary second-order fields. Boundary modes and
  // boundary second-order fields are ODE-evolved (spatially constant) so
  // their derivatives are zero, but we include them here because the DG
  // infrastructure's moving-mesh code iterates over all variables and expects
  // derivatives for each.
  using gradient_variables =
      tmpl::append<original_evolved_variables_tags, auxiliary_variables_tags,
                   boundary_mode_tags, boundary_second_order_tags>;

  // variables_tag = gradient_variables (which is all tags in the correct order)
  using variables_tag = ::Tags::Variables<gradient_variables>;

  using variables_tag_list = typename variables_tag::tags_list;

  // gradients_tags is used by the FD path (SoTimeDerivative) for FD
  // derivative computation — must be the original evolved variables only
  using gradients_tags = original_evolved_variables_tags;

  using compute_volume_time_derivative_terms = LdgTimeDerivative;

  using compute_largest_characteristic_speed =
      ComputeLargestCharacteristicSpeed<Frame::Inertial>;
};

namespace Tags {
/// \brief Tags sent for second-order Ccz4 evolution.
using spacetime_reconstruction_tags = System::variables_tag_list;
}  // namespace Tags

}  // namespace Ccz4::fd
