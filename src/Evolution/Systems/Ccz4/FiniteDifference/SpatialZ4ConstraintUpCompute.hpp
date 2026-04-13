// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/Z4Constraint.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Compute tag for the upper spatial Z4 constraint
 * \f$Z^i = \frac{1}{2}\phi^2 (\hat{\Gamma}^i - \tilde{\Gamma}^i)\f$.
 *
 * \details Recomputed on demand from the evolved variables
 * `Ccz4::Tags::ConformalFactor`, `Ccz4::Tags::ConformalMetric`,
 * `Ccz4::Tags::GammaHat`, and `Ccz4::Tags::FieldD`. Used at observation
 * time to monitor the Z4 constraint of the current state.
 */
struct SpatialZ4ConstraintUpCompute
    : ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>,
      db::ComputeTag {
  using base = ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::Ccz4::Tags::ConformalFactor<DataVector>,
                 ::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                 ::Ccz4::Tags::GammaHat<DataVector, 3>,
                 ::Ccz4::Tags::FieldD<DataVector, 3>>;

  static void function(
      const gsl::not_null<return_type*> upper_spatial_z4_constraint,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::ii<DataVector, 3>& conformal_metric,
      const tnsr::I<DataVector, 3>& gamma_hat,
      const tnsr::ijj<DataVector, 3>& field_d) {
    const size_t num_pts = get(conformal_factor).size();

    // half phi^2
    Scalar<DataVector> half_conformal_factor_squared(num_pts);
    get(half_conformal_factor_squared) =
        0.5 * square(get(conformal_factor));

    // inverse conformal metric
    const auto det_and_inv = determinant_and_inverse(conformal_metric);
    const tnsr::II<DataVector, 3>& inv_conformal_metric = det_and_inv.second;

    // conformal Christoffel of the second kind Gamma-tilde^k_{ij}
    tnsr::Ijj<DataVector, 3> conformal_christoffel_second_kind(num_pts);
    ::Ccz4::conformal_christoffel_second_kind(
        make_not_null(&conformal_christoffel_second_kind), inv_conformal_metric,
        field_d);

    // contracted conformal Christoffel Gamma-tilde^i = gamma^{jk} Gamma^i_{jk}
    tnsr::I<DataVector, 3> contracted_conformal_christoffel_second_kind(
        num_pts);
    ::Ccz4::contracted_conformal_christoffel_second_kind(
        make_not_null(&contracted_conformal_christoffel_second_kind),
        inv_conformal_metric, conformal_christoffel_second_kind);

    // gamma_hat^i - Gamma-tilde^i
    tnsr::I<DataVector, 3> gamma_hat_minus_contracted_conformal_christoffel(
        num_pts);
    ::tenex::evaluate<ti::I>(
        make_not_null(&gamma_hat_minus_contracted_conformal_christoffel),
        gamma_hat(ti::I) -
            contracted_conformal_christoffel_second_kind(ti::I));

    // Z^i = (1/2) phi^2 (gamma_hat^i - Gamma-tilde^i)
    ::Ccz4::upper_spatial_z4_constraint(
        upper_spatial_z4_constraint, half_conformal_factor_squared,
        gamma_hat_minus_contracted_conformal_christoffel);
  }
};
}  // namespace Ccz4::fd
