// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Add the ::Ccz4::Tags::SpatialZ4ConstraintUp to the DataBox
 *
 */
struct AddUpperSpatialZ4Constraint : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags =
      tmpl::list<::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>>;
  using argument_tags = tmpl::list<gr::Tags::Lapse<DataVector>>;

  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> upper_spatial_z4_constraint,
      const Scalar<DataVector>& lapse) {
    // This is a dummy initial value. The actual Z4 constraint of the initial
    // data is printed at the first time step
    *upper_spatial_z4_constraint = make_with_value<tnsr::I<DataVector, 3>>(
        get(lapse).size(), 0.0);
  }
};
}  // namespace Ccz4::fd
