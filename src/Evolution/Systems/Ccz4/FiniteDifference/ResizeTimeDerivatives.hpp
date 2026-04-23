// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Resize db::add_tag_prefix<::Tags::dt, variables_tag> to subcell mesh
 *
 */
struct ResizeTimeDerivatives : tt::ConformsTo<db::protocols::Mutator> {
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, System::variables_tag>;
  using return_tags = tmpl::list<dt_variables_tag>;
  using argument_tags = tmpl::list<gr::Tags::Lapse<DataVector>>;

  static void apply(
      const gsl::not_null<typename dt_variables_tag::type*> dt_vars_ptr,
      const Scalar<DataVector>& lapse) {
    // dummy values; the actual values are printed at the next step
    dt_vars_ptr->initialize(get(lapse).size(), 0.0);
  }
};
}  // namespace Ccz4::fd
