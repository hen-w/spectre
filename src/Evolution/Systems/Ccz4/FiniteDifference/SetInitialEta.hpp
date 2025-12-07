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
 * \brief Sets the ::Ccz4::Tags::Eta in the Gamma-driver condition
 * in Ccz4 evolution system.
 *
 */
struct SetInitialEta : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<::Ccz4::Tags::Eta<DataVector>>;
  using argument_tags = tmpl::list<gr::Tags::Lapse<DataVector>>;

  static void apply(const gsl::not_null<Scalar<DataVector>*> eta,
                    const Scalar<DataVector>& lapse) {
    *eta = make_with_value<Scalar<DataVector>>(get(lapse).size(), 0.0);
  }
};
}  // namespace Ccz4::fd
