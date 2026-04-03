// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Sets the ::Ccz4::Tags::Eta in the Gamma-driver condition
 * in Ccz4 evolution system using the constant value from
 * ::Ccz4::fd::Tags::EtaConstant.
 *
 */
struct SetInitialEta : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<::Ccz4::Tags::Eta<DataVector>>;
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>, ::Ccz4::fd::Tags::EtaConstant>;

  static void apply(const gsl::not_null<Scalar<DataVector>*> eta,
                    const Scalar<DataVector>& lapse,
                    const double eta_constant) {
    *eta = make_with_value<Scalar<DataVector>>(get(lapse).size(), eta_constant);
  }
};
}  // namespace Ccz4::fd
