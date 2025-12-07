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
 * \brief Sets the ::Ccz4::Tags::K0 in the 1+log slicing condition
 * in Ccz4 evolution system to the initial value of the trace of
 * the extrinsic curvature.
 */
struct SetK0 : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<::Ccz4::Tags::K0<DataVector>>;
  using argument_tags =
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataVector>>;

  static void apply(const gsl::not_null<Scalar<DataVector>*> k_0,
                    const Scalar<DataVector>& trace_extrinsic_curvature) {
    // there are two kinds of 1+log slicing conditions.
    // in trumpet Schwarzschild, we set K0 to zero instead of
    // the initial value of K
    *k_0 = make_with_value<Scalar<DataVector>>(trace_extrinsic_curvature, 0.0);
  }
};
}  // namespace Ccz4::fd
