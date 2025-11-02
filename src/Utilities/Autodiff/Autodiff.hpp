// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <autodiff/common/numbertraits.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/reverse/var.hpp>

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Utilities/Simd/Simd.hpp"
#include "Utilities/TMPL.hpp"

using ad_supported_maps = tmpl::list<>;

namespace autodiff::detail {
/// Template specialization for simd::batch<double> to treat it as arithmetic.
// The major difficulty we have with DataVector working with autodiff is
// DataVector does not have a scalar broadcast constructor, which is expected
// in the seed function for autodiff dual type.
template <>
struct ArithmeticTraits<simd::batch<double>> {
  static constexpr bool isArithmetic = true;
};
}  // namespace autodiff::detail
