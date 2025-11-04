// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <autodiff/common/numbertraits.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/reverse/var.hpp>

#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Simd/Simd.hpp"

using BatchType = simd::batch<double>;
using SecondOrderDual = autodiff::HigherOrderDual<2, BatchType>;
using SecondOrderDualNum = autodiff::HigherOrderDual<2, double>;

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

namespace MakeWithValueImpls {
template <typename T>
struct MakeWithValueImpl<autodiff::HigherOrderDual<2, double>, T> {
  static SPECTRE_ALWAYS_INLINE autodiff::HigherOrderDual<2, double> apply(
      const T& /* input */, const double value) {
    return {value};
  }
};

template <typename T>
struct MakeWithValueImpl<autodiff::HigherOrderDual<2, simd::batch<double>>, T> {
  static SPECTRE_ALWAYS_INLINE autodiff::HigherOrderDual<2, simd::batch<double>>
  apply(const T& /* input */, const double value) {
    return {value};
  }
};
}  // namespace MakeWithValueImpls
