// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <autodiff/common/numbertraits.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/reverse/var.hpp>

#include "Utilities/Simd/Simd.hpp"

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
struct MakeWithValueImpl<autodiff::HigherOrderDual<2, simd::batch<double>>, T> {
  static SPECTRE_ALWAYS_INLINE autodiff::HigherOrderDual<2, simd::batch<double>>
  apply(const T& /* input */, const double value) {
    return {value};
  }
};

template <typename T>
struct MakeWithValueImpl<autodiff::HigherOrderDual<2, double>, T> {
  static SPECTRE_ALWAYS_INLINE autodiff::HigherOrderDual<2, double> apply(
      const T& /* input */, const double value) {
    return {value};
  }
};
}  // namespace MakeWithValueImpls

SPECTRE_ALWAYS_INLINE size_t
get_size(const autodiff::HigherOrderDual<2, simd::batch<double>>& /*t*/) {
  return 1;
}

SPECTRE_ALWAYS_INLINE size_t
get_size(const autodiff::HigherOrderDual<2, double>& /*t*/) {
  return 1;
}

SPECTRE_ALWAYS_INLINE decltype(auto) get_element(
    const autodiff::HigherOrderDual<2, simd::batch<double>>& t,
    const size_t /*i*/) {
  return t;
}

SPECTRE_ALWAYS_INLINE decltype(auto) get_element(
    autodiff::HigherOrderDual<2, simd::batch<double>>& t, const size_t /*i*/) {
  return t;
}

SPECTRE_ALWAYS_INLINE decltype(auto) get_element(
    const autodiff::HigherOrderDual<2, double>& t, const size_t /*i*/) {
  return t;
}

SPECTRE_ALWAYS_INLINE decltype(auto) get_element(
    autodiff::HigherOrderDual<2, double>& t, const size_t /*i*/) {
  return t;
}
