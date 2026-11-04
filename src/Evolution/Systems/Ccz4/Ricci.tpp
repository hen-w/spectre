// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/Ricci.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4 {
template <typename DataType, size_t Dim, typename Frame, typename TensorType>
void spatial_ricci_tensor(
    const gsl::not_null<tnsr::ii<DataType, Dim, Frame>*> result,
    const tnsr::Ijj<DataType, Dim, Frame>& conformal_christoffel_second_kind,
    const tnsr::ij<DataType, Dim, Frame>&
        contracted_d_conformal_christoffel_difference,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const TensorType& d_field_p) {
  constexpr double dim_minus_two = static_cast<double>(Dim) - 2.0;
  using ContractedConformalChristoffel =
      ::Tags::Tempi<0, Dim, Frame, DataType>;
  using CovariantDerivFieldP = ::Tags::Tempii<1, Dim, Frame, DataType>;
  using TraceTerm = ::Tags::TempScalar<2, DataType>;
  TempBuffer<tmpl::list<ContractedConformalChristoffel, CovariantDerivFieldP,
                        TraceTerm>>
      buffer(get_size(get<0, 0>(conformal_spatial_metric)));

  // \tilde{\Gamma}^m_{lm}
  auto& contracted_conformal_christoffel =
      get<ContractedConformalChristoffel>(buffer);
  tenex::evaluate<ti::l>(
      make_not_null(&contracted_conformal_christoffel),
      conformal_christoffel_second_kind(ti::M, ti::l, ti::m));

  // H_{ij} = \partial_{(i} P_{j)} - \tilde{\Gamma}^k_{ij} P_k
  auto& covariant_d_field_p = get<CovariantDerivFieldP>(buffer);
  if constexpr (std::is_same_v<TensorType, tnsr::ii<DataType, Dim, Frame>>) {
    tenex::evaluate<ti::i, ti::j>(
        make_not_null(&covariant_d_field_p),
        d_field_p(ti::i, ti::j) -
            conformal_christoffel_second_kind(ti::K, ti::i, ti::j) *
                field_p(ti::k));
  } else {
    tenex::evaluate<ti::i, ti::j>(
        make_not_null(&covariant_d_field_p),
        0.5 * (d_field_p(ti::i, ti::j) + d_field_p(ti::j, ti::i)) -
            conformal_christoffel_second_kind(ti::K, ti::i, ti::j) *
                field_p(ti::k));
  }

  // \tilde{\gamma}^{kl} (H_{kl} - (n - 2) P_k P_l)
  auto& trace_term = get<TraceTerm>(buffer);
  tenex::evaluate(make_not_null(&trace_term),
                  inverse_conformal_spatial_metric(ti::K, ti::L) *
                      (covariant_d_field_p(ti::k, ti::l) -
                       dim_minus_two * field_p(ti::k) * field_p(ti::l)));

  tenex::evaluate<ti::i, ti::j>(
      result,
      // \tilde{R}_{ij}
      contracted_d_conformal_christoffel_difference(ti::i, ti::j) +
          conformal_christoffel_second_kind(ti::L, ti::i, ti::j) *
              contracted_conformal_christoffel(ti::l) -
          conformal_christoffel_second_kind(ti::L, ti::i, ti::m) *
              conformal_christoffel_second_kind(ti::M, ti::l, ti::j) +
          // Terms of the conformal factor
          dim_minus_two * (covariant_d_field_p(ti::i, ti::j) +
                           field_p(ti::i) * field_p(ti::j)) +
          conformal_spatial_metric(ti::i, ti::j) * trace_term());
}

template <typename DataType, size_t Dim, typename Frame, typename TensorType>
tnsr::ii<DataType, Dim, Frame> spatial_ricci_tensor(
    const tnsr::Ijj<DataType, Dim, Frame>& conformal_christoffel_second_kind,
    const tnsr::ij<DataType, Dim, Frame>&
        contracted_d_conformal_christoffel_difference,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const TensorType& d_field_p) {
  tnsr::ii<DataType, Dim, Frame> result{};
  spatial_ricci_tensor(make_not_null(&result),
                       conformal_christoffel_second_kind,
                       contracted_d_conformal_christoffel_difference,
                       conformal_spatial_metric,
                       inverse_conformal_spatial_metric, field_p, d_field_p);
  return result;
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)
#define TTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE(_, data)                                              \
  template void Ccz4::spatial_ricci_tensor(                               \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*> \
          result,                                                         \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&               \
          conformal_christoffel_second_kind,                              \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>&                \
          contracted_d_conformal_christoffel_difference,                  \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_spatial_metric,                                       \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_conformal_spatial_metric,                               \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_p,        \
      const TTYPE(data)<DTYPE(data), DIM(data), FRAME(data)>& d_field_p); \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                  \
  Ccz4::spatial_ricci_tensor(                                             \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&               \
          conformal_christoffel_second_kind,                              \
      const tnsr::ij<DTYPE(data), DIM(data), FRAME(data)>&                \
          contracted_d_conformal_christoffel_difference,                  \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
          conformal_spatial_metric,                                       \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_conformal_spatial_metric,                               \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& field_p,        \
      const TTYPE(data)<DTYPE(data), DIM(data), FRAME(data)>& d_field_p);

// Instantiations are split into several compilation units to reduce
// compiler memory consumption.
