// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Hessian.hpp"

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "Utilities/Autodiff/Autodiff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::Hessian {
template <size_t Dim, typename TargetFrame>
auto inv_hessian(
    const ElementMap<Dim, TargetFrame>& map,
    const ::InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            TargetFrame>& inverse_jac,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& source_coords)
    -> ::InverseHessian<DataVector, Dim, Frame::ElementLogical, TargetFrame> {
  // first compute hessian
  using BatchType = simd::batch<double>;
  using SecondOrderDual = autodiff::HigherOrderDual<2, BatchType>;
  using SecondOrderDualNum = autodiff::HigherOrderDual<2, double>;

  const size_t num_pts = get<0>(source_coords).size();
  ::Hessian<DataVector, Dim, Frame::ElementLogical, TargetFrame> hessian{
      num_pts};
  ::InverseHessian<DataVector, Dim, Frame::ElementLogical, TargetFrame>
      inverse_hessian{num_pts};

  // manual vectorization with xsimd
  const size_t vec_end = (num_pts / BatchType::size) * BatchType::size;
  for (size_t pts_index = 0; pts_index < vec_end;
       pts_index += BatchType::size) {
    tnsr::I<SecondOrderDual, Dim, Frame::ElementLogical> dual_source_coords;

    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          ((get<Is>(dual_source_coords) =
                BatchType::load_aligned(&(get<Is>(source_coords))[pts_index])),
           ...);
        }
        (std::make_index_sequence<Dim>{});

        autodiff::detail::seed<1>(dual_source_coords.get(i), 1.0);
        autodiff::detail::seed<2>(dual_source_coords.get(j), 1.0);

        const auto dual_target_coords = map(dual_source_coords);
        for (size_t k = 0; k < Dim; ++k) {
          const auto deriv_kij =
              autodiff::derivative<2>(dual_target_coords.get(k));
          deriv_kij.store_aligned(&hessian.get(k, i, j)[pts_index]);
        }
      }
    }
  }
  // dealing with the tail
  for (size_t pts_index = vec_end; pts_index < num_pts; ++pts_index) {
    tnsr::I<SecondOrderDualNum, Dim, Frame::ElementLogical> dual_source_coords;

    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          ((get<Is>(dual_source_coords) =
                gsl::at(get<Is>(source_coords), pts_index)),
           ...);
        }
        (std::make_index_sequence<Dim>{});
        autodiff::detail::seed<1>(dual_source_coords.get(i), 1.0);
        autodiff::detail::seed<2>(dual_source_coords.get(j), 1.0);

        const auto dual_target_coords = map(dual_source_coords);
        for (size_t k = 0; k < Dim; ++k) {
          hessian.get(k, i, j)[pts_index] =
              autodiff::derivative<2>(dual_target_coords.get(k));
        }
      }
    }
  }

  // piece together the inverse hessian from hessian and inverse jacobian
  ::tenex::evaluate<ti::I, ti::m, ti::n>(
      make_not_null(&inverse_hessian),
      -1.0 * inverse_jac(ti::I, ti::j) * inverse_jac(ti::K, ti::m) *
          inverse_jac(ti::L, ti::n) * hessian(ti::J, ti::k, ti::l));

  return inverse_hessian;
}
}  // namespace domain::Hessian

// For dual frame evolutions the ElementMap only goes to the grid frame
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

// Explicitly instantiate inv_hessian for the supported dimensions and frames
#define INSTANTIATION(r, data)                                                \
  template ::InverseHessian<DataVector, GET_DIM(data), Frame::ElementLogical, \
                            GET_FRAME(data)>                                  \
  domain::Hessian::inv_hessian<GET_DIM(data), GET_FRAME(data)>(               \
      const ElementMap<GET_DIM(data), GET_FRAME(data)>&,                      \
      const ::InverseJacobian<DataVector, GET_DIM(data),                      \
                              Frame::ElementLogical, GET_FRAME(data)>&,       \
      const tnsr::I<DataVector, GET_DIM(data), Frame::ElementLogical>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (Frame::Inertial, Frame::Grid))

#undef GET_DIM
#undef GET_FRAME
#undef INSTANTIATION
