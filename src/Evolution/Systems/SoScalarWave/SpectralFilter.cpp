// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/FilledCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/HollowCylinder.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// The spectral filter acts on the evolved variables, matching the filter
// registered by the executable (`Filters::Filter<Dim,
// variables_tag::tags_list>`). Phi is an auxiliary variable and is not
// filtered.
template <size_t Dim>
using tags_for_filter =
    typename SoScalarWave::System<Dim>::variables_tag::tags_list;
}  // namespace

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template class Filters::Hypercube<DIM(data), tags_for_filter<DIM(data)>>; \
  template class Filters::None<DIM(data), tags_for_filter<DIM(data)>>;      \
  template struct evolution::dg::Initialization::SpectralFilters<           \
      DIM(data), tags_for_filter<DIM(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

template class Filters::HollowCylinder<tags_for_filter<3>>;
template class Filters::FilledCylinder<tags_for_filter<3>>;

#undef DIM
#undef INSTANTIATE
