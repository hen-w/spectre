// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/SpectralFilter.hpp"

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"

namespace {
using tags_1d = typename ScalarWave::System<1>::variables_tag::tags_list;
using tags_2d = typename ScalarWave::System<2>::variables_tag::tags_list;
using tags_3d = typename ScalarWave::System<3>::variables_tag::tags_list;
}  // namespace

template class Filters::runtime::Hypercube<1, tags_1d>;
template class Filters::runtime::None<1, tags_1d>;
template struct evolution::dg::Initialization::SpectralFilters<1, tags_1d>;
template class Filters::runtime::Hypercube<2, tags_2d>;
template class Filters::runtime::None<2, tags_2d>;
template struct evolution::dg::Initialization::SpectralFilters<2, tags_2d>;
template class Filters::runtime::Hypercube<3, tags_3d>;
template class Filters::runtime::None<3, tags_3d>;
template struct evolution::dg::Initialization::SpectralFilters<3, tags_3d>;
