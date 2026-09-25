// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/SpectralFilter.hpp"

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.tpp"

namespace {
// The full CCZ4 evolved-variables list (9 original + 4 LDG auxiliary + 4
// boundary second-order fields). Taken from the system so the instantiated
// types are guaranteed to match the executable's
// `system::variables_tag::tags_list`.
using tags_for_filter = Ccz4::fd::System::variables_tag_list;
}  // namespace

template class Filters::runtime::Hypercube<3, tags_for_filter>;
template class Filters::runtime::None<3, tags_for_filter>;
template class Filters::runtime::SphericalShell<tags_for_filter>;
template struct evolution::dg::Initialization::SpectralFilters<3,
                                                               tags_for_filter>;
