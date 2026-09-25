// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/ScalarWave/System.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Factory.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarWave {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief A `tmpl::list` of all concrete `Filters::runtime` filter types for
 * the first-order scalar wave system in `Dim` spatial dimensions.
 *
 * Only `Filters::runtime::None` and `Filters::runtime::Hypercube` are
 * registered. `Filters::runtime::SphericalShell` would additionally require
 * the system-specific `ylm::TensorYlm` filter specializations; it is not
 * registered because ScalarWave is not filtered on Ylm shells on this
 * branch. `None` supports every mesh, so Ylm-shell domains parse and run
 * with an all-`None` filter configuration.
 */
template <size_t Dim>
using all_runtime_filters = Filters::runtime::all_filters<
    Dim, typename System<Dim>::variables_tag::tags_list>;
}  // namespace ScalarWave
