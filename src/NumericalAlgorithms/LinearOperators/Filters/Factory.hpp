// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "Utilities/TMPL.hpp"

namespace Filters::runtime {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief A `tmpl::list` of all concrete
 * `Filters::runtime::Filter<Dim, TagList>` subclasses available for a given
 * `Dim` and `TagList`.
 *
 * Use this alias in a metavariables `factory_creation::factory_classes` map to
 * register all filters without having to enumerate them individually:
 * ```cpp
 * tmpl::pair<Filters::runtime::Filter<volume_dim, FilterTagList>,
 *            Filters::runtime::all_filters<volume_dim, FilterTagList>>,
 * ```
 *
 * Note: `Filters::runtime::SphericalShell` is not included here because it
 * requires system-specific instantiation of
 * `ylm::TensorYlm::apply_tensor_ylm_filter`. Executables that support it
 * must add it to their `factory_classes` manually and include the
 * appropriate system-specific header.
 */
template <size_t Dim, typename TagList>
using all_filters = tmpl::list<Hypercube<Dim, TagList>, None<Dim, TagList>>;
}  // namespace Filters::runtime
