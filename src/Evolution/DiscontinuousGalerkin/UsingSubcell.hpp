// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits/CreateGetStaticMemberVariableOrDefault.hpp"

namespace evolution::dg {
/*!
 * \brief If `Metavars` has a `SubcellOptions` member struct and
 * `SubcellOptions::subcell_enabled` is `true` then inherits from
 * `std::true_type`, otherwise inherits from `std::false_type`.
 *
 * \note This check is intentionally not inside the `DgSubcell` library so that
 * executables that do not use subcell do not need to link against it.
 */
template <typename Metavars, typename = std::void_t<>>
struct using_subcell : std::false_type {};

/// \cond
template <typename Metavars>
struct using_subcell<Metavars, std::void_t<typename Metavars::SubcellOptions>>
    : std::bool_constant<Metavars::SubcellOptions::subcell_enabled> {};
/// \endcond

/*!
 * \brief If `Metavars` has a `SubcellOptions` member struct and
 * `SubcellOptions::subcell_enabled` is `true` then is `true`, otherwise
 * `false`.
 *
 * \note This check is intentionally not inside the `DgSubcell` library so that
 * executables that do not use subcell do not need to link against it.
 */
template <typename Metavars>
constexpr bool using_subcell_v = using_subcell<Metavars>::value;

namespace detail {
CREATE_GET_STATIC_MEMBER_VARIABLE_OR_DEFAULT(
    active_grids_fixed_at_initialization)
}  // namespace detail

/*!
 * \brief If `Metavars` has a `SubcellOptions` member struct and
 * `SubcellOptions::active_grids_fixed_at_initialization` is `true` then
 * inherits from `std::true_type`, otherwise (also if the flag is absent)
 * inherits from `std::false_type`.
 *
 * \details The flag declares that the active grid (DG or finite differences)
 * of every element is the one selected at initialization by
 * `evolution::dg::subcell::Actions::SetSubcellGrid` and never changes (no
 * troubled-cell indicator, no AMR). Ghost data are then sent only to
 * neighbors that use them, according to the neighbor active grids set at
 * initialization by `evolution::dg::subcell::Actions::SetNeighborActiveGrids`:
 * elements using DG send none to neighbors using DG, and elements using finite
 * differences send none to neighbors using DG in the first of two ghost
 * exchanges (`IsAuxiliary`), which those neighbors discard. The messages
 * themselves (destinations, number, temporal ids) are unchanged.
 *
 * \note This check is intentionally not inside the `DgSubcell` library so that
 * executables that do not use subcell do not need to link against it.
 */
template <typename Metavars, typename = std::void_t<>>
struct active_grids_fixed_at_initialization : std::false_type {};

template <typename Metavars>
struct active_grids_fixed_at_initialization<
    Metavars, std::void_t<typename Metavars::SubcellOptions>>
    : std::bool_constant<
          detail::get_active_grids_fixed_at_initialization_or_default_v<
              typename Metavars::SubcellOptions, false>> {};

/// `true` if `active_grids_fixed_at_initialization<Metavars>` inherits from
/// `std::true_type`.
template <typename Metavars>
constexpr bool active_grids_fixed_at_initialization_v =
    active_grids_fixed_at_initialization<Metavars>::value;
}  // namespace evolution::dg
