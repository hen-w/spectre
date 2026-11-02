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
CREATE_GET_STATIC_MEMBER_VARIABLE_OR_DEFAULT(
    reuse_auxiliary_finite_difference_ghost_data)
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

/*!
 * \brief If `Metavars` has a `SubcellOptions` member struct and
 * `SubcellOptions::reuse_auxiliary_finite_difference_ghost_data` is `true`
 * then inherits from `std::true_type`, otherwise (also if the flag is absent)
 * inherits from `std::false_type`.
 *
 * \details The flag is for systems with two ghost exchanges per right-hand
 * side (see `active_grids_fixed_at_initialization`) that change only the
 * trailing components of the ghost data of the elements using finite
 * differences between the exchanges, and zero them in the first exchange. It
 * declares that these elements use, from the ghost data of their
 * finite-difference neighbors, only the leading components holding the
 * tensors `SubcellOptions::finite_difference_neighbor_ghost_tags`. Then:
 * - in the first exchange, an element using finite differences slices or
 *   interpolates only these leading components for its finite-difference
 *   neighbors, and sends them in the full ghost-data layout with the
 *   remaining components zero;
 * - in the second exchange, it sends no message to its finite-difference
 *   neighbors and keeps the ghost data that it received from them in the
 *   first exchange, along with their meshes, mortar time step ids and TCI
 *   decisions. Its messages to and from neighbors using DG are unchanged.
 *
 * The RDMP data of the finite-difference neighbors are then received only in
 * the first exchange; with the active grids fixed at initialization there is
 * no troubled-cell indicator to use them. The flag requires
 * `active_grids_fixed_at_initialization` and a system without fluxes.
 * Enabling donor extension with this flag is an error: the optimization does
 * not support `SubcellOptions::enable_extension_directions()`.
 *
 * \note This check is intentionally not inside the `DgSubcell` library so that
 * executables that do not use subcell do not need to link against it.
 */
template <typename Metavars, typename = std::void_t<>>
struct reuse_auxiliary_finite_difference_ghost_data : std::false_type {};

template <typename Metavars>
struct reuse_auxiliary_finite_difference_ghost_data<
    Metavars, std::void_t<typename Metavars::SubcellOptions>>
    : std::bool_constant<
          detail::get_reuse_auxiliary_finite_difference_ghost_data_or_default_v<
              typename Metavars::SubcellOptions, false>> {};

/// `true` if `reuse_auxiliary_finite_difference_ghost_data<Metavars>` inherits
/// from `std::true_type`.
template <typename Metavars>
constexpr bool reuse_auxiliary_finite_difference_ghost_data_v =
    reuse_auxiliary_finite_difference_ghost_data<Metavars>::value;
}  // namespace evolution::dg
