// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

namespace evolution::dg::Actions::detail {
CREATE_HAS_TYPE_ALIAS(boundary_conditions_base)
CREATE_HAS_TYPE_ALIAS_V(boundary_conditions_base)

CREATE_HAS_TYPE_ALIAS(inverse_spatial_metric_tag)
CREATE_HAS_TYPE_ALIAS_V(inverse_spatial_metric_tag)

template <bool HasInverseSpatialMetricTag = false>
struct inverse_spatial_metric_tag_impl {
  template <typename System>
  using f = tmpl::list<>;
};

template <>
struct inverse_spatial_metric_tag_impl<true> {
  template <typename System>
  using f = tmpl::list<typename System::inverse_spatial_metric_tag>;
};

template <typename System>
using inverse_spatial_metric_tag = typename inverse_spatial_metric_tag_impl<
    has_inverse_spatial_metric_tag_v<System>>::template f<System>;

CREATE_GET_TYPE_ALIAS_OR_DEFAULT(auxiliary_variables)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(dg_auxiliary_boundary_terms_volume_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(dg_auxiliary_package_data_temporary_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(dg_auxiliary_package_data_volume_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(dg_auxiliary_package_field_tags)

// `gradient_variables` is prepended since `partial_derivatives` requires the
// differentiated tags to be the *leading* tags of the source `Variables`.
template <typename System>
using evolved_and_auxiliary_vars_tags = tmpl::remove_duplicates<
    tmpl::append<typename System::gradient_variables,
                 typename System::variables_tag::tags_list,
                 get_auxiliary_variables_or_default_t<System, tmpl::list<>>>>;

template <bool HasPrimitiveVars = false>
struct get_primitive_vars {
  template <typename BoundaryCorrection>
  using f = tmpl::list<>;

  template <typename BoundaryCondition>
  using boundary_condition_interior_tags = tmpl::list<>;
};

template <>
struct get_primitive_vars<true> {
  template <typename BoundaryCorrection>
  using f = typename BoundaryCorrection::dg_package_data_primitive_tags;

  template <typename BoundaryCondition>
  using boundary_condition_interior_tags =
      typename BoundaryCondition::dg_interior_primitive_variables_tags;
};

template <bool HasPrimitiveAndConservativeVars, typename BoundaryCorrection>
using boundary_correction_primitive_tags = typename get_primitive_vars<
    HasPrimitiveAndConservativeVars>::template f<BoundaryCorrection>;

template <bool HasPrimitiveAndConservativeVars, typename BoundaryCondition>
using boundary_condition_primitive_tags =
    typename get_primitive_vars<HasPrimitiveAndConservativeVars>::
        template boundary_condition_interior_tags<BoundaryCondition>;

template <typename BoundaryCorrection, typename = std::void_t<>>
struct interior_tags_for_boundary_correction {
  using type = tmpl::list<>;
};

template <typename BoundaryCorrection>
struct interior_tags_for_boundary_correction<
    BoundaryCorrection,
    std::void_t<typename BoundaryCorrection::
                    dg_project_from_interior_for_boundary_condition>> {
  using type = typename BoundaryCorrection::
      dg_project_from_interior_for_boundary_condition;
};

template <typename BoundaryCondition, typename = std::void_t<>>
struct derivative_tags_for_boundary_condition {
  using type = tmpl::list<>;
};

template <typename BoundaryCondition>
struct derivative_tags_for_boundary_condition<
    BoundaryCondition,
    std::void_t<typename BoundaryCondition::dg_interior_derivative_tags>> {
  using type = typename BoundaryCondition::dg_interior_derivative_tags;
};

template <typename System, bool = System::has_primitive_and_conservative_vars>
struct get_primitive_vars_tags_from_system_impl {
  using type = typename System::primitive_variables_tag::tags_list;
};

template <typename System>
struct get_primitive_vars_tags_from_system_impl<System, false> {
  using type = tmpl::list<>;
};

/// Returns a `tmpl::list` of the primitive tags. The list is empty if the
/// system does not have primitive tags.
template <typename System>
using get_primitive_vars_tags_from_system =
    typename get_primitive_vars_tags_from_system_impl<System>::type;

template <typename BoundaryCondition, typename = std::void_t<>>
struct get_dt_vars_from_boundary_condition_impl {
  using type = tmpl::list<>;
};

template <typename BoundaryCondition>
struct get_dt_vars_from_boundary_condition_impl<
    BoundaryCondition,
    std::void_t<typename BoundaryCondition::dg_interior_dt_vars_tags>> {
  using type = typename BoundaryCondition::dg_interior_dt_vars_tags;
};

/// Returns the `dg_interior_dt_vars_tags` if the boundary condition specifies
/// them, otherwise returns an empty list.
template <typename BoundaryCondition>
using get_dt_vars_from_boundary_condition =
    typename get_dt_vars_from_boundary_condition_impl<BoundaryCondition>::type;

template <typename BoundaryCondition, typename = std::void_t<>>
struct get_deriv_vars_from_boundary_condition_impl {
  using type = tmpl::list<>;
};

template <typename BoundaryCondition>
struct get_deriv_vars_from_boundary_condition_impl<
    BoundaryCondition,
    std::void_t<typename BoundaryCondition::dg_interior_deriv_vars_tags>> {
  using type = typename BoundaryCondition::dg_interior_deriv_vars_tags;
};

/// Returns the `dg_interior_deriv_vars_tags` if the boundary condition
/// specifies them, otherwise returns an empty list.
template <typename BoundaryCondition>
using get_deriv_vars_from_boundary_condition =
    typename get_deriv_vars_from_boundary_condition_impl<
        BoundaryCondition>::type;

// The interior inputs to a boundary condition's
// `boundary_field_time_derivatives` method (the boundary-evolved-fields
// facility). They are declared separately from the `dg_ghost` interior inputs
// so that a boundary condition can request a different projected interior state
// for its boundary-field time derivative. A boundary condition that does not
// opt into boundary-evolved fields defines none of them, so these
// "detect-or-default" metafunctions yield the declared list when present and an
// empty `tmpl::list<>` otherwise. Per the facility design the assembled
// interior inputs must be a subset of the interior face fields projected for
// this boundary condition. `apply_boundary_condition_impl` only compile-checks
// that each tag is a member of the interior-face `Variables` type, not that it
// was actually projected; in practice a boundary condition lists here only tags
// it also feeds to `dg_ghost`, which are projected (a member-but-unprojected
// tag would read signaling NaN, trapped in debug).
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(
    boundary_field_time_derivatives_evolved_variables_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(boundary_field_time_derivatives_primitive_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(boundary_field_time_derivatives_temporary_tags)

// The assembled interior inputs to `boundary_field_time_derivatives`, in the
// order evolved, primitive, temporary; empty for a non-opting boundary
// condition.
template <typename BoundaryCondition>
using boundary_field_time_derivatives_interior_tags = tmpl::append<
    get_boundary_field_time_derivatives_evolved_variables_tags_or_default_t<
        BoundaryCondition, tmpl::list<>>,
    get_boundary_field_time_derivatives_primitive_tags_or_default_t<
        BoundaryCondition, tmpl::list<>>,
    get_boundary_field_time_derivatives_temporary_tags_or_default_t<
        BoundaryCondition, tmpl::list<>>>;

// Detect whether a boundary condition defines the facility's derivative member
// function `boundary_field_time_derivatives`, to fail loud on a boundary
// condition that defines the method but forgets to opt in via
// `boundary_evolved_variables` (without the opt-in the facility block is
// discarded and the method is silently never called). SpECTRE's member-function
// trait `CREATE_IS_CALLABLE` does not apply here: it tests callability with a
// specific argument list, and this method's argument types derive from the very
// `boundary_evolved_variables` / `boundary_field_time_derivatives_*_tags`
// declarations a mis-wired boundary condition omits, so those types are unknown
// at the point we must check. We therefore detect the member's existence with
// the standard SFINAE detection idiom -- the same `void_t` structure
// `CREATE_IS_CALLABLE` and `CREATE_HAS_TYPE_ALIAS` are built from, keyed on the
// member itself rather than on a call or a type alias.
template <typename BoundaryCondition, typename = std::void_t<>>
struct has_boundary_field_time_derivatives : std::false_type {};
template <typename BoundaryCondition>
struct has_boundary_field_time_derivatives<
    BoundaryCondition,
    std::void_t<decltype(&BoundaryCondition::boundary_field_time_derivatives)>>
    : std::true_type {};
template <typename BoundaryCondition>
constexpr bool has_boundary_field_time_derivatives_v =
    has_boundary_field_time_derivatives<BoundaryCondition>::value;
}  // namespace evolution::dg::Actions::detail
