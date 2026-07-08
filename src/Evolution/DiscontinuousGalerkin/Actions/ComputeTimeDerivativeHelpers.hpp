// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"
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

// The LDG auxiliary pass uses additional `dg_auxiliary_*` aliases on the
// boundary correction. Non-LDG corrections do not define them, so these
// "detect-or-default" metafunctions yield the alias when present and an empty
// `tmpl::list<>` otherwise. This keeps both arms of `tmpl::conditional_t`
// well-formed even when `ComputeAuxiliary` is `false`.
CREATE_HAS_TYPE_ALIAS(dg_auxiliary_package_data_temporary_tags)
CREATE_HAS_TYPE_ALIAS_V(dg_auxiliary_package_data_temporary_tags)

template <typename T, bool = has_dg_auxiliary_package_data_temporary_tags_v<T>>
struct get_dg_auxiliary_package_data_temporary_tags_or_empty {
  using type = tmpl::list<>;
};

template <typename T>
struct get_dg_auxiliary_package_data_temporary_tags_or_empty<T, true> {
  using type = typename T::dg_auxiliary_package_data_temporary_tags;
};

template <typename T>
using get_dg_auxiliary_package_data_temporary_tags_or_empty_t =
    typename get_dg_auxiliary_package_data_temporary_tags_or_empty<T>::type;

CREATE_HAS_TYPE_ALIAS(dg_auxiliary_package_field_tags)
CREATE_HAS_TYPE_ALIAS_V(dg_auxiliary_package_field_tags)

template <typename T, bool = has_dg_auxiliary_package_field_tags_v<T>>
struct get_dg_auxiliary_package_field_tags_or_empty {
  using type = tmpl::list<>;
};

template <typename T>
struct get_dg_auxiliary_package_field_tags_or_empty<T, true> {
  using type = typename T::dg_auxiliary_package_field_tags;
};

template <typename T>
using get_dg_auxiliary_package_field_tags_or_empty_t =
    typename get_dg_auxiliary_package_field_tags_or_empty<T>::type;

CREATE_HAS_TYPE_ALIAS(dg_auxiliary_package_data_volume_tags)
CREATE_HAS_TYPE_ALIAS_V(dg_auxiliary_package_data_volume_tags)

template <typename T, bool = has_dg_auxiliary_package_data_volume_tags_v<T>>
struct get_dg_auxiliary_package_data_volume_tags_or_empty {
  using type = tmpl::list<>;
};

template <typename T>
struct get_dg_auxiliary_package_data_volume_tags_or_empty<T, true> {
  using type = typename T::dg_auxiliary_package_data_volume_tags;
};

template <typename T>
using get_dg_auxiliary_package_data_volume_tags_or_empty_t =
    typename get_dg_auxiliary_package_data_volume_tags_or_empty<T>::type;

CREATE_HAS_TYPE_ALIAS(dg_auxiliary_boundary_terms_volume_tags)
CREATE_HAS_TYPE_ALIAS_V(dg_auxiliary_boundary_terms_volume_tags)

template <typename T, bool = has_dg_auxiliary_boundary_terms_volume_tags_v<T>>
struct get_dg_auxiliary_boundary_terms_volume_tags_or_empty {
  using type = tmpl::list<>;
};

template <typename T>
struct get_dg_auxiliary_boundary_terms_volume_tags_or_empty<T, true> {
  using type = typename T::dg_auxiliary_boundary_terms_volume_tags;
};

template <typename T>
using get_dg_auxiliary_boundary_terms_volume_tags_or_empty_t =
    typename get_dg_auxiliary_boundary_terms_volume_tags_or_empty<T>::type;

// An evolution `System` may declare an `auxiliary_variables` type alias whose
// first derivatives are computed in the physical DG pass and supplied to the
// volume time-derivative terms (sourced from the auxiliary-variable storage).
// Systems without that alias do not define it, so this "detect-or-default"
// metafunction yields the alias when present and an empty `tmpl::list<>`
// otherwise.
CREATE_HAS_TYPE_ALIAS(auxiliary_variables)
CREATE_HAS_TYPE_ALIAS_V(auxiliary_variables)

template <typename T, bool = has_auxiliary_variables_v<T>>
struct get_auxiliary_variables_or_empty {
  using type = tmpl::list<>;
};

template <typename T>
struct get_auxiliary_variables_or_empty<T, true> {
  using type = typename T::auxiliary_variables;
};

template <typename T>
using get_auxiliary_variables_or_empty_t =
    typename get_auxiliary_variables_or_empty<T>::type;

// The `Variables` that the volume time-derivative terms differentiate: the
// evolved variables (`variables_tag`) together with the auxiliary variables
// (`auxiliary_variables`).
//
// `gradient_variables` is prepended even though it is a subset of
// `variables_tag union auxiliary_variables`, which makes the `append` look
// redundant. The reason is ordering, not content: `partial_derivatives`
// requires the differentiated tags to be the *leading* tags of the source
// `Variables` (it takes the first `size(gradient_variables)` tags as the
// differentiated block), so `gradient_variables` must physically lead. The
// trailing `variables_tag` / `auxiliary_variables` then supply any remaining
// fields - those read by the moving-mesh terms but not themselves
// differentiated - and `remove_duplicates` drops the overlap with the head.
template <typename System>
using vars_to_differentiate_tags = tmpl::remove_duplicates<
    tmpl::append<typename System::gradient_variables,
                 typename System::variables_tag::tags_list,
                 get_auxiliary_variables_or_empty_t<System>>>;

// The evolved-variable-like fields a DG boundary correction reads on a face,
// listed in the order the framework projects them to the face: the evolved
// variables (`System::variables_tag`), followed - for the physical pass only -
// by the auxiliary variables (`System::auxiliary_variables`). The physical
// boundary correction may read auxiliary variables (e.g. an LDG numerical flux
// reads the auxiliary gradient), so they are projected after the evolved
// variables; the boundary correction's `dg_package_data` must take its inputs
// in this order. The auxiliary pass computes the auxiliary variables and does
// not read them, so it projects only the evolved variables.
template <typename System, bool ComputeAuxiliary>
using dg_boundary_correction_projected_evolved_tags = tmpl::conditional_t<
    ComputeAuxiliary, typename System::variables_tag::tags_list,
    tmpl::remove_duplicates<
        tmpl::append<typename System::variables_tag::tags_list,
                     get_auxiliary_variables_or_empty_t<System>>>>;

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
}  // namespace evolution::dg::Actions::detail
