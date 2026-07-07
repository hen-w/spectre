// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <typeinfo>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BoundaryConditions/Cartoon.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/BoundaryConditionsImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/InternalMortarDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace evolution::dg::Actions {
namespace detail {
// Single-argument wrappers around the detect-or-default auxiliary-tag
// metafunctions so they can be used as brigand lazy metafunctions in
// `tmpl::transform`.
template <typename T>
struct get_dg_auxiliary_package_temporary_tags {
  using type = get_dg_auxiliary_package_data_temporary_tags_or_empty_t<T>;
};
template <typename T>
struct get_dg_auxiliary_package_field_tags {
  using type = get_dg_auxiliary_package_field_tags_or_empty_t<T>;
};

// The combined tags that the LDG auxiliary pass would have to project from the
// (uninitialized) volume time-derivative buffers if a registered boundary
// correction or condition declared them. The pass deliberately skips the
// volume time-derivative computation, so these buffers are left uninitialized;
// the guard `auxiliary_send_reads_only_initialized_face_data` below asserts
// these combined lists are all empty.
//
// We build the lists with `tmpl::transform`/`tmpl::flatten` over the factory
// lists (mirroring how the action itself gathers tags across all possible
// boundary corrections/conditions), rather than per-class predicates, so we
// avoid binding the non-type `Dim` parameter into a `tmpl::all` lambda.

// Per-boundary-condition tags that `apply_boundary_condition_on_face` would
// project from the volume temporaries: `dg_interior_temporary_tags` minus the
// `Coordinates` tag (which is always available). `dg_interior_temporary_tags`
// is a required alias on every boundary condition. `DimType` carries `Dim` as a
// `tmpl::size_t` so this can be used as a lazy metafunction (with `tmpl::pin`)
// in `tmpl::transform`.
template <typename DimType, typename BoundaryCondition>
struct boundary_condition_volume_temporary_tags {
  using type =
      tmpl::remove<typename BoundaryCondition::dg_interior_temporary_tags,
                   domain::Tags::Coordinates<DimType::value, Frame::Inertial>>;
};

// Per-boundary-condition tags projected from the volume partial derivatives
// (`dg_interior_deriv_vars_tags`, optional) and from the volume time
// derivative (`dg_interior_dt_vars_tags`, optional). We reuse the existing
// detect-or-default metafunctions, which default to an empty list.
template <typename BoundaryCondition>
struct boundary_condition_deriv_tags {
  using type = get_deriv_vars_from_boundary_condition<BoundaryCondition>;
};
template <typename BoundaryCondition>
struct boundary_condition_dt_tags {
  using type = get_dt_vars_from_boundary_condition<BoundaryCondition>;
};

// `true` iff the LDG auxiliary send reads only initialized face data, i.e. it
// never projects any of the volume time-derivative buffers (volume
// temporaries, fluxes, partial derivatives), the inverse spatial metric, or
// the volume time derivative to the boundary. The auxiliary pass deliberately
// skips the volume time-derivative computation, so all of those volume buffers
// are left uninitialized; this guard confirms nothing reads them.
template <typename EvolutionSystem, typename Metavariables>
struct auxiliary_send_reads_only_initialized_face_data {
 private:
  using derived_boundary_corrections =
      tmpl::at<typename Metavariables::factory_creation::factory_classes,
               evolution::BoundaryCorrection>;
  // Mirror the physical boundary-condition path (BoundaryConditionsImpl.hpp):
  // the Cartoon/None/Periodic marker conditions are handled specially and do
  // not declare the interior-data interface (e.g. dg_interior_temporary_tags),
  // so they are excluded before inspecting that interface.
  using derived_boundary_conditions = tmpl::remove_if<
      tmpl::at<typename Metavariables::factory_creation::factory_classes,
               typename EvolutionSystem::boundary_conditions_base>,
      tmpl::or_<
          std::is_base_of<domain::BoundaryConditions::MarkAsCartoon, tmpl::_1>,
          std::is_base_of<domain::BoundaryConditions::MarkAsNone, tmpl::_1>,
          std::is_base_of<domain::BoundaryConditions::MarkAsPeriodic,
                          tmpl::_1>>>;

  // (3) Volume temporaries the boundary corrections would project.
  using all_correction_volume_temporary_tags = tmpl::flatten<
      tmpl::transform<derived_boundary_corrections,
                      get_dg_auxiliary_package_temporary_tags<tmpl::_1>>>;
  // (6) Volume temporaries the boundary conditions would project.
  using all_condition_volume_temporary_tags = tmpl::flatten<tmpl::transform<
      derived_boundary_conditions,
      boundary_condition_volume_temporary_tags<
          tmpl::pin<tmpl::size_t<EvolutionSystem::volume_dim>>, tmpl::_1>>>;
  // (4) Partial derivatives the boundary conditions would project.
  using all_condition_deriv_tags =
      tmpl::flatten<tmpl::transform<derived_boundary_conditions,
                                    boundary_condition_deriv_tags<tmpl::_1>>>;
  // (5) Volume time derivatives the boundary conditions would project.
  using all_condition_dt_tags =
      tmpl::flatten<tmpl::transform<derived_boundary_conditions,
                                    boundary_condition_dt_tags<tmpl::_1>>>;

 public:
  static constexpr bool value =
      // (1) No fluxes are projected (system is flux-free).
      tmpl::size<typename EvolutionSystem::flux_variables>::value == 0 and
      // (2) The system has no inverse spatial metric to project.
      not has_inverse_spatial_metric_tag_v<EvolutionSystem> and
      // (3) No boundary correction projects volume temporaries.
      tmpl::size<all_correction_volume_temporary_tags>::value == 0 and
      // (6) No boundary condition projects volume temporaries.
      tmpl::size<all_condition_volume_temporary_tags>::value == 0 and
      // (4) No boundary condition projects partial derivatives.
      tmpl::size<all_condition_deriv_tags>::value == 0 and
      // (5) No boundary condition projects the volume time derivative.
      tmpl::size<all_condition_dt_tags>::value == 0;
};

template <typename EvolutionSystem, typename Metavariables>
constexpr bool auxiliary_send_reads_only_initialized_face_data_v =
    auxiliary_send_reads_only_initialized_face_data<EvolutionSystem,
                                                    Metavariables>::value;
}  // namespace detail

/*!
 * \brief Packages and sends the auxiliary-pass boundary data for the local
 * discontinuous Galerkin (LDG) method.
 *
 * This action performs the *send* half of the LDG auxiliary communication. It
 * mirrors the boundary-data portion of `ComputeTimeDerivative::apply` (the part
 * before the time-derivative computation) but uses the auxiliary package-data
 * interface (`dg_auxiliary_package_field_tags`, `dg_auxiliary_package_data`,
 * `dg_auxiliary_package_data_volume_tags`) and sends on the auxiliary inbox
 * channel `evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox` with
 * `IsAuxiliary` set to `true`.
 *
 * The action does NOT compute the volume time derivative, fluxes, flux
 * divergence, or partial derivatives. It only:
 *
 * 1. Computes the internal mortar data in auxiliary mode
 *    (`detail::internal_mortar_data` with `ComputeAuxiliary` set to `true`).
 * 2. Applies the auxiliary external boundary conditions
 *    (`detail::apply_boundary_conditions_on_all_external_faces` with
 *    `ComputeAuxiliary` set to `true`).
 * 3. Sends the packaged auxiliary mortar data to the neighbors via
 *    `detail::send_boundary_data` with `IsAuxiliary` set to `true`.
 *
 * Only conforming, DG (no subcell), global time stepping is supported for now;
 * this is enforced by `static_assert`s.
 */
template <size_t Dim, typename EvolutionSystem, bool LocalTimeStepping,
          bool UseNodegroupDgElements>
struct SendAuxiliaryBoundaryData {
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          Dim, UseNodegroupDgElements, /*IsAuxiliary=*/true>>;
  using const_global_cache_tags =
      tmpl::list<::dg::Tags::Formulation, evolution::Tags::BoundaryCorrection,
                 domain::Tags::ExternalBoundaryConditions<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* /*meta*/);  // NOLINT const
};

template <size_t Dim, typename EvolutionSystem, bool LocalTimeStepping,
          bool UseNodegroupDgElements>
template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
          typename ActionList, typename ParallelComponent,
          typename Metavariables>
Parallel::iterable_action_return_t SendAuxiliaryBoundaryData<
    Dim, EvolutionSystem, LocalTimeStepping, UseNodegroupDgElements>::
    apply(db::DataBox<DbTagsList>& box,
          tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
          Parallel::GlobalCache<Metavariables>& cache,
          const ArrayIndex& /*array_index*/, ActionList /*meta*/,
          const ParallelComponent* const /*meta*/) {  // NOLINT const
  static_assert(not LocalTimeStepping,
                "Local time stepping is not supported for the LDG auxiliary "
                "pass.");
  static_assert(not evolution::dg::using_subcell_v<Metavariables>,
                "Subcell is not supported for the LDG auxiliary pass.");
  static_assert(UseNodegroupDgElements ==
                    Parallel::is_dg_element_collection_v<ParallelComponent>,
                "The action SendAuxiliaryBoundaryData is told by the "
                "template parameter UseNodegroupDgElements that it is being "
                "used with a DgElementCollection, but the ParallelComponent "
                "is not a DgElementCollection. You need to change the "
                "template parameter on the SendAuxiliaryBoundaryData action "
                "in your action list.");
  static_assert(
      detail::auxiliary_send_reads_only_initialized_face_data_v<EvolutionSystem,
                                                                Metavariables>,
      "The LDG auxiliary pass does not compute the volume time derivative, so "
      "it cannot supply volume temporaries, fluxes, partial derivatives, an "
      "inverse spatial metric, or interior time derivatives to the boundary "
      "corrections/conditions. A boundary correction or condition registered "
      "for this system requires one of these on the boundary, which the "
      "auxiliary send does not support.");

  using variables_tag = typename EvolutionSystem::variables_tag;
  using partial_derivative_tags = typename EvolutionSystem::gradient_variables;
  using flux_variables = typename EvolutionSystem::flux_variables;
  using compute_volume_time_derivative_terms =
      typename EvolutionSystem::compute_volume_time_derivative_terms;

  const Mesh<Dim>& mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
  const Element<Dim>& element = db::get<domain::Tags::Element<Dim>>(box);
  ASSERT(alg::all_of(mesh.basis(),
                     [&mesh](const Spectral::Basis current_basis) {
                       return current_basis == mesh.basis(0);
                     }) or
             element.topologies() != domain::topologies::hypercube<Dim>,
         "An isotropic basis must be used in the evolution code. While "
         "theoretically this restriction could be lifted, the simplification "
         "it offers are quite substantial. Relaxing this assumption is likely "
         "to require quite a bit of careful code refactoring and debugging.");
  ASSERT(alg::all_of(mesh.quadrature(),
                     [&mesh](const Spectral::Quadrature current_quadrature) {
                       return current_quadrature == mesh.quadrature(0);
                     }) or
             element.topologies() != domain::topologies::hypercube<Dim>,
         "An isotropic quadrature must be used in the evolution code. While "
         "theoretically this restriction could be lifted, the simplification "
         "it offers are quite substantial. Relaxing this assumption is likely "
         "to require quite a bit of careful code refactoring and debugging.");

  const auto& boundary_correction =
      db::get<evolution::Tags::BoundaryCorrection>(box);
  using derived_boundary_corrections =
      tmpl::at<typename Metavariables::factory_creation::factory_classes,
               evolution::BoundaryCorrection>;

  // To avoid a second allocation in internal_mortar_data, we allocate the
  // variables needed to construct the fields on the faces here along with
  // everything else. This requires us to know all the tags necessary to apply
  // boundary corrections. However, since we pick boundary corrections at
  // runtime, we just gather all possible tags from all possible boundary
  // corrections and lump them into the allocation. This may result in a
  // larger-than-necessary allocation, but it won't be that much larger.
  //
  // We collect the auxiliary package tags (via the detect-or-default
  // metafunctions) since this action packages using dg_auxiliary_package_data.
  using all_dg_auxiliary_package_temporary_tags = tmpl::transform<
      derived_boundary_corrections,
      detail::get_dg_auxiliary_package_temporary_tags<tmpl::_1>>;
  using all_primitive_tags_for_face =
      tmpl::transform<derived_boundary_corrections,
                      detail::get_primitive_tags_for_face<
                          tmpl::pin<EvolutionSystem>, tmpl::_1>>;
  using fluxes_tags = db::wrap_tags_in<::Tags::Flux, flux_variables,
                                       tmpl::size_t<Dim>, Frame::Inertial>;
  using dg_package_data_projected_tags =
      tmpl::list<typename variables_tag::tags_list, fluxes_tags,
                 all_dg_auxiliary_package_temporary_tags,
                 all_primitive_tags_for_face>;
  using all_face_temporary_tags =
      tmpl::remove_duplicates<tmpl::flatten<tmpl::push_back<
          tmpl::list<dg_package_data_projected_tags,
                     detail::inverse_spatial_metric_tag<EvolutionSystem>>,
          detail::OneOverNormalVectorMagnitude, detail::NormalVector<Dim>>>>;
  // To avoid additional allocations in internal_mortar_data, we provide a
  // buffer used to compute the packaged data before it has to be projected to
  // the mortar. We get all auxiliary mortar tags for similar reasons as
  // described above.
  using all_mortar_tags = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::transform<derived_boundary_corrections,
                      detail::get_dg_auxiliary_package_field_tags<tmpl::_1>>>>;

  // We also don't use the number of volume mesh grid points. We instead use the
  // max number of grid points from each face. That way, our allocation will be
  // large enough to hold any face and we can reuse the allocation for each face
  // without having to resize it.
  size_t num_face_temporary_grid_points = 0;
  {
    for (const auto& [direction, neighbors_in_direction] :
         element.neighbors()) {
      (void)neighbors_in_direction;
      const auto face_mesh = mesh.slice_away(direction.dimension());
      num_face_temporary_grid_points = std::max(
          num_face_temporary_grid_points, face_mesh.number_of_grid_points());
    }
  }

  // Allocate the Variables classes needed by the auxiliary boundary-data
  // computation.
  //
  // The volume time-derivative buffers (`temporaries`, `volume_fluxes`,
  // `partial_derivs`) are still allocated, correctly sized, here even though
  // the auxiliary pass does NOT compute the volume time derivative. They are
  // left uninitialized: the auxiliary-pass external-boundary-condition helper
  // (`apply_boundary_conditions_on_all_external_faces`) projects `temporaries`
  // and `partial_derivs` to the face whenever the boundary condition declares
  // interior temporary/derivative tags, so these buffers must have the correct
  // size. We allocate them rather than running `volume_terms`, which is the
  // expensive computation we are skipping. (Allocating only the face buffers
  // is not safe in general because that external-boundary helper reads the
  // volume buffers regardless of the auxiliary/physical mode.)
  using VarsTemporaries =
      Variables<typename compute_volume_time_derivative_terms::temporary_tags>;
  using VarsFluxes =
      Variables<db::wrap_tags_in<::Tags::Flux, flux_variables,
                                 tmpl::size_t<Dim>, Frame::Inertial>>;
  using VarsPartialDerivatives =
      Variables<db::wrap_tags_in<::Tags::deriv, partial_derivative_tags,
                                 tmpl::size_t<Dim>, Frame::Inertial>>;
  using VarsFaceTemporaries = Variables<all_face_temporary_tags>;
  using DgPackagedDataVarsOnFace = Variables<all_mortar_tags>;
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const size_t buffer_size =
      (VarsTemporaries::number_of_independent_components +
       VarsFluxes::number_of_independent_components +
       VarsPartialDerivatives::number_of_independent_components) *
          number_of_grid_points +
      // Different number of grid points. See explanation above where
      // num_face_temporary_grid_points is defined
      (VarsFaceTemporaries::number_of_independent_components +
       DgPackagedDataVarsOnFace::number_of_independent_components) *
          num_face_temporary_grid_points;
  auto buffer = cpp20::make_unique_for_overwrite<double[]>(buffer_size);
#ifdef SPECTRE_NAN_INIT
  std::fill(&buffer[0], &buffer[buffer_size],
            std::numeric_limits<double>::signaling_NaN());
#endif
  VarsTemporaries temporaries{
      &buffer[0], VarsTemporaries::number_of_independent_components *
                      number_of_grid_points};
  VarsFluxes volume_fluxes{
      &buffer[VarsTemporaries::number_of_independent_components *
              number_of_grid_points],
      VarsFluxes::number_of_independent_components * number_of_grid_points};
  VarsPartialDerivatives partial_derivs{
      &buffer[(VarsTemporaries::number_of_independent_components +
               VarsFluxes::number_of_independent_components) *
              number_of_grid_points],
      VarsPartialDerivatives::number_of_independent_components *
          number_of_grid_points};
  // Lighter weight data structure than a Variables to avoid passing even more
  // templates to internal_mortar_data.
  gsl::span<double> face_temporaries = gsl::make_span<double>(
      &buffer[(VarsTemporaries::number_of_independent_components +
               VarsFluxes::number_of_independent_components +
               VarsPartialDerivatives::number_of_independent_components) *
              number_of_grid_points],
      // Different number of grid points. See explanation above where
      // num_face_temporary_grid_points is defined
      VarsFaceTemporaries::number_of_independent_components *
          num_face_temporary_grid_points);
  gsl::span<double> packaged_data_buffer = gsl::make_span<double>(
      &buffer[(VarsTemporaries::number_of_independent_components +
               VarsFluxes::number_of_independent_components +
               VarsPartialDerivatives::number_of_independent_components) *
                  number_of_grid_points +
              VarsFaceTemporaries::number_of_independent_components *
                  num_face_temporary_grid_points],
      // Different number of grid points. See explanation above where
      // num_face_temporary_grid_points is defined
      DgPackagedDataVarsOnFace::number_of_independent_components *
          num_face_temporary_grid_points);

  const Variables<detail::get_primitive_vars_tags_from_system<EvolutionSystem>>*
      primitive_vars{nullptr};
  if constexpr (EvolutionSystem::has_primitive_and_conservative_vars) {
    primitive_vars =
        &db::get<typename EvolutionSystem::primitive_variables_tag>(box);
  }

  static_assert(
      tmpl::all<derived_boundary_corrections, std::is_final<tmpl::_1>>::value,
      "All createable classes for boundary corrections must be marked "
      "final.");
  tmpl::for_each<derived_boundary_corrections>([&boundary_correction, &box,
                                                &partial_derivs,
                                                &primitive_vars, &temporaries,
                                                &volume_fluxes,
                                                &packaged_data_buffer,
                                                &face_temporaries](
                                                   auto derived_correction_v) {
    using DerivedCorrection = tmpl::type_from<decltype(derived_correction_v)>;
    if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
      // Compute internal boundary quantities on the mortar in auxiliary
      // mode for sides of the element that have neighbors, i.e. they are
      // not an external side.
      // Note: this call mutates:
      //  - evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>,
      //  - evolution::dg::Tags::MortarData<Dim>
      detail::internal_mortar_data<EvolutionSystem, Dim,
                                   /*ComputeAuxiliary=*/true>(
          make_not_null(&box), make_not_null(&face_temporaries),
          make_not_null(&packaged_data_buffer),
          dynamic_cast<const DerivedCorrection&>(boundary_correction),
          db::get<variables_tag>(box), volume_fluxes, temporaries,
          primitive_vars,
          typename DerivedCorrection::dg_auxiliary_package_data_volume_tags{});

      detail::apply_boundary_conditions_on_all_external_faces<
          EvolutionSystem, Dim, /*ComputeAuxiliary=*/true>(
          make_not_null(&box),
          dynamic_cast<const DerivedCorrection&>(boundary_correction),
          temporaries, volume_fluxes, partial_derivs, primitive_vars);
    }
  });

  detail::send_boundary_data<Dim, EvolutionSystem, LocalTimeStepping,
                             UseNodegroupDgElements, /*IsAuxiliary=*/true,
                             ParallelComponent>(
      make_not_null(&cache), make_not_null(&box), volume_fluxes);
  return {Parallel::AlgorithmExecution::Continue, std::nullopt};
}
}  // namespace evolution::dg::Actions
