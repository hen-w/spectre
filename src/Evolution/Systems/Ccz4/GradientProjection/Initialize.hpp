// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection {
/*!
 * \brief Initialization mutator that builds the static per-element
 * `LocalOperators` and adds the (empty) buffers that every right-hand side
 * evaluation fills: the right-hand-side targets `Tags::RhsTargets` and the
 * chart data `Tags::StepChartFields` and `Tags::StepCharts`.
 *
 * It is an `ERROR` if the grid-to-inertial map is time dependent, since the
 * cached operators are valid only for a static mesh.
 */
struct InitializeLocalOperators {
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using simple_tags = tmpl::list<Tags::LocalOperators, Tags::RhsTargets,
                                 Tags::StepChartFields, Tags::StepCharts>;
  using compute_tags = tmpl::list<>;
  using return_tags = tmpl::list<Tags::LocalOperators>;
  using argument_tags = tmpl::list<
      domain::Tags::Mesh<3>,
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
      domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                  Frame::Inertial>>;

  static void apply(
      gsl::not_null<LocalOperators*> local_operators, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian,
      const Scalar<DataVector>& det_inverse_jacobian,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
          grid_to_inertial_map);
};

namespace Actions {
/*!
 * \brief Set the physical fields from the analytic initial data and zero the
 * stored gradients, on the active grid (the DG grid, or with DG-subcell the
 * subcell grid of elements using finite differences).
 *
 * \details This replaces `evolution::Initialization::Actions::SetVariables`,
 * which would request the gradients from the initial data (with DG-subcell,
 * `SetSubcellGrid` must therefore not set the variables). The gradients are
 * zeroed (rather than left uninitialized) because generic actions such as
 * `RandomizeVariables` operate on the whole evolved `Variables`; they are
 * overwritten before every use.
 */
struct SetInitialData {
  using simple_tags_from_options = tmpl::list<::Tags::Time>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    using derived_classes =
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 evolution::initial_data::InitialData>;
    const double initial_time = db::get<::Tags::Time>(box);
    const auto& logical_coords =
        [&box]() -> const tnsr::I<DataVector, 3, Frame::ElementLogical>& {
      if constexpr (evolution::dg::using_subcell_v<Metavariables>) {
        if (db::get<evolution::dg::subcell::Tags::ActiveGrid>(box) ==
            evolution::dg::subcell::ActiveGrid::Subcell) {
          return db::get<evolution::dg::subcell::Tags::Coordinates<
              3, Frame::ElementLogical>>(box);
        }
      }
      return db::get<domain::Tags::Coordinates<3, Frame::ElementLogical>>(box);
    }();
    const auto inertial_coords =
        db::get<::domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                              Frame::Inertial>>(
            box)(db::get<::domain::Tags::ElementMap<3, Frame::Grid>>(box)(
                     logical_coords),
                 initial_time, db::get<::domain::Tags::FunctionsOfTime>(box));
    using variables_tag = typename System::variables_tag;
    call_with_dynamic_type<void, derived_classes>(
        &db::get<evolution::initial_data::Tags::InitialData>(box),
        [&box, &initial_time, &inertial_coords](const auto* const data) {
          using initial_data_subclass = std::decay_t<decltype(*data)>;
          if constexpr (is_analytic_data_v<initial_data_subclass> or
                        is_analytic_solution_v<initial_data_subclass>) {
            db::mutate<variables_tag>(
                [&data, &initial_time, &inertial_coords](
                    const gsl::not_null<typename variables_tag::type*> vars) {
                  vars->initialize(get<0>(inertial_coords).size(), 0.0);
                  vars->assign_subset(evolution::Initialization::initial_data(
                      *data, inertial_coords, initial_time,
                      typename System::physical_tags{}));
                },
                make_not_null(&box));
          } else {
            ERROR(
                "The CCZ4 gradient-projection scheme requires analytic "
                "initial data.");
          }
        });
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief On elements using finite differences, check that the options the
 * DG-FD coupling requires are set.
 *
 * \details The gradients that finite-difference elements send to their DG
 * neighbors (`ComputeFiniteDifferenceGradients`) require the volume to be on
 * the algebraic-constraint surface, which
 * `Ccz4::fd::Tags::ConstrainedEvolution` enforces at the start of each
 * finite-difference right-hand side, and the DG elements always evolve the
 * lapse and shift
 * (`Ccz4::fd::Tags::EvolveLapseAndShift`). It is an `ERROR` if either option
 * is `false` on an element whose active grid is the subcell grid. Elements
 * using DG do not depend on these options. The active grid is fixed at
 * initialization (there is no troubled-cell indicator), so the check is done
 * once, after `evolution::dg::subcell::Actions::SetSubcellGrid`.
 */
struct CheckFiniteDifferenceOptions {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(evolution::dg::using_subcell_v<Metavariables>,
                  "CheckFiniteDifferenceOptions is only used with DG-subcell.");
    if (db::get<evolution::dg::subcell::Tags::ActiveGrid>(box) ==
        evolution::dg::subcell::ActiveGrid::Subcell) {
      const auto& element_id = db::get<domain::Tags::Element<3>>(box).id();
      if (not db::get<::Ccz4::fd::Tags::ConstrainedEvolution>(box)) {
        ERROR("Element " << element_id
                         << " uses finite differences, which in the CCZ4 "
                            "gradient-projection DG-FD scheme require "
                            "ConstrainedEvolution: true.");
      }
      if (not db::get<::Ccz4::fd::Tags::EvolveLapseAndShift>(box)) {
        ERROR("Element " << element_id
                         << " uses finite differences, which in the CCZ4 "
                            "gradient-projection DG-FD scheme require "
                            "EvolveLapseAndShift: true (the DG elements always "
                            "evolve the lapse and shift).");
      }
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Ccz4::GradientProjection
