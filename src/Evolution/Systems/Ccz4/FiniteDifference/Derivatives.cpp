// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.tpp"
#include "NumericalAlgorithms/FiniteDifference/SecondPartialDerivatives.hpp"
#include "NumericalAlgorithms/FiniteDifference/SecondPartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
namespace {
// The fields `System::second_derivative_tags` as runs of consecutive evolved
// tags, which the finite-difference kernels differentiate in place, one run at
// a time.
using second_derivative_runs = tmpl::list<
    tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
               ::Ccz4::Tags::ConformalFactor<DataVector>>,
    tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>>>;
static_assert(std::is_same_v<tmpl::join<second_derivative_runs>,
                             System::second_derivative_tags>,
              "The runs must hold the fields System::second_derivative_tags "
              "in order.");

// The data of the consecutive tags `SubsetTags` within `data`, which holds the
// tensors of `TagsList` in the layout of a `Variables`.
template <typename SubsetTags, typename TagsList>
gsl::span<const double> subset_data(const gsl::span<const double> data) {
  // Use a Variables view to get the offset of the subset without having to do
  // pointer math.
  const Variables<TagsList> view{
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<double*>(data.data()), data.size()};
  const auto subset = view.template reference_subset<SubsetTags>();
  return {subset.data(), subset.size()};
}

// The volume data and the neighbor ghost data of the consecutive evolved tags
// `SubsetTags`.
template <typename SubsetTags>
std::pair<gsl::span<const double>, DirectionMap<3, gsl::span<const double>>>
volume_and_ghost_data(
    const Variables<typename System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data) {
  using NeighborVariables = Variables<Tags::spacetime_reconstruction_tags>;
  DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};
  for (const auto& [directional_element_id, ghost_data] : all_ghost_data) {
    const DataVector& neighbor_data =
        ghost_data.neighbor_ghost_data_for_reconstruction();
    ASSERT(
        neighbor_data.size() %
                NeighborVariables::number_of_independent_components ==
            0,
        "Amount of reconstruction data sent ("
            << neighbor_data.size() << ") from " << directional_element_id
            << " is not a multiple of the number of reconstruction variables "
            << NeighborVariables::number_of_independent_components);
    ghost_cell_vars.insert(std::pair{
        directional_element_id.direction(),
        subset_data<SubsetTags, Tags::spacetime_reconstruction_tags>(
            gsl::make_span(neighbor_data.data(), neighbor_data.size()))});
  }
  const auto volume_subset =
      volume_evolved_variables.template reference_subset<SubsetTags>();
  return {gsl::make_span(volume_subset.data(), volume_subset.size()),
          std::move(ghost_cell_vars)};
}
}  // namespace

void spacetime_derivatives(
    const gsl::not_null<Variables<
        db::wrap_tags_in<::Tags::deriv, typename System::gradients_tags,
                         tmpl::size_t<3>, Frame::Inertial>>*>
        result,
    const Variables<typename System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const size_t& deriv_order, const Mesh<3>& volume_mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_jacobian) {
  using gradients_tags = typename System::gradients_tags;
  if (UNLIKELY(result->number_of_grid_points() !=
               volume_evolved_variables.number_of_grid_points())) {
    result->initialize(volume_evolved_variables.number_of_grid_points());
  }

  const auto [volume_Ccz4_vars, ghost_cell_vars] =
      volume_and_ghost_data<gradients_tags>(volume_evolved_variables,
                                            all_ghost_data);

  ::fd::partial_derivatives<gradients_tags>(
      result, volume_Ccz4_vars, ghost_cell_vars, volume_mesh,
      Variables<gradients_tags>::number_of_independent_components, deriv_order,
      cell_centered_logical_to_inertial_inv_jacobian);
}

void spacetime_derivatives(
    const gsl::not_null<Variables<
        db::wrap_tags_in<::Tags::deriv, System::second_derivative_tags,
                         tmpl::size_t<3>, Frame::Inertial>>*>
        result,
    const Variables<typename System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const size_t& deriv_order, const Mesh<3>& volume_mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_jacobian) {
  if (UNLIKELY(result->number_of_grid_points() !=
               volume_evolved_variables.number_of_grid_points())) {
    result->initialize(volume_evolved_variables.number_of_grid_points());
  }

  tmpl::for_each<second_derivative_runs>(
      [&]<typename RunTags>(tmpl::type_<RunTags> /*meta*/) {
        const auto [volume_run_vars, ghost_cell_run_vars] =
            volume_and_ghost_data<RunTags>(volume_evolved_variables,
                                           all_ghost_data);
        auto run_result = result->template reference_subset<db::wrap_tags_in<
            ::Tags::deriv, RunTags, tmpl::size_t<3>, Frame::Inertial>>();
        ::fd::partial_derivatives<RunTags>(
            make_not_null(&run_result), volume_run_vars, ghost_cell_run_vars,
            volume_mesh, Variables<RunTags>::number_of_independent_components,
            deriv_order, cell_centered_logical_to_inertial_inv_jacobian);
      });
}

void second_spacetime_derivatives(
    const gsl::not_null<
        Variables<db::wrap_tags_in<::Tags::second_deriv, System::gradients_tags,
                                   tmpl::size_t<3>, Frame::Inertial>>*>
        result,
    const Variables<typename System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const size_t& deriv_order, const Mesh<3>& volume_mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_jacobian,
    const InverseHessian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_hessian) {
  ASSERT(deriv_order == 4,
         "Only fourth order second partial derivatives have been implemented.");

  using gradients_tags = typename System::gradients_tags;
  if (UNLIKELY(result->number_of_grid_points() !=
               volume_evolved_variables.number_of_grid_points())) {
    result->initialize(volume_evolved_variables.number_of_grid_points());
  }

  const auto [volume_Ccz4_vars, ghost_cell_vars] =
      volume_and_ghost_data<gradients_tags>(volume_evolved_variables,
                                            all_ghost_data);

  ::fd::second_partial_derivatives<gradients_tags>(
      result, volume_Ccz4_vars, ghost_cell_vars, volume_mesh,
      Variables<gradients_tags>::number_of_independent_components, deriv_order,
      cell_centered_logical_to_inertial_inv_jacobian,
      cell_centered_logical_to_inertial_inv_hessian);
}

void first_and_second_spacetime_derivatives(
    const gsl::not_null<
        Variables<db::wrap_tags_in<::Tags::deriv, System::gradients_tags,
                                   tmpl::size_t<3>, Frame::Inertial>>*>
        first_derivatives,
    const gsl::not_null<Variables<
        db::wrap_tags_in<::Tags::second_deriv, System::second_derivative_tags,
                         tmpl::size_t<3>, Frame::Inertial>>*>
        second_derivatives,
    const Variables<typename System::variables_tag::tags_list>&
        volume_evolved_variables,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>&
        all_ghost_data,
    const size_t& deriv_order, const Mesh<3>& volume_mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_jacobian,
    const InverseHessian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>&
        cell_centered_logical_to_inertial_inv_hessian) {
  ASSERT(deriv_order == 4,
         "Only fourth order second partial derivatives have been implemented.");

  using gradients_tags = typename System::gradients_tags;
  const size_t number_of_points =
      volume_evolved_variables.number_of_grid_points();
  if (UNLIKELY(first_derivatives->number_of_grid_points() !=
               number_of_points)) {
    first_derivatives->initialize(number_of_points);
  }
  if (UNLIKELY(second_derivatives->number_of_grid_points() !=
               number_of_points)) {
    second_derivatives->initialize(number_of_points);
  }

  const auto [volume_Ccz4_vars, ghost_cell_vars] =
      volume_and_ghost_data<gradients_tags>(volume_evolved_variables,
                                            all_ghost_data);

  // The logical first derivatives of all the physical fields. Those of the
  // second-derivative fields are reused in the inverse-Hessian terms of their
  // second derivatives.
  DataVector logical_derivatives_buffer(3 * volume_Ccz4_vars.size());
  std::array<gsl::span<double>, 3> logical_derivatives{};
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(logical_derivatives, d) =
        gsl::make_span(&logical_derivatives_buffer[d * volume_Ccz4_vars.size()],
                       volume_Ccz4_vars.size());
  }
  ::fd::partial_derivatives<gradients_tags>(
      first_derivatives, make_not_null(&logical_derivatives), volume_Ccz4_vars,
      ghost_cell_vars, volume_mesh,
      Variables<gradients_tags>::number_of_independent_components, deriv_order,
      cell_centered_logical_to_inertial_inv_jacobian);

  tmpl::for_each<second_derivative_runs>([&]<typename RunTags>(
                                             tmpl::type_<RunTags> /*meta*/) {
    const auto [volume_run_vars, ghost_cell_run_vars] =
        volume_and_ghost_data<RunTags>(volume_evolved_variables,
                                       all_ghost_data);
    std::array<gsl::span<const double>, 3> run_logical_derivatives{};
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(run_logical_derivatives, d) =
          subset_data<RunTags, gradients_tags>(gsl::at(logical_derivatives, d));
    }
    auto run_second_derivatives =
        second_derivatives->template reference_subset<db::wrap_tags_in<
            ::Tags::second_deriv, RunTags, tmpl::size_t<3>, Frame::Inertial>>();
    ::fd::second_partial_derivatives<RunTags>(
        make_not_null(&run_second_derivatives), run_logical_derivatives,
        volume_run_vars, ghost_cell_run_vars, volume_mesh,
        Variables<RunTags>::number_of_independent_components, deriv_order,
        cell_centered_logical_to_inertial_inv_jacobian,
        cell_centered_logical_to_inertial_inv_hessian);
  });
}
}  // namespace Ccz4::fd
