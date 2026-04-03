// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Systems/SoScalarWave/BoundaryConditions/TimeDerivativeDirichlet.hpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace SoScalarWave::Actions {

/*!
 * \brief After internal boundary corrections have been applied, overwrite
 * \f$\partial_t \Psi\f$ and \f$\partial_t \Pi\f$ at external boundary face
 * points with analytic values for `TimeDerivativeDirichlet` boundary
 * conditions.
 *
 * \details This action is intended to run immediately after
 * `ApplyBoundaryCorrectionsToTimeDerivative`.  At corner points that lie on
 * both an external boundary face and an internal interface, the internal DG
 * boundary correction (added by `ApplyBoundaryCorrectionsToTimeDerivative`)
 * contaminates the time derivative that was set to analytic values by the
 * `dg_time_derivative` method of `TimeDerivativeDirichlet` during
 * `ComputeTimeDerivative`.  This action fixes that by overwriting the time
 * derivatives of Psi and Pi at all external face points where the boundary
 * condition is `TimeDerivativeDirichlet`, ensuring the final values are
 * exactly the analytic time derivatives.
 *
 * Currently only Gauss-Lobatto quadrature is supported.
 */
template <size_t Dim>
struct OverwriteExternalBoundaryDt {
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt,
                         typename SoScalarWave::System<Dim>::variables_tag>;
  using return_tags = tmpl::list<dt_variables_tag>;
  using argument_tags =
      tmpl::list<domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
                 domain::Tags::ExternalBoundaryConditions<Dim>,
                 domain::Tags::ElementMap<Dim, Frame::Grid>,
                 domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                             Frame::Inertial>,
                 ::Tags::Time, domain::Tags::FunctionsOfTime>;

  static void apply(
      const gsl::not_null<Variables<typename dt_variables_tag::tags_list>*>
          dt_vars,
      const Element<Dim>& element, const Mesh<Dim>& mesh,
      const std::vector<DirectionMap<
          Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>&
          all_boundary_conditions,
      const ElementMap<Dim, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
          moving_mesh_map,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) {
    if (element.external_boundaries().empty()) {
      return;
    }

    ASSERT(mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto,
           "OverwriteExternalBoundaryDt currently only supports "
           "Gauss-Lobatto quadrature, but got "
               << mesh.quadrature(0));

    using dt_psi_tag = ::Tags::dt<SoScalarWave::Tags::Psi>;
    using dt_pi_tag = ::Tags::dt<SoScalarWave::Tags::Pi>;

    const auto& boundary_conditions =
        all_boundary_conditions.at(element.id().block_id());

    for (const auto& direction : element.external_boundaries()) {
      const auto& bc_base = *boundary_conditions.at(direction);

      // Only act on TimeDerivativeDirichlet boundary conditions.
      const auto* const td_bc =
          dynamic_cast<const SoScalarWave::BoundaryConditions::
                           TimeDerivativeDirichlet<Dim>*>(&bc_base);
      if (td_bc == nullptr) {
        continue;
      }

      // Compute face coordinates (same as BoundaryConditionsImpl.hpp)
      const auto face_mesh = mesh.slice_away(direction.dimension());
      const size_t num_face_pts = face_mesh.number_of_grid_points();
      const auto face_coords =
          moving_mesh_map(logical_to_grid_map(interface_logical_coordinates(
                              face_mesh, direction)),
                          time, functions_of_time);

      // Project current volume dt values to the face using
      // project_tensors_to_boundary, the same function used by
      // BoundaryConditionsImpl.hpp to project interior data to faces.
      using dt_tags_to_read = tmpl::list<dt_psi_tag, dt_pi_tag>;
      Variables<dt_tags_to_read> current_dt_on_face{num_face_pts};
      ::dg::project_tensors_to_boundary<dt_tags_to_read>(
          make_not_null(&current_dt_on_face), *dt_vars, mesh, direction);

      // Call dg_time_derivative to compute the correction.  It returns
      //   correction = -current_dt + analytic_dt
      // so that after add_slice_to_data the result is exactly analytic_dt.
      // The normal_covector and face_mesh_velocity are unused by
      // TimeDerivativeDirichlet, so we pass a dummy normal and std::nullopt.
      Scalar<DataVector> dt_psi_correction{num_face_pts};
      Scalar<DataVector> dt_pi_correction{num_face_pts};
      tnsr::i<DataVector, Dim, Frame::Inertial> dt_phi_correction{num_face_pts};
      const tnsr::i<DataVector, Dim, Frame::Inertial> unused_normal{
          num_face_pts, 0.0};
      td_bc->dg_time_derivative(
          make_not_null(&dt_psi_correction), make_not_null(&dt_pi_correction),
          make_not_null(&dt_phi_correction), std::nullopt, unused_normal,
          face_coords, get<dt_psi_tag>(current_dt_on_face),
          get<dt_pi_tag>(current_dt_on_face), time);

      // Build a full correction Variables (same tags as dt_variables_tag)
      // so we can use add_slice_to_data which requires matching tag lists.
      using dt_variables_tags = typename dt_variables_tag::tags_list;
      Variables<dt_variables_tags> correction{num_face_pts, 0.0};
      get<dt_psi_tag>(correction) = std::move(dt_psi_correction);
      get<dt_pi_tag>(correction) = std::move(dt_pi_correction);
      // dt_phi_correction is 0 from dg_time_derivative, matching the
      // 0.0-initialized correction Variables.

      // Add the correction to the volume data at the boundary slice.
      // For GL, add_slice_to_data adds the face data to the volume at
      // the boundary slice, matching the pattern in BoundaryConditionsImpl.hpp.
      add_slice_to_data(dt_vars, correction, mesh.extents(),
                        direction.dimension(),
                        index_to_slice_at(mesh.extents(), direction));
    }
  }
};

}  // namespace SoScalarWave::Actions
