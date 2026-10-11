// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/NeighborPackagedData.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <typeinfo>
#include <vector>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.tpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/MeshForGhostData.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceConstrainedEvolution.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UnlimitedDeg4Prim.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection {
PackagedFaceData package_finite_difference_face(
    Variables<System::variables_tag_list> reconstructed_vars,
    const tnsr::i<DataVector, 3>& normal_covector,
    const BoundaryCorrections::UpwindGradientProjection& boundary_correction,
    const Direction<3>& direction) {
  ::Ccz4::fd::enforce_algebraic_constraints(
      make_not_null(&get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(
          reconstructed_vars)),
      make_not_null(
          &get<::Ccz4::Tags::ATilde<DataVector, 3>>(reconstructed_vars)));
  PackagedFaceData packaged{reconstructed_vars.number_of_grid_points()};
  evolution::dg::Actions::detail::dg_package_data<System>(
      make_not_null(&packaged), boundary_correction, reconstructed_vars,
      normal_covector, {std::nullopt}, direction, System::variables_tag_list{});
  return packaged;
}

DirectionalIdMap<3, DataVector> NeighborPackagedData::apply(
    const db::Access& box,
    const std::vector<DirectionalId<3>>& mortars_to_reconstruct_to) {
  DirectionalIdMap<3, DataVector> neighbor_package_data{};
  if (mortars_to_reconstruct_to.empty()) {
    return neighbor_package_data;
  }
  using Correction = BoundaryCorrections::UpwindGradientProjection;
  const auto& boundary_correction =
      db::get<evolution::Tags::BoundaryCorrection>(box);
  if (typeid(boundary_correction) != typeid(Correction)) {
    ERROR(
        "The CCZ4 gradient-projection DG-FD coupling requires the "
        "UpwindGradientProjection boundary correction.");
  }
  if (db::get<domain::Tags::MeshVelocity<3>>(box).has_value()) {
    ERROR("Moving meshes are not supported by the CCZ4 DG-FD coupling.");
  }

  const Mesh<3>& dg_mesh = db::get<domain::Tags::Mesh<3>>(box);
  const Mesh<3>& subcell_mesh =
      db::get<evolution::dg::subcell::Tags::Mesh<3>>(box);
  const auto volume_vars_subcell = evolution::dg::subcell::fd::project(
      db::get<System::variables_tag>(box), dg_mesh, subcell_mesh.extents());
  const auto& ghost_subcell_data =
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box);
  const auto& mesh_for_ghost_data =
      db::get<evolution::dg::subcell::Tags::MeshForGhostData<3>>(box);
  const auto& mortar_meshes = db::get<evolution::dg::Tags::MortarMesh<3>>(box);
  const auto& neighbor_meshes = db::get<domain::Tags::NeighborMesh<3>>(box);
  const ::Ccz4::fd::Reconstructor& reconstructor =
      db::get<::Ccz4::fd::Tags::Reconstructor>(box);
  const auto& subcell_options =
      db::get<evolution::dg::subcell::Tags::SubcellOptions<3>>(box);
  const Element<3>& element = db::get<domain::Tags::Element<3>>(box);

  for (const auto& mortar_id : mortars_to_reconstruct_to) {
    const Direction<3>& direction = mortar_id.direction();
    if (not mesh_for_ghost_data.contains(mortar_id) or
        mesh_for_ghost_data.at(mortar_id).basis(0) !=
            Spectral::Basis::FiniteDifference) {
      ERROR("The neighbor "
            << mortar_id
            << " sent no boundary-correction data but its ghost data are not "
               "finite-difference data, so they carry no gradients.");
    }
    // The reconstructed package is stored on the local face mesh
    // (`neighbor_reconstructed_face_solution`) while the local mortar data are
    // projected to the mortar mesh, and the reconstruction needs ghost data on
    // the local subcell mesh: only conforming neighbors of equal order are
    // supported.
    const Mesh<2> face_mesh = dg_mesh.slice_away(direction.dimension());
    if (mortar_meshes.at(mortar_id) != face_mesh or
        mesh_for_ghost_data.at(mortar_id) != subcell_mesh) {
      ERROR(
          "The CCZ4 DG-FD coupling supports only conforming neighbors of "
          "equal order, but on element "
          << element.id() << " the mortar " << mortar_id
          << " has the mortar mesh " << mortar_meshes.at(mortar_id)
          << " (local face mesh " << face_mesh
          << ") and the finite-difference neighbor's ghost data are on "
          << mesh_for_ghost_data.at(mortar_id) << " (local subcell mesh "
          << subcell_mesh << "; neighbor DG mesh as received: "
          << (neighbor_meshes.contains(mortar_id)
                  ? get_output(neighbor_meshes.at(mortar_id))
                  : std::string{"none"})
          << ").");
    }
    const size_t number_of_face_points =
        subcell_mesh.extents().slice_away(direction.dimension()).product();

    Variables<System::variables_tag_list> vars_on_face{number_of_face_points};
    call_with_dynamic_type<
        void, typename ::Ccz4::fd::Reconstructor::creatable_classes>(
        &reconstructor,
        [&element, &mortar_id, &ghost_subcell_data, &subcell_mesh,
         &vars_on_face, &volume_vars_subcell](const auto& typed_reconstructor) {
          typed_reconstructor->reconstruct_fd_neighbor(
              make_not_null(&vars_on_face), volume_vars_subcell, element,
              ghost_subcell_data, subcell_mesh, mortar_id.direction());
        });

    // The outward normal of the neighbor, projected to the FD face. The
    // correction uses only the local DG normal.
    const auto& dg_normal_covector =
        get<evolution::dg::Tags::NormalCovector<3>>(
            *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<3>>(box)
                 .at(direction));
    tnsr::i<DataVector, 3> normal_covector{number_of_face_points};
    for (size_t i = 0; i < 3; ++i) {
      const DataVector reversed_normal = -1.0 * dg_normal_covector.get(i);
      normal_covector.get(i) = evolution::dg::subcell::fd::project(
          reversed_normal, dg_mesh.slice_away(direction.dimension()),
          subcell_mesh.extents().slice_away(direction.dimension()));
    }

    const PackagedFaceData packaged = package_finite_difference_face(
        std::move(vars_on_face), normal_covector,
        dynamic_cast<const Correction&>(boundary_correction), direction);
    const PackagedFaceData dg_packaged =
        evolution::dg::subcell::fd::reconstruct(
            packaged, dg_mesh.slice_away(direction.dimension()),
            subcell_mesh.extents().slice_away(direction.dimension()),
            subcell_options.reconstruction_method());

    // The reconstruction onto the DG face can overshoot.
    const auto& potentials = get<Tags::PackagedPotentials>(dg_packaged);
    if (min(potentials.get(PotentialIndex::conformal_factor)) <= 0.0 or
        min(potentials.get(PotentialIndex::lapse)) <= 0.0) {
      ERROR(
          "The conformal factor or lapse of the finite-difference "
          "neighbor's package on the DG face of mortar "
          << mortar_id << " is not positive: conformal factor "
          << potentials.get(PotentialIndex::conformal_factor) << ", lapse "
          << potentials.get(PotentialIndex::lapse));
    }
    DataVector& result = neighbor_package_data[mortar_id];
    result.destructive_resize(dg_packaged.size());
    std::copy(dg_packaged.data(), dg_packaged.data() + dg_packaged.size(),
              result.begin());
  }
  return neighbor_package_data;
}
}  // namespace Ccz4::GradientProjection

template void evolution::dg::subcell::neighbor_reconstructed_face_solution<
    3, Ccz4::GradientProjection::NeighborPackagedData>(
    gsl::not_null<db::Access*> box);
