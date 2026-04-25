// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/NeighborPackagedData.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.tpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/PackageDataImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/Ccz4/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UnlimitedDeg4Prim.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
template <bool IsAuxiliary>
DirectionalIdMap<3, DataVector> NeighborPackagedDataImpl<IsAuxiliary>::apply(
    const db::Access& box,
    const std::vector<DirectionalId<3>>& mortars_to_reconstruct_to) {
  using system = Ccz4::fd::System;
  using variables_tag_list = system::variables_tag_list;

  DirectionalIdMap<3, DataVector> neighbor_package_data{};
  if (mortars_to_reconstruct_to.empty()) {
    return neighbor_package_data;
  }

  ASSERT(not db::get<domain::Tags::MeshVelocity<3>>(box).has_value(),
         "Moving mesh not yet supported for CCZ4 DG-subcell.");

  const Mesh<3>& dg_mesh = db::get<domain::Tags::Mesh<3>>(box);
  const Mesh<3>& subcell_mesh =
      db::get<evolution::dg::subcell::Tags::Mesh<3>>(box);

  // Project DG volume data to subcell (FD) cell centers
  const auto volume_vars_subcell = evolution::dg::subcell::fd::project(
      db::get<typename system::variables_tag>(box), dg_mesh,
      subcell_mesh.extents());

  const auto& ghost_subcell_data =
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box);

  const Ccz4::fd::Reconstructor& recons =
      db::get<Ccz4::fd::Tags::Reconstructor>(box);

  const auto& boundary_correction =
      db::get<evolution::Tags::BoundaryCorrection>(box);
  using derived_boundary_corrections =
      Ccz4::BoundaryCorrections::standard_boundary_corrections<3>;

  const auto& subcell_options =
      db::get<evolution::dg::subcell::Tags::SubcellOptions<3>>(box);

  // The projected_tags list used by both dg_package_data and
  // dg_auxiliary_package_data: all 17 variables in variables_tag_list.
  // (flux_variables = empty, temporary_tags = empty, primitive_tags = empty)
  using dg_package_data_projected_tags = variables_tag_list;

  tmpl::for_each<derived_boundary_corrections>([&](auto derived_correction_v) {
    using DerivedCorrection = tmpl::type_from<decltype(derived_correction_v)>;

    if (typeid(boundary_correction) != typeid(DerivedCorrection)) {
      return;
    }

    using pkg_field_tags = tmpl::conditional_t<
        IsAuxiliary,
        typename DerivedCorrection::dg_auxiliary_package_field_tags,
        typename DerivedCorrection::dg_package_field_tags>;

    Variables<pkg_field_tags> packaged_data{0};

    for (const auto& mortar_id : mortars_to_reconstruct_to) {
      const Direction<3>& direction = mortar_id.direction();

      const size_t num_face_pts =
          subcell_mesh.extents().slice_away(direction.dimension()).product();

      // Allocate face variables: all 17 tags, zero-initialized so that
      // boundary second-order tags (not reconstructed) are zero.
      Variables<variables_tag_list> vars_on_face{num_face_pts, 0.0};

      // Create a view over just the 13 reconstructed tags for the
      // reconstruction call.
      auto reconstructed_vars_on_face =
          vars_on_face.template reference_subset<tags_list_for_reconstruct>();

      // FD reconstruct neighbor data to the shared face
      call_with_dynamic_type<void,
                             typename Ccz4::fd::Reconstructor::creatable_classes>(
          &recons,
          [&element = db::get<domain::Tags::Element<3>>(box), &mortar_id,
           &ghost_subcell_data, &subcell_mesh, &reconstructed_vars_on_face,
           &volume_vars_subcell](const auto& reconstructor) {
            reconstructor->reconstruct_fd_neighbor(
                make_not_null(&reconstructed_vars_on_face),
                volume_vars_subcell, element, ghost_subcell_data, subcell_mesh,
                mortar_id.direction());
          });

      // Get normal covector, negate (outward for neighbor = inward for us),
      // and project from DG face to FD face.
      tnsr::i<DataVector, 3, Frame::Inertial> normal_covector =
          get<evolution::dg::Tags::NormalCovector<3>>(
              *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<3>>(box)
                   .at(direction));
      for (auto& t : normal_covector) {
        t *= -1.0;
      }
      const auto dg_normal_covector = normal_covector;
      for (size_t i = 0; i < 3; ++i) {
        normal_covector.get(i) = evolution::dg::subcell::fd::project(
            dg_normal_covector.get(i),
            dg_mesh.slice_away(direction.dimension()),
            subcell_mesh.extents().slice_away(direction.dimension()));
      }

      // Package data on FD face.  Zero-initialize because
      // dg_package_data intentionally skips boundary second-order fields
      // (they are unused by dg_boundary_terms), but fd::reconstruct
      // operates on the entire Variables buffer.
      packaged_data.initialize(num_face_pts, 0.0);
      if constexpr (IsAuxiliary) {
        evolution::dg::Actions::detail::dg_auxiliary_package_data<system>(
            make_not_null(&packaged_data),
            dynamic_cast<const DerivedCorrection&>(boundary_correction),
            vars_on_face, normal_covector, {std::nullopt}, direction,
            dg_package_data_projected_tags{});
      } else {
        evolution::dg::Actions::detail::dg_package_data<system>(
            make_not_null(&packaged_data),
            dynamic_cast<const DerivedCorrection&>(boundary_correction),
            vars_on_face, normal_covector, {std::nullopt}, direction,
            dg_package_data_projected_tags{});
      }

      // Interpolate packaged data from FD face to DG face
      auto dg_packaged_data = evolution::dg::subcell::fd::reconstruct(
          packaged_data, dg_mesh.slice_away(direction.dimension()),
          subcell_mesh.extents().slice_away(direction.dimension()),
          subcell_options.reconstruction_method());

      DataVector dg_packaged_data_view{dg_packaged_data.data(),
                                       dg_packaged_data.size()};
      neighbor_package_data[mortar_id] = DataVector{dg_packaged_data.size()};
      std::copy(dg_packaged_data_view.begin(), dg_packaged_data_view.end(),
                neighbor_package_data[mortar_id].begin());
    }
  });

  return neighbor_package_data;
}

template class NeighborPackagedDataImpl<true>;
template class NeighborPackagedDataImpl<false>;
}  // namespace Ccz4::fd

// Explicit template instantiations required by the DG-subcell framework.
template void evolution::dg::subcell::neighbor_reconstructed_face_solution<
    3, Ccz4::fd::NeighborPackagedDataImpl<false>>(
    gsl::not_null<db::Access*> box);
template void evolution::dg::subcell::neighbor_reconstructed_face_solution<
    3, Ccz4::fd::NeighborPackagedDataImpl<true>>(
    gsl::not_null<db::Access*> box);
