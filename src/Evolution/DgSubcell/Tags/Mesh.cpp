// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Tags/Mesh.hpp"

#include "Evolution/DgSubcell/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell::Tags {
template <size_t VolumeDim>
void MeshCompute<VolumeDim>::function(
    const gsl::not_null<return_type*> subcell_mesh,
    const ::Mesh<VolumeDim>& dg_mesh) {
  // Subcell FD meshes are only meaningful for Legendre or Chebyshev DG
  // meshes.  For other bases (e.g. SphericalHarmonic shells that are
  // always DG-only) return a small valid FD dummy mesh.  The result is
  // never used at runtime — ObserverMeshCompute selects the DG mesh
  // when ActiveGrid == Dg — but the DataBox evaluates all argument_tags
  // of ObserverMeshCompute before calling its function, so this compute
  // tag must produce a valid FD mesh to avoid crashes in downstream
  // compute tags.
  if (dg_mesh.basis() !=
          make_array<VolumeDim>(Spectral::Basis::Legendre) and
      dg_mesh.basis() !=
          make_array<VolumeDim>(Spectral::Basis::Chebyshev)) {
    *subcell_mesh = ::Mesh<VolumeDim>{3, Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered};
    return;
  }
  *subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template struct MeshCompute<GET_DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef GET_DIM
}  // namespace evolution::dg::subcell::Tags
