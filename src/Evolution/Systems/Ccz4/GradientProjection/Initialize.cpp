// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/Initialize.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::GradientProjection {
void InitializeLocalOperators::apply(
    const gsl::not_null<LocalOperators*> local_operators, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const Scalar<DataVector>& det_inverse_jacobian,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    const size_t shell_angular_modes_to_kill) {
  if (not grid_to_inertial_map.is_identity()) {
    ERROR(
        "The CCZ4 gradient-projection scheme supports only static meshes, but "
        "the grid-to-inertial map is not the identity.");
  }
  *local_operators =
      LocalOperators{mesh, inverse_jacobian, det_inverse_jacobian,
                     shell_angular_modes_to_kill};
}
}  // namespace Ccz4::GradientProjection
