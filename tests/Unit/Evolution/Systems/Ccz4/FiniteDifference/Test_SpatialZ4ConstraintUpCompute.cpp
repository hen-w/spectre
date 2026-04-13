// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SpatialZ4ConstraintUpCompute.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::fd {
namespace {

// On flat space (gamma-tilde = delta, phi = 1, gamma-hat^i = 0,
// field_d = 0) the contracted conformal Christoffel is zero, so
// Z^i = (1/2) phi^2 (gamma-hat^i - Gamma-tilde^i) = 0.
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.Fd.SpatialZ4ConstraintUpCompute",
    "[Unit][Evolution]") {
  const size_t num_pts = 5;

  Scalar<DataVector> conformal_factor{DataVector{num_pts, 1.0}};
  tnsr::ii<DataVector, 3> conformal_metric(num_pts, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    conformal_metric.get(i, i) = 1.0;
  }
  tnsr::I<DataVector, 3> gamma_hat(num_pts, 0.0);
  tnsr::ijj<DataVector, 3> field_d(num_pts, 0.0);

  auto box = db::create<
      db::AddSimpleTags<::Ccz4::Tags::ConformalFactor<DataVector>,
                        ::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                        ::Ccz4::Tags::GammaHat<DataVector, 3>,
                        ::Ccz4::Tags::FieldD<DataVector, 3>>,
      db::AddComputeTags<SpatialZ4ConstraintUpCompute>>(
      std::move(conformal_factor), std::move(conformal_metric),
      std::move(gamma_hat), std::move(field_d));

  const auto& z_up =
      db::get<::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>>(box);
  for (size_t i = 0; i < 3; ++i) {
    for (const auto& val : z_up.get(i)) {
      CHECK(val == 0.0);
    }
  }
}
}  // namespace
}  // namespace Ccz4::fd
