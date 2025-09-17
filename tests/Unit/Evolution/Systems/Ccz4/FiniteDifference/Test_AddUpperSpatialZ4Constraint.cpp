// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/AddUpperSpatialZ4Constraint.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::fd {
namespace {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.AddUpperSpatialZ4Constraint",
                  "[Unit][Evolution]") {
  const size_t num_pts = 5;
  Scalar<DataVector> lapse{DataVector{num_pts, 1.5}};

  tnsr::I<DataVector, 3> initial_spatial_z4_constraint{};

  auto box = db::create<
      db::AddSimpleTags<gr::Tags::Lapse<DataVector>,
                        ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>>>(
      std::move(lapse), std::move(initial_spatial_z4_constraint));

  db::mutate_apply<AddUpperSpatialZ4Constraint>(make_not_null(&box));

  initial_spatial_z4_constraint =
      db::get<::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>>(box);
  for (size_t i = 0; i < 3; ++i) {
    for (auto& val : initial_spatial_z4_constraint.get(i)) {
      CHECK(val == 0.0);
    }
  }
}
}  // namespace
}  // namespace Ccz4::fd
