// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ResizeTimeDerivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::fd {
namespace {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.ResizeTimeDerivatives",
                  "[Unit][Evolution]") {
  const size_t num_pts = 5;
  Scalar<DataVector> lapse{DataVector{num_pts, 1.5}};

  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, Ccz4::fd::System::variables_tag>;
  // Initialize with wrong size to test resizing
  typename dt_variables_tag::type time_derivs{3, 99.0};

  auto box = db::create<
      db::AddSimpleTags<gr::Tags::Lapse<DataVector>, dt_variables_tag>>(
      std::move(lapse), std::move(time_derivs));

  // Check initial size is wrong
  CHECK(get<dt_variables_tag>(box).number_of_grid_points() == 3);

  db::mutate_apply<ResizeTimeDerivatives>(make_not_null(&box));

  // Check that time derivatives are resized to match lapse size
  CHECK(get<dt_variables_tag>(box).number_of_grid_points() == num_pts);

  // Check that all variables in time derivatives are initialized to 0.0
  const auto& dt_vars = get<dt_variables_tag>(box);

  // Test all variables from the Ccz4::fd::System
  CHECK(get<::Tags::dt<::Ccz4::Tags::ConformalMetric<DataVector, 3>>>(
            dt_vars) == tnsr::ii<DataVector, 3>{DataVector{num_pts, 0.0}});
  CHECK(get<::Tags::dt<::Ccz4::Tags::ConformalFactor<DataVector>>>(dt_vars) ==
        Scalar<DataVector>{DataVector{num_pts, 0.0}});
  CHECK(get<::Tags::dt<::Ccz4::Tags::ATilde<DataVector, 3>>>(dt_vars) ==
        tnsr::ii<DataVector, 3>{DataVector{num_pts, 0.0}});
  CHECK(get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
            dt_vars) == Scalar<DataVector>{DataVector{num_pts, 0.0}});
  CHECK(get<::Tags::dt<::Ccz4::Tags::Theta<DataVector>>>(dt_vars) ==
        Scalar<DataVector>{DataVector{num_pts, 0.0}});
  CHECK(get<::Tags::dt<::Ccz4::Tags::GammaHat<DataVector, 3>>>(dt_vars) ==
        tnsr::I<DataVector, 3>{DataVector{num_pts, 0.0}});
  CHECK(get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(dt_vars) ==
        Scalar<DataVector>{DataVector{num_pts, 0.0}});
  CHECK(get<::Tags::dt<gr::Tags::Shift<DataVector, 3>>>(dt_vars) ==
        tnsr::I<DataVector, 3>{DataVector{num_pts, 0.0}});
  CHECK(get<::Tags::dt<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>(
            dt_vars) == tnsr::I<DataVector, 3>{DataVector{num_pts, 0.0}});
}

}  // namespace
}  // namespace Ccz4::fd
