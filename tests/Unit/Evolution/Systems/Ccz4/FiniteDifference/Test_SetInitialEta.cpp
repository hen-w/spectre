// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SetInitialEta.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::fd {
namespace {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.SetInitialEta",
                  "[Unit][Evolution]") {
  const size_t num_pts = 5;
  Scalar<DataVector> lapse{DataVector{num_pts, 1.5}};

  Scalar<DataVector> initial_eta{};

  auto box = db::create<db::AddSimpleTags<gr::Tags::Lapse<DataVector>,
                                          ::Ccz4::Tags::Eta<DataVector>>>(
      std::move(lapse), std::move(initial_eta));

  db::mutate_apply<SetInitialEta>(make_not_null(&box));

  CHECK(get<::Ccz4::Tags::Eta<DataVector>>(box) ==
        Scalar<DataVector>{DataVector{num_pts, 0.0}});
}

}  // namespace
}  // namespace Ccz4::fd
