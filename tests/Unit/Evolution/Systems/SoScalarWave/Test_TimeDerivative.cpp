// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Evolution/Systems/SoScalarWave/TimeDerivative.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SoPlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Separate tags for testing to avoid DataBox conflicts with temporary_tags
struct TemporaryDtPsiCopy : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct TemporaryDtPiCopy : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
void check_du_dt(const size_t npts, const double time) {
  const Mesh<Dim> mesh{npts, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};

  SoScalarWave::Solutions::SoPlaneWave<Dim> solution(
      make_array<Dim>(0.1), make_array<Dim>(0.0),
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2));

  tnsr::I<DataVector, Dim> x = [&]() {
    auto logical_coords = logical_coordinates(mesh);
    tnsr::I<DataVector, Dim> coords{pow<Dim>(npts)};
    for (size_t i = 0; i < Dim; ++i) {
      coords.get(i) = std::move(logical_coords.get(i));
    }
    return coords;
  }();

  auto local_check_du_dt = [&]() {
    InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
        inv_jac{pow<Dim>(npts)};
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        inv_jac.get(i, j) = (i == j ? 1.0 : 0.0);
      }
    }

    Variables<tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi>>
        evolved_vars(pow<Dim>(npts));
    // Initialize with analytic solution values
    get<SoScalarWave::Tags::Psi>(evolved_vars) = solution.psi(x, time);
    get<SoScalarWave::Tags::Pi>(evolved_vars) =
        Scalar<DataVector>(-1.0 * solution.dpsi_dt(x, time).get());

    auto box = db::create<db::AddSimpleTags<
        Tags::dt<SoScalarWave::Tags::Psi>, Tags::dt<SoScalarWave::Tags::Pi>,
        TemporaryDtPsiCopy, TemporaryDtPiCopy,
        Tags::Variables<
            tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi>>,
        domain::Tags::Mesh<Dim>,
        domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                      Frame::Inertial>>>(
        Scalar<DataVector>(pow<Dim>(npts), 0.0),
        Scalar<DataVector>(pow<Dim>(npts), 0.0),
        Scalar<DataVector>(pow<Dim>(npts), 0.0),
        Scalar<DataVector>(pow<Dim>(npts), 0.0), evolved_vars, mesh, inv_jac);

    db::mutate_apply<tmpl::list<Tags::dt<SoScalarWave::Tags::Psi>,
                                Tags::dt<SoScalarWave::Tags::Pi>,
                                TemporaryDtPsiCopy, TemporaryDtPiCopy>,
                     typename SoScalarWave::TimeDerivative<Dim>::argument_tags>(
        SoScalarWave::TimeDerivative<Dim>{}, make_not_null(&box));

    CHECK_ITERABLE_APPROX(db::get<Tags::dt<SoScalarWave::Tags::Psi>>(box),
                          solution.dpsi_dt(x, time));
    CHECK_ITERABLE_APPROX(
        db::get<Tags::dt<SoScalarWave::Tags::Pi>>(box),
        Scalar<DataVector>(-1.0 * solution.d2psi_dt2(x, time).get()));

    // Check that temporary tags are also populated correctly
    CHECK_ITERABLE_APPROX(db::get<TemporaryDtPsiCopy>(box),
                          db::get<Tags::dt<SoScalarWave::Tags::Psi>>(box));
    CHECK_ITERABLE_APPROX(db::get<TemporaryDtPiCopy>(box),
                          db::get<Tags::dt<SoScalarWave::Tags::Pi>>(box));
  };

  // Test the time derivative computation
  local_check_du_dt();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.SoScalarWave.TimeDerivative",
                  "[Unit][Evolution]") {
  constexpr double time = 0.7;
  check_du_dt<1>(3, time);
  check_du_dt<2>(3, time);
  check_du_dt<3>(3, time);
}
