// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Ccz4/Ccz4WrappedGr.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/MomentumConstraintCompute.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::fd {
namespace {

// [[TimeOut, 20]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.Fd.MomentumConstraintCompute",
    "[Unit][Evolution]") {
  // Minkowski: phi=1, gamma_tilde=delta, A_tilde=0, K=0, D=0, P=0 -> M^i=0.
  {
    const Mesh<3> mesh(5, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
    const size_t num_pts = mesh.number_of_grid_points();

    InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
        inv_jac(num_pts, 0.0);
    for (size_t i = 0; i < 3; ++i) {
      inv_jac.get(i, i) = 1.0;
    }

    Scalar<DataVector> conformal_factor{DataVector{num_pts, 1.0}};
    tnsr::ii<DataVector, 3> conformal_metric(num_pts, 0.0);
    for (size_t i = 0; i < 3; ++i) {
      conformal_metric.get(i, i) = 1.0;
    }
    tnsr::ii<DataVector, 3> a_tilde(num_pts, 0.0);
    Scalar<DataVector> trace_extrinsic_curvature{DataVector{num_pts, 0.0}};
    tnsr::ijj<DataVector, 3> field_d(num_pts, 0.0);
    tnsr::i<DataVector, 3> field_p(num_pts, 0.0);

    const auto box = db::create<
        db::AddSimpleTags<
            Ccz4::Tags::ConformalFactor<DataVector>,
            Ccz4::Tags::ConformalMetric<DataVector, 3>,
            Ccz4::Tags::ATilde<DataVector, 3>,
            gr::Tags::TraceExtrinsicCurvature<DataVector>,
            Ccz4::Tags::FieldD<DataVector, 3>,
            Ccz4::Tags::FieldP<DataVector, 3>, domain::Tags::Mesh<3>,
            domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                          Frame::Inertial>>,
        db::AddComputeTags<MomentumConstraintCompute>>(
        std::move(conformal_factor), std::move(conformal_metric),
        std::move(a_tilde), std::move(trace_extrinsic_curvature),
        std::move(field_d), std::move(field_p), mesh, std::move(inv_jac));

    const auto& momentum =
        db::get<gr::Tags::MomentumConstraint<DataVector, 3, Frame::Inertial>>(
            box);
    for (size_t i = 0; i < 3; ++i) {
      for (const auto& val : momentum.get(i)) {
        CHECK(val == approx(0.0));
      }
    }
  }

  // KerrSchild: verify spectral convergence of M^i to zero.
  const Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::KerrSchild>
      wrapped_solution(1.0, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}});
  const double center = 5.0;
  const double half_length = 1.0;

  auto compute_max_m = [&](const size_t num_1d) {
    const Mesh<3> mesh(num_1d, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
    const size_t num_pts = mesh.number_of_grid_points();

    const auto logical_coords = logical_coordinates(mesh);
    tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords(num_pts);
    for (size_t d = 0; d < 3; ++d) {
      inertial_coords.get(d) =
          center + half_length * logical_coords.get(d);
    }

    InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
        inv_jac(num_pts, 0.0);
    for (size_t d = 0; d < 3; ++d) {
      inv_jac.get(d, d) = 1.0 / half_length;
    }

    const auto ccz4_vars = wrapped_solution.variables(
        inertial_coords, 0.0,
        tmpl::list<Ccz4::Tags::ConformalFactor<DataVector>,
                   Ccz4::Tags::ConformalMetric<DataVector, 3>,
                   Ccz4::Tags::ATilde<DataVector, 3>,
                   gr::Tags::TraceExtrinsicCurvature<DataVector>,
                   Ccz4::Tags::FieldD<DataVector, 3>,
                   Ccz4::Tags::FieldP<DataVector, 3>>{});

    const auto box = db::create<
        db::AddSimpleTags<
            Ccz4::Tags::ConformalFactor<DataVector>,
            Ccz4::Tags::ConformalMetric<DataVector, 3>,
            Ccz4::Tags::ATilde<DataVector, 3>,
            gr::Tags::TraceExtrinsicCurvature<DataVector>,
            Ccz4::Tags::FieldD<DataVector, 3>,
            Ccz4::Tags::FieldP<DataVector, 3>, domain::Tags::Mesh<3>,
            domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                          Frame::Inertial>>,
        db::AddComputeTags<MomentumConstraintCompute>>(
        get<Ccz4::Tags::ConformalFactor<DataVector>>(ccz4_vars),
        get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(ccz4_vars),
        get<Ccz4::Tags::ATilde<DataVector, 3>>(ccz4_vars),
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(ccz4_vars),
        get<Ccz4::Tags::FieldD<DataVector, 3>>(ccz4_vars),
        get<Ccz4::Tags::FieldP<DataVector, 3>>(ccz4_vars),
        mesh, std::move(inv_jac));

    const auto& momentum =
        db::get<gr::Tags::MomentumConstraint<DataVector, 3, Frame::Inertial>>(
            box);
    double max_m = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      for (const auto& val : momentum.get(i)) {
        max_m = std::max(max_m, std::abs(val));
      }
    }
    return max_m;
  };

  const double m6 = compute_max_m(6);
  const double m8 = compute_max_m(8);
  const double m10 = compute_max_m(10);
  const double m12 = compute_max_m(12);

  CAPTURE(m6);
  CAPTURE(m8);
  CAPTURE(m10);
  CAPTURE(m12);

  CHECK(m8 < m6 / 10.0);
  CHECK(m10 < m8 / 10.0);
  CHECK(m12 < m10 / 10.0);
  CHECK(m12 < 1.0e-10);
}

}  // namespace
}  // namespace Ccz4::fd
