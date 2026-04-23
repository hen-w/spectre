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
#include "Evolution/Systems/Ccz4/FiniteDifference/HamiltonianConstraintCompute.hpp"
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
    "Unit.Evolution.Systems.Ccz4.Fd.HamiltonianConstraintCompute",
    "[Unit][Evolution]") {
  // Minkowski: phi=1, gamma_tilde=delta, A_tilde=0, K=0, D=0, P=0 -> H=0.
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
        db::AddComputeTags<HamiltonianConstraintCompute>>(
        std::move(conformal_factor), std::move(conformal_metric),
        std::move(a_tilde), std::move(trace_extrinsic_curvature),
        std::move(field_d), std::move(field_p), mesh, std::move(inv_jac));

    const auto& hamiltonian =
        db::get<gr::Tags::HamiltonianConstraint<DataVector>>(box);
    for (const auto& val : get(hamiltonian)) {
      CHECK(val == approx(0.0));
    }
  }

  // KerrSchild: verify spectral convergence of H to zero.
  const Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::KerrSchild>
      wrapped_solution(1.0, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}});
  const double center = 5.0;
  const double half_length = 1.0;

  auto compute_max_h = [&](const size_t num_1d) {
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
        db::AddComputeTags<HamiltonianConstraintCompute>>(
        get<Ccz4::Tags::ConformalFactor<DataVector>>(ccz4_vars),
        get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(ccz4_vars),
        get<Ccz4::Tags::ATilde<DataVector, 3>>(ccz4_vars),
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(ccz4_vars),
        get<Ccz4::Tags::FieldD<DataVector, 3>>(ccz4_vars),
        get<Ccz4::Tags::FieldP<DataVector, 3>>(ccz4_vars),
        mesh, std::move(inv_jac));

    const auto& hamiltonian =
        db::get<gr::Tags::HamiltonianConstraint<DataVector>>(box);
    double max_h = 0.0;
    for (const auto& val : get(hamiltonian)) {
      max_h = std::max(max_h, std::abs(val));
    }
    return max_h;
  };

  const double h6 = compute_max_h(6);
  const double h8 = compute_max_h(8);
  const double h10 = compute_max_h(10);
  const double h12 = compute_max_h(12);

  CAPTURE(h6);
  CAPTURE(h8);
  CAPTURE(h10);
  CAPTURE(h12);

  CHECK(h8 < h6 / 10.0);
  CHECK(h10 < h8 / 10.0);
  CHECK(h12 < h10 / 10.0);
  CHECK(h12 < 1.0e-10);
}

}  // namespace
}  // namespace Ccz4::fd
