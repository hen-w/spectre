// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/Systems/Ccz4/ATilde.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/LdgTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Ccz4::fd {
namespace {

using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

void test_minkowski() {
  constexpr size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  constexpr size_t points_per_dimension = 5;

  // DG mesh (GaussLobatto) — the LDG path uses spectral derivatives
  const Mesh<SpatialDim> mesh{points_per_dimension, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
  const size_t num_pts = mesh.number_of_grid_points();

  // Affine coordinate map [-1,1]^3 -> physical domain
  const std::array<double, SpatialDim> lower_bound{-2., 0., -0.5};
  const std::array<double, SpatialDim> upper_bound{2., 2., -0.1};
  const std::array<double, SpatialDim> coords_range{
      upper_bound[0] - lower_bound[0], upper_bound[1] - lower_bound[1],
      upper_bound[2] - lower_bound[2]};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  const auto logical_coords = logical_coordinates(mesh);
  const auto x = coord_map(logical_coords);

  // Diagonal inverse Jacobian for affine map
  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical, FrameType>
      inv_jacobian{num_pts, 0.0};
  for (size_t i = 0; i < SpatialDim; ++i) {
    inv_jacobian.get(i, i) = 2.0 / gsl::at(coords_range, i);
  }

  // Used-for-size DataVector
  const DataVector lapse_size(num_pts, 0.0);

  // Get Minkowski initial data (all 13 variables)
  auto evolved_vars = TestHelpers::Ccz4::fd::detail::Minkowski::
      compute_prim_solution_for_Minkowski(x);

  // The Minkowski helper does not initialize auxiliary fields (FieldA, FieldB,
  // FieldD, FieldP). For Minkowski they are all zero (spatial derivatives of
  // constant evolved variables).
  get<::Ccz4::Tags::FieldA<DataVector, 3>>(evolved_vars) =
      make_with_value<tnsr::i<DataVector, 3>>(lapse_size, 0.0);
  get<::Ccz4::Tags::FieldB<DataVector, 3>>(evolved_vars) =
      make_with_value<tnsr::iJ<DataVector, 3>>(lapse_size, 0.0);
  get<::Ccz4::Tags::FieldD<DataVector, 3>>(evolved_vars) =
      make_with_value<tnsr::ijj<DataVector, 3>>(lapse_size, 0.0);
  get<::Ccz4::Tags::FieldP<DataVector, 3>>(evolved_vars) =
      make_with_value<tnsr::i<DataVector, 3>>(lapse_size, 0.0);

  // Extract the 13 variable tensors
  const auto& conformal_metric =
      get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(evolved_vars);
  const auto& conformal_factor =
      get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars);
  const auto& a_tilde = get<::Ccz4::Tags::ATilde<DataVector, 3>>(evolved_vars);
  const auto& trace_extrinsic_curvature =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars);
  const auto& theta = get<::Ccz4::Tags::Theta<DataVector>>(evolved_vars);
  const auto& gamma_hat =
      get<::Ccz4::Tags::GammaHat<DataVector, 3>>(evolved_vars);
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(evolved_vars);
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(evolved_vars);
  const auto& b =
      get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(evolved_vars);
  const auto& field_a = get<::Ccz4::Tags::FieldA<DataVector, 3>>(evolved_vars);
  const auto& field_b = get<::Ccz4::Tags::FieldB<DataVector, 3>>(evolved_vars);
  const auto& field_d = get<::Ccz4::Tags::FieldD<DataVector, 3>>(evolved_vars);
  const auto& field_p = get<::Ccz4::Tags::FieldP<DataVector, 3>>(evolved_vars);

  // Compute partial derivatives on the spectral mesh.
  // For Minkowski, all fields are constant so derivatives are zero.
  const auto d_conformal_metric =
      partial_derivative(conformal_metric, mesh, inv_jacobian);
  const auto d_conformal_factor =
      partial_derivative(conformal_factor, mesh, inv_jacobian);
  const auto d_a_tilde = partial_derivative(a_tilde, mesh, inv_jacobian);
  const auto d_trace_extrinsic_curvature =
      partial_derivative(trace_extrinsic_curvature, mesh, inv_jacobian);
  const auto d_theta = partial_derivative(theta, mesh, inv_jacobian);
  const auto d_gamma_hat = partial_derivative(gamma_hat, mesh, inv_jacobian);
  const auto d_lapse = partial_derivative(lapse, mesh, inv_jacobian);
  const auto d_shift = partial_derivative(shift, mesh, inv_jacobian);
  const auto d_b = partial_derivative(b, mesh, inv_jacobian);
  const auto d_field_a_raw = partial_derivative(field_a, mesh, inv_jacobian);
  const auto d_field_b_raw = partial_derivative(field_b, mesh, inv_jacobian);
  const auto d_field_d_raw = partial_derivative(field_d, mesh, inv_jacobian);
  const auto d_field_p_raw = partial_derivative(field_p, mesh, inv_jacobian);

  // Allocate dt output variables, NaN-initialized
  const auto nan_val = std::numeric_limits<double>::signaling_NaN();
  auto dt_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3>>(lapse_size, nan_val);
  auto dt_conformal_factor =
      make_with_value<Scalar<DataVector>>(lapse_size, nan_val);
  auto dt_a_tilde =
      make_with_value<tnsr::ii<DataVector, 3>>(lapse_size, nan_val);
  auto dt_trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(lapse_size, nan_val);
  auto dt_theta = make_with_value<Scalar<DataVector>>(lapse_size, nan_val);
  auto dt_gamma_hat =
      make_with_value<tnsr::I<DataVector, 3>>(lapse_size, nan_val);
  auto dt_lapse = make_with_value<Scalar<DataVector>>(lapse_size, nan_val);
  auto dt_shift = make_with_value<tnsr::I<DataVector, 3>>(lapse_size, nan_val);
  auto dt_b = make_with_value<tnsr::I<DataVector, 3>>(lapse_size, nan_val);
  auto dt_field_a =
      make_with_value<tnsr::i<DataVector, 3>>(lapse_size, nan_val);
  auto dt_field_b =
      make_with_value<tnsr::iJ<DataVector, 3>>(lapse_size, nan_val);
  auto dt_field_d =
      make_with_value<tnsr::ijj<DataVector, 3>>(lapse_size, nan_val);
  auto dt_field_p =
      make_with_value<tnsr::i<DataVector, 3>>(lapse_size, nan_val);

  // Boundary second-order field dt outputs (zero-initialized)
  auto dt_boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3>>(lapse_size, 0.0);
  auto dt_boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(lapse_size, 0.0);
  auto dt_boundary_lapse = make_with_value<Scalar<DataVector>>(lapse_size, 0.0);
  auto dt_boundary_shift =
      make_with_value<tnsr::I<DataVector, 3>>(lapse_size, 0.0);

  // Boundary second-order field derivative inputs (zero-valued)
  const auto d_boundary_conformal_metric =
      make_with_value<tnsr::ijj<DataVector, 3>>(lapse_size, 0.0);
  const auto d_boundary_conformal_factor =
      make_with_value<tnsr::i<DataVector, 3>>(lapse_size, 0.0);
  const auto d_boundary_lapse =
      make_with_value<tnsr::i<DataVector, 3>>(lapse_size, 0.0);
  const auto d_boundary_shift =
      make_with_value<tnsr::iJ<DataVector, 3>>(lapse_size, 0.0);

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  // Call LdgTimeDerivative::apply directly
  LdgTimeDerivative::apply(
      make_not_null(&dt_conformal_metric), make_not_null(&dt_conformal_factor),
      make_not_null(&dt_a_tilde), make_not_null(&dt_trace_extrinsic_curvature),
      make_not_null(&dt_theta), make_not_null(&dt_gamma_hat),
      make_not_null(&dt_lapse), make_not_null(&dt_shift), make_not_null(&dt_b),
      make_not_null(&dt_field_a), make_not_null(&dt_field_b),
      make_not_null(&dt_field_d), make_not_null(&dt_field_p),
      make_not_null(&dt_boundary_conformal_metric),
      make_not_null(&dt_boundary_conformal_factor),
      make_not_null(&dt_boundary_lapse), make_not_null(&dt_boundary_shift),
      // partial derivatives
      d_conformal_metric, d_conformal_factor, d_a_tilde,
      d_trace_extrinsic_curvature, d_theta, d_gamma_hat, d_lapse, d_shift, d_b,
      d_field_a_raw, d_field_b_raw, d_field_d_raw, d_field_p_raw,
      d_boundary_conformal_metric, d_boundary_conformal_factor,
      d_boundary_lapse, d_boundary_shift,
      // argument_tags (variable values)
      conformal_metric, conformal_factor, a_tilde, trace_extrinsic_curvature,
      theta, gamma_hat, lapse, shift, b, field_a, field_b, field_d, field_p,
      // kappa parameters
      kappa_1, kappa_2, kappa_3,
      // eta, k_0, evolve_lapse_and_shift
      Scalar<DataVector>(num_pts, 0.0), Scalar<DataVector>(num_pts, 0.0), true,
      Element<SpatialDim>{}, mesh,
      std::vector<DirectionMap<
          SpatialDim,
          std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>{},
      inv_jacobian);

  // Verify all 13 dt variables are zero
  const DataVector zero(num_pts, 0.0);

  for (const auto& component : dt_conformal_metric) {
    CAPTURE("dt_conformal_metric");
    CHECK_ITERABLE_APPROX(component, zero);
  }
  CAPTURE("dt_conformal_factor");
  CHECK_ITERABLE_APPROX(get(dt_conformal_factor), zero);

  for (const auto& component : dt_a_tilde) {
    CAPTURE("dt_a_tilde");
    CHECK_ITERABLE_APPROX(component, zero);
  }
  CAPTURE("dt_trace_extrinsic_curvature");
  CHECK_ITERABLE_APPROX(get(dt_trace_extrinsic_curvature), zero);

  CAPTURE("dt_theta");
  CHECK_ITERABLE_APPROX(get(dt_theta), zero);

  for (const auto& component : dt_gamma_hat) {
    CAPTURE("dt_gamma_hat");
    CHECK_ITERABLE_APPROX(component, zero);
  }
  CAPTURE("dt_lapse");
  CHECK_ITERABLE_APPROX(get(dt_lapse), zero);

  for (const auto& component : dt_shift) {
    CAPTURE("dt_shift");
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (const auto& component : dt_b) {
    CAPTURE("dt_b");
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (const auto& component : dt_field_a) {
    CAPTURE("dt_field_a");
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (const auto& component : dt_field_b) {
    CAPTURE("dt_field_b");
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (const auto& component : dt_field_d) {
    CAPTURE("dt_field_d");
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (const auto& component : dt_field_p) {
    CAPTURE("dt_field_p");
    CHECK_ITERABLE_APPROX(component, zero);
  }
}

void test_kerrschild() {
  constexpr size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  constexpr size_t points_per_dimension = 10;
  const bool evolve_lapse_and_shift = true;
  const bool evolve_shift = evolve_lapse_and_shift;

  // DG mesh (GaussLobatto) — the LDG path uses spectral derivatives
  const Mesh<SpatialDim> mesh{points_per_dimension, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
  const size_t num_pts = mesh.number_of_grid_points();

  // Affine coordinate map — away from singularity (same as SO test)
  const std::array<double, SpatialDim> lower_bound{0.8, 1.0, 1.3};
  const std::array<double, SpatialDim> upper_bound{1.2, 1.2, 1.4};
  const std::array<double, SpatialDim> coords_range{
      upper_bound[0] - lower_bound[0], upper_bound[1] - lower_bound[1],
      upper_bound[2] - lower_bound[2]};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  const auto logical_coords = logical_coordinates(mesh);
  const auto x = coord_map(logical_coords);

  // Diagonal inverse Jacobian for affine map
  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical, FrameType>
      inv_jacobian{num_pts, 0.0};
  for (size_t i = 0; i < SpatialDim; ++i) {
    inv_jacobian.get(i, i) = 2.0 / gsl::at(coords_range, i);
  }

  // Setup KerrSchild solution
  const double mass = 2.0;
  const std::array<double, SpatialDim> spin{{0.2, 0.4, 0.8}};
  const std::array<double, SpatialDim> center{{0.2, 0.5, 0.1}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const double t = std::numeric_limits<double>::signaling_NaN();
  const double f = System::f;

  // Get evolved variables (9 evolved fields, NOT auxiliaries)
  auto evolved_vars = TestHelpers::Ccz4::fd::detail::KerrSchild::
      compute_prim_solution_for_KerrSchild(x, t, f, evolve_shift, solution);

  // Extract evolved variable references
  const auto& conformal_metric =
      get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(evolved_vars);
  const auto& conformal_factor_scalar =
      get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars);
  const auto& a_tilde = get<::Ccz4::Tags::ATilde<DataVector, 3>>(evolved_vars);
  const auto& trace_extrinsic_curvature =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars);
  const auto& theta = get<::Ccz4::Tags::Theta<DataVector>>(evolved_vars);
  const auto& gamma_hat =
      get<::Ccz4::Tags::GammaHat<DataVector, 3>>(evolved_vars);
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(evolved_vars);
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(evolved_vars);
  const auto& b =
      get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(evolved_vars);

  // Compute auxiliary fields analytically from KerrSchild solution
  const auto kerrschild_vars = solution.variables(
      x, t, typename gr::Solutions::KerrSchild::tags<DataVector>{});

  // FieldA: A_i = d_i(alpha) / alpha
  const auto& d_lapse_analytic =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<SpatialDim>,
                        FrameType>>(kerrschild_vars);
  tnsr::i<DataVector, SpatialDim> field_a(num_pts);
  for (size_t i = 0; i < SpatialDim; ++i) {
    field_a.get(i) = d_lapse_analytic.get(i) / get(lapse);
  }

  // FieldB: B_{ki} = d_k(beta^i)
  const auto& d_shift_analytic =
      get<::Tags::deriv<gr::Tags::Shift<DataVector, SpatialDim, FrameType>,
                        tmpl::size_t<SpatialDim>, FrameType>>(kerrschild_vars);
  const auto& field_b = d_shift_analytic;

  // FieldD: D_{kij} = 0.5 * d_k(tilde_gamma_{ij})
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim, FrameType>>(
          kerrschild_vars);
  const auto& d_spatial_metric = get<
      ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, SpatialDim, FrameType>,
                    tmpl::size_t<SpatialDim>, FrameType>>(kerrschild_vars);
  const auto det_spatial_metric = determinant(spatial_metric);
  const auto conformal_factor = pow(get(det_spatial_metric), -1.0 / 6.0);
  Scalar<DataVector> conformal_factor_squared{};
  get(conformal_factor_squared) = square(conformal_factor);
  const auto& d_det_spatial_metric =
      get<gr::Tags::DerivDetSpatialMetric<DataVector, SpatialDim, FrameType>>(
          kerrschild_vars);
  const auto d_conformal_spatial_metric =
      TestHelpers::Ccz4::fd::detail::KerrSchild::get_d_conformal_spatial_metric<
          SpatialDim, FrameType>(conformal_factor_squared, spatial_metric,
                                 d_spatial_metric, d_det_spatial_metric);
  const auto field_d = TestHelpers::Ccz4::fd::detail::KerrSchild::get_field_d<
      SpatialDim, FrameType>(d_conformal_spatial_metric);

  // FieldP: P_i = d_i(ln(phi)) = -d_i(det(gamma)) / (6 * det(gamma))
  tnsr::i<DataVector, SpatialDim> field_p(num_pts);
  for (size_t i = 0; i < SpatialDim; ++i) {
    field_p.get(i) =
        -d_det_spatial_metric.get(i) / (6.0 * get(det_spatial_metric));
  }

  // Set auxiliary fields in evolved_vars for the Variables container
  get<::Ccz4::Tags::FieldA<DataVector, 3>>(evolved_vars) = field_a;
  get<::Ccz4::Tags::FieldB<DataVector, 3>>(evolved_vars) = field_b;
  get<::Ccz4::Tags::FieldD<DataVector, 3>>(evolved_vars) = field_d;
  get<::Ccz4::Tags::FieldP<DataVector, 3>>(evolved_vars) = field_p;

  // Compute spectral partial derivatives of all 13 variables
  const auto d_conformal_metric_spec =
      partial_derivative(conformal_metric, mesh, inv_jacobian);
  const auto d_conformal_factor_spec =
      partial_derivative(conformal_factor_scalar, mesh, inv_jacobian);
  const auto d_a_tilde = partial_derivative(a_tilde, mesh, inv_jacobian);
  const auto d_trace_extrinsic_curvature =
      partial_derivative(trace_extrinsic_curvature, mesh, inv_jacobian);
  const auto d_theta = partial_derivative(theta, mesh, inv_jacobian);
  const auto d_gamma_hat = partial_derivative(gamma_hat, mesh, inv_jacobian);
  const auto d_lapse_spectral = partial_derivative(lapse, mesh, inv_jacobian);
  const auto d_shift_spec = partial_derivative(shift, mesh, inv_jacobian);
  const auto d_b = partial_derivative(b, mesh, inv_jacobian);
  const auto d_field_a_raw = partial_derivative(field_a, mesh, inv_jacobian);
  const auto d_field_b_raw = partial_derivative(field_b, mesh, inv_jacobian);
  const auto d_field_d_raw = partial_derivative(field_d, mesh, inv_jacobian);
  const auto d_field_p_raw = partial_derivative(field_p, mesh, inv_jacobian);

  // Compute eta, k_0, and slicing condition
  const auto eta = make_with_value<Scalar<DataVector>>(lapse, 0.1);
  const Scalar<DataVector> slicing_condition =
      TestHelpers::Ccz4::fd::detail::KerrSchild::get_slicing_condition(
          ::Ccz4::SlicingConditionType::Log, lapse);
  const auto k_0 = TestHelpers::Ccz4::fd::detail::KerrSchild::get_k_0_kerr(
      shift, lapse, d_lapse_spectral, slicing_condition, theta,
      trace_extrinsic_curvature);

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  // Allocate dt output variables, NaN-initialized
  const auto nan_val = std::numeric_limits<double>::signaling_NaN();
  const DataVector used_for_size(num_pts, nan_val);
  auto dt_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3>>(used_for_size, nan_val);
  auto dt_conformal_factor =
      make_with_value<Scalar<DataVector>>(used_for_size, nan_val);
  auto dt_a_tilde =
      make_with_value<tnsr::ii<DataVector, 3>>(used_for_size, nan_val);
  auto dt_trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(used_for_size, nan_val);
  auto dt_theta = make_with_value<Scalar<DataVector>>(used_for_size, nan_val);
  auto dt_gamma_hat =
      make_with_value<tnsr::I<DataVector, 3>>(used_for_size, nan_val);
  auto dt_lapse = make_with_value<Scalar<DataVector>>(used_for_size, nan_val);
  auto dt_shift =
      make_with_value<tnsr::I<DataVector, 3>>(used_for_size, nan_val);
  auto dt_b = make_with_value<tnsr::I<DataVector, 3>>(used_for_size, nan_val);
  auto dt_field_a =
      make_with_value<tnsr::i<DataVector, 3>>(used_for_size, nan_val);
  auto dt_field_b =
      make_with_value<tnsr::iJ<DataVector, 3>>(used_for_size, nan_val);
  auto dt_field_d =
      make_with_value<tnsr::ijj<DataVector, 3>>(used_for_size, nan_val);
  auto dt_field_p =
      make_with_value<tnsr::i<DataVector, 3>>(used_for_size, nan_val);

  // Boundary second-order field dt outputs (zero-initialized)
  auto dt_boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3>>(used_for_size, 0.0);
  auto dt_boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto dt_boundary_lapse =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  auto dt_boundary_shift =
      make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 0.0);

  // Boundary second-order field derivative inputs (zero-valued)
  const auto d_boundary_conformal_metric =
      make_with_value<tnsr::ijj<DataVector, 3>>(used_for_size, 0.0);
  const auto d_boundary_conformal_factor =
      make_with_value<tnsr::i<DataVector, 3>>(used_for_size, 0.0);
  const auto d_boundary_lapse =
      make_with_value<tnsr::i<DataVector, 3>>(used_for_size, 0.0);
  const auto d_boundary_shift =
      make_with_value<tnsr::iJ<DataVector, 3>>(used_for_size, 0.0);

  // Call LdgTimeDerivative::apply
  LdgTimeDerivative::apply(
      make_not_null(&dt_conformal_metric), make_not_null(&dt_conformal_factor),
      make_not_null(&dt_a_tilde), make_not_null(&dt_trace_extrinsic_curvature),
      make_not_null(&dt_theta), make_not_null(&dt_gamma_hat),
      make_not_null(&dt_lapse), make_not_null(&dt_shift), make_not_null(&dt_b),
      make_not_null(&dt_field_a), make_not_null(&dt_field_b),
      make_not_null(&dt_field_d), make_not_null(&dt_field_p),
      make_not_null(&dt_boundary_conformal_metric),
      make_not_null(&dt_boundary_conformal_factor),
      make_not_null(&dt_boundary_lapse), make_not_null(&dt_boundary_shift),
      // partial derivatives
      d_conformal_metric_spec, d_conformal_factor_spec, d_a_tilde,
      d_trace_extrinsic_curvature, d_theta, d_gamma_hat, d_lapse_spectral,
      d_shift_spec, d_b, d_field_a_raw, d_field_b_raw, d_field_d_raw,
      d_field_p_raw, d_boundary_conformal_metric,
      d_boundary_conformal_factor, d_boundary_lapse, d_boundary_shift,
      // argument_tags (variable values)
      conformal_metric, conformal_factor_scalar, a_tilde,
      trace_extrinsic_curvature, theta, gamma_hat, lapse, shift, b, field_a,
      field_b, field_d, field_p,
      // kappa parameters
      kappa_1, kappa_2, kappa_3,
      // eta, k_0, evolve_lapse_and_shift
      eta, k_0, evolve_lapse_and_shift, Element<SpatialDim>{}, mesh,
      std::vector<DirectionMap<
          SpatialDim,
          std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>{},
      inv_jacobian);

  // dt_b: expected non-zero value from eq 12i
  const tnsr::I<DataVector, SpatialDim, FrameType> dt_b_expected =
      TestHelpers::Ccz4::fd::detail::KerrSchild::get_dt_b_kerr_expected(
          evolve_shift, eta, shift, d_gamma_hat, b, d_b);

  // dt_field_a: expected non-zero because the production code assumes d_k_0 = 0
  // but the test uses a spatially-varying k_0 to make dt_lapse = 0.
  // From LdgTimeDerivative.hpp eq 12j (with d_k_0 = 0):
  //   dt_A_k = -2*(d_K(k) - 2*d_theta(k)) + B_k^l*A_l + beta^l*d_l(A_k)
  // Symmetrize d_field_a_raw the same way the production code does.
  tnsr::ii<DataVector, SpatialDim> d_field_a_sym(num_pts);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      d_field_a_sym.get(i, j) =
          0.5 * (d_field_a_raw.get(i, j) + d_field_a_raw.get(j, i));
    }
  }
  tnsr::i<DataVector, SpatialDim> dt_field_a_expected(num_pts);
  for (size_t k = 0; k < SpatialDim; ++k) {
    dt_field_a_expected.get(k) =
        -2.0 * (d_trace_extrinsic_curvature.get(k) - 2.0 * d_theta.get(k));
    for (size_t l = 0; l < SpatialDim; ++l) {
      dt_field_a_expected.get(k) += field_b.get(k, l) * field_a.get(l) +
                                    shift.get(l) * d_field_a_sym.get(k, l);
    }
  }

  // Check results with tolerance
  const Approx custom_approx =
      Approx::custom().epsilon(1.0e-10).scale(*std::max_element(
          evolved_vars.data(), evolved_vars.data() + evolved_vars.size() - 1));

  // Check all dt vars except dt_b and dt_field_a are zero
  const auto zero = DataVector(num_pts, 0.0);
  for (const auto& component : dt_conformal_metric) {
    CAPTURE("dt_conformal_metric");
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, custom_approx);
  }
  CHECK_ITERABLE_CUSTOM_APPROX(get(dt_conformal_factor), zero, custom_approx);
  for (const auto& component : dt_a_tilde) {
    CAPTURE("dt_a_tilde");
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, custom_approx);
  }
  CHECK_ITERABLE_CUSTOM_APPROX(get(dt_trace_extrinsic_curvature), zero,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get(dt_theta), zero, custom_approx);
  for (const auto& component : dt_gamma_hat) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, custom_approx);
  }
  CHECK_ITERABLE_CUSTOM_APPROX(get(dt_lapse), zero, custom_approx);
  for (const auto& component : dt_shift) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, custom_approx);
  }
  CHECK_ITERABLE_CUSTOM_APPROX(dt_field_a, dt_field_a_expected, custom_approx);
  for (const auto& component : dt_field_b) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, custom_approx);
  }
  for (const auto& component : dt_field_d) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, custom_approx);
  }
  for (const auto& component : dt_field_p) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, custom_approx);
  }
  CHECK_ITERABLE_CUSTOM_APPROX(dt_b, dt_b_expected, custom_approx);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.FiniteDifference.LdgTimeDerivative",
    "[Unit][Evolution]") {
  test_minkowski();
  test_kerrschild();
}

}  // namespace
}  // namespace Ccz4::fd
