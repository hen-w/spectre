// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/TimeDerivativeDirichlet.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Characteristics.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Overwrites the time derivative at external boundary face nodes
 * by freezing incoming characteristic modes for the LDG (DG) discretization.
 *
 * \details This MutateApply action runs after
 * `ApplyBoundaryCorrectionsToTimeDerivative` in the DG step. For each
 * external face with a `TimeDerivativeDirichlet` boundary condition, it:
 * 1. Computes partial derivatives via spectral differentiation
 * 2. Decomposes the time derivative into characteristic fields
 * 3. Zeros all incoming dt characteristic modes
 * 4. Inverse-transforms and overwrites dt at the boundary face
 *
 * This is a simpler alternative to `OverwriteExternalBoundaryDt` (CRPBC):
 * no constraint-preserving or radiation-preserving corrections, just clean
 * freezing of incoming modes.
 *
 * \warning Assumes Gauss-Lobatto quadrature (boundary nodes exist).
 */
struct OverwriteExternalBoundaryDtDirichlet {
  static constexpr size_t Dim = 3;
  using System = ::Ccz4::fd::System;
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, System::variables_tag>;

  using return_tags = tmpl::list<dt_variables_tag>;

  using argument_tags = tmpl::list<
      domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
      domain::Tags::ExternalBoundaryConditions<Dim>,
      domain::Tags::Coordinates<Dim, Frame::Inertial>,
      domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      System::variables_tag,
      ::Ccz4::Tags::K0<DataVector>,
      ::Ccz4::fd::Tags::EvolveLapseAndShift>;

  static void apply(
      const gsl::not_null<Variables<typename dt_variables_tag::tags_list>*>
          dt_vars,
      const Element<Dim>& element,
      const Mesh<Dim>& mesh,
      const std::vector<DirectionMap<
          Dim,
          std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>&
          all_boundary_conditions,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*inertial_coords*/,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const Variables<typename System::variables_tag::tags_list>& evolved_vars,
      const Scalar<DataVector>& k_0,
      const bool evolve_lapse_and_shift) {
    // Phase A: Boundary detection
    if (element.external_boundaries().empty()) {
      return;
    }

    std::vector<Direction<Dim>> td_dirichlet_directions;
    const auto& block_boundary_conditions =
        all_boundary_conditions.at(element.id().block_id());
    for (const auto& direction : element.external_boundaries()) {
      const auto* td_bc = dynamic_cast<
          const Ccz4::BoundaryConditions::TimeDerivativeDirichlet*>(
          block_boundary_conditions.at(direction).get());
      if (td_bc != nullptr) {
        td_dirichlet_directions.push_back(direction);
      }
    }
    if (td_dirichlet_directions.empty()) {
      return;
    }

    // ASSERT(evolve_lapse_and_shift,
    //        "TimeDerivativeDirichlet BC requires evolving lapse and shift.");

    // ASSERT(mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto and
    //            mesh.quadrature(1) == Spectral::Quadrature::GaussLobatto and
    //            mesh.quadrature(2) == Spectral::Quadrature::GaussLobatto,
    //        "OverwriteExternalBoundaryDtDirichlet requires Gauss-Lobatto "
    //        "quadrature but got "
    //            << mesh.quadrature(0) << ", " << mesh.quadrature(1) << ", "
    //            << mesh.quadrature(2));

    const size_t num_pts = mesh.number_of_grid_points();

    // Extract evolved fields
    const auto& conformal_metric =
        get<::Ccz4::Tags::ConformalMetric<DataVector, Dim>>(evolved_vars);
    const auto& conformal_factor =
        get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars);
    const auto& a_tilde =
        get<::Ccz4::Tags::ATilde<DataVector, Dim>>(evolved_vars);
    const auto& trace_extrinsic_curvature =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars);
    const auto& theta =
        get<::Ccz4::Tags::Theta<DataVector>>(evolved_vars);
    const auto& gamma_hat =
        get<::Ccz4::Tags::GammaHat<DataVector, Dim>>(evolved_vars);
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(evolved_vars);
    const auto& shift = get<gr::Tags::Shift<DataVector, Dim>>(evolved_vars);
    const auto& field_a =
        get<::Ccz4::Tags::FieldA<DataVector, Dim>>(evolved_vars);
    const auto& field_b =
        get<::Ccz4::Tags::FieldB<DataVector, Dim>>(evolved_vars);
    const auto& field_d =
        get<::Ccz4::Tags::FieldD<DataVector, Dim>>(evolved_vars);
    const auto& field_p =
        get<::Ccz4::Tags::FieldP<DataVector, Dim>>(evolved_vars);

    // Phase B: Compute partial derivatives via spectral differentiation
    const auto partial_derivs = partial_derivatives<
        typename System::variables_tag::tags_list>(evolved_vars, mesh,
                                                   inv_jacobian);

    // Extract raw derivatives of auxiliary fields
    const auto& d_field_a_raw =
        get<::Tags::deriv<::Ccz4::Tags::FieldA<DataVector, Dim>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(partial_derivs);
    const auto& d_field_b_raw =
        get<::Tags::deriv<::Ccz4::Tags::FieldB<DataVector, Dim>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(partial_derivs);
    const auto& d_field_d_raw =
        get<::Tags::deriv<::Ccz4::Tags::FieldD<DataVector, Dim>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(partial_derivs);
    const auto& d_field_p_raw =
        get<::Tags::deriv<::Ccz4::Tags::FieldP<DataVector, Dim>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(partial_derivs);
    const auto& d_a_tilde =
        get<::Tags::deriv<::Ccz4::Tags::ATilde<DataVector, Dim>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(partial_derivs);
    const auto& d_trace_extrinsic_curvature =
        get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(partial_derivs);
    const auto& d_theta =
        get<::Tags::deriv<::Ccz4::Tags::Theta<DataVector>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(partial_derivs);
    const auto& d_b =
        get<::Tags::deriv<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(partial_derivs);

    // Phase C: Symmetrize auxiliary field derivatives (full volume)
    tnsr::ii<DataVector, Dim> d_field_a(num_pts);
    ::tenex::evaluate<ti::i, ti::j>(
        make_not_null(&d_field_a),
        0.5 * (d_field_a_raw(ti::i, ti::j) + d_field_a_raw(ti::j, ti::i)));

    tnsr::iiJ<DataVector, Dim> d_field_b(num_pts);
    ::tenex::evaluate<ti::i, ti::j, ti::K>(
        make_not_null(&d_field_b),
        0.5 * (d_field_b_raw(ti::i, ti::j, ti::K) +
               d_field_b_raw(ti::j, ti::i, ti::K)));

    tnsr::iijj<DataVector, Dim> d_field_d(num_pts);
    ::tenex::evaluate<ti::i, ti::j, ti::k, ti::l>(
        make_not_null(&d_field_d),
        0.5 * (d_field_d_raw(ti::i, ti::j, ti::k, ti::l) +
               d_field_d_raw(ti::j, ti::i, ti::k, ti::l)));

    tnsr::ii<DataVector, Dim> d_field_p(num_pts);
    ::tenex::evaluate<ti::i, ti::j>(
        make_not_null(&d_field_p),
        0.5 * (d_field_p_raw(ti::i, ti::j) + d_field_p_raw(ti::j, ti::i)));

    // Loop over each TimeDerivativeDirichlet face
    for (const auto& direction : td_dirichlet_directions) {
      const size_t normal_dim = direction.dimension();
      const size_t N_normal = mesh.extents(normal_dim);
      size_t num_face_pts = 1;
      for (size_t d = 0; d < Dim; ++d) {
        if (d != normal_dim) {
          num_face_pts *= mesh.extents(d);
        }
      }
      const size_t inner_stride = [&]() {
        size_t s = 1;
        for (size_t d = 0; d < normal_dim; ++d) {
          s *= mesh.extents(d);
        }
        return s;
      }();
      const size_t outermost_layer =
          (direction.side() == Side::Upper) ? N_normal - 1 : 0;

      // Volume index mapping from (face_idx, layer) to linear index
      auto volume_index = [&](size_t face_idx, size_t layer) -> size_t {
        return (face_idx % inner_stride) +
               inner_stride * layer +
               inner_stride * N_normal * (face_idx / inner_stride);
      };

      // Slice helpers
      auto slice_scalar = [&](Scalar<DataVector>& face,
                              const Scalar<DataVector>& vol) {
        get(face).destructive_resize(num_face_pts);
        for (size_t fp = 0; fp < num_face_pts; ++fp) {
          get(face)[fp] = get(vol)[volume_index(fp, outermost_layer)];
        }
      };

      auto slice_tensor = [&]<typename TensorType>(TensorType& face,
                                                    const TensorType& vol) {
        for (size_t ti = 0; ti < vol.size(); ++ti) {
          face[ti].destructive_resize(num_face_pts);
          for (size_t fp = 0; fp < num_face_pts; ++fp) {
            face[ti][fp] = vol[ti][volume_index(fp, outermost_layer)];
          }
        }
      };

      // Slice evolved variables to outermost face
      tnsr::ii<DataVector, Dim> conformal_metric_face;
      slice_tensor(conformal_metric_face, conformal_metric);
      Scalar<DataVector> conformal_factor_face;
      slice_scalar(conformal_factor_face, conformal_factor);
      tnsr::ii<DataVector, Dim> a_tilde_face;
      slice_tensor(a_tilde_face, a_tilde);
      Scalar<DataVector> trace_K_face;
      slice_scalar(trace_K_face, trace_extrinsic_curvature);
      Scalar<DataVector> theta_face;
      slice_scalar(theta_face, theta);
      tnsr::I<DataVector, Dim> gamma_hat_face;
      slice_tensor(gamma_hat_face, gamma_hat);
      Scalar<DataVector> lapse_face;
      slice_scalar(lapse_face, lapse);
      tnsr::I<DataVector, Dim> shift_face;
      slice_tensor(shift_face, shift);
      tnsr::i<DataVector, Dim> field_a_face;
      slice_tensor(field_a_face, field_a);
      tnsr::iJ<DataVector, Dim> field_b_face;
      slice_tensor(field_b_face, field_b);
      tnsr::ijj<DataVector, Dim> field_d_face;
      slice_tensor(field_d_face, field_d);
      tnsr::i<DataVector, Dim> field_p_face;
      slice_tensor(field_p_face, field_p);

      // Compute first derivatives on face from auxiliary fields
      tnsr::ijj<DataVector, Dim> d_conformal_metric_face(num_face_pts);
      ::tenex::evaluate<ti::i, ti::j, ti::k>(
          make_not_null(&d_conformal_metric_face),
          2.0 * field_d_face(ti::i, ti::j, ti::k));
      tnsr::i<DataVector, Dim> d_conformal_factor_face(num_face_pts);
      ::tenex::evaluate<ti::i>(
          make_not_null(&d_conformal_factor_face),
          conformal_factor_face() * field_p_face(ti::i));
      tnsr::i<DataVector, Dim> d_lapse_face(num_face_pts);
      ::tenex::evaluate<ti::i>(
          make_not_null(&d_lapse_face),
          lapse_face() * field_a_face(ti::i));
      tnsr::iJ<DataVector, Dim> d_shift_face(num_face_pts);
      ::tenex::evaluate<ti::i, ti::J>(
          make_not_null(&d_shift_face),
          field_b_face(ti::i, ti::J));
      tnsr::ijj<DataVector, Dim> d_a_tilde_face;
      slice_tensor(d_a_tilde_face, d_a_tilde);
      tnsr::i<DataVector, Dim> d_trace_K_face;
      slice_tensor(d_trace_K_face, d_trace_extrinsic_curvature);
      tnsr::i<DataVector, Dim> d_theta_face;
      slice_tensor(d_theta_face, d_theta);
      tnsr::iJ<DataVector, Dim> d_b_face;
      slice_tensor(d_b_face, d_b);

      // Slice symmetrized auxiliary derivatives to face
      tnsr::ii<DataVector, Dim> d_field_a_face;
      slice_tensor(d_field_a_face, d_field_a);
      tnsr::iiJ<DataVector, Dim> d_field_b_face;
      slice_tensor(d_field_b_face, d_field_b);
      tnsr::iijj<DataVector, Dim> d_field_d_face;
      slice_tensor(d_field_d_face, d_field_d);
      tnsr::ii<DataVector, Dim> d_field_p_face;
      slice_tensor(d_field_p_face, d_field_p);

      // Phase D: Compute face geometry
      Scalar<DataVector> conformal_factor_squared_face(num_face_pts);
      get(conformal_factor_squared_face) = square(get(conformal_factor_face));

      auto [det_conformal_metric_face, inv_conformal_metric_face] =
          determinant_and_inverse(conformal_metric_face);

      tnsr::II<DataVector, Dim> inv_spatial_metric_face(num_face_pts);
      ::tenex::evaluate<ti::I, ti::J>(
          make_not_null(&inv_spatial_metric_face),
          conformal_factor_squared_face() *
              inv_conformal_metric_face(ti::I, ti::J));

      tnsr::ii<DataVector, Dim> spatial_metric_face(num_face_pts);
      ::tenex::evaluate<ti::i, ti::j>(
          make_not_null(&spatial_metric_face),
          conformal_metric_face(ti::i, ti::j) /
              conformal_factor_squared_face());

      // Compute unit normal from inverse Jacobian
      InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
          outermost_inv_jacobian;
      slice_tensor(outermost_inv_jacobian, inv_jacobian);

      tnsr::i<DataVector, Dim> unnormalized_normal_one_form(num_face_pts);
      for (size_t i = 0; i < Dim; ++i) {
        unnormalized_normal_one_form.get(i) =
            static_cast<double>(direction.sign()) *
            outermost_inv_jacobian.get(normal_dim, i);
      }

      tnsr::I<DataVector, Dim> unit_normal_vector(num_face_pts);
      ::tenex::evaluate<ti::I>(
          make_not_null(&unit_normal_vector),
          inv_spatial_metric_face(ti::I, ti::J) *
              unnormalized_normal_one_form(ti::j));

      Scalar<DataVector> magnitude(num_face_pts);
      ::tenex::evaluate(
          make_not_null(&magnitude),
          sqrt(unit_normal_vector(ti::I) *
               unnormalized_normal_one_form(ti::i)));
      ::tenex::evaluate<ti::I>(
          make_not_null(&unit_normal_vector),
          unit_normal_vector(ti::I) / magnitude());

      const tnsr::i<DataVector, Dim> unit_normal_one_form =
          raise_or_lower_index(unit_normal_vector, spatial_metric_face);

      // Phase E: Compute second derivatives from auxiliary field derivatives
      tnsr::ii<DataVector, Dim> d_d_lapse_face(num_face_pts);
      ::tenex::evaluate<ti::i, ti::j>(
          make_not_null(&d_d_lapse_face),
          lapse_face() * (d_field_a_face(ti::i, ti::j) +
                          field_a_face(ti::i) * field_a_face(ti::j)));

      tnsr::ii<DataVector, Dim> d_d_conformal_factor_face(num_face_pts);
      ::tenex::evaluate<ti::i, ti::j>(
          make_not_null(&d_d_conformal_factor_face),
          conformal_factor_face() *
              (d_field_p_face(ti::i, ti::j) +
               field_p_face(ti::i) * field_p_face(ti::j)));

      tnsr::iijj<DataVector, Dim> d_d_conformal_metric_face(num_face_pts);
      ::tenex::evaluate<ti::i, ti::j, ti::k, ti::l>(
          make_not_null(&d_d_conformal_metric_face),
          2.0 * d_field_d_face(ti::i, ti::j, ti::k, ti::l));

      tnsr::iiJ<DataVector, Dim> d_d_shift_face(num_face_pts);
      ::tenex::evaluate<ti::i, ti::j, ti::K>(
          make_not_null(&d_d_shift_face),
          d_field_b_face(ti::i, ti::j, ti::K));

      // k_minus_k0_minus_2_theta_c for dt_d_lapse computation
      constexpr double c_param = 1.0;
      Scalar<DataVector> k_0_face;
      slice_scalar(k_0_face, k_0);
      Scalar<DataVector> k_minus_k0_minus_2_theta_c_face(num_face_pts);
      get(k_minus_k0_minus_2_theta_c_face) =
          get(trace_K_face) - get(k_0_face) -
          2.0 * c_param * get(theta_face);

      // Phase E2: Compute dt of partial derivatives from PDE
      constexpr double one_third = 1.0 / 3.0;
      constexpr double f_param = System::f;
      const bool shifting_shift = System::shifting_shift;

      // Slice dt vars to outermost face
      Scalar<DataVector> outermost_dt_trace_K;
      slice_scalar(outermost_dt_trace_K,
                   get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<
                       DataVector>>>(*dt_vars));
      tnsr::ii<DataVector, Dim> outermost_dt_a_tilde;
      slice_tensor(outermost_dt_a_tilde,
                   get<::Tags::dt<::Ccz4::Tags::ATilde<DataVector, Dim>>>(
                       *dt_vars));
      Scalar<DataVector> outermost_dt_theta;
      slice_scalar(outermost_dt_theta,
                   get<::Tags::dt<::Ccz4::Tags::Theta<DataVector>>>(*dt_vars));
      tnsr::I<DataVector, Dim> outermost_dt_gamma_hat;
      slice_tensor(
          outermost_dt_gamma_hat,
          get<::Tags::dt<::Ccz4::Tags::GammaHat<DataVector, Dim>>>(*dt_vars));
      tnsr::I<DataVector, Dim> outermost_dt_b;
      slice_tensor(
          outermost_dt_b,
          get<::Tags::dt<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>>>(
              *dt_vars));

      // dt of spatial derivatives
      tnsr::ijj<DataVector, Dim> outermost_dt_d_conformal_metric(
          num_face_pts);
      ::tenex::evaluate<ti::k, ti::i, ti::j>(
          make_not_null(&outermost_dt_d_conformal_metric),
          -2.0 * (d_a_tilde_face(ti::k, ti::i, ti::j) * lapse_face() +
                  a_tilde_face(ti::i, ti::j) * d_lapse_face(ti::k)) +
              d_conformal_metric_face(ti::k, ti::l, ti::i) *
                  d_shift_face(ti::j, ti::L) +
              d_conformal_metric_face(ti::k, ti::l, ti::j) *
                  d_shift_face(ti::i, ti::L) +
              conformal_metric_face(ti::i, ti::l) *
                  d_d_shift_face(ti::k, ti::j, ti::L) +
              conformal_metric_face(ti::j, ti::l) *
                  d_d_shift_face(ti::k, ti::i, ti::L) -
              2.0 * one_third *
                  (conformal_metric_face(ti::i, ti::j) *
                       d_d_shift_face(ti::k, ti::l, ti::L) +
                   d_shift_face(ti::l, ti::L) *
                       d_conformal_metric_face(ti::k, ti::i, ti::j)) +
              shift_face(ti::L) *
                  d_d_conformal_metric_face(ti::k, ti::l, ti::i, ti::j) +
              d_shift_face(ti::k, ti::L) *
                  d_conformal_metric_face(ti::l, ti::i, ti::j));

      tnsr::i<DataVector, Dim> outermost_dt_d_conformal_factor(num_face_pts);
      ::tenex::evaluate<ti::k>(
          make_not_null(&outermost_dt_d_conformal_factor),
          one_third *
                  (d_trace_K_face(ti::k) * conformal_factor_face() *
                       lapse_face() +
                   trace_K_face() * d_conformal_factor_face(ti::k) *
                       lapse_face() +
                   trace_K_face() * d_lapse_face(ti::k) *
                       conformal_factor_face() -
                   conformal_factor_face() *
                       d_d_shift_face(ti::k, ti::l, ti::L) -
                   d_conformal_factor_face(ti::k) *
                       d_shift_face(ti::l, ti::L)) +
              shift_face(ti::L) *
                  d_d_conformal_factor_face(ti::k, ti::l) +
              d_shift_face(ti::k, ti::L) * d_conformal_factor_face(ti::l));

      tnsr::i<DataVector, Dim> outermost_dt_d_lapse(num_face_pts);
      ::tenex::evaluate<ti::k>(
          make_not_null(&outermost_dt_d_lapse),
          -2.0 * (d_lapse_face(ti::k) *
                      k_minus_k0_minus_2_theta_c_face() +
                  d_trace_K_face(ti::k) * lapse_face() -
                  2.0 * d_theta_face(ti::k) * c_param * lapse_face()) +
              d_shift_face(ti::k, ti::L) * d_lapse_face(ti::l) +
              shift_face(ti::L) * d_d_lapse_face(ti::k, ti::l));

      tnsr::iJ<DataVector, Dim> outermost_dt_d_shift(num_face_pts);
      ::tenex::evaluate<ti::k, ti::I>(
          make_not_null(&outermost_dt_d_shift),
          f_param * d_b_face(ti::k, ti::I));
      if (shifting_shift) {
        ::tenex::update<ti::k, ti::I>(
            make_not_null(&outermost_dt_d_shift),
            outermost_dt_d_shift(ti::k, ti::I) +
                d_shift_face(ti::k, ti::L) * d_shift_face(ti::l, ti::I) +
                shift_face(ti::L) * d_d_shift_face(ti::k, ti::l, ti::I));
      }

      // Phase F: dt characteristic fields and speeds
      auto dt_char_fields = dt_characteristic_fields(
          unit_normal_one_form, conformal_metric_face, conformal_factor_face,
          lapse_face, shift_face, outermost_dt_trace_K, outermost_dt_a_tilde,
          outermost_dt_theta, outermost_dt_gamma_hat, outermost_dt_b,
          outermost_dt_d_conformal_metric, outermost_dt_d_conformal_factor,
          outermost_dt_d_lapse, outermost_dt_d_shift, f_param);

      const auto char_speeds = characteristic_speeds(
          lapse_face, shift_face, conformal_factor_face, f_param,
          unit_normal_one_form);

      // Phase F2: Speed validation
      for (size_t i = 0; i < num_face_pts; ++i) {
        if (char_speeds[0][i] < 0.0 or char_speeds[1][i] >= 0.0 or
            char_speeds[5][i] < 0.0 or char_speeds[6][i] >= 0.0 or
            char_speeds[12][i] < 0.0 or char_speeds[13][i] >= 0.0 or
            char_speeds[14][i] < 0.0 or char_speeds[15][i] >= 0.0) {
          ERROR(
              "TimeDerivativeDirichlet BCs require asymptotically "
              "Minkowskian coordinates but the characteristic speeds at "
              "boundary point "
              << i << " are char_speeds[0] = " << char_speeds[0][i]
              << ", char_speeds[1] = " << char_speeds[1][i]
              << ", char_speeds[5] = " << char_speeds[5][i]
              << ", char_speeds[6] = " << char_speeds[6][i]
              << ", char_speeds[12] = " << char_speeds[12][i]
              << ", char_speeds[13] = " << char_speeds[13][i]
              << ", char_speeds[14] = " << char_speeds[14][i]
              << ", char_speeds[15] = " << char_speeds[15][i]);
        }
      }

      // Phase G: Zero all incoming dt characteristic modes
      // UTensorMinus (speed[1], always incoming)
      get<::Tags::dt<Tags::UTensorMinus<DataVector, Dim, Frame::Inertial>>>(
          dt_char_fields) =
          make_with_value<tnsr::ii<DataVector, Dim>>(lapse_face, 0.0);

      // UVector2Minus (speed[4], always incoming)
      get<::Tags::dt<Tags::UVector2Minus<DataVector, Dim, Frame::Inertial>>>(
          dt_char_fields) =
          make_with_value<tnsr::i<DataVector, Dim>>(lapse_face, 0.0);

      // UVector3Minus (speed[6], always incoming)
      get<::Tags::dt<Tags::UVector3Minus<DataVector, Dim, Frame::Inertial>>>(
          dt_char_fields) =
          make_with_value<tnsr::i<DataVector, Dim>>(lapse_face, 0.0);

      // UVector1Zero (speed[2], incoming when speed < 0, point-by-point)
      {
        auto& dt_u_vector1_zero =
            get<::Tags::dt<Tags::UVector1Zero<DataVector, Dim,
                                              Frame::Inertial>>>(
                dt_char_fields);
        for (size_t s = 0; s < num_face_pts; ++s) {
          if (char_speeds[2][s] < 0.0) {
            for (size_t j = 0; j < Dim; ++j) {
              dt_u_vector1_zero.get(j)[s] = 0.0;
            }
          }
        }
      }

      // UScalar1Zero (speed[7], incoming when speed < 0, point-by-point)
      {
        auto& dt_u_scalar1_zero =
            get<::Tags::dt<Tags::UScalar1Zero<DataVector>>>(dt_char_fields);
        for (size_t s = 0; s < num_face_pts; ++s) {
          if (char_speeds[7][s] < 0.0) {
            get(dt_u_scalar1_zero)[s] = 0.0;
          }
        }
      }

      // UScalar2Minus (speed[9], always incoming)
      get<::Tags::dt<Tags::UScalar2Minus<DataVector>>>(dt_char_fields) =
          make_with_value<Scalar<DataVector>>(lapse_face, 0.0);

      // UScalar3Minus (speed[11], always incoming)
      get<::Tags::dt<Tags::UScalar3Minus<DataVector>>>(dt_char_fields) =
          make_with_value<Scalar<DataVector>>(lapse_face, 0.0);

      // UScalar4Minus (speed[13], always incoming)
      get<::Tags::dt<Tags::UScalar4Minus<DataVector>>>(dt_char_fields) =
          make_with_value<Scalar<DataVector>>(lapse_face, 0.0);

      // UScalar5Minus (speed[15], always incoming)
      get<::Tags::dt<Tags::UScalar5Minus<DataVector>>>(dt_char_fields) =
          make_with_value<Scalar<DataVector>>(lapse_face, 0.0);

      // Phase H: Inverse characteristic transform
      const auto modified_dt_evolved_vars =
          dt_evolved_space_from_dt_characteristic_fields(
              get<::Tags::dt<Tags::UTensorPlus<DataVector, Dim,
                                               Frame::Inertial>>>(
                  dt_char_fields),
              get<::Tags::dt<Tags::UTensorMinus<DataVector, Dim,
                                                Frame::Inertial>>>(
                  dt_char_fields),
              get<::Tags::dt<Tags::UVector1Zero<DataVector, Dim,
                                                Frame::Inertial>>>(
                  dt_char_fields),
              get<::Tags::dt<Tags::UVector2Plus<DataVector, Dim,
                                                Frame::Inertial>>>(
                  dt_char_fields),
              get<::Tags::dt<Tags::UVector2Minus<DataVector, Dim,
                                                 Frame::Inertial>>>(
                  dt_char_fields),
              get<::Tags::dt<Tags::UVector3Plus<DataVector, Dim,
                                                Frame::Inertial>>>(
                  dt_char_fields),
              get<::Tags::dt<Tags::UVector3Minus<DataVector, Dim,
                                                 Frame::Inertial>>>(
                  dt_char_fields),
              get<::Tags::dt<Tags::UScalar1Zero<DataVector>>>(dt_char_fields),
              get<::Tags::dt<Tags::UScalar2Plus<DataVector>>>(dt_char_fields),
              get<::Tags::dt<Tags::UScalar2Minus<DataVector>>>(dt_char_fields),
              get<::Tags::dt<Tags::UScalar3Plus<DataVector>>>(dt_char_fields),
              get<::Tags::dt<Tags::UScalar3Minus<DataVector>>>(dt_char_fields),
              get<::Tags::dt<Tags::UScalar4Plus<DataVector>>>(dt_char_fields),
              get<::Tags::dt<Tags::UScalar4Minus<DataVector>>>(dt_char_fields),
              get<::Tags::dt<Tags::UScalar5Plus<DataVector>>>(dt_char_fields),
              get<::Tags::dt<Tags::UScalar5Minus<DataVector>>>(dt_char_fields),
              unit_normal_one_form, conformal_metric_face,
              conformal_factor_face, lapse_face, shift_face, f_param);

      // Phase I: Overwrite dt at boundary face — directly modified variables
      // (a_tilde, trace_K, theta, gamma_hat, b)
      {
        auto& dt_a_tilde_vol =
            get<::Tags::dt<::Ccz4::Tags::ATilde<DataVector, Dim>>>(*dt_vars);
        const auto& modified_dt_a_tilde =
            get<::Tags::dt<::Ccz4::Tags::ATilde<DataVector, Dim>>>(
                modified_dt_evolved_vars);
        for (size_t ti = 0; ti < dt_a_tilde_vol.size(); ++ti) {
          for (size_t fp = 0; fp < num_face_pts; ++fp) {
            dt_a_tilde_vol[ti][volume_index(fp, outermost_layer)] =
                modified_dt_a_tilde[ti][fp];
          }
        }
      }
      {
        auto& dt_K_vol =
            get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
                *dt_vars);
        const auto& modified_dt_K =
            get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
                modified_dt_evolved_vars);
        for (size_t fp = 0; fp < num_face_pts; ++fp) {
          get(dt_K_vol)[volume_index(fp, outermost_layer)] =
              get(modified_dt_K)[fp];
        }
      }
      {
        auto& dt_theta_vol =
            get<::Tags::dt<::Ccz4::Tags::Theta<DataVector>>>(*dt_vars);
        const auto& modified_dt_theta =
            get<::Tags::dt<::Ccz4::Tags::Theta<DataVector>>>(
                modified_dt_evolved_vars);
        for (size_t fp = 0; fp < num_face_pts; ++fp) {
          get(dt_theta_vol)[volume_index(fp, outermost_layer)] =
              get(modified_dt_theta)[fp];
        }
      }
      {
        auto& dt_gamma_hat_vol =
            get<::Tags::dt<::Ccz4::Tags::GammaHat<DataVector, Dim>>>(
                *dt_vars);
        const auto& modified_dt_gamma_hat =
            get<::Tags::dt<::Ccz4::Tags::GammaHat<DataVector, Dim>>>(
                modified_dt_evolved_vars);
        for (size_t i = 0; i < Dim; ++i) {
          for (size_t fp = 0; fp < num_face_pts; ++fp) {
            dt_gamma_hat_vol.get(i)[volume_index(fp, outermost_layer)] =
                modified_dt_gamma_hat.get(i)[fp];
          }
        }
      }
      {
        auto& dt_b_vol =
            get<::Tags::dt<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>>>(
                *dt_vars);
        const auto& modified_dt_b =
            get<::Tags::dt<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>>>(
                modified_dt_evolved_vars);
        for (size_t i = 0; i < Dim; ++i) {
          for (size_t fp = 0; fp < num_face_pts; ++fp) {
            dt_b_vol.get(i)[volume_index(fp, outermost_layer)] =
                modified_dt_b.get(i)[fp];
          }
        }
      }

      // Phase J: Reconstruct dt for metric/gauge vars using spectral
      // differentiation matrix
      const auto& modified_dt_dn_conformal_metric =
          get<::Tags::dt<Tags::DnConformalMetric<DataVector, Dim,
                                                 Frame::Inertial>>>(
              modified_dt_evolved_vars);
      const auto& modified_dt_dn_conformal_factor =
          get<::Tags::dt<Tags::DnConformalFactor<DataVector>>>(
              modified_dt_evolved_vars);
      const auto& modified_dt_dn_lapse =
          get<::Tags::dt<Tags::DnLapse<DataVector>>>(
              modified_dt_evolved_vars);
      const auto& modified_dt_dn_shift =
          get<::Tags::dt<Tags::DnShift<DataVector, Dim, Frame::Inertial>>>(
              modified_dt_evolved_vars);

      // Compute jacobian_factor
      const tnsr::I<DataVector, Dim, Frame::ElementLogical>
          inv_jacobian_dot_normal = ::tenex::evaluate<ti::I>(
              unit_normal_vector(ti::J) *
              outermost_inv_jacobian(ti::I, ti::j));
      Scalar<DataVector> jacobian_factor(num_face_pts);
      get(jacobian_factor) = inv_jacobian_dot_normal.get(normal_dim);

      // Get the 1D spectral differentiation matrix in the normal direction
      const auto diff_matrix = Spectral::differentiation_matrix(
          mesh.slice_through(normal_dim));
      const double d_NN = diff_matrix(outermost_layer, outermost_layer);

      // Lambda to reconstruct dt at the outermost face point for a tensor
      // component, using the spectral differentiation matrix
      const auto reconstruct_dt_component =
          [&](double* dt_field_data, const double* modified_dn_dt_data) {
            for (size_t face_idx = 0; face_idx < num_face_pts; ++face_idx) {
              double sum_interior = 0.0;
              for (size_t j = 0; j < N_normal; ++j) {
                if (j == outermost_layer) {
                  continue;
                }
                sum_interior += diff_matrix(outermost_layer, j) *
                                dt_field_data[volume_index(face_idx, j)];
              }
              dt_field_data[volume_index(face_idx, outermost_layer)] =
                  (modified_dn_dt_data[face_idx] /
                       get(jacobian_factor)[face_idx] -
                   sum_interior) /
                  d_NN;
            }
          };

      // Reconstruct dt_conformal_metric
      {
        auto& dt_cm_vol =
            get<::Tags::dt<::Ccz4::Tags::ConformalMetric<DataVector, Dim>>>(
                *dt_vars);
        for (size_t ti = 0; ti < dt_cm_vol.size(); ++ti) {
          reconstruct_dt_component(
              dt_cm_vol[ti].data(),
              modified_dt_dn_conformal_metric[ti].data());
        }
      }

      // Reconstruct dt_conformal_factor
      {
        auto& dt_cf_vol =
            get<::Tags::dt<::Ccz4::Tags::ConformalFactor<DataVector>>>(
                *dt_vars);
        reconstruct_dt_component(
            get(dt_cf_vol).data(),
            get(modified_dt_dn_conformal_factor).data());
      }

      // Reconstruct dt_lapse
      {
        auto& dt_lapse_vol =
            get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(*dt_vars);
        reconstruct_dt_component(
            get(dt_lapse_vol).data(),
            get(modified_dt_dn_lapse).data());
      }

      // Reconstruct dt_shift
      {
        auto& dt_shift_vol =
            get<::Tags::dt<gr::Tags::Shift<DataVector, Dim>>>(*dt_vars);
        for (size_t i = 0; i < Dim; ++i) {
          reconstruct_dt_component(
              dt_shift_vol.get(i).data(),
              modified_dt_dn_shift.get(i).data());
        }
      }
    }  // end loop over td_dirichlet_directions
  }
};
}  // namespace Ccz4::fd
