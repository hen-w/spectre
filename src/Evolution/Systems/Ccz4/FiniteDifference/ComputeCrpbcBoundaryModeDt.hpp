// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/ConstraintsRadiationPreserving.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Characteristics.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/RadiationCharacteristics.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Ricci.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/Z4Constraint.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Computes the CRPBC boundary-mode time derivatives using
 * fully corrected dt values.
 *
 * \details This MutateApply action runs **after**
 * `ApplyBoundaryCorrectionsToTimeDerivative` in the DG step, so the dt
 * fields it reads already include DG boundary corrections. For each
 * external face with a `ConstraintsRadiationPreserving` boundary
 * condition, it:
 * 1. Computes face geometry (metrics, Christoffels, Ricci, Z4 constraint)
 * 2. Computes dt characteristic fields from corrected dt evolved vars
 * 3. Applies constraint-preserving equations (Eq1-Eq3)
 * 4. Applies radiation-preserving correction for UTensorMinus
 * 5. Scatters the 4 boundary-mode dt values back to the volume
 */
struct ComputeCrpbcBoundaryModeDt {
  static constexpr size_t Dim = 3;
  using System = ::Ccz4::fd::System;
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, System::variables_tag>;

  using return_tags = tmpl::list<dt_variables_tag>;

  using argument_tags =
      tmpl::list<domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
                 domain::Tags::ExternalBoundaryConditions<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>,
                 System::variables_tag, ::Ccz4::fd::Tags::EvolveLapseAndShift>;

  static void apply(
      const gsl::not_null<Variables<typename dt_variables_tag::tags_list>*>
          dt_vars,
      const Element<Dim>& element, const Mesh<Dim>& mesh,
      const std::vector<DirectionMap<
          Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>&
          all_boundary_conditions,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const Variables<typename System::variables_tag::tags_list>& evolved_vars,
      const bool evolve_lapse_and_shift) {
    // Early return if no external boundaries
    if (element.external_boundaries().empty()) {
      return;
    }

    // Detect CRPBC faces
    std::vector<Direction<Dim>> crpbc_directions;
    const auto& block_boundary_conditions =
        all_boundary_conditions.at(element.id().block_id());
    for (const auto& direction : element.external_boundaries()) {
      const auto* crpbc = dynamic_cast<
          const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving*>(
          block_boundary_conditions.at(direction).get());
      if (crpbc != nullptr) {
        crpbc_directions.push_back(direction);
      }
    }

    if (crpbc_directions.empty()) {
      return;
    }

    // All CRPBC faces share the same BC object; retrieve penalty multiplier
    // once.
    // *** THIS IS A BUG; CANNOT ASSUME SAME PENALTY MULTIPLIER ON ALL FACES.
    // *** FIX IN FUTURE. ***
    const auto* crpbc_ptr = dynamic_cast<
        const Ccz4::BoundaryConditions::ConstraintsRadiationPreserving*>(
        block_boundary_conditions.at(crpbc_directions[0]).get());
    const double penalty_multiplier = crpbc_ptr->penalty_multiplier();

    ASSERT(evolve_lapse_and_shift,
           "ConstraintsRadiationPreserving BC requires evolving lapse and "
           "shift.");

    // Extract evolved fields from the Variables container
    const auto& conformal_metric =
        get<::Ccz4::Tags::ConformalMetric<DataVector, Dim>>(evolved_vars);
    const auto& conformal_factor =
        get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars);
    const auto& a_tilde =
        get<::Ccz4::Tags::ATilde<DataVector, Dim>>(evolved_vars);
    const auto& trace_extrinsic_curvature =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars);
    const auto& theta = get<::Ccz4::Tags::Theta<DataVector>>(evolved_vars);
    const auto& gamma_hat =
        get<::Ccz4::Tags::GammaHat<DataVector, Dim>>(evolved_vars);
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(evolved_vars);
    const auto& shift = get<gr::Tags::Shift<DataVector, Dim>>(evolved_vars);
    const auto& field_a =
        get<::Ccz4::Tags::FieldA<DataVector, Dim>>(evolved_vars);
    const auto& field_d =
        get<::Ccz4::Tags::FieldD<DataVector, Dim>>(evolved_vars);
    const auto& field_p =
        get<::Ccz4::Tags::FieldP<DataVector, Dim>>(evolved_vars);

    // Extract dt fields from the dt_vars container
    const auto& dt_conformal_metric =
        get<::Tags::dt<::Ccz4::Tags::ConformalMetric<DataVector, Dim>>>(
            *dt_vars);
    const auto& dt_conformal_factor =
        get<::Tags::dt<::Ccz4::Tags::ConformalFactor<DataVector>>>(*dt_vars);
    const auto& dt_a_tilde =
        get<::Tags::dt<::Ccz4::Tags::ATilde<DataVector, Dim>>>(*dt_vars);
    const auto& dt_trace_extrinsic_curvature =
        get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
            *dt_vars);
    const auto& dt_theta =
        get<::Tags::dt<::Ccz4::Tags::Theta<DataVector>>>(*dt_vars);
    const auto& dt_gamma_hat =
        get<::Tags::dt<::Ccz4::Tags::GammaHat<DataVector, Dim>>>(*dt_vars);
    const auto& dt_lapse =
        get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(*dt_vars);
    const auto& dt_shift =
        get<::Tags::dt<gr::Tags::Shift<DataVector, Dim>>>(*dt_vars);
    const auto& dt_b =
        get<::Tags::dt<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>>>(
            *dt_vars);
    auto& dt_u_scalar3_minus =
        get<::Tags::dt<Tags::UScalar3Minus<DataVector>>>(*dt_vars);
    auto& dt_u_vector2_minus =
        get<::Tags::dt<Tags::UVector2Minus<DataVector, Dim, Frame::Inertial>>>(
            *dt_vars);
    auto& dt_u_scalar2_minus =
        get<::Tags::dt<Tags::UScalar2Minus<DataVector>>>(*dt_vars);
    auto& dt_u_tensor_minus =
        get<::Tags::dt<Tags::UTensorMinus<DataVector, Dim, Frame::Inertial>>>(
            *dt_vars);

    // Analytic dt auxiliary fields pre-computed in LdgTimeDerivative
    const auto& dt_field_a_vol =
        get<::Tags::dt<::Ccz4::Tags::FieldA<DataVector, Dim>>>(*dt_vars);
    const auto& dt_field_b_vol =
        get<::Tags::dt<::Ccz4::Tags::FieldB<DataVector, Dim>>>(*dt_vars);
    const auto& dt_field_d_vol =
        get<::Tags::dt<::Ccz4::Tags::FieldD<DataVector, Dim>>>(*dt_vars);
    const auto& dt_field_p_vol =
        get<::Tags::dt<::Ccz4::Tags::FieldP<DataVector, Dim>>>(*dt_vars);

    const size_t num_pts = mesh.number_of_grid_points();

    // Compute partial derivatives of evolved vars for aux field symmetrization
    const auto partial_derivs =
        partial_derivatives<typename System::variables_tag::tags_list>(
            evolved_vars, mesh, inv_jacobian);

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
        get<::Tags::deriv<::Ccz4::Tags::Theta<DataVector>, tmpl::size_t<Dim>,
                          Frame::Inertial>>(partial_derivs);
    const auto& d_gamma_hat =
        get<::Tags::deriv<::Ccz4::Tags::GammaHat<DataVector, Dim>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(partial_derivs);

    // Symmetrize derivatives of auxiliary fields
    tnsr::ii<DataVector, Dim> d_field_a(num_pts);
    ::tenex::evaluate<ti::i, ti::j>(
        make_not_null(&d_field_a),
        0.5 * (d_field_a_raw(ti::i, ti::j) + d_field_a_raw(ti::j, ti::i)));

    tnsr::iiJ<DataVector, Dim> d_field_b(num_pts);
    ::tenex::evaluate<ti::i, ti::j, ti::K>(
        make_not_null(&d_field_b), 0.5 * (d_field_b_raw(ti::i, ti::j, ti::K) +
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

    constexpr double f_param_crpbc = System::f;

    for (const auto& direction : crpbc_directions) {
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

      auto volume_index = [&](size_t face_idx, size_t layer) -> size_t {
        return (face_idx % inner_stride) + inner_stride * layer +
               inner_stride * N_normal * (face_idx / inner_stride);
      };

      // Grid spacing: Euclidean distance between outermost and
      // second-outermost grid points in the normal direction.
      const size_t second_outermost_layer =
          (direction.side() == Side::Upper) ? N_normal - 2 : 1;
      DataVector dist_sq(num_face_pts, 0.0);
      for (size_t d = 0; d < Dim; ++d) {
        for (size_t fp = 0; fp < num_face_pts; ++fp) {
          const double dx =
              inertial_coords.get(d)[volume_index(fp, outermost_layer)] -
              inertial_coords.get(d)[volume_index(fp, second_outermost_layer)];
          dist_sq[fp] += dx * dx;
        }
      }
      // penalty_strength = penalty_multiplier / h (face-point DataVector)
      Scalar<DataVector> penalty_strength(num_face_pts);
      get(penalty_strength) = penalty_multiplier / sqrt(dist_sq);

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

      // Slice evolved variables to face
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
      tnsr::ijj<DataVector, Dim> field_d_face;
      slice_tensor(field_d_face, field_d);
      tnsr::i<DataVector, Dim> field_p_face;
      slice_tensor(field_p_face, field_p);

      // Reconstruct first derivatives from aux fields
      tnsr::ijj<DataVector, Dim> d_conformal_metric_face(num_face_pts);
      ::tenex::evaluate<ti::i, ti::j, ti::k>(
          make_not_null(&d_conformal_metric_face),
          2.0 * field_d_face(ti::i, ti::j, ti::k));
      tnsr::i<DataVector, Dim> d_conformal_factor_face(num_face_pts);
      ::tenex::evaluate<ti::i>(make_not_null(&d_conformal_factor_face),
                               conformal_factor_face() * field_p_face(ti::i));

      // Slice derivative quantities
      tnsr::ijj<DataVector, Dim> d_a_tilde_face;
      slice_tensor(d_a_tilde_face, d_a_tilde);
      tnsr::i<DataVector, Dim> d_trace_K_face;
      slice_tensor(d_trace_K_face, d_trace_extrinsic_curvature);
      tnsr::i<DataVector, Dim> d_theta_face;
      slice_tensor(d_theta_face, d_theta);
      tnsr::iJ<DataVector, Dim> d_gamma_hat_face;
      slice_tensor(d_gamma_hat_face, d_gamma_hat);

      // Slice symmetrized aux derivatives
      tnsr::iijj<DataVector, Dim> d_field_d_face;
      slice_tensor(d_field_d_face, d_field_d);
      tnsr::ii<DataVector, Dim> d_field_p_face;
      slice_tensor(d_field_p_face, d_field_p);

      // Slice fields needed for dt_d_* recovery
      tnsr::i<DataVector, Dim> field_a_face;
      slice_tensor(field_a_face, field_a);
      Scalar<DataVector> outermost_dt_lapse_face;
      slice_scalar(outermost_dt_lapse_face, dt_lapse);
      Scalar<DataVector> outermost_dt_conformal_factor_face;
      slice_scalar(outermost_dt_conformal_factor_face, dt_conformal_factor);

      // Slice analytic dt auxiliary fields to face
      tnsr::i<DataVector, Dim> outermost_dt_field_a;
      slice_tensor(outermost_dt_field_a, dt_field_a_vol);
      tnsr::iJ<DataVector, Dim> outermost_dt_field_b;
      slice_tensor(outermost_dt_field_b, dt_field_b_vol);
      tnsr::ijj<DataVector, Dim> outermost_dt_field_d;
      slice_tensor(outermost_dt_field_d, dt_field_d_vol);
      tnsr::i<DataVector, Dim> outermost_dt_field_p;
      slice_tensor(outermost_dt_field_p, dt_field_p_vol);

      // Face temporaries
      Scalar<DataVector> conformal_factor_squared_face(num_face_pts);
      ::tenex::evaluate(make_not_null(&conformal_factor_squared_face),
                        conformal_factor_face() * conformal_factor_face());

      auto [det_conformal_metric_face, inv_conformal_metric_face] =
          determinant_and_inverse(conformal_metric_face);

      tnsr::II<DataVector, Dim> inv_spatial_metric_face(num_face_pts);
      ::tenex::evaluate<ti::I, ti::J>(
          make_not_null(&inv_spatial_metric_face),
          conformal_factor_squared_face() *
              inv_conformal_metric_face(ti::I, ti::J));

      tnsr::ii<DataVector, Dim> spatial_metric_face(num_face_pts);
      ::tenex::evaluate<ti::i, ti::j>(make_not_null(&spatial_metric_face),
                                      conformal_metric_face(ti::i, ti::j) /
                                          conformal_factor_squared_face());

      tnsr::iJJ<DataVector, Dim> field_d_up_face(num_face_pts);
      ::tenex::evaluate<ti::k, ti::I, ti::J>(
          make_not_null(&field_d_up_face),
          inv_conformal_metric_face(ti::I, ti::N) *
              inv_conformal_metric_face(ti::M, ti::J) *
              field_d_face(ti::k, ti::n, ti::m));

      auto conformal_christoffel_face =
          ::Ccz4::conformal_christoffel_second_kind(inv_conformal_metric_face,
                                                    field_d_face);

      auto d_conformal_christoffel_face =
          ::Ccz4::deriv_conformal_christoffel_second_kind(
              inv_conformal_metric_face, field_d_face, d_field_d_face,
              field_d_up_face);

      auto christoffel_face = ::Ccz4::christoffel_second_kind(
          conformal_metric_face, inv_conformal_metric_face, field_p_face,
          conformal_christoffel_face);

      tnsr::i<DataVector, Dim> contracted_christoffel_face(num_face_pts);
      ::tenex::evaluate<ti::l>(make_not_null(&contracted_christoffel_face),
                               christoffel_face(ti::M, ti::l, ti::m));

      tnsr::ij<DataVector, Dim> contracted_d_conformal_christoffel_diff_face(
          num_face_pts);
      ::tenex::evaluate<ti::i, ti::j>(
          make_not_null(&contracted_d_conformal_christoffel_diff_face),
          d_conformal_christoffel_face(ti::m, ti::M, ti::i, ti::j) -
              d_conformal_christoffel_face(ti::j, ti::M, ti::i, ti::m));

      tnsr::I<DataVector, Dim> contracted_field_d_up_face(num_face_pts);
      ::tenex::evaluate<ti::L>(make_not_null(&contracted_field_d_up_face),
                               field_d_up_face(ti::m, ti::M, ti::L));

      auto spatial_ricci_face = ::Ccz4::spatial_ricci_tensor(
          christoffel_face, contracted_christoffel_face,
          contracted_d_conformal_christoffel_diff_face, conformal_metric_face,
          inv_conformal_metric_face, field_d_face, field_d_up_face,
          contracted_field_d_up_face, field_p_face, d_field_p_face);

      auto contracted_conformal_christoffel_face =
          ::Ccz4::contracted_conformal_christoffel_second_kind(
              inv_conformal_metric_face, conformal_christoffel_face);

      tnsr::I<DataVector, Dim> gamma_hat_minus_contracted_cc_face(num_face_pts);
      ::tenex::evaluate<ti::I>(
          make_not_null(&gamma_hat_minus_contracted_cc_face),
          gamma_hat_face(ti::I) - contracted_conformal_christoffel_face(ti::I));

      auto spatial_z4_constraint_face = ::Ccz4::spatial_z4_constraint(
          conformal_metric_face, gamma_hat_minus_contracted_cc_face);

      auto d_contracted_cc_face =
          ::Ccz4::deriv_contracted_conformal_christoffel_second_kind(
              inv_conformal_metric_face, field_d_up_face,
              conformal_christoffel_face, d_conformal_christoffel_face);

      tnsr::iJ<DataVector, Dim> d_gamma_hat_minus_contracted_cc_face(
          num_face_pts);
      ::tenex::evaluate<ti::i, ti::J>(
          make_not_null(&d_gamma_hat_minus_contracted_cc_face),
          d_gamma_hat_face(ti::i, ti::J) - d_contracted_cc_face(ti::i, ti::J));

      tnsr::ij<DataVector, Dim> d_z4_face(num_face_pts);
      ::tenex::evaluate<ti::i, ti::j>(
          make_not_null(&d_z4_face),
          field_d_face(ti::i, ti::j, ti::l) *
                  gamma_hat_minus_contracted_cc_face(ti::L) +
              0.5 * conformal_metric_face(ti::j, ti::l) *
                  d_gamma_hat_minus_contracted_cc_face(ti::i, ti::L));

      // Unit normal
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
      ::tenex::evaluate<ti::I>(make_not_null(&unit_normal_vector),
                               inv_spatial_metric_face(ti::I, ti::J) *
                                   unnormalized_normal_one_form(ti::j));
      Scalar<DataVector> magnitude(num_face_pts);
      ::tenex::evaluate(make_not_null(&magnitude),
                        sqrt(unit_normal_vector(ti::I) *
                             unnormalized_normal_one_form(ti::i)));
      ::tenex::evaluate<ti::I>(make_not_null(&unit_normal_vector),
                               unit_normal_vector(ti::I) / magnitude());
      const tnsr::i<DataVector, Dim> unit_normal_one_form =
          raise_or_lower_index(unit_normal_vector, spatial_metric_face);

      // dt of spatial derivatives from analytic dt_field_a/b/d/p
      // (pre-computed in LdgTimeDerivative, avoiding spectral differentiation).
      // Recovery: ∂_k ∂_t f = f * ∂_t(field_f) + ∂_t f * field_f
      tnsr::ijj<DataVector, Dim> outermost_dt_d_conformal_metric(num_face_pts);
      ::tenex::evaluate<ti::k, ti::i, ti::j>(
          make_not_null(&outermost_dt_d_conformal_metric),
          2.0 * outermost_dt_field_d(ti::k, ti::i, ti::j));

      tnsr::i<DataVector, Dim> outermost_dt_d_lapse(num_face_pts);
      ::tenex::evaluate<ti::k>(
          make_not_null(&outermost_dt_d_lapse),
          lapse_face() * outermost_dt_field_a(ti::k) +
              outermost_dt_lapse_face() * field_a_face(ti::k));

      tnsr::i<DataVector, Dim> outermost_dt_d_conformal_factor(num_face_pts);
      ::tenex::evaluate<ti::k>(
          make_not_null(&outermost_dt_d_conformal_factor),
          conformal_factor_face() * outermost_dt_field_p(ti::k) +
              outermost_dt_conformal_factor_face() * field_p_face(ti::k));

      tnsr::iJ<DataVector, Dim> outermost_dt_d_shift(num_face_pts);
      ::tenex::evaluate<ti::k, ti::I>(make_not_null(&outermost_dt_d_shift),
                                      outermost_dt_field_b(ti::k, ti::I));

      // Slice dt of evolved variables
      Scalar<DataVector> outermost_dt_trace_K;
      slice_scalar(outermost_dt_trace_K, dt_trace_extrinsic_curvature);
      tnsr::ii<DataVector, Dim> outermost_dt_a_tilde;
      slice_tensor(outermost_dt_a_tilde, dt_a_tilde);
      Scalar<DataVector> outermost_dt_theta;
      slice_scalar(outermost_dt_theta, dt_theta);
      tnsr::I<DataVector, Dim> outermost_dt_gamma_hat;
      slice_tensor(outermost_dt_gamma_hat, dt_gamma_hat);
      tnsr::I<DataVector, Dim> outermost_dt_b;
      slice_tensor(outermost_dt_b, dt_b);

      // dt characteristic fields
      auto dt_char_fields = dt_characteristic_fields(
          unit_normal_one_form, conformal_metric_face, conformal_factor_face,
          lapse_face, shift_face, outermost_dt_trace_K, outermost_dt_a_tilde,
          outermost_dt_theta, outermost_dt_gamma_hat, outermost_dt_b,
          outermost_dt_d_conformal_metric, outermost_dt_d_conformal_factor,
          outermost_dt_d_lapse, outermost_dt_d_shift, f_param_crpbc);

      // Characteristic speeds
      const auto char_speeds =
          characteristic_speeds(lapse_face, shift_face, conformal_factor_face,
                                f_param_crpbc, unit_normal_one_form);

      // Speed validation
      for (size_t i = 0; i < num_face_pts; ++i) {
        if (char_speeds[0][i] < 0.0 or char_speeds[1][i] >= 0.0 or
            char_speeds[5][i] < 0.0 or char_speeds[6][i] >= 0.0 or
            char_speeds[12][i] < 0.0 or char_speeds[13][i] >= 0.0 or
            char_speeds[14][i] < 0.0 or char_speeds[15][i] >= 0.0) {
          ERROR(
              "CRPBC requires asymptotically Minkowskian coordinates but "
              "the characteristic speeds at an outer boundary point are "
              "char_speeds[0] = "
              << char_speeds[0][i] << ", char_speeds[1] = " << char_speeds[1][i]
              << ", char_speeds[5] = " << char_speeds[5][i]
              << ", char_speeds[6] = " << char_speeds[6][i]
              << ", char_speeds[12] = " << char_speeds[12][i]
              << ", char_speeds[13] = " << char_speeds[13][i]
              << ", char_speeds[14] = " << char_speeds[14][i]
              << ", char_speeds[15] = " << char_speeds[15][i]);
        }
      }

      // Constraint-preserving equations
      const auto q_mixed = gr::transverse_projection_operator(
          unit_normal_vector, unit_normal_one_form);

      Scalar<DataVector> dn_theta(num_face_pts);
      ::tenex::evaluate(make_not_null(&dn_theta),
                        unit_normal_vector(ti::I) * d_theta_face(ti::i));

      Scalar<DataVector> beta_dot_d_theta(num_face_pts);
      ::tenex::evaluate(make_not_null(&beta_dot_d_theta),
                        shift_face(ti::K) * d_theta_face(ti::k));

      const auto& dt_u_scalar3_plus =
          get<::Tags::dt<Tags::UScalar3Plus<DataVector>>>(dt_char_fields);
      auto& dt_u_scalar3_minus_field2 =
          get<::Tags::dt<Tags::UScalar3Minus<DataVector>>>(dt_char_fields);

      // Eq1: dt UScalar3Minus
      ::tenex::evaluate(
          make_not_null(&dt_u_scalar3_minus_field2),
          dt_u_scalar3_plus() +
              4.0 / conformal_factor_squared_face() *
                  (-lapse_face() * dn_theta() + beta_dot_d_theta() -
                   penalty_strength() * theta_face()));

      // Eq2: dt UVector2Minus
      const auto& dt_u_vector2_plus =
          get<::Tags::dt<Tags::UVector2Plus<DataVector, Dim, Frame::Inertial>>>(
              dt_char_fields);
      auto& dt_u_vector2_minus_field = get<
          ::Tags::dt<Tags::UVector2Minus<DataVector, Dim, Frame::Inertial>>>(
          dt_char_fields);

      tnsr::i<DataVector, Dim> eq2_term2(num_face_pts);
      ::tenex::evaluate<ti::i>(
          make_not_null(&eq2_term2),
          q_mixed(ti::J, ti::i) * inv_conformal_metric_face(ti::K, ti::L) *
              q_mixed(ti::M, ti::l) *
              outermost_dt_d_conformal_metric(ti::m, ti::j, ti::k));

      tnsr::i<DataVector, Dim> n_dot_dZ(num_face_pts);
      ::tenex::evaluate<ti::m>(
          make_not_null(&n_dot_dZ),
          unit_normal_vector(ti::I) * d_z4_face(ti::i, ti::m));

      tnsr::i<DataVector, Dim> beta_dot_dZ(num_face_pts);
      ::tenex::evaluate<ti::m>(make_not_null(&beta_dot_dZ),
                               shift_face(ti::K) * d_z4_face(ti::k, ti::m));

      tnsr::i<DataVector, Dim> eq2_term3(num_face_pts);
      ::tenex::evaluate<ti::i>(
          make_not_null(&eq2_term3),
          q_mixed(ti::M, ti::i) *
              (-lapse_face() * n_dot_dZ(ti::m) + beta_dot_dZ(ti::m) -
               penalty_strength() * spatial_z4_constraint_face(ti::m)));

      ::tenex::evaluate<ti::i>(
          make_not_null(&dt_u_vector2_minus_field),
          -dt_u_vector2_plus(ti::i) +
              2.0 / conformal_factor_squared_face() * eq2_term2(ti::i) +
              4.0 / conformal_factor_squared_face() * eq2_term3(ti::i));

      // Eq3: dt UScalar2Minus
      const auto& dt_u_scalar2_plus =
          get<::Tags::dt<Tags::UScalar2Plus<DataVector>>>(dt_char_fields);
      auto& dt_u_scalar2_minus_field =
          get<::Tags::dt<Tags::UScalar2Minus<DataVector>>>(dt_char_fields);

      Scalar<DataVector> phi4(num_face_pts);
      ::tenex::evaluate(
          make_not_null(&phi4),
          conformal_factor_squared_face() * conformal_factor_squared_face());

      Scalar<DataVector> eq3_term_B(num_face_pts);
      ::tenex::evaluate(
          make_not_null(&eq3_term_B),
          unit_normal_one_form(ti::i) * q_mixed(ti::M, ti::l) *
              inv_conformal_metric_face(ti::I, ti::J) *
              inv_conformal_metric_face(ti::K, ti::L) *
              outermost_dt_d_conformal_metric(ti::m, ti::j, ti::k));

      Scalar<DataVector> dn_Zn(num_face_pts);
      ::tenex::evaluate(make_not_null(&dn_Zn),
                        unit_normal_vector(ti::I) * n_dot_dZ(ti::i));
      Scalar<DataVector> beta_dot_d_Zn(num_face_pts);
      ::tenex::evaluate(make_not_null(&beta_dot_d_Zn),
                        unit_normal_vector(ti::I) * beta_dot_dZ(ti::i));

      ::tenex::evaluate(
          make_not_null(&dt_u_scalar2_minus_field),
          dt_u_scalar2_plus() -
              0.5 * phi4() *
                  (dt_u_scalar3_plus() + dt_u_scalar3_minus_field2()) +
              phi4() * eq3_term_B() +
              2.0 * conformal_factor_squared_face() *
                  (-lapse_face() * dn_Zn() + beta_dot_d_Zn() -
                   penalty_strength() * unit_normal_vector(ti::M) *
                       spatial_z4_constraint_face(ti::m)));

      // Radiation-preserving correction for UTensorMinus
      Scalar<DataVector> half_cfs_face(num_face_pts);
      ::tenex::evaluate(make_not_null(&half_cfs_face),
                        0.5 * conformal_factor_squared_face());

      const auto radiation_char_fields = radiation_characteristic_fields(
          conformal_factor_face, conformal_factor_squared_face,
          conformal_metric_face, spatial_metric_face, inv_spatial_metric_face,
          trace_K_face, a_tilde_face, d_conformal_factor_face, d_trace_K_face,
          d_conformal_metric_face, d_a_tilde_face, spatial_ricci_face,
          christoffel_face, unit_normal_one_form);
      const auto& c_tensor_minus =
          get<Tags::CTensorMinus<DataVector, Dim, Frame::Inertial>>(
              radiation_char_fields);
      auto& dt_u_tensor_minus_field =
          get<::Tags::dt<Tags::UTensorMinus<DataVector, Dim, Frame::Inertial>>>(
              dt_char_fields);

      ::tenex::update<ti::i, ti::j>(
          make_not_null(&dt_u_tensor_minus_field),
          dt_u_tensor_minus_field(ti::i, ti::j) -
              (lapse_face() + shift_face(ti::K) * unit_normal_one_form(ti::k)) *
                  conformal_factor_squared_face() *
                  c_tensor_minus(ti::i, ti::j));

      // Scatter: store dt of the 4 boundary mode fields back to volume
      // at face node positions
      const auto& face_dt_u_scalar3_minus =
          get<::Tags::dt<Tags::UScalar3Minus<DataVector>>>(dt_char_fields);
      const auto& face_dt_u_vector2_minus = get<
          ::Tags::dt<Tags::UVector2Minus<DataVector, Dim, Frame::Inertial>>>(
          dt_char_fields);
      const auto& face_dt_u_scalar2_minus =
          get<::Tags::dt<Tags::UScalar2Minus<DataVector>>>(dt_char_fields);
      const auto& face_dt_u_tensor_minus =
          get<::Tags::dt<Tags::UTensorMinus<DataVector, Dim, Frame::Inertial>>>(
              dt_char_fields);

      for (size_t fp = 0; fp < num_face_pts; ++fp) {
        const size_t vol_idx = volume_index(fp, outermost_layer);
        get(dt_u_scalar3_minus)[vol_idx] = get(face_dt_u_scalar3_minus)[fp];
        get(dt_u_scalar2_minus)[vol_idx] = get(face_dt_u_scalar2_minus)[fp];
        for (size_t i = 0; i < Dim; ++i) {
          dt_u_vector2_minus.get(i)[vol_idx] =
              face_dt_u_vector2_minus.get(i)[fp];
        }
        for (size_t ti = 0; ti < dt_u_tensor_minus.size(); ++ti) {
          dt_u_tensor_minus[ti][vol_idx] = face_dt_u_tensor_minus[ti][fp];
        }
      }
    }  // end loop over crpbc_directions
  }
};
}  // namespace Ccz4::fd
