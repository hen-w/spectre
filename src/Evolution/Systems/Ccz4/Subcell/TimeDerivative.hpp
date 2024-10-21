// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/AsAccess.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TimeDerivative.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::Subcell {
template <size_t Dim, class DbTagsList>
static void apply(const gsl::not_null<db::DataBox<DbTagsList>*> box) {
  // we assume the evolved variables, their time derivatives, and ln_lapse,
  // ln_conformal_factor are in the data box.
  constexpr double one_half = 1.0 / 2.0;

  // compute the first derivatives of the evolved variables
  using evolved_vars_tag = typename Ccz4::System::variables_tag;
  using gradients_tags = typename Ccz4::System::gradients_tags;

  const auto& evolved_vars = db::get<evolved_vars_tag>(*box);
  const Mesh<3>& subcell_mesh =
      db::get<evolution::dg::subcell::Tags::Mesh<3>>(*box);
  const size_t num_pts = subcell_mesh.number_of_grid_points();
  Variables<db::wrap_tags_in<::Tags::deriv, gradients_tags, tmpl::size_t<3>,
                             Frame::Inertial>>
      cell_centered_Ccz4_derivs{num_pts};
  const fd::Reconstructor& recons = db::get<fd::Tags::Reconstructor>(*box);
  const auto& cell_centered_logical_to_inertial_inv_jacobian = db::get<
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<3>>(
      *box);

  Ccz4::fd::spacetime_derivatives(
      make_not_null(&cell_centered_Ccz4_derivs), evolved_vars,
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
          *box),
      recons.ghost_zone_size() * 2, subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian);

  // calculate the four auxiliary fields in eq. 6
  // auxiliary variables NOT evolved in SO-CCZ4
  const gsl::not_null<tnsr::i<DataVector, Dim>*> field_a;
  ::tenex::evaluate<ti::i>(
      field_a,
      get<::Tags::deriv<Tags::LogLapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>>(cell_centered_Ccz4_derivs)(ti::i));

  const gsl::not_null<tnsr::iJ<DataVector, Dim>*> field_b;
  ::tenex::evaluate<ti::k, ti::I>(
      field_b, get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>,
                                 tmpl::size_t<3>, Frame::Inertial>>(
                   cell_centered_Ccz4_derivs)(ti::k, ti::I));

  const gsl::not_null<tnsr::ijj<DataVector, Dim>*> field_d;
  ::tenex::evaluate<ti::k, ti::i, ti::j>(
      field_d,
      one_half * get<::Tags::deriv<Tags::ConformalMetric<DataVector, 3>,
                                   tmpl::size_t<3>, Frame::Inertial>>(
                     cell_centered_Ccz4_derivs)(ti::k, ti::i, ti::j));

  const gsl::not_null<tnsr::i<DataVector, Dim>*> field_p;
  ::tenex::evaluate<ti::i>(
      field_p,
      get<::Tags::deriv<Tags::LogConformalFactor<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>>(cell_centered_Ccz4_derivs)(ti::i));

  // compute second derivatives of the evolved variables
  Variables<db::wrap_tags_in<::Tags::deriv,
                             db::wrap_tags_in<::Tags::deriv, gradients_tags,
                                              tmpl::size_t<3>, Frame::Inertial>,
                             tmpl::size_t<3>, Frame::Inertial>>
      cell_centered_Ccz4_second_derivs{num_pts};

  Ccz4::fd::second_derivatives(
      make_not_null(&cell_centered_Ccz4_second_derivs), evolved_vars,
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
          *box),
      4 /*deriv order for special second deriv stencil*/, subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian);

  // compute spatial derivative of the four auxiliary fields
  const gsl::not_null<tnsr::ij<DataVector, Dim>*> d_field_a;
  ::tenex::evaluate<ti::i, ti::j>(
      d_field_a,
      get<::Tags::deriv<::Tags::deriv<Tags::LogLapse<DataVector>,
                                      tmpl::size_t<3>, Frame::Inertial>,
                        tmpl::size_t<3>, Frame::Inertial>>(
          cell_centered_Ccz4_second_derivs)(ti::i, ti::j));

  const gsl::not_null<tnsr::ijK<DataVector, Dim>*> d_field_b;
  ::tenex::evaluate<ti::j, ti::k, ti::I>(
      d_field_b,
      get<::Tags::deriv<::Tags::deriv<gr::Tags::Shift<DataVector, 3>,
                                      tmpl::size_t<3>, Frame::Inertial>,
                        tmpl::size_t<3>, Frame::Inertial>>(
          cell_centered_Ccz4_second_derivs)(ti::j, ti::k, ti::I));

  const gsl::not_null<tnsr::ijkk<DataVector, Dim>*> d_field_d;
  ::tenex::evaluate<ti::l, ti::k, ti::i, ti::j>(
      d_field_d,
      get<::Tags::deriv<::Tags::deriv<Tags::ConformalMetric<DataVector, 3>,
                                      tmpl::size_t<3>, Frame::Inertial>,
                        tmpl::size_t<3>, Frame::Inertial>>(
          cell_centered_Ccz4_second_derivs)(ti::l, ti::k, ti::i, ti::j));

  const gsl::not_null<tnsr::ij<DataVector, Dim>*> d_field_p;
  ::tenex::evaluate<ti::j, ti::i>(
      d_field_p,
      get<::Tags::deriv<::Tags::deriv<Tags::LogConformalFactor<DataVector>,
                                      tmpl::size_t<3>, Frame::Inertial>,
                        tmpl::size_t<3>, Frame::Inertial>>(
          cell_centered_Ccz4_second_derivs)(ti::j, ti::i));

  // intialize containers to be supplied in the DG TimeDerivative.cpp apply
  // function quantities we need for computing eq 4, 13 - 27
  const Scalar<DataVector> conformal_factor_squared;
  const Scalar<DataVector> det_conformal_spatial_metric;
  const tnsr::II<DataVector, Dim> inv_conformal_spatial_metric;
  const tnsr::II<DataVector, Dim> inv_spatial_metric;
  const Scalar<DataVector> slicing_condition;    // g(\alpha)
  const Scalar<DataVector> d_slicing_condition;  // g'(\alpha)
  const tnsr::II<DataVector, Dim> inv_a_tilde;
  // temporary expressions
  const tnsr::ij<DataVector, Dim> a_tilde_times_field_b;
  const tnsr::ii<DataVector, Dim>
      a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde;
  const Scalar<DataVector> contracted_field_b;
  const tnsr::ijK<DataVector, Dim> symmetrized_d_field_b;
  const tnsr::i<DataVector, Dim> contracted_symmetrized_d_field_b;
  const tnsr::ijk<DataVector, Dim> field_b_times_field_d;
  const tnsr::i<DataVector, Dim> field_d_up_times_a_tilde;
  const tnsr::I<DataVector, Dim> contracted_field_d_up;    // temp for eq 18 -20
  const Scalar<DataVector> half_conformal_factor_squared;  // temp for eq 25
  const tnsr::ij<DataVector, Dim> conformal_metric_times_field_b;
  const tnsr::ijk<DataVector, Dim> conformal_metric_times_symmetrized_d_field_b;
  const tnsr::ii<DataVector, Dim> conformal_metric_times_trace_a_tilde;
  const tnsr::i<DataVector, Dim> inv_conformal_metric_times_d_a_tilde;
  const tnsr::I<DataVector, Dim>
      gamma_hat_minus_contracted_conformal_christoffel;
  const tnsr::iJ<DataVector, Dim>
      d_gamma_hat_minus_contracted_conformal_christoffel;
  const tnsr::i<DataVector, Dim>
      contracted_christoffel_second_kind;  // temp for eq 18 -20
  const tnsr::ij<DataVector, Dim>
      contracted_d_conformal_christoffel_difference;  // temp for eq 18 -20
  const Scalar<DataVector> k_minus_2_theta_c;
  const Scalar<DataVector> k_minus_k0_minus_2_theta_c;
  const Scalar<DataVector> ln_lapse;
  const tnsr::ii<DataVector, Dim> lapse_times_a_tilde;
  const tnsr::ijj<DataVector, Dim> lapse_times_d_a_tilde;
  const tnsr::i<DataVector, Dim> lapse_times_field_a;
  const tnsr::ii<DataVector, Dim> lapse_times_conformal_spatial_metric;
  const Scalar<DataVector> lapse_times_slicing_condition;
  const Scalar<DataVector>
      lapse_times_ricci_scalar_plus_divergence_z4_constraint;
  const tnsr::I<DataVector, Dim> shift_times_deriv_gamma_hat;
  const tnsr::ii<DataVector, Dim> inv_tau_times_conformal_metric;
  // expressions and identities needed for evolution equations: eq 13 - 27
  const Scalar<DataVector> trace_a_tilde;                              // eq 13
  const tnsr::iJJ<DataVector, Dim> field_d_up;                         // eq 14
  const tnsr::Ijj<DataVector, Dim> conformal_christoffel_second_kind;  // eq 15
  const tnsr::iJkk<DataVector, Dim>
      d_conformal_christoffel_second_kind;                   // eq 16
  const tnsr::Ijj<DataVector, Dim> christoffel_second_kind;  // eq 17
  const tnsr::ii<DataVector, Dim> spatial_ricci_tensor;      // eq 18 - 20
  const tnsr::ij<DataVector, Dim> grad_grad_lapse;           // eq 21
  const Scalar<DataVector> divergence_lapse;                 // eq 22
  const tnsr::I<DataVector, Dim>
      contracted_conformal_christoffel_second_kind;  // eq 23
  const tnsr::iJ<DataVector, Dim>
      d_contracted_conformal_christoffel_second_kind;                   // eq 24
  const tnsr::i<DataVector, Dim> spatial_z4_constraint;                 // eq 25
  const tnsr::I<DataVector, Dim> upper_spatial_z4_constraint;           // eq 25
  const tnsr::ij<DataVector, Dim> grad_spatial_z4_constraint;           // eq 26
  const Scalar<DataVector> ricci_scalar_plus_divergence_z4_constraint;  // eq 27
  // free params
  const double c = 1.0;  // set to 1.0 to recover the SO-CCZ4 system in Dumbser
  const double cleaning_speed = 1.0;  // e in the paper
  const Scalar<DataVector> eta;
  const double f = get<Tags::GammaDriverParam>(*box);
  const Scalar<DataVector> k_0;
  const double kappa_1 = get<Tags::Kappa1>(*box);
  const double kappa_2 = get<Tags::Kappa2>(*box);
  ;
  const double kappa_3 = get<Tags::Kappa3>(*box);
  ;
  const double one_over_relaxation_time =
      1.0;  // \tau^{-1}; this shouldn't matter in SO-CCZ4
  const EvolveShift evolve_shift =
      EvolveShift::True;  // always evolve shift in SO-CCZ4
  const SlicingConditionType slicing_condition_type =
      SlicingConditionType::Log;  // always use 1+log slicing

  // feed into the dg TimeDerivative.cpp apply function to calculate the time
  // derivatives of the evolved vairables
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, evolved_vars_tag>;
  db::mutate<dt_variables_tag>(
      [&]() {
        apply(
            // LHS time derivatives of evolved variables: eq 4a - 4i
            get<::Tags::dt<Tags::ConformalMetric<DataVector, 3>>>(
                *box),                                                 // eq 4a
            get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(*box),        // eq 4g
            get<::Tags::dt<gr::Tags::Shift<DataVector, 3>>>(*box),     // eq 4h
            get<::Tags::dt<Tags::ConformalFactor<DataVector>>>(*box),  // eq 4c
            get<::Tags::dt<Tags::ATilde<DataVector, 3>>>(*box),        // eq 4b
            get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
                *box),                                             // eq 4d
            get<::Tags::dt<Tags::Theta<DataVector>>>(*box),        // eq 4e
            get<::Tags::dt<Tags::GammaHat<DataVector, 3>>>(*box),  // eq 4f
            get<::Tags::dt<Tags::Fieldb<DataVector, 3>>>(*box),    // eq 4i

            // quantities we need for computing eq 4, 13 - 27
            make_not_null(&conformal_factor_squared),
            make_not_null(&det_conformal_spatial_metric),
            make_not_null(&inv_conformal_spatial_metric),
            make_not_null(&inv_spatial_metric),
            make_not_null(&slicing_condition),    // g(\alpha)
            make_not_null(&d_slicing_condition),  // g'(\alpha)
            make_not_null(&inv_a_tilde),
            // temporary expressions
            make_not_null(&a_tilde_times_field_b),
            make_not_null(
                &a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde),
            make_not_null(&contracted_field_b),
            make_not_null(&symmetrized_d_field_b),
            make_not_null(&contracted_symmetrized_d_field_b),
            make_not_null(&field_b_times_field_d),
            make_not_null(&field_d_up_times_a_tilde),
            make_not_null(&contracted_field_d_up),  // temp for eq 18 -20
            make_not_null(&half_conformal_factor_squared),  // temp for eq 25
            make_not_null(&conformal_metric_times_field_b),
            make_not_null(&conformal_metric_times_symmetrized_d_field_b),
            make_not_null(&conformal_metric_times_trace_a_tilde),
            make_not_null(&inv_conformal_metric_times_d_a_tilde),
            make_not_null(&gamma_hat_minus_contracted_conformal_christoffel),
            make_not_null(&d_gamma_hat_minus_contracted_conformal_christoffel),
            make_not_null(
                &contracted_christoffel_second_kind),  // temp for eq 18 -20
            make_not_null(
                &contracted_d_conformal_christoffel_difference),  // temp for eq
                                                                  // 18 -20
            make_not_null(&k_minus_2_theta_c),
            make_not_null(&k_minus_k0_minus_2_theta_c),
            make_not_null(&ln_lapse), make_not_null(&lapse_times_a_tilde),
            make_not_null(&lapse_times_d_a_tilde),
            make_not_null(&lapse_times_field_a),
            make_not_null(&lapse_times_conformal_spatial_metric),
            make_not_null(&lapse_times_slicing_condition),
            make_not_null(
                &lapse_times_ricci_scalar_plus_divergence_z4_constraint),
            make_not_null(&shift_times_deriv_gamma_hat),
            make_not_null(&inv_tau_times_conformal_metric),
            // expressions and identities needed for evolution equations: eq 13
            // - 27
            make_not_null(&trace_a_tilde),                        // eq 13
            make_not_null(&field_d_up),                           // eq 14
            make_not_null(&conformal_christoffel_second_kind),    // eq 15
            make_not_null(&d_conformal_christoffel_second_kind),  // eq 16
            make_not_null(&christoffel_second_kind),              // eq 17
            make_not_null(&spatial_ricci_tensor),                 // eq 18 - 20
            make_not_null(&grad_grad_lapse),                      // eq 21
            make_not_null(&divergence_lapse),                     // eq 22
            make_not_null(
                &contracted_conformal_christoffel_second_kind),  // eq 23
            make_not_null(
                &d_contracted_conformal_christoffel_second_kind),  // eq 24
            make_not_null(&spatial_z4_constraint),                 // eq 25
            make_not_null(&upper_spatial_z4_constraint),           // eq 25
            make_not_null(&grad_spatial_z4_constraint),            // eq 26
            make_not_null(
                &ricci_scalar_plus_divergence_z4_constraint),  // eq 27
            field_a,  // auxiliary variables NOT evolved in SO-CCZ4
            field_b, field_d, field_p,
            d_field_a,  // spatial derivative of auxiliary variables
            d_field_b, d_field_d, d_field_p,
            // free params
            c, cleaning_speed,  // e in the paper
            eta, f, k_0, kappa_1, kappa_2, kappa_3,
            one_over_relaxation_time,  // \tau^{-1}
            evolve_shift, slicing_condition_type,
            // evolved variables
            get<Tags::ConformalMetric<DataVector, 3>>(*box),
            get<gr::Tags::Lapse<DataVector>>(*box),
            get<gr::Tags::Shift<DataVector, 3>>(*box),
            get<Tags::ConformalFactor<DataVector>>(*box),
            get<Tags::ATilde<DataVector, 3>>(*box),
            get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(*box),
            get<Tags::Theta<DataVector>>(*box),
            get<Tags::GammaHat<DataVector, 3>>(*box),
            get<Tags::Fieldb<DataVector, 3>>(*box),
            // spatial derivatives of evolved variables
            get<::Tags::deriv<Tags::ATilde<DataVector, 3>, tmpl::size_t<3>,
                              Frame::Inertial>>,
            get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                              tmpl::size_t<3>, Frame::Inertial>>,
            get<::Tags::deriv<Tags::Theta<DataVector>, tmpl::size_t<3>,
                              Frame::Inertial>>,
            get<::Tags::deriv<Tags::GammaHat<DataVector, 3>, tmpl::size_t<3>,
                              Frame::Inertial>>,
            get<::Tags::deriv<Tags::Fieldb<DataVector, 3>, tmpl::size_t<3>,
                              Frame::Inertial>>);
      },
      box);
}
}  // namespace Ccz4::Subcell
