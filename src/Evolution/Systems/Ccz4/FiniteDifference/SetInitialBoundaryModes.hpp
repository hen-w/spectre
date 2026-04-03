// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/ConstraintsRadiationPreserving.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Characteristics.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Initialize boundary mode evolved variables from the initial data
 * at CRPBC external boundary faces.
 *
 * \details The boundary mode variables (UScalar3Minus, UVector2Minus,
 * UScalar2Minus, UTensorMinus) are initialized to the incoming characteristic
 * field values computed from the initial data. At interior points and
 * non-CRPBC faces, the boundary modes remain zero.
 */
struct SetInitialBoundaryModes : tt::ConformsTo<db::protocols::Mutator> {
  static constexpr size_t Dim = 3;
  using System = ::Ccz4::fd::System;

  // Boundary mode tags that are mutated
  using return_tags =
      tmpl::list<Tags::UScalar3Minus<DataVector>,
                 Tags::UVector2Minus<DataVector, Dim, Frame::Inertial>,
                 Tags::UScalar2Minus<DataVector>,
                 Tags::UTensorMinus<DataVector, Dim, Frame::Inertial>>;

  // Read-only evolved/auxiliary variable tags and domain tags
  using argument_tags = tmpl::list<
      ::Ccz4::Tags::ConformalMetric<DataVector, Dim>,
      ::Ccz4::Tags::ConformalFactor<DataVector>,
      ::Ccz4::Tags::ATilde<DataVector, Dim>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Ccz4::Tags::Theta<DataVector>, ::Ccz4::Tags::GammaHat<DataVector, Dim>,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, Dim>,
      ::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>,
      ::Ccz4::Tags::FieldA<DataVector, Dim>,
      ::Ccz4::Tags::FieldB<DataVector, Dim>,
      ::Ccz4::Tags::FieldD<DataVector, Dim>,
      ::Ccz4::Tags::FieldP<DataVector, Dim>, domain::Tags::Element<Dim>,
      domain::Tags::Mesh<Dim>, domain::Tags::ExternalBoundaryConditions<Dim>,
      domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      ::Ccz4::fd::Tags::EvolveLapseAndShift>;

  static void apply(
      const gsl::not_null<Scalar<DataVector>*> vol_u_scalar3_minus,
      const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          vol_u_vector2_minus,
      const gsl::not_null<Scalar<DataVector>*> vol_u_scalar2_minus,
      const gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          vol_u_tensor_minus,
      const tnsr::ii<DataVector, Dim>& conformal_metric,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::ii<DataVector, Dim>& a_tilde,
      const Scalar<DataVector>& trace_k, const Scalar<DataVector>& theta,
      const tnsr::I<DataVector, Dim>& gamma_hat,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
      const tnsr::I<DataVector, Dim>& b_aux,
      const tnsr::i<DataVector, Dim>& field_a,
      const tnsr::iJ<DataVector, Dim>& field_b,
      const tnsr::ijj<DataVector, Dim>& field_d,
      const tnsr::i<DataVector, Dim>& field_p, const Element<Dim>& element,
      const Mesh<Dim>& mesh,
      const std::vector<DirectionMap<
          Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>&
          all_boundary_conditions,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const bool evolve_lapse_and_shift) {
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

    ASSERT(evolve_lapse_and_shift,
           "ConstraintsRadiationPreserving BC requires evolving lapse and "
           "shift.");

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

      // Slice evolved variables and auxiliary fields to face
      tnsr::ii<DataVector, Dim> conformal_metric_face;
      slice_tensor(conformal_metric_face, conformal_metric);
      Scalar<DataVector> conformal_factor_face;
      slice_scalar(conformal_factor_face, conformal_factor);
      tnsr::ii<DataVector, Dim> a_tilde_face;
      slice_tensor(a_tilde_face, a_tilde);
      Scalar<DataVector> trace_k_face;
      slice_scalar(trace_k_face, trace_k);
      Scalar<DataVector> theta_face;
      slice_scalar(theta_face, theta);
      tnsr::I<DataVector, Dim> gamma_hat_face;
      slice_tensor(gamma_hat_face, gamma_hat);
      Scalar<DataVector> lapse_face;
      slice_scalar(lapse_face, lapse);
      tnsr::I<DataVector, Dim> shift_face;
      slice_tensor(shift_face, shift);
      tnsr::I<DataVector, Dim> b_aux_face;
      slice_tensor(b_aux_face, b_aux);
      tnsr::i<DataVector, Dim> field_a_face;
      slice_tensor(field_a_face, field_a);
      tnsr::iJ<DataVector, Dim> field_b_face;
      slice_tensor(field_b_face, field_b);
      tnsr::ijj<DataVector, Dim> field_d_face;
      slice_tensor(field_d_face, field_d);
      tnsr::i<DataVector, Dim> field_p_face;
      slice_tensor(field_p_face, field_p);

      // Reconstruct spatial derivatives from auxiliary fields (tenex)
      tnsr::ijj<DataVector, Dim> d_conformal_metric(num_face_pts);
      ::tenex::evaluate<ti::k, ti::i, ti::j>(
          make_not_null(&d_conformal_metric),
          2.0 * field_d_face(ti::k, ti::i, ti::j));

      tnsr::i<DataVector, Dim> d_conformal_factor(num_face_pts);
      ::tenex::evaluate<ti::i>(make_not_null(&d_conformal_factor),
                               conformal_factor_face() * field_p_face(ti::i));

      tnsr::i<DataVector, Dim> d_lapse(num_face_pts);
      ::tenex::evaluate<ti::i>(make_not_null(&d_lapse),
                               lapse_face() * field_a_face(ti::i));

      tnsr::iJ<DataVector, Dim> d_shift(num_face_pts);
      ::tenex::evaluate<ti::i, ti::J>(make_not_null(&d_shift),
                                      field_b_face(ti::i, ti::J));

      // Compute physical inverse spatial metric for normal normalization
      Scalar<DataVector> conformal_factor_squared(num_face_pts);
      ::tenex::evaluate(make_not_null(&conformal_factor_squared),
                        conformal_factor_face() * conformal_factor_face());

      const auto [face_det_conformal_metric, face_inv_conformal_metric] =
          determinant_and_inverse(conformal_metric_face);

      tnsr::II<DataVector, Dim> inv_spatial_metric(num_face_pts);
      ::tenex::evaluate<ti::I, ti::J>(
          make_not_null(&inv_spatial_metric),
          conformal_factor_squared() * face_inv_conformal_metric(ti::I, ti::J));

      tnsr::ii<DataVector, Dim> spatial_metric(num_face_pts);
      ::tenex::evaluate<ti::i, ti::j>(
          make_not_null(&spatial_metric),
          conformal_metric_face(ti::i, ti::j) / conformal_factor_squared());

      // Compute unit outward normal from inverse Jacobian
      // (bare loop necessary: indexing by runtime normal_dim)
      InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
          outermost_inv_jacobian;
      slice_tensor(outermost_inv_jacobian, inv_jacobian);

      tnsr::i<DataVector, Dim> unnormalized_normal(num_face_pts);
      for (size_t i = 0; i < Dim; ++i) {
        unnormalized_normal.get(i) = static_cast<double>(direction.sign()) *
                                     outermost_inv_jacobian.get(normal_dim, i);
      }

      // Normalize with physical spatial metric (tenex)
      tnsr::I<DataVector, Dim> unit_normal_vector(num_face_pts);
      ::tenex::evaluate<ti::I>(
          make_not_null(&unit_normal_vector),
          inv_spatial_metric(ti::I, ti::J) * unnormalized_normal(ti::j));
      Scalar<DataVector> magnitude(num_face_pts);
      ::tenex::evaluate(
          make_not_null(&magnitude),
          sqrt(unit_normal_vector(ti::I) * unnormalized_normal(ti::i)));
      ::tenex::evaluate<ti::I>(make_not_null(&unit_normal_vector),
                               unit_normal_vector(ti::I) / magnitude());
      const tnsr::i<DataVector, Dim> unit_normal_one_form =
          raise_or_lower_index(unit_normal_vector, spatial_metric);

      // Compute characteristic fields
      typename Tags::CharacteristicFields<
          DataVector, Dim, Frame::Inertial>::type char_fields(num_face_pts);
      ::Ccz4::fd::characteristic_fields<Frame::Inertial>(
          make_not_null(&char_fields), unit_normal_one_form,
          conformal_metric_face, conformal_factor_face, lapse_face, shift_face,
          trace_k_face, a_tilde_face, theta_face, gamma_hat_face, b_aux_face,
          d_conformal_metric, d_conformal_factor, d_lapse, d_shift, System::f);

      // Extract the 4 incoming modes
      const auto& face_u_scalar3_minus =
          get<Tags::UScalar3Minus<DataVector>>(char_fields);
      const auto& face_u_vector2_minus =
          get<Tags::UVector2Minus<DataVector, Dim, Frame::Inertial>>(
              char_fields);
      const auto& face_u_scalar2_minus =
          get<Tags::UScalar2Minus<DataVector>>(char_fields);
      const auto& face_u_tensor_minus =
          get<Tags::UTensorMinus<DataVector, Dim, Frame::Inertial>>(
              char_fields);

      // Scatter face values back to volume boundary mode variables
      for (size_t fp = 0; fp < num_face_pts; ++fp) {
        const size_t vol_idx = volume_index(fp, outermost_layer);
        get(*vol_u_scalar3_minus)[vol_idx] = get(face_u_scalar3_minus)[fp];
        get(*vol_u_scalar2_minus)[vol_idx] = get(face_u_scalar2_minus)[fp];
        for (size_t i = 0; i < Dim; ++i) {
          vol_u_vector2_minus->get(i)[vol_idx] =
              face_u_vector2_minus.get(i)[fp];
        }
        for (size_t ti = 0; ti < vol_u_tensor_minus->size(); ++ti) {
          (*vol_u_tensor_minus)[ti][vol_idx] = face_u_tensor_minus[ti][fp];
        }
      }
    }
  }
};
}  // namespace Ccz4::fd
