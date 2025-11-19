// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief If constrained_evolution is true, enforce
 * the unit determinant constraint on the conformal spatial metric
 * and traceless constraint on the ATilde
 */
struct EnforceConstrainedEvolution : tt::ConformsTo<db::protocols::Mutator> {
  static constexpr size_t dim = System::volume_dim;
  using return_tags = tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, dim>,
                                 ::Ccz4::Tags::ATilde<DataVector, dim>>;
  using argument_tags = tmpl::list<::Ccz4::fd::Tags::ConstrainedEvolution>;

  static void apply(
      const gsl::not_null<tnsr::ii<DataVector, dim>*> conformal_spatial_metric,
      const gsl::not_null<tnsr::ii<DataVector, dim>*> a_tilde,
      const bool constrained_evolution) {
    if (constrained_evolution) {
      Scalar<DataVector> det_conformal_spatial_metric{};
      determinant(make_not_null(&det_conformal_spatial_metric),
                  *conformal_spatial_metric);
      if (min(get(det_conformal_spatial_metric)) <= 0.0) {
        ERROR(
            "The determinant of the conformal spatial metric is non-positive: "
            << get(det_conformal_spatial_metric));
      }
      get(det_conformal_spatial_metric) =
          pow(get(det_conformal_spatial_metric), -1.0 / 3.0);
      ::tenex::update<ti::i, ti::j>(
          conformal_spatial_metric,
          det_conformal_spatial_metric() *
              (*conformal_spatial_metric)(ti::i, ti::j));

      Scalar<DataVector> trace_a_tilde{};
      tnsr::II<DataVector, dim> inv_conformal_spatial_metric{};
      determinant_and_inverse(make_not_null(&det_conformal_spatial_metric),
                              make_not_null(&inv_conformal_spatial_metric),
                              *conformal_spatial_metric);
      ::tenex::evaluate(make_not_null(&trace_a_tilde),
                        (inv_conformal_spatial_metric)(ti::I, ti::J) *
                            (*a_tilde)(ti::i, ti::j));
      ::tenex::update<ti::i, ti::j>(
          a_tilde,
          (*a_tilde)(ti::i, ti::j) -
              trace_a_tilde() * (*conformal_spatial_metric)(ti::i, ti::j) / 3.);
    }
  }
};
}  // namespace Ccz4::fd
