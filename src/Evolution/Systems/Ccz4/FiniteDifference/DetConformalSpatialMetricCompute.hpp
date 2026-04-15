// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TempTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Compute tag for \f$\det\tilde{\gamma}_{ij}\f$, the determinant of
 * the conformal spatial metric.
 *
 * \details Recomputed on demand from the evolved variable
 * `Ccz4::Tags::ConformalMetric`. The CCZ4 formulation enforces
 * \f$\det\tilde{\gamma}_{ij} = 1\f$ analytically, so this quantity can be
 * observed to monitor how well that algebraic constraint is preserved during
 * the evolution.
 */
struct DetConformalSpatialMetricCompute
    : ::Ccz4::Tags::DetConformalSpatialMetric<DataVector>,
      db::ComputeTag {
  using base = ::Ccz4::Tags::DetConformalSpatialMetric<DataVector>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, 3>>;

  static void function(
      const gsl::not_null<return_type*> det_conformal_metric,
      const tnsr::ii<DataVector, 3>& conformal_metric) {
    determinant(det_conformal_metric, conformal_metric);
  }
};
}  // namespace Ccz4::fd
