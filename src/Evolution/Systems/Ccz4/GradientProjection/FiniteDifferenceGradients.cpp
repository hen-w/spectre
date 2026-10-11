// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/FiniteDifferenceGradients.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::GradientProjection {
void finite_difference_potential_gradients(
    const gsl::not_null<Tags::PotentialGradients::type*> gradients,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::ijj<DataVector, 3>& d_conformal_metric,
    const tnsr::i<DataVector, 3>& d_conformal_factor,
    const tnsr::i<DataVector, 3>& d_lapse,
    const tnsr::iJ<DataVector, 3>& d_shift) {
  const size_t number_of_points = get<0, 0>(conformal_metric).size();
  for (size_t s = 0; s < number_of_points; ++s) {
    Matrix3 metric{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        metric(i, j) = conformal_metric.get(i, j)[s];
      }
    }
    const LogMetricChart full_chart = LogMetricChart::from_metric(metric);
    const auto& h = full_chart.log_eigenvalues();
    const double log_determinant_over_two = h[0] + h[1] + h[2];
    if (std::abs(log_determinant_over_two) > chart_trace_tolerance) {
      ERROR(
          "The conformal metric of a finite-difference element is off the "
          "unit-determinant surface: log(det(conformal metric))/2 = "
          << log_determinant_over_two << " (tolerance " << chart_trace_tolerance
          << ").");
    }
    // The chart of the trace-free H, as on the DG elements.
    const std::array<double, 5> log_metric =
        stf_encode(full_chart.log_metric());
    const LogMetricChart chart{stf_decode(log_metric.data())};
    for (size_t i = 0; i < 3; ++i) {
      Matrix3 metric_derivative{};
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < 3; ++k) {
          metric_derivative(j, k) = d_conformal_metric.get(i, j, k)[s];
        }
      }
      const std::array<double, 5> q_h =
          stf_encode(chart.inverse_tangent(metric_derivative));
      for (size_t c = 0; c < 5; ++c) {
        gradients->get(i, PotentialIndex::log_metric + c)[s] = gsl::at(q_h, c);
      }
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    gradients->get(i, PotentialIndex::conformal_factor) =
        d_conformal_factor.get(i);
    gradients->get(i, PotentialIndex::lapse) = d_lapse.get(i);
    for (size_t k = 0; k < 3; ++k) {
      gradients->get(i, PotentialIndex::shift + k) = d_shift.get(i, k);
    }
  }
}
}  // namespace Ccz4::GradientProjection
