// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace TestHelpers::Ccz4GradientProjection {
namespace gp = ::Ccz4::GradientProjection;

/// Row-major flattening of a 3x3 matrix.
inline DataVector flatten(const gp::Matrix3& matrix) {
  DataVector result{9};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result[3 * i + j] = matrix(i, j);
    }
  }
  return result;
}

inline gp::Matrix3 unflatten(const DataVector& flat) {
  gp::Matrix3 result{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result(i, j) = flat[3 * i + j];
    }
  }
  return result;
}

template <size_t Size>
DataVector to_data_vector(const std::array<double, Size>& values) {
  DataVector result{Size};
  for (size_t i = 0; i < Size; ++i) {
    result[i] = gsl::at(values, i);
  }
  return result;
}

inline DataVector to_data_vector(const gp::GradientVector& q) {
  DataVector result{30};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      result[10 * i + a] = gsl::at(gsl::at(q, i), a);
    }
  }
  return result;
}

inline double max_abs(const gp::Matrix3& matrix) {
  double result = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result = std::max(result, std::abs(matrix(i, j)));
    }
  }
  return result;
}

inline double max_abs_difference(const gp::Matrix3& a, const DataVector& b) {
  double result = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result = std::max(result, std::abs(a(i, j) - b[3 * i + j]));
    }
  }
  return result;
}

inline gp::Matrix3 random_symmetric(const gsl::not_null<std::mt19937*> gen,
                                    const double scale) {
  std::normal_distribution<double> dist(0.0, scale);
  gp::Matrix3 result{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result(i, j) = result(j, i) = dist(*gen);
    }
  }
  return result;
}

/// A random rotation from the Cayley transform of an antisymmetric matrix.
inline gp::Matrix3 random_rotation(const gsl::not_null<std::mt19937*> gen) {
  std::normal_distribution<double> dist(0.0, 1.0);
  gp::Matrix3 antisymmetric{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i + 1; j < 3; ++j) {
      antisymmetric(i, j) = dist(*gen);
      antisymmetric(j, i) = -antisymmetric(i, j);
    }
  }
  gp::Matrix3 identity{};
  for (size_t i = 0; i < 3; ++i) {
    identity(i, i) = 1.0;
  }
  const gp::Matrix3 inverse = blaze::inv(identity - antisymmetric);
  return (identity + antisymmetric) * inverse;
}

/// \f$H=R\,\mathrm{diag}(h)R^T\f$ with trace-free \f$h\f$.
inline gp::Matrix3 log_metric_with_spectrum(
    const gsl::not_null<std::mt19937*> gen, const std::array<double, 3>& h) {
  const gp::Matrix3 rotation = random_rotation(gen);
  gp::Matrix3 diagonal{};
  const double mean = (h[0] + h[1] + h[2]) / 3.0;
  for (size_t i = 0; i < 3; ++i) {
    diagonal(i, i) = gsl::at(h, i) - mean;
  }
  return rotation * diagonal * blaze::trans(rotation);
}

/*!
 * \brief An admissible chart state: anisotropic log-metric with entries of
 * size `h_scale`, lapse and (physical) conformal factor in the given ranges,
 * shift of size `shift_scale`, velocities and gradients of size `field_scale`.
 */
struct RandomState {
  gp::PotentialVector u{};
  gp::VelocityVector w{};
  gp::GradientVector q{};
};

inline RandomState random_state(
    const gsl::not_null<std::mt19937*> gen, const double h_scale,
    const double shift_scale, const double field_scale,
    const std::array<double, 2>& lapse_range = {{0.4, 1.5}},
    const std::array<double, 2>& phi_range = {{0.6, 1.3}}) {
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> lapse(lapse_range[0], lapse_range[1]);
  std::uniform_real_distribution<double> phi(phi_range[0], phi_range[1]);
  RandomState result{};
  for (size_t a = 0; a < 5; ++a) {
    gsl::at(result.u, a) = h_scale * normal(*gen);
  }
  result.u[gp::PotentialIndex::conformal_factor] = phi(*gen);
  result.u[gp::PotentialIndex::lapse] = lapse(*gen);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result.u, gp::PotentialIndex::shift + i) =
        shift_scale * normal(*gen);
  }
  for (size_t a = 0; a < gp::number_of_velocities; ++a) {
    gsl::at(result.w, a) = field_scale * normal(*gen);
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      gsl::at(gsl::at(result.q, i), a) = field_scale * normal(*gen);
    }
  }
  return result;
}

inline std::array<double, 3> random_unit_vector(
    const gsl::not_null<std::mt19937*> gen) {
  std::normal_distribution<double> normal(0.0, 1.0);
  std::array<double, 3> result{{normal(*gen), normal(*gen), normal(*gen)}};
  const double norm = std::sqrt(result[0] * result[0] + result[1] * result[1] +
                                result[2] * result[2]);
  for (auto& component : result) {
    component /= norm;
  }
  return result;
}
/*!
 * \brief Owning chart data at a set of points: the local \f$H\f$, \f$a\f$, the
 * other potentials and velocities as physical fields, and the physical
 * conformal metric \f$e^{2H}\f$ and \f$\tilde A=\frac12\mathcal T_Ha\f$
 * consistent with them; `potentials()` and `velocities()` are the views used
 * by the production code.
 */
struct ChartData {
  explicit ChartData(const size_t number_of_points)
      : log_conformal_metric(number_of_points),
        conformal_factor(number_of_points),
        lapse(number_of_points),
        shift(number_of_points),
        a(number_of_points),
        trace_extrinsic_curvature(number_of_points),
        theta(number_of_points),
        gamma_hat(number_of_points),
        b(number_of_points),
        conformal_metric(number_of_points),
        a_tilde(number_of_points) {}

  gp::Tags::LogConformalMetric::type log_conformal_metric;
  Scalar<DataVector> conformal_factor;
  Scalar<DataVector> lapse;
  tnsr::I<DataVector, 3> shift;
  gp::Tags::TransformedATilde::type a;
  Scalar<DataVector> trace_extrinsic_curvature;
  Scalar<DataVector> theta;
  tnsr::I<DataVector, 3> gamma_hat;
  tnsr::I<DataVector, 3> b;
  tnsr::ii<DataVector, 3> conformal_metric;
  tnsr::ii<DataVector, 3> a_tilde;

  gp::PotentialFields potentials() const {
    return {log_conformal_metric, conformal_factor, lapse, shift};
  }
  gp::VelocityFields velocities() const {
    return {a, trace_extrinsic_curvature, theta, gamma_hat, b};
  }

  void set(const size_t point, const gp::PotentialVector& u,
           const gp::VelocityVector& w) {
    for (size_t c = 0; c < 5; ++c) {
      log_conformal_metric.get(c)[point] =
          gsl::at(u, gp::PotentialIndex::log_metric + c);
      a.get(c)[point] = gsl::at(w, gp::VelocityIndex::a + c);
    }
    get(conformal_factor)[point] = u[gp::PotentialIndex::conformal_factor];
    get(lapse)[point] = u[gp::PotentialIndex::lapse];
    get(trace_extrinsic_curvature)[point] =
        w[gp::VelocityIndex::trace_extrinsic_curvature];
    get(theta)[point] = w[gp::VelocityIndex::theta];
    for (size_t i = 0; i < 3; ++i) {
      shift.get(i)[point] = gsl::at(u, gp::PotentialIndex::shift + i);
      gamma_hat.get(i)[point] = gsl::at(w, gp::VelocityIndex::gamma_hat + i);
      b.get(i)[point] = gsl::at(w, gp::VelocityIndex::b + i);
    }
    const gp::LogMetricChart chart{
        gp::stf_decode(&u[gp::PotentialIndex::log_metric])};
    const gp::Matrix3 g = chart.metric();
    const gp::Matrix3 physical_a_tilde =
        0.5 * chart.tangent(gp::stf_decode(&w[gp::VelocityIndex::a]));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        conformal_metric.get(i, j)[point] = g(i, j);
        a_tilde.get(i, j)[point] = physical_a_tilde(i, j);
      }
    }
  }
};

/// Geometry of an element: an affine map, or a curved polynomial map whose
// mass matrix is not separable.
struct Geometry {
  Mesh<3> mesh;
  tnsr::I<DataVector, 3, Frame::ElementLogical> logical;
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian;
  Scalar<DataVector> det_inverse_jacobian;
  double exact_volume;  // only meaningful for the affine map
};

inline Geometry make_geometry(const std::array<size_t, 3>& extents,
                              const bool curved) {
  Geometry result{};
  result.mesh = Mesh<3>{extents, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  result.logical = logical_coordinates(result.mesh);
  const size_t n = result.mesh.number_of_grid_points();
  const std::array<double, 3> widths{{1.3, 0.7, 2.1}};
  Jacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial> jacobian{n,
                                                                           0.0};
  for (size_t i = 0; i < 3; ++i) {
    jacobian.get(i, i) = 0.5 * gsl::at(widths, i);
  }
  if (curved) {
    // x^i = w_i xi^i / 2 + 0.08 * (xi^{i+1})^2 * xi^{i+2} (cyclic)
    for (size_t i = 0; i < 3; ++i) {
      const size_t j = (i + 1) % 3;
      const size_t k = (i + 2) % 3;
      jacobian.get(i, j) +=
          0.16 * result.logical.get(j) * result.logical.get(k);
      jacobian.get(i, k) += 0.08 * square(result.logical.get(j));
    }
  }
  const auto det_and_inverse = determinant_and_inverse(jacobian);
  result.det_inverse_jacobian =
      Scalar<DataVector>{1.0 / get(det_and_inverse.first)};
  result.inverse_jacobian = det_and_inverse.second;
  result.exact_volume = widths[0] * widths[1] * widths[2];
  return result;
}

/// Lagrange interpolation weights at `target` for five nodes (independent of
/// the production finite-difference reconstruction stencils).
inline std::array<double, 5> lagrange_weights(
    const std::array<double, 5>& nodes, const double target) {
  std::array<double, 5> weights{};
  for (size_t m = 0; m < 5; ++m) {
    double weight = 1.0;
    for (size_t l = 0; l < 5; ++l) {
      if (l != m) {
        weight *= (target - gsl::at(nodes, l)) /
                  (gsl::at(nodes, m) - gsl::at(nodes, l));
      }
    }
    gsl::at(weights, m) = weight;
  }
  return weights;
}

}  // namespace TestHelpers::Ccz4GradientProjection
