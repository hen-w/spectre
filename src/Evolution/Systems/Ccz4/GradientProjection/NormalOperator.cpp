// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/NormalOperator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::GradientProjection {
namespace {
using Vector3 = std::array<double, 3>;

// Relative tolerance below which a characteristic speed is treated as zero
// (sonic point) or two speeds of coupled blocks as coincident (loss of strong
// hyperbolicity). It guards numerical degeneracy only; it is not a physical
// cutoff.
constexpr double degeneracy_tolerance = 1.0e-10;

// Offsets of the parts of a normal-subspace vector.
constexpr size_t z_a = VelocityIndex::a;
constexpr size_t z_k = VelocityIndex::trace_extrinsic_curvature;
constexpr size_t z_theta = VelocityIndex::theta;
constexpr size_t z_gamma = VelocityIndex::gamma_hat;
constexpr size_t z_b = VelocityIndex::b;
constexpr size_t z_r = number_of_velocities;
constexpr size_t z_r_h = z_r + PotentialIndex::log_metric;
constexpr size_t z_r_phi = z_r + PotentialIndex::conformal_factor;
constexpr size_t z_r_lapse = z_r + PotentialIndex::lapse;
constexpr size_t z_r_beta = z_r + PotentialIndex::shift;

// Characteristic field offsets.
constexpr size_t c_tensor = 0;
constexpr size_t c_vector = 4;
constexpr size_t c_scalar = 14;

Vector3 matrix_times(const Matrix3& m, const double* const v) {
  Vector3 result{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) = m(i, 0) * v[0] + m(i, 1) * v[1] + m(i, 2) * v[2];
  }
  return result;
}

double dot(const Vector3& a, const double* const b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double bilinear(const Vector3& a, const Matrix3& m, const Vector3& b) {
  double result = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result += gsl::at(a, i) * m(i, j) * gsl::at(b, j);
    }
  }
  return result;
}

// sym(a (x) b) = (a b^T + b a^T) / 2
Matrix3 symmetric_outer(const Vector3& a, const Vector3& b) {
  Matrix3 result{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result(i, j) =
          0.5 * (gsl::at(a, i) * gsl::at(b, j) + gsl::at(a, j) * gsl::at(b, i));
    }
  }
  return result;
}

// Characteristic fields of the block [[v, p], [q, v]] with c = sqrt(pq):
// right eigenvectors (p, +-c), left eigenvectors (1/(2p), +-1/(2c)).
std::array<double, 2> block_fields(const double p, const double c,
                                   const double x, const double y) {
  return {{0.5 * x / p + 0.5 * y / c, 0.5 * x / p - 0.5 * y / c}};
}

std::array<double, 2> block_values(const double p, const double c,
                                   const double plus, const double minus) {
  return {{p * (plus + minus), c * (plus - minus)}};
}

// The Gamma-driver block on (gamma, b, beta),
// gamma' = b' = v gamma + c_s beta, beta' = f b, has the kernel field
// d = gamma - b and on its range (p, s) = (b, (beta + v d / c_s) / f) the block
// [[v, c_s f], [1, 0]] with eigenvalues
// lambda_pm = v/2 +- sqrt(v^2/4 + c_s f).
std::array<double, 3> shift_fields(const double v, const double c_s,
                                   const double f, const double gamma,
                                   const double b, const double beta) {
  const double root = std::sqrt(0.25 * v * v + c_s * f);
  const double lambda_plus = 0.5 * v + root;
  const double lambda_minus = 0.5 * v - root;
  const double d = gamma - b;
  const double p = b;
  const double s = (beta + v * d / c_s) / f;
  return {{(p - lambda_minus * s) / (2.0 * root),
           (-p + lambda_plus * s) / (2.0 * root), d}};
}

std::array<double, 3> shift_values(const double v, const double c_s,
                                   const double f, const double plus,
                                   const double minus, const double kernel) {
  const double root = std::sqrt(0.25 * v * v + c_s * f);
  const double p = (0.5 * v + root) * plus + (0.5 * v - root) * minus;
  const double s = plus + minus;
  return {{p + kernel, p, f * s - v * kernel / c_s}};
}

// Euclidean split of a full jump: the normal gradient part r joins the
// velocities in the normal-subspace vector z and the tangential remainder is
// returned separately.
void split_jump(const gsl::not_null<NormalSubspaceVector*> z,
                const gsl::not_null<GradientVector*> tangential,
                const std::array<double, 3>& normal,
                const VelocityVector& jump_w, const GradientVector& jump_q) {
  const double normal_squared = dot(normal, normal.data());
  for (size_t a = 0; a < number_of_velocities; ++a) {
    gsl::at(*z, a) = gsl::at(jump_w, a);
  }
  for (size_t a = 0; a < number_of_potentials; ++a) {
    const double r =
        (normal[0] * gsl::at(jump_q[0], a) + normal[1] * gsl::at(jump_q[1], a) +
         normal[2] * gsl::at(jump_q[2], a)) /
        normal_squared;
    gsl::at(*z, number_of_velocities + a) = r;
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(gsl::at(*tangential, i), a) =
          gsl::at(gsl::at(jump_q, i), a) - gsl::at(normal, i) * r;
    }
  }
}

// The conformal factor is stored directly, so its positivity (which the
// characteristic structure needs) is no longer implied by the chart.
double checked_conformal_factor(const PotentialVector& u) {
  const double conformal_factor = u[PotentialIndex::conformal_factor];
  if (not(conformal_factor > 0.0)) {
    ERROR(
        "The conformal factor must be positive for the normal operator of the "
        "CCZ4 target operator, but is "
        << conformal_factor << " (lapse " << u[PotentialIndex::lapse] << ")");
  }
  return conformal_factor;
}

void check_gamma_driver_parameter(const double f) {
  if (not(f > 0.0)) {
    ERROR(
        "The Gamma-driver parameter f must be positive for the "
        "characteristic speeds of the CCZ4 target operator, but is "
        << f);
  }
}

double relative_gap(const double a, const double b) {
  const double scale = std::max(std::abs(a), std::abs(b));
  return scale == 0.0 ? 0.0 : std::abs(a - b) / scale;
}
}  // namespace

NormalSubspaceVector normal_flux(const LogMetricChart& chart,
                                 const PotentialVector& u,
                                 const std::array<double, 3>& normal,
                                 const VelocityVector& w,
                                 const GradientVector& q, const double f) {
  const double lapse = u[PotentialIndex::lapse];
  const double conformal_factor = checked_conformal_factor(u);
  const double phi_squared = conformal_factor * conformal_factor;
  const Matrix3 metric = chart.metric();
  const Matrix3 inverse_metric = chart.inverse_metric();
  const double advection = dot(normal, &u[PotentialIndex::shift]);
  const Vector3 raised_normal = matrix_times(inverse_metric, normal.data());

  // Gradient components contracted with the raised normal.
  const auto raised_gradient = [&raised_normal, &q](const size_t component) {
    return raised_normal[0] * q[0][component] +
           raised_normal[1] * q[1][component] +
           raised_normal[2] * q[2][component];
  };
  const auto gradient_covector = [&q](const size_t component,
                                      const double factor) {
    return Vector3{{factor * q[0][component], factor * q[1][component],
                    factor * q[2][component]}};
  };

  // The principal terms of the conformal factor enter through
  // FieldP_i = q_{phi,i} / phi.
  const Vector3 lowered_gamma =
      matrix_times(metric, &w[VelocityIndex::gamma_hat]);
  Matrix3 big_u =
      lapse * (symmetric_outer(normal, lowered_gamma) +
               symmetric_outer(
                   normal, gradient_covector(PotentialIndex::conformal_factor,
                                             1.0 / conformal_factor))) -
      symmetric_outer(normal, gradient_covector(PotentialIndex::lapse, 1.0));
  double trace = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      trace += inverse_metric(i, j) * big_u(i, j);
    }
  }
  big_u = phi_squared * (big_u - (trace / 3.0) * metric);
  const std::array<double, 5> metric_source =
      stf_encode(2.0 * chart.inverse_tangent(big_u));

  NormalSubspaceVector result{};
  for (size_t a = 0; a < 5; ++a) {
    gsl::at(result, z_a + a) =
        advection * gsl::at(w, VelocityIndex::a + a) -
        lapse * phi_squared * raised_gradient(PotentialIndex::log_metric + a) +
        gsl::at(metric_source, a);
  }
  const double normal_gamma = dot(normal, &w[VelocityIndex::gamma_hat]);
  const double raised_field_p =
      raised_gradient(PotentialIndex::conformal_factor) / conformal_factor;
  const double raised_alpha = raised_gradient(PotentialIndex::lapse);
  const double trace_k = w[VelocityIndex::trace_extrinsic_curvature];
  const double theta = w[VelocityIndex::theta];
  result[z_k] = advection * trace_k +
                lapse * phi_squared * (normal_gamma + 4.0 * raised_field_p) -
                phi_squared * raised_alpha;
  result[z_theta] =
      advection * theta +
      lapse * phi_squared * (0.5 * normal_gamma + 2.0 * raised_field_p);
  double divergence = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    divergence += gsl::at(gsl::at(q, i), PotentialIndex::shift + i);
  }
  for (size_t k = 0; k < 3; ++k) {
    const double value = advection * gsl::at(w, VelocityIndex::gamma_hat + k) +
                         lapse * gsl::at(raised_normal, k) *
                             (2.0 * theta - 4.0 * trace_k / 3.0) +
                         raised_gradient(PotentialIndex::shift + k) +
                         gsl::at(raised_normal, k) * divergence / 3.0;
    gsl::at(result, z_gamma + k) = value;
    gsl::at(result, z_b + k) = value;
  }
  const PotentialVector rhs = potential_rhs(chart, u, w, q, f);
  for (size_t a = 0; a < number_of_potentials; ++a) {
    gsl::at(result, z_r + a) = gsl::at(rhs, a);
  }
  return result;
}

void embed_normal_subspace_vector(const gsl::not_null<VelocityVector*> result_w,
                                  const gsl::not_null<GradientVector*> result_q,
                                  const std::array<double, 3>& normal,
                                  const NormalSubspaceVector& z) {
  for (size_t a = 0; a < number_of_velocities; ++a) {
    gsl::at(*result_w, a) = gsl::at(z, a);
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < number_of_potentials; ++a) {
      gsl::at(gsl::at(*result_q, i), a) =
          gsl::at(normal, i) * gsl::at(z, z_r + a);
    }
  }
}

void apply_normal_matrix(const gsl::not_null<VelocityVector*> result_w,
                         const gsl::not_null<GradientVector*> result_q,
                         const LogMetricChart& chart, const PotentialVector& u,
                         const std::array<double, 3>& normal,
                         const VelocityVector& w, const GradientVector& q,
                         const double f) {
  embed_normal_subspace_vector(result_w, result_q, normal,
                               normal_flux(chart, u, normal, w, q, f));
}

NormalCharacteristics::NormalCharacteristics(
    const LogMetricChart& chart, const PotentialVector& u,
    const std::array<double, 3>& normal, const double f)
    : chart_(chart),
      sqrt_metric_(chart.sqrt_metric()),
      inverse_sqrt_metric_(chart.inverse_sqrt_metric()),
      shift_{{u[PotentialIndex::shift], u[PotentialIndex::shift + 1],
              u[PotentialIndex::shift + 2]}},
      lapse_(u[PotentialIndex::lapse]),
      conformal_factor_(checked_conformal_factor(u)),
      f_(f) {
  check_gamma_driver_parameter(f_);
  if (not(lapse_ > 0.0)) {
    ERROR(
        "The lapse must be positive for the characteristic decomposition, "
        "but is "
        << lapse_);
  }
  const Vector3 isotropic_normal =
      matrix_times(inverse_sqrt_metric_, normal.data());
  normal_magnitude_ = std::sqrt(isotropic_normal[0] * isotropic_normal[0] +
                                isotropic_normal[1] * isotropic_normal[1] +
                                isotropic_normal[2] * isotropic_normal[2]);
  if (not(normal_magnitude_ > 0.0)) {
    ERROR("The normal covector vanishes: (" << normal[0] << ", " << normal[1]
                                            << ", " << normal[2] << ")");
  }
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(unit_normal_, i) = gsl::at(isotropic_normal, i) / normal_magnitude_;
  }
  advection_ = dot(normal, &u[PotentialIndex::shift]) / normal_magnitude_;

  // A fixed orthonormal tangent basis.
  size_t axis = 0;
  for (size_t i = 1; i < 3; ++i) {
    if (std::abs(gsl::at(unit_normal_, i)) <
        std::abs(gsl::at(unit_normal_, axis))) {
      axis = i;
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(tangent_1_, i) =
        (i == axis ? 1.0 : 0.0) -
        gsl::at(unit_normal_, axis) * gsl::at(unit_normal_, i);
  }
  const double tangent_norm = std::sqrt(dot(tangent_1_, tangent_1_.data()));
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(tangent_1_, i) /= tangent_norm;
  }
  tangent_2_ = {
      {unit_normal_[1] * tangent_1_[2] - unit_normal_[2] * tangent_1_[1],
       unit_normal_[2] * tangent_1_[0] - unit_normal_[0] * tangent_1_[2],
       unit_normal_[0] * tangent_1_[1] - unit_normal_[1] * tangent_1_[0]}};

  const double v = advection_;
  const double alpha = lapse_;
  const double phi = conformal_factor_;
  const double light = alpha * phi;
  const double gauge = phi * std::sqrt(2.0 * alpha);
  const double transverse_root = std::sqrt(0.25 * v * v + f);
  const double longitudinal_root = std::sqrt(0.25 * v * v + 4.0 * f / 3.0);
  const std::array<double, 2> light_speeds{{v + light, v - light}};
  const std::array<double, 2> gauge_speeds{{v + gauge, v - gauge}};
  const std::array<double, 2> longitudinal_speeds{
      {0.5 * v + longitudinal_root, 0.5 * v - longitudinal_root}};

  const auto report = [this, &normal, &light_speeds, &gauge_speeds,
                       &longitudinal_speeds](const std::string& what) {
    ERROR(what << "\nlapse = " << lapse_
               << ", conformal factor = " << conformal_factor_
               << ", Gamma-driver f = " << f_ << ", normal = (" << normal[0]
               << ", " << normal[1] << ", " << normal[2]
               << "), conformal norm of the normal = " << normal_magnitude_
               << ", shift . normal / norm = " << advection_
               << "\nIsotropic-frame speeds: light/Theta " << light_speeds[0]
               << ", " << light_speeds[1] << "; 1+log " << gauge_speeds[0]
               << ", " << gauge_speeds[1] << "; longitudinal shift "
               << longitudinal_speeds[0] << ", " << longitudinal_speeds[1]);
  };
  for (const double speed : light_speeds) {
    if (std::abs(speed) <= degeneracy_tolerance * (std::abs(v) + light)) {
      report(
          "A light-cone characteristic speed of the CCZ4 target operator "
          "vanishes (sonic point); the normal matrix is not diagonalizable.");
    }
  }
  for (const double speed : gauge_speeds) {
    if (std::abs(speed) <= degeneracy_tolerance * (std::abs(v) + gauge)) {
      report(
          "A 1+log characteristic speed of the CCZ4 target operator vanishes "
          "(sonic point); the normal matrix is not diagonalizable.");
    }
  }
  for (const double shift_speed : longitudinal_speeds) {
    for (const double speed : light_speeds) {
      if (relative_gap(shift_speed, speed) <= degeneracy_tolerance) {
        report(
            "The longitudinal Gamma-driver speed coincides with a light-cone "
            "speed: the puncture gauge is not strongly hyperbolic here.");
      }
    }
    for (const double speed : gauge_speeds) {
      if (relative_gap(shift_speed, speed) <= degeneracy_tolerance) {
        report(
            "The longitudinal Gamma-driver speed coincides with a 1+log "
            "speed: the puncture gauge is not strongly hyperbolic here.");
      }
    }
  }

  // Decoupling of the longitudinal shift block (see the class documentation):
  // e_gamma = e_b = a1 Theta + a2 psi + a1p P + a2p s and
  // e_beta = c1 Theta + c2 psi + c1p P + c2p s solve the Sylvester equations
  // S E - E B = -C of the shift block S, the (Theta, psi) and (P, s) blocks B
  // and their couplings C into the shift rates. delta_theta and delta_lapse
  // vanish exactly at the collisions rejected above.
  const double phi_squared = phi * phi;
  const double d_theta = f - 0.75 * square(light);
  const double delta_theta =
      4.0 / 3.0 * square(d_theta) - 0.75 * square(v * light);
  const double c1 = 2.0 * alpha / 3.0 * f * d_theta / delta_theta;
  const double a1 = 2.0 * alpha / 3.0 * f * v / delta_theta;
  const double c2 = 0.375 * alpha * phi_squared * a1;
  const double a2 = (0.5 * alpha * phi_squared * c1 + v * c2) / f;
  theta_coupling_ = {{a1, a2, c1, c2}};
  const double d_lapse = f - 1.5 * alpha * phi_squared;
  const double delta_lapse =
      4.0 / 3.0 * square(d_lapse) - 1.5 * alpha * phi_squared * square(v);
  const double c1p = 4.0 * alpha / 3.0 * f * d_lapse / delta_lapse;
  const double a1p = 4.0 * alpha / 3.0 * f * v / delta_lapse;
  const double c2p = -0.75 * phi_squared * a1p;
  const double a2p = (c2p * v - c1p * phi_squared) / f;
  lapse_coupling_ = {{a1p, a2p, c1p, c2p}};

  const double m = normal_magnitude_;
  for (size_t polarization = 0; polarization < 2; ++polarization) {
    speeds_[c_tensor + 2 * polarization] = m * light_speeds[0];
    speeds_[c_tensor + 2 * polarization + 1] = m * light_speeds[1];
  }
  for (size_t tangent = 0; tangent < 2; ++tangent) {
    const size_t base = c_vector + 5 * tangent;
    speeds_[base] = m * light_speeds[0];
    speeds_[base + 1] = m * light_speeds[1];
    speeds_[base + 2] = m * (0.5 * v + transverse_root);
    speeds_[base + 3] = m * (0.5 * v - transverse_root);
    speeds_[base + 4] = 0.0;
  }
  speeds_[c_scalar] = m * light_speeds[0];
  speeds_[c_scalar + 1] = m * light_speeds[1];
  speeds_[c_scalar + 2] = m * light_speeds[0];
  speeds_[c_scalar + 3] = m * light_speeds[1];
  speeds_[c_scalar + 4] = m * gauge_speeds[0];
  speeds_[c_scalar + 5] = m * gauge_speeds[1];
  speeds_[c_scalar + 6] = m * longitudinal_speeds[0];
  speeds_[c_scalar + 7] = m * longitudinal_speeds[1];
  speeds_[c_scalar + 8] = 0.0;
}

bool NormalCharacteristics::is_kernel_field(const size_t index) {
  return index == c_vector + 4 or index == c_vector + 9 or
         index == c_scalar + 8;
}

NormalSubspaceVector NormalCharacteristics::to_isotropic(
    const NormalSubspaceVector& z) const {
  const double m = normal_magnitude_;
  const auto transform_tensor = [this](const double* const components) {
    return stf_encode(0.5 * inverse_sqrt_metric_ *
                      chart_.tangent(stf_decode(components)) *
                      inverse_sqrt_metric_);
  };
  NormalSubspaceVector result{};
  const std::array<double, 5> a = transform_tensor(&z[z_a]);
  const std::array<double, 5> r_h = transform_tensor(&z[z_r_h]);
  for (size_t i = 0; i < 5; ++i) {
    gsl::at(result, z_a + i) = gsl::at(a, i);
    gsl::at(result, z_r_h + i) = m * gsl::at(r_h, i);
  }
  result[z_k] = z[z_k];
  result[z_theta] = z[z_theta];
  // The isotropic scalar rate is that of log(phi): R_phi = phi R_log(phi).
  result[z_r_phi] = m * z[z_r_phi] / conformal_factor_;
  result[z_r_lapse] = m * z[z_r_lapse];
  const Vector3 gamma = matrix_times(sqrt_metric_, &z[z_gamma]);
  const Vector3 b = matrix_times(sqrt_metric_, &z[z_b]);
  const Vector3 beta = matrix_times(sqrt_metric_, &z[z_r_beta]);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, z_gamma + i) = gsl::at(gamma, i);
    gsl::at(result, z_b + i) = gsl::at(b, i);
    gsl::at(result, z_r_beta + i) = m * gsl::at(beta, i);
  }
  return result;
}

NormalSubspaceVector NormalCharacteristics::from_isotropic(
    const NormalSubspaceVector& z) const {
  const double m = normal_magnitude_;
  const auto transform_tensor = [this](const double* const components) {
    return stf_encode(
        2.0 * chart_.inverse_tangent(sqrt_metric_ * stf_decode(components) *
                                     sqrt_metric_));
  };
  NormalSubspaceVector result{};
  const std::array<double, 5> a = transform_tensor(&z[z_a]);
  const std::array<double, 5> r_h = transform_tensor(&z[z_r_h]);
  for (size_t i = 0; i < 5; ++i) {
    gsl::at(result, z_a + i) = gsl::at(a, i);
    gsl::at(result, z_r_h + i) = gsl::at(r_h, i) / m;
  }
  result[z_k] = z[z_k];
  result[z_theta] = z[z_theta];
  result[z_r_phi] = conformal_factor_ * z[z_r_phi] / m;
  result[z_r_lapse] = z[z_r_lapse] / m;
  const Vector3 gamma = matrix_times(inverse_sqrt_metric_, &z[z_gamma]);
  const Vector3 b = matrix_times(inverse_sqrt_metric_, &z[z_b]);
  const Vector3 beta = matrix_times(inverse_sqrt_metric_, &z[z_r_beta]);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, z_gamma + i) = gsl::at(gamma, i);
    gsl::at(result, z_b + i) = gsl::at(b, i);
    gsl::at(result, z_r_beta + i) = gsl::at(beta, i) / m;
  }
  return result;
}

NormalSubspaceVector NormalCharacteristics::characteristic_fields(
    const NormalSubspaceVector& z) const {
  return fields_from_isotropic(to_isotropic(z));
}

NormalSubspaceVector NormalCharacteristics::fields_from_isotropic(
    const NormalSubspaceVector& iso) const {
  const double v = advection_;
  const double alpha = lapse_;
  const double phi = conformal_factor_;
  const double light = alpha * phi;
  const double light_p = -alpha * phi * phi;
  const Vector3& n = unit_normal_;
  const std::array<Vector3, 2> tangents{{tangent_1_, tangent_2_}};
  const Matrix3 big_a = stf_decode(&iso[z_a]);
  const Matrix3 h = stf_decode(&iso[z_r_h]);
  const double* const gamma = &iso[z_gamma];
  const double* const b = &iso[z_b];
  const double* const beta = &iso[z_r_beta];
  NormalSubspaceVector result{};

  // Transverse-traceless sector.
  const std::array<double, 2> tt_a{
      {(bilinear(tangents[0], big_a, tangents[0]) -
        bilinear(tangents[1], big_a, tangents[1])) /
           std::sqrt(2.0),
       std::sqrt(2.0) * bilinear(tangents[0], big_a, tangents[1])}};
  const std::array<double, 2> tt_h{
      {(bilinear(tangents[0], h, tangents[0]) -
        bilinear(tangents[1], h, tangents[1])) /
           std::sqrt(2.0),
       std::sqrt(2.0) * bilinear(tangents[0], h, tangents[1])}};
  for (size_t polarization = 0; polarization < 2; ++polarization) {
    const auto fields =
        block_fields(light_p, light, gsl::at(tt_a, polarization),
                     gsl::at(tt_h, polarization));
    result[c_tensor + 2 * polarization] = fields[0];
    result[c_tensor + 2 * polarization + 1] = fields[1];
  }

  // Vector sector.
  for (size_t tangent = 0; tangent < 2; ++tangent) {
    const Vector3& t = gsl::at(tangents, tangent);
    const size_t base = c_vector + 5 * tangent;
    const double gamma_t = dot(t, gamma);
    const auto light_fields =
        block_fields(light_p, light, bilinear(t, big_a, n),
                     bilinear(t, h, n) - 0.5 * gamma_t);
    const auto shift =
        shift_fields(v, 1.0, f_, gamma_t, dot(t, b), dot(t, beta));
    gsl::at(result, base) = light_fields[0];
    gsl::at(result, base + 1) = light_fields[1];
    gsl::at(result, base + 2) = shift[0];
    gsl::at(result, base + 3) = shift[1];
    gsl::at(result, base + 4) = shift[2];
  }

  // Scalar sector.
  const double x = 1.5 * bilinear(n, big_a, n);
  const double y = 1.5 * bilinear(n, h, n);
  const double trace_k = iso[z_k];
  const double theta = iso[z_theta];
  const double gamma_n = dot(n, gamma);
  const double ell = iso[z_r_phi];
  const double s = iso[z_r_lapse];
  const double big_p = trace_k - 2.0 * theta;
  const double psi = gamma_n + 4.0 * ell;
  const auto light_fields =
      block_fields(light_p, light, x - big_p, y - (gamma_n + ell));
  const auto theta_fields =
      block_fields(0.5 * alpha * phi * phi, light, theta, psi);
  const auto gauge_fields =
      block_fields(-phi * phi, phi * std::sqrt(2.0 * alpha), big_p, s);
  const double e_gamma = theta_coupling_[0] * theta + theta_coupling_[1] * psi +
                         lapse_coupling_[0] * big_p + lapse_coupling_[1] * s;
  const double e_beta = theta_coupling_[2] * theta + theta_coupling_[3] * psi +
                        lapse_coupling_[2] * big_p + lapse_coupling_[3] * s;
  const auto shift = shift_fields(v, 4.0 / 3.0, f_, gamma_n - e_gamma,
                                  dot(n, b) - e_gamma, dot(n, beta) - e_beta);
  result[c_scalar] = light_fields[0];
  result[c_scalar + 1] = light_fields[1];
  result[c_scalar + 2] = theta_fields[0];
  result[c_scalar + 3] = theta_fields[1];
  result[c_scalar + 4] = gauge_fields[0];
  result[c_scalar + 5] = gauge_fields[1];
  result[c_scalar + 6] = shift[0];
  result[c_scalar + 7] = shift[1];
  result[c_scalar + 8] = shift[2];
  return result;
}

NormalSubspaceVector
NormalCharacteristics::normal_fields_from_characteristic_fields(
    const NormalSubspaceVector& c) const {
  return from_isotropic(isotropic_from_fields(c));
}

NormalSubspaceVector NormalCharacteristics::isotropic_from_fields(
    const NormalSubspaceVector& c) const {
  const double v = advection_;
  const double alpha = lapse_;
  const double phi = conformal_factor_;
  const double light = alpha * phi;
  const double light_p = -alpha * phi * phi;
  const Vector3& n = unit_normal_;
  const std::array<Vector3, 2> tangents{{tangent_1_, tangent_2_}};

  // Scalar sector.
  const auto theta_psi = block_values(0.5 * alpha * phi * phi, light,
                                      c[c_scalar + 2], c[c_scalar + 3]);
  const auto p_s = block_values(-phi * phi, phi * std::sqrt(2.0 * alpha),
                                c[c_scalar + 4], c[c_scalar + 5]);
  const double theta = theta_psi[0];
  const double psi = theta_psi[1];
  const double big_p = p_s[0];
  const double s = p_s[1];
  const double e_gamma = theta_coupling_[0] * theta + theta_coupling_[1] * psi +
                         lapse_coupling_[0] * big_p + lapse_coupling_[1] * s;
  const double e_beta = theta_coupling_[2] * theta + theta_coupling_[3] * psi +
                        lapse_coupling_[2] * big_p + lapse_coupling_[3] * s;
  const auto shift = shift_values(v, 4.0 / 3.0, f_, c[c_scalar + 6],
                                  c[c_scalar + 7], c[c_scalar + 8]);
  const double gamma_n = shift[0] + e_gamma;
  const double b_n = shift[1] + e_gamma;
  const double beta_n = shift[2] + e_beta;
  const double ell = 0.25 * (psi - gamma_n);
  const double trace_k = big_p + 2.0 * theta;
  const auto x_y = block_values(light_p, light, c[c_scalar], c[c_scalar + 1]);
  const double x = x_y[0] + big_p;
  const double y = x_y[1] + (gamma_n + ell);

  Matrix3 big_a{};
  Matrix3 h{};
  Vector3 gamma{};
  Vector3 b{};
  Vector3 beta{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      const double q_ij =
          gsl::at(n, i) * gsl::at(n, j) - (i == j ? 1.0 : 0.0) / 3.0;
      big_a(i, j) = x * q_ij;
      h(i, j) = y * q_ij;
    }
    gsl::at(gamma, i) = gamma_n * gsl::at(n, i);
    gsl::at(b, i) = b_n * gsl::at(n, i);
    gsl::at(beta, i) = beta_n * gsl::at(n, i);
  }

  // Vector sector.
  for (size_t tangent = 0; tangent < 2; ++tangent) {
    const Vector3& t = gsl::at(tangents, tangent);
    const size_t base = c_vector + 5 * tangent;
    const auto shift_t =
        shift_values(v, 1.0, f_, gsl::at(c, base + 2), gsl::at(c, base + 3),
                     gsl::at(c, base + 4));
    const auto light_t =
        block_values(light_p, light, gsl::at(c, base), gsl::at(c, base + 1));
    const double x_t = light_t[0];
    const double y_t = light_t[1] + 0.5 * shift_t[0];
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        const double sym =
            gsl::at(n, i) * gsl::at(t, j) + gsl::at(t, i) * gsl::at(n, j);
        big_a(i, j) += x_t * sym;
        h(i, j) += y_t * sym;
      }
      gsl::at(gamma, i) += shift_t[0] * gsl::at(t, i);
      gsl::at(b, i) += shift_t[1] * gsl::at(t, i);
      gsl::at(beta, i) += shift_t[2] * gsl::at(t, i);
    }
  }

  // Transverse-traceless sector.
  for (size_t polarization = 0; polarization < 2; ++polarization) {
    const auto values =
        block_values(light_p, light, gsl::at(c, c_tensor + 2 * polarization),
                     gsl::at(c, c_tensor + 2 * polarization + 1));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        const double basis = polarization == 0
                                 ? (tangents[0][i] * tangents[0][j] -
                                    tangents[1][i] * tangents[1][j]) /
                                       std::sqrt(2.0)
                                 : (tangents[0][i] * tangents[1][j] +
                                    tangents[1][i] * tangents[0][j]) /
                                       std::sqrt(2.0);
        big_a(i, j) += values[0] * basis;
        h(i, j) += values[1] * basis;
      }
    }
  }

  NormalSubspaceVector iso{};
  const std::array<double, 5> encoded_a = stf_encode(big_a);
  const std::array<double, 5> encoded_h = stf_encode(h);
  for (size_t i = 0; i < 5; ++i) {
    gsl::at(iso, z_a + i) = gsl::at(encoded_a, i);
    gsl::at(iso, z_r_h + i) = gsl::at(encoded_h, i);
  }
  iso[z_k] = trace_k;
  iso[z_theta] = theta;
  iso[z_r_phi] = ell;
  iso[z_r_lapse] = s;
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(iso, z_gamma + i) = gsl::at(gamma, i);
    gsl::at(iso, z_b + i) = gsl::at(b, i);
    gsl::at(iso, z_r_beta + i) = gsl::at(beta, i);
  }
  return iso;
}

NormalSubspaceVector NormalCharacteristics::apply_sign(
    const NormalSubspaceVector& z) const {
  NormalSubspaceVector fields = characteristic_fields(z);
  for (size_t k = 0; k < number_of_normal_components; ++k) {
    if (is_kernel_field(k)) {
      gsl::at(fields, k) = 0.0;
    } else if (gsl::at(speeds_, k) < 0.0) {
      gsl::at(fields, k) = -gsl::at(fields, k);
    }
  }
  return normal_fields_from_characteristic_fields(fields);
}

NormalSubspaceVector NormalCharacteristics::isotropic_stationary_coupling(
    const std::array<double, 3>& normal,
    const GradientVector& tangential) const {
  const double v = advection_;
  const double alpha = lapse_;
  const double phi = conformal_factor_;
  const double m = normal_magnitude_;
  const double phi_squared = phi * phi;
  const Vector3& n = unit_normal_;
  const std::array<Vector3, 2> tangents{{tangent_1_, tangent_2_}};

  // Isotropic ingredients of F(0,t): the raised-normal- and shift-contracted
  // H-gradient chart tangents, the isotropic shift-gradient matrix, and the
  // conformal-factor and lapse gradient contractions.
  const Vector3 raised_normal =
      matrix_times(chart_.inverse_metric(), normal.data());
  std::array<double, 5> h_nu{};
  std::array<double, 5> h_beta{};
  Vector3 tau_phi{};
  Vector3 tau_alpha{};
  Matrix3 shift_gradient{};
  for (size_t i = 0; i < 3; ++i) {
    const PotentialVector& t_i = gsl::at(tangential, i);
    for (size_t a = 0; a < 5; ++a) {
      gsl::at(h_nu, a) += gsl::at(raised_normal, i) *
                          gsl::at(t_i, PotentialIndex::log_metric + a);
      gsl::at(h_beta, a) +=
          gsl::at(shift_, i) * gsl::at(t_i, PotentialIndex::log_metric + a);
    }
    gsl::at(tau_phi, i) = t_i[PotentialIndex::conformal_factor];
    gsl::at(tau_alpha, i) = t_i[PotentialIndex::lapse];
    for (size_t k = 0; k < 3; ++k) {
      shift_gradient(i, k) = gsl::at(t_i, PotentialIndex::shift + k);
    }
  }
  const Matrix3 theta_nu = 0.5 * inverse_sqrt_metric_ *
                           chart_.tangent(stf_decode(h_nu.data())) *
                           inverse_sqrt_metric_;
  const Matrix3 theta_beta = 0.5 * inverse_sqrt_metric_ *
                             chart_.tangent(stf_decode(h_beta.data())) *
                             inverse_sqrt_metric_;
  const Matrix3 iso_shift_gradient =
      inverse_sqrt_metric_ * shift_gradient * sqrt_metric_;
  const double trace_shift_gradient =
      shift_gradient(0, 0) + shift_gradient(1, 1) + shift_gradient(2, 2);
  Vector3 kappa{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(kappa, i) =
        alpha * gsl::at(tau_phi, i) / phi - gsl::at(tau_alpha, i);
  }
  const Vector3 iso_kappa = matrix_times(inverse_sqrt_metric_, kappa.data());
  const double normal_kappa = dot(n, iso_kappa.data());
  const double nu_tau_phi = dot(raised_normal, tau_phi.data());
  const double nu_tau_alpha = dot(raised_normal, tau_alpha.data());
  const double beta_tau_phi = dot(shift_, tau_phi.data());
  const double beta_tau_alpha = dot(shift_, tau_alpha.data());

  // The isotropic components of F(0,t), in the variable convention of
  // to_isotropic. The GammaHat and b rows are equal and the shift-rate rows
  // vanish (R_beta = f b at w = 0).
  Matrix3 a_bar = -alpha * phi_squared * theta_nu +
                  m * phi_squared * symmetric_outer(n, iso_kappa);
  Matrix3 h_bar = theta_beta;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      h_bar(i, j) +=
          0.5 * (iso_shift_gradient(i, j) + iso_shift_gradient(j, i));
    }
    a_bar(i, i) -= m * phi_squared * normal_kappa / 3.0;
    h_bar(i, i) -= trace_shift_gradient / 3.0;
  }
  h_bar *= m;
  const double k_bar =
      4.0 * alpha * phi * nu_tau_phi - phi_squared * nu_tau_alpha;
  const double theta_bar = 2.0 * alpha * phi * nu_tau_phi;
  const double ell_bar = m * (beta_tau_phi / phi - trace_shift_gradient / 3.0);
  const double s_bar = m * beta_tau_alpha;
  Vector3 gamma_bar{};
  for (size_t i = 0; i < 3; ++i) {
    double transposed_shift_normal = 0.0;
    for (size_t j = 0; j < 3; ++j) {
      transposed_shift_normal += iso_shift_gradient(j, i) * gsl::at(n, j);
    }
    gsl::at(gamma_bar, i) = m * (transposed_shift_normal +
                                 trace_shift_gradient * gsl::at(n, i) / 3.0);
  }

  // Sector solves of M(Kt) = F(0,t) with zero kernel characteristic content,
  // in the unit-normal-normalized variables (all right-hand sides divided by
  // the conformal normal magnitude). The 2x2 blocks are inverted by their
  // adjugates over the determinants, which are the products of the paired
  // characteristic speeds and are kept away from zero by the constructor's
  // degeneracy errors; the Gamma-driver blocks divide only by f and c_s.
  const double inv_m = 1.0 / m;
  const double det_light = v * v - alpha * alpha * phi_squared;
  const double det_gauge = v * v - 2.0 * alpha * phi_squared;
  const double light_p = -alpha * phi_squared;
  const double light_q = -alpha;
  const auto solve_light = [v, light_p, light_q, det_light](const double b_1,
                                                            const double b_2) {
    return std::array<double, 2>{{(v * b_1 - light_p * b_2) / det_light,
                                  (v * b_2 - light_q * b_1) / det_light}};
  };

  // Transverse-traceless sector.
  const std::array<double, 2> tt_a_bar{
      {(bilinear(tangents[0], a_bar, tangents[0]) -
        bilinear(tangents[1], a_bar, tangents[1])) *
           inv_m / std::sqrt(2.0),
       std::sqrt(2.0) * bilinear(tangents[0], a_bar, tangents[1]) * inv_m}};
  const std::array<double, 2> tt_h_bar{
      {(bilinear(tangents[0], h_bar, tangents[0]) -
        bilinear(tangents[1], h_bar, tangents[1])) *
           inv_m / std::sqrt(2.0),
       std::sqrt(2.0) * bilinear(tangents[0], h_bar, tangents[1]) * inv_m}};
  std::array<std::array<double, 2>, 2> coupling_tt{};
  for (size_t polarization = 0; polarization < 2; ++polarization) {
    gsl::at(coupling_tt, polarization) = solve_light(
        gsl::at(tt_a_bar, polarization), gsl::at(tt_h_bar, polarization));
  }

  // Vector sector: the driver solution has zero GammaHat and b components, so
  // the light block decouples with the plain H projection.
  std::array<std::array<double, 2>, 2> coupling_vector_light{};
  std::array<double, 2> coupling_vector_beta{};
  for (size_t tangent = 0; tangent < 2; ++tangent) {
    const Vector3& t = gsl::at(tangents, tangent);
    const double b_a = bilinear(t, a_bar, n) * inv_m;
    const double b_gamma = dot(t, gamma_bar.data()) * inv_m;
    const double b_h = bilinear(t, h_bar, n) * inv_m - 0.5 * b_gamma;
    gsl::at(coupling_vector_light, tangent) = solve_light(b_a, b_h);
    gsl::at(coupling_vector_beta, tangent) = b_gamma;
  }

  // Scalar sector, through the same decoupling as the transforms.
  const double b_x = 1.5 * bilinear(n, a_bar, n) * inv_m;
  const double b_y = 1.5 * bilinear(n, h_bar, n) * inv_m;
  const double b_theta = theta_bar * inv_m;
  const double b_k = k_bar * inv_m;
  const double b_ell = ell_bar * inv_m;
  const double b_s = s_bar * inv_m;
  const double b_gamma_n = dot(n, gamma_bar.data()) * inv_m;
  const double b_psi = b_gamma_n + 4.0 * b_ell;
  const double b_big_p = b_k - 2.0 * b_theta;
  const double e_gamma_b =
      theta_coupling_[0] * b_theta + theta_coupling_[1] * b_psi +
      lapse_coupling_[0] * b_big_p + lapse_coupling_[1] * b_s;
  const double e_beta_b =
      theta_coupling_[2] * b_theta + theta_coupling_[3] * b_psi +
      lapse_coupling_[2] * b_big_p + lapse_coupling_[3] * b_s;
  const double k_theta =
      (v * b_theta - 0.5 * alpha * phi_squared * b_psi) / det_light;
  const double k_psi = (v * b_psi - 2.0 * alpha * b_theta) / det_light;
  const double k_big_p = (v * b_big_p + phi_squared * b_s) / det_gauge;
  const double k_s = (v * b_s + 2.0 * alpha * b_big_p) / det_gauge;
  const auto light_scalar = solve_light(b_x - b_big_p, b_y - b_gamma_n - b_ell);
  const double k_gamma_tilde = -e_beta_b / f_;
  const double k_beta_tilde =
      0.75 * (b_gamma_n - e_gamma_b - v * k_gamma_tilde);
  const double k_gamma_n = k_gamma_tilde + theta_coupling_[0] * k_theta +
                           theta_coupling_[1] * k_psi +
                           lapse_coupling_[0] * k_big_p +
                           lapse_coupling_[1] * k_s;
  const double k_beta_n =
      k_beta_tilde + theta_coupling_[2] * k_theta + theta_coupling_[3] * k_psi +
      lapse_coupling_[2] * k_big_p + lapse_coupling_[3] * k_s;
  const double k_ell = 0.25 * (k_psi - k_gamma_n);
  const double k_trace = k_big_p + 2.0 * k_theta;
  const double k_x = light_scalar[0] + k_big_p;
  const double k_y = light_scalar[1] + k_gamma_n + k_ell;

  // Assemble the isotropic components, mirroring the inverse transform. The
  // kernel condition makes the GammaHat and b components equal.
  Matrix3 a_coupling{};
  Matrix3 h_coupling{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      const double q_ij =
          gsl::at(n, i) * gsl::at(n, j) - (i == j ? 1.0 : 0.0) / 3.0;
      a_coupling(i, j) = k_x * q_ij;
      h_coupling(i, j) = k_y * q_ij;
      for (size_t tangent = 0; tangent < 2; ++tangent) {
        const Vector3& t = gsl::at(tangents, tangent);
        const double sym =
            gsl::at(n, i) * gsl::at(t, j) + gsl::at(t, i) * gsl::at(n, j);
        a_coupling(i, j) += gsl::at(coupling_vector_light, tangent)[0] * sym;
        h_coupling(i, j) += gsl::at(coupling_vector_light, tangent)[1] * sym;
      }
      const double basis_plus =
          (tangents[0][i] * tangents[0][j] - tangents[1][i] * tangents[1][j]) /
          std::sqrt(2.0);
      const double basis_cross =
          (tangents[0][i] * tangents[1][j] + tangents[1][i] * tangents[0][j]) /
          std::sqrt(2.0);
      a_coupling(i, j) +=
          coupling_tt[0][0] * basis_plus + coupling_tt[1][0] * basis_cross;
      h_coupling(i, j) +=
          coupling_tt[0][1] * basis_plus + coupling_tt[1][1] * basis_cross;
    }
  }
  NormalSubspaceVector coupling{};
  const std::array<double, 5> encoded_a = stf_encode(a_coupling);
  const std::array<double, 5> encoded_h = stf_encode(h_coupling);
  for (size_t i = 0; i < 5; ++i) {
    gsl::at(coupling, z_a + i) = gsl::at(encoded_a, i);
    gsl::at(coupling, z_r_h + i) = gsl::at(encoded_h, i);
  }
  coupling[z_k] = k_trace;
  coupling[z_theta] = k_theta;
  coupling[z_r_phi] = k_ell;
  coupling[z_r_lapse] = k_s;
  for (size_t i = 0; i < 3; ++i) {
    const double beta_i = k_beta_n * gsl::at(n, i) +
                          coupling_vector_beta[0] * gsl::at(tangents[0], i) +
                          coupling_vector_beta[1] * gsl::at(tangents[1], i);
    gsl::at(coupling, z_gamma + i) = k_gamma_n * gsl::at(n, i);
    gsl::at(coupling, z_b + i) = k_gamma_n * gsl::at(n, i);
    gsl::at(coupling, z_r_beta + i) = beta_i;
  }
  return coupling;
}

void NormalCharacteristics::full_characteristic_fields(
    const gsl::not_null<NormalSubspaceVector*> fields,
    const gsl::not_null<GradientVector*> tangential,
    const std::array<double, 3>& normal, const VelocityVector& jump_w,
    const GradientVector& jump_q) const {
  NormalSubspaceVector z{};
  split_jump(make_not_null(&z), tangential, normal, jump_w, jump_q);
  NormalSubspaceVector iso = to_isotropic(z);
  const NormalSubspaceVector coupling =
      isotropic_stationary_coupling(normal, *tangential);
  for (size_t k = 0; k < number_of_normal_components; ++k) {
    gsl::at(iso, k) += gsl::at(coupling, k);
  }
  *fields = fields_from_isotropic(iso);
}

void NormalCharacteristics::full_fields_from_characteristic_fields(
    const gsl::not_null<VelocityVector*> jump_w,
    const gsl::not_null<GradientVector*> jump_q,
    const std::array<double, 3>& normal, const NormalSubspaceVector& fields,
    const GradientVector& tangential) const {
  const NormalSubspaceVector coupling =
      isotropic_stationary_coupling(normal, tangential);
  NormalSubspaceVector iso = isotropic_from_fields(fields);
  for (size_t k = 0; k < number_of_normal_components; ++k) {
    gsl::at(iso, k) -= gsl::at(coupling, k);
  }
  embed_normal_subspace_vector(jump_w, jump_q, normal, from_isotropic(iso));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < number_of_potentials; ++a) {
      gsl::at(gsl::at(*jump_q, i), a) += gsl::at(gsl::at(tangential, i), a);
    }
  }
}

void NormalCharacteristics::add_kernel_penalty(
    const gsl::not_null<VelocityVector*> result_w,
    const gsl::not_null<GradientVector*> result_q,
    const std::array<double, 3>& normal, const VelocityVector& jump_w,
    const double weight) const {
  const double v = advection_;
  const double m = normal_magnitude_;
  const Vector3& n = unit_normal_;
  Vector3 difference{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(difference, i) = gsl::at(jump_w, VelocityIndex::gamma_hat + i) -
                             gsl::at(jump_w, VelocityIndex::b + i);
  }
  const Vector3 iso_difference = matrix_times(sqrt_metric_, difference.data());
  const double amplitude_normal = dot(n, iso_difference.data());
  Vector3 tangential_part{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(tangential_part, i) =
        gsl::at(iso_difference, i) - amplitude_normal * gsl::at(n, i);
  }
  // The kernel eigenvectors' inverse-transform images: GammaHat receives the
  // jump difference itself, b nothing, and the rate slots the closed forms of
  // the derivation note.
  Matrix3 h_iso = symmetric_outer(n, tangential_part);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      h_iso(i, j) +=
          0.75 * amplitude_normal *
          (gsl::at(n, i) * gsl::at(n, j) - (i == j ? 1.0 : 0.0) / 3.0);
    }
  }
  const std::array<double, 5> rate_h = stf_encode(
      2.0 * chart_.inverse_tangent(sqrt_metric_ * h_iso * sqrt_metric_));
  Vector3 beta_direction{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(beta_direction, i) =
        gsl::at(tangential_part, i) + 0.75 * amplitude_normal * gsl::at(n, i);
  }
  const Vector3 rate_beta =
      matrix_times(inverse_sqrt_metric_, beta_direction.data());
  PotentialVector rate{};
  for (size_t a = 0; a < 5; ++a) {
    gsl::at(rate, PotentialIndex::log_metric + a) = gsl::at(rate_h, a) / m;
  }
  rate[PotentialIndex::conformal_factor] =
      -conformal_factor_ * amplitude_normal / (4.0 * m);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(rate, PotentialIndex::shift + i) = -v * gsl::at(rate_beta, i) / m;
  }
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(*result_w, VelocityIndex::gamma_hat + i) +=
        weight * gsl::at(difference, i);
    for (size_t a = 0; a < number_of_potentials; ++a) {
      gsl::at(gsl::at(*result_q, i), a) +=
          weight * gsl::at(normal, i) * gsl::at(rate, a);
    }
  }
}

void NormalCharacteristics::apply_upwind_and_penalties(
    const gsl::not_null<VelocityVector*> result_w,
    const gsl::not_null<GradientVector*> result_q,
    const std::array<double, 3>& normal, const VelocityVector& jump_w,
    const GradientVector& jump_q, const NormalSubspaceVector& flux,
    const double upwind_factor, const double sigma_p, const double sigma_t,
    const double sigma_g) const {
  if (sigma_p == sigma_t) {
    // Equal propagating and stationary-gradient penalties commute with the
    // full transform up to the kernel projector, so the penalty is sigma_p
    // times the jump plus (sigma_g - sigma_p) P_g; the upwinding is the sign
    // route on the flux.
    NormalSubspaceVector weighted = characteristic_fields(flux);
    for (size_t k = 0; k < number_of_normal_components; ++k) {
      if (is_kernel_field(k)) {
        gsl::at(weighted, k) = 0.0;
      } else if (gsl::at(speeds_, k) < 0.0) {
        gsl::at(weighted, k) *= -upwind_factor;
      } else {
        gsl::at(weighted, k) *= upwind_factor;
      }
    }
    embed_normal_subspace_vector(
        result_w, result_q, normal,
        normal_fields_from_characteristic_fields(weighted));
    for (size_t a = 0; a < number_of_velocities; ++a) {
      gsl::at(*result_w, a) += sigma_p * gsl::at(jump_w, a);
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < number_of_potentials; ++a) {
        gsl::at(gsl::at(*result_q, i), a) +=
            sigma_p * gsl::at(gsl::at(jump_q, i), a);
      }
    }
    if (sigma_g != sigma_p) {
      add_kernel_penalty(result_w, result_q, normal, jump_w, sigma_g - sigma_p);
    }
    return;
  }
  // General path: the full forward transform, the diagonal weights, and the
  // inverse transform. The flux is not needed here.
  NormalSubspaceVector z{};
  GradientVector tangential{};
  split_jump(make_not_null(&z), make_not_null(&tangential), normal, jump_w,
             jump_q);
  const NormalSubspaceVector coupling =
      isotropic_stationary_coupling(normal, tangential);
  NormalSubspaceVector iso = to_isotropic(z);
  for (size_t k = 0; k < number_of_normal_components; ++k) {
    gsl::at(iso, k) += gsl::at(coupling, k);
  }
  NormalSubspaceVector weighted = fields_from_isotropic(iso);
  for (size_t k = 0; k < number_of_normal_components; ++k) {
    if (is_kernel_field(k)) {
      gsl::at(weighted, k) *= sigma_g;
    } else {
      gsl::at(weighted, k) *=
          upwind_factor * std::abs(gsl::at(speeds_, k)) + sigma_p;
    }
  }
  NormalSubspaceVector iso_out = isotropic_from_fields(weighted);
  for (size_t k = 0; k < number_of_normal_components; ++k) {
    gsl::at(iso_out, k) -= sigma_t * gsl::at(coupling, k);
  }
  embed_normal_subspace_vector(result_w, result_q, normal,
                               from_isotropic(iso_out));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < number_of_potentials; ++a) {
      gsl::at(gsl::at(*result_q, i), a) +=
          sigma_t * gsl::at(gsl::at(tangential, i), a);
    }
  }
}

void apply_absolute_normal_matrix(const gsl::not_null<VelocityVector*> result_w,
                                  const gsl::not_null<GradientVector*> result_q,
                                  const LogMetricChart& chart,
                                  const PotentialVector& u,
                                  const std::array<double, 3>& normal,
                                  const VelocityVector& w,
                                  const GradientVector& q, const double f) {
  const NormalCharacteristics characteristics{chart, u, normal, f};
  embed_normal_subspace_vector(
      result_w, result_q, normal,
      characteristics.apply_sign(normal_flux(chart, u, normal, w, q, f)));
}

double largest_characteristic_speed(const LogMetricChart& chart,
                                    const PotentialVector& u,
                                    const std::array<double, 3>& normal,
                                    const double f) {
  check_gamma_driver_parameter(f);
  const Vector3 raised = matrix_times(chart.inverse_metric(), normal.data());
  const double m = std::sqrt(dot(raised, normal.data()));
  const double v = std::abs(dot(normal, &u[PotentialIndex::shift]));
  const double lapse = u[PotentialIndex::lapse];
  const double phi = checked_conformal_factor(u);
  return std::max({v + lapse * phi * m, v + phi * m * std::sqrt(2.0 * lapse),
                   0.5 * v + std::sqrt(0.25 * v * v + 4.0 * f * m * m / 3.0)});
}
}  // namespace Ccz4::GradientProjection
