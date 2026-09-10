// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/Ccz4/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Ccz4/BoundaryCorrections/LaxFriedrichs.hpp"
#include "Evolution/Systems/Ccz4/BoundaryCorrections/ProposedFlux.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

// This test mirrors the coverage of Test_LaxFriedrichs.cpp for the packaging,
// auxiliary packaging, boundary terms, auxiliary boundary terms, boundary
// override, serialization, and all-output-initialization cases, updating the
// three modified physical corrections (ATilde, K, Theta) with independent
// component-loop formulas. It then adds regressions that pin the
// ProposedFlux-vs-LaxFriedrichs delta, the trace-free property of Q, the use
// of the correct per-side conformal metric, no-op recovery of LaxFriedrichs,
// consistency, and factory/serialization round-trips.

namespace {

constexpr size_t face_size = 25;  // 5x5 face

// Normal covector for upper_xi face: (1, 0, 0)
tnsr::i<DataVector, 3, Frame::Inertial> make_unit_normal(
    const size_t num_points) {
  auto n = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(num_points), 0.0);
  get<0>(n) = 1.0;
  return n;
}

// Normal covector pointing in -x, used for the exterior side.
tnsr::i<DataVector, 3, Frame::Inertial> make_negative_unit_normal(
    const size_t num_points) {
  auto n = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(num_points), 0.0);
  get<0>(n) = -1.0;
  return n;
}

// Compute inverse of a 3x3 symmetric matrix at a single point (mirrors the
// helper in Test_LaxFriedrichs.cpp).
void invert_3x3_symmetric(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& metric,
    tnsr::II<DataVector, 3, Frame::Inertial>* inv_metric, const size_t q) {
  std::array<std::array<double, 3>, 3> m{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      m[i][j] = metric.get(i, j)[q];
    }
  }
  const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                     m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                     m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const double inv_det = 1.0 / det;
  inv_metric->get(0, 0)[q] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
  inv_metric->get(0, 1)[q] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
  inv_metric->get(0, 2)[q] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
  inv_metric->get(1, 1)[q] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
  inv_metric->get(1, 2)[q] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
  inv_metric->get(2, 2)[q] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
  inv_metric->get(1, 0)[q] = inv_metric->get(0, 1)[q];
  inv_metric->get(2, 0)[q] = inv_metric->get(0, 2)[q];
  inv_metric->get(2, 1)[q] = inv_metric->get(1, 2)[q];
}

// Generate a well-conditioned conformal metric: delta_{ij} + eps * random_{ij}
// (mirrors Test_LaxFriedrichs.cpp).
template <typename Generator>
tnsr::ii<DataVector, 3, Frame::Inertial> make_random_conformal_metric(
    const gsl::not_null<Generator*> gen, const size_t num_points) {
  std::uniform_real_distribution<> small_dist(-0.1, 0.1);
  auto perturbation =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          gen, small_dist, DataVector(num_points));
  auto result = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(num_points), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result.get(i, j) = perturbation.get(i, j);
      if (i == j) {
        result.get(i, j) += 1.0;
      }
    }
  }
  return result;
}

// Build a constant (per-point identical) symmetric metric from its six unique
// components. The caller is responsible for supplying an SPD matrix.
tnsr::ii<DataVector, 3, Frame::Inertial> make_constant_metric(
    const double xx, const double xy, const double xz, const double yy,
    const double yz, const double zz) {
  auto result = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  result.get(0, 0) = DataVector(face_size, xx);
  result.get(0, 1) = DataVector(face_size, xy);
  result.get(0, 2) = DataVector(face_size, xz);
  result.get(1, 1) = DataVector(face_size, yy);
  result.get(1, 2) = DataVector(face_size, yz);
  result.get(2, 2) = DataVector(face_size, zz);
  return result;
}

// T = B_k^k, the trace of FieldB (an independent component-loop computation).
DataVector trace_field_b(
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& field_b) {
  DataVector result(face_size, 0.0);
  for (size_t k = 0; k < 3; ++k) {
    result += field_b.get(k, k);
  }
  return result;
}

// Q_ij = 0.5 * (g_jk B_i^k + g_ik B_j^k) - (1/3) g_ij T, computed with the
// given conformal metric (an independent component-loop computation).
tnsr::ii<DataVector, 3, Frame::Inertial> compute_q(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& conformal_metric,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& field_b) {
  const DataVector trace = trace_field_b(field_b);
  auto result = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      DataVector value(face_size, 0.0);
      for (size_t k = 0; k < 3; ++k) {
        value += 0.5 * (conformal_metric.get(j, k) * field_b.get(i, k) +
                        conformal_metric.get(i, k) * field_b.get(j, k));
      }
      value -= (1.0 / 3.0) * conformal_metric.get(i, j) * trace;
      result.get(i, j) = value;
    }
  }
  return result;
}

// Bundle of all face fields consumed by dg_boundary_terms.
struct FaceData {
  tnsr::ii<DataVector, 3, Frame::Inertial> conformal_metric;
  Scalar<DataVector> conformal_factor;
  tnsr::ii<DataVector, 3, Frame::Inertial> a_tilde;
  Scalar<DataVector> trace_extrinsic_curvature;
  Scalar<DataVector> theta;
  tnsr::I<DataVector, 3, Frame::Inertial> gamma_hat;
  Scalar<DataVector> lapse;
  tnsr::I<DataVector, 3, Frame::Inertial> shift;
  tnsr::I<DataVector, 3, Frame::Inertial> auxiliary_shift_b;
  tnsr::i<DataVector, 3, Frame::Inertial> field_a;
  tnsr::iJ<DataVector, 3, Frame::Inertial> field_b;
  tnsr::ijj<DataVector, 3, Frame::Inertial> field_d;
  tnsr::i<DataVector, 3, Frame::Inertial> field_p;
  tnsr::i<DataVector, 3, Frame::Inertial> normal_covector;
};

// Fill all fields (except the caller-supplied metric and normal) with random
// values drawn from `dist`.
template <typename Generator>
FaceData make_random_face_data(
    const gsl::not_null<Generator*> gen, std::uniform_real_distribution<> dist,
    tnsr::ii<DataVector, 3, Frame::Inertial> conformal_metric,
    tnsr::i<DataVector, 3, Frame::Inertial> normal_covector) {
  FaceData data{};
  data.conformal_metric = std::move(conformal_metric);
  data.normal_covector = std::move(normal_covector);
  data.conformal_factor = make_with_random_values<Scalar<DataVector>>(
      gen, dist, DataVector(face_size));
  data.a_tilde =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          gen, dist, DataVector(face_size));
  data.trace_extrinsic_curvature = make_with_random_values<Scalar<DataVector>>(
      gen, dist, DataVector(face_size));
  data.theta = make_with_random_values<Scalar<DataVector>>(
      gen, dist, DataVector(face_size));
  data.gamma_hat =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          gen, dist, DataVector(face_size));
  data.lapse = make_with_random_values<Scalar<DataVector>>(
      gen, dist, DataVector(face_size));
  get(data.lapse) = 0.5 + abs(get(data.lapse));
  data.shift = make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
      gen, dist, DataVector(face_size));
  data.auxiliary_shift_b =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          gen, dist, DataVector(face_size));
  data.field_a =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          gen, dist, DataVector(face_size));
  data.field_b =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          gen, dist, DataVector(face_size));
  data.field_d =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          gen, dist, DataVector(face_size));
  data.field_p =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          gen, dist, DataVector(face_size));
  return data;
}

// Bundle of all corrections produced by dg_boundary_terms.
struct Corrections {
  tnsr::ii<DataVector, 3, Frame::Inertial> conformal_metric;
  Scalar<DataVector> conformal_factor;
  tnsr::ii<DataVector, 3, Frame::Inertial> a_tilde;
  Scalar<DataVector> trace_extrinsic_curvature;
  Scalar<DataVector> theta;
  tnsr::I<DataVector, 3, Frame::Inertial> gamma_hat;
  Scalar<DataVector> lapse;
  tnsr::I<DataVector, 3, Frame::Inertial> shift;
  tnsr::I<DataVector, 3, Frame::Inertial> auxiliary_shift_b;
  tnsr::i<DataVector, 3, Frame::Inertial> field_a;
  tnsr::iJ<DataVector, 3, Frame::Inertial> field_b;
  tnsr::ijj<DataVector, 3, Frame::Inertial> field_d;
  tnsr::i<DataVector, 3, Frame::Inertial> field_p;
  tnsr::ii<DataVector, 3, Frame::Inertial> boundary_conformal_metric;
  Scalar<DataVector> boundary_conformal_factor;
  Scalar<DataVector> boundary_lapse;
  tnsr::I<DataVector, 3, Frame::Inertial> boundary_shift;
};

// A nonzero sentinel makes missing writes to zero-valued outputs observable.
Corrections make_corrections(const double value = 1.234) {
  Corrections c{};
  c.conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), value);
  c.conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), value);
  c.a_tilde = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), value);
  c.trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), value);
  c.theta = make_with_value<Scalar<DataVector>>(DataVector(face_size), value);
  c.gamma_hat = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), value);
  c.lapse = make_with_value<Scalar<DataVector>>(DataVector(face_size), value);
  c.shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), value);
  c.auxiliary_shift_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), value);
  c.field_a = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), value);
  c.field_b = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), value);
  c.field_d = make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), value);
  c.field_p = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), value);
  c.boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), value);
  c.boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), value);
  c.boundary_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), value);
  c.boundary_shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), value);
  return c;
}

// Invoke dg_boundary_terms for the requested ForExternalBoundary and return all
// corrections. Works for any correction type with the LaxFriedrichs signature.
template <bool ForExternalBoundary, typename Correction>
Corrections run_boundary_terms(const Correction& correction,
                               const FaceData& interior,
                               const FaceData& exterior,
                               const dg::Formulation formulation) {
  Corrections c = make_corrections();
  correction.template dg_boundary_terms<ForExternalBoundary>(
      make_not_null(&c.conformal_metric), make_not_null(&c.conformal_factor),
      make_not_null(&c.a_tilde), make_not_null(&c.trace_extrinsic_curvature),
      make_not_null(&c.theta), make_not_null(&c.gamma_hat),
      make_not_null(&c.lapse), make_not_null(&c.shift),
      make_not_null(&c.auxiliary_shift_b), make_not_null(&c.field_a),
      make_not_null(&c.field_b), make_not_null(&c.field_d),
      make_not_null(&c.field_p), make_not_null(&c.boundary_conformal_metric),
      make_not_null(&c.boundary_conformal_factor),
      make_not_null(&c.boundary_lapse), make_not_null(&c.boundary_shift),
      interior.conformal_metric, interior.conformal_factor, interior.a_tilde,
      interior.trace_extrinsic_curvature, interior.theta, interior.gamma_hat,
      interior.lapse, interior.shift, interior.auxiliary_shift_b,
      interior.field_a, interior.field_b, interior.field_d, interior.field_p,
      interior.normal_covector, exterior.conformal_metric,
      exterior.conformal_factor, exterior.a_tilde,
      exterior.trace_extrinsic_curvature, exterior.theta, exterior.gamma_hat,
      exterior.lapse, exterior.shift, exterior.auxiliary_shift_b,
      exterior.field_a, exterior.field_b, exterior.field_d, exterior.field_p,
      exterior.normal_covector, formulation);
  return c;
}

// Convenience overload for interior faces (ForExternalBoundary = false).
template <typename Correction>
Corrections run_interior_boundary_terms(const Correction& correction,
                                        const FaceData& interior,
                                        const FaceData& exterior,
                                        const dg::Formulation formulation) {
  return run_boundary_terms<false>(correction, interior, exterior, formulation);
}

void check_corrections_approx_equal(const Corrections& a,
                                    const Corrections& b) {
  CHECK_ITERABLE_APPROX(a.conformal_metric, b.conformal_metric);
  CHECK_ITERABLE_APPROX(a.conformal_factor, b.conformal_factor);
  CHECK_ITERABLE_APPROX(a.a_tilde, b.a_tilde);
  CHECK_ITERABLE_APPROX(a.trace_extrinsic_curvature,
                        b.trace_extrinsic_curvature);
  CHECK_ITERABLE_APPROX(a.theta, b.theta);
  CHECK_ITERABLE_APPROX(a.gamma_hat, b.gamma_hat);
  CHECK_ITERABLE_APPROX(a.lapse, b.lapse);
  CHECK_ITERABLE_APPROX(a.shift, b.shift);
  CHECK_ITERABLE_APPROX(a.auxiliary_shift_b, b.auxiliary_shift_b);
  CHECK_ITERABLE_APPROX(a.field_a, b.field_a);
  CHECK_ITERABLE_APPROX(a.field_b, b.field_b);
  CHECK_ITERABLE_APPROX(a.field_d, b.field_d);
  CHECK_ITERABLE_APPROX(a.field_p, b.field_p);
  CHECK_ITERABLE_APPROX(a.boundary_conformal_metric,
                        b.boundary_conformal_metric);
  CHECK_ITERABLE_APPROX(a.boundary_conformal_factor,
                        b.boundary_conformal_factor);
  CHECK_ITERABLE_APPROX(a.boundary_lapse, b.boundary_lapse);
  CHECK_ITERABLE_APPROX(a.boundary_shift, b.boundary_shift);
}

// Check that all corrections except (ATilde, K, Theta) match LaxFriedrichs, and
// that the three modified corrections equal LaxFriedrichs plus the expected
// experimental delta computed independently from Q and T. eff_tau1 is the
// effective tau1 for the invoked face (0 when the modification is suppressed).
void check_proposed_equals_lax_plus_delta(const Corrections& proposed,
                                          const Corrections& lax,
                                          const FaceData& interior,
                                          const FaceData& exterior,
                                          const double eff_tau1) {
  // Unmodified corrections must be identical to LaxFriedrichs.
  CHECK_ITERABLE_APPROX(proposed.conformal_metric, lax.conformal_metric);
  CHECK_ITERABLE_APPROX(proposed.conformal_factor, lax.conformal_factor);
  CHECK_ITERABLE_APPROX(proposed.gamma_hat, lax.gamma_hat);
  CHECK_ITERABLE_APPROX(proposed.lapse, lax.lapse);
  CHECK_ITERABLE_APPROX(proposed.shift, lax.shift);
  CHECK_ITERABLE_APPROX(proposed.auxiliary_shift_b, lax.auxiliary_shift_b);
  CHECK_ITERABLE_APPROX(proposed.field_a, lax.field_a);
  CHECK_ITERABLE_APPROX(proposed.field_b, lax.field_b);
  CHECK_ITERABLE_APPROX(proposed.field_d, lax.field_d);
  CHECK_ITERABLE_APPROX(proposed.field_p, lax.field_p);
  CHECK_ITERABLE_APPROX(proposed.boundary_conformal_metric,
                        lax.boundary_conformal_metric);
  CHECK_ITERABLE_APPROX(proposed.boundary_conformal_factor,
                        lax.boundary_conformal_factor);
  CHECK_ITERABLE_APPROX(proposed.boundary_lapse, lax.boundary_lapse);
  CHECK_ITERABLE_APPROX(proposed.boundary_shift, lax.boundary_shift);

  const DataVector trace_int = trace_field_b(interior.field_b);
  const DataVector trace_ext = trace_field_b(exterior.field_b);
  const DataVector trace_jump =
      trace_ext / get(exterior.lapse) - trace_int / get(interior.lapse);
  const auto q_int = compute_q(interior.conformal_metric, interior.field_b);
  const auto q_ext = compute_q(exterior.conformal_metric, exterior.field_b);

  // Delta relative to LaxFriedrichs:
  //   K      : +0.5  * eff_tau1 * [T / alpha]
  //   Theta  : +0.25 * eff_tau1 * [T / alpha]
  //   ATilde : +0.5  * eff_tau1 * [Q / alpha]
  auto expected_K = lax.trace_extrinsic_curvature;
  get(expected_K) += 0.5 * eff_tau1 * trace_jump;
  CHECK_ITERABLE_APPROX(proposed.trace_extrinsic_curvature, expected_K);

  auto expected_theta = lax.theta;
  get(expected_theta) += 0.25 * eff_tau1 * trace_jump;
  CHECK_ITERABLE_APPROX(proposed.theta, expected_theta);

  auto expected_a_tilde = lax.a_tilde;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      expected_a_tilde.get(i, j) += 0.5 * eff_tau1 *
                                    (q_ext.get(i, j) / get(exterior.lapse) -
                                     q_int.get(i, j) / get(interior.lapse));
    }
  }
  CHECK_ITERABLE_APPROX(proposed.a_tilde, expected_a_tilde);
}

// ---------------------------------------------------------------------------
// Packaging: identical to LaxFriedrichs (unchanged by ProposedFlux).
// ---------------------------------------------------------------------------
void test_dg_package_data() {
  const auto direction = Direction<3>::upper_xi();
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::ProposedFlux<3> correction(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 2.0);

  const auto conformal_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto conformal_factor = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto a_tilde =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto trace_extrinsic_curvature =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&gen), dist,
                                                  DataVector(face_size));
  const auto theta = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto gamma_hat =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto auxiliary_shift_b =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_a =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_d =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_p =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto normal_covector = make_unit_normal(face_size);

  auto pkg_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto pkg_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_a_tilde = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_gamma_hat = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_auxiliary_shift_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto pkg_field_a = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_field_b = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_field_d = make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_field_p = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_normal_covector =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);

  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;
  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
      std::nullopt;

  const double result = correction.dg_package_data(
      make_not_null(&pkg_conformal_metric),
      make_not_null(&pkg_conformal_factor), make_not_null(&pkg_a_tilde),
      make_not_null(&pkg_trace_extrinsic_curvature), make_not_null(&pkg_theta),
      make_not_null(&pkg_gamma_hat), make_not_null(&pkg_lapse),
      make_not_null(&pkg_shift), make_not_null(&pkg_auxiliary_shift_b),
      make_not_null(&pkg_field_a), make_not_null(&pkg_field_b),
      make_not_null(&pkg_field_d), make_not_null(&pkg_field_p),
      make_not_null(&pkg_normal_covector), conformal_metric, conformal_factor,
      a_tilde, trace_extrinsic_curvature, theta, gamma_hat, lapse, shift,
      auxiliary_shift_b, field_a, field_b, field_d, field_p,
      boundary_conformal_metric, boundary_conformal_factor, boundary_lapse,
      boundary_shift, normal_covector, mesh_velocity, normal_dot_mesh_velocity,
      direction);

  CHECK(result == 0.0);
  CHECK_ITERABLE_APPROX(pkg_conformal_metric, conformal_metric);
  CHECK_ITERABLE_APPROX(pkg_conformal_factor, conformal_factor);
  CHECK_ITERABLE_APPROX(pkg_a_tilde, a_tilde);
  CHECK_ITERABLE_APPROX(pkg_trace_extrinsic_curvature,
                        trace_extrinsic_curvature);
  CHECK_ITERABLE_APPROX(pkg_theta, theta);
  CHECK_ITERABLE_APPROX(pkg_gamma_hat, gamma_hat);
  CHECK_ITERABLE_APPROX(pkg_lapse, lapse);
  CHECK_ITERABLE_APPROX(pkg_shift, shift);
  CHECK_ITERABLE_APPROX(pkg_auxiliary_shift_b, auxiliary_shift_b);
  CHECK_ITERABLE_APPROX(pkg_field_a, field_a);
  CHECK_ITERABLE_APPROX(pkg_field_b, field_b);
  CHECK_ITERABLE_APPROX(pkg_field_d, field_d);
  CHECK_ITERABLE_APPROX(pkg_field_p, field_p);
  CHECK_ITERABLE_APPROX(pkg_normal_covector, normal_covector);
}

void test_dg_auxiliary_package_data() {
  const auto direction = Direction<3>::upper_xi();
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::ProposedFlux<3> correction(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 2.0);

  const auto conformal_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto conformal_factor = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto a_tilde =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto trace_extrinsic_curvature =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&gen), dist,
                                                  DataVector(face_size));
  const auto theta = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto gamma_hat =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto auxiliary_shift_b =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_a =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_d =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_p =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto normal_covector = make_unit_normal(face_size);

  auto pkg_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto pkg_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_normal_covector =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);

  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;
  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
      std::nullopt;

  const double result = correction.dg_auxiliary_package_data(
      make_not_null(&pkg_conformal_metric),
      make_not_null(&pkg_conformal_factor), make_not_null(&pkg_lapse),
      make_not_null(&pkg_shift), make_not_null(&pkg_normal_covector),
      conformal_metric, conformal_factor, a_tilde, trace_extrinsic_curvature,
      theta, gamma_hat, lapse, shift, auxiliary_shift_b, field_a, field_b,
      field_d, field_p, boundary_conformal_metric, boundary_conformal_factor,
      boundary_lapse, boundary_shift, normal_covector, mesh_velocity,
      normal_dot_mesh_velocity, direction);

  CHECK(result == 0.0);
  CHECK_ITERABLE_APPROX(pkg_conformal_metric, conformal_metric);
  CHECK_ITERABLE_APPROX(pkg_conformal_factor, conformal_factor);
  CHECK_ITERABLE_APPROX(pkg_lapse, lapse);
  CHECK_ITERABLE_APPROX(pkg_shift, shift);
  CHECK_ITERABLE_APPROX(pkg_normal_covector, normal_covector);
}

// ---------------------------------------------------------------------------
// Physical boundary terms: independent component-loop expected values,
// including the experimental Q/T modifications for ATilde, K, Theta.
// ---------------------------------------------------------------------------
void test_dg_boundary_terms() {
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::ProposedFlux<3> correction(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  const auto interior = make_random_face_data(
      make_not_null(&gen), dist,
      make_random_conformal_metric(make_not_null(&gen), face_size),
      make_unit_normal(face_size));
  const auto exterior = make_random_face_data(
      make_not_null(&gen), dist,
      make_random_conformal_metric(make_not_null(&gen), face_size),
      make_negative_unit_normal(face_size));

  const auto corr = run_interior_boundary_terms(
      correction, interior, exterior, dg::Formulation::StrongInertial);

  // Zero corrections.
  const auto zero_ii =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto zero_scalar =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto zero_I = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  CHECK_ITERABLE_APPROX(corr.conformal_metric, zero_ii);
  CHECK_ITERABLE_APPROX(corr.conformal_factor, zero_scalar);
  CHECK_ITERABLE_APPROX(corr.lapse, zero_scalar);
  CHECK_ITERABLE_APPROX(corr.shift, zero_I);
  CHECK_ITERABLE_APPROX(corr.boundary_conformal_metric, zero_ii);
  CHECK_ITERABLE_APPROX(corr.boundary_conformal_factor, zero_scalar);
  CHECK_ITERABLE_APPROX(corr.boundary_lapse, zero_scalar);
  CHECK_ITERABLE_APPROX(corr.boundary_shift, zero_I);

  // Precompute inverse conformal metrics and derived quantities per point.
  auto inv_cm_int = make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto inv_cm_ext = make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  for (size_t q = 0; q < face_size; ++q) {
    invert_3x3_symmetric(interior.conformal_metric, &inv_cm_int, q);
    invert_3x3_symmetric(exterior.conformal_metric, &inv_cm_ext, q);
  }

  DataVector n_dot_shift_int(face_size, 0.0);
  DataVector n_dot_shift_ext(face_size, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    n_dot_shift_int += interior.normal_covector.get(i) * interior.shift.get(i);
    n_dot_shift_ext += exterior.normal_covector.get(i) * exterior.shift.get(i);
  }

  auto icm_dot_n_int = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto icm_dot_n_ext = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  for (size_t I = 0; I < 3; ++I) {
    for (size_t j = 0; j < 3; ++j) {
      icm_dot_n_int.get(I) +=
          inv_cm_int.get(I, j) * interior.normal_covector.get(j);
      icm_dot_n_ext.get(I) +=
          inv_cm_ext.get(I, j) * exterior.normal_covector.get(j);
    }
  }

  const DataVector cf_sq_int =
      get(interior.conformal_factor) * get(interior.conformal_factor);
  const DataVector cf_sq_ext =
      get(exterior.conformal_factor) * get(exterior.conformal_factor);

  DataVector gh_dot_n_int(face_size, 0.0);
  DataVector gh_dot_n_ext(face_size, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    gh_dot_n_int += interior.gamma_hat.get(i) * interior.normal_covector.get(i);
    gh_dot_n_ext += exterior.gamma_hat.get(i) * exterior.normal_covector.get(i);
  }

  // --- K boundary correction ---
  auto compute_k_flux_dot_normal =
      [&](const DataVector& n_dot_beta,
          const tnsr::I<DataVector, 3, Frame::Inertial>& icm_n,
          const Scalar<DataVector>& K, const Scalar<DataVector>& alpha,
          const DataVector& phi_sq,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fa,
          const tnsr::II<DataVector, 3, Frame::Inertial>& inv_g,
          const tnsr::ijj<DataVector, 3, Frame::Inertial>& fd,
          const DataVector& gh_n,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fp) {
        DataVector result(face_size, 0.0);
        result -= n_dot_beta * get(K);
        for (size_t I = 0; I < 3; ++I) {
          result += get(alpha) * phi_sq * icm_n.get(I) * fa.get(I);
        }
        for (size_t k = 0; k < 3; ++k) {
          for (size_t I = 0; I < 3; ++I) {
            for (size_t J = 0; J < 3; ++J) {
              result += get(alpha) * phi_sq * inv_g.get(I, J) *
                        fd.get(k, I, J) * icm_n.get(k);
            }
          }
        }
        result -= get(alpha) * phi_sq * gh_n;
        for (size_t I = 0; I < 3; ++I) {
          result -= 4.0 * get(alpha) * phi_sq * icm_n.get(I) * fp.get(I);
        }
        return result;
      };

  const DataVector k_flux_int = compute_k_flux_dot_normal(
      n_dot_shift_int, icm_dot_n_int, interior.trace_extrinsic_curvature,
      interior.lapse, cf_sq_int, interior.field_a, inv_cm_int, interior.field_d,
      gh_dot_n_int, interior.field_p);
  const DataVector k_flux_ext = compute_k_flux_dot_normal(
      n_dot_shift_ext, icm_dot_n_ext, exterior.trace_extrinsic_curvature,
      exterior.lapse, cf_sq_ext, exterior.field_a, inv_cm_ext, exterior.field_d,
      gh_dot_n_ext, exterior.field_p);

  // Experimental jump: [K - T / alpha].
  const DataVector trace_b_int = trace_field_b(interior.field_b);
  const DataVector trace_b_ext = trace_field_b(exterior.field_b);
  auto expected_corr_K =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  get(expected_corr_K) = -0.5 * tau2 * (k_flux_ext + k_flux_int) -
                         0.5 * tau1 *
                             ((get(exterior.trace_extrinsic_curvature) -
                               trace_b_ext / get(exterior.lapse)) -
                              (get(interior.trace_extrinsic_curvature) -
                               trace_b_int / get(interior.lapse)));
  CHECK_ITERABLE_APPROX(corr.trace_extrinsic_curvature, expected_corr_K);

  // --- theta boundary correction ---
  auto compute_theta_flux_dot_normal =
      [&](const DataVector& n_dot_beta, const Scalar<DataVector>& th,
          const Scalar<DataVector>& alpha, const DataVector& phi_sq,
          const tnsr::I<DataVector, 3, Frame::Inertial>& icm_n,
          const tnsr::ijj<DataVector, 3, Frame::Inertial>& fd,
          const DataVector& gh_n,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fp,
          const tnsr::II<DataVector, 3, Frame::Inertial>& inv_g) {
        DataVector result(face_size, 0.0);
        result -= n_dot_beta * get(th);
        for (size_t k = 0; k < 3; ++k) {
          for (size_t I = 0; I < 3; ++I) {
            for (size_t J = 0; J < 3; ++J) {
              result += 0.5 * get(alpha) * phi_sq * inv_g.get(I, J) *
                        fd.get(k, I, J) * icm_n.get(k);
            }
          }
        }
        result -= 0.5 * get(alpha) * phi_sq * gh_n;
        for (size_t I = 0; I < 3; ++I) {
          result -= 0.5 * 4.0 * get(alpha) * phi_sq * icm_n.get(I) * fp.get(I);
        }
        return result;
      };

  const DataVector theta_flux_int = compute_theta_flux_dot_normal(
      n_dot_shift_int, interior.theta, interior.lapse, cf_sq_int, icm_dot_n_int,
      interior.field_d, gh_dot_n_int, interior.field_p, inv_cm_int);
  const DataVector theta_flux_ext = compute_theta_flux_dot_normal(
      n_dot_shift_ext, exterior.theta, exterior.lapse, cf_sq_ext, icm_dot_n_ext,
      exterior.field_d, gh_dot_n_ext, exterior.field_p, inv_cm_ext);

  // Experimental jump: [Theta - T / (2 alpha)].
  auto expected_corr_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  get(expected_corr_theta) =
      -0.5 * tau2 * (theta_flux_ext + theta_flux_int) -
      0.5 * tau1 *
          ((get(exterior.theta) - 0.5 * trace_b_ext / get(exterior.lapse)) -
           (get(interior.theta) - 0.5 * trace_b_int / get(interior.lapse)));
  CHECK_ITERABLE_APPROX(corr.theta, expected_corr_theta);

  // --- a_tilde boundary correction ---
  auto compute_a_tilde_flux =
      [&](const DataVector& n_dot_beta,
          const tnsr::ii<DataVector, 3, Frame::Inertial>& at,
          const Scalar<DataVector>& alpha, const DataVector& phi_sq,
          const tnsr::i<DataVector, 3, Frame::Inertial>& n_cov,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fa,
          const tnsr::ii<DataVector, 3, Frame::Inertial>& cm,
          const tnsr::ijj<DataVector, 3, Frame::Inertial>& fd,
          const tnsr::I<DataVector, 3, Frame::Inertial>& gh,
          const DataVector& gh_n,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fp,
          const tnsr::I<DataVector, 3, Frame::Inertial>& icm_n,
          const tnsr::II<DataVector, 3, Frame::Inertial>& inv_g) {
        tnsr::ii<DataVector, 3, Frame::Inertial> result(face_size, 0.0);
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = i; j < 3; ++j) {
            auto& r = result.get(i, j);
            r -= n_dot_beta * at.get(i, j);
            const DataVector alp_phi2 = get(alpha) * phi_sq;
            r += alp_phi2 * (0.5 * n_cov.get(i) * fa.get(j) +
                             0.5 * n_cov.get(j) * fa.get(i));
            DataVector icm_n_dot_fa(face_size, 0.0);
            for (size_t K = 0; K < 3; ++K) {
              icm_n_dot_fa += icm_n.get(K) * fa.get(K);
            }
            r -= alp_phi2 * cm.get(i, j) * icm_n_dot_fa / 3.0;
            for (size_t K = 0; K < 3; ++K) {
              r += alp_phi2 * icm_n.get(K) * fd.get(K, i, j);
            }
            DataVector trace_term(face_size, 0.0);
            for (size_t K = 0; K < 3; ++K) {
              for (size_t M = 0; M < 3; ++M) {
                for (size_t N = 0; N < 3; ++N) {
                  trace_term +=
                      inv_g.get(M, N) * icm_n.get(K) * fd.get(K, M, N);
                }
              }
            }
            r -= alp_phi2 * cm.get(i, j) * trace_term / 3.0;
            for (size_t K = 0; K < 3; ++K) {
              r -= alp_phi2 * (0.5 * n_cov.get(i) * cm.get(j, K) * gh.get(K) +
                               0.5 * n_cov.get(j) * cm.get(i, K) * gh.get(K));
            }
            r += alp_phi2 * cm.get(i, j) * gh_n / 3.0;
            r -= alp_phi2 * (0.5 * n_cov.get(i) * fp.get(j) +
                             0.5 * n_cov.get(j) * fp.get(i));
            DataVector icm_n_dot_fp(face_size, 0.0);
            for (size_t K = 0; K < 3; ++K) {
              icm_n_dot_fp += icm_n.get(K) * fp.get(K);
            }
            r += alp_phi2 * cm.get(i, j) * icm_n_dot_fp / 3.0;
          }
        }
        return result;
      };

  const auto at_flux_int = compute_a_tilde_flux(
      n_dot_shift_int, interior.a_tilde, interior.lapse, cf_sq_int,
      interior.normal_covector, interior.field_a, interior.conformal_metric,
      interior.field_d, interior.gamma_hat, gh_dot_n_int, interior.field_p,
      icm_dot_n_int, inv_cm_int);
  const auto at_flux_ext = compute_a_tilde_flux(
      n_dot_shift_ext, exterior.a_tilde, exterior.lapse, cf_sq_ext,
      exterior.normal_covector, exterior.field_a, exterior.conformal_metric,
      exterior.field_d, exterior.gamma_hat, gh_dot_n_ext, exterior.field_p,
      icm_dot_n_ext, inv_cm_ext);

  // Experimental jump: [ATilde_ij - Q_ij / alpha].
  const auto q_int = compute_q(interior.conformal_metric, interior.field_b);
  const auto q_ext = compute_q(exterior.conformal_metric, exterior.field_b);
  auto expected_corr_a_tilde =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      expected_corr_a_tilde.get(i, j) =
          -0.5 * tau2 * (at_flux_ext.get(i, j) + at_flux_int.get(i, j)) -
          0.5 * tau1 *
              ((exterior.a_tilde.get(i, j) -
                q_ext.get(i, j) / get(exterior.lapse)) -
               (interior.a_tilde.get(i, j) -
                q_int.get(i, j) / get(interior.lapse)));
    }
  }
  CHECK_ITERABLE_APPROX(corr.a_tilde, expected_corr_a_tilde);

  // --- gamma_hat boundary correction (unmodified) ---
  auto compute_gamma_hat_flux =
      [&](const DataVector& n_dot_beta,
          const tnsr::I<DataVector, 3, Frame::Inertial>& gh,
          const Scalar<DataVector>& alpha,
          const tnsr::I<DataVector, 3, Frame::Inertial>& icm_n,
          const Scalar<DataVector>& trace_K, const Scalar<DataVector>& th,
          const tnsr::iJ<DataVector, 3, Frame::Inertial>& fb,
          const tnsr::II<DataVector, 3, Frame::Inertial>& inv_g,
          const tnsr::i<DataVector, 3, Frame::Inertial>& n_cov) {
        tnsr::I<DataVector, 3, Frame::Inertial> result(face_size, 0.0);
        for (size_t I = 0; I < 3; ++I) {
          auto& r = result.get(I);
          r -= n_dot_beta * gh.get(I);
          r += (4.0 / 3.0) * get(alpha) * icm_n.get(I) * get(trace_K);
          r -= 2.0 * get(alpha) * icm_n.get(I) * get(th);
          for (size_t J = 0; J < 3; ++J) {
            r -= icm_n.get(J) * fb.get(J, I);
          }
          for (size_t j = 0; j < 3; ++j) {
            r -= icm_n.get(I) * fb.get(j, j) / 6.0;
          }
          for (size_t kk = 0; kk < 3; ++kk) {
            for (size_t j = 0; j < 3; ++j) {
              r -= inv_g.get(I, kk) * fb.get(kk, j) * n_cov.get(j) / 6.0;
            }
          }
        }
        return result;
      };

  const auto gh_flux_int = compute_gamma_hat_flux(
      n_dot_shift_int, interior.gamma_hat, interior.lapse, icm_dot_n_int,
      interior.trace_extrinsic_curvature, interior.theta, interior.field_b,
      inv_cm_int, interior.normal_covector);
  const auto gh_flux_ext = compute_gamma_hat_flux(
      n_dot_shift_ext, exterior.gamma_hat, exterior.lapse, icm_dot_n_ext,
      exterior.trace_extrinsic_curvature, exterior.theta, exterior.field_b,
      inv_cm_ext, exterior.normal_covector);

  auto expected_corr_gamma_hat =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t I = 0; I < 3; ++I) {
    expected_corr_gamma_hat.get(I) =
        -0.5 * tau2 * (gh_flux_ext.get(I) + gh_flux_int.get(I)) -
        0.5 * tau1 * (exterior.gamma_hat.get(I) - interior.gamma_hat.get(I));
  }
  CHECK_ITERABLE_APPROX(corr.gamma_hat, expected_corr_gamma_hat);

  // --- auxiliary_shift_b boundary correction (unmodified) ---
  auto expected_corr_b =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t I = 0; I < 3; ++I) {
    expected_corr_b.get(I) =
        -0.5 * tau2 * (gh_flux_ext.get(I) + gh_flux_int.get(I)) -
        0.5 * tau1 *
            (exterior.auxiliary_shift_b.get(I) -
             interior.auxiliary_shift_b.get(I));
  }
  CHECK_ITERABLE_APPROX(corr.auxiliary_shift_b, expected_corr_b);

  // --- field_a boundary correction (unmodified) ---
  auto compute_field_a_flux =
      [&](const DataVector& n_dot_beta,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fa,
          const Scalar<DataVector>& K, const Scalar<DataVector>& th,
          const tnsr::i<DataVector, 3, Frame::Inertial>& n_cov) {
        tnsr::i<DataVector, 3, Frame::Inertial> result(face_size, 0.0);
        for (size_t k = 0; k < 3; ++k) {
          result.get(k) = -n_dot_beta * fa.get(k) +
                          2.0 * n_cov.get(k) * get(K) -
                          4.0 * n_cov.get(k) * get(th);
        }
        return result;
      };
  const auto fa_flux_int = compute_field_a_flux(
      n_dot_shift_int, interior.field_a, interior.trace_extrinsic_curvature,
      interior.theta, interior.normal_covector);
  const auto fa_flux_ext = compute_field_a_flux(
      n_dot_shift_ext, exterior.field_a, exterior.trace_extrinsic_curvature,
      exterior.theta, exterior.normal_covector);
  auto expected_corr_field_a =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t k = 0; k < 3; ++k) {
    expected_corr_field_a.get(k) =
        -0.5 * tau2 * (fa_flux_ext.get(k) + fa_flux_int.get(k)) -
        0.5 * tau1 * (exterior.field_a.get(k) - interior.field_a.get(k));
  }
  CHECK_ITERABLE_APPROX(corr.field_a, expected_corr_field_a);

  // --- field_b boundary correction (unmodified) ---
  constexpr double f_param = Ccz4::fd::System::f;
  auto compute_field_b_flux =
      [&](const DataVector& n_dot_beta,
          const tnsr::iJ<DataVector, 3, Frame::Inertial>& fb,
          const tnsr::I<DataVector, 3, Frame::Inertial>& aux_b,
          const tnsr::i<DataVector, 3, Frame::Inertial>& n_cov) {
        tnsr::iJ<DataVector, 3, Frame::Inertial> result(face_size, 0.0);
        for (size_t k = 0; k < 3; ++k) {
          for (size_t I = 0; I < 3; ++I) {
            if constexpr (Ccz4::fd::System::shifting_shift) {
              result.get(k, I) = -n_dot_beta * fb.get(k, I) -
                                 f_param * n_cov.get(k) * aux_b.get(I);
            } else {
              result.get(k, I) = -f_param * n_cov.get(k) * aux_b.get(I);
            }
          }
        }
        return result;
      };
  const auto fb_flux_int = compute_field_b_flux(
      n_dot_shift_int, interior.field_b, interior.auxiliary_shift_b,
      interior.normal_covector);
  const auto fb_flux_ext = compute_field_b_flux(
      n_dot_shift_ext, exterior.field_b, exterior.auxiliary_shift_b,
      exterior.normal_covector);
  auto expected_corr_field_b =
      make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t k = 0; k < 3; ++k) {
    for (size_t I = 0; I < 3; ++I) {
      expected_corr_field_b.get(k, I) =
          -0.5 * tau2 * (fb_flux_ext.get(k, I) + fb_flux_int.get(k, I)) -
          0.5 * tau1 *
              (exterior.field_b.get(k, I) - interior.field_b.get(k, I));
    }
  }
  CHECK_ITERABLE_APPROX(corr.field_b, expected_corr_field_b);

  // --- field_d boundary correction (unmodified) ---
  auto compute_field_d_flux =
      [&](const DataVector& n_dot_beta,
          const tnsr::ijj<DataVector, 3, Frame::Inertial>& fd,
          const tnsr::ii<DataVector, 3, Frame::Inertial>& cm,
          const tnsr::iJ<DataVector, 3, Frame::Inertial>& fb,
          const tnsr::i<DataVector, 3, Frame::Inertial>& n_cov,
          const Scalar<DataVector>& alpha,
          const tnsr::ii<DataVector, 3, Frame::Inertial>& at) {
        tnsr::ijj<DataVector, 3, Frame::Inertial> result(face_size, 0.0);
        DataVector contracted_fb(face_size, 0.0);
        for (size_t l = 0; l < 3; ++l) {
          contracted_fb += fb.get(l, l);
        }
        for (size_t k = 0; k < 3; ++k) {
          for (size_t i = 0; i < 3; ++i) {
            for (size_t j = i; j < 3; ++j) {
              auto& r = result.get(k, i, j);
              r = -n_dot_beta * fd.get(k, i, j);
              for (size_t l = 0; l < 3; ++l) {
                r -=
                    0.25 * cm.get(l, i) *
                    (n_cov.get(k) * fb.get(j, l) + n_cov.get(j) * fb.get(k, l));
                r -=
                    0.25 * cm.get(l, j) *
                    (n_cov.get(k) * fb.get(i, l) + n_cov.get(i) * fb.get(k, l));
              }
              r += (1.0 / 6.0) * cm.get(i, j) * (n_cov.get(k) * contracted_fb);
              for (size_t l = 0; l < 3; ++l) {
                r += (1.0 / 6.0) * cm.get(i, j) * n_cov.get(l) * fb.get(k, l);
              }
              r += get(alpha) * n_cov.get(k) * at.get(i, j);
            }
          }
        }
        return result;
      };
  const auto fd_flux_int = compute_field_d_flux(
      n_dot_shift_int, interior.field_d, interior.conformal_metric,
      interior.field_b, interior.normal_covector, interior.lapse,
      interior.a_tilde);
  const auto fd_flux_ext = compute_field_d_flux(
      n_dot_shift_ext, exterior.field_d, exterior.conformal_metric,
      exterior.field_b, exterior.normal_covector, exterior.lapse,
      exterior.a_tilde);
  auto expected_corr_field_d =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        expected_corr_field_d.get(k, i, j) =
            -0.5 * tau2 *
                (fd_flux_ext.get(k, i, j) + fd_flux_int.get(k, i, j)) -
            0.5 * tau1 *
                (exterior.field_d.get(k, i, j) - interior.field_d.get(k, i, j));
      }
    }
  }
  CHECK_ITERABLE_APPROX(corr.field_d, expected_corr_field_d);

  // --- field_p boundary correction (unmodified) ---
  auto compute_field_p_flux =
      [&](const DataVector& n_dot_beta,
          const tnsr::i<DataVector, 3, Frame::Inertial>& fp,
          const Scalar<DataVector>& alpha, const Scalar<DataVector>& K,
          const tnsr::iJ<DataVector, 3, Frame::Inertial>& fb,
          const tnsr::i<DataVector, 3, Frame::Inertial>& n_cov) {
        tnsr::i<DataVector, 3, Frame::Inertial> result(face_size, 0.0);
        DataVector contracted_fb(face_size, 0.0);
        for (size_t l = 0; l < 3; ++l) {
          contracted_fb += fb.get(l, l);
        }
        for (size_t k = 0; k < 3; ++k) {
          result.get(k) = -n_dot_beta * fp.get(k) -
                          (get(alpha) / 3.0) * n_cov.get(k) * get(K) +
                          (1.0 / 6.0) * n_cov.get(k) * contracted_fb;
          for (size_t l = 0; l < 3; ++l) {
            result.get(k) += (1.0 / 6.0) * n_cov.get(l) * fb.get(k, l);
          }
        }
        return result;
      };
  const auto fp_flux_int =
      compute_field_p_flux(n_dot_shift_int, interior.field_p, interior.lapse,
                           interior.trace_extrinsic_curvature, interior.field_b,
                           interior.normal_covector);
  const auto fp_flux_ext =
      compute_field_p_flux(n_dot_shift_ext, exterior.field_p, exterior.lapse,
                           exterior.trace_extrinsic_curvature, exterior.field_b,
                           exterior.normal_covector);
  auto expected_corr_field_p =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t k = 0; k < 3; ++k) {
    expected_corr_field_p.get(k) =
        -0.5 * tau2 * (fp_flux_ext.get(k) + fp_flux_int.get(k)) -
        0.5 * tau1 * (exterior.field_p.get(k) - interior.field_p.get(k));
  }
  CHECK_ITERABLE_APPROX(corr.field_p, expected_corr_field_p);
}

// ---------------------------------------------------------------------------
// Auxiliary boundary terms: identical to LaxFriedrichs (unchanged).
// ---------------------------------------------------------------------------
void test_dg_auxiliary_boundary_terms() {
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::ProposedFlux<3> correction(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 2.0);

  const auto conformal_metric_int =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto conformal_factor_int = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse_int = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift_int =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto normal_covector_int = make_unit_normal(face_size);

  const auto conformal_metric_ext =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto conformal_factor_ext = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse_ext = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift_ext =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto normal_covector_ext = make_negative_unit_normal(face_size);

  Corrections corr = make_corrections();
  correction.dg_auxiliary_boundary_terms(
      make_not_null(&corr.conformal_metric),
      make_not_null(&corr.conformal_factor), make_not_null(&corr.a_tilde),
      make_not_null(&corr.trace_extrinsic_curvature),
      make_not_null(&corr.theta), make_not_null(&corr.gamma_hat),
      make_not_null(&corr.lapse), make_not_null(&corr.shift),
      make_not_null(&corr.auxiliary_shift_b), make_not_null(&corr.field_a),
      make_not_null(&corr.field_b), make_not_null(&corr.field_d),
      make_not_null(&corr.field_p),
      make_not_null(&corr.boundary_conformal_metric),
      make_not_null(&corr.boundary_conformal_factor),
      make_not_null(&corr.boundary_lapse), make_not_null(&corr.boundary_shift),
      conformal_metric_int, conformal_factor_int, lapse_int, shift_int,
      normal_covector_int, conformal_metric_ext, conformal_factor_ext,
      lapse_ext, shift_ext, normal_covector_ext,
      dg::Formulation::StrongInertial);

  const auto zero_ii =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto zero_scalar =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto zero_I = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  CHECK_ITERABLE_APPROX(corr.conformal_metric, zero_ii);
  CHECK_ITERABLE_APPROX(corr.conformal_factor, zero_scalar);
  CHECK_ITERABLE_APPROX(corr.a_tilde, zero_ii);
  CHECK_ITERABLE_APPROX(corr.trace_extrinsic_curvature, zero_scalar);
  CHECK_ITERABLE_APPROX(corr.theta, zero_scalar);
  CHECK_ITERABLE_APPROX(corr.gamma_hat, zero_I);
  CHECK_ITERABLE_APPROX(corr.lapse, zero_scalar);
  CHECK_ITERABLE_APPROX(corr.shift, zero_I);
  CHECK_ITERABLE_APPROX(corr.auxiliary_shift_b, zero_I);

  CHECK_ITERABLE_APPROX(corr.boundary_conformal_metric, zero_ii);
  CHECK_ITERABLE_APPROX(corr.boundary_conformal_factor, zero_scalar);
  CHECK_ITERABLE_APPROX(corr.boundary_lapse, zero_scalar);
  CHECK_ITERABLE_APPROX(corr.boundary_shift, zero_I);

  const DataVector log_lapse_int = log(get(lapse_int));
  const DataVector log_lapse_ext = log(get(lapse_ext));
  const DataVector log_cf_int = log(get(conformal_factor_int));
  const DataVector log_cf_ext = log(get(conformal_factor_ext));

  auto expected_corr_field_a =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    expected_corr_field_a.get(i) =
        0.5 * (log_lapse_int * normal_covector_int.get(i) +
               log_lapse_ext * normal_covector_ext.get(i));
  }
  CHECK_ITERABLE_APPROX(corr.field_a, expected_corr_field_a);

  auto expected_corr_field_b =
      make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t J = 0; J < 3; ++J) {
      expected_corr_field_b.get(i, J) =
          0.5 * (shift_int.get(J) * normal_covector_int.get(i) +
                 shift_ext.get(J) * normal_covector_ext.get(i));
    }
  }
  CHECK_ITERABLE_APPROX(corr.field_b, expected_corr_field_b);

  auto expected_corr_field_d =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = j; k < 3; ++k) {
        expected_corr_field_d.get(i, j, k) =
            0.5 *
            (0.5 * conformal_metric_int.get(j, k) * normal_covector_int.get(i) +
             0.5 * conformal_metric_ext.get(j, k) * normal_covector_ext.get(i));
      }
    }
  }
  CHECK_ITERABLE_APPROX(corr.field_d, expected_corr_field_d);

  auto expected_corr_field_p =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    expected_corr_field_p.get(i) =
        0.5 * (log_cf_int * normal_covector_int.get(i) +
               log_cf_ext * normal_covector_ext.get(i));
  }
  CHECK_ITERABLE_APPROX(corr.field_p, expected_corr_field_p);
}

// ---------------------------------------------------------------------------
// Regression: ProposedFlux differs from LaxFriedrichs ONLY in the three
// physical corrections (ATilde, K, Theta), by the independently computed
// delta. Exercises both dg::Formulation values and the interior / external
// (central and non-central) faces.
// ---------------------------------------------------------------------------
void test_compare_against_lax_friedrichs() {
  const double tau1 = 1.5;
  const double tau2 = 2.3;

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  // Distinct non-diagonal SPD conformal metrics per side (constant per point),
  // strongly non-identity to catch Cartesian or wrong-side lowering.
  const auto metric_int = make_constant_metric(2.0, 0.3, -0.2, 1.5, 0.1, 1.2);
  const auto metric_ext =
      make_constant_metric(1.3, -0.4, 0.25, 1.8, -0.15, 2.1);

  const auto interior = make_random_face_data(
      make_not_null(&gen), dist, metric_int, make_unit_normal(face_size));
  const auto exterior =
      make_random_face_data(make_not_null(&gen), dist, metric_ext,
                            make_negative_unit_normal(face_size));

  for (const auto formulation :
       {dg::Formulation::StrongInertial, dg::Formulation::WeakInertial}) {
    for (const bool use_central : {true, false}) {
      const Ccz4::BoundaryCorrections::ProposedFlux<3> proposed(tau1, tau2,
                                                                use_central);
      const Ccz4::BoundaryCorrections::LaxFriedrichs<3> lax(tau1, tau2,
                                                            use_central);

      // Interior faces: modification active with eff_tau1 = tau1.
      const auto proposed_int = run_interior_boundary_terms(
          proposed, interior, exterior, formulation);
      const auto lax_int =
          run_interior_boundary_terms(lax, interior, exterior, formulation);
      check_proposed_equals_lax_plus_delta(proposed_int, lax_int, interior,
                                           exterior, tau1);

      // External faces: eff_tau1 = 0 when use_central, else tau1.
      const auto proposed_ext =
          run_boundary_terms<true>(proposed, interior, exterior, formulation);
      const auto lax_ext =
          run_boundary_terms<true>(lax, interior, exterior, formulation);
      const double eff_tau1 = use_central ? 0.0 : tau1;
      check_proposed_equals_lax_plus_delta(proposed_ext, lax_ext, interior,
                                           exterior, eff_tau1);

      // When the modification is suppressed (external + central flux), the
      // ProposedFlux external face must match LaxFriedrichs exactly.
      if (use_central) {
        check_corrections_approx_equal(proposed_ext, lax_ext);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Regression: distinct non-diagonal SPD metrics. Q is trace-free with respect
// to each side's own conformal metric, and its coefficients/signs match a hand
// computation. Also confirms that using the wrong side's metric would change
// the result.
// ---------------------------------------------------------------------------
void test_q_trace_free_and_metric_lowering() {
  const auto metric_int = make_constant_metric(2.0, 0.3, -0.2, 1.5, 0.1, 1.2);
  const auto metric_ext =
      make_constant_metric(1.3, -0.4, 0.25, 1.8, -0.15, 2.1);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto field_b_int =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b_ext =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));

  const auto q_int = compute_q(metric_int, field_b_int);
  const auto q_ext = compute_q(metric_ext, field_b_ext);

  // Trace-free with respect to each side's own metric: g^{ij} Q_ij = 0.
  auto inv_int = make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto inv_ext = make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  for (size_t q = 0; q < face_size; ++q) {
    invert_3x3_symmetric(metric_int, &inv_int, q);
    invert_3x3_symmetric(metric_ext, &inv_ext, q);
  }
  DataVector trace_q_int(face_size, 0.0);
  DataVector trace_q_ext(face_size, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      trace_q_int += inv_int.get(i, j) * q_int.get(i, j);
      trace_q_ext += inv_ext.get(i, j) * q_ext.get(i, j);
    }
  }
  const DataVector zero(face_size, 0.0);
  CHECK_ITERABLE_APPROX(trace_q_int, zero);
  CHECK_ITERABLE_APPROX(trace_q_ext, zero);

  // Using the wrong side's metric changes Q (its trace w.r.t. the true metric
  // would then be nonzero), confirming per-side lowering matters.
  const auto q_int_wrong_metric = compute_q(metric_ext, field_b_int);
  DataVector trace_q_wrong(face_size, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      trace_q_wrong += inv_int.get(i, j) * q_int_wrong_metric.get(i, j);
    }
  }
  bool differs = false;
  for (size_t q = 0; q < face_size; ++q) {
    if (std::abs(trace_q_wrong[q]) > 1.0e-10) {
      differs = true;
      break;
    }
  }
  CHECK(differs);

  // Explicit single-point coefficient/sign check for Q_01 with the interior
  // metric at the first grid point.
  const size_t point = 0;
  double t_hand = 0.0;
  for (size_t k = 0; k < 3; ++k) {
    t_hand += field_b_int.get(k, k)[point];
  }
  double q01_hand = 0.0;
  for (size_t k = 0; k < 3; ++k) {
    q01_hand +=
        0.5 * (metric_int.get(1, k)[point] * field_b_int.get(0, k)[point] +
               metric_int.get(0, k)[point] * field_b_int.get(1, k)[point]);
  }
  q01_hand -= (1.0 / 3.0) * metric_int.get(0, 1)[point] * t_hand;
  CHECK(q_int.get(0, 1)[point] == approx(q01_hand));
}

// ---------------------------------------------------------------------------
// Regression: tangential-only, nonsymmetric FieldB with x-normal, to catch
// normal-projection and index-transpose bugs. Uses hand-built data and checks
// the modified corrections against LaxFriedrichs plus the independent delta.
// ---------------------------------------------------------------------------
void test_tangential_nonsymmetric_field_b() {
  const double tau1 = 0.75;
  const double tau2 = 1.0;
  const Ccz4::BoundaryCorrections::ProposedFlux<3> proposed(tau1, tau2);
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> lax(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  // Near-identity SPD metrics keep inverses well-conditioned; a mild off-
  // diagonal ensures lowering mixes components.
  const auto metric_int = make_constant_metric(1.0, 0.2, 0.0, 1.0, 0.1, 1.0);
  const auto metric_ext = make_constant_metric(1.0, -0.15, 0.05, 1.0, 0.2, 1.0);

  auto interior = make_random_face_data(make_not_null(&gen), dist, metric_int,
                                        make_unit_normal(face_size));
  auto exterior = make_random_face_data(make_not_null(&gen), dist, metric_ext,
                                        make_negative_unit_normal(face_size));

  // Overwrite FieldB with tangential-only (lower derivative index in {y,z},
  // i.e. B_0^k = 0), nonsymmetric, nonzero-trace, off-diagonal data. With the
  // x-normal, this discriminates against any bug that projects FieldB on the
  // normal (which would zero the retained lower-index-0 row) or transposes its
  // indices.
  interior.field_b = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  interior.field_b.get(1, 1) = DataVector(face_size, 0.4);   // B_y^y
  interior.field_b.get(2, 2) = DataVector(face_size, -0.9);  // B_z^z
  interior.field_b.get(1, 2) = DataVector(face_size, 0.7);   // B_y^z
  interior.field_b.get(2, 1) = DataVector(face_size, -0.3);  // B_z^y (!= B_y^z)
  interior.field_b.get(1, 0) = DataVector(face_size, 0.5);   // B_y^x
  exterior.field_b = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  exterior.field_b.get(1, 1) = DataVector(face_size, -0.6);  // B_y^y
  exterior.field_b.get(2, 2) = DataVector(face_size, 0.2);   // B_z^z
  exterior.field_b.get(2, 1) = DataVector(face_size, 0.8);   // B_z^y
  exterior.field_b.get(1, 2) =
      DataVector(face_size, -0.25);                          // B_y^z (!= B_z^y)
  exterior.field_b.get(2, 0) = DataVector(face_size, -0.4);  // B_z^x

  const auto proposed_corr = run_interior_boundary_terms(
      proposed, interior, exterior, dg::Formulation::StrongInertial);
  const auto lax_corr = run_interior_boundary_terms(
      lax, interior, exterior, dg::Formulation::StrongInertial);
  check_proposed_equals_lax_plus_delta(proposed_corr, lax_corr, interior,
                                       exterior, tau1);

  // The B jump is nonzero, so the three modified corrections must actually
  // differ from LaxFriedrichs here (guards against an accidental no-op).
  bool a_tilde_differs = false;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      for (size_t q = 0; q < face_size; ++q) {
        if (proposed_corr.a_tilde.get(i, j)[q] !=
            lax_corr.a_tilde.get(i, j)[q]) {
          a_tilde_differs = true;
        }
      }
    }
  }
  CHECK(a_tilde_differs);
  bool k_differs = false;
  for (size_t q = 0; q < face_size; ++q) {
    if (get(proposed_corr.trace_extrinsic_curvature)[q] !=
        get(lax_corr.trace_extrinsic_curvature)[q]) {
      k_differs = true;
    }
  }
  CHECK(k_differs);
}

// ---------------------------------------------------------------------------
// Regression: tau1 = 0 makes ProposedFlux identical to LaxFriedrichs (the
// modification lives entirely in the tau1 penalty). Also checks that a range
// of tau1 (including 1 and a larger value) and arbitrary tau2 are handled.
// ---------------------------------------------------------------------------
void test_tau1_zero_identity_and_parameters() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto interior = make_random_face_data(
      make_not_null(&gen), dist,
      make_random_conformal_metric(make_not_null(&gen), face_size),
      make_unit_normal(face_size));
  const auto exterior = make_random_face_data(
      make_not_null(&gen), dist,
      make_random_conformal_metric(make_not_null(&gen), face_size),
      make_negative_unit_normal(face_size));

  // tau1 = 0: exactly LaxFriedrichs for every output and any tau2.
  for (const double tau2 : {1.0, 2.3}) {
    const Ccz4::BoundaryCorrections::ProposedFlux<3> proposed(0.0, tau2);
    const Ccz4::BoundaryCorrections::LaxFriedrichs<3> lax(0.0, tau2);
    const auto proposed_corr = run_interior_boundary_terms(
        proposed, interior, exterior, dg::Formulation::StrongInertial);
    const auto lax_corr = run_interior_boundary_terms(
        lax, interior, exterior, dg::Formulation::StrongInertial);
    check_corrections_approx_equal(proposed_corr, lax_corr);
  }

  // Nonzero tau1 (including 1 and a larger value) with arbitrary tau2: the
  // delta relationship still holds.
  for (const double tau1 : {1.0, 4.5}) {
    for (const double tau2 : {1.0, 2.3}) {
      const Ccz4::BoundaryCorrections::ProposedFlux<3> proposed(tau1, tau2);
      const Ccz4::BoundaryCorrections::LaxFriedrichs<3> lax(tau1, tau2);
      const auto proposed_corr = run_interior_boundary_terms(
          proposed, interior, exterior, dg::Formulation::StrongInertial);
      const auto lax_corr = run_interior_boundary_terms(
          lax, interior, exterior, dg::Formulation::StrongInertial);
      check_proposed_equals_lax_plus_delta(proposed_corr, lax_corr, interior,
                                           exterior, tau1);
    }
  }
}

// Equal FieldB traces with unequal lapses must give a nonzero modified jump.
// Conversely, equal FieldB/alpha must give zero modified physical jumps.
// These cases distinguish side-local division from a face-averaged lapse and
// pin the unit-lapse limit without using the general Q helper as an oracle.
void test_side_local_lapse() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const double tau1 = 0.8;
  const Ccz4::BoundaryCorrections::ProposedFlux<3> proposed(tau1, 0.0);
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> lax(tau1, 0.0);
  auto interior =
      make_random_face_data(make_not_null(&gen), dist,
                            make_constant_metric(1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
                            make_unit_normal(face_size));
  for (auto& component : interior.a_tilde) {
    component = 0.0;
  }
  get(interior.trace_extrinsic_curvature) = 0.0;
  get(interior.theta) = 0.0;
  get(interior.lapse) = 1.0;
  for (auto& component : interior.field_b) {
    component = 0.0;
  }
  interior.field_b.get(0, 0) = 1.0;
  interior.field_b.get(1, 1) = 2.0;
  interior.field_b.get(2, 2) = 3.0;
  interior.field_b.get(1, 0) = 4.0;
  interior.field_b.get(2, 1) = -2.0;
  auto exterior = interior;
  exterior.normal_covector = make_negative_unit_normal(face_size);

  for (const bool unit_lapse : {true, false}) {
    CAPTURE(unit_lapse);
    if (not unit_lapse) {
      for (size_t q = 0; q < face_size; ++q) {
        get(interior.lapse)[q] = 0.5 + 0.01 * static_cast<double>(q);
        get(exterior.lapse)[q] = 1.5 + 0.02 * static_cast<double>(q);
      }
    }
    const auto corr = run_interior_boundary_terms(
        proposed, interior, exterior, dg::Formulation::StrongInertial);
    const DataVector inverse_lapse_jump =
        1.0 / get(exterior.lapse) - 1.0 / get(interior.lapse);
    CHECK_ITERABLE_APPROX(get(corr.trace_extrinsic_curvature),
                          3.0 * tau1 * inverse_lapse_jump);
    CHECK_ITERABLE_APPROX(get(corr.theta), 1.5 * tau1 * inverse_lapse_jump);
    // T=6, Q_xx=-1, Q_xy=2, Q_yz=-1 for the explicit FieldB above.
    CHECK_ITERABLE_APPROX(corr.a_tilde.get(0, 0),
                          -0.5 * tau1 * inverse_lapse_jump);
    CHECK_ITERABLE_APPROX(corr.a_tilde.get(0, 1), tau1 * inverse_lapse_jump);
    CHECK_ITERABLE_APPROX(corr.a_tilde.get(1, 2),
                          -0.5 * tau1 * inverse_lapse_jump);
    check_proposed_equals_lax_plus_delta(
        corr,
        run_interior_boundary_terms(lax, interior, exterior,
                                    dg::Formulation::StrongInertial),
        interior, exterior, tau1);
  }

  for (auto& component : interior.field_b) {
    component *= get(interior.lapse);
  }
  for (auto& component : exterior.field_b) {
    component *= get(exterior.lapse);
  }
  const auto corr = run_interior_boundary_terms(
      proposed, interior, exterior, dg::Formulation::StrongInertial);
  const DataVector zero(face_size, 0.0);
  CHECK_ITERABLE_APPROX(get(corr.trace_extrinsic_curvature), zero);
  CHECK_ITERABLE_APPROX(get(corr.theta), zero);
  for (const auto& component : corr.a_tilde) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
}

// ---------------------------------------------------------------------------
// Regression: FieldB = 0 makes T = 0 and Q = 0, so ProposedFlux recovers
// LaxFriedrichs exactly for every output.
// ---------------------------------------------------------------------------
void test_field_b_zero_recovers_lax_friedrichs() {
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::ProposedFlux<3> proposed(tau1, tau2);
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> lax(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  auto interior = make_random_face_data(
      make_not_null(&gen), dist,
      make_random_conformal_metric(make_not_null(&gen), face_size),
      make_unit_normal(face_size));
  auto exterior = make_random_face_data(
      make_not_null(&gen), dist,
      make_random_conformal_metric(make_not_null(&gen), face_size),
      make_negative_unit_normal(face_size));
  interior.field_b = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  exterior.field_b = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);

  const auto proposed_corr = run_interior_boundary_terms(
      proposed, interior, exterior, dg::Formulation::StrongInertial);
  const auto lax_corr = run_interior_boundary_terms(
      lax, interior, exterior, dg::Formulation::StrongInertial);
  check_corrections_approx_equal(proposed_corr, lax_corr);
}

// ---------------------------------------------------------------------------
// Regression: equal interior/exterior states with opposing normals produce
// zero corrections for every output (consistency of the boundary correction).
// ---------------------------------------------------------------------------
void test_consistency_equal_states() {
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::ProposedFlux<3> proposed(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto shared = make_random_face_data(
      make_not_null(&gen), dist,
      make_random_conformal_metric(make_not_null(&gen), face_size),
      make_unit_normal(face_size));

  // Exterior copies every field but flips the normal.
  auto exterior = shared;
  exterior.normal_covector = make_negative_unit_normal(face_size);

  const auto zero = make_corrections(0.0);
  for (const auto formulation :
       {dg::Formulation::StrongInertial, dg::Formulation::WeakInertial}) {
    check_corrections_approx_equal(
        run_interior_boundary_terms(proposed, shared, exterior, formulation),
        zero);
    check_corrections_approx_equal(
        run_boundary_terms<true>(proposed, shared, exterior, formulation),
        zero);
  }
}

// ---------------------------------------------------------------------------
// Regression: external boundary override. With UseCentralFluxAtBoundary=false
// the external face equals the interior behavior; with =true it suppresses the
// experimental modification and equals central-flux LaxFriedrichs. Default is
// true.
// ---------------------------------------------------------------------------
void test_use_central_flux_at_boundary() {
  const double tau1 = 1.5;
  const double tau2 = 2.3;

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto interior = make_random_face_data(
      make_not_null(&gen), dist,
      make_random_conformal_metric(make_not_null(&gen), face_size),
      make_unit_normal(face_size));
  const auto exterior = make_random_face_data(
      make_not_null(&gen), dist,
      make_random_conformal_metric(make_not_null(&gen), face_size),
      make_negative_unit_normal(face_size));

  // Default UseCentralFluxAtBoundary is true.
  const Ccz4::BoundaryCorrections::ProposedFlux<3> correction_default(tau1,
                                                                      tau2);
  const Ccz4::BoundaryCorrections::ProposedFlux<3> correction_central(
      tau1, tau2, true);
  const Ccz4::BoundaryCorrections::ProposedFlux<3> correction_no_central(
      tau1, tau2, false);

  const auto ext_default = run_boundary_terms<true>(
      correction_default, interior, exterior, dg::Formulation::StrongInertial);
  const auto ext_central = run_boundary_terms<true>(
      correction_central, interior, exterior, dg::Formulation::StrongInertial);
  // Default must match explicit true.
  check_corrections_approx_equal(ext_default, ext_central);

  // With no_central, the external face equals interior behavior.
  const auto ext_no_central =
      run_boundary_terms<true>(correction_no_central, interior, exterior,
                               dg::Formulation::StrongInertial);
  const auto interior_no_central =
      run_interior_boundary_terms(correction_no_central, interior, exterior,
                                  dg::Formulation::StrongInertial);
  check_corrections_approx_equal(ext_no_central, interior_no_central);

  // With central flux, the external face equals central-flux LaxFriedrichs.
  const Ccz4::BoundaryCorrections::LaxFriedrichs<3> lax_central(tau1, tau2,
                                                                true);
  const auto lax_ext_central = run_boundary_terms<true>(
      lax_central, interior, exterior, dg::Formulation::StrongInertial);
  check_corrections_approx_equal(ext_central, lax_ext_central);

  // Central and non-central external K must actually differ (tau1 != 0,
  // tau2 != 1).
  bool differs = false;
  for (size_t q = 0; q < face_size; ++q) {
    if (get(ext_central.trace_extrinsic_curvature)[q] !=
        get(ext_no_central.trace_extrinsic_curvature)[q]) {
      differs = true;
      break;
    }
  }
  CHECK(differs);
}

// ---------------------------------------------------------------------------
// Packaging serialization (mirrors Test_LaxFriedrichs.cpp): the packaged data
// is independent of tau, but this pins that dg_package_data survives a
// serialization round-trip.
// ---------------------------------------------------------------------------
void test_serialization() {
  const Ccz4::BoundaryCorrections::ProposedFlux<3> correction(1.5, 2.3, false);
  const auto deserialized = serialize_and_deserialize(correction);
  const auto direction = Direction<3>::upper_xi();
  const auto normal = make_unit_normal(face_size);

  auto pkg_cm = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_cf = make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_at = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_K = make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_gh = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  auto pkg_shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_b = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_fa = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_fb = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_fd = make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_fp = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  auto pkg_n = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);

  const auto cm = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 1.0);
  const auto cf =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 1.0);
  const auto at = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.5);
  const auto K =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.1);
  const auto theta =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.2);
  const auto gh = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.3);
  const auto lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 1.0);
  const auto shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto b = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto fa = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto fb = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto fd = make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto fp = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto bcm = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const auto bcf =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto blapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto bshift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);

  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;
  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
      std::nullopt;

  const double result_orig = correction.dg_package_data(
      make_not_null(&pkg_cm), make_not_null(&pkg_cf), make_not_null(&pkg_at),
      make_not_null(&pkg_K), make_not_null(&pkg_theta), make_not_null(&pkg_gh),
      make_not_null(&pkg_lapse), make_not_null(&pkg_shift),
      make_not_null(&pkg_b), make_not_null(&pkg_fa), make_not_null(&pkg_fb),
      make_not_null(&pkg_fd), make_not_null(&pkg_fp), make_not_null(&pkg_n), cm,
      cf, at, K, theta, gh, lapse, shift, b, fa, fb, fd, fp, bcm, bcf, blapse,
      bshift, normal, mesh_velocity, normal_dot_mesh_velocity, direction);
  const auto pkg_n_orig = pkg_n;

  for (auto& component : pkg_n) {
    component = 0.0;
  }
  const double result_deser = deserialized.dg_package_data(
      make_not_null(&pkg_cm), make_not_null(&pkg_cf), make_not_null(&pkg_at),
      make_not_null(&pkg_K), make_not_null(&pkg_theta), make_not_null(&pkg_gh),
      make_not_null(&pkg_lapse), make_not_null(&pkg_shift),
      make_not_null(&pkg_b), make_not_null(&pkg_fa), make_not_null(&pkg_fb),
      make_not_null(&pkg_fd), make_not_null(&pkg_fp), make_not_null(&pkg_n), cm,
      cf, at, K, theta, gh, lapse, shift, b, fa, fb, fd, fp, bcm, bcf, blapse,
      bshift, normal, mesh_velocity, normal_dot_mesh_velocity, direction);

  CHECK(result_orig == result_deser);
  CHECK_ITERABLE_APPROX(pkg_n, pkg_n_orig);
}

// ---------------------------------------------------------------------------
// Factory / YAML construction through the actual factory list, get_clone, and
// serialization. Boundary terms are exercised after cloning / serialization
// with nonzero B jumps and nondefault parameters, which packaging alone cannot
// verify (packaging is independent of tau1/tau2).
// ---------------------------------------------------------------------------
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        evolution::BoundaryCorrection,
        Ccz4::BoundaryCorrections::standard_boundary_corrections<3>>>;
  };
};

void test_factory_creation_and_serialization() {
  register_factory_classes_with_charm<Metavariables>();

  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const auto created =
      TestHelpers::test_creation<std::unique_ptr<evolution::BoundaryCorrection>,
                                 Metavariables>(
          "ProposedFlux:\n"
          "  Tau1: 1.5\n"
          "  Tau2: 2.3\n"
          "  UseCentralFluxAtBoundary: false\n");
  REQUIRE(created != nullptr);
  const auto& created_ref =
      dynamic_cast<const Ccz4::BoundaryCorrections::ProposedFlux<3>&>(*created);

  const auto cloned = created->get_clone();
  REQUIRE(cloned != nullptr);
  const auto& cloned_ref =
      dynamic_cast<const Ccz4::BoundaryCorrections::ProposedFlux<3>&>(*cloned);

  const auto serialized = serialize_and_deserialize(created);
  REQUIRE(serialized != nullptr);
  const auto& serialized_ref =
      dynamic_cast<const Ccz4::BoundaryCorrections::ProposedFlux<3>&>(
          *serialized);

  // Reference directly constructed with the same nondefault parameters.
  const Ccz4::BoundaryCorrections::ProposedFlux<3> expected(tau1, tau2, false);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto interior = make_random_face_data(
      make_not_null(&gen), dist,
      make_random_conformal_metric(make_not_null(&gen), face_size),
      make_unit_normal(face_size));
  const auto exterior = make_random_face_data(
      make_not_null(&gen), dist,
      make_random_conformal_metric(make_not_null(&gen), face_size),
      make_negative_unit_normal(face_size));

  // Use the external face (ForExternalBoundary=true) with nonzero B jumps so
  // that both tau1 (via the penalty, including the Q/T modification) and tau2
  // (via the central flux) affect the result. With UseCentralFluxAtBoundary
  // false this uses the deserialized tau1/tau2, which packaging cannot check.
  const auto expected_corr = run_boundary_terms<true>(
      expected, interior, exterior, dg::Formulation::StrongInertial);
  const auto created_corr = run_boundary_terms<true>(
      created_ref, interior, exterior, dg::Formulation::StrongInertial);
  const auto cloned_corr = run_boundary_terms<true>(
      cloned_ref, interior, exterior, dg::Formulation::StrongInertial);
  const auto serialized_corr = run_boundary_terms<true>(
      serialized_ref, interior, exterior, dg::Formulation::StrongInertial);

  check_corrections_approx_equal(created_corr, expected_corr);
  check_corrections_approx_equal(cloned_corr, expected_corr);
  check_corrections_approx_equal(serialized_corr, expected_corr);

  // YAML requires all options explicitly. The true flag must reproduce the
  // C++ constructor's default and suppress the external-face modification.
  const auto created_default =
      TestHelpers::test_creation<std::unique_ptr<evolution::BoundaryCorrection>,
                                 Metavariables>(
          "ProposedFlux:\n"
          "  Tau1: 1.5\n"
          "  Tau2: 2.3\n"
          "  UseCentralFluxAtBoundary: true\n");
  const auto& created_default_ref =
      dynamic_cast<const Ccz4::BoundaryCorrections::ProposedFlux<3>&>(
          *created_default);
  const Ccz4::BoundaryCorrections::ProposedFlux<3> expected_default(tau1, tau2);
  const auto created_default_corr = run_boundary_terms<true>(
      created_default_ref, interior, exterior, dg::Formulation::StrongInertial);
  const auto expected_default_corr = run_boundary_terms<true>(
      expected_default, interior, exterior, dg::Formulation::StrongInertial);
  check_corrections_approx_equal(created_default_corr, expected_default_corr);
}

// ---------------------------------------------------------------------------
// Regression: all packaged components are written (no leftover sNaN), mirroring
// Test_LaxFriedrichs.cpp.
// ---------------------------------------------------------------------------
void test_dg_package_data_all_components_written() {
  using package_tags =
      Ccz4::BoundaryCorrections::ProposedFlux<3>::dg_package_field_tags;
  Variables<package_tags> packaged_vars(face_size);

  const auto direction = Direction<3>::upper_xi();
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::ProposedFlux<3> correction(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 2.0);

  const auto conformal_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto conformal_factor = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto a_tilde =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto trace_extrinsic_curvature =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&gen), dist,
                                                  DataVector(face_size));
  const auto theta = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto gamma_hat =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto auxiliary_shift_b =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_a =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_d =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_p =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto normal_covector = make_unit_normal(face_size);
  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;
  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
      std::nullopt;

  correction.dg_package_data(
      make_not_null(
          &get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::ConformalFactor<DataVector>>(packaged_vars)),
      make_not_null(&get<::Ccz4::Tags::ATilde<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(packaged_vars)),
      make_not_null(&get<::Ccz4::Tags::Theta<DataVector>>(packaged_vars)),
      make_not_null(&get<::Ccz4::Tags::GammaHat<DataVector, 3>>(packaged_vars)),
      make_not_null(&get<gr::Tags::Lapse<DataVector>>(packaged_vars)),
      make_not_null(&get<gr::Tags::Shift<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(packaged_vars)),
      make_not_null(&get<::Ccz4::Tags::FieldA<DataVector, 3>>(packaged_vars)),
      make_not_null(&get<::Ccz4::Tags::FieldB<DataVector, 3>>(packaged_vars)),
      make_not_null(&get<::Ccz4::Tags::FieldD<DataVector, 3>>(packaged_vars)),
      make_not_null(&get<::Ccz4::Tags::FieldP<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::NormalCovector<DataVector, 3>>(packaged_vars)),
      conformal_metric, conformal_factor, a_tilde, trace_extrinsic_curvature,
      theta, gamma_hat, lapse, shift, auxiliary_shift_b, field_a, field_b,
      field_d, field_p, boundary_conformal_metric, boundary_conformal_factor,
      boundary_lapse, boundary_shift, normal_covector, mesh_velocity,
      normal_dot_mesh_velocity, direction);

  for (size_t i = 0; i < packaged_vars.size(); ++i) {
    CAPTURE(i);
    CHECK(not std::isnan(packaged_vars.data()[i]));
  }
}

void test_dg_auxiliary_package_data_all_components_written() {
  using aux_package_tags = Ccz4::BoundaryCorrections::ProposedFlux<
      3>::dg_auxiliary_package_field_tags;
  Variables<aux_package_tags> packaged_vars(face_size);

  const auto direction = Direction<3>::upper_xi();
  const double tau1 = 1.5;
  const double tau2 = 2.3;
  const Ccz4::BoundaryCorrections::ProposedFlux<3> correction(tau1, tau2);

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 2.0);

  const auto conformal_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto conformal_factor = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto a_tilde =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto trace_extrinsic_curvature =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&gen), dist,
                                                  DataVector(face_size));
  const auto theta = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto gamma_hat =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), dist, DataVector(face_size));
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto auxiliary_shift_b =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_a =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_b =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_d =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto field_p =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(face_size));
  const auto boundary_conformal_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto boundary_conformal_factor =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_lapse =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  const auto boundary_shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  const auto normal_covector = make_unit_normal(face_size);
  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;
  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
      std::nullopt;

  correction.dg_auxiliary_package_data(
      make_not_null(
          &get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::ConformalFactor<DataVector>>(packaged_vars)),
      make_not_null(&get<gr::Tags::Lapse<DataVector>>(packaged_vars)),
      make_not_null(&get<gr::Tags::Shift<DataVector, 3>>(packaged_vars)),
      make_not_null(
          &get<::Ccz4::Tags::NormalCovector<DataVector, 3>>(packaged_vars)),
      conformal_metric, conformal_factor, a_tilde, trace_extrinsic_curvature,
      theta, gamma_hat, lapse, shift, auxiliary_shift_b, field_a, field_b,
      field_d, field_p, boundary_conformal_metric, boundary_conformal_factor,
      boundary_lapse, boundary_shift, normal_covector, mesh_velocity,
      normal_dot_mesh_velocity, direction);

  for (size_t i = 0; i < packaged_vars.size(); ++i) {
    CAPTURE(i);
    CHECK(not std::isnan(packaged_vars.data()[i]));
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.BoundaryCorrections.ProposedFlux",
    "[Unit][Evolution]") {
  test_dg_package_data();
  test_dg_auxiliary_package_data();
  test_dg_boundary_terms();
  test_dg_auxiliary_boundary_terms();
  test_compare_against_lax_friedrichs();
  test_q_trace_free_and_metric_lowering();
  test_tangential_nonsymmetric_field_b();
  test_tau1_zero_identity_and_parameters();
  test_side_local_lapse();
  test_field_b_zero_recovers_lax_friedrichs();
  test_consistency_equal_states();
  test_use_central_flux_at_boundary();
  test_serialization();
  test_factory_creation_and_serialization();
  test_dg_package_data_all_components_written();
  test_dg_auxiliary_package_data_all_components_written();
}
