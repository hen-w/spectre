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
#include "Evolution/DiscontinuousGalerkin/InterfaceOrientation.hpp"
#include "Evolution/Systems/Ccz4/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Ccz4/BoundaryCorrections/ParityFlux.hpp"
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

// Tests for Ccz4::BoundaryCorrections::ParityFlux. In addition to packaging
// checks, these verify the structural properties the parity candidate must
// satisfy: consistency (identical states give the central flux with zero
// penalty), single-valuedness/antisymmetry of the implied numerical flux
// (including the feedback term under a prefix flip), the Cartesian parity
// table for the one-sided trace selection, the feedback term (zero at zero
// shift and a hand-computed oblique value), and option plumbing.

namespace {

using Orientation = evolution::dg::InterfaceOrientation;
constexpr size_t face_size = 25;  // 5x5 face

tnsr::i<DataVector, 3, Frame::Inertial> make_axis_normal(const size_t axis,
                                                         const double sign) {
  auto n = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  n.get(axis) = sign;
  return n;
}

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

tnsr::II<DataVector, 3, Frame::Inertial> inverse_metric(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& metric) {
  auto inv = make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  for (size_t q = 0; q < face_size; ++q) {
    invert_3x3_symmetric(metric, &inv, q);
  }
  return inv;
}

template <typename Generator>
tnsr::ii<DataVector, 3, Frame::Inertial> make_random_conformal_metric(
    const gsl::not_null<Generator*> gen) {
  std::uniform_real_distribution<> small_dist(-0.1, 0.1);
  auto perturbation =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          gen, small_dist, DataVector(face_size));
  auto result = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
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

// Reference computation of GammaTilde_h^i = inv^{jk} inv^{il} (D_jkl + D_kjl -
// D_ljk) using the given conformal metric (independent component loop).
tnsr::I<DataVector, 3, Frame::Inertial> compute_gamma_tilde_h(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& conformal_metric,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& field_d) {
  const auto inv = inverse_metric(conformal_metric);
  auto result = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          result.get(i) += inv.get(j, k) * inv.get(i, l) *
                           (field_d.get(j, k, l) + field_d.get(k, j, l) -
                            field_d.get(l, j, k));
        }
      }
    }
  }
  return result;
}

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
  tnsr::I<DataVector, 3, Frame::Inertial> gamma_tilde_h;
  tnsr::i<DataVector, 3, Frame::Inertial> normal_covector;
};

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
  get(data.conformal_factor) = 0.5 + abs(get(data.conformal_factor));
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
  data.gamma_tilde_h =
      compute_gamma_tilde_h(data.conformal_metric, data.field_d);
  return data;
}

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

template <bool ForExternalBoundary>
Corrections run_boundary_terms(
    const Ccz4::BoundaryCorrections::ParityFlux<3>& correction,
    const FaceData& interior, const FaceData& exterior,
    const Orientation orientation) {
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
      interior.gamma_tilde_h, interior.normal_covector,
      exterior.conformal_metric, exterior.conformal_factor, exterior.a_tilde,
      exterior.trace_extrinsic_curvature, exterior.theta, exterior.gamma_hat,
      exterior.lapse, exterior.shift, exterior.auxiliary_shift_b,
      exterior.field_a, exterior.field_b, exterior.field_d, exterior.field_p,
      exterior.gamma_tilde_h, exterior.normal_covector, orientation,
      dg::Formulation::StrongInertial);
  return c;
}

Corrections run_auxiliary_boundary_terms(
    const Ccz4::BoundaryCorrections::ParityFlux<3>& correction,
    const FaceData& interior, const FaceData& exterior,
    const Orientation orientation) {
  Corrections c = make_corrections();
  correction.dg_auxiliary_boundary_terms(
      make_not_null(&c.conformal_metric), make_not_null(&c.conformal_factor),
      make_not_null(&c.a_tilde), make_not_null(&c.trace_extrinsic_curvature),
      make_not_null(&c.theta), make_not_null(&c.gamma_hat),
      make_not_null(&c.lapse), make_not_null(&c.shift),
      make_not_null(&c.auxiliary_shift_b), make_not_null(&c.field_a),
      make_not_null(&c.field_b), make_not_null(&c.field_d),
      make_not_null(&c.field_p), make_not_null(&c.boundary_conformal_metric),
      make_not_null(&c.boundary_conformal_factor),
      make_not_null(&c.boundary_lapse), make_not_null(&c.boundary_shift),
      interior.conformal_metric, interior.conformal_factor, interior.lapse,
      interior.shift, interior.normal_covector, exterior.conformal_metric,
      exterior.conformal_factor, exterior.lapse, exterior.shift,
      exterior.normal_covector, orientation, dg::Formulation::StrongInertial);
  return c;
}

// --- independent component-loop interior fluxes dot normal (full flux,
// including the advective -shift.n w term), adapted from Test_ProposedFlux ---

DataVector k_flux_dot_normal(const FaceData& d) {
  const auto inv = inverse_metric(d.conformal_metric);
  DataVector n_dot_beta(face_size, 0.0);
  DataVector gh_dot_n(face_size, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    n_dot_beta += d.normal_covector.get(i) * d.shift.get(i);
    gh_dot_n += d.gamma_hat.get(i) * d.normal_covector.get(i);
  }
  const DataVector cf_sq = get(d.conformal_factor) * get(d.conformal_factor);
  DataVector icm_n[3] = {DataVector(face_size, 0.0), DataVector(face_size, 0.0),
                         DataVector(face_size, 0.0)};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      icm_n[i] += inv.get(i, j) * d.normal_covector.get(j);
    }
  }
  DataVector result(face_size, 0.0);
  result -= n_dot_beta * get(d.trace_extrinsic_curvature);
  for (size_t i = 0; i < 3; ++i) {
    result += get(d.lapse) * cf_sq * icm_n[i] * d.field_a.get(i);
  }
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        result += get(d.lapse) * cf_sq * inv.get(i, j) *
                  d.field_d.get(k, i, j) * icm_n[k];
      }
    }
  }
  result -= get(d.lapse) * cf_sq * gh_dot_n;
  for (size_t i = 0; i < 3; ++i) {
    result -= 4.0 * get(d.lapse) * cf_sq * icm_n[i] * d.field_p.get(i);
  }
  return result;
}

DataVector theta_flux_dot_normal(const FaceData& d) {
  const auto inv = inverse_metric(d.conformal_metric);
  DataVector n_dot_beta(face_size, 0.0);
  DataVector gh_dot_n(face_size, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    n_dot_beta += d.normal_covector.get(i) * d.shift.get(i);
    gh_dot_n += d.gamma_hat.get(i) * d.normal_covector.get(i);
  }
  const DataVector cf_sq = get(d.conformal_factor) * get(d.conformal_factor);
  DataVector icm_n[3] = {DataVector(face_size, 0.0), DataVector(face_size, 0.0),
                         DataVector(face_size, 0.0)};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      icm_n[i] += inv.get(i, j) * d.normal_covector.get(j);
    }
  }
  DataVector result(face_size, 0.0);
  result -= n_dot_beta * get(d.theta);
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        result += 0.5 * get(d.lapse) * cf_sq * inv.get(i, j) *
                  d.field_d.get(k, i, j) * icm_n[k];
      }
    }
  }
  result -= 0.5 * get(d.lapse) * cf_sq * gh_dot_n;
  for (size_t i = 0; i < 3; ++i) {
    result -= 0.5 * 4.0 * get(d.lapse) * cf_sq * icm_n[i] * d.field_p.get(i);
  }
  return result;
}

tnsr::ii<DataVector, 3, Frame::Inertial> a_tilde_flux_dot_normal(
    const FaceData& d) {
  const auto inv = inverse_metric(d.conformal_metric);
  DataVector n_dot_beta(face_size, 0.0);
  DataVector gh_dot_n(face_size, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    n_dot_beta += d.normal_covector.get(i) * d.shift.get(i);
    gh_dot_n += d.gamma_hat.get(i) * d.normal_covector.get(i);
  }
  const DataVector cf_sq = get(d.conformal_factor) * get(d.conformal_factor);
  DataVector icm_n[3] = {DataVector(face_size, 0.0), DataVector(face_size, 0.0),
                         DataVector(face_size, 0.0)};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      icm_n[i] += inv.get(i, j) * d.normal_covector.get(j);
    }
  }
  auto result = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  const DataVector alp_phi2 = get(d.lapse) * cf_sq;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      auto& r = result.get(i, j);
      r -= n_dot_beta * d.a_tilde.get(i, j);
      r += alp_phi2 * (0.5 * d.normal_covector.get(i) * d.field_a.get(j) +
                       0.5 * d.normal_covector.get(j) * d.field_a.get(i));
      DataVector icm_n_dot_fa(face_size, 0.0);
      for (size_t k = 0; k < 3; ++k) {
        icm_n_dot_fa += icm_n[k] * d.field_a.get(k);
      }
      r -= alp_phi2 * d.conformal_metric.get(i, j) * icm_n_dot_fa / 3.0;
      for (size_t k = 0; k < 3; ++k) {
        r += alp_phi2 * icm_n[k] * d.field_d.get(k, i, j);
      }
      DataVector trace_term(face_size, 0.0);
      for (size_t k = 0; k < 3; ++k) {
        for (size_t mm = 0; mm < 3; ++mm) {
          for (size_t nn = 0; nn < 3; ++nn) {
            trace_term += inv.get(mm, nn) * icm_n[k] * d.field_d.get(k, mm, nn);
          }
        }
      }
      r -= alp_phi2 * d.conformal_metric.get(i, j) * trace_term / 3.0;
      for (size_t k = 0; k < 3; ++k) {
        r -= alp_phi2 * (0.5 * d.normal_covector.get(i) *
                             d.conformal_metric.get(j, k) * d.gamma_hat.get(k) +
                         0.5 * d.normal_covector.get(j) *
                             d.conformal_metric.get(i, k) * d.gamma_hat.get(k));
      }
      r += alp_phi2 * d.conformal_metric.get(i, j) * gh_dot_n / 3.0;
      r -= alp_phi2 * (0.5 * d.normal_covector.get(i) * d.field_p.get(j) +
                       0.5 * d.normal_covector.get(j) * d.field_p.get(i));
      DataVector icm_n_dot_fp(face_size, 0.0);
      for (size_t k = 0; k < 3; ++k) {
        icm_n_dot_fp += icm_n[k] * d.field_p.get(k);
      }
      r += alp_phi2 * d.conformal_metric.get(i, j) * icm_n_dot_fp / 3.0;
    }
  }
  return result;
}

tnsr::I<DataVector, 3, Frame::Inertial> gamma_hat_flux_dot_normal(
    const FaceData& d) {
  const auto inv = inverse_metric(d.conformal_metric);
  DataVector n_dot_beta(face_size, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    n_dot_beta += d.normal_covector.get(i) * d.shift.get(i);
  }
  DataVector icm_n[3] = {DataVector(face_size, 0.0), DataVector(face_size, 0.0),
                         DataVector(face_size, 0.0)};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      icm_n[i] += inv.get(i, j) * d.normal_covector.get(j);
    }
  }
  auto result = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    auto& r = result.get(i);
    r -= n_dot_beta * d.gamma_hat.get(i);
    r += (4.0 / 3.0) * get(d.lapse) * icm_n[i] *
         get(d.trace_extrinsic_curvature);
    r -= 2.0 * get(d.lapse) * icm_n[i] * get(d.theta);
    for (size_t j = 0; j < 3; ++j) {
      r -= icm_n[j] * d.field_b.get(j, i);
    }
    for (size_t j = 0; j < 3; ++j) {
      r -= icm_n[i] * d.field_b.get(j, j) / 6.0;
    }
    for (size_t k = 0; k < 3; ++k) {
      for (size_t j = 0; j < 3; ++j) {
        r -= inv.get(i, k) * d.field_b.get(k, j) * d.normal_covector.get(j) /
             6.0;
      }
    }
  }
  return result;
}

void check_scalar_zero(const Scalar<DataVector>& s) {
  const auto zero =
      make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
  CHECK_ITERABLE_APPROX(s, zero);
}
void check_vector_zero(const tnsr::I<DataVector, 3, Frame::Inertial>& v) {
  const auto zero = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  CHECK_ITERABLE_APPROX(v, zero);
}
void check_covector_zero(const tnsr::i<DataVector, 3, Frame::Inertial>& v) {
  const auto zero = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  CHECK_ITERABLE_APPROX(v, zero);
}
void check_sym_zero(const tnsr::ii<DataVector, 3, Frame::Inertial>& t) {
  const auto zero = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  CHECK_ITERABLE_APPROX(t, zero);
}
void check_iJ_zero(const tnsr::iJ<DataVector, 3, Frame::Inertial>& t) {
  const auto zero = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  CHECK_ITERABLE_APPROX(t, zero);
}
void check_ijj_zero(const tnsr::ijj<DataVector, 3, Frame::Inertial>& t) {
  const auto zero = make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), 0.0);
  CHECK_ITERABLE_APPROX(t, zero);
}

// ---------------------------------------------------------------------------
// (a) packaging
// ---------------------------------------------------------------------------
void test_dg_package_data() {
  const Ccz4::BoundaryCorrections::ParityFlux<3> correction(1.5, 2.3, 1.1);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 2.0);
  const auto data =
      make_random_face_data(make_not_null(&gen), dist,
                            make_random_conformal_metric(make_not_null(&gen)),
                            make_axis_normal(0, 1.0));

  auto pkg =
      make_random_face_data(make_not_null(&gen), dist,
                            make_random_conformal_metric(make_not_null(&gen)),
                            make_axis_normal(0, 1.0));
  auto pkg_gamma_tilde_h =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
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
  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;
  const std::optional<Scalar<DataVector>> normal_dot_mesh_velocity =
      std::nullopt;

  const double result = correction.dg_package_data(
      make_not_null(&pkg.conformal_metric),
      make_not_null(&pkg.conformal_factor), make_not_null(&pkg.a_tilde),
      make_not_null(&pkg.trace_extrinsic_curvature), make_not_null(&pkg.theta),
      make_not_null(&pkg.gamma_hat), make_not_null(&pkg.lapse),
      make_not_null(&pkg.shift), make_not_null(&pkg.auxiliary_shift_b),
      make_not_null(&pkg.field_a), make_not_null(&pkg.field_b),
      make_not_null(&pkg.field_d), make_not_null(&pkg.field_p),
      make_not_null(&pkg_gamma_tilde_h), make_not_null(&pkg.normal_covector),
      data.conformal_metric, data.conformal_factor, data.a_tilde,
      data.trace_extrinsic_curvature, data.theta, data.gamma_hat, data.lapse,
      data.shift, data.auxiliary_shift_b, data.field_a, data.field_b,
      data.field_d, data.field_p, boundary_conformal_metric,
      boundary_conformal_factor, boundary_lapse, boundary_shift,
      data.normal_covector, mesh_velocity, normal_dot_mesh_velocity,
      Direction<3>::upper_xi());

  CHECK(result == 0.0);
  CHECK_ITERABLE_APPROX(pkg.conformal_metric, data.conformal_metric);
  CHECK_ITERABLE_APPROX(pkg.a_tilde, data.a_tilde);
  CHECK_ITERABLE_APPROX(pkg.field_d, data.field_d);
  CHECK_ITERABLE_APPROX(pkg.normal_covector, data.normal_covector);
  const auto expected_gamma_tilde_h =
      compute_gamma_tilde_h(data.conformal_metric, data.field_d);
  CHECK_ITERABLE_APPROX(pkg_gamma_tilde_h, expected_gamma_tilde_h);

  // auxiliary packaging
  auto apkg =
      make_random_face_data(make_not_null(&gen), dist,
                            make_random_conformal_metric(make_not_null(&gen)),
                            make_axis_normal(0, 1.0));
  const double aresult = correction.dg_auxiliary_package_data(
      make_not_null(&apkg.conformal_metric),
      make_not_null(&apkg.conformal_factor), make_not_null(&apkg.lapse),
      make_not_null(&apkg.shift), make_not_null(&apkg.normal_covector),
      data.conformal_metric, data.conformal_factor, data.a_tilde,
      data.trace_extrinsic_curvature, data.theta, data.gamma_hat, data.lapse,
      data.shift, data.auxiliary_shift_b, data.field_a, data.field_b,
      data.field_d, data.field_p, boundary_conformal_metric,
      boundary_conformal_factor, boundary_lapse, boundary_shift,
      data.normal_covector, mesh_velocity, normal_dot_mesh_velocity,
      Direction<3>::upper_xi());
  CHECK(aresult == 0.0);
  CHECK_ITERABLE_APPROX(apkg.conformal_metric, data.conformal_metric);
  CHECK_ITERABLE_APPROX(apkg.lapse, data.lapse);
  CHECK_ITERABLE_APPROX(apkg.shift, data.shift);
  CHECK_ITERABLE_APPROX(apkg.normal_covector, data.normal_covector);
}

// ---------------------------------------------------------------------------
// (b) consistency: identical states and auxiliaries with opposite normals give
// the central flux with zero penalty -- every correction vanishes.
// ---------------------------------------------------------------------------
void test_consistency() {
  const Ccz4::BoundaryCorrections::ParityFlux<3> correction(1.5, 0.7, 1.3);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto interior =
      make_random_face_data(make_not_null(&gen), dist,
                            make_random_conformal_metric(make_not_null(&gen)),
                            make_axis_normal(0, 1.0));
  FaceData exterior = interior;
  exterior.normal_covector = make_axis_normal(0, -1.0);

  for (const auto orientation :
       {Orientation::InteriorIsLower, Orientation::InteriorIsUpper}) {
    const auto corr =
        run_boundary_terms<false>(correction, interior, exterior, orientation);
    check_scalar_zero(corr.trace_extrinsic_curvature);
    check_scalar_zero(corr.theta);
    check_sym_zero(corr.a_tilde);
    check_vector_zero(corr.gamma_hat);
    check_vector_zero(corr.auxiliary_shift_b);
    check_vector_zero(corr.shift);
    check_covector_zero(corr.field_a);
    check_iJ_zero(corr.field_b);
    check_ijj_zero(corr.field_d);
    check_covector_zero(corr.field_p);
    // unconditionally-zero outputs
    check_sym_zero(corr.conformal_metric);
    check_scalar_zero(corr.conformal_factor);
    check_scalar_zero(corr.lapse);

    const auto acorr = run_auxiliary_boundary_terms(correction, interior,
                                                    exterior, orientation);
    check_covector_zero(acorr.field_a);
    check_covector_zero(acorr.field_p);
    check_iJ_zero(acorr.field_b);
    check_ijj_zero(acorr.field_d);
  }
}

// ---------------------------------------------------------------------------
// (c) single-valuedness / antisymmetry. Evaluate from both elements'
// perspectives (swapped data, flipped normals, flipped orientation). The
// implied outward-normal numerical fluxes must be opposite. This exercises the
// feedback term under the prefix flip as well.
// ---------------------------------------------------------------------------
void test_single_valuedness() {
  const Ccz4::BoundaryCorrections::ParityFlux<3> correction(1.5, 0.7, 1.3);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  // Axis-aligned and oblique face normals; the latter exercises the
  // reflection machinery of the physical pass away from the Cartesian case.
  const double inv_sqrt3 = 1.0 / sqrt(3.0);
  auto oblique_normal =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), inv_sqrt3);
  auto oblique_normal_neg =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), -inv_sqrt3);
  const std::array<std::pair<tnsr::i<DataVector, 3, Frame::Inertial>,
                             tnsr::i<DataVector, 3, Frame::Inertial>>,
                   2>
      normal_pairs{
          {{make_axis_normal(0, 1.0), make_axis_normal(0, -1.0)},
           {std::move(oblique_normal), std::move(oblique_normal_neg)}}};
  for (const auto& [normal_p, normal_q] : normal_pairs) {
    auto data_p = make_random_face_data(
        make_not_null(&gen), dist,
        make_random_conformal_metric(make_not_null(&gen)), normal_p);
    auto data_q = make_random_face_data(
        make_not_null(&gen), dist,
        make_random_conformal_metric(make_not_null(&gen)), normal_q);

    // Element A: interior = P (own normal +x), exterior = Q (own normal -x),
    // interior is Lower.
    const auto corr_a = run_boundary_terms<false>(correction, data_p, data_q,
                                                  Orientation::InteriorIsLower);
    // Element B is the neighbor: interior = Q (own normal -x), exterior = P
    // (own normal +x), interior is Upper. Each side keeps its own outward
    // normal.
    const auto corr_b = run_boundary_terms<false>(correction, data_q, data_p,
                                                  Orientation::InteriorIsUpper);

    // Interior fluxes dot each element's own (interior) normal.
    const DataVector fint_k_a = k_flux_dot_normal(data_p);
    const DataVector fint_k_b = k_flux_dot_normal(data_q);
    const DataVector fint_th_a = theta_flux_dot_normal(data_p);
    const DataVector fint_th_b = theta_flux_dot_normal(data_q);
    const auto fint_at_a = a_tilde_flux_dot_normal(data_p);
    const auto fint_at_b = a_tilde_flux_dot_normal(data_q);
    const auto fint_gh_a = gamma_hat_flux_dot_normal(data_p);
    const auto fint_gh_b = gamma_hat_flux_dot_normal(data_q);

    // n . f* = correction + n . f_int must be opposite for the two elements.
    Approx custom = Approx::custom().epsilon(1.0e-11).scale(1.0);
    for (size_t q = 0; q < face_size; ++q) {
      CHECK((get(corr_a.trace_extrinsic_curvature)[q] + fint_k_a[q]) ==
            custom(-(get(corr_b.trace_extrinsic_curvature)[q] + fint_k_b[q])));
      CHECK((get(corr_a.theta)[q] + fint_th_a[q]) ==
            custom(-(get(corr_b.theta)[q] + fint_th_b[q])));
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          CHECK(
              (corr_a.a_tilde.get(i, j)[q] + fint_at_a.get(i, j)[q]) ==
              custom(-(corr_b.a_tilde.get(i, j)[q] + fint_at_b.get(i, j)[q])));
        }
        // gamma_hat and b share the same interior flux (shifting_shift=false).
        CHECK((corr_a.gamma_hat.get(i)[q] + fint_gh_a.get(i)[q]) ==
              custom(-(corr_b.gamma_hat.get(i)[q] + fint_gh_b.get(i)[q])));
        CHECK((corr_a.auxiliary_shift_b.get(i)[q] + fint_gh_a.get(i)[q]) ==
              custom(
                  -(corr_b.auxiliary_shift_b.get(i)[q] + fint_gh_b.get(i)[q])));
        // shift has no interior flux; its correction is antisymmetric.
        CHECK(corr_a.shift.get(i)[q] == custom(-corr_b.shift.get(i)[q]));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// (d) Cartesian reduction: on an n = e_m face the one-sided auxiliary-trace
// selection reproduces the parity table. A component is taken entirely from
// one side exactly when it is reflection-odd across axis m; the aux-pass
// corrections (n_i (w_int - w^*), Tau2 = 1) then vanish for the even
// components and equal n_i (w_int - w_ext) for the odd ones.
// ---------------------------------------------------------------------------
void test_cartesian_parity() {
  // Masks (bit b set => flips under reflection across axis b): vector
  // component j has mask (1<<j); symmetric-tensor component (j,k) has mask
  // (1<<j)^(1<<k); scalars have mask 0. These reproduce the published
  // U_PARITY/V_PARITY tables entry-by-entry.
  const Ccz4::BoundaryCorrections::ParityFlux<3> correction(0.0, 1.0, 1.0);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  for (size_t m = 0; m < 3; ++m) {
    auto interior =
        make_random_face_data(make_not_null(&gen), dist,
                              make_random_conformal_metric(make_not_null(&gen)),
                              make_axis_normal(m, 1.0));
    FaceData exterior =
        make_random_face_data(make_not_null(&gen), dist,
                              make_random_conformal_metric(make_not_null(&gen)),
                              make_axis_normal(m, -1.0));
    const auto corr = run_auxiliary_boundary_terms(
        correction, interior, exterior, Orientation::InteriorIsLower);

    // field_a from ln(lapse): scalar (even) -> correction identically zero.
    check_covector_zero(corr.field_a);
    check_covector_zero(corr.field_p);

    // field_b from shift beta^j: vector. Correction (i,j) nonzero only for
    // i = m, and then only for the reflection-odd component j = m.
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        const unsigned mask = 1u << j;
        const bool odd = (mask & (1u << m)) != 0u;
        DataVector expected(face_size, 0.0);
        if (i == m and odd) {
          expected = interior.shift.get(j) - exterior.shift.get(j);
        }
        CHECK_ITERABLE_APPROX(corr.field_b.get(i, j), expected);
      }
    }

    // field_d from conformal metric gamma_tilde_jk: symmetric tensor. The
    // reconstruction carries 1/2; correction (i,j,k) nonzero only for i = m
    // and the reflection-odd components (exactly one of j,k equals m).
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = j; k < 3; ++k) {
          const unsigned mask = (1u << j) ^ (1u << k);
          const bool odd = (mask & (1u << m)) != 0u;
          DataVector expected(face_size, 0.0);
          if (i == m and odd) {
            expected = 0.5 * (interior.conformal_metric.get(j, k) -
                              exterior.conformal_metric.get(j, k));
          }
          CHECK_ITERABLE_APPROX(corr.field_d.get(i, j, k), expected);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// (e) feedback term. For a simple configuration (flat metric, unit lapse/
// conformal factor, only field_d and shift nonzero) the gamma_hat and b
// corrections reduce to the GammaTilde_h penalty plus the feedback term:
//   corr^i = 0.5 tau1 Delta_K GammaTilde_h^i + feedback^i,
//   feedback^i = -2 (beta_F . nhat_pref) R^i_j Delta_K J^j,
//   J_l = nhat^k nhat^m D_klm.
// At zero shift the feedback vanishes identically.
// ---------------------------------------------------------------------------
void test_feedback() {
  const double tau1 = 1.7;
  const Ccz4::BoundaryCorrections::ParityFlux<3> correction(tau1, 1.0, 1.0);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  const auto flat_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          DataVector(face_size), 0.0);
  auto make_simple =
      [&](const bool with_shift,
          const tnsr::i<DataVector, 3, Frame::Inertial>& normal) {
        FaceData d{};
        d.conformal_metric = flat_metric;
        for (size_t i = 0; i < 3; ++i) {
          d.conformal_metric.get(i, i) = DataVector(face_size, 1.0);
        }
        d.conformal_factor =
            make_with_value<Scalar<DataVector>>(DataVector(face_size), 1.0);
        d.lapse =
            make_with_value<Scalar<DataVector>>(DataVector(face_size), 1.0);
        d.a_tilde = make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
            DataVector(face_size), 0.0);
        d.trace_extrinsic_curvature =
            make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
        d.theta =
            make_with_value<Scalar<DataVector>>(DataVector(face_size), 0.0);
        d.gamma_hat = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
            DataVector(face_size), 0.0);
        d.auxiliary_shift_b =
            make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
                DataVector(face_size), 0.0);
        d.field_a = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
            DataVector(face_size), 0.0);
        d.field_b = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
            DataVector(face_size), 0.0);
        d.field_p = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
            DataVector(face_size), 0.0);
        d.field_d =
            make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
                make_not_null(&gen), dist, DataVector(face_size));
        d.shift = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
            DataVector(face_size), 0.0);
        if (with_shift) {
          d.shift =
              make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
                  make_not_null(&gen), dist, DataVector(face_size));
        }
        d.normal_covector = normal;
        d.gamma_tilde_h = compute_gamma_tilde_h(d.conformal_metric, d.field_d);
        return d;
      };

  // Oblique unit normal (1,1,1)/sqrt(3).
  const double inv_sqrt3 = 1.0 / sqrt(3.0);
  auto oblique = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), inv_sqrt3);
  auto oblique_neg = make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
      DataVector(face_size), -inv_sqrt3);

  for (const bool with_shift : {false, true}) {
    const auto interior = make_simple(with_shift, oblique);
    FaceData exterior = make_simple(with_shift, oblique_neg);

    const auto corr = run_boundary_terms<false>(correction, interior, exterior,
                                                Orientation::InteriorIsLower);

    // Reference: 0.5 tau1 Delta_K GammaTilde_h + feedback.
    std::array<double, 3> nhat{{inv_sqrt3, inv_sqrt3, inv_sqrt3}};
    std::array<std::array<double, 3>, 3> reflection{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        gsl::at(gsl::at(reflection, i), j) =
            (i == j ? 1.0 : 0.0) - 2.0 * gsl::at(nhat, i) * gsl::at(nhat, j);
      }
    }
    DataVector beta_dot_nhat(face_size, 0.0);
    for (size_t i = 0; i < 3; ++i) {
      beta_dot_nhat += 0.5 * (interior.shift.get(i) + exterior.shift.get(i)) *
                       gsl::at(nhat, i);
    }
    // s = +1 (InteriorIsLower).
    const DataVector beta_dot_nhat_pref = beta_dot_nhat;
    auto j_int = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
        DataVector(face_size), 0.0);
    auto j_ext = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
        DataVector(face_size), 0.0);
    for (size_t l = 0; l < 3; ++l) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t mm = 0; mm < 3; ++mm) {
          j_int.get(l) += gsl::at(nhat, k) * gsl::at(nhat, mm) *
                          interior.field_d.get(k, l, mm);
          j_ext.get(l) += gsl::at(nhat, k) * gsl::at(nhat, mm) *
                          exterior.field_d.get(k, l, mm);
        }
      }
    }
    for (size_t i = 0; i < 3; ++i) {
      DataVector reflected_delta_j(face_size, 0.0);
      for (size_t jj = 0; jj < 3; ++jj) {
        reflected_delta_j += gsl::at(gsl::at(reflection, i), jj) *
                             (j_ext.get(jj) - j_int.get(jj));
      }
      const DataVector feedback = -2.0 * beta_dot_nhat_pref * reflected_delta_j;
      // penalty argument is (gamma_hat - Gt) with gamma_hat = 0:
      //   -0.5 tau1 Delta_K(-Gt) = +0.5 tau1 Delta_K Gt
      const DataVector expected =
          0.5 * tau1 *
              (exterior.gamma_tilde_h.get(i) - interior.gamma_tilde_h.get(i)) +
          feedback;
      CHECK_ITERABLE_APPROX(corr.gamma_hat.get(i), expected);
      CHECK_ITERABLE_APPROX(corr.auxiliary_shift_b.get(i), expected);
      if (not with_shift) {
        // feedback must be identically zero; correction is the pure penalty.
        const DataVector pure_penalty =
            0.5 * tau1 *
            (exterior.gamma_tilde_h.get(i) - interior.gamma_tilde_h.get(i));
        CHECK_ITERABLE_APPROX(corr.gamma_hat.get(i), pure_penalty);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// (g) physical-pass Cartesian one-sided selection. With shared coefficients
// (equal metric, lapse, conformal factor on both sides), zero shift, and
// tau1 = 0, the parity combination minus the interior flux reduces, for
// InteriorIsLower, to: (g_ext - g_int) for reflection-even output components
// and exactly zero for reflection-odd ones. This pins the side assignment of
// the physical pass (a globally swapped assignment would give the transposed
// pattern) component by component on an n = e_x face.
// ---------------------------------------------------------------------------
void test_physical_one_sided_selection() {
  const Ccz4::BoundaryCorrections::ParityFlux<3> correction(0.0, 1.0, 1.0);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  auto interior =
      make_random_face_data(make_not_null(&gen), dist,
                            make_random_conformal_metric(make_not_null(&gen)),
                            make_axis_normal(0, 1.0));
  get(interior.lapse) = 1.3;
  get(interior.conformal_factor) = 0.9;
  for (size_t i = 0; i < 3; ++i) {
    interior.shift.get(i) = 0.0;
  }
  FaceData exterior = make_random_face_data(make_not_null(&gen), dist,
                                            interior.conformal_metric,
                                            make_axis_normal(0, -1.0));
  exterior.lapse = interior.lapse;
  exterior.conformal_factor = interior.conformal_factor;
  for (size_t i = 0; i < 3; ++i) {
    exterior.shift.get(i) = 0.0;
  }

  const auto corr = run_boundary_terms<false>(correction, interior, exterior,
                                              Orientation::InteriorIsLower);

  // Reference g's: the loop fluxes evaluated with the interior normal on both
  // sides. Shared coefficients make the face coefficient state equal to the
  // side-local one, and zero shift removes the advective term.
  FaceData ext_ref = exterior;
  ext_ref.normal_covector = interior.normal_covector;
  const DataVector g_k_int = k_flux_dot_normal(interior);
  const DataVector g_k_ext = k_flux_dot_normal(ext_ref);
  const DataVector g_th_int = theta_flux_dot_normal(interior);
  const DataVector g_th_ext = theta_flux_dot_normal(ext_ref);
  const auto g_at_int = a_tilde_flux_dot_normal(interior);
  const auto g_at_ext = a_tilde_flux_dot_normal(ext_ref);
  const auto g_gh_int = gamma_hat_flux_dot_normal(interior);
  const auto g_gh_ext = gamma_hat_flux_dot_normal(ext_ref);

  Approx custom = Approx::custom().epsilon(1.0e-11).scale(1.0);
  for (size_t q = 0; q < face_size; ++q) {
    // scalars: even -> g_ext - g_int.
    CHECK(get(corr.trace_extrinsic_curvature)[q] ==
          custom(g_k_ext[q] - g_k_int[q]));
    CHECK(get(corr.theta)[q] == custom(g_th_ext[q] - g_th_int[q]));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        // ATilde_ij: even iff x appears zero or two times among (i, j).
        const bool even = ((i == 0 ? 1 : 0) + (j == 0 ? 1 : 0)) != 1;
        const double expected =
            even ? g_at_ext.get(i, j)[q] - g_at_int.get(i, j)[q] : 0.0;
        CHECK(corr.a_tilde.get(i, j)[q] == custom(expected));
      }
      // GammaHat^i, b^i: odd for i = x, even otherwise.
      const double expected_vec =
          i == 0 ? 0.0 : g_gh_ext.get(i)[q] - g_gh_int.get(i)[q];
      CHECK(corr.gamma_hat.get(i)[q] == custom(expected_vec));
      CHECK(corr.auxiliary_shift_b.get(i)[q] == custom(expected_vec));
    }
  }
}

// ---------------------------------------------------------------------------
// (h) external boundary, central branch: with UseCentralFluxAtBoundary = true
// and the ExternalBoundary orientation the corrections are the plain central
// flux -(f_int . n_int + f_ext . n_ext)/2 with no penalty, and the shift
// correction vanishes.
// ---------------------------------------------------------------------------
void test_external_central() {
  const Ccz4::BoundaryCorrections::ParityFlux<3> correction(1.5, 0.7, 1.3,
                                                            true);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto interior =
      make_random_face_data(make_not_null(&gen), dist,
                            make_random_conformal_metric(make_not_null(&gen)),
                            make_axis_normal(1, 1.0));
  const auto exterior =
      make_random_face_data(make_not_null(&gen), dist,
                            make_random_conformal_metric(make_not_null(&gen)),
                            make_axis_normal(1, -1.0));
  const auto corr = run_boundary_terms<true>(correction, interior, exterior,
                                             Orientation::ExternalBoundary);

  const DataVector g_k =
      -0.5 * (k_flux_dot_normal(interior) + k_flux_dot_normal(exterior));
  const DataVector g_th = -0.5 * (theta_flux_dot_normal(interior) +
                                  theta_flux_dot_normal(exterior));
  const auto g_at_int = a_tilde_flux_dot_normal(interior);
  const auto g_at_ext = a_tilde_flux_dot_normal(exterior);
  const auto g_gh_int = gamma_hat_flux_dot_normal(interior);
  const auto g_gh_ext = gamma_hat_flux_dot_normal(exterior);

  Approx custom = Approx::custom().epsilon(1.0e-11).scale(1.0);
  for (size_t q = 0; q < face_size; ++q) {
    CHECK(get(corr.trace_extrinsic_curvature)[q] == custom(g_k[q]));
    CHECK(get(corr.theta)[q] == custom(g_th[q]));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        CHECK(corr.a_tilde.get(i, j)[q] ==
              custom(-0.5 * (g_at_int.get(i, j)[q] + g_at_ext.get(i, j)[q])));
      }
      CHECK(corr.gamma_hat.get(i)[q] ==
            custom(-0.5 * (g_gh_int.get(i)[q] + g_gh_ext.get(i)[q])));
      CHECK(corr.shift.get(i)[q] == custom(0.0));
    }
  }
}

// ---------------------------------------------------------------------------
// (f) option plumbing: pup, get_clone, factory creation.
// ---------------------------------------------------------------------------
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        evolution::BoundaryCorrection,
        Ccz4::BoundaryCorrections::standard_boundary_corrections<3>>>;
  };
};

void test_options() {
  register_factory_classes_with_charm<Metavariables>();

  const Ccz4::BoundaryCorrections::ParityFlux<3> correction(1.5, 0.7, 1.3,
                                                            false);
  const auto serialized = serialize_and_deserialize(correction);
  const auto cloned = correction.get_clone();

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto interior =
      make_random_face_data(make_not_null(&gen), dist,
                            make_random_conformal_metric(make_not_null(&gen)),
                            make_axis_normal(0, 1.0));
  const auto exterior =
      make_random_face_data(make_not_null(&gen), dist,
                            make_random_conformal_metric(make_not_null(&gen)),
                            make_axis_normal(0, -1.0));
  const auto corr = run_boundary_terms<false>(correction, interior, exterior,
                                              Orientation::InteriorIsLower);
  const auto corr_serialized = run_boundary_terms<false>(
      serialized, interior, exterior, Orientation::InteriorIsLower);
  CHECK_ITERABLE_APPROX(corr.gamma_hat, corr_serialized.gamma_hat);
  CHECK_ITERABLE_APPROX(corr.a_tilde, corr_serialized.a_tilde);
  CHECK_ITERABLE_APPROX(corr.shift, corr_serialized.shift);

  const auto created =
      TestHelpers::test_creation<std::unique_ptr<evolution::BoundaryCorrection>,
                                 Metavariables>(
          "ParityFlux:\n"
          "  Tau1: 1.5\n"
          "  Tau2: 0.7\n"
          "  ShiftPenalty: 1.3\n"
          "  UseCentralFluxAtBoundary: false\n");
  const auto* const created_parity =
      dynamic_cast<const Ccz4::BoundaryCorrections::ParityFlux<3>*>(
          created.get());
  REQUIRE(created_parity != nullptr);
  const auto corr_created = run_boundary_terms<false>(
      *created_parity, interior, exterior, Orientation::InteriorIsLower);
  CHECK_ITERABLE_APPROX(corr.gamma_hat, corr_created.gamma_hat);

  // YAML requires all options explicitly; ShiftPenalty = 1.0 and
  // UseCentralFluxAtBoundary = true reproduce the C++ constructor defaults.
  const auto created_defaults =
      TestHelpers::test_creation<std::unique_ptr<evolution::BoundaryCorrection>,
                                 Metavariables>(
          "ParityFlux:\n"
          "  Tau1: 1.0\n"
          "  Tau2: 1.0\n"
          "  ShiftPenalty: 1.0\n"
          "  UseCentralFluxAtBoundary: true\n");
  const auto* const created_defaults_parity =
      dynamic_cast<const Ccz4::BoundaryCorrections::ParityFlux<3>*>(
          created_defaults.get());
  REQUIRE(created_defaults_parity != nullptr);
  const Ccz4::BoundaryCorrections::ParityFlux<3> expected_defaults(1.0, 1.0);
  const auto corr_defaults =
      run_boundary_terms<false>(*created_defaults_parity, interior, exterior,
                                Orientation::InteriorIsLower);
  const auto corr_expected_defaults = run_boundary_terms<false>(
      expected_defaults, interior, exterior, Orientation::InteriorIsLower);
  CHECK_ITERABLE_APPROX(corr_defaults.gamma_hat,
                        corr_expected_defaults.gamma_hat);
  CHECK_ITERABLE_APPROX(corr_defaults.shift, corr_expected_defaults.shift);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.BoundaryCorrections.ParityFlux",
                  "[Unit][Evolution]") {
  test_dg_package_data();
  test_consistency();
  test_single_valuedness();
  test_cartesian_parity();
  test_feedback();
  test_physical_one_sided_selection();
  test_external_central();
  test_options();
}
}  // namespace
