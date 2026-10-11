// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection {
/// The ordinary derivatives of the local transformed curvature \f$a\f$.
using DerivTransformedATilde =
    ::Tags::deriv<Tags::TransformedATilde, tmpl::size_t<3>, Frame::Inertial>;
/// The ordinary derivatives \f$D_iq_j\f$ of the stored gradients.
using DerivPotentialGradients =
    ::Tags::deriv<Tags::PotentialGradients, tmpl::size_t<3>, Frame::Inertial>;

/*!
 * \brief Non-owning references to the chart potentials
 * \f$u=(H,\phi,\alpha,\beta)\f$: the local logarithmic metric and the
 * physical conformal factor, lapse and shift.
 */
struct PotentialFields {
  const Tags::LogConformalMetric::type& log_conformal_metric;
  const Scalar<DataVector>& conformal_factor;
  const Scalar<DataVector>& lapse;
  const tnsr::I<DataVector, 3>& shift;
};

/*!
 * \brief Non-owning references to the chart velocities
 * \f$w=(a,K,\Theta,\hat\Gamma,b)\f$: the local transformed curvature and the
 * physical \f$K,\Theta,\hat\Gamma,b\f$.
 */
struct VelocityFields {
  const Tags::TransformedATilde::type& a;
  const Scalar<DataVector>& trace_extrinsic_curvature;
  const Scalar<DataVector>& theta;
  const tnsr::I<DataVector, 3>& gamma_hat;
  const tnsr::I<DataVector, 3>& b;
};

/// \brief Non-owning references to the ordinary derivatives \f$D_iw\f$ of the
/// chart velocities, see `VelocityFields`.
struct VelocityDerivatives {
  const DerivTransformedATilde::type& a;
  const tnsr::i<DataVector, 3>& trace_extrinsic_curvature;
  const tnsr::i<DataVector, 3>& theta;
  const tnsr::iJ<DataVector, 3>& gamma_hat;
  const tnsr::iJ<DataVector, 3>& b;
};

/// \cond
inline PotentialVector potentials_at(const PotentialFields& u,
                                     const size_t point) {
  PotentialVector result{};
  for (size_t c = 0; c < 5; ++c) {
    gsl::at(result, PotentialIndex::log_metric + c) =
        u.log_conformal_metric.get(c)[point];
  }
  result[PotentialIndex::conformal_factor] = get(u.conformal_factor)[point];
  result[PotentialIndex::lapse] = get(u.lapse)[point];
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, PotentialIndex::shift + i) = u.shift.get(i)[point];
  }
  return result;
}

inline VelocityVector velocities_at(const VelocityFields& w,
                                    const size_t point) {
  VelocityVector result{};
  for (size_t c = 0; c < 5; ++c) {
    gsl::at(result, VelocityIndex::a + c) = w.a.get(c)[point];
  }
  result[VelocityIndex::trace_extrinsic_curvature] =
      get(w.trace_extrinsic_curvature)[point];
  result[VelocityIndex::theta] = get(w.theta)[point];
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, VelocityIndex::gamma_hat + i) = w.gamma_hat.get(i)[point];
    gsl::at(result, VelocityIndex::b + i) = w.b.get(i)[point];
  }
  return result;
}

// D_i w at a point.
inline VelocityVector velocity_derivatives_at(const VelocityDerivatives& d_w,
                                              const size_t i,
                                              const size_t point) {
  VelocityVector result{};
  for (size_t c = 0; c < 5; ++c) {
    gsl::at(result, VelocityIndex::a + c) = d_w.a.get(i, c)[point];
  }
  result[VelocityIndex::trace_extrinsic_curvature] =
      d_w.trace_extrinsic_curvature.get(i)[point];
  result[VelocityIndex::theta] = d_w.theta.get(i)[point];
  for (size_t k = 0; k < 3; ++k) {
    gsl::at(result, VelocityIndex::gamma_hat + k) =
        d_w.gamma_hat.get(i, k)[point];
    gsl::at(result, VelocityIndex::b + k) = d_w.b.get(i, k)[point];
  }
  return result;
}

inline PotentialVector potentials_at(
    const Tags::ChartVector<number_of_potentials>& u, const size_t point) {
  PotentialVector result{};
  for (size_t a = 0; a < number_of_potentials; ++a) {
    gsl::at(result, a) = u.get(a)[point];
  }
  return result;
}

inline VelocityVector velocities_at(
    const Tags::ChartVector<number_of_velocities>& w, const size_t point) {
  VelocityVector result{};
  for (size_t a = 0; a < number_of_velocities; ++a) {
    gsl::at(result, a) = w.get(a)[point];
  }
  return result;
}

inline GradientVector gradients_at(const Tags::PotentialGradients::type& q,
                                   const size_t point) {
  GradientVector result{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < number_of_potentials; ++a) {
      gsl::at(gsl::at(result, i), a) = q.get(i, a)[point];
    }
  }
  return result;
}

// D_i q_j at a point, `[j][A]`.
inline GradientVector gradient_derivatives_at(
    const DerivPotentialGradients::type& d_q, const size_t i,
    const size_t point) {
  GradientVector result{};
  for (size_t j = 0; j < 3; ++j) {
    for (size_t a = 0; a < number_of_potentials; ++a) {
      gsl::at(gsl::at(result, j), a) = d_q.get(i, j, a)[point];
    }
  }
  return result;
}

inline void set_velocities_at(
    const gsl::not_null<Tags::ChartVector<number_of_velocities>*> w,
    const size_t point, const VelocityVector& values) {
  for (size_t a = 0; a < number_of_velocities; ++a) {
    w->get(a)[point] = gsl::at(values, a);
  }
}

inline void set_gradients_at(
    const gsl::not_null<Tags::PotentialGradients::type*> q, const size_t point,
    const GradientVector& values) {
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < number_of_potentials; ++a) {
      q->get(i, a)[point] = gsl::at(gsl::at(values, i), a);
    }
  }
}

inline void set_potentials_at(
    const gsl::not_null<Tags::ChartVector<number_of_potentials>*> u,
    const size_t point, const PotentialVector& values) {
  for (size_t a = 0; a < number_of_potentials; ++a) {
    u->get(a)[point] = gsl::at(values, a);
  }
}
/// \endcond
}  // namespace Ccz4::GradientProjection
