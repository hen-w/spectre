// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::GradientProjection {
/// \cond
inline PotentialVector potentials_at(const Tags::ChartU::type& u,
                                     const size_t point) {
  PotentialVector result{};
  for (size_t a = 0; a < number_of_potentials; ++a) {
    gsl::at(result, a) = u.get(a)[point];
  }
  return result;
}

inline VelocityVector velocities_at(const Tags::ChartW::type& w,
                                    const size_t point) {
  VelocityVector result{};
  for (size_t a = 0; a < number_of_velocities; ++a) {
    gsl::at(result, a) = w.get(a)[point];
  }
  return result;
}

inline GradientVector gradients_at(const Tags::ChartQ::type& q,
                                   const size_t point) {
  GradientVector result{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < number_of_potentials; ++a) {
      gsl::at(gsl::at(result, i), a) = q.get(i, a)[point];
    }
  }
  return result;
}

inline void set_velocities_at(const gsl::not_null<Tags::ChartW::type*> w,
                              const size_t point,
                              const VelocityVector& values) {
  for (size_t a = 0; a < number_of_velocities; ++a) {
    w->get(a)[point] = gsl::at(values, a);
  }
}

inline void set_gradients_at(const gsl::not_null<Tags::ChartQ::type*> q,
                             const size_t point, const GradientVector& values) {
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < number_of_potentials; ++a) {
      q->get(i, a)[point] = gsl::at(gsl::at(values, i), a);
    }
  }
}

inline void set_potentials_at(const gsl::not_null<Tags::ChartU::type*> u,
                              const size_t point,
                              const PotentialVector& values) {
  for (size_t a = 0; a < number_of_potentials; ++a) {
    u->get(a)[point] = gsl::at(values, a);
  }
}
/// \endcond
}  // namespace Ccz4::GradientProjection
