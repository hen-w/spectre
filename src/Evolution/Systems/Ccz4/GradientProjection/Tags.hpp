// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"

namespace Ccz4::GradientProjection::Tags {
/*!
 * \brief The chart potentials \f$u\f$, ordered as in
 * `Ccz4::GradientProjection::PotentialIndex`.
 *
 * \details These and the other chart fields are scratch storage in the evolved
 * `Variables`: they are overwritten from the physical fields at the start of
 * every right-hand-side evaluation and are never consumed after being advanced
 * by the time stepper. Their time derivatives hold temporary targets, not
 * physical rates. The tensors have a frame-free index of dimension larger than
 * three and so cannot be observed.
 */
struct ChartU : db::SimpleTag {
  using type = Tensor<
      DataVector, Symmetry<1>,
      index_list<SpatialIndex<number_of_potentials, UpLo::Up, Frame::NoFrame>>>;
};

/// \brief The chart velocities \f$w\f$, ordered as in
/// `Ccz4::GradientProjection::VelocityIndex`. See `ChartU`.
struct ChartW : db::SimpleTag {
  using type = Tensor<
      DataVector, Symmetry<1>,
      index_list<SpatialIndex<number_of_velocities, UpLo::Up, Frame::NoFrame>>>;
};

/*!
 * \brief The element-local ordinary gradients \f$q_i=D_iu\f$ of the chart
 * potentials, `get(i, A)`. See `ChartU`.
 */
struct ChartQ : db::SimpleTag {
  using type = Tensor<
      DataVector, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<number_of_potentials, UpLo::Up, Frame::NoFrame>>>;
};

/// \brief The static per-element fit and filter operators, see
/// `Ccz4::GradientProjection::LocalOperators`.
struct LocalOperators : db::SimpleTag {
  using type = ::Ccz4::GradientProjection::LocalOperators;
};
}  // namespace Ccz4::GradientProjection::Tags
