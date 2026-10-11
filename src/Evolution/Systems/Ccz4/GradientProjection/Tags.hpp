// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection::Tags {
/// A vector of `Size` chart components at each grid point.
template <size_t Size>
using ChartVector =
    Tensor<DataVector, Symmetry<1>,
           index_list<SpatialIndex<Size, UpLo::Up, Frame::NoFrame>>>;

/*!
 * \brief The logarithmic conformal metric \f$H=\frac12\log\tilde\gamma\f$ in
 * the basis of `stf_encode`.
 *
 * \details Only constructed in local temporaries from the physical conformal
 * metric, where it is differentiated or packaged; it is never stored.
 */
struct LogConformalMetric : db::SimpleTag {
  using type = ChartVector<5>;
};

/*!
 * \brief The transformed trace-free extrinsic curvature
 * \f$a=2\mathcal T_H^{-1}\tilde A\f$ in the basis of `stf_encode`.
 *
 * \details Only constructed in local temporaries, see `LogConformalMetric`.
 */
struct TransformedATilde : db::SimpleTag {
  using type = ChartVector<5>;
};

/// The element-local gradients \f$q_i=D_iu\f$ stored after the physical
/// fields, see `Ccz4::fd::Tags::PotentialGradients`.
using PotentialGradients = ::Ccz4::fd::Tags::PotentialGradients;
static_assert(PotentialGradients::type::index_dim(1) == number_of_potentials,
              "The stored gradients must have one component per potential.");

/// \brief The target rate of the chart velocities \f$w\f$, ordered as in
/// `Ccz4::GradientProjection::VelocityIndex`.
struct VelocityTarget : db::SimpleTag {
  using type = ChartVector<number_of_velocities>;
};

/// \brief The target rates of the gradients \f$q_i\f$, `get(i, A)`.
struct GradientTarget : db::SimpleTag {
  using type = PotentialGradients::type;
};

/*!
 * \brief The potential-rate input, ordered as in
 * `Ccz4::GradientProjection::PotentialIndex`: the original potential
 * right-hand side \f$R\f$ plus the lifted internal value jumps, whose
 * mass-weighted mean prescribes the mean of the fitted potential rate.
 */
struct PotentialRateTarget : db::SimpleTag {
  using type = ChartVector<number_of_potentials>;
};

/*!
 * \brief The right-hand-side targets: the corrected velocity rates
 * \f$w_t^*\f$, gradient rates \f$q_{i,t}^*\f$ and the potential-rate input.
 *
 * \details This non-evolved buffer receives the volume targets and all lifted
 * boundary corrections (it is the system's `boundary_correction_target_tag`),
 * and lives from the volume evaluation to `FinalizeTimeDerivative`. It is not
 * part of the evolved state or the time-stepper history.
 */
using RhsTargets = ::Tags::Variables<
    tmpl::list<VelocityTarget, GradientTarget, PotentialRateTarget>>;

/// \brief The chart velocities \f$w\f$ packaged on a face.
struct PackagedVelocities : db::SimpleTag {
  using type = ChartVector<number_of_velocities>;
};

/// \brief The gradients \f$q_i\f$ packaged on a face.
struct PackagedGradients : db::SimpleTag {
  using type = PotentialGradients::type;
};

/// \brief The chart potentials \f$u\f$ packaged on a face.
struct PackagedPotentials : db::SimpleTag {
  using type = ChartVector<number_of_potentials>;
};

/// \brief The static per-element fit and filter operators, see
/// `Ccz4::GradientProjection::LocalOperators`.
struct LocalOperators : db::SimpleTag {
  using type = ::Ccz4::GradientProjection::LocalOperators;
};
}  // namespace Ccz4::GradientProjection::Tags
