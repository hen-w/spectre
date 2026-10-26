// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// Option tags of the CCZ4 gradient-projection scheme.
namespace Ccz4::GradientProjection::OptionTags {
/// \brief The gain of the top-shell filter of the velocities, see
/// `Ccz4::GradientProjection::Tags::TopShellFilterGain`.
struct TopShellFilterGain {
  using type = double;
  static constexpr Options::String help = {
      "The factor multiplying the highest tensor modal shell of the potential "
      "rates and of GammaHat - b on DG elements before every right-hand side "
      "(at fixed potentials and gradients): 1 leaves the state unfiltered, 0 "
      "removes the shell. On spherical-harmonic shells it multiplies the "
      "highest radial degree within the retained angular space instead, and "
      "1 still removes the angular content outside that space (see "
      "ShellAngularModesToKill)."};
  static type lower_bound() { return 0.0; }
  static type upper_bound() { return 1.0; }
  using group = ::dg::OptionTags::DiscontinuousGalerkinGroup;
};

/// \brief The number of angular bands removed by the velocity filter on
/// spherical-harmonic shells, see
/// `Ccz4::GradientProjection::Tags::ShellAngularModesToKill`.
struct ShellAngularModesToKill {
  using type = size_t;
  static constexpr Options::String help = {
      "The number of top angular bands removed by the velocity filter on "
      "spherical-harmonic shells: the retained angular space is the range of "
      "the native tensor spherical-harmonic filter with this many removed "
      "bands (0 still removes the tensor content of total degree above L). "
      "It must not exceed L on any shell element and is ignored on other "
      "elements."};
  using group = ::dg::OptionTags::DiscontinuousGalerkinGroup;
};
}  // namespace Ccz4::GradientProjection::OptionTags

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
 * \details Constructed from the physical conformal metric, where it is
 * differentiated or packaged. It is not evolved: within a step it is kept only
 * from `ComputeVolumeTargets` to `FinalizeTimeDerivative`, see
 * `StepChartFields`.
 */
struct LogConformalMetric : db::SimpleTag {
  using type = ChartVector<5>;
};

/*!
 * \brief The transformed trace-free extrinsic curvature
 * \f$a=2\mathcal T_H^{-1}\tilde A\f$ in the basis of `stf_encode`.
 *
 * \details Constructed and kept like `LogConformalMetric`.
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

/*!
 * \brief The local chart fields \f$(H,a)\f$ of the current state.
 *
 * \details `ComputeVolumeTargets` computes them from the physical fields and
 * `FinalizeTimeDerivative` reuses them; the evolved variables do not change in
 * between. Like `RhsTargets`, this buffer is not part of the evolved state or
 * the time-stepper history.
 */
using StepChartFields =
    ::Tags::Variables<tmpl::list<LogConformalMetric, TransformedATilde>>;

/// \brief The `LogMetricChart` of \f$H\f$ at each grid point of the current
/// state, kept like `StepChartFields`.
struct StepCharts : db::SimpleTag {
  using type = std::vector<LogMetricChart>;
};

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

/// \brief The gain \f$g\f$ of the filter \f$F=\Pi+g(I-\Pi)\f$ on the
/// highest tensor modal shell, see
/// `Ccz4::GradientProjection::LocalOperators::apply_filter` and
/// `Ccz4::GradientProjection::filter_velocities`; on spherical-harmonic
/// shells the radial gain of `Ccz4::GradientProjection::GeneralShellFilter`.
struct TopShellFilterGain : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::TopShellFilterGain>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type gain) { return gain; }
};

/// \brief The number of angular bands removed by the velocity filter on
/// spherical-harmonic shells, see
/// `Ccz4::GradientProjection::ShellFilterAngularSpaces`.
struct ShellAngularModesToKill : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::ShellAngularModesToKill>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type modes) { return modes; }
};
}  // namespace Ccz4::GradientProjection::Tags
