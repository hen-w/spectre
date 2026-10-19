// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/AuxiliaryGhostData.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Filter.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/FinalizeTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TimeDerivative.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/UpdateU.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection {
namespace detail {
template <typename SendAuxiliaryGhostData, typename DrainAuxiliaryGhostData>
using step_actions = tmpl::flatten<tmpl::list<
    SendAuxiliaryGhostData, ::Actions::MutateApply<ComputeVolumeTargets>,
    evolution::dg::Actions::ComputeTimeDerivative<3, System, AllStepChoosers,
                                                  false, false>,
    evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<3, false>,
    DrainAuxiliaryGhostData, ::Actions::MutateApply<FinalizeTimeDerivative>,
    ::Actions::MutateApply<RecordTimeStepperData<System>>,
    evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
    ::Actions::MutateApply<UpdateU<System, false>>,
    ::Actions::MutateApply<CleanHistory<System>>,
    ::Actions::MutateApply<EnforceAlgebraicConstraints>>>;
}  // namespace detail

/*!
 * \brief The actions of one right-hand side evaluation and global time step.
 *
 * \details
 * 1. `ComputeVolumeTargets` recomputes \f$q_i=D_iu\f$ from the physical state,
 *    filters the velocities (`filter_velocities`, the end-of-step filter of
 *    the scheme applied just before the right-hand side it feeds), evaluates
 *    the volume targets into `Tags::RhsTargets` and zeroes the time
 *    derivatives.
 * 2. `ComputeTimeDerivative` (which skips its generic volume terms) applies
 *    the external boundary conditions and sends the packaged chart fields:
 *    the single DG exchange of the right-hand side.
 * 3. `ApplyBoundaryCorrectionsToTimeDerivative` receives the neighbor data and
 *    lifts the target corrections and the internal value-jump input into
 *    `Tags::RhsTargets`.
 * 4. `FinalizeTimeDerivative` fits the potential rates, applies the
 *    compatibility correction and writes the physical rates.
 * 5. `RecordTimeStepperData` records the state and the finalized rates in the
 *    history. Adams-Bashforth restarts from the recorded state, so the state
 *    must not change between here and `UpdateU`; dense observations restore it.
 * 6. `UpdateU` and `CleanHistory` advance the state.
 * 7. `EnforceAlgebraicConstraints` projects the advanced physical state onto
 *    the algebraic-constraint surface without modifying the history.
 *
 * Only global time stepping is supported.
 */
using step_actions = detail::step_actions<tmpl::list<>, tmpl::list<>>;

/*!
 * \brief The actions of one right-hand side evaluation and global time step
 * of an element using DG in the DG-subcell scheme.
 *
 * \details The same as `step_actions`, except that the element first sends
 * its ghost data to the auxiliary inbox
 * (`evolution::dg::Actions::SendAuxiliaryGhostData`), from which its
 * finite-difference neighbors compute their stored gradients, and discards
 * the auxiliary ghost data it receives after its boundary corrections
 * (`evolution::dg::Actions::DrainAuxiliaryGhostData`).
 */
using dg_subcell_step_actions = detail::step_actions<
    evolution::dg::Actions::SendAuxiliaryGhostData<3, System, false>,
    evolution::dg::Actions::DrainAuxiliaryGhostData<3, false>>;
}  // namespace Ccz4::GradientProjection
