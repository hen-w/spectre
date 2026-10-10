// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Filter.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/FinalizeTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/UpdateU.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection {
/*!
 * \brief The actions of one right-hand side evaluation and global time step.
 *
 * \details
 * 1. `PrepareChartFields` overwrites the chart fields from the physical state
 *    and reconstructs \f$q_i=D_iu\f$.
 * 2. `ComputeTimeDerivative` evaluates the volume targets, applies the
 *    external boundary conditions and sends the packaged chart fields: the
 *    single DG exchange of the right-hand side.
 * 3. `ApplyBoundaryCorrectionsToTimeDerivative` receives the neighbor data and
 *    lifts the target corrections and the internal value-jump input.
 * 4. `FinalizeTimeDerivative` fits the potential rates, applies the
 *    compatibility correction and writes the physical rates.
 * 5. `RecordTimeStepperData` records the state and the finalized rates in the
 *    history. Adams-Bashforth restarts from the recorded state, so the state
 *    must not change between here and `UpdateU`; dense observations restore it.
 * 6. `UpdateU` and `CleanHistory` advance the state.
 * 7. `FilterAndEnforceConstraints` filters the advanced physical state and
 *    enforces the algebraic constraints without modifying the history.
 *
 * Only global time stepping is supported.
 */
using step_actions = tmpl::list<
    ::Actions::MutateApply<PrepareChartFields>,
    evolution::dg::Actions::ComputeTimeDerivative<3, System, AllStepChoosers,
                                                  false, false>,
    evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<3, false>,
    ::Actions::MutateApply<FinalizeTimeDerivative>,
    ::Actions::MutateApply<RecordTimeStepperData<System>>,
    evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
    ::Actions::MutateApply<UpdateU<System, false>>,
    ::Actions::MutateApply<CleanHistory<System>>,
    ::Actions::MutateApply<FilterAndEnforceConstraints>>;
}  // namespace Ccz4::GradientProjection
