// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/Actions/TakeTimeStep.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ApplyFilter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceConstrainedEvolution.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceTracelessDtConformalMetric.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/GhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SoTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/FiniteDifferenceGradients.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/StepActions.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/UpdateU.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection {
/*!
 * \brief The actions of one right-hand side evaluation and global time step
 * of the DG-subcell scheme, on DG element arrays.
 *
 * \details The numerical method of each element (fixed at initialization) is
 * selected by `evolution::dg::subcell::Actions::SelectNumericalMethod`:
 *
 * - Elements using DG run `dg_subcell_step_actions`.
 * - Elements using finite differences enforce the algebraic constraints,
 *   exchange the physical fields (with zero gradients) through the auxiliary
 *   inbox, compute the gradients for their DG neighbors
 *   (`Actions::ComputeFiniteDifferenceGradients`), exchange the physical
 *   fields and the gradients, and take the finite-difference step with
 *   `Ccz4::fd::SoTimeDerivative` and same-time Kreiss-Oliger dissipation.
 *
 * Every element sends both exchanges to all its neighbors; how they are
 * consumed depends only on the receiver's method. Each exchange replaces all
 * ghost data of a finite-difference element, so after each exchange the
 * element adds the ghost data of its external boundaries from the boundary
 * conditions (`Ccz4::fd::Actions::SetBoundaryConditionGhostData`), at the
 * current time.
 */
using subcell_step_actions = tmpl::flatten<tmpl::list<
    evolution::dg::subcell::Actions::SelectNumericalMethod,

    ::Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
    dg_subcell_step_actions,
    ::Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,

    ::Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
    ::Actions::MutateApply<::Ccz4::fd::EnforceConstrainedEvolution>,

    // -- First exchange: the physical fields, auxiliary inbox --
    evolution::dg::subcell::Actions::SendDataForReconstruction<
        3, ::Ccz4::fd::GhostVariables, false, true>,
    evolution::dg::subcell::Actions::ReceiveAndSendDataForReconstruction<
        3, ::Ccz4::fd::GhostVariables, false, true>,
    evolution::dg::subcell::Actions::ReceiveDataForReconstruction<3, true>,
    ::Ccz4::fd::Actions::SetBoundaryConditionGhostData,

    // -- Between the exchanges: the gradients for the DG neighbors --
    Actions::ComputeFiniteDifferenceGradients,

    // -- Second exchange: the physical fields and the gradients --
    evolution::dg::subcell::Actions::SendDataForReconstruction<
        3, ::Ccz4::fd::GhostVariablesPhysical, false>,
    evolution::dg::subcell::Actions::ReceiveAndSendDataForReconstruction<
        3, ::Ccz4::fd::GhostVariablesPhysical, false>,
    evolution::dg::subcell::Actions::ReceiveDataForReconstruction<3>,
    ::Ccz4::fd::Actions::SetBoundaryConditionGhostData,

    ::Actions::Label<
        evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback>,

    // subcell actions
    evolution::dg::subcell::fd::Actions::TakeTimeStep<
        ::Ccz4::fd::SoTimeDerivative>,
    ::Actions::MutateApply<::Ccz4::fd::AddKreissOligerDissipation>,
    ::Actions::MutateApply<::Ccz4::fd::EnforceTracelessDtConformalMetric>,
    ::Actions::MutateApply<RecordTimeStepperData<System>>,
    evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
    ::Actions::MutateApply<UpdateU<System, false>>,
    ::Actions::MutateApply<CleanHistory<System>>,
    ::Actions::Label<evolution::dg::subcell::Actions::Labels::EndOfSolvers>>>;
}  // namespace Ccz4::GradientProjection
