// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.tpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"

// The runtime filters and spectral-filter initialization of the evolved
// variables (shared with Ccz4::fd::System) are instantiated in
// Evolution/Systems/Ccz4/SpectralFilter.cpp.
template class CleanHistory<Ccz4::GradientProjection::System>;
template class RecordTimeStepperData<Ccz4::GradientProjection::System>;
template class UpdateU<Ccz4::GradientProjection::System, false>;
template class Events::ObserveTimeStep<Ccz4::GradientProjection::System>;
