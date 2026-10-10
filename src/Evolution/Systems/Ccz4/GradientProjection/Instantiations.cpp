// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.tpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"

namespace {
using tags_for_filter = Ccz4::GradientProjection::System::variables_tag_list;
}  // namespace

template class CleanHistory<Ccz4::GradientProjection::System>;
template class RecordTimeStepperData<Ccz4::GradientProjection::System>;
template class UpdateU<Ccz4::GradientProjection::System, false>;
template class Filters::runtime::None<3, tags_for_filter>;
template class Events::ObserveTimeStep<Ccz4::GradientProjection::System>;
template struct evolution::dg::Initialization::SpectralFilters<3,
                                                               tags_for_filter>;
