// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"

// Explicit instantiations needed for linking

template class RecordTimeStepperData<Ccz4::fd::System>;
template class UpdateU<Ccz4::fd::System, false>;
template class UpdateU<Ccz4::fd::System, true>;
