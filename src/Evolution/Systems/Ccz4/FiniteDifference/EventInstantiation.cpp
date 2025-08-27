// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.tpp"

template class Events::ObserveTimeStep<Ccz4::fd::System>;
