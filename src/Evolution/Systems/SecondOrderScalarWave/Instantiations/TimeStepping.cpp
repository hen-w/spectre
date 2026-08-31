// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SecondOrderScalarWave/System.hpp"
#include "Time/ChangeTimeStepperOrder.tpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"
#include "Utilities/GenerateInstantiations.hpp"

// The system's variables_tag is a tmpl::list (split volume/boundary
// variables), so these instantiate the mutators' list specializations: one
// history per entry, the boundary entry stepped alongside the volume one.

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                           \
  template class ChangeTimeStepperOrder<                                 \
      SecondOrderScalarWave::System<DIM(data)>>;                         \
  template class CleanHistory<SecondOrderScalarWave::System<DIM(data)>>; \
  template class RecordTimeStepperData<                                  \
      SecondOrderScalarWave::System<DIM(data)>>;                         \
  template class UpdateU<SecondOrderScalarWave::System<DIM(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
