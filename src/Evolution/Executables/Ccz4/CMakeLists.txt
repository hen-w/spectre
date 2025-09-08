# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE EvolveCcz4)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  EvolveCcz4.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Actions
  Ccz4
  Charmxx::main
  DgSubcell
  DiscontinuousGalerkin
  DomainCreators
  Events
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  FiniteDifference
  GeneralRelativity
  GeneralRelativitySolutions
  LinearOperators
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  SoCcz4
  Time
  Utilities
  )
