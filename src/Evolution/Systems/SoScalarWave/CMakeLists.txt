# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SoScalarWave)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  TimeDerivative.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Actions/OverwriteExternalBoundaryDt.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivative.hpp
  UpdateAuxiliaryVariables.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainBoundaryConditions
  ErrorHandling
  Events
  LinearOperators
  Options
  Utilities
  INTERFACE
  Initialization
  Parallel
  Serialization
  PRIVATE
  WaveEquationSolutions
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(Instantiations)
