# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SoCcz4)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ApplyFilter.cpp
  Characteristics.cpp
  ConstraintCharacteristics.cpp
  Derivatives.cpp
  DummyReconstructor.cpp
  EnforceConstrainedEvolution.cpp
  EventInstantiation.cpp
  Filter.cpp
  GhostData.cpp
  RadiationCharacteristics.cpp
  Reconstructor.cpp
  RegisterDerivedWithCharm.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AddUpperSpatialZ4Constraint.hpp
  ApplyFilter.hpp
  BoundaryConditionGhostData.hpp
  Characteristics.hpp
  ConstraintCharacteristics.hpp
  Derivatives.hpp
  DummyReconstructor.hpp
  EnforceConstrainedEvolution.hpp
  Filter.hpp
  GhostData.hpp
  RadiationCharacteristics.hpp
  Reconstructor.hpp
  RegisterDerivedWithCharm.hpp
  ResizeTimeDerivatives.hpp
  SetInitialEta.hpp
  SetK0.hpp
  SoTimeDerivative.hpp
  System.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Ccz4
  DataStructures
  DomainStructure
  ErrorHandling
  DgSubcell
  FiniteDifference
  GeneralRelativity
  GeneralRelativitySolutions
  Options
  Spectral
  Utilities
  INTERFACE
  )

add_subdirectory(../BoundaryConditions ${CMAKE_CURRENT_BINARY_DIR}/BoundaryConditions)
