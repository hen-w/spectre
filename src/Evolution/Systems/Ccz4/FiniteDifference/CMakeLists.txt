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
  EnforceTracelessDerivConformalMetric.cpp
  EnforceTracelessDtConformalMetric.cpp
  EventInstantiation.cpp
  Filter.cpp
  GhostData.cpp
  RadiationCharacteristics.cpp
  Reconstructor.cpp
  RegisterDerivedWithCharm.cpp
  VolumeTerms.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApplyFilter.hpp
  BoundaryConditionGhostData.hpp
  Characteristics.hpp
  ComputeCrpbcBoundaryModeDt.hpp
  ConstraintCharacteristics.hpp
  Derivatives.hpp
  DummyReconstructor.hpp
  EnforceConstrainedEvolution.hpp
  EnforceTracelessDerivConformalMetric.hpp
  EnforceTracelessDtConformalMetric.hpp
  Filter.hpp
  GhostData.hpp
  LdgTimeDerivative.hpp
  OverwriteExternalBoundaryDt.hpp
  OverwriteExternalBoundaryDtDirichlet.hpp
  RadiationCharacteristics.hpp
  Reconstructor.hpp
  RegisterDerivedWithCharm.hpp
  ResizeTimeDerivatives.hpp
  SetInitialEta.hpp
  SetInitialBoundaryModes.hpp
  SetK0.hpp
  SoTimeDerivative.hpp
  SpatialZ4ConstraintUpCompute.hpp
  System.hpp
  Tags.hpp
  UpdateAuxiliaryVariables.hpp
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
  Options
  Spectral
  Utilities
  INTERFACE
  )

add_subdirectory(../BoundaryConditions ${CMAKE_CURRENT_BINARY_DIR}/BoundaryConditions)
add_subdirectory(../BoundaryCorrections ${CMAKE_CURRENT_BINARY_DIR}/BoundaryCorrections)
