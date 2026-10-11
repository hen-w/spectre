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
  UnlimitedDeg4Prim.cpp
  EnforceConstrainedEvolution.cpp
  EnforceTracelessDtConformalMetric.cpp
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
  ApplyFilter.hpp
  Characteristics.hpp
  ConstraintCharacteristics.hpp
  Derivatives.hpp
  DetConformalSpatialMetricCompute.hpp
  ReconstructWork.hpp
  UnlimitedDeg4Prim.hpp
  EnforceConstrainedEvolution.hpp
  EnforceTracelessDtConformalMetric.hpp
  Filter.hpp
  ConstraintEnergyCompute.hpp
  GhostData.hpp
  HamiltonianConstraintCompute.hpp
  MomentumConstraintCompute.hpp
  RadiationCharacteristics.hpp
  Reconstructor.hpp
  RegisterDerivedWithCharm.hpp
  ResizeTimeDerivatives.hpp
  SetInitialEta.hpp
  SetK0.hpp
  SoTimeDerivative.hpp
  SpatialZ4ConstraintUpCompute.hpp
  System.hpp
  Tags.hpp
  TraceATildeCompute.hpp
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
