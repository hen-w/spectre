# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Ccz4GradientProjection)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BoundaryConditions/DirichletAnalytic.cpp
  BoundaryCorrections/UpwindGradientProjection.cpp
  Chart.cpp
  Filter.cpp
  FinalizeTimeDerivative.cpp
  Initialize.cpp
  Instantiations.cpp
  LocalOperators.cpp
  NormalOperator.cpp
  PrepareChartFields.cpp
  TimeDerivative.cpp
  VolumeTerms.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryConditions/DirichletAnalytic.hpp
  BoundaryCorrections/UpwindGradientProjection.hpp
  Chart.hpp
  Filter.hpp
  FinalizeTimeDerivative.hpp
  Initialize.hpp
  LocalOperators.hpp
  NormalOperator.hpp
  PointwiseFields.hpp
  PrepareChartFields.hpp
  StepActions.hpp
  System.hpp
  Tags.hpp
  TimeDerivative.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blaze
  Ccz4
  DataStructures
  Domain
  DomainStructure
  ErrorHandling
  Evolution
  GeneralRelativity
  GeneralRelativitySolutions
  LinearOperators
  Observer
  Options
  Serialization
  SoCcz4
  Spectral
  Time
  Utilities
  )
