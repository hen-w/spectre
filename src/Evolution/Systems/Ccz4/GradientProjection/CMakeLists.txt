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
  FiniteDifferenceGradients.cpp
  Initialize.cpp
  Instantiations.cpp
  LocalOperators.cpp
  NeighborPackagedData.cpp
  NormalOperator.cpp
  PrepareChartFields.cpp
  TimeDerivative.cpp
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
  FiniteDifferenceGradients.hpp
  Initialize.hpp
  LocalOperators.hpp
  NeighborPackagedData.hpp
  NormalOperator.hpp
  PointwiseFields.hpp
  PrepareChartFields.hpp
  StepActions.hpp
  SubcellStepActions.hpp
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
  DgSubcell
  Domain
  DomainStructure
  ErrorHandling
  Evolution
  FiniteDifference
  GeneralRelativity
  GeneralRelativitySolutions
  LinearOperators
  Observer
  Options
  Serialization
  SoCcz4
  Spectral
  SphericalHarmonics
  Time
  Utilities
  )
