# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Ccz4GradientProjection")

set(LIBRARY_SOURCES
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryCorrections/Test_UpwindGradientProjection.cpp
  Test_Chart.cpp
  Test_ErrorControl.cpp
  Test_Filter.cpp
  Test_FinalizeTimeDerivative.cpp
  Test_FiniteDifferenceGradients.cpp
  Test_HybridStepActions.cpp
  Test_Initialize.cpp
  Test_LocalOperators.cpp
  Test_NeighborPackagedData.cpp
  Test_NormalOperator.cpp
  Test_PrepareChartFields.cpp
  Test_StepActions.cpp
  Test_TimeDerivative.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Ccz4
  Ccz4GradientProjection
  ConstraintDamping
  CoordinateMaps
  DataStructures
  DgSubcell
  DiscontinuousGalerkin
  Domain
  DomainBoundaryConditions
  DomainCreators
  DomainStructure
  EventsAndDenseTriggers
  Evolution
  Framework
  GeneralRelativitySolutions
  LinearOperators
  Options
  SoCcz4
  Spectral
  Time
  Utilities
  )
