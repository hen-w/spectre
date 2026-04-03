# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_SoScalarWave)

set(LIBRARY_SOURCES
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryConditions/Test_Periodic.cpp
  BoundaryCorrections/Test_CgCollocation.cpp
  Test_Tags.cpp
  Test_TimeDerivative.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  MathFunctions
  SoScalarWave
  Time
  Utilities
  WaveEquationSolutions
  )
