# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BoundaryCondition.cpp
  ConstraintsRadiationPreserving.cpp
  DirichletAnalytic.cpp
  Sommerfeld.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCondition.hpp
  ConstraintsRadiationPreserving.hpp
  DirichletAnalytic.hpp
  Sommerfeld.hpp
  Factory.hpp
  )
