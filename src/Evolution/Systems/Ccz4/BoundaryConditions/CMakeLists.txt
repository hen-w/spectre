# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BoundaryCondition.cpp
  ConstraintsRadiationPreserving.cpp
  CRPBCExp1.cpp
  CRPBCExp2.cpp
  CRPBCExp3.cpp
  DirichletAnalytic.cpp
  DirichletCharacteristics.cpp
  Sommerfeld.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCondition.hpp
  ConstraintsRadiationPreserving.hpp
  CRPBCExp1.hpp
  CRPBCExp2.hpp
  CRPBCExp3.hpp
  DirichletAnalytic.hpp
  DirichletCharacteristics.hpp
  Sommerfeld.hpp
  Factory.hpp
  )
