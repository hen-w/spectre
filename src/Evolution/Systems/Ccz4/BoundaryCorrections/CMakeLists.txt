# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  LaxFriedrichs.cpp
  ParityFlux.cpp
  ProposedFlux.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  LaxFriedrichs.hpp
  ParityFlux.hpp
  ProposedFlux.hpp
  )
