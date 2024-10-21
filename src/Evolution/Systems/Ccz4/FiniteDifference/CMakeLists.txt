# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Derivatives.cpp
  FilterOptions.cpp
  Filters.cpp
  Reconstructor.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Derivatives.hpp
  Factory.hpp
  FilterOptions.hpp
  Filters.hpp
  FiniteDifference.hpp
  ReconstructWork.hpp
  ReconstructWork.tpp
  Tag.hpp
  Reconstructor.hpp
  )
