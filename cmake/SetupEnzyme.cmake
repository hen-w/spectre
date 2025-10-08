# Distributed under the MIT License.
# See LICENSE.txt for details.

option(SPECTRE_ENABLE_ENZYME "Enable the Enzyme Clang plugin" OFF)
set(ENZYME_PLUGIN "" CACHE FILEPATH "Path to the Enzyme plugin shared library (e.g. ClangEnzyme-*.so/.dylib)")
set(ENZYME_INCLUDE_DIR "" CACHE PATH "Path to Enzyme include directory (optional)")

if (NOT SPECTRE_ENABLE_ENZYME)
  return()
endif()

if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR "SPECTRE_ENABLE_ENZYME requires Clang/clang++")
endif()

# Import Enzyme CMake package so imported targets like LLDEnzymeFlags/ClangEnzymeFlags exist.
# Set Enzyme_DIR to the directory containing EnzymeConfig.cmake if CMake cannot find it.
find_package(Enzyme CONFIG REQUIRED)

if (NOT ENZYME_PLUGIN)
  message(FATAL_ERROR "SPECTRE_ENABLE_ENZYME=ON but ENZYME_PLUGIN is not set")
endif()

if (NOT EXISTS "${ENZYME_PLUGIN}")
  message(FATAL_ERROR "ENZYME_PLUGIN not found: ${ENZYME_PLUGIN}")
endif()

if (ENZYME_INCLUDE_DIR AND NOT EXISTS "${ENZYME_INCLUDE_DIR}")
  message(FATAL_ERROR "ENZYME_INCLUDE_DIR not found: ${ENZYME_INCLUDE_DIR}")
endif()

# EXACT flags requested: -fplugin=<ClangEnzyme-*.so>
# Force unconstrained FP to avoid constrained FP intrinsics that Enzyme
# doesn't currently handle.
target_compile_options(
  SpectreFlags
  INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:
    -fplugin=${ENZYME_PLUGIN};
  >
)

if (ENZYME_INCLUDE_DIR)
  target_include_directories(SpectreFlags INTERFACE ${ENZYME_INCLUDE_DIR})
endif()

