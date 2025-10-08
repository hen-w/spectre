# Distributed under the MIT License.
# See LICENSE.txt for details.

option(SPECTRE_ENABLE_CLAD "Enable the clad Clang plugin" OFF)
set(CLAD_PLUGIN "" CACHE FILEPATH "Path to the clad plugin shared library (e.g. libclad.so/.dylib)")
set(CLAD_INCLUDE_DIR "" CACHE PATH "Path to clad include directory (contains clad/)")

if (NOT SPECTRE_ENABLE_CLAD)
  return()
endif()

if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR "SPECTRE_ENABLE_CLAD requires Clang/clang++")
endif()

if (NOT CLAD_PLUGIN)
  message(FATAL_ERROR "SPECTRE_ENABLE_CLAD=ON but CLAD_PLUGIN is not set")
endif()

if (NOT EXISTS "${CLAD_PLUGIN}")
  message(FATAL_ERROR "CLAD_PLUGIN not found: ${CLAD_PLUGIN}")
endif()

if (CLAD_INCLUDE_DIR AND NOT EXISTS "${CLAD_INCLUDE_DIR}")
  message(FATAL_ERROR "CLAD_INCLUDE_DIR not found: ${CLAD_INCLUDE_DIR}")
endif()

# Define macro to disable clad numerical differentiation fallback
# (was incorrectly added to compile options previously)
target_compile_definitions(SpectreFlags INTERFACE CLAD_NO_NUM_DIFF)

target_compile_options(
  SpectreFlags
  INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:
    -fplugin=${CLAD_PLUGIN};
  >
)

if (CLAD_INCLUDE_DIR)
  target_include_directories(SpectreFlags INTERFACE ${CLAD_INCLUDE_DIR})
endif()

# Ensure -lstdc++ -lm are on link lines
target_link_libraries(SpectreFlags INTERFACE stdc++ m)
target_link_options(SpectreFlags INTERFACE -lstdc++ -lm)