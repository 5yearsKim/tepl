cmake_minimum_required(VERSION 3.20)
project(tutorial_cpp LANGUAGES CXX)

set(TEPL_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../../bazel-bin/tepl"
    CACHE FILEPATH "Path to the TEPL compiler")
set(EGGC_SOURCE_DIR "" CACHE PATH "Use an existing egg-c checkout")
if(NOT EXISTS "${TEPL_EXECUTABLE}")
  message(FATAL_ERROR "Build the compiler with bazel build //:tepl, or set TEPL_EXECUTABLE")
endif()

# Build only the egg-c header library.
set(BUILD_TESTING OFF)
set(EGGC_BUILD_EXAMPLES OFF)
set(EGGC_BUILD_BENCHMARKS OFF)
if(EGGC_SOURCE_DIR)
  add_subdirectory("${EGGC_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/eggc" EXCLUDE_FROM_ALL)
else()
  include(FetchContent)
  FetchContent_Declare(eggc
    GIT_REPOSITORY https://github.com/5yearsKim/egg-c.git
    GIT_TAG 0c28bd5050b85ed10e915b5348c27f760ed31ae5)
  FetchContent_MakeAvailable(eggc)
endif()

enable_testing()
foreach(example IN ITEMS basic analyze)
  set(pattern "${CMAKE_CURRENT_SOURCE_DIR}/pattern_${example}")
  set(generated "${CMAKE_CURRENT_SOURCE_DIR}/src/tepl_pattern_${example}")
  file(GLOB_RECURSE tepl_sources CONFIGURE_DEPENDS "${pattern}/*.tepl")

  # Regenerate headers when the TEPL source or compiler changes.
  add_custom_command(
    OUTPUT "${generated}/generated.h"
    COMMAND "${TEPL_EXECUTABLE}" generate "${pattern}"
            --target cpp --cpp-namespace "tepl_pattern_${example}"
            --no-format --out "${generated}"
    DEPENDS "${TEPL_EXECUTABLE}" ${tepl_sources}
    VERBATIM)

  add_executable(pattern_${example}
    "src/${example}.cpp" "${generated}/generated.h")
  target_link_libraries(pattern_${example} PRIVATE eggc::eggc)
  target_compile_features(pattern_${example} PRIVATE cxx_std_20)
  set_target_properties(pattern_${example} PROPERTIES CXX_EXTENSIONS OFF)
  add_test(NAME pattern_${example} COMMAND pattern_${example})
endforeach()
