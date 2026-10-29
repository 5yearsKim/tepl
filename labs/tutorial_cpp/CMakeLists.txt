cmake_minimum_required(VERSION 3.20)
project(tutorial_cpp LANGUAGES CXX)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
set(TEPL_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../../bazel-bin/tepl" CACHE FILEPATH "TEPL compiler")
set(EGGC_SOURCE_DIR "" CACHE PATH "Use an existing egg-c checkout")
if(NOT EXISTS "${TEPL_EXECUTABLE}")
  message(FATAL_ERROR "Build the compiler with bazel build //:tepl, or set TEPL_EXECUTABLE")
endif()
set(BUILD_TESTING OFF)
set(EGGC_BUILD_EXAMPLES OFF)
set(EGGC_BUILD_BENCHMARKS OFF)
if(EGGC_SOURCE_DIR)
  add_subdirectory("${EGGC_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/eggc" EXCLUDE_FROM_ALL)
else()
  include(FetchContent)
  FetchContent_Declare(eggc
    URL https://codeload.github.com/5yearsKim/egg-c/tar.gz/0c28bd5050b85ed10e915b5348c27f760ed31ae5
    URL_HASH SHA256=89d2ffbd7999edb65ae3c180d1c9caf72117fb1e8f1f39bf90107e4bb8d8b4f9)
  FetchContent_MakeAvailable(eggc)
endif()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/cpp" "${CMAKE_CURRENT_BINARY_DIR}/tepl-runtime")
set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
add_custom_command(
  OUTPUT "${generated}/generated.h"
  COMMAND "${TEPL_EXECUTABLE}" generate "${CMAKE_CURRENT_SOURCE_DIR}/program.tepl"
          --target cpp --cpp-namespace tutorial_ir --out "${generated}" --no-format
  DEPENDS "${TEPL_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/program.tepl"
  VERBATIM)
add_executable(tutorial_cpp src/main.cpp "${generated}/generated.h")
target_include_directories(tutorial_cpp PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(tutorial_cpp PRIVATE tepl::runtime)
set_target_properties(tutorial_cpp PROPERTIES CXX_EXTENSIONS OFF)
enable_testing()
add_test(NAME tutorial_cpp COMMAND tutorial_cpp)
