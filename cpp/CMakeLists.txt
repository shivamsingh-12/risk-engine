cmake_minimum_required(VERSION 3.20)
project(qre LANGUAGES CXX)

#standard lang and build type
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

#can change from terminal in case
option(QRE_BUILD_TESTS "Build GoogleTest test suite" ON)
option(QRE_BUILD_BENCHMARKS "Build Google Benchmark suite" ON)
option(QRE_ENABLE_WARNINGS "Enable strict warnings" ON)
option(QRE_ENABLE_SANITIZERS "Enable ASan/UBSan (debug builds)" OFF)

#system installed eigen is best else fall back to FetchContent
find_package(Eigen3 3.4 QUIET NO_MODULE)
 
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen3)
endif()

#library
add_library(qre
  src/core/placeholder.cpp
  src/linalg/placeholder.cpp
  src/montecarlo/placeholder.cpp
  src/optimization/placeholder.cpp
  src/pricing/placeholder.cpp
  src/risk/placeholder.cpp
  src/stats/placeholder.cpp
)
target_include_directories(qre
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
 
target_link_libraries(qre PUBLIC Eigen3::Eigen)
#warnings - warnings every build, but sanitizers only for CI runs

if(QRE_ENABLE_WARNINGS)
  target_compile_options(qre PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
  )
endif()

#sanitizer section
#cmake -B build-asan -DQRE_ENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
if(QRE_ENABLE_SANITIZERS)
  target_compile_options(qre PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(qre PUBLIC -fsanitize=address,undefined)
endif()

#subdirectories
if(QRE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
 
if(QRE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
