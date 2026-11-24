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
option(QRE_BUILD_BENCHMARKS "Build Google Benchmark suite" OFF)
option(QRE_ENABLE_WARNINGS "Enable strict warnings" ON)
option(QRE_BUILD_EXPERIMENTS "Build standalone experiment executables" ON)
option(QRE_ENABLE_SANITIZERS "Enable ASan/UBSan (debug builds)" OFF)
option(QRE_BUILD_APPS "Build the qre_run production executable" ON)

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

#library: header-only, so an INTERFACE target that carries the include path and Eigen
add_library(qre INTERFACE)
target_include_directories(qre
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(qre INTERFACE Eigen3::Eigen)
target_compile_features(qre INTERFACE cxx_std_23)

#strict warnings for our own executables (tests, apps, experiments); link qre_warnings to opt in
add_library(qre_warnings INTERFACE)
if(QRE_ENABLE_WARNINGS)
  target_compile_options(qre_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
  )
endif()

#sanitizers (CI runs a Debug build with these):
#cmake -B build-asan -S cpp -DQRE_ENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
if(QRE_ENABLE_SANITIZERS)
  target_compile_options(qre INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  target_link_options(qre INTERFACE -fsanitize=address,undefined)
endif()

#ThreadSanitizer (CI, Phase 22): checks the lock-free queue and the streaming pipeline for data races
#cmake -B build-tsan -S cpp -DQRE_ENABLE_TSAN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
option(QRE_ENABLE_TSAN "Build with ThreadSanitizer (not together with QRE_ENABLE_SANITIZERS)" OFF)
if(QRE_ENABLE_TSAN)
  if(QRE_ENABLE_SANITIZERS)
    message(FATAL_ERROR "ThreadSanitizer cannot be combined with AddressSanitizer")
  endif()
  target_compile_options(qre INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(qre INTERFACE -fsanitize=thread)
endif()

#coverage (CI): cmake -B build-cov -S cpp -DQRE_ENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
option(QRE_ENABLE_COVERAGE "Instrument for gcov/llvm-cov coverage" OFF)
if(QRE_ENABLE_COVERAGE)
  target_compile_options(qre INTERFACE --coverage -O0)
  target_link_options(qre INTERFACE --coverage)
endif()

#subdirectories
if(QRE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
 
if(QRE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if(QRE_BUILD_EXPERIMENTS)
  add_subdirectory(experiments)
endif()
if(QRE_BUILD_APPS)
  add_subdirectory(apps)
endif()