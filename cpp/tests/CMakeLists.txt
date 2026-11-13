include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
#keep gtest from overriding the parent project's compiler/runtime settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

#only test files with content so far; add the rest as their phases get tests
add_executable(qre_tests
    test_black_scholes.cpp
    test_gbm_path.cpp
    test_var_es.cpp
    test_data_hardening.cpp
    test_backtest.cpp
    test_ewma.cpp
    test_garch.cpp
    test_student_t.cpp
    test_phase8.cpp
    test_heston.cpp
    test_phase10.cpp
    test_phase11.cpp
    test_phase12.cpp
    test_phase13.cpp
    test_phase14.cpp
    test_phase15.cpp
    test_phase16.cpp
    test_phase17.cpp
    test_phase18.cpp
)
target_link_libraries(qre_tests PRIVATE qre GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(qre_tests)

#Phase 14: end-to-end run of the production executable on the real data (fast models), from the repo root
if(QRE_BUILD_APPS)
  add_test(NAME qre_run_smoke
           COMMAND qre_run --config configs/ci_smoke.conf --output ${CMAKE_BINARY_DIR}/qre_run_smoke
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..)
endif()
