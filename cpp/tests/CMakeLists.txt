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

#one file per topic; every C++ module also has a Python counterpart in python_reference/tests
add_executable(qre_tests
    test_black_scholes.cpp
    test_gbm_path.cpp
    test_var_es.cpp
    test_data_hardening.cpp
    test_backtest.cpp
    test_ewma.cpp
    test_garch.cpp
    test_student_t.cpp
    test_stochastic_processes.cpp
    test_heston.cpp
    test_shrinkage.cpp
    test_evt.cpp
    test_advanced_mc.cpp
    test_adaptive.cpp
    test_engine.cpp
    test_regressions.cpp
    test_credit.cpp
    test_deep_hedging.cpp
    test_execution.cpp
    test_stress.cpp
    test_portfolio.cpp
    test_stream.cpp
)
find_package(Threads REQUIRED)
target_link_libraries(qre_tests PRIVATE qre qre_warnings GTest::gtest_main Threads::Threads)

include(GoogleTest)
gtest_discover_tests(qre_tests)

#Phase 14: end-to-end run of the production executable on the real data (fast models), from the repo root
if(QRE_BUILD_APPS)
  add_test(NAME qre_run_smoke
           COMMAND qre_run --config configs/ci_smoke.conf --output ${CMAKE_BINARY_DIR}/qre_run_smoke
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..)
  #Phase 22: qre_stream starts, streams for two seconds on a free port and exits cleanly
  add_test(NAME qre_stream_smoke
           COMMAND qre_stream --data data/market_data/adjusted_close.csv --holdings AAPL=100,JPM=50
                   --port 0 --rate 5000 --duration 2
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..)
  set_tests_properties(qre_stream_smoke PROPERTIES PASS_REGULAR_EXPRESSION "qre_stream: stopped")
endif()
