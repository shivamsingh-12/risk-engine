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
)
target_link_libraries(qre_tests PRIVATE qre GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(qre_tests)
