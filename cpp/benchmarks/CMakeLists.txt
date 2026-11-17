#Google Benchmark suite: cmake -B build -S cpp -DCMAKE_BUILD_TYPE=Release -DQRE_BUILD_BENCHMARKS=ON
#Run from the repo root (the engine benchmark reads configs/ and data/): build/benchmarks/qre_bench
include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

add_executable(qre_bench
    bench_pricing.cpp
    bench_linalg.cpp
    bench_risk.cpp
)
target_link_libraries(qre_bench PRIVATE qre qre_warnings benchmark::benchmark_main)
