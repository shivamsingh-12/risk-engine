//Covariance benchmarks: sample covariance, Ledoit-Wolf shrinkage and random-matrix clipping on T = 500 days
#include <benchmark/benchmark.h>
#include <random>
#include <Eigen/Dense>
#include "qre/core/rng.hpp"
#include "qre/linalg/covariance.hpp"
#include "qre/linalg/shrinkage.hpp"

using namespace qre;

namespace{
ReturnMatrix returns(Eigen::Index n_assets){
    Rng rng(11);
    std::normal_distribution<double> nd(0.0, 0.01);
    ReturnMatrix r(500, n_assets);
    for(Eigen::Index i = 0; i < r.rows(); ++i)
        for(Eigen::Index j = 0; j < n_assets; ++j) r(i, j) = nd(rng.engine());
    return r;
}

void BM_SampleCov(benchmark::State& state){
    const ReturnMatrix r = returns(state.range(0));
    for(auto _ : state) benchmark::DoNotOptimize(linalg::sample_cov(r));
}
void BM_LedoitWolf(benchmark::State& state){
    const ReturnMatrix r = returns(state.range(0));
    for(auto _ : state) benchmark::DoNotOptimize(linalg::ledoit_wolf(r));
}
void BM_RmtClip(benchmark::State& state){
    const ReturnMatrix r = returns(state.range(0));
    for(auto _ : state) benchmark::DoNotOptimize(linalg::rmt_clip(r));
}
BENCHMARK(BM_SampleCov)->Arg(10)->Arg(50)->Arg(200)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_LedoitWolf)->Arg(10)->Arg(50)->Arg(200)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RmtClip)->Arg(10)->Arg(50)->Arg(200)->Unit(benchmark::kMicrosecond);
}
