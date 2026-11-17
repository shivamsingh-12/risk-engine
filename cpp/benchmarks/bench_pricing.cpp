//Pricing benchmarks: Black-Scholes, Monte Carlo, Sobol points, and basket Greeks by AAD vs bumping
#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include "qre/core/rng.hpp"
#include "qre/montecarlo/gbm_path.hpp"
#include "qre/montecarlo/greeks.hpp"
#include "qre/montecarlo/qmc.hpp"
#include "qre/pricing/black_scholes.hpp"

using namespace qre;

namespace{
const pricing::BSParams kCall{100.0, 105.0, 0.03, 0.2, 1.0, pricing::OptionType::Call};

void BM_BlackScholesPrice(benchmark::State& state){
    pricing::BSParams p = kCall;
    for(auto _ : state){
        benchmark::DoNotOptimize(p);
        benchmark::DoNotOptimize(pricing::price(p));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BlackScholesPrice);

void BM_ImpliedVol(benchmark::State& state){
    const double target = pricing::price(kCall);
    for(auto _ : state) benchmark::DoNotOptimize(pricing::implied_vol(target, kCall));
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ImpliedVol);

//items = simulated paths
void BM_McEuropean(benchmark::State& state){
    const auto n = static_cast<std::size_t>(state.range(0));
    Rng rng(42);
    for(auto _ : state) benchmark::DoNotOptimize(montecarlo::mc_european_price(kCall, n, rng));
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_McEuropean)->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->Unit(benchmark::kMillisecond);

//items = normal draws (points x dimensions)
void BM_SobolNormals(benchmark::State& state){
    const auto dim = static_cast<std::size_t>(state.range(0));
    for(auto _ : state) benchmark::DoNotOptimize(montecarlo::qmc::sobol_normals(16384, dim));
    state.SetItemsProcessed(state.iterations() * 16384 * state.range(0));
}
BENCHMARK(BM_SobolNormals)->Arg(1)->Arg(16)->Arg(64)->Unit(benchmark::kMillisecond);

struct Basket{
    Eigen::VectorXd spots, vols, weights;
    Eigen::MatrixXd corr, z;
    explicit Basket(Eigen::Index n){
        spots = Eigen::VectorXd::Constant(n, 100.0);
        vols = Eigen::VectorXd::LinSpaced(n, 0.15, 0.35);
        weights = Eigen::VectorXd::Constant(n, 1.0 / static_cast<double>(n));
        corr = Eigen::MatrixXd::Constant(n, n, 0.3);
        corr.diagonal().setOnes();
        Rng rng(7);
        std::normal_distribution<double> nd;
        z.resize(4096, n);
        for(Eigen::Index i = 0; i < z.rows(); ++i)
            for(Eigen::Index j = 0; j < n; ++j) z(i, j) = nd(rng.engine());
    }
};

//2N + 2 sensitivities of an N-asset basket call on 4,096 paths: the price alone, then three ways to get the Greeks
void BM_BasketPrice(benchmark::State& state){
    const Basket b(state.range(0));
    for(auto _ : state)
        benchmark::DoNotOptimize(montecarlo::basket_call_price(b.spots, b.vols, b.corr, b.weights, 100.0, 0.03, 1.0, b.z));
}
void BM_BasketGreeksAad(benchmark::State& state){
    const Basket b(state.range(0));
    for(auto _ : state)
        benchmark::DoNotOptimize(montecarlo::basket_call_aad(b.spots, b.vols, b.corr, b.weights, 100.0, 0.03, 1.0, b.z));
}
void BM_BasketGreeksPathwise(benchmark::State& state){
    const Basket b(state.range(0));
    for(auto _ : state)
        benchmark::DoNotOptimize(montecarlo::basket_call_pathwise(b.spots, b.vols, b.corr, b.weights, 100.0, 0.03, 1.0, b.z));
}
void BM_BasketGreeksBump(benchmark::State& state){
    const Basket b(state.range(0));
    for(auto _ : state)
        benchmark::DoNotOptimize(montecarlo::basket_call_bump(b.spots, b.vols, b.corr, b.weights, 100.0, 0.03, 1.0, b.z));
}
BENCHMARK(BM_BasketPrice)->Arg(10)->Arg(50)->Arg(100)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BasketGreeksAad)->Arg(10)->Arg(50)->Arg(100)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BasketGreeksPathwise)->Arg(10)->Arg(50)->Arg(100)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BasketGreeksBump)->Arg(10)->Arg(50)->Arg(100)->Unit(benchmark::kMillisecond);
}
