//Risk benchmarks: single-model fits on a 250-day window, and the full daily run across thread counts.
//The engine benchmark reads configs/daily_risk.conf and the bundled data, so run from the repo root.
#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include "qre/core/rng.hpp"
#include "qre/engine/config.hpp"
#include "qre/engine/engine.hpp"
#include "qre/risk/evt.hpp"
#include "qre/risk/garch.hpp"
#include "qre/risk/var_es.hpp"

using namespace qre;

namespace{
std::vector<double> window(){
    Rng rng(3);
    std::student_t_distribution<double> t(5.0);
    std::vector<double> r(250);
    for(double& x : r) x = 0.01 * t(rng.engine());
    return r;
}

void BM_HistoricalVarEs(benchmark::State& state){
    const auto r = window();
    for(auto _ : state) benchmark::DoNotOptimize(risk::historical_var_es(r, 0.99));
}
void BM_GarchFit(benchmark::State& state){
    const auto r = window();
    for(auto _ : state) benchmark::DoNotOptimize(risk::fit_garch(r));
}
void BM_GarchTFit(benchmark::State& state){
    const auto r = window();
    for(auto _ : state) benchmark::DoNotOptimize(risk::fit_garch(r, risk::GarchDist::StudentT));
}
BENCHMARK(BM_HistoricalVarEs)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GarchFit)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GarchTFit)->Unit(benchmark::kMillisecond);

//The daily risk run of configs/daily_risk.conf: 8 models x 2 confidence levels x 503 backtest days
void BM_DailyRiskRun(benchmark::State& state){
    engine::RunConfig cfg;
    try{
        cfg = engine::load_config("configs/daily_risk.conf");
    } catch(const std::exception& e){
        state.SkipWithError("run from the repo root: configs/daily_risk.conf not found");
        return;
    }
    for(auto _ : state) benchmark::DoNotOptimize(engine::run_engine(cfg, static_cast<std::size_t>(state.range(0))));
}
BENCHMARK(BM_DailyRiskRun)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Unit(benchmark::kMillisecond)->UseRealTime()->Iterations(1);
}
