//Phase 14 experiment: is the production engine reproducible, how does it scale, and where does the time go?
//Part 1: reproducibility. The full daily job (configs/daily_risk.conf: 8 models x 2 confidence levels x 503 days)
//        run with 1, 2, 3, 4 and 8 threads, twice each; every forecast compared bit for bit with the 1-thread run
//Part 2: scaling. Wall time of the forecast stage for each thread count (machine dependent, written separately)
//Part 3: cost per model. Serial time per forecast for each model (machine dependent)
//Part 4: GARCH fits for the cross-language check. For every backtest day: the C++ GARCH(1,1) (normal) fit of the
//        portfolio returns -- alpha, beta, log-likelihood -- so the Python script can compare its own optimizer's
//        answer day by day (python_reference/experiments/production/cross_check.py)
//Run from the repo root
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "qre/engine/config.hpp"
#include "qre/engine/engine.hpp"
#include "qre/engine/parallel.hpp"
#include "qre/risk/garch.hpp"
using namespace qre;
using namespace qre::engine;

namespace {
double max_abs_diff(const EngineResult& a, const EngineResult& b){
    double d = 0.0;
    for(std::size_t c = 0; c < a.results.size(); ++c){
        const ConfidenceResult& x = a.results[c];
        const ConfidenceResult& y = b.results[c];
        d = std::max({d, (x.forecasts.var - y.forecasts.var).cwiseAbs().maxCoeff(), (x.forecasts.es - y.forecasts.es).cwiseAbs().maxCoeff()});
        for(std::size_t i = 0; i < x.combined.var.size(); ++i) d = std::max(d, std::abs(x.combined.var[i] - y.combined.var[i]));
        d = std::max(d, std::abs(x.next_combined.var - y.next_combined.var));
    }
    return d;
}
bool bit_identical(const EngineResult& a, const EngineResult& b){
    for(std::size_t c = 0; c < a.results.size(); ++c){
        const ConfidenceResult& x = a.results[c];
        const ConfidenceResult& y = b.results[c];
        if(!(x.forecasts.var == y.forecasts.var) || !(x.forecasts.es == y.forecasts.es) || x.combined.var != y.combined.var ||
           x.combined.es != y.combined.es || !(x.next_weights == y.next_weights)) return false;
        for(std::size_t k = 0; k < x.next_day.size(); ++k){
            if(x.next_day[k].var != y.next_day[k].var || x.next_day[k].es != y.next_day[k].es) return false;
        }
    }
    return true;
}
}

int main(){
    const std::string out_dir = "cpp/experiments/production/results/";
    std::ofstream c1(out_dir + "reproducibility.csv"), c2(out_dir + "thread_scaling.csv"), c3(out_dir + "model_cost.csv"),
                  c4(out_dir + "cpp_garch_fits.csv");
    if(!c1.is_open() || !c2.is_open() || !c3.is_open() || !c4.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    for(auto* f : {&c1, &c2, &c3, &c4}) f->precision(12);
    RunConfig cfg;
    try{ cfg = load_config("configs/daily_risk.conf"); }
    catch(const std::exception& e){
        std::cerr << e.what() << "\nRun from repo root \n";
        return 1;
    }

    //------------------------------------------------------------ Parts 1 and 2
    c1 << "threads,run,bit_identical_to_1_thread,max_abs_difference\n";
    c2 << "threads,hardware_threads,forecast_ms,speedup,efficiency\n";
    StageTimes base_t;
    const EngineResult base = run_engine(cfg, 1, &base_t);
    std::cout << "1 thread: " << base_t.forecast_ms / 1000.0 << " s for " << base_t.tasks << " forecasts\n";
    const std::size_t hw = resolve_threads(0);
    c2 << 1 << "," << hw << "," << base_t.forecast_ms << ",1,1\n";
    for(std::size_t th : {1u, 2u, 3u, 4u, 8u}){
        double best = 1e300;
        for(int run = 1; run <= 2; ++run){
            if(th == 1 && run == 1){
                c1 << "1,1,yes,0\n";
                continue;
            }
            StageTimes t;
            const EngineResult r = run_engine(cfg, th, &t);
            best = std::min(best, t.forecast_ms);
            c1 << th << "," << run << "," << (bit_identical(base, r) ? "yes" : "NO") << "," << max_abs_diff(base, r) << "\n";
        }
        if(th > 1){
            c2 << th << "," << hw << "," << best << "," << base_t.forecast_ms / best << "," << base_t.forecast_ms / best / static_cast<double>(th) << "\n";
        }
        std::cout << th << " threads done\n";
    }

    //------------------------------------------------------------ Part 3
    {
        const Eigen::Index T = base.prices.rows();
        std::vector<double> ms_per;
        std::vector<std::size_t> count;
        for(const std::string& m : cfg.models){
            const auto a = std::chrono::steady_clock::now();
            std::size_t n = 0;
            for(double c : cfg.confidence){
                for(Eigen::Index t = cfg.window + 1; t <= T; ++t, ++n){
                    (void)risk::forecast_var_es(base.prices, base.weights, t, cfg.window, c, model_from_name(m),
                                                static_cast<std::size_t>(cfg.mc_paths), static_cast<uint64_t>(cfg.seed), 252, cfg.lambda);
                }
            }
            ms_per.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
            count.push_back(n);
        }
        double total = 0.0;
        for(double x : ms_per) total += x;
        c3 << "model,forecasts,total_ms,ms_per_forecast,share_of_job\n";
        for(std::size_t k = 0; k < cfg.models.size(); ++k){
            c3 << cfg.models[k] << "," << count[k] << "," << ms_per[k] << "," << ms_per[k] / static_cast<double>(count[k]) << ","
               << ms_per[k] / total << "\n";
        }
    }

    //------------------------------------------------------------ Part 4
    c4 << "date,alpha,beta,loglik,converged\n";
    for(Eigen::Index t = cfg.window + 1; t <= base.prices.rows(); ++t){      //t = rows -> the next-day forecast
        const ReturnMatrix rets = data::simple_return(data::pit_window(base.prices, t, cfg.window));
        const risk::GarchFit f = risk::fit_garch(risk::portfolio_returns(rets, base.weights));
        c4 << (t < base.prices.rows() ? base.dates[static_cast<std::size_t>(t)] : std::string("next")) << "," << f.alpha << "," << f.beta << "," << f.loglik << "," << (f.converged ? 1 : 0) << "\n";
    }
    std::cout << "Results in " << out_dir << "\n";
    return 0;
}
