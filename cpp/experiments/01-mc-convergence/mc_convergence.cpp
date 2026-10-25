//Phase 6 experiment 1: Monte Carlo convergence of an option price under N, with variance reduction toggles
//For each N and method, R independent replications (each with its own spawned seed) estimate the price;
//error is measured against the Black-Scholes closed form. RMSE over replications estimates the
//standard error, which should fall like N^(-1/2) -> log-log slope near -0.5
//C++ only: Black-Scholes (Phase 5, cross-validated in Python) is the independent truth, so a second MC
//implementation would add nothing; the Python reference tests already check the same scaling
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "qre/core/rng.hpp"
#include "qre/montecarlo/gbm_path.hpp"
#include "qre/pricing/black_scholes.hpp"
using qre::Rng;
using qre::montecarlo::McOptions;
using qre::pricing::BSParams;
using qre::pricing::OptionType;

int main(){
    const std::string output_path = "cpp/experiments/01-mc-convergence/results/mc_convergence_results.csv";
    std::ofstream out(output_path);
    if(!out.is_open()){
        std::cerr << "Failed to open output file " << output_path << "\n";
        std::cerr << "Run from repo root \n";
        return 1;
    }
    out.precision(10);
    out << "n,method,replications,rmse,mean_reported_se,rmse_ratio_per_10x_n\n";

    const BSParams p{100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call};
    const double exact = qre::pricing::price(p);
    const int reps = 100;
    const std::vector<std::size_t> n_grid = {100, 1000, 10000, 100000};
    struct Method{ std::string name; McOptions opts; };
    const std::vector<Method> methods = {
        {"plain", {false, false}},
        {"antithetic", {true, false}},
        {"control_variate", {false, true}},
        {"both", {true, true}},
    };

    Rng master(2026);   //one master seed -> every replication gets its own spawned stream
    for(const Method& m : methods){
        std::vector<double> log_n, log_rmse;
        double prev_rmse = 0.0;
        for(std::size_t n : n_grid){
            double sum_sq = 0.0, sum_se = 0.0;
            for(int r = 0; r < reps; ++r){
                Rng rng = master.spawn(static_cast<uint64_t>(r) + 1);
                const auto res = qre::montecarlo::mc_european_price(p, n, rng, m.opts);
                const double e = res.price - exact;
                sum_sq += e * e;
                sum_se += res.std_error;
            }
            const double rmse = std::sqrt(sum_sq / reps);
            const double ratio = prev_rmse > 0.0 ? prev_rmse / rmse : 0.0;
            out << n << "," << m.name << "," << reps << "," << rmse << "," << sum_se / reps << "," << ratio << "\n";
            std::cout << m.name << " n=" << n << " rmse=" << rmse << " mean_se=" << sum_se / reps
                      << " ratio=" << ratio << "\n";
            log_n.push_back(std::log(static_cast<double>(n)));
            log_rmse.push_back(std::log(rmse));
            prev_rmse = rmse;
        }
        //Least-squares slope of log(rmse) on log(n)
        double mx = 0, my = 0;
        for(std::size_t i = 0; i < log_n.size(); ++i){ mx += log_n[i]; my += log_rmse[i]; }
        mx /= static_cast<double>(log_n.size()); my /= static_cast<double>(log_n.size());
        double sxy = 0, sxx = 0;
        for(std::size_t i = 0; i < log_n.size(); ++i){ sxy += (log_n[i] - mx) * (log_rmse[i] - my); sxx += (log_n[i] - mx) * (log_n[i] - mx); }
        std::cout << m.name << " log-log slope = " << sxy / sxx << "\n\n";
    }
    out.close();
    std::cout << "Results written to " << output_path << "\n";
    return 0;
}
