//Phase 6 experiment 2: Monte Carlo vs Black-Scholes across moneyness, and the payoff of variance reduction
//For each strike and method: R replications at fixed N. Reports bias, RMSE, how often the reported
//95% interval (price +/- 1.96 se) covers the exact price, the variance reduction factor against plain MC,
//and accuracy per unit of compute (1 / (variance x wall time)) relative to plain MC
//C++ only: the exact answer is the Phase 5 closed form, so Black-Scholes is the independent check
//Wall-clock timings depend on the machine and compiler flags -> compare ratios, not absolute times
#include <chrono>
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
    const std::string output_path = "cpp/experiments/03-bs-vs-mc/results/mc_vs_bs_results.csv";
    std::ofstream out(output_path);
    if(!out.is_open()){
        std::cerr << "Failed to open output file " << output_path << "\n";
        std::cerr << "Run from repo root \n";
        return 1;
    }
    out.precision(10);
    out << "moneyness,method,bs_price,mean_mc_price,bias,rmse,mean_reported_se,coverage_95,variance_reduction_factor,time_per_run_ms,relative_efficiency\n";

    const double spot = 100.0;
    const std::size_t n = 5000;
    const int reps = 400;
    const double moneyness_grid[] = {0.8, 0.9, 1.0, 1.1, 1.2};   //K/S
    struct Method{ std::string name; McOptions opts; };
    const std::vector<Method> methods = {
        {"plain", {false, false}},
        {"antithetic", {true, false}},
        {"control_variate", {false, true}},
        {"both", {true, true}},
    };

    Rng master(2027);
    for(double m : moneyness_grid){
        const BSParams p{spot, spot * m, 0.05, 0.2, 1.0, OptionType::Call};
        const double exact = qre::pricing::price(p);
        double plain_var = 0.0, plain_cost = 0.0;
        for(const Method& meth : methods){
            double sum_price = 0.0, sum_sq = 0.0, sum_se = 0.0;
            int covered = 0;
            const auto t0 = std::chrono::steady_clock::now();
            for(int r = 0; r < reps; ++r){
                Rng rng = master.spawn(static_cast<uint64_t>(r) + 1);
                const auto res = qre::montecarlo::mc_european_price(p, n, rng, meth.opts);
                const double e = res.price - exact;
                sum_price += res.price;
                sum_sq += e * e;
                sum_se += res.std_error;
                if(std::abs(e) <= 1.96 * res.std_error) ++covered;
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
            const double mse = sum_sq / reps;
            if(meth.name == "plain"){ plain_var = mse; plain_cost = mse * ms; }
            const double vrf = plain_var / mse;
            const double eff = plain_cost / (mse * ms);
            out << m << "," << meth.name << "," << exact << "," << sum_price / reps << "," << sum_price / reps - exact
                << "," << std::sqrt(mse) << "," << sum_se / reps << "," << static_cast<double>(covered) / reps
                << "," << vrf << "," << ms << "," << eff << "\n";
            std::cout << "K/S=" << m << " " << meth.name << " rmse=" << std::sqrt(mse) << " cov95="
                      << static_cast<double>(covered) / reps << " VRF=" << vrf << " time_ms=" << ms
                      << " eff=" << eff << "\n";
        }
        std::cout << "\n";
    }
    out.close();
    std::cout << "Results written to " << output_path << "\n";
    return 0;
}
