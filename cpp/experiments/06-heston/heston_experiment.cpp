//Phase 9 experiment A: Heston pricing, simulation and the shape of the smile
//Part 1: full-truncation Euler Monte Carlo vs the semi-closed form, by number of time steps, for parameters that
//        satisfy the Feller condition, mildly violate it, and violate it badly. How large is the discretization
//        bias, and how often does the simulated variance hit zero?
//Part 2: the term structure of the smile. ATM implied vol and skew (IV at 90% minus IV at 110% of spot) from 1 week
//        to 5 years, for Heston and for the Phase 8 Merton jump-diffusion
//C++ only: Heston prices are cross-validated against the Python reference (which also matches the original 1993
//formula) in the test suites
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "qre/models/heston.hpp"
#include "qre/models/merton.hpp"
#include "qre/pricing/black_scholes.hpp"
using namespace qre;
using qre::models::HestonParams;
using qre::pricing::OptionType;

namespace {
//Share of paths whose (unfloored) variance goes below zero at least once
double share_hitting_zero(const HestonParams& p, double T, std::size_t steps, std::size_t n, Rng& rng){
    const double h = T / static_cast<double>(steps);
    std::normal_distribution<double> z(0.0, 1.0);
    std::size_t hit = 0;
    for(std::size_t i = 0; i < n; ++i){
        double v = p.v0;
        bool touched = false;
        for(std::size_t s = 0; s < steps; ++s){
            const double vp = std::max(v, 0.0);
            v += p.kappa * (p.theta - vp) * h + p.sigma * std::sqrt(vp * h) * z(rng.engine());
            if(v < 0.0) touched = true;
        }
        if(touched) ++hit;
    }
    return static_cast<double>(hit) / static_cast<double>(n);
}
double iv(double price, double K, double T){
    const auto r = pricing::implied_vol(price, {100.0, K, 0.0, 0.2, T, OptionType::Call});
    return r.has_value() ? r->root : std::nan("");
}
}

int main(){
    const std::string out_dir = "cpp/experiments/06-heston/results/";
    std::ofstream bias(out_dir + "mc_bias.csv"), smile(out_dir + "smile_term_structure.csv");
    if(!bias.is_open() || !smile.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    bias.precision(10); smile.precision(10);

    //------------------------------------------------------------ Part 1
    struct Case{ const char* name; HestonParams p; };
    const Case cases[] = {
        {"feller_holds", {3.0, 0.05, 0.3, -0.6, 0.04}},          //2 kappa theta = 0.30 vs sigma^2 = 0.09
        {"feller_mildly_violated", {2.0, 0.04, 0.5, -0.7, 0.04}},//0.16 vs 0.25
        {"feller_badly_violated", {1.0, 0.04, 1.0, -0.7, 0.04}}, //0.08 vs 1.00
    };
    bias << "case,feller_ratio,steps,exact,mc_price,mc_se,bias,bias_in_se,share_paths_variance_below_zero\n";
    Rng rng(2032);
    for(const auto& c : cases){
        const double exact = models::heston_price(c.p, 100.0, 100.0, 0.0, 1.0);
        const double ratio = 2.0 * c.p.kappa * c.p.theta / (c.p.sigma * c.p.sigma);
        for(std::size_t steps : {4u, 16u, 64u, 256u}){
            const auto r = models::heston_mc_price(c.p, 100.0, 100.0, 0.0, 1.0, steps, 200000, rng);
            const double zero = share_hitting_zero(c.p, 1.0, steps, 20000, rng);
            bias << c.name << "," << ratio << "," << steps << "," << exact << "," << r.price << "," << r.std_error << ","
                 << r.price - exact << "," << (r.price - exact) / r.std_error << "," << zero << "\n";
            std::cout << c.name << " (2k.th/s^2=" << ratio << ") steps " << steps << ": exact " << exact << " mc " << r.price
                      << " +- " << r.std_error << " bias " << r.price - exact << " (" << (r.price - exact) / r.std_error
                      << " se)  v<0 on " << zero * 100 << "% of paths\n";
        }
    }

    //------------------------------------------------------------ Part 2
    const HestonParams hp{2.0, 0.04, 0.6, -0.7, 0.04};
    smile << "model,maturity,iv_90,iv_100,iv_110,skew_90_minus_110\n";
    std::cout << "\n";
    for(double T : {1.0 / 52.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0}){
        const double h90 = iv(models::heston_price(hp, 100.0, 90.0, 0.0, T), 90.0, T);
        const double h100 = iv(models::heston_price(hp, 100.0, 100.0, 0.0, T), 100.0, T);
        const double h110 = iv(models::heston_price(hp, 100.0, 110.0, 0.0, T), 110.0, T);
        const models::MertonParams m90{100.0, 90.0, 0.0, 0.2, T, 1.0, -0.1, 0.15}, m100{100.0, 100.0, 0.0, 0.2, T, 1.0, -0.1, 0.15},
                                   m110{100.0, 110.0, 0.0, 0.2, T, 1.0, -0.1, 0.15};
        const double j90 = iv(models::merton_price(m90), 90.0, T), j100 = iv(models::merton_price(m100), 100.0, T),
                     j110 = iv(models::merton_price(m110), 110.0, T);
        smile << "heston," << T << "," << h90 << "," << h100 << "," << h110 << "," << h90 - h110 << "\n";
        smile << "merton," << T << "," << j90 << "," << j100 << "," << j110 << "," << j90 - j110 << "\n";
        std::cout << "T=" << T << ": Heston ATM " << h100 << " skew " << h90 - h110 << " | Merton ATM " << j100 << " skew " << j90 - j110 << "\n";
    }
    std::cout << "Results written to " << out_dir << "\n";
    return 0;
}
