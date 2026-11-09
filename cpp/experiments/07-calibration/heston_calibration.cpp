//Phase 9 experiment B: calibrating Heston to an implied volatility surface
//Part 1: identifiability. Generate a noiseless Heston surface from known parameters, calibrate from 8 different
//        starting points, and see which parameters come back the same and which trade off against each other.
//        Then add realistic quote noise (0.2 vol points) and repeat
//Part 2: model fit. Calibrate Heston to the Phase 8 Merton jump-diffusion surface (which has steep short-dated
//        skew) and look at where the fit breaks down by maturity
//Objective: RMS of vega-weighted price errors ~ implied vol RMSE (models/heston.hpp), Nelder-Mead
//C++ only: Heston pricing and calibration are cross-validated against the Python reference in the test suites
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "qre/models/heston.hpp"
#include "qre/models/merton.hpp"
#include "qre/pricing/black_scholes.hpp"
using namespace qre;
using qre::models::HestonParams;
using qre::pricing::OptionType;

namespace {
double iv_of(double price, double K, double T){
    const auto r = pricing::implied_vol(price, {100.0, K, 0.0, 0.2, T, OptionType::Call});
    return r.has_value() ? r->root : std::nan("");
}
}

int main(){
    const std::string out_dir = "cpp/experiments/07-calibration/results/";
    std::ofstream ident(out_dir + "identifiability.csv"), fit(out_dir + "heston_vs_merton_surface.csv");
    if(!ident.is_open() || !fit.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    ident.precision(10); fit.precision(10);
    const double rate = 0.0;

    //------------------------------------------------------------ Part 1
    const HestonParams truth{1.5, 0.05, 0.6, -0.65, 0.035};
    const std::vector<double> strikes{80, 85, 90, 95, 100, 105, 110, 115, 120}, mats{0.25, 1.0, 2.0};
    std::vector<std::vector<double>> clean, noisy;
    std::mt19937_64 eng(2033);
    std::normal_distribution<double> noise(0.0, 0.002);
    for(double T : mats){
        std::vector<double> c, n;
        for(double K : strikes){
            const double v = iv_of(models::heston_price(truth, 100.0, K, rate, T), K, T);
            c.push_back(v); n.push_back(v + noise(eng));
        }
        clean.push_back(c); noisy.push_back(n);
    }
    //Eight starting points spread over a plausible region
    std::uniform_real_distribution<double> uk(0.3, 6.0), ut(0.01, 0.12), us(0.1, 1.5), ur(-0.95, 0.2), uv(0.01, 0.12);
    std::vector<HestonParams> starts;
    for(int i = 0; i < 8; ++i) starts.push_back({uk(eng), ut(eng), us(eng), ur(eng), uv(eng)});

    ident << "surface,start,kappa,theta,sigma,rho,v0,rmse_vol_points,converged\n";
    ident << "truth,-," << truth.kappa << "," << truth.theta << "," << truth.sigma << "," << truth.rho << "," << truth.v0 << ",0,\n";
    for(const auto& [label, ivs] : {std::pair<std::string, std::vector<std::vector<double>>>{"noiseless", clean}, {"noise_0.2vol", noisy}}){
        const auto quotes = models::make_quotes(100.0, rate, strikes, mats, ivs);
        for(std::size_t i = 0; i < starts.size(); ++i){
            const auto c = models::calibrate_heston(100.0, rate, quotes, {starts[i]}, 4000);
            ident << label << "," << i << "," << c.params.kappa << "," << c.params.theta << "," << c.params.sigma << "," << c.params.rho << ","
                  << c.params.v0 << "," << c.rmse * 100.0 << "," << (c.converged ? "true" : "false") << "\n";
            std::cout << label << " start " << i << ": kappa " << c.params.kappa << " theta " << c.params.theta << " sigma " << c.params.sigma
                      << " rho " << c.params.rho << " v0 " << c.params.v0 << "  rmse " << c.rmse * 100.0 << " vol pts\n";
        }
    }

    //------------------------------------------------------------ Part 2
    const std::vector<double> mstrikes{80, 85, 90, 95, 100, 105, 110, 115, 120}, mmats{0.1, 0.25, 0.5, 1.0, 2.0};
    std::vector<std::vector<double>> merton_iv;
    for(double T : mmats){
        std::vector<double> row;
        for(double K : mstrikes) row.push_back(iv_of(models::merton_price({100.0, K, rate, 0.2, T, 1.0, -0.1, 0.15}), K, T));
        merton_iv.push_back(row);
    }
    const auto mq = models::make_quotes(100.0, rate, mstrikes, mmats, merton_iv);
    const auto mfit = models::calibrate_heston(100.0, rate, mq);
    std::cout << "\nHeston fit to Merton surface: kappa " << mfit.params.kappa << " theta " << mfit.params.theta << " sigma " << mfit.params.sigma
              << " rho " << mfit.params.rho << " v0 " << mfit.params.v0 << "  overall rmse " << mfit.rmse * 100.0 << " vol pts\n";
    std::ofstream fit_params(out_dir + "heston_fit_to_merton_params.csv");
    fit_params.precision(10);
    fit_params << "kappa,theta,sigma,rho,v0,rmse_vol_points,converged,feller\n"
               << mfit.params.kappa << "," << mfit.params.theta << "," << mfit.params.sigma << "," << mfit.params.rho << "," << mfit.params.v0 << ","
               << mfit.rmse * 100.0 << "," << (mfit.converged ? "true" : "false") << "," << (mfit.params.feller() ? "true" : "false") << "\n";
    fit << "maturity,strike,merton_iv,heston_iv,error_vol_points\n";
    for(std::size_t i = 0; i < mmats.size(); ++i){
        double sq = 0.0, worst = 0.0;
        for(std::size_t j = 0; j < mstrikes.size(); ++j){
            const double h = iv_of(models::heston_price(mfit.params, 100.0, mstrikes[j], rate, mmats[i]), mstrikes[j], mmats[i]);
            const double e = (h - merton_iv[i][j]) * 100.0;
            sq += e * e; worst = std::max(worst, std::abs(e));
            fit << mmats[i] << "," << mstrikes[j] << "," << merton_iv[i][j] << "," << h << "," << e << "\n";
        }
        std::cout << "  T=" << mmats[i] << ": rmse " << std::sqrt(sq / static_cast<double>(mstrikes.size())) << " vol pts, worst " << worst << "\n";
    }
    std::cout << "Results written to " << out_dir << "\n";
    return 0;
}
