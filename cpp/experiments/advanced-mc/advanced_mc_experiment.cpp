//Phase 12 experiment: quasi-Monte Carlo, the Brownian bridge, and how to get Greeks from a simulation
//Part 1: QMC convergence. Geometric Asian call (exact price known) with 4, 16 and 64 monitoring dates, so 4, 16 and
//        64 dimensions. Pseudo-random MC vs randomly shifted Sobol (incremental paths) vs Sobol + Brownian bridge,
//        N = 2^7 .. 2^16 paths, RMSE over 32 independent randomizations, and the fitted convergence rate.
//        Arithmetic Asian (no closed form): standard error across randomizations at N = 2^14
//Part 2: Greek estimators. European call delta and vega by pathwise, likelihood ratio and bumping; digital call delta,
//        where pathwise is identically 0, by likelihood ratio and by bumping with several bump sizes
//Part 3: AAD cost. Basket call on N = 1..100 assets, all 2N + 2 sensitivities by AAD, by hand-coded pathwise
//        and by central-difference bumping, timed against one price. Accuracy is written separately from timing,
//        because timing depends on the machine
//C++ only: Sobol, Brownian bridge, Asian pricing, the Greek estimators and AAD are cross-validated against the Python
//reference (Sobol vs scipy.stats.qmc bit for bit, AAD vs analytic Black-Scholes Greeks)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include "qre/montecarlo/asian.hpp"
#include "qre/montecarlo/greeks.hpp"
#include "qre/montecarlo/qmc.hpp"
#include "qre/pricing/black_scholes.hpp"
using namespace qre;
using namespace qre::montecarlo;

namespace {
const char* name(PathMethod m){
    return m == PathMethod::Mc ? "mc" : (m == PathMethod::Sobol ? "sobol" : "sobol_bridge");
}
Eigen::VectorXd normals(std::size_t n, Rng& rng){
    const std::vector<double> d = distributions::normal::sample(n, 0.0, 1.0, rng);
    return Eigen::Map<const Eigen::VectorXd>(d.data(), static_cast<Eigen::Index>(n));
}
GreekEstimate mean_se(const Eigen::ArrayXd& x){
    const double n = static_cast<double>(x.size()), m = x.mean();
    return {m, std::sqrt((x - m).square().sum() / (n - 1.0) / n)};
}
template<class F> double median_ms(F&& f, int reps = 5){
    std::vector<double> t;
    for(int r = 0; r < reps; ++r){
        const auto a = std::chrono::steady_clock::now();
        f();
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
}

int main(){
    const std::string out_dir = "cpp/experiments/advanced-mc/results/";
    std::ofstream c1(out_dir + "qmc_convergence.csv"), c1s(out_dir + "qmc_rates.csv"), c1a(out_dir + "qmc_arithmetic.csv"),
                  c2(out_dir + "greek_estimators.csv"), c3a(out_dir + "aad_accuracy.csv"), c3t(out_dir + "aad_timing.csv");
    if(!c1.is_open() || !c1s.is_open() || !c1a.is_open() || !c2.is_open() || !c3a.is_open() || !c3t.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    for(auto* f : {&c1, &c1s, &c1a, &c2, &c3a, &c3t}) f->precision(10);

    //------------------------------------------------------------ Part 1
    const double S = 100.0, K = 100.0, r = 0.05, vol = 0.2, T = 1.0;
    const int reps = 32;
    const PathMethod methods[3] = {PathMethod::Mc, PathMethod::Sobol, PathMethod::SobolBridge};
    c1 << "steps,method,n_paths,rmse,rel_rmse_pct\n";
    c1s << "steps,method,rate,rmse_at_65536,mc_paths_equivalent_factor\n";
    for(Eigen::Index m : {4, 16, 64}){
        const double exact = geometric_asian_call(S, K, r, vol, T, m);
        double rmse_mc_last = 0.0;
        for(PathMethod method : methods){
            Rng rng(1200 + static_cast<uint64_t>(m) * 10 + static_cast<uint64_t>(method));
            std::vector<double> lx, ly;
            double last = 0.0;
            for(int k = 7; k <= 16; ++k){
                const std::size_t n = std::size_t{1} << k;
                double se = 0.0;
                for(int rep = 0; rep < reps; ++rep){
                    const double e = asian_call_estimate(S, K, r, vol, T, m, n, method, Average::Geometric, rng) - exact;
                    se += e * e;
                }
                const double rmse = std::sqrt(se / reps);
                c1 << m << "," << name(method) << "," << n << "," << rmse << "," << 100.0 * rmse / exact << "\n";
                lx.push_back(std::log(static_cast<double>(n)));
                ly.push_back(std::log(rmse));
                last = rmse;
            }
            //least-squares slope of log RMSE on log N: -0.5 for MC, towards -1 for QMC
            const double mx = std::accumulate(lx.begin(), lx.end(), 0.0) / static_cast<double>(lx.size());
            const double my = std::accumulate(ly.begin(), ly.end(), 0.0) / static_cast<double>(ly.size());
            double sxy = 0.0, sxx = 0.0;
            for(std::size_t i = 0; i < lx.size(); ++i){ sxy += (lx[i] - mx) * (ly[i] - my); sxx += (lx[i] - mx) * (lx[i] - mx); }
            if(method == PathMethod::Mc) rmse_mc_last = last;
            c1s << m << "," << name(method) << "," << sxy / sxx << "," << last << "," << (rmse_mc_last / last) * (rmse_mc_last / last) << "\n";
        }
        std::cout << "Part 1: " << m << " dates done\n";
    }
    c1a << "steps,method,n_paths,price,std_error,variance_reduction_vs_mc\n";
    for(Eigen::Index m : {16, 64}){
        double var_mc = 0.0;
        for(PathMethod method : methods){
            Rng rng(1300 + static_cast<uint64_t>(m) * 10 + static_cast<uint64_t>(method));
            std::vector<double> est;
            for(int rep = 0; rep < reps; ++rep) est.push_back(asian_call_estimate(S, K, r, vol, T, m, 16384, method, Average::Arithmetic, rng));
            const double mu = std::accumulate(est.begin(), est.end(), 0.0) / reps;
            double v = 0.0;
            for(double e : est) v += (e - mu) * (e - mu);
            v /= (reps - 1);
            if(method == PathMethod::Mc) var_mc = v;
            c1a << m << "," << name(method) << ",16384," << mu << "," << std::sqrt(v) << "," << var_mc / v << "\n";
        }
    }

    //------------------------------------------------------------ Part 2
    c2 << "option,greek,method,param,estimate,std_error,exact\n";
    {
        const double s = 100, k = 105, rr = 0.03, v = 0.25, t = 0.8;
        const pricing::BSParams p{s, k, rr, v, t, pricing::OptionType::Call};
        Rng rng(1400);
        const Eigen::VectorXd z = normals(100000, rng);
        const PathwiseGreeks pw = european_call_pathwise(s, k, rr, v, t, z);
        const LrGreeks lr = european_call_lr(s, k, rr, v, t, z);
        auto call = [&](double s0, double v0){
            return std::exp(-rr * t) * (s0 * ((rr - 0.5 * v0 * v0) * t + v0 * std::sqrt(t) * z.array()).exp() - k).max(0.0);
        };
        const GreekEstimate bd = mean_se((call(s + 1.0, v) - call(s - 1.0, v)) / 2.0);
        const GreekEstimate bv = mean_se((call(s, v + 0.0025) - call(s, v - 0.0025)) / 0.005);
        c2 << "european_call,delta,pathwise,," << pw.delta.value << "," << pw.delta.std_error << "," << pricing::delta(p) << "\n"
           << "european_call,delta,likelihood_ratio,," << lr.delta.value << "," << lr.delta.std_error << "," << pricing::delta(p) << "\n"
           << "european_call,delta,bump,h=1," << bd.value << "," << bd.std_error << "," << pricing::delta(p) << "\n"
           << "european_call,vega,pathwise,," << pw.vega.value << "," << pw.vega.std_error << "," << pricing::vega(p) << "\n"
           << "european_call,vega,likelihood_ratio,," << lr.vega.value << "," << lr.vega.std_error << "," << pricing::vega(p) << "\n"
           << "european_call,vega,bump,h=0.0025," << bv.value << "," << bv.std_error << "," << pricing::vega(p) << "\n";
    }
    {
        const double s = 100, k = 100, rr = 0.02, v = 0.2, t = 0.5;
        const double exact = digital_call_delta(s, k, rr, v, t);
        Rng rng(1401);
        const Eigen::VectorXd z = normals(100000, rng);
        const GreekEstimate lr = digital_call_delta_lr(s, k, rr, v, t, z);
        c2 << "digital_call,delta,pathwise,,0,0," << exact << "\n"
           << "digital_call,delta,likelihood_ratio,," << lr.value << "," << lr.std_error << "," << exact << "\n";
        for(double h : {0.01, 0.1, 1.0, 5.0, 20.0}){
            const GreekEstimate b = digital_call_delta_bump(s, k, rr, v, t, z, h);
            c2 << "digital_call,delta,bump,h=" << h << "," << b.value << "," << b.std_error << "," << exact << "\n";
        }
    }

    //------------------------------------------------------------ Part 3
    c3a << "n_assets,n_sensitivities,aad_vs_pathwise_max_abs_diff,aad_vs_bump_max_rel_diff_delta,aad_vs_bump_max_rel_diff_vega,price,delta_1,vega_1\n";
    c3t << "n_assets,n_sensitivities,price_ms,pathwise_ms,aad_ms,bump_ms,aad_over_price,bump_over_price,bump_over_aad\n";
    const std::size_t n_paths = 20000;
    for(Eigen::Index n : {1, 2, 5, 10, 20, 50, 100}){
        Eigen::VectorXd spots = Eigen::VectorXd::Constant(n, 100.0), w = Eigen::VectorXd::Constant(n, 1.0 / static_cast<double>(n));
        Eigen::VectorXd vols(n);
        for(Eigen::Index i = 0; i < n; ++i) vols(i) = n == 1 ? 0.25 : 0.15 + 0.2 * static_cast<double>(i) / static_cast<double>(n - 1);
        Eigen::MatrixXd corr = Eigen::MatrixXd::Constant(n, n, 0.3);
        corr.diagonal().setOnes();
        Rng rng(1500 + static_cast<uint64_t>(n));
        const std::vector<double> d = distributions::normal::sample(n_paths * static_cast<std::size_t>(n), 0.0, 1.0, rng);
        const Eigen::MatrixXd z = Eigen::Map<const Eigen::MatrixXd>(d.data(), static_cast<Eigen::Index>(n_paths), n);
        const double k = 100.0, rr = 0.03, t = 1.0;
        const BasketGreeks pw = basket_call_pathwise(spots, vols, corr, w, k, rr, t, z);
        const BasketGreeks ad = basket_call_aad(spots, vols, corr, w, k, rr, t, z);
        const BasketGreeks bp = basket_call_bump(spots, vols, corr, w, k, rr, t, z, 1e-4);
        double diff = std::max(std::abs(ad.d_rate - pw.d_rate), std::abs(ad.d_strike - pw.d_strike));
        diff = std::max({diff, (ad.d_spots - pw.d_spots).cwiseAbs().maxCoeff(), (ad.d_vols - pw.d_vols).cwiseAbs().maxCoeff()});
        const double rel_d = ((ad.d_spots - bp.d_spots).array() / ad.d_spots.array()).abs().maxCoeff();
        const double rel_v = ((ad.d_vols - bp.d_vols).array() / ad.d_vols.array()).abs().maxCoeff();
        c3a << n << "," << 2 * n + 2 << "," << diff << "," << rel_d << "," << rel_v << "," << ad.price << ","
            << ad.d_spots(0) << "," << ad.d_vols(0) << "\n";
        double sink = 0.0;
        const double t_price = median_ms([&]{ sink += basket_call_price(spots, vols, corr, w, k, rr, t, z); });
        const double t_pw = median_ms([&]{ sink += basket_call_pathwise(spots, vols, corr, w, k, rr, t, z).price; });
        const double t_aad = median_ms([&]{ sink += basket_call_aad(spots, vols, corr, w, k, rr, t, z).price; });
        const double t_bump = median_ms([&]{ sink += basket_call_bump(spots, vols, corr, w, k, rr, t, z).price; }, n >= 50 ? 1 : 3);
        c3t << n << "," << 2 * n + 2 << "," << t_price << "," << t_pw << "," << t_aad << "," << t_bump << ","
            << t_aad / t_price << "," << t_bump / t_price << "," << t_bump / t_aad << "\n";
        std::cout << "Part 3: N = " << n << (sink > 0.0 ? "" : " (no price?)") << "\n";
    }
    std::cout << "Results in " << out_dir << "\n";
    return 0;
}
