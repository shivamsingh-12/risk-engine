//Phase 16 experiment: the Merton structural credit model -- what its spreads look like, how first-passage default
//changes default probabilities, and how well a firm's asset volatility can be recovered from its equity
//Part 1: credit spread term structures for four firms (assets 100, r = 3%): safe (debt face 40, asset vol 20%),
//        medium (70, 25%), risky (95, 30%), distressed (120, 30%), maturities 0.1 to 30 years
//Part 2: Merton (default only at T) vs Black-Cox (default the first time assets touch the debt level) real-world
//        default probabilities for the medium firm with asset drift 8%, horizons 0.25 to 10 years, with a
//        Monte Carlo check of the Black-Cox formula
//Part 3: estimation. Simulated firms (assets 100, debt 60, asset vol 25%, one year of daily data, 300 firms per
//        scenario): asset vol from the static two-equation solve with historical equity vol vs the KMV iteration,
//        in a normal year (drift +8%) and a year of falling assets (drift -30%)
//Part 4: the six bundled stocks with HYPOTHETICAL debt (face value 0.5, 1 and 2 times market equity, one-year
//        maturity, r = 4%): the bundled data has no balance sheets, so this only shows how equity volatility and
//        leverage translate into asset volatility, distance to default and spreads. It is not a credit assessment
//        of these companies
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "qre/credit/merton_credit.hpp"
#include "qre/data/csv_loader.hpp"
#include "qre/data/quality.hpp"
using namespace qre;
using namespace qre::credit;

namespace {
double ann_log_vol(const std::vector<double>& x){
    const std::size_t n = x.size() - 1;
    double m = 0.0, s = 0.0;
    for(std::size_t i = 0; i < n; ++i) m += std::log(x[i + 1] / x[i]);
    m /= static_cast<double>(n);
    for(std::size_t i = 0; i < n; ++i){ const double d = std::log(x[i + 1] / x[i]) - m; s += d * d; }
    return std::sqrt(s / static_cast<double>(n - 1) * 252.0);
}
}

int main(){
    const std::string out_dir = "cpp/experiments/credit/results/";
    std::ofstream c1(out_dir + "term_structure.csv"), c2(out_dir + "first_passage.csv"), c3(out_dir + "estimation.csv"),
                  c3s(out_dir + "estimation_summary.csv"), c4(out_dir + "hypothetical_leverage.csv");
    if(!c1.is_open() || !c2.is_open() || !c3.is_open() || !c3s.is_open() || !c4.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    for(auto* f : {&c1, &c2, &c3, &c3s, &c4}) f->precision(10);
    const double r = 0.03;

    //------------------------------------------------------------ Part 1
    c1 << "firm,debt_face,asset_vol,maturity,spread_bp,risk_neutral_pd,recovery,equity,leverage_market\n";
    struct Firm{ const char* name; double D, sigma; };
    const Firm firms[4] = {{"safe", 40, 0.20}, {"medium", 70, 0.25}, {"risky", 95, 0.30}, {"distressed", 120, 0.30}};
    for(const Firm& f : firms){
        for(double T : {0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0}){
            const double e = equity_value(100.0, f.D, r, f.sigma, T);
            c1 << f.name << "," << f.D << "," << f.sigma << "," << T << "," << 1e4 * credit_spread(100.0, f.D, r, f.sigma, T) << ","
               << risk_neutral_pd(100.0, f.D, r, f.sigma, T) << "," << recovery_rate(100.0, f.D, r, f.sigma, T) << "," << e << ","
               << (100.0 - e) / 100.0 << "\n";
        }
    }

    //------------------------------------------------------------ Part 2
    c2 << "horizon,merton_pd,black_cox_pd,ratio,mc_pd,mc_se\n";
    for(double T : {0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0}){
        const double pm = physical_pd(100.0, 70.0, 0.08, 0.25, T), pb = black_cox_pd(100.0, 70.0, 0.08, 0.25, T);
        const auto steps = static_cast<int>(std::lround(252.0 * T));
        const double mc = first_passage_mc(100.0, 70.0, 0.08, 0.25, T, 100000, steps, 2100 + static_cast<uint64_t>(steps));
        c2 << T << "," << pm << "," << pb << "," << pb / pm << "," << mc << "," << std::sqrt(mc * (1 - mc) / 100000.0) << "\n";
    }
    std::cout << "Parts 1-2 done\n";

    //------------------------------------------------------------ Part 3
    c3 << "scenario,firm,true_sigma_sample,naive_sigma,kmv_sigma,true_pd_rn,naive_pd_rn,kmv_pd_rn,last_V,kmv_iterations\n";
    c3s << "scenario,estimator,sigma_bias,sigma_rmse,pd_bias,pd_rmse,pd_mean_true\n";
    for(double mu : {0.08, -0.30}){
        const std::string scen = mu > 0 ? "normal_year" : "falling_assets";
        std::mt19937_64 eng(mu > 0 ? 2101 : 2102);
        std::normal_distribution<double> nd(0.0, 1.0);
        const double sig = 0.25, D = 60.0, dt = 1.0 / 252.0;
        double nb = 0, nr = 0, kb = 0, kr = 0, npb = 0, npr = 0, kpb = 0, kpr = 0, ptrue = 0;
        const int firms_n = 300;
        for(int k = 0; k < firms_n; ++k){
            std::vector<double> V{100.0}, E;
            for(int t = 0; t < 250; ++t) V.push_back(V.back() * std::exp((mu - 0.5 * sig * sig) * dt + sig * std::sqrt(dt) * nd(eng)));
            for(double v : V) E.push_back(equity_value(v, D, r, sig, 1.0));
            const double se = ann_log_vol(E);
            const AssetSolution naive = solve_assets(E.back(), se, D, r, 1.0);
            const KmvResult km = kmv_iterative(E, D, r);
            const double pt = risk_neutral_pd(V.back(), D, r, sig, 1.0);
            const double pn = risk_neutral_pd(naive.V, D, r, naive.sigma, 1.0);
            const double pk = risk_neutral_pd(km.assets.back(), D, r, km.sigma, 1.0);
            c3 << scen << "," << k << "," << ann_log_vol(V) << "," << naive.sigma << "," << km.sigma << "," << pt << "," << pn << "," << pk
               << "," << V.back() << "," << km.iterations << "\n";
            nb += naive.sigma - sig; nr += (naive.sigma - sig) * (naive.sigma - sig);
            kb += km.sigma - sig; kr += (km.sigma - sig) * (km.sigma - sig);
            npb += pn - pt; npr += (pn - pt) * (pn - pt); kpb += pk - pt; kpr += (pk - pt) * (pk - pt); ptrue += pt;
        }
        const double n = firms_n;
        c3s << scen << ",static_two_equation," << nb / n << "," << std::sqrt(nr / n) << "," << npb / n << "," << std::sqrt(npr / n) << "," << ptrue / n << "\n";
        c3s << scen << ",kmv_iterative," << kb / n << "," << std::sqrt(kr / n) << "," << kpb / n << "," << std::sqrt(kpr / n) << "," << ptrue / n << "\n";
    }
    std::cout << "Part 3 done\n";

    //------------------------------------------------------------ Part 4
    data::PriceTable table;
    try{ table = data::load_price_csv("data/market_data/adjusted_close.csv"); }
    catch(const std::exception& e){
        std::cerr << e.what() << "\nRun from repo root \n";
        return 1;
    }
    const auto hardened = data::harden(table.prices);
    if(!hardened.report.clean()){ std::cerr << "price data failed the quality report\n"; return 1; }
    const double r4 = 0.04;
    c4 << "ticker,equity_vol,debt_to_equity,asset_vol,asset_to_equity,distance_to_default_rn,risk_neutral_pd,spread_bp\n";
    const Eigen::Index T = hardened.prices.rows();
    for(Eigen::Index j = 0; j < hardened.prices.cols(); ++j){
        std::vector<double> E;
        for(Eigen::Index t = T - 251; t < T; ++t) E.push_back(100.0 * hardened.prices(t, j) / hardened.prices(T - 1, j));
        const double se = ann_log_vol(E);
        for(double lev : {0.5, 1.0, 2.0}){
            const double D = 100.0 * lev;
            const KmvResult km = kmv_iterative(E, D, r4);
            const double V = km.assets.back();
            c4 << table.tickers[static_cast<std::size_t>(j)] << "," << se << "," << lev << "," << km.sigma << "," << V / 100.0 << ","
               << distance_to_default(V, D, r4, km.sigma, 1.0) << "," << risk_neutral_pd(V, D, r4, km.sigma, 1.0) << ","
               << 1e4 * credit_spread(V, D, r4, km.sigma, 1.0) << "\n";
        }
    }
    std::cout << "Results in " << out_dir << "\n";
    return 0;
}
