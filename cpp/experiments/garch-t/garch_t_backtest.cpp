//Student-t GARCH experiment: do fat-tailed shocks fix the ES understatement of the normal models?
//Part 1: rolling out-of-sample backtest (same setup as Experiments 9-11) of GARCH with Student-t shocks next to
//        GARCH-normal, FHS-GARCH and the earlier models, at 99% and 97.5%
//Part 2: where do the fat tails come from? Compare the excess kurtosis of raw portfolio returns with that of
//        GARCH-standardized residuals (volatility clustering alone makes returns look fat-tailed), and track the
//        estimated nu over the backtest
//C++ only: Student-t functions and the GARCH-t fit are cross-validated against scipy / arch in the tests
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "qre/data/csv_loader.hpp"
#include "qre/data/point_in_time.hpp"
#include "qre/data/quality.hpp"
#include "qre/risk/backtest.hpp"
#include "qre/risk/garch.hpp"
using namespace qre::risk;

namespace {
std::size_t breaches_in(const BacktestSeries& s, const std::vector<std::string>& dates, const std::string& year){
    std::size_t n = 0;
    for(std::size_t i = 0; i < s.rows.size(); ++i)
        if(dates[static_cast<std::size_t>(s.rows[i])].substr(0, 4) == year && s.loss[i] > s.var[i]) ++n;
    return n;
}
double excess_kurtosis(const std::vector<double>& x){
    const double m = qre::estimators::empirical::sample_mean(x);
    double m2 = 0.0, m4 = 0.0;
    for(double v : x){ const double d = v - m; m2 += d * d; m4 += d * d * d * d; }
    m2 /= static_cast<double>(x.size()); m4 /= static_cast<double>(x.size());
    return m4 / (m2 * m2) - 3.0;
}
double quantile(std::vector<double> v, double q){
    std::sort(v.begin(), v.end());
    const double h = static_cast<double>(v.size() - 1) * q;
    const auto lo = static_cast<std::size_t>(std::floor(h));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (h - static_cast<double>(lo)) * (v[hi] - v[lo]);
}
}

int main(){
    const std::string out_dir = "cpp/experiments/garch-t/results/";
    qre::data::PriceTable table;
    try{ table = qre::data::load_price_csv("data/market_data/adjusted_close.csv"); }
    catch(const std::exception& e){
        std::cerr << e.what() << "\nRun from repo root \n";
        return 1;
    }
    const auto quality = qre::data::harden(table.prices);
    if(!quality.report.clean()){
        std::cerr << "price data failed the quality report\n";
        return 1;
    }
    std::ofstream summary(out_dir + "method_comparison.csv");
    std::ofstream tails(out_dir + "tail_diagnostics.csv");
    std::ofstream nus(out_dir + "daily_nu.csv");
    if(!summary.is_open() || !tails.is_open() || !nus.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    summary.precision(10); tails.precision(10); nus.precision(10);

    const qre::PriceMatrix& prices = quality.prices;
    const qre::WeightVector w = qre::WeightVector::Constant(prices.cols(), 1.0 / static_cast<double>(prices.cols()));
    const Eigen::Index window = 250;

    //------------------------------------------------------------ Part 1
    struct M{ const char* name; BacktestMethod m; };
    const M methods[] = {{"historical", BacktestMethod::Historical}, {"parametric", BacktestMethod::Parametric},
                         {"fhs", BacktestMethod::FilteredHistorical}, {"garch", BacktestMethod::GarchParametric},
                         {"fhs_garch", BacktestMethod::GarchFiltered}, {"garch_t", BacktestMethod::GarchStudentT}};
    summary << "confidence,method,n_obs,breaches,expected,breaches_2022,breaches_2023,kupiec_p,christoffersen_p,cc_p,z2,mean_var,mean_es\n";
    for(double conf : {0.99, 0.975}){
        for(const M& m : methods){
            const BacktestSeries s = rolling_backtest(prices, w, window, conf, m.m, 20000, 2029, 252, 0.94);
            const BacktestSummary r = summarize(s, conf);
            double mv = 0.0, me = 0.0;
            for(std::size_t i = 0; i < s.var.size(); ++i){ mv += s.var[i]; me += s.es[i]; }
            mv /= static_cast<double>(s.var.size()); me /= static_cast<double>(s.var.size());
            summary << conf << "," << m.name << "," << r.n_obs << "," << r.n_breaches << "," << static_cast<double>(r.n_obs) * (1.0 - conf) << ","
                    << breaches_in(s, table.dates, "2022") << "," << breaches_in(s, table.dates, "2023") << "," << r.kupiec.p_value << ","
                    << r.christoffersen.p_value << "," << r.conditional.p_value << "," << r.z2 << "," << mv << "," << me << "\n";
            std::cout << "c=" << conf << " " << m.name << ": " << r.n_breaches << " (2022 " << breaches_in(s, table.dates, "2022") << ", 2023 "
                      << breaches_in(s, table.dates, "2023") << ") Kupiec p=" << r.kupiec.p_value << " Christ p=" << r.christoffersen.p_value
                      << " CC p=" << r.conditional.p_value << " Z2=" << r.z2 << " meanVaR=" << mv << " meanES=" << me << "\n";
        }
    }

    //------------------------------------------------------------ Part 2
    const std::vector<double> all = portfolio_returns(qre::data::simple_return(prices), w);
    const GarchFit gn = fit_garch(all);
    const GarchFit gt = fit_garch(all, GarchDist::StudentT);
    const auto h = garch_variance_path(all, gn.omega, gn.alpha, gn.beta, gn.long_run_var);
    std::vector<double> z(all.size());
    for(std::size_t t = 0; t < all.size(); ++t) z[t] = all[t] / std::sqrt(h[t]);
    tails << "series,n,excess_kurtosis,garch_normal_loglik,garch_t_loglik,nu_full_history\n";
    tails << "raw_portfolio_returns," << all.size() << "," << excess_kurtosis(all) << "," << gn.loglik << "," << gt.loglik << "," << gt.nu << "\n";
    tails << "garch_standardized_residuals," << z.size() << "," << excess_kurtosis(z) << ",,,\n";
    std::cout << "\nfull history: raw excess kurtosis " << excess_kurtosis(all) << ", GARCH-standardized residuals " << excess_kurtosis(z)
              << "\nGARCH-normal loglik " << gn.loglik << ", GARCH-t loglik " << gt.loglik << " (nu = " << gt.nu
              << ", alpha " << gt.alpha << ", beta " << gt.beta << ")\n";

    std::vector<double> nu_250;
    nus << "date,nu_250_day_window\n";
    for(Eigen::Index t = window + 1; t < prices.rows(); ++t){
        const auto r = portfolio_returns(qre::data::simple_return(qre::data::pit_window(prices, t, window)), w);
        const double nu = fit_garch(r, GarchDist::StudentT).nu;
        nu_250.push_back(nu);
        nus << table.dates[static_cast<std::size_t>(t)] << "," << nu << "\n";
    }
    const double share_big = static_cast<double>(std::count_if(nu_250.begin(), nu_250.end(), [](double x){ return x > 30.0; })) / static_cast<double>(nu_250.size());
    std::cout << "nu on 250-day windows: median " << quantile(nu_250, 0.5) << " [p10 " << quantile(nu_250, 0.1) << ", p90 "
              << quantile(nu_250, 0.9) << "], share above 30: " << share_big << "\n";
    tails << "nu_250_day_windows," << nu_250.size() << ",,,," << quantile(nu_250, 0.5) << "\n";
    std::cout << "Results written to " << out_dir << "\n";
    return 0;
}
