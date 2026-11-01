//GARCH experiment: does estimating the decay (GARCH) beat fixing it (EWMA), and how stable are the estimates?
//Part 1: rolling out-of-sample backtest (same setup as Experiments 9 and 10) of GARCH-normal and FHS-GARCH
//        next to EWMA, FHS and the equal-weight baselines, at 99% and 97.5%
//Part 2: fit diagnostics. GARCH is refitted every day on the 250-day portfolio return window; record the
//        estimated alpha, beta and persistence each day, and compare with a 500-day estimation window
//        over the forecast days both can cover. Also a single fit on the whole history for reference
//C++ only: GARCH and Nelder-Mead are cross-validated against scipy (and the arch package) in the tests
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

double quantile(std::vector<double> v, double q){
    std::sort(v.begin(), v.end());
    const double h = static_cast<double>(v.size() - 1) * q;
    const auto lo = static_cast<std::size_t>(std::floor(h));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (h - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

double change_sd(const std::vector<double>& v){
    std::vector<double> d;
    for(std::size_t i = 1; i < v.size(); ++i) d.push_back(v[i] - v[i - 1]);
    return qre::estimators::empirical::sample_stddev(d);
}
}

int main(){
    const std::string out_dir = "cpp/experiments/garch/results/";
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
    std::ofstream params(out_dir + "daily_garch_params.csv");
    std::ofstream diag(out_dir + "fit_diagnostics.csv");
    if(!summary.is_open() || !params.is_open() || !diag.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    summary.precision(10); params.precision(10); diag.precision(10);

    const qre::PriceMatrix& prices = quality.prices;
    const qre::WeightVector w = qre::WeightVector::Constant(prices.cols(), 1.0 / static_cast<double>(prices.cols()));
    const Eigen::Index window = 250;

    //------------------------------------------------------------ Part 1: backtest
    struct M{ const char* name; BacktestMethod m; };
    const M methods[] = {{"historical", BacktestMethod::Historical}, {"parametric", BacktestMethod::Parametric},
                         {"ewma", BacktestMethod::EwmaParametric}, {"fhs", BacktestMethod::FilteredHistorical},
                         {"garch", BacktestMethod::GarchParametric}, {"fhs_garch", BacktestMethod::GarchFiltered}};
    summary << "confidence,method,n_obs,breaches,expected,breaches_2022,breaches_2023,kupiec_p,christoffersen_p,cc_p,z2,mean_var,var_daily_change_sd\n";
    for(double conf : {0.99, 0.975}){
        for(const M& m : methods){
            const BacktestSeries s = rolling_backtest(prices, w, window, conf, m.m, 20000, 2029, 252, 0.94);
            const BacktestSummary r = summarize(s, conf);
            double mean_var = 0.0;
            for(double v : s.var) mean_var += v;
            mean_var /= static_cast<double>(s.var.size());
            summary << conf << "," << m.name << "," << r.n_obs << "," << r.n_breaches << "," << static_cast<double>(r.n_obs) * (1.0 - conf) << ","
                    << breaches_in(s, table.dates, "2022") << "," << breaches_in(s, table.dates, "2023") << "," << r.kupiec.p_value << ","
                    << r.christoffersen.p_value << "," << r.conditional.p_value << "," << r.z2 << "," << mean_var << "," << change_sd(s.var) << "\n";
            std::cout << "c=" << conf << " " << m.name << ": " << r.n_breaches << " (2022 " << breaches_in(s, table.dates, "2022") << ", 2023 "
                      << breaches_in(s, table.dates, "2023") << ") Kupiec p=" << r.kupiec.p_value << " Christ p=" << r.christoffersen.p_value
                      << " CC p=" << r.conditional.p_value << " Z2=" << r.z2 << " meanVaR=" << mean_var << " dVaR sd=" << change_sd(s.var) << "\n";
        }
    }

    //------------------------------------------------------------ Part 2: fit diagnostics
    const std::vector<double> all = portfolio_returns(qre::data::simple_return(prices), w);
    const GarchFit full = fit_garch(all);
    std::cout << "\nfull-history fit (" << all.size() << " returns): alpha=" << full.alpha << " beta=" << full.beta
              << " persistence=" << full.persistence() << "\n";

    params << "date,window,alpha,beta,persistence,converged\n";
    diag << "window,n_fits,alpha_median,alpha_p10,alpha_p90,beta_median,beta_p10,beta_p90,persistence_median,"
            "share_persistence_above_0_99,share_alpha_below_0_01,share_not_converged,alpha_daily_change_sd,persistence_daily_change_sd\n";
    diag << "full_history,1," << full.alpha << ",,," << full.beta << ",,," << full.persistence() << ",,,,,\n";
    const Eigen::Index first_common = 501;     //first forecast day a 500-day window can serve
    for(Eigen::Index win : {Eigen::Index(250), Eigen::Index(500)}){
        std::vector<double> a, b, p;
        std::size_t not_conv = 0;
        for(Eigen::Index t = first_common; t < prices.rows(); ++t){
            const auto rets = portfolio_returns(qre::data::simple_return(qre::data::pit_window(prices, t, win)), w);
            const GarchFit f = fit_garch(rets);
            a.push_back(f.alpha); b.push_back(f.beta); p.push_back(f.persistence());
            if(!f.converged) ++not_conv;
            params << table.dates[static_cast<std::size_t>(t)] << "," << win << "," << f.alpha << "," << f.beta << "," << f.persistence()
                   << "," << (f.converged ? "true" : "false") << "\n";
        }
        const double n = static_cast<double>(a.size());
        const auto share = [&](const std::vector<double>& v, auto pred){ return static_cast<double>(std::count_if(v.begin(), v.end(), pred)) / n; };
        const double hi_p = share(p, [](double x){ return x > 0.99; });
        const double lo_a = share(a, [](double x){ return x < 0.01; });
        diag << win << "," << a.size() << "," << quantile(a, 0.5) << "," << quantile(a, 0.1) << "," << quantile(a, 0.9) << ","
             << quantile(b, 0.5) << "," << quantile(b, 0.1) << "," << quantile(b, 0.9) << "," << quantile(p, 0.5) << ","
             << hi_p << "," << lo_a << "," << static_cast<double>(not_conv) / n << "," << change_sd(a) << "," << change_sd(p) << "\n";
        std::cout << "window " << win << " (" << a.size() << " fits from " << table.dates[static_cast<std::size_t>(first_common)] << "): alpha median "
                  << quantile(a, 0.5) << " [" << quantile(a, 0.1) << ", " << quantile(a, 0.9) << "]  beta median " << quantile(b, 0.5)
                  << " [" << quantile(b, 0.1) << ", " << quantile(b, 0.9) << "]  persistence median " << quantile(p, 0.5)
                  << "  share p>0.99 " << hi_p << "  share alpha<0.01 " << lo_a << "  not converged " << not_conv
                  << "  d(alpha) sd " << change_sd(a) << "  d(persistence) sd " << change_sd(p) << "\n";
    }
    std::cout << "Results written to " << out_dir << "\n";
    return 0;
}
