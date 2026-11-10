//Phase 11 experiment: tail risk with extreme value theory, and how uncertain risk numbers really are
//Part 1: known tail. Student-t losses with 4 degrees of freedom (true tail index xi = 0.25), 2,000 observations,
//        200 replications. VaR at 99%, 99.5%, 99.9% from historical, normal and peaks-over-threshold (POT) vs the true
//        quantile; tail index from the GPD fit at several thresholds and from Hill
//Part 2: real data. POT tail fit of the 6-stock portfolio's daily losses, compared with historical and normal, plus
//        iid and block bootstrap intervals for historical VaR/ES
//Part 3: backtest conditional EVT (GARCH + GPD) against FHS, FHS-GARCH and GARCH-t, with how often xi hits its bound
//Part 4: model risk. Next-day 99% VaR and ES from all ten of the engine's models on the last 250 days, compared with
//        the statistical uncertainty of a single model (bootstrap interval)
//C++ only: GPD, POT, Hill and conditional EVT are cross-validated against the Python reference (GPD vs scipy)
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "qre/data/csv_loader.hpp"
#include "qre/data/point_in_time.hpp"
#include "qre/data/quality.hpp"
#include "qre/distributions/student_t.hpp"
#include "qre/linalg/covariance.hpp"
#include "qre/risk/backtest.hpp"
#include "qre/risk/evt.hpp"
using namespace qre;
using namespace qre::risk;

namespace {
double mean(const std::vector<double>& v){ double s = 0; for(double x : v) s += x; return s / static_cast<double>(v.size()); }
double rmse_vs(const std::vector<double>& v, double truth){ double s = 0; for(double x : v) s += (x - truth) * (x - truth); return std::sqrt(s / static_cast<double>(v.size())); }
std::size_t breaches_in(const BacktestSeries& s, const std::vector<std::string>& dates, const std::string& year){
    std::size_t n = 0;
    for(std::size_t i = 0; i < s.rows.size(); ++i)
        if(dates[static_cast<std::size_t>(s.rows[i])].substr(0, 4) == year && s.loss[i] > s.var[i]) ++n;
    return n;
}
}

int main(){
    const std::string out_dir = "cpp/experiments/09-evt/results/";
    std::ofstream p1(out_dir + "known_tail.csv"), p1b(out_dir + "threshold_sensitivity.csv"), p2(out_dir + "real_tail.csv"),
                  p3(out_dir + "backtest.csv"), p4(out_dir + "model_risk.csv");
    if(!p1.is_open() || !p1b.is_open() || !p2.is_open() || !p3.is_open() || !p4.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    for(auto* f : {&p1, &p1b, &p2, &p3, &p4}) f->precision(10);

    //------------------------------------------------------------ Part 1
    const double nu = 4.0;
    const int reps = 200;
    const std::size_t n = 2000;
    std::mt19937_64 eng(2035);
    std::student_t_distribution<double> tdist(nu);
    const double levels[3] = {0.99, 0.995, 0.999};
    std::vector<double> est[3][3];              //[level][method: historical, normal, pot]
    std::vector<double> xi_th[4], xi_hill;
    const double thresholds[4] = {0.80, 0.90, 0.95, 0.98};
    int bound_hits[4] = {0, 0, 0, 0};
    for(int r = 0; r < reps; ++r){
        std::vector<double> x(n);
        for(auto& v : x) v = tdist(eng);
        const double sd = estimators::empirical::sample_stddev(x), mu = estimators::empirical::sample_mean(x);
        for(int l = 0; l < 3; ++l){
            est[l][0].push_back(var_es_from_losses(x, levels[l]).var);
            est[l][1].push_back(mu + sd * risk::detail::normal_quantile(levels[l]));
            est[l][2].push_back(pot_var_es(x, levels[l], 0.90).risk.var);
        }
        for(int k = 0; k < 4; ++k){
            const PotResult pr = pot_var_es(x, 0.999, thresholds[k]);
            xi_th[k].push_back(pr.xi);
            bound_hits[k] += pr.at_bound ? 1 : 0;
        }
        xi_hill.push_back(hill_estimator(x, 200));
    }
    p1 << "confidence,true_quantile,method,mean_estimate,bias_pct,rmse_pct\n";
    const char* mnames[3] = {"historical", "normal", "pot_evt"};
    for(int l = 0; l < 3; ++l){
        const double truth = distributions::student_t::quantile(levels[l], nu);
        for(int m = 0; m < 3; ++m){
            const double b = 100.0 * (mean(est[l][m]) / truth - 1.0), e = 100.0 * rmse_vs(est[l][m], truth) / truth;
            p1 << levels[l] << "," << truth << "," << mnames[m] << "," << mean(est[l][m]) << "," << b << "," << e << "\n";
            std::cout << "t4 VaR" << levels[l] << " (true " << truth << ") " << mnames[m] << ": bias " << b << "%  rmse " << e << "%\n";
        }
    }
    p1b << "estimator,threshold_quantile,exceedances,mean_xi,sd_xi,xi_bound_hits,true_xi\n";
    for(int k = 0; k < 4; ++k){
        p1b << "gpd_mle," << thresholds[k] << "," << static_cast<double>(n) * (1.0 - thresholds[k]) << "," << mean(xi_th[k]) << ","
            << estimators::empirical::sample_stddev(xi_th[k]) << "," << bound_hits[k] << "," << 1.0 / nu << "\n";
        std::cout << "GPD xi at threshold " << thresholds[k] << ": mean " << mean(xi_th[k]) << " sd " << estimators::empirical::sample_stddev(xi_th[k])
                  << " bound hits " << bound_hits[k] << "\n";
    }
    p1b << "hill,0.90,200," << mean(xi_hill) << "," << estimators::empirical::sample_stddev(xi_hill) << ",0," << 1.0 / nu << "\n";
    std::cout << "Hill xi (k=200): mean " << mean(xi_hill) << " sd " << estimators::empirical::sample_stddev(xi_hill) << "\n\n";

    //------------------------------------------------------------ data
    data::PriceTable table;
    try{ table = data::load_price_csv("data/market_data/adjusted_close.csv"); }
    catch(const std::exception& e){
        std::cerr << e.what() << "\nRun from repo root \n";
        return 1;
    }
    const auto quality = data::harden(table.prices);
    if(!quality.report.clean()){
        std::cerr << "price data failed the quality report\n";
        return 1;
    }
    const PriceMatrix& prices = quality.prices;
    const WeightVector w = WeightVector::Constant(prices.cols(), 1.0 / static_cast<double>(prices.cols()));
    const auto port = portfolio_returns(data::simple_return(prices), w);
    std::vector<double> losses(port.size());
    for(std::size_t i = 0; i < port.size(); ++i) losses[i] = -port[i];

    //------------------------------------------------------------ Part 2
    const double sd = estimators::empirical::sample_stddev(losses);
    p2 << "confidence,historical_var,historical_es,normal_var,normal_es,pot_var,pot_es,pot_xi,pot_beta,threshold,exceedances\n";
    for(double c : {0.99, 0.995, 0.999}){
        const VarEs h = var_es_from_losses(losses, c);
        const VarEs nrm = parametric_var_es(0.0, sd, c);
        const PotResult pr = pot_var_es(losses, c, 0.90);
        p2 << c << "," << h.var << "," << h.es << "," << nrm.var << "," << nrm.es << "," << pr.risk.var << "," << pr.risk.es << ","
           << pr.xi << "," << pr.beta << "," << pr.threshold << "," << pr.n_exceed << "\n";
        std::cout << "real c=" << c << ": hist " << h.var << "/" << h.es << "  normal " << nrm.var << "/" << nrm.es << "  POT " << pr.risk.var
                  << "/" << pr.risk.es << "  (xi " << pr.xi << ", " << pr.n_exceed << " exceedances)\n";
    }
    for(std::size_t k : {25u, 50u, 75u, 100u}) std::cout << "real Hill xi k=" << k << ": " << hill_estimator(losses, k) << "\n";
    Rng brng(2036);
    const BootstrapCi iid = bootstrap_var_es(losses, 0.99, 2000, 1, brng);
    const BootstrapCi blk = bootstrap_var_es(losses, 0.99, 2000, 10, brng);
    std::ofstream p2b(out_dir + "real_bootstrap.csv");
    p2b.precision(10);
    p2b << "bootstrap,block,var99,var_lo,var_hi,es99,es_lo,es_hi\n";
    const VarEs h99 = var_es_from_losses(losses, 0.99);
    p2b << "iid,1," << h99.var << "," << iid.var_lo << "," << iid.var_hi << "," << h99.es << "," << iid.es_lo << "," << iid.es_hi << "\n";
    p2b << "block,10," << h99.var << "," << blk.var_lo << "," << blk.var_hi << "," << h99.es << "," << blk.es_lo << "," << blk.es_hi << "\n";
    std::cout << "bootstrap 90% CI hist VaR99 " << h99.var << ": iid [" << iid.var_lo << ", " << iid.var_hi << "] block [" << blk.var_lo << ", " << blk.var_hi
              << "]; ES99 " << h99.es << ": iid [" << iid.es_lo << ", " << iid.es_hi << "] block [" << blk.es_lo << ", " << blk.es_hi << "]\n\n";

    //------------------------------------------------------------ Part 3
    struct M{ const char* name; BacktestMethod m; };
    const M methods[] = {{"fhs", BacktestMethod::FilteredHistorical}, {"fhs_garch", BacktestMethod::GarchFiltered},
                         {"garch_t", BacktestMethod::GarchStudentT}, {"garch_evt", BacktestMethod::GarchEvt}};
    p3 << "confidence,method,breaches,expected,breaches_2022,breaches_2023,kupiec_p,christoffersen_p,cc_p,z2,mean_var,mean_es\n";
    for(double c : {0.99, 0.975}){
        for(const M& m : methods){
            const BacktestSeries s = rolling_backtest(prices, w, 250, c, m.m, 20000, 2029, 252, 0.94);
            const BacktestSummary r = summarize(s, c);
            double mv = 0, me = 0;
            for(std::size_t i = 0; i < s.var.size(); ++i){ mv += s.var[i]; me += s.es[i]; }
            mv /= static_cast<double>(s.var.size()); me /= static_cast<double>(s.var.size());
            p3 << c << "," << m.name << "," << r.n_breaches << "," << static_cast<double>(r.n_obs) * (1.0 - c) << "," << breaches_in(s, table.dates, "2022") << ","
               << breaches_in(s, table.dates, "2023") << "," << r.kupiec.p_value << "," << r.christoffersen.p_value << "," << r.conditional.p_value << ","
               << r.z2 << "," << mv << "," << me << "\n";
            std::cout << "c=" << c << " " << m.name << ": " << r.n_breaches << " (2022 " << breaches_in(s, table.dates, "2022") << ", 2023 "
                      << breaches_in(s, table.dates, "2023") << ") Kupiec p=" << r.kupiec.p_value << " Christ p=" << r.christoffersen.p_value
                      << " Z2=" << r.z2 << " meanVaR=" << mv << " meanES=" << me << "\n";
        }
    }
    {   //xi diagnostics for conditional EVT on every backtest window
        std::vector<double> xis;
        int hits = 0;
        for(Eigen::Index t = 251; t < prices.rows(); ++t){
            const auto r = portfolio_returns(data::simple_return(data::pit_window(prices, t, 250)), w);
            const GarchFit g = fit_garch(r);
            const auto hh = garch_variance_path(r, g.omega, g.alpha, g.beta, g.long_run_var);
            std::vector<double> z(r.size());
            for(std::size_t i = 0; i < r.size(); ++i) z[i] = -r[i] / std::sqrt(hh[i]);
            const PotResult pr = pot_var_es(z, 0.99, 0.90);
            xis.push_back(pr.xi);
            hits += pr.at_bound ? 1 : 0;
        }
        std::vector<double> sorted = xis;
        std::sort(sorted.begin(), sorted.end());
        std::ofstream p3b(out_dir + "conditional_evt_xi.csv");
        p3b.precision(10);
        p3b << "windows,exceedances_per_window,xi_median,xi_p10,xi_p90,xi_bound_hits\n"
            << xis.size() << ",25," << sorted[sorted.size() / 2] << "," << sorted[sorted.size() / 10] << "," << sorted[sorted.size() * 9 / 10] << "," << hits << "\n";
        std::cout << "conditional EVT xi over " << xis.size() << " windows: median " << sorted[sorted.size() / 2] << " [p10 " << sorted[sorted.size() / 10]
                  << ", p90 " << sorted[sorted.size() * 9 / 10] << "], bound hits " << hits << "\n\n";
    }

    //------------------------------------------------------------ Part 4
    const Eigen::Index T = prices.rows();
    const ReturnMatrix rets = data::simple_return(data::pit_window(prices, T, 250));
    const auto pr = portfolio_returns(rets, w);
    std::vector<double> lw(pr.size());
    for(std::size_t i = 0; i < pr.size(); ++i) lw[i] = -pr[i];
    const CovarianceMatrix cov = linalg::sample_cov(rets);
    Rng mc_rng(2037);
    struct F{ const char* name; VarEs v; };
    const std::vector<F> f = {
        {"historical", historical_var_es(pr, 0.99)},
        {"parametric", parametric_var_es(0.0, portfolio_vol(w, cov), 0.99)},
        {"monte_carlo", monte_carlo_var_es(w, cov, 0.99, 1, 200000, mc_rng)},
        {"ewma", ewma_parametric_var_es(rets, w, 0.99, 0.94)},
        {"fhs", fhs_var_es(pr, 0.99, 0.94)},
        {"garch", garch_var_es(pr, 0.99)},
        {"fhs_garch", garch_var_es(pr, 0.99, true)},
        {"garch_t", garch_var_es(pr, 0.99, false, GarchDist::StudentT)},
        {"pot_evt", pot_var_es(lw, 0.99, 0.90).risk},
        {"garch_evt", garch_evt_var_es(pr, 0.99)},
    };
    p4 << "model,var99,es99\n";
    std::vector<double> vs, es;
    for(const auto& m : f){
        p4 << m.name << "," << m.v.var << "," << m.v.es << "\n";
        vs.push_back(m.v.var); es.push_back(m.v.es);
        std::cout << "next day after " << table.dates.back() << " " << m.name << ": VaR99 " << m.v.var << " ES99 " << m.v.es << "\n";
    }
    Rng b2(2038);
    const BootstrapCi last = bootstrap_var_es(lw, 0.99, 2000, 10, b2);
    const auto [vmin, vmax] = std::minmax_element(vs.begin(), vs.end());
    const auto [emin, emax] = std::minmax_element(es.begin(), es.end());
    std::ofstream p4b(out_dir + "model_risk_summary.csv");
    p4b.precision(10);
    p4b << "measure,var_low,var_high,es_low,es_high\n";
    p4b << "across_10_models," << *vmin << "," << *vmax << "," << *emin << "," << *emax << "\n";
    p4b << "historical_block_bootstrap_90pct," << last.var_lo << "," << last.var_hi << "," << last.es_lo << "," << last.es_hi << "\n";
    std::cout << "model range VaR99 [" << *vmin << ", " << *vmax << "]  ES99 [" << *emin << ", " << *emax << "]\n";
    std::cout << "historical block-bootstrap 90% CI VaR99 [" << last.var_lo << ", " << last.var_hi << "]  ES99 [" << last.es_lo << ", " << last.es_hi << "]\n";
    std::cout << "Results written to " << out_dir << "\n";
    return 0;
}
