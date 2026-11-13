//Phase 18 experiment: optimal execution with Almgren-Chriss -- what risk aversion buys, and at what price
//Part 1: the paper's example (sell 1 million shares at $50 over 5 days, 30% annual volatility, eps = 1/16, eta = 2.5e-6,
//        gamma = 2.5e-7): schedules for lambda = 0 (TWAP), 1e-7, 1e-6, 1e-5, with expected cost, standard deviation, kappa
//Part 2: the efficient frontier (lambda from 1e-8 to 1e-4) and the closed form checked against the direct
//        tridiagonal solve
//Part 3: Monte Carlo of the shortfall (200,000 runs): TWAP vs optimal (lambda = 1e-6) vs selling everything at once,
//        mean / sd / 95% VaR and CVaR against the formulas
//Part 4: the execution horizon: lambda = 1e-6 with 1 to 20 days available (5 trading periods per day)
//Part 5: illustrative volatility sensitivity: the paper's impact parameters with each bundled stock's daily return
//        volatility (last 250 days) applied to a $50 price. Impact parameters are not estimated from these stocks
//        (the bundled data has no volumes or trades), so these are not cost estimates for them
//Run from the repo root
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "qre/data/csv_loader.hpp"
#include "qre/data/quality.hpp"
#include "qre/execution/almgren_chriss.hpp"
using namespace qre;
using namespace qre::execution;

namespace {
struct Tail{ double mean, sd, var95, cvar95; };
Tail tail(std::vector<double> L){
    const double n = static_cast<double>(L.size());
    double m = 0.0;
    for(double x : L) m += x;
    m /= n;
    double s = 0.0;
    for(double x : L) s += (x - m) * (x - m);
    std::sort(L.begin(), L.end());
    const auto idx = static_cast<std::size_t>(std::ceil(0.95 * n)) - 1;
    double c = 0.0;
    for(std::size_t i = idx; i < L.size(); ++i) c += L[i];
    return {m, std::sqrt(s / (n - 1.0)), L[idx], c / static_cast<double>(L.size() - idx)};
}
}

int main(){
    const std::string out_dir = "cpp/experiments/execution/results/";
    std::ofstream c1(out_dir + "schedules.csv"), c1s(out_dir + "schedule_summary.csv"), c2(out_dir + "frontier.csv"),
                  c3(out_dir + "monte_carlo.csv"), c4(out_dir + "horizon.csv"), c5(out_dir + "volatility_sensitivity.csv");
    if(!c1.is_open() || !c1s.is_open() || !c2.is_open() || !c3.is_open() || !c4.is_open() || !c5.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    for(auto* f : {&c1, &c1s, &c2, &c3, &c4, &c5}) f->precision(10);
    const AcParams p = paper_example();

    //------------------------------------------------------------ Part 1
    c1 << "lambda,day,holdings,sold_in_period\n";
    c1s << "lambda,kappa_per_day,time_scale_days,expected_cost,sd_cost,expected_cost_bp,sd_cost_bp,sold_day1_pct,max_abs_diff_vs_numerical\n";
    const double notional = p.X * p.S0;
    for(double lam : {0.0, 1e-7, 1e-6, 1e-5}){
        const auto x = optimal_holdings(p, lam), xn = solve_numerically(p, lam);
        double diff = 0.0;
        for(std::size_t j = 0; j < x.size(); ++j) diff = std::max(diff, std::abs(x[j] - xn[j]));
        for(std::size_t j = 0; j < x.size(); ++j) c1 << lam << "," << j << "," << x[j] << "," << (j ? x[j - 1] - x[j] : 0.0) << "\n";
        const double k = kappa(p, lam), E = expected_cost(p, x), sd = std::sqrt(cost_variance(p, x));
        c1s << lam << "," << k << "," << (k > 0 ? 1.0 / k : 0.0) << "," << E << "," << sd << "," << 1e4 * E / notional << "," << 1e4 * sd / notional
            << "," << 100.0 * (x[0] - x[1]) / p.X << "," << diff << "\n";
    }

    //------------------------------------------------------------ Part 2
    c2 << "lambda,expected_cost,sd_cost,max_abs_diff_vs_numerical\n";
    for(int i = 0; i <= 24; ++i){
        const double lam = std::pow(10.0, -8.0 + 4.0 * i / 24.0);
        const auto x = optimal_holdings(p, lam), xn = solve_numerically(p, lam);
        double diff = 0.0;
        for(std::size_t j = 0; j < x.size(); ++j) diff = std::max(diff, std::abs(x[j] - xn[j]));
        c2 << lam << "," << expected_cost(p, x) << "," << std::sqrt(cost_variance(p, x)) << "," << diff << "\n";
    }

    //------------------------------------------------------------ Part 3
    c3 << "strategy,formula_mean,formula_sd,mc_mean,mc_sd,mc_var95,mc_cvar95,mean_plus_lambda_var\n";
    std::vector<double> immediate(static_cast<std::size_t>(p.N) + 1, 0.0);
    immediate[0] = p.X;
    struct Strat{ const char* name; std::vector<double> x; };
    for(const Strat& s : {Strat{"twap", twap_holdings(p)}, Strat{"optimal_lambda_1e-6", optimal_holdings(p, 1e-6)}, Strat{"sell_all_first_period", immediate}}){
        const Tail t = tail(simulate_shortfall(p, s.x, 200000, 1801));
        const double E = expected_cost(p, s.x), V = cost_variance(p, s.x);
        c3 << s.name << "," << E << "," << std::sqrt(V) << "," << t.mean << "," << t.sd << "," << t.var95 << "," << t.cvar95 << "," << E + 1e-6 * V << "\n";
    }
    std::cout << "Parts 1-3 done\n";

    //------------------------------------------------------------ Part 4
    c4 << "days,periods,expected_cost,sd_cost,mean_plus_lambda_var,sold_day1_pct,remaining_after_half_pct\n";
    for(int days : {1, 2, 5, 10, 20}){
        AcParams q = p;
        q.T = days;
        q.N = 5 * days;
        const auto x = optimal_holdings(q, 1e-6);
        const double E = expected_cost(q, x), V = cost_variance(q, x);
        c4 << days << "," << q.N << "," << E << "," << std::sqrt(V) << "," << E + 1e-6 * V << "," << 100.0 * (q.X - x[5]) / q.X << ","
           << 100.0 * x[static_cast<std::size_t>(q.N / 2)] / q.X << "\n";
    }

    //------------------------------------------------------------ Part 5
    data::PriceTable table;
    try{ table = data::load_price_csv("data/market_data/adjusted_close.csv"); }
    catch(const std::exception& e){
        std::cerr << e.what() << "\nRun from repo root \n";
        return 1;
    }
    const auto hardened = data::harden(table.prices);
    if(!hardened.report.clean()){ std::cerr << "price data failed the quality report\n"; return 1; }
    c5 << "ticker,daily_return_vol,sigma_dollars_per_day,kappa_per_day,sold_day1_pct,expected_cost_bp,sd_cost_bp,twap_sd_cost_bp\n";
    const Eigen::Index T = hardened.prices.rows();
    for(Eigen::Index j = 0; j < hardened.prices.cols(); ++j){
        double m = 0.0, s = 0.0;
        const int n = 250;
        for(Eigen::Index t = T - n; t < T; ++t) m += std::log(hardened.prices(t, j) / hardened.prices(t - 1, j));
        m /= n;
        for(Eigen::Index t = T - n; t < T; ++t){ const double d = std::log(hardened.prices(t, j) / hardened.prices(t - 1, j)) - m; s += d * d; }
        const double vol = std::sqrt(s / (n - 1));
        AcParams q = p;
        q.sigma = vol * q.S0;
        const auto x = optimal_holdings(q, 1e-6), tw = twap_holdings(q);
        c5 << table.tickers[static_cast<std::size_t>(j)] << "," << vol << "," << q.sigma << "," << kappa(q, 1e-6) << "," << 100.0 * (q.X - x[1]) / q.X << ","
           << 1e4 * expected_cost(q, x) / notional << "," << 1e4 * std::sqrt(cost_variance(q, x)) / notional << ","
           << 1e4 * std::sqrt(cost_variance(q, tw)) / notional << "\n";
    }
    std::cout << "Results in " << out_dir << "\n";
    return 0;
}
