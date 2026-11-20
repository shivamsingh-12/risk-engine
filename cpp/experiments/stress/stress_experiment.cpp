//Experiment 25 (Phase 20): stress testing. Run from the repo root:
//    build/experiments/stress_experiment
//Reads data/market_data/long_history_close.csv (tools/fetch_data.py) and writes cpp/experiments/stress/results/
//  Part 1  replay of ten crises on seven portfolios (buy-and-hold)
//  Part 2  did rolling 99% VaR models cover the crisis days? (daily rebalanced, the backtest convention)
//  Part 3  conditional Gaussian shocks: predict each asset's crisis return from SPY's, with calm vs crisis covariance
//  Part 4  how wrong is a beta proxy for an asset with no history in the crisis?
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "qre/data/csv_loader.hpp"
#include "qre/linalg/covariance.hpp"
#include "qre/risk/backtest.hpp"
#include "qre/risk/stress.hpp"

using namespace qre;
using namespace qre::risk;

namespace{
const std::string kOut = "cpp/experiments/stress/results/";

struct Portfolio{
    std::string key, name;
    std::vector<std::pair<std::string, double>> holdings;
};

const std::vector<Portfolio> kPortfolios = {
    {"sp500", "S&P 500 (SPY)", {{"SPY", 1.0}}},
    {"tech", "Nasdaq 100 (QQQ)", {{"QQQ", 1.0}}},
    {"sixty_forty", "60/40 (SPY/IEF)", {{"SPY", 0.6}, {"IEF", 0.4}}},
    {"six_stocks", "Six stocks, equal weight", {{"AAPL", 1.0 / 6}, {"CAT", 1.0 / 6}, {"JNJ", 1.0 / 6}, {"JPM", 1.0 / 6},
                                                {"PG", 1.0 / 6}, {"XOM", 1.0 / 6}}},
    {"multi_asset", "Multi-asset", {{"SPY", 0.30}, {"EFA", 0.15}, {"EEM", 0.05}, {"IEF", 0.25}, {"LQD", 0.10},
                                    {"GLD", 0.10}, {"VNQ", 0.05}}},
    {"credit", "Corporate credit (HYG/LQD)", {{"HYG", 0.5}, {"LQD", 0.5}}},
    {"financials", "Financials (XLF)", {{"XLF", 1.0}}},
};

const std::vector<std::string> kAssets = {"QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "LQD", "HYG", "GLD", "USO", "VNQ",
                                          "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
                                          "AAPL", "JPM", "XOM", "JNJ", "PG", "CAT"};

struct Method{ std::string name; BacktestMethod m; };
const std::vector<Method> kMethods = {
    {"historical", BacktestMethod::Historical}, {"parametric", BacktestMethod::Parametric},
    {"ewma", BacktestMethod::EwmaParametric}, {"fhs", BacktestMethod::FilteredHistorical},
    {"garch_t", BacktestMethod::GarchStudentT}, {"garch_evt", BacktestMethod::GarchEvt}};

data::PriceTable g_table;

Eigen::Index col(const std::string& t){
    const auto it = std::find(g_table.tickers.begin(), g_table.tickers.end(), t);
    if(it == g_table.tickers.end()) throw InvalidInputError("ticker " + t + " not in the price file");
    return static_cast<Eigen::Index>(it - g_table.tickers.begin());
}

Eigen::MatrixXd columns(const std::vector<std::string>& tickers, Eigen::Index r0, Eigen::Index r1){
    Eigen::MatrixXd m(r1 - r0 + 1, static_cast<Eigen::Index>(tickers.size()));
    for(std::size_t j = 0; j < tickers.size(); ++j) m.col(static_cast<Eigen::Index>(j)) = g_table.prices.block(r0, col(tickers[j]), r1 - r0 + 1, 1);
    return m;
}

//daily simple returns of rows [r0, r1] (r1 - r0 rows)
Eigen::MatrixXd returns(const std::vector<std::string>& tickers, Eigen::Index r0, Eigen::Index r1){
    const Eigen::MatrixXd p = columns(tickers, r0, r1);
    return (p.bottomRows(p.rows() - 1).array() / p.topRows(p.rows() - 1).array() - 1.0).matrix();
}

//every ticker has a price on every row in [r0, r1]
bool complete(const std::vector<std::string>& tickers, Eigen::Index r0, Eigen::Index r1){
    return r0 >= 0 && columns(tickers, r0, r1).allFinite();
}

std::pair<std::vector<std::string>, Eigen::VectorXd> split(const Portfolio& p){
    std::vector<std::string> t;
    Eigen::VectorXd w(static_cast<Eigen::Index>(p.holdings.size()));
    for(std::size_t i = 0; i < p.holdings.size(); ++i){ t.push_back(p.holdings[i].first); w(static_cast<Eigen::Index>(i)) = p.holdings[i].second; }
    return {t, w};
}

double rms(const std::vector<double>& e){
    double s = 0.0;
    for(double x : e) s += x * x;
    return std::sqrt(s / static_cast<double>(e.size()));
}
double mean_abs(const std::vector<double>& e){
    double s = 0.0;
    for(double x : e) s += std::abs(x);
    return s / static_cast<double>(e.size());
}

//Scenarios counted in totals: Lehman sits inside the GFC window, so it is reported but not added twice
bool counted(const Scenario& s){ return s.key != "lehman"; }
bool overlaps(const Scenario& a, const Scenario& b){ return a.start <= b.end && b.start <= a.end; }

void part1(){
    std::ofstream f(kOut + "replay.csv");
    f << "portfolio,scenario,start,end,days,total_return,max_drawdown,worst_day,worst_10d\n";
    std::printf("\nPart 1: buy-and-hold replay, total return (max drawdown)\n%-28s", "");
    for(const auto& s : kScenarios) std::printf(" %14.14s", std::string(s.key).c_str());
    std::printf("\n");
    for(const auto& p : kPortfolios){
        const auto [t, w] = split(p);
        std::printf("%-28s", p.name.c_str());
        for(const auto& s : kScenarios){
            const auto [i0, i1] = window_indices(g_table.dates, s.start, s.end);
            const Replay r = replay(returns(t, i0, i1), w);
            f << p.key << "," << s.key << "," << g_table.dates[static_cast<std::size_t>(i0)] << "," << g_table.dates[static_cast<std::size_t>(i1)]
              << "," << r.days << "," << r.total_return << "," << r.max_drawdown << "," << r.worst_day << "," << r.worst_10d << "\n";
            std::printf("  %+6.1f%% (%4.1f)", 100 * r.total_return, 100 * r.max_drawdown);
        }
        std::printf("\n");
    }
}

void part2(){
    std::ofstream f(kOut + "var_coverage.csv"), g(kOut + "var_at_start.csv");
    f << "portfolio,scenario,method,days,breaches,expected,worst_loss,var_on_worst_day,worst_over_var,"
         "mean_breach_loss,mean_breach_es\n";
    g << "portfolio,scenario,var1_hist,var1_garch_t,var10_hist_sqrt10,var10_garch_t_sqrt10,worst_10d_loss,scenario_loss\n";
    struct Tot{ int days = 0, breaches = 0; std::vector<double> ratio; double bl = 0.0, be = 0.0; int nb = 0; };
    std::map<std::string, Tot> tot;
    const double conf = 0.99;
    for(const auto& p : kPortfolios){
        const auto [t, w] = split(p);
        const Eigen::MatrixXd prices = columns(t, 0, g_table.prices.rows() - 1);
        for(const auto& s : kScenarios){
            const auto [i0, i1] = window_indices(g_table.dates, s.start, s.end);
            if(!complete(t, i0 - 250, i1)){
                std::printf("  (skipped %s in %s: less than 250 days of history before the scenario)\n", p.name.c_str(),
                            std::string(s.key).c_str());
                continue;
            }
            for(const auto& m : kMethods){
                int breaches = 0;
                double worst = -1e9, var_w = 0.0, bl = 0.0, be = 0.0;
                for(Eigen::Index day = i0 + 1; day <= i1; ++day){
                    const VarEs ve = forecast_var_es(prices, w, day, 250, conf, m.m);
                    const double loss = realized_loss(prices, w, day);
                    if(loss > ve.var){ ++breaches; bl += loss; be += ve.es; }
                    if(loss > worst){ worst = loss; var_w = ve.var; }
                }
                const int n = static_cast<int>(i1 - i0);
                f << p.key << "," << s.key << "," << m.name << "," << n << "," << breaches << "," << 0.01 * n << ","
                  << worst << "," << var_w << "," << worst / var_w << "," << (breaches ? bl / breaches : NAN) << ","
                  << (breaches ? be / breaches : NAN) << "\n";
                if(counted(s)){
                    auto& T = tot[m.name];
                    T.days += n; T.breaches += breaches; T.ratio.push_back(worst / var_w);
                    T.bl += bl; T.be += be; T.nb += breaches;
                }
            }
            //10-day risk measured at the start, against the worst 10 days and the whole scenario (buy-and-hold)
            const VarEs h = forecast_var_es(prices, w, i0 + 1, 250, conf, BacktestMethod::Historical);
            const VarEs gt = forecast_var_es(prices, w, i0 + 1, 250, conf, BacktestMethod::GarchStudentT);
            const Replay r = replay(returns(t, i0, i1), w);
            g << p.key << "," << s.key << "," << h.var << "," << gt.var << "," << std::sqrt(10.0) * h.var << ","
              << std::sqrt(10.0) * gt.var << "," << -r.worst_10d << "," << -r.total_return << "\n";
        }
    }
    std::ofstream sm(kOut + "var_coverage_summary.csv");
    sm << "method,days,breaches,expected,breach_rate,median_worst_over_var,max_worst_over_var,breach_loss_over_es\n";
    std::printf("\nPart 2: rolling 99%% VaR inside the crises (7 portfolios x 9 non-overlapping scenarios)\n");
    std::printf("%-12s %6s %9s %9s %8s %12s %10s %12s\n", "method", "days", "breaches", "expected", "rate", "median w/VaR", "max w/VaR", "loss/ES brch");
    for(const auto& m : kMethods){
        auto& T = tot[m.name];
        std::vector<double> r = T.ratio;
        std::sort(r.begin(), r.end());
        const double med = r.size() % 2 ? r[r.size() / 2] : 0.5 * (r[r.size() / 2 - 1] + r[r.size() / 2]);
        sm << m.name << "," << T.days << "," << T.breaches << "," << 0.01 * T.days << "," << double(T.breaches) / T.days << ","
           << med << "," << r.back() << "," << T.bl / T.be << "\n";
        std::printf("%-12s %6d %9d %9.1f %7.1f%% %12.2f %10.2f %12.2f\n", m.name.c_str(), T.days, T.breaches, 0.01 * T.days,
                    100.0 * T.breaches / T.days, med, r.back(), T.bl / T.be);
    }
}

void part3(){
    std::ofstream f(kOut + "conditional_predictions.csv"), sm(kOut + "conditional_summary.csv");
    f << "scenario,asset,realized,beta_calm,beta_crisis,pred_calm,pred_crisis\n";
    sm << "scenario,spy_return,rmse_calm,rmse_crisis,mae_calm,mae_crisis,sign_errors_calm,sign_errors_crisis,assets\n";
    std::printf("\nPart 3: predict each asset's crisis return from SPY's (Gaussian conditioning), error across the assets\n");
    std::printf("%-15s %8s %10s %12s %10s %12s\n", "scenario", "SPY", "RMSE calm", "RMSE crisis", "signs calm", "signs crisis");
    std::vector<double> ec_all, ex_all;
    for(const auto& s : kScenarios){
        const auto [i0, i1] = window_indices(g_table.dates, s.start, s.end);
        //assets with prices over the calm window, the scenario and every pooled crisis window
        std::vector<std::string> all{"SPY"};
        for(const auto& a : kAssets){
            bool ok = complete({a}, i0 - 500, i1);
            for(const auto& o : kScenarios){
                if(overlaps(o, s)) continue;
                const auto [j0, j1] = window_indices(g_table.dates, o.start, o.end);
                ok = ok && complete({a}, j0, j1);
            }
            if(ok) all.push_back(a);
        }
        //calm: the 500 trading days before the scenario; crisis: the daily returns of every other non-overlapping scenario
        const CovarianceMatrix calm = linalg::sample_cov(returns(all, i0 - 500, i0));
        std::vector<Eigen::MatrixXd> blocks;
        Eigen::Index rows = 0;
        for(const auto& o : kScenarios){
            if(overlaps(o, s)) continue;
            const auto [j0, j1] = window_indices(g_table.dates, o.start, o.end);
            blocks.push_back(returns(all, j0, j1));
            rows += blocks.back().rows();
        }
        Eigen::MatrixXd pooled(rows, static_cast<Eigen::Index>(all.size()));
        Eigen::Index at = 0;
        for(const auto& b : blocks){ pooled.middleRows(at, b.rows()) = b; at += b.rows(); }
        const CovarianceMatrix crisis = linalg::sample_cov(pooled);
        const Eigen::MatrixXd px = columns(all, i0, i1);
        const double spy = px(px.rows() - 1, 0) / px(0, 0) - 1.0;
        const Shock a = conditional_shock(calm, {0}, {spy});
        const Shock b = conditional_shock(crisis, {0}, {spy});
        std::vector<double> ec, ex;
        int sc = 0, sx = 0;
        for(std::size_t j = 1; j < all.size(); ++j){
            const auto J = static_cast<Eigen::Index>(j);
            const double real = px(px.rows() - 1, J) / px(0, J) - 1.0;
            f << s.key << "," << all[j] << "," << real << "," << calm(J, 0) / calm(0, 0) << "," << crisis(J, 0) / crisis(0, 0)
              << "," << a.mean(J) << "," << b.mean(J) << "\n";
            ec.push_back(a.mean(J) - real); ex.push_back(b.mean(J) - real);
            sc += (a.mean(J) > 0) != (real > 0);
            sx += (b.mean(J) > 0) != (real > 0);
        }
        if(counted(s)){ ec_all.insert(ec_all.end(), ec.begin(), ec.end()); ex_all.insert(ex_all.end(), ex.begin(), ex.end()); }
        sm << s.key << "," << spy << "," << rms(ec) << "," << rms(ex) << "," << mean_abs(ec) << "," << mean_abs(ex) << ","
           << sc << "," << sx << "," << ec.size() << "\n";
        std::printf("%-15s %+7.1f%% %9.1f%% %11.1f%% %7d/%zu %9d/%zu\n", std::string(s.key).c_str(), 100 * spy, 100 * rms(ec),
                    100 * rms(ex), sc, ec.size(), sx, ex.size());
    }
    sm << "all_counted,," << rms(ec_all) << "," << rms(ex_all) << "," << mean_abs(ec_all) << "," << mean_abs(ex_all) << ",,,"
       << ec_all.size() << "\n";
    std::printf("%-15s %8s %9.1f%% %11.1f%%   (9 non-overlapping scenarios)\n", "all", "", 100 * rms(ec_all), 100 * rms(ex_all));
}

void part4(){
    std::ofstream f(kOut + "proxy_accuracy.csv"), sm(kOut + "proxy_summary.csv"), pf(kOut + "proxy_portfolios.csv");
    f << "scenario,asset,beta_after,actual,proxy,error\n";
    sm << "scenario,mae,rmse,mean_error,assets\n";
    pf << "portfolio,scenario,actual,all_proxied,error\n";
    std::vector<std::string> all = kAssets;
    all.insert(all.begin(), "SPY");
    std::printf("\nPart 4: an asset replaced by beta x SPY, beta from the 3 years AFTER the scenario (as for a new listing)\n");
    std::printf("%-15s %10s %10s %12s\n", "scenario", "MAE", "RMSE", "mean error");
    std::map<std::string, std::vector<double>> pf_err;
    std::vector<double> all_err;
    for(const auto& s : kScenarios){
        const auto [i0, i1] = window_indices(g_table.dates, s.start, s.end);
        const Eigen::Index a0 = i1 + 1, a1 = std::min<Eigen::Index>(i1 + 756, g_table.prices.rows() - 1);
        const Eigen::MatrixXd after = returns(all, a0, a1);
        const Eigen::MatrixXd win = returns(all, i0, i1);
        Eigen::VectorXd betas(static_cast<Eigen::Index>(all.size()));
        std::vector<double> mk(after.col(0).data(), after.col(0).data() + after.rows());
        for(Eigen::Index j = 0; j < betas.size(); ++j){
            std::vector<double> x(after.col(j).data(), after.col(j).data() + after.rows());
            betas(j) = beta(x, mk);
        }
        //proxy path: every asset's daily return = beta * SPY's
        Eigen::MatrixXd prox(win.rows(), win.cols());
        for(Eigen::Index j = 0; j < win.cols(); ++j) prox.col(j) = betas(j) * win.col(0);
        std::vector<double> err;
        for(Eigen::Index j = 1; j < win.cols(); ++j){
            const double actual = (1.0 + win.col(j).array()).prod() - 1.0;
            const double proxy = (1.0 + prox.col(j).array()).prod() - 1.0;
            f << s.key << "," << all[static_cast<std::size_t>(j)] << "," << betas(j) << "," << actual << "," << proxy << "," << proxy - actual << "\n";
            err.push_back(proxy - actual);
        }
        if(counted(s)) all_err.insert(all_err.end(), err.begin(), err.end());
        const double me = std::accumulate(err.begin(), err.end(), 0.0) / static_cast<double>(err.size());
        sm << s.key << "," << mean_abs(err) << "," << rms(err) << "," << me << "," << err.size() << "\n";
        std::printf("%-15s %9.1f%% %9.1f%% %+11.1f%%\n", std::string(s.key).c_str(), 100 * mean_abs(err), 100 * rms(err), 100 * me);
        //portfolios with every holding proxied
        for(const auto& p : kPortfolios){
            const auto [t, w] = split(p);
            Eigen::MatrixXd act(win.rows(), static_cast<Eigen::Index>(t.size())), px(win.rows(), static_cast<Eigen::Index>(t.size()));
            for(std::size_t k = 0; k < t.size(); ++k){
                const auto J = static_cast<Eigen::Index>(std::find(all.begin(), all.end(), t[k]) - all.begin());
                act.col(static_cast<Eigen::Index>(k)) = win.col(J);
                px.col(static_cast<Eigen::Index>(k)) = prox.col(J);
            }
            const double ra = replay(act, w).total_return, rp = replay(px, w).total_return;
            pf << p.key << "," << s.key << "," << ra << "," << rp << "," << rp - ra << "\n";
            if(counted(s)) pf_err[p.key].push_back(rp - ra);
        }
    }
    std::printf("%-15s %9.1f%% %9.1f%%   (9 non-overlapping scenarios)\n", "all assets", 100 * mean_abs(all_err), 100 * rms(all_err));
    std::printf("Portfolios fully proxied, mean absolute error of the scenario return:\n");
    for(const auto& p : kPortfolios) std::printf("  %-28s %5.1f%%\n", p.name.c_str(), 100 * mean_abs(pf_err[p.key]));
}
}

int main(){
    try{
        g_table = data::load_price_csv("data/market_data/long_history_close.csv");
    } catch(const std::exception& e){
        std::cerr << "run from the repo root after tools/fetch_data.py: " << e.what() << "\n";
        return 1;
    }
    std::filesystem::create_directories(kOut);
    std::printf("Prices %s to %s, %zu tickers\n", g_table.dates.front().c_str(), g_table.dates.back().c_str(), g_table.tickers.size());
    const auto t0 = std::chrono::steady_clock::now();
    part1();
    part2();
    part3();
    part4();
    std::printf("\n%.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}
