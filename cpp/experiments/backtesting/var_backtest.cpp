//Backtesting experiment: are the Phase 7 VaR/ES models right out of sample, and how much can the data prove?
//Part 1: rolling 1-day backtest of historical, parametric and Monte Carlo VaR/ES on the real 6-stock history.
//        Each day's forecast uses only the previous 250 days (point-in-time), then meets that day's loss.
//Part 2: size and power of the Kupiec test itself, by simulation: with N forecast days, how often does it
//        wrongly reject a correct model (size), and how often does it catch a model that is too optimistic?
//C++ only: the test statistics are cross-validated against scipy in both test suites
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "qre/data/csv_loader.hpp"
#include "qre/data/quality.hpp"
#include "qre/risk/backtest.hpp"
using namespace qre::risk;

int main(){
    const std::string out_dir = "cpp/experiments/backtesting/results/";
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
    std::ofstream summary(out_dir + "backtest_summary.csv");
    std::ofstream daily(out_dir + "daily_forecasts_99.csv");
    std::ofstream power(out_dir + "kupiec_size_power.csv");
    if(!summary.is_open() || !daily.is_open() || !power.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    summary.precision(10); daily.precision(10); power.precision(6);

    const qre::PriceMatrix& prices = quality.prices;
    const Eigen::Index n = prices.cols();
    const qre::WeightVector w = qre::WeightVector::Constant(n, 1.0 / static_cast<double>(n));
    const Eigen::Index window = 250;

    //------------------------------------------------------------ Part 1
    summary << "confidence,method,n_obs,breaches,expected_breaches,breach_rate,kupiec_lr,kupiec_p,"
               "christoffersen_lr,christoffersen_p,cc_lr,cc_p,traffic_light_last250,z2\n";
    struct M{ const char* name; BacktestMethod m; };
    const M methods[] = {{"historical", BacktestMethod::Historical}, {"parametric", BacktestMethod::Parametric},
                         {"monte_carlo", BacktestMethod::MonteCarlo}};
    std::vector<BacktestSeries> at99;
    for(double conf : {0.99, 0.975}){
        for(const M& m : methods){
            const BacktestSeries s = rolling_backtest(prices, w, window, conf, m.m, 20000, 2029);
            const BacktestSummary r = summarize(s, conf);
            if(conf == 0.99) at99.push_back(s);
            summary << conf << "," << m.name << "," << r.n_obs << "," << r.n_breaches << "," << static_cast<double>(r.n_obs) * (1.0 - conf)
                    << "," << r.breach_rate << "," << r.kupiec.lr << "," << r.kupiec.p_value << "," << r.christoffersen.lr << ","
                    << r.christoffersen.p_value << "," << r.conditional.lr << "," << r.conditional.p_value << ","
                    << to_string(r.traffic_light) << "," << r.z2 << "\n";
            std::cout << "c=" << conf << " " << m.name << ": breaches " << r.n_breaches << "/" << r.n_obs << " (expected "
                      << static_cast<double>(r.n_obs) * (1.0 - conf) << ")  Kupiec p=" << r.kupiec.p_value
                      << "  Christoffersen p=" << r.christoffersen.p_value << "  CC p=" << r.conditional.p_value
                      << "  light=" << to_string(r.traffic_light) << "  Z2=" << r.z2 << "\n";
        }
    }
    daily << "date,realized_loss,hist_var99,param_var99,mc_var99,hist_es99,param_es99,mc_es99\n";
    for(std::size_t i = 0; i < at99[0].rows.size(); ++i){
        daily << table.dates[static_cast<std::size_t>(at99[0].rows[i])] << "," << at99[0].loss[i] << ","
              << at99[0].var[i] << "," << at99[1].var[i] << "," << at99[2].var[i] << ","
              << at99[0].es[i] << "," << at99[1].es[i] << "," << at99[2].es[i] << "\n";
    }
    //Breaches by calendar year at 99%: a backtest verdict depends heavily on which window you look at
    std::ofstream by_year(out_dir + "by_year_99.csv");
    by_year << "year,method,n_obs,breaches,expected_breaches,kupiec_p,traffic_light\n";
    for(const std::string year : {"2022", "2023"}){
        for(std::size_t m = 0; m < at99.size(); ++m){
            std::vector<bool> b;
            for(std::size_t i = 0; i < at99[m].rows.size(); ++i){
                if(table.dates[static_cast<std::size_t>(at99[m].rows[i])].substr(0, 4) == year) b.push_back(at99[m].loss[i] > at99[m].var[i]);
            }
            const auto x = static_cast<std::size_t>(std::count(b.begin(), b.end(), true));
            const TestResult k = kupiec_pof(b.size(), x, 0.01);
            by_year << year << "," << methods[m].name << "," << b.size() << "," << x << "," << static_cast<double>(b.size()) * 0.01 << ","
                    << k.p_value << "," << to_string(traffic_light(x, b.size(), 0.01)) << "\n";
            std::cout << year << " " << methods[m].name << ": " << x << "/" << b.size() << " Kupiec p=" << k.p_value
                      << " light=" << to_string(traffic_light(x, b.size(), 0.01)) << "\n";
        }
    }
    std::cout << "first forecast " << table.dates[static_cast<std::size_t>(at99[0].rows.front())]
              << ", last " << table.dates[static_cast<std::size_t>(at99[0].rows.back())] << "\n\n";

    //------------------------------------------------------------ Part 2
    //Breach counts of a model whose true breach probability is p_true, judged against nominal alpha
    power << "alpha,n_days,true_breach_prob,true_over_nominal,reject_rate_5pct\n";
    std::mt19937_64 eng(2030);
    const int reps = 20000;
    for(double alpha : {0.01, 0.025}){
        for(std::size_t days : {250u, 502u, 1000u, 2500u}){
            for(double mult : {1.0, 1.5, 2.0}){
                std::binomial_distribution<std::size_t> dist(days, alpha * mult);
                int rejected = 0;
                for(int i = 0; i < reps; ++i) if(kupiec_pof(days, dist(eng), alpha).p_value < 0.05) ++rejected;
                const double rate = static_cast<double>(rejected) / reps;
                power << alpha << "," << days << "," << alpha * mult << "," << mult << "," << rate << "\n";
                std::cout << "alpha=" << alpha << " days=" << days << " true=" << mult << "x  reject=" << rate << "\n";
            }
        }
    }
    std::cout << "Results written to " << out_dir << "\n";
    return 0;
}
