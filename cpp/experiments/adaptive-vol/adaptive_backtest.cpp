//Adaptive volatility experiment: does EWMA / filtered historical simulation fix what the backtest found?
//The equal-weight 250-day models breached in clusters during 2022 and never in 2023. This reruns the same
//rolling out-of-sample backtest with the two adaptive methods next to the three originals, then sweeps the
//EWMA decay lambda. Also reports the cost of adapting: average VaR (capital) and how much VaR moves day to day
//C++ only: EWMA/FHS are cross-validated against the Python reference in both test suites
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "qre/data/csv_loader.hpp"
#include "qre/data/quality.hpp"
#include "qre/risk/backtest.hpp"
using namespace qre::risk;

namespace {
struct Extra{ double mean_var; double var_change_sd; };

//Average forecast VaR and the standard deviation of its daily change (how jumpy the model is)
Extra extras(const BacktestSeries& s){
    double sum = 0.0;
    for(double v : s.var) sum += v;
    std::vector<double> d;
    for(std::size_t i = 1; i < s.var.size(); ++i) d.push_back(s.var[i] - s.var[i - 1]);
    return Extra{sum / static_cast<double>(s.var.size()), qre::estimators::empirical::sample_stddev(d)};
}

std::size_t breaches_in(const BacktestSeries& s, const std::vector<std::string>& dates, const std::string& year){
    std::size_t n = 0;
    for(std::size_t i = 0; i < s.rows.size(); ++i)
        if(dates[static_cast<std::size_t>(s.rows[i])].substr(0, 4) == year && s.loss[i] > s.var[i]) ++n;
    return n;
}
}

int main(){
    const std::string out_dir = "cpp/experiments/adaptive-vol/results/";
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
    std::ofstream sweep(out_dir + "lambda_sweep.csv");
    std::ofstream daily(out_dir + "daily_var99.csv");
    if(!summary.is_open() || !sweep.is_open() || !daily.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    summary.precision(10); sweep.precision(10); daily.precision(10);

    const qre::PriceMatrix& prices = quality.prices;
    const qre::WeightVector w = qre::WeightVector::Constant(prices.cols(), 1.0 / static_cast<double>(prices.cols()));
    const Eigen::Index window = 250;
    struct M{ const char* name; BacktestMethod m; };
    const M methods[] = {{"historical", BacktestMethod::Historical}, {"parametric", BacktestMethod::Parametric},
                         {"monte_carlo", BacktestMethod::MonteCarlo}, {"ewma", BacktestMethod::EwmaParametric},
                         {"fhs", BacktestMethod::FilteredHistorical}};

    //------------------------------------------------------------ Part 1: five methods, lambda = 0.94
    summary << "confidence,method,n_obs,breaches,expected,breaches_2022,breaches_2023,kupiec_p,christoffersen_p,cc_p,z2,"
               "mean_var,var_daily_change_sd\n";
    std::vector<BacktestSeries> at99;
    for(double conf : {0.99, 0.975}){
        for(const M& m : methods){
            const BacktestSeries s = rolling_backtest(prices, w, window, conf, m.m, 20000, 2029, 252, 0.94);
            const BacktestSummary r = summarize(s, conf);
            const Extra e = extras(s);
            if(conf == 0.99) at99.push_back(s);
            summary << conf << "," << m.name << "," << r.n_obs << "," << r.n_breaches << "," << static_cast<double>(r.n_obs) * (1.0 - conf) << ","
                    << breaches_in(s, table.dates, "2022") << "," << breaches_in(s, table.dates, "2023") << ","
                    << r.kupiec.p_value << "," << r.christoffersen.p_value << "," << r.conditional.p_value << "," << r.z2 << ","
                    << e.mean_var << "," << e.var_change_sd << "\n";
            std::cout << "c=" << conf << " " << m.name << ": " << r.n_breaches << " (2022 " << breaches_in(s, table.dates, "2022")
                      << ", 2023 " << breaches_in(s, table.dates, "2023") << ")  Kupiec p=" << r.kupiec.p_value
                      << "  Christ p=" << r.christoffersen.p_value << "  CC p=" << r.conditional.p_value << "  Z2=" << r.z2
                      << "  meanVaR=" << e.mean_var << "  dVaR sd=" << e.var_change_sd << "\n";
        }
    }
    daily << "date,realized_loss";
    for(const M& m : methods) daily << "," << m.name << "_var99";
    daily << "\n";
    for(std::size_t i = 0; i < at99[0].rows.size(); ++i){
        daily << table.dates[static_cast<std::size_t>(at99[0].rows[i])] << "," << at99[0].loss[i];
        for(const auto& s : at99) daily << "," << s.var[i];
        daily << "\n";
    }

    //------------------------------------------------------------ Part 2: lambda sweep
    sweep << "confidence,method,lambda,breaches,breaches_2022,breaches_2023,kupiec_p,christoffersen_p,cc_p,z2,mean_var,var_daily_change_sd\n";
    std::cout << "\n";
    for(double conf : {0.99, 0.975}){
        for(const M& m : {methods[3], methods[4]}){
            for(double lam : {0.90, 0.94, 0.97, 0.99}){
                const BacktestSeries s = rolling_backtest(prices, w, window, conf, m.m, 20000, 2029, 252, lam);
                const BacktestSummary r = summarize(s, conf);
                const Extra e = extras(s);
                sweep << conf << "," << m.name << "," << lam << "," << r.n_breaches << "," << breaches_in(s, table.dates, "2022") << ","
                      << breaches_in(s, table.dates, "2023") << "," << r.kupiec.p_value << "," << r.christoffersen.p_value << ","
                      << r.conditional.p_value << "," << r.z2 << "," << e.mean_var << "," << e.var_change_sd << "\n";
                std::cout << "c=" << conf << " " << m.name << " lam=" << lam << ": " << r.n_breaches << " (2022 "
                          << breaches_in(s, table.dates, "2022") << ", 2023 " << breaches_in(s, table.dates, "2023") << ")  Kupiec p="
                          << r.kupiec.p_value << "  Christ p=" << r.christoffersen.p_value << "  Z2=" << r.z2
                          << "  meanVaR=" << e.mean_var << "  dVaR sd=" << e.var_change_sd << "\n";
            }
        }
    }
    std::cout << "Results written to " << out_dir << "\n";
    return 0;
}
