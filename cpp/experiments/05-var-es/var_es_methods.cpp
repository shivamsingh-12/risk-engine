//Phase 7 experiment: Historical vs parametric vs Monte Carlo VaR/ES on a real equal-weight portfolio
//Runs the whole Phase 7 pipeline on the Phase 3 price history (AAPL, CAT, JNJ, JPM, PG, XOM, 2021-2023)
//and reports each method at several confidence levels, the Euler risk contribution of each stock,
//and a calibration check: how often did the portfolio actually lose more than each method's 1-day VaR?
//C++ only: the numerical methods are cross-validated against numpy/scipy in the test suite, and this
//experiment is about how the methods behave on real data rather than about a new mathematical fact
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "qre/core/rng.hpp"
#include "qre/risk/pipeline.hpp"
using qre::risk::VarEs;

//Minimal loader for this experiment only: header row of tickers, first column is the date
static bool load_prices(const std::string& path, qre::PriceMatrix& prices, std::vector<std::string>& tickers){
    std::ifstream in(path);
    if(!in.is_open()) return false;
    std::string line;
    std::getline(in, line);
    std::stringstream hs(line);
    std::string cell;
    std::getline(hs, cell, ',');   //Date
    while(std::getline(hs, cell, ',')) tickers.push_back(cell);
    std::vector<std::vector<double>> rows;
    while(std::getline(in, line)){
        if(line.empty()) continue;
        std::stringstream ls(line);
        std::getline(ls, cell, ',');
        std::vector<double> row;
        while(std::getline(ls, cell, ',')){
            row.push_back(cell.empty() ? std::numeric_limits<double>::quiet_NaN() : std::stod(cell));
        }
        rows.push_back(row);
    }
    prices.resize(static_cast<Eigen::Index>(rows.size()), static_cast<Eigen::Index>(tickers.size()));
    for(std::size_t r = 0; r < rows.size(); ++r)
        for(std::size_t c = 0; c < tickers.size(); ++c)
            prices(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = rows[r][c];
    return true;
}

int main(){
    const std::string data_path = "data/market_data/adjusted_close.csv";
    const std::string out_dir = "cpp/experiments/05-var-es/results/";
    qre::PriceMatrix prices;
    std::vector<std::string> tickers;
    if(!load_prices(data_path, prices, tickers)){
        std::cerr << "Failed to open " << data_path << "\nRun from repo root \n";
        return 1;
    }
    std::ofstream methods(out_dir + "var_es_methods.csv");
    std::ofstream contrib(out_dir + "risk_contributions.csv");
    std::ofstream stats(out_dir + "portfolio_stats.csv");
    if(!methods.is_open() || !contrib.is_open() || !stats.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    methods.precision(10); contrib.precision(10); stats.precision(10);

    const Eigen::Index n_assets = prices.cols();
    const qre::WeightVector w = qre::WeightVector::Constant(n_assets, 1.0 / static_cast<double>(n_assets));
    const qre::ReturnMatrix returns = qre::data::simple_return(prices);
    const std::vector<double> port = qre::risk::portfolio_returns(returns, w);

    //Portfolio return distribution summary -> why the methods will disagree
    const double mean = qre::estimators::empirical::sample_mean(port);
    const double sd = qre::estimators::empirical::sample_stddev(port);
    double m3 = 0, m4 = 0, worst = 0;
    for(double r : port){
        const double z = (r - mean) / sd;
        m3 += z * z * z; m4 += z * z * z * z;
        worst = std::min(worst, r);
    }
    const double n = static_cast<double>(port.size());
    stats << "n_returns,mean_daily,daily_vol,skew,excess_kurtosis,worst_day\n";
    stats << port.size() << "," << mean << "," << sd << "," << m3 / n << "," << m4 / n - 3.0 << "," << worst << "\n";
    std::cout << "n=" << port.size() << " mean=" << mean << " vol=" << sd << " skew=" << m3 / n
              << " excess_kurtosis=" << m4 / n - 3.0 << " worst=" << worst << "\n";

    methods << "confidence,method,var,es,es_over_var,observed_exceedance_rate,expected_exceedance_rate\n";
    qre::Rng rng(2028);
    for(double conf : {0.95, 0.975, 0.99}){
        qre::risk::RiskConfig cfg;
        cfg.confidence = conf;
        cfg.n_paths = 1000000;
        const auto rep = qre::risk::run_risk_pipeline(prices, w, rng, cfg);
        struct Row{ const char* name; VarEs v; };
        for(const Row& row : {Row{"historical", rep.historical}, Row{"parametric", rep.parametric}, Row{"monte_carlo", rep.monte_carlo}}){
            std::size_t exceed = 0;
            for(double r : port) if(-r > row.v.var) ++exceed;
            const double rate = static_cast<double>(exceed) / n;
            methods << conf << "," << row.name << "," << row.v.var << "," << row.v.es << "," << row.v.es / row.v.var
                    << "," << rate << "," << 1.0 - conf << "\n";
            std::cout << "c=" << conf << " " << row.name << " VaR=" << row.v.var << " ES=" << row.v.es
                      << " ES/VaR=" << row.v.es / row.v.var << " exceed=" << rate << " (nominal " << 1.0 - conf << ")\n";
        }
        if(conf == 0.99){
            contrib << "ticker,weight,risk_contribution,share_of_portfolio_vol,standalone_vol\n";
            const qre::CovarianceMatrix cov = qre::linalg::sample_cov(returns);
            for(Eigen::Index i = 0; i < n_assets; ++i){
                contrib << tickers[static_cast<std::size_t>(i)] << "," << w(i) << "," << rep.risk_contributions(i) << ","
                        << rep.risk_contributions(i) / rep.portfolio_vol << "," << std::sqrt(cov(i, i)) << "\n";
                std::cout << tickers[static_cast<std::size_t>(i)] << " share=" << rep.risk_contributions(i) / rep.portfolio_vol
                          << " standalone_vol=" << std::sqrt(cov(i, i)) << "\n";
            }
            std::cout << "portfolio_vol=" << rep.portfolio_vol << "\n";
        }
    }
    std::cout << "Results written to " << out_dir << "\n";
    return 0;
}
