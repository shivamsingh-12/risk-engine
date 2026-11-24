//Experiment 26 (Phase 21): portfolio construction, component risk and 10-day VaR. Run from the repo root:
//    build/experiments/portfolio_experiment
//Reads data/market_data/long_history_close.csv and writes cpp/experiments/portfolio/results/
//  Part 1  eight construction methods out of sample on three universes, monthly rebalancing, 2007 to 2026
//  Part 2  does today's component ES predict each holding's share of next year's tail losses?
//  Part 3  10-day VaR: five methods backtested on non-overlapping 10-day periods
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "qre/data/csv_loader.hpp"
#include "qre/linalg/covariance.hpp"
#include "qre/linalg/shrinkage.hpp"
#include "qre/portfolio/construction.hpp"
#include "qre/risk/backtest.hpp"
#include "qre/risk/component.hpp"
#include "qre/risk/ewma.hpp"
#include "qre/risk/horizon.hpp"

using namespace qre;

namespace{
const std::string kOut = "cpp/experiments/portfolio/results/";
constexpr Eigen::Index kEstimation = 500;      //days of history behind every estimate
constexpr Eigen::Index kRebalance = 21;        //trading days between rebalances
constexpr double kCost = 0.0010;               //10 bp per unit of one-way turnover, for the net figures
constexpr double kGamma = 5.0;                 //mean-variance risk aversion
constexpr double kShort = -0.3;                //lower bound for the methods that may short

std::vector<std::string> g_dates;
Eigen::MatrixXd g_prices;
std::vector<std::string> g_tickers;

Eigen::Index col(const std::string& t){
    const auto it = std::find(g_tickers.begin(), g_tickers.end(), t);
    if(it == g_tickers.end()) throw InvalidInputError("ticker " + t + " not in the price file");
    return static_cast<Eigen::Index>(it - g_tickers.begin());
}

//daily simple returns for the tickers, all rows (row t = return from t-1 to t; row 0 is NaN)
Eigen::MatrixXd returns_of(const std::vector<std::string>& t){
    Eigen::MatrixXd r(g_prices.rows(), static_cast<Eigen::Index>(t.size()));
    r.row(0).setConstant(NAN);
    for(std::size_t j = 0; j < t.size(); ++j){
        const Eigen::Index c = col(t[j]);
        for(Eigen::Index i = 1; i < g_prices.rows(); ++i) r(i, static_cast<Eigen::Index>(j)) = g_prices(i, c) / g_prices(i - 1, c) - 1.0;
    }
    return r;
}

//first row from which every ticker has prices on all later rows
Eigen::Index first_full_row(const Eigen::MatrixXd& r){
    Eigen::Index first = 1;
    for(Eigen::Index i = 1; i < r.rows(); ++i) if(!r.row(i).allFinite()) first = i + 1;
    return first;
}

double ann_vol(const std::vector<double>& x){
    const double m = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    double s = 0.0;
    for(double v : x) s += (v - m) * (v - m);
    return std::sqrt(s / static_cast<double>(x.size() - 1) * 252.0);
}

// ------------------------------------------------------------------------------------------------ Part 1
struct Universe{ std::string key, name; std::vector<std::string> tickers; };
const std::vector<Universe> kUniverses = {
    {"sectors", "9 US sector ETFs", {"XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"}},
    {"multi_asset", "9 asset-class ETFs", {"SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "VNQ"}},
    {"stocks", "14 large US stocks", {"AAPL", "AMZN", "BAC", "CAT", "GOOGL", "GS", "JNJ", "JPM", "KO", "MSFT", "NVDA", "PG", "WMT", "XOM"}},
};

struct Choice{ Eigen::VectorXd w; Eigen::MatrixXd cov; };       //weights and the covariance they were judged by
using Method = std::function<Choice(const Eigen::MatrixXd&)>;   //from a window of returns

std::vector<std::pair<std::string, Method>> methods(){
    using namespace portfolio;
    auto sample = [](const Eigen::MatrixXd& r){ return linalg::sample_cov(r); };
    auto lw = [](const Eigen::MatrixXd& r){ return linalg::ledoit_wolf(r).cov; };
    auto mean = [](const Eigen::MatrixXd& r){ return r.colwise().mean().transpose().eval(); };
    return {
        {"equal", [=](const Eigen::MatrixXd& r){ return Choice{equal_weight(r.cols()), sample(r)}; }},
        {"inverse_vol", [=](const Eigen::MatrixXd& r){ auto s = sample(r); return Choice{inverse_vol(s), s}; }},
        {"min_var_sample", [=](const Eigen::MatrixXd& r){ auto s = sample(r); return Choice{min_variance(s, make_bounds(r.cols())).weights, s}; }},
        {"min_var_lw", [=](const Eigen::MatrixXd& r){ auto s = lw(r); return Choice{min_variance(s, make_bounds(r.cols())).weights, s}; }},
        {"min_var_lw_short", [=](const Eigen::MatrixXd& r){ auto s = lw(r); return Choice{min_variance(s, make_bounds(r.cols(), kShort, 1.0)).weights, s}; }},
        {"risk_parity", [=](const Eigen::MatrixXd& r){ auto s = lw(r); return Choice{risk_parity(s).weights, s}; }},
        {"mean_var", [=](const Eigen::MatrixXd& r){ auto s = lw(r); return Choice{mean_variance(mean(r), s, kGamma, make_bounds(r.cols())).weights, s}; }},
        {"mean_var_short", [=](const Eigen::MatrixXd& r){ auto s = lw(r); return Choice{mean_variance(mean(r), s, kGamma, make_bounds(r.cols(), kShort, 1.0)).weights, s}; }},
    };
}

void part1(){
    std::ofstream f(kOut + "construction.csv"), wf(kOut + "construction_weights.csv"), yf(kOut + "construction_by_year.csv");
    f << "universe,method,start,end,days,ann_return,ann_vol,return_over_vol,max_drawdown,turnover_per_rebalance,"
         "ann_return_net,avg_holdings,avg_gross,avg_short,vol_ratio_realized_over_predicted\n";
    wf << "universe,method,ticker,avg_weight,min_weight,max_weight\n";
    yf << "universe,method,year,return\n";
    for(const auto& u : kUniverses){
        const Eigen::MatrixXd r = returns_of(u.tickers);
        const Eigen::Index n = r.cols(), start = first_full_row(r) + kEstimation, end = r.rows();
        std::printf("\nPart 1, %s: %s to %s\n%-17s %8s %7s %6s %8s %9s %8s %6s %6s %8s\n", u.name.c_str(),
                    g_dates[static_cast<std::size_t>(start)].c_str(), g_dates.back().c_str(), "method", "return", "vol", "r/vol",
                    "max DD", "turnover", "net 10bp", "hold", "gross", "vol x");
        for(const auto& [name, m] : methods()){
            std::vector<double> daily, net_daily;
            Eigen::VectorXd w = Eigen::VectorXd::Zero(n), target(n);
            double turnover = 0.0, holdings = 0.0, gross = 0.0, shrt = 0.0, pred_var = 0.0, real_var = 0.0;
            int rebalances = 0;
            Eigen::VectorXd wsum = Eigen::VectorXd::Zero(n), wmin = Eigen::VectorXd::Constant(n, 1e9), wmax = Eigen::VectorXd::Constant(n, -1e9);
            std::map<std::string, double> year_growth;
            for(Eigen::Index t = start; t < end; t += kRebalance){
                const Choice c = m(r.middleRows(t - kEstimation, kEstimation));
                target = c.w;
                const double to = 0.5 * (target - w).cwiseAbs().sum();
                if(rebalances > 0) turnover += to;
                const double cost = kCost * (rebalances > 0 ? to : 1.0);     //the first purchase is a full turnover
                holdings += static_cast<double>((target.array().abs() > 0.01).count());
                gross += target.cwiseAbs().sum();
                shrt += -target.cwiseMin(0.0).sum();
                wsum += target;
                wmin = wmin.cwiseMin(target);
                wmax = wmax.cwiseMax(target);
                ++rebalances;
                //hold until the next rebalance: weights drift with prices
                w = target;
                const Eigen::Index stop = std::min(end, t + kRebalance);
                std::vector<double> month;
                for(Eigen::Index d = t; d < stop; ++d){
                    const double rp = w.dot(r.row(d));
                    daily.push_back(rp);
                    net_daily.push_back(d == t ? (1.0 - cost) * (1.0 + rp) - 1.0 : rp);
                    month.push_back(rp);
                    w = (w.array() * (1.0 + r.row(d).transpose().array())).matrix() / (1.0 + rp);
                    year_growth[g_dates[static_cast<std::size_t>(d)].substr(0, 4)] += std::log1p(rp);
                }
                if(month.size() >= 10){
                    pred_var += target.dot(c.cov * target) * static_cast<double>(month.size());
                    for(double x : month) real_var += x * x;
                }
            }
            double g = 1.0, peak = 1.0, mdd = 0.0;
            for(double x : daily){ g *= 1.0 + x; peak = std::max(peak, g); mdd = std::max(mdd, 1.0 - g / peak); }
            const double years = static_cast<double>(daily.size()) / 252.0;
            const double ret = std::pow(g, 1.0 / years) - 1.0;
            double gn = 1.0;
            for(double x : net_daily) gn *= 1.0 + x;
            const double net = std::pow(gn, 1.0 / years) - 1.0;
            const double vol = ann_vol(daily);
            const double tpr = turnover / std::max(1, rebalances - 1);
            const double ratio = std::sqrt(real_var / pred_var);
            f << u.key << "," << name << "," << g_dates[static_cast<std::size_t>(start)] << "," << g_dates.back() << "," << daily.size() << ","
              << ret << "," << vol << "," << ret / vol << "," << mdd << "," << tpr << "," << net << "," << holdings / rebalances << ","
              << gross / rebalances << "," << shrt / rebalances << "," << ratio << "\n";
            for(Eigen::Index j = 0; j < n; ++j)
                wf << u.key << "," << name << "," << u.tickers[static_cast<std::size_t>(j)] << "," << wsum(j) / rebalances << "," << wmin(j) << "," << wmax(j) << "\n";
            for(const auto& [y, lg] : year_growth) yf << u.key << "," << name << "," << y << "," << std::expm1(lg) << "\n";
            std::printf("%-17s %7.2f%% %6.2f%% %6.2f %7.1f%% %8.1f%% %7.2f%% %6.1f %6.2f %8.2f\n", name.c_str(), 100 * ret, 100 * vol,
                        ret / vol, 100 * mdd, 100 * tpr, 100 * net, holdings / rebalances, gross / rebalances, ratio);
        }
    }
}

// ------------------------------------------------------------------------------------------------ Part 2
struct Fixed{ std::string key, name; std::vector<std::pair<std::string, double>> holdings; };
const std::vector<Fixed> kFixed = {
    {"sixty_forty", "60/40 (SPY/IEF)", {{"SPY", 0.6}, {"IEF", 0.4}}},
    {"multi_asset", "Multi-asset", {{"SPY", 0.30}, {"EFA", 0.15}, {"EEM", 0.05}, {"IEF", 0.25}, {"LQD", 0.10}, {"GLD", 0.10}, {"VNQ", 0.05}}},
    {"six_stocks", "Six stocks, equal weight", {{"AAPL", 1.0 / 6}, {"CAT", 1.0 / 6}, {"JNJ", 1.0 / 6}, {"JPM", 1.0 / 6}, {"PG", 1.0 / 6}, {"XOM", 1.0 / 6}}},
    {"sp500", "S&P 500 (SPY)", {{"SPY", 1.0}}},
    {"tech", "Nasdaq 100 (QQQ)", {{"QQQ", 1.0}}},
    {"financials", "Financials (XLF)", {{"XLF", 1.0}}},
    {"credit", "Corporate credit (HYG/LQD)", {{"HYG", 0.5}, {"LQD", 0.5}}},
};

std::pair<std::vector<std::string>, Eigen::VectorXd> split(const Fixed& p){
    std::vector<std::string> t;
    Eigen::VectorXd w(static_cast<Eigen::Index>(p.holdings.size()));
    for(std::size_t i = 0; i < p.holdings.size(); ++i){ t.push_back(p.holdings[i].first); w(static_cast<Eigen::Index>(i)) = p.holdings[i].second; }
    return {t, w};
}

void part2(){
    std::ofstream f(kOut + "component_prediction.csv"), sm(kOut + "component_summary.csv"), now(kOut + "component_latest.csv");
    f << "portfolio,date,ticker,weight,pred_parametric,pred_historical,realized\n";
    sm << "portfolio,dates,mae_weight,mae_parametric,mae_historical,mean_weight_share,mean_pred_parametric,mean_realized\n";
    now << "portfolio,ticker,weight,share_vol,component_var99_param,component_es975_param,component_var99_hist,component_es975_hist,share_es975_hist\n";
    std::printf("\nPart 2: predicting each holding's share of the next 250 days' 97.5%% ES (mean absolute error, share points)\n");
    std::printf("%-26s %6s %10s %12s %12s\n", "portfolio", "dates", "weights", "parametric", "historical");
    for(const auto& p : kFixed){
        if(p.holdings.size() < 2) continue;
        const auto [t, w] = split(p);
        const Eigen::MatrixXd r = returns_of(t);
        const Eigen::Index start = first_full_row(r) + kEstimation, end = r.rows();
        double ew = 0.0, ep = 0.0, eh = 0.0;
        Eigen::VectorXd mw = Eigen::VectorXd::Zero(w.size()), mp = mw, mr = mw;
        int n = 0;
        for(Eigen::Index d = start; d + 250 <= end; d += kRebalance){
            const Eigen::MatrixXd past = r.middleRows(d - 250, 250), next = r.middleRows(d, 250);
            const auto pc = risk::parametric_components(w, linalg::sample_cov(past), 0.975);
            const auto hc = risk::historical_components(past, w, 0.975);
            const auto rc = risk::historical_components(next, w, 0.975);
            const Eigen::VectorXd sp = pc.component_es / pc.es, sh = hc.component_es / hc.es, sr = rc.component_es / rc.es;
            for(Eigen::Index j = 0; j < w.size(); ++j)
                f << p.key << "," << g_dates[static_cast<std::size_t>(d)] << "," << t[static_cast<std::size_t>(j)] << "," << w(j) << "," << sp(j) << "," << sh(j) << "," << sr(j) << "\n";
            ew += (w - sr).cwiseAbs().mean();
            ep += (sp - sr).cwiseAbs().mean();
            eh += (sh - sr).cwiseAbs().mean();
            mw += w; mp += sp; mr += sr;
            ++n;
        }
        sm << p.key << "," << n << "," << ew / n << "," << ep / n << "," << eh / n << ",";
        for(Eigen::Index j = 0; j < w.size(); ++j) sm << (j ? ";" : "") << t[static_cast<std::size_t>(j)] << "=" << mw(j) / n;
        sm << ",";
        for(Eigen::Index j = 0; j < w.size(); ++j) sm << (j ? ";" : "") << t[static_cast<std::size_t>(j)] << "=" << mp(j) / n;
        sm << ",";
        for(Eigen::Index j = 0; j < w.size(); ++j) sm << (j ? ";" : "") << t[static_cast<std::size_t>(j)] << "=" << mr(j) / n;
        sm << "\n";
        std::printf("%-26s %6d %9.1f %12.1f %12.1f\n", p.name.c_str(), n, 100 * ew / n, 100 * ep / n, 100 * eh / n);
        std::printf("    mean shares   weight / predicted (param) / realized:");
        for(Eigen::Index j = 0; j < w.size(); ++j) std::printf("  %s %.0f/%.0f/%.0f", t[static_cast<std::size_t>(j)].c_str(), 100 * mw(j) / n, 100 * mp(j) / n, 100 * mr(j) / n);
        std::printf("\n");
        //latest snapshot
        const Eigen::MatrixXd last = r.bottomRows(250);
        const Eigen::MatrixXd s = linalg::sample_cov(last);
        const auto p99 = risk::parametric_components(w, s, 0.99), p975 = risk::parametric_components(w, s, 0.975);
        const auto h99 = risk::historical_components(last, w, 0.99), h975 = risk::historical_components(last, w, 0.975);
        const Eigen::VectorXd sv = portfolio::risk_contribution_shares(w, s);
        for(Eigen::Index j = 0; j < w.size(); ++j)
            now << p.key << "," << t[static_cast<std::size_t>(j)] << "," << w(j) << "," << sv(j) << "," << p99.component_var(j) << "," << p975.component_es(j) << ","
                << h99.component_var(j) << "," << h975.component_es(j) << "," << h975.component_es(j) / h975.es << "\n";
    }
}

// ------------------------------------------------------------------------------------------------ Part 3
void part3(){
    std::ofstream f(kOut + "horizon_forecasts.csv"), sm(kOut + "horizon_summary.csv");
    f << "portfolio,date,confidence,loss,sqrt10_hist,sqrt10_fhs_ewma,overlapping,garch_normal,fhs_garch\n";
    sm << "portfolio,confidence,method,periods,breaches,expected,breach_rate,kupiec_p,mean_var,loss_over_es_on_breach\n";
    const std::vector<std::string> names = {"sqrt10_hist", "sqrt10_fhs_ewma", "overlapping", "garch_normal", "fhs_garch"};
    const std::vector<double> confs = {0.99, 0.975};
    struct Acc{ int n = 0, b = 0; double var = 0.0, bl = 0.0, be = 0.0; };
    std::map<std::pair<double, std::string>, Acc> pooled;
    std::printf("\nPart 3: 10-day VaR on non-overlapping 10-day periods (breach rate; 1%% and 2.5%% expected)\n");
    std::printf("%-26s %6s", "portfolio", "n");
    for(const auto& m : names) std::printf(" %16s", m.c_str());
    std::printf("\n");
    for(const auto& p : kFixed){
        const auto [t, w] = split(p);
        const Eigen::MatrixXd r = returns_of(t);
        const Eigen::Index start = first_full_row(r) + kEstimation, end = r.rows();
        std::map<std::pair<double, std::string>, Acc> acc;
        for(Eigen::Index d = start; d + 10 <= end; d += 10){
            const Eigen::VectorXd pr500 = r.middleRows(d - kEstimation, kEstimation) * w;
            const std::vector<double> win500(pr500.data(), pr500.data() + pr500.size());
            const std::vector<double> win250(win500.end() - 250, win500.end());
            double g = 1.0;
            for(Eigen::Index k = d; k < d + 10; ++k) g *= 1.0 + w.dot(r.row(k));
            const double loss = -(g - 1.0);
            const risk::GarchFit fit = risk::fit_garch(win500);
            for(double c : confs){
                const std::vector<risk::VarEs> v = {
                    risk::sqrt_time_var_es(win250, c, 10),
                    [&]{ auto x = risk::fhs_var_es(win250, c); return risk::VarEs{x.var * std::sqrt(10.0), x.es * std::sqrt(10.0)}; }(),
                    risk::overlapping_var_es(win500, c, 10),
                    risk::garch_normal_var_es(win500, c, 10, &fit),
                    risk::fhs_garch_var_es(win500, c, 10, 5000, static_cast<uint64_t>(d), &fit),
                };
                f << p.key << "," << g_dates[static_cast<std::size_t>(d)] << "," << c << "," << loss;
                for(std::size_t k = 0; k < v.size(); ++k){
                    f << "," << v[k].var;
                    for(auto* a : {&acc[{c, names[k]}], &pooled[{c, names[k]}]}){
                        ++a->n; a->var += v[k].var;
                        if(loss > v[k].var){ ++a->b; a->bl += loss; a->be += v[k].es; }
                    }
                }
                f << "\n";
            }
        }
        for(double c : confs){
            std::printf("%-26s %6d", (p.name + (c == 0.99 ? " 99%" : " 97.5%")).c_str(), acc[{c, names[0]}].n);
            for(const auto& m : names){
                const Acc& a = acc[{c, m}];
                const auto k = risk::kupiec_pof(static_cast<std::size_t>(a.n), static_cast<std::size_t>(a.b), 1.0 - c);
                sm << p.key << "," << c << "," << m << "," << a.n << "," << a.b << "," << (1.0 - c) * a.n << "," << double(a.b) / a.n << ","
                   << k.p_value << "," << a.var / a.n << "," << (a.b ? a.bl / a.be : NAN) << "\n";
                std::printf(" %9.1f%% p%4.2f", 100.0 * a.b / a.n, k.p_value);
            }
            std::printf("\n");
        }
    }
    for(double c : confs){
        std::printf("%-26s %6d", (std::string("all portfolios") + (c == 0.99 ? " 99%" : " 97.5%")).c_str(), pooled[{c, names[0]}].n);
        for(const auto& m : names){
            const Acc& a = pooled[{c, m}];
            const auto k = risk::kupiec_pof(static_cast<std::size_t>(a.n), static_cast<std::size_t>(a.b), 1.0 - c);
            sm << "all," << c << "," << m << "," << a.n << "," << a.b << "," << (1.0 - c) * a.n << "," << double(a.b) / a.n << ","
               << k.p_value << "," << a.var / a.n << "," << (a.b ? a.bl / a.be : NAN) << "\n";
            std::printf(" %9.2f%% p%4.2f", 100.0 * a.b / a.n, k.p_value);
        }
        std::printf("\n");
    }
}
}

int main(){
    data::PriceTable table;
    try{
        table = data::load_price_csv("data/market_data/long_history_close.csv");
    } catch(const std::exception& e){
        std::fprintf(stderr, "run from the repo root after tools/fetch_data.py: %s\n", e.what());
        return 1;
    }
    //keep the US trading calendar: drop the rows on which SPY has no price (index-only holidays)
    const auto spy = static_cast<Eigen::Index>(std::find(table.tickers.begin(), table.tickers.end(), "SPY") - table.tickers.begin());
    std::vector<Eigen::Index> keep;
    for(Eigen::Index i = 0; i < table.prices.rows(); ++i) if(std::isfinite(table.prices(i, spy))) keep.push_back(i);
    g_prices.resize(static_cast<Eigen::Index>(keep.size()), table.prices.cols());
    for(std::size_t k = 0; k < keep.size(); ++k){
        g_prices.row(static_cast<Eigen::Index>(k)) = table.prices.row(keep[k]);
        g_dates.push_back(table.dates[static_cast<std::size_t>(keep[k])]);
    }
    g_tickers = table.tickers;
    std::filesystem::create_directories(kOut);
    std::printf("Prices %s to %s, %zu trading days\n", g_dates.front().c_str(), g_dates.back().c_str(), g_dates.size());
    const auto t0 = std::chrono::steady_clock::now();
    part1();
    part2();
    part3();
    std::printf("\n%.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}
