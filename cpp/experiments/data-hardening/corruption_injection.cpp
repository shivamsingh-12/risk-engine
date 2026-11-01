//Data-hardening experiment: how much do realistic data errors distort the Phase 7 risk numbers,
//and how much of the damage does the hardened data layer detect and repair?
//Start from the real 6-stock history (clean, verified by the quality report), inject one corruption at
//a time plus all of them together, then compare two treatments against the clean baseline:
//  naive    = Phase 7 pipeline straight on the corrupted prices (its own short forward fill only)
//  hardened = harden() -> apply the split candidates the report flags (the analyst confirms them)
//             -> Phase 7 pipeline
//Separately, a survivorship check: one stock delists partway through, compare dropping it from the
//whole history (cleaning.hpp's drop_missing_assets) against the point-in-time universe
//C++ only: the hardening logic is cross-validated against the Python reference in the test suites
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "qre/core/rng.hpp"
#include "qre/data/cleaning.hpp"
#include "qre/data/corporate_actions.hpp"
#include "qre/data/csv_loader.hpp"
#include "qre/data/point_in_time.hpp"
#include "qre/data/quality.hpp"
#include "qre/risk/pipeline.hpp"
using qre::PriceMatrix;
using qre::WeightVector;

namespace {
struct Metrics{ double hist_var, hist_es, param_var, worst_day; };

Metrics measure(const PriceMatrix& prices, const WeightVector& w){
    qre::Rng rng(7);
    qre::risk::RiskConfig cfg;
    cfg.confidence = 0.99;
    cfg.n_paths = 20000;        //Monte Carlo is not reported here; keep it cheap
    const auto rep = qre::risk::run_risk_pipeline(prices, w, rng, cfg);
    const PriceMatrix filled = qre::data::forward_fill(prices, cfg.max_fill_days);
    const auto port = qre::risk::portfolio_returns(qre::data::simple_return(filled), w);
    double worst = 0.0;
    for(double r : port) worst = std::min(worst, r);
    return Metrics{rep.historical.var, rep.historical.es, rep.parametric.var, worst};
}

PriceMatrix harden_and_confirm(const PriceMatrix& raw, qre::data::QualityReport& report){
    const auto res = qre::data::harden(raw);
    report = res.report;
    std::vector<qre::data::CorporateAction> confirmed;
    for(const auto& s : report.split_candidates){
        confirmed.push_back({s.asset, s.row, qre::data::ActionType::Split, s.ratio});
    }
    return qre::data::back_adjust(res.prices, confirmed);
}

double pct(double x, double base){ return 100.0 * (x / base - 1.0); }
}

int main(){
    const std::string out_dir = "cpp/experiments/data-hardening/results/";
    qre::data::PriceTable table;
    try{ table = qre::data::load_price_csv("data/market_data/adjusted_close.csv"); }
    catch(const std::exception& e){
        std::cerr << e.what() << "\nRun from repo root \n";
        return 1;
    }
    std::ofstream out(out_dir + "corruption_results.csv");
    std::ofstream surv(out_dir + "survivorship_results.csv");
    if(!out.is_open() || !surv.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    out.precision(10); surv.precision(10);

    const PriceMatrix& real = table.prices;
    const Eigen::Index n = real.cols();
    const WeightVector w = WeightVector::Constant(n, 1.0 / static_cast<double>(n));
    auto idx = [&](const std::string& t){
        for(std::size_t i = 0; i < table.tickers.size(); ++i) if(table.tickers[i] == t) return static_cast<Eigen::Index>(i);
        throw std::runtime_error("ticker not found: " + t);
    };
    const Eigen::Index AAPL = idx("AAPL"), XOM = idx("XOM"), JNJ = idx("JNJ"), PG = idx("PG"), CAT = idx("CAT");
    const double nan = std::numeric_limits<double>::quiet_NaN();

    //The real data must be clean, otherwise the baseline itself is suspect
    const auto base_check = qre::data::harden(real);
    std::cout << "real data quality report clean: " << (base_check.report.clean() ? "yes" : "NO") << "\n";
    if(!base_check.report.clean()) return 1;
    const Metrics base = measure(real, w);

    struct Scenario{ std::string name; std::string what; PriceMatrix prices; };
    std::vector<Scenario> scenarios;
    {   PriceMatrix p = real; p.block(400, AAPL, p.rows() - 400, 1) /= 4.0;
        scenarios.push_back({"unadjusted_split", "AAPL 4-for-1 split left unadjusted from " + table.dates[400], p}); }
    {   PriceMatrix p = real; p(200, XOM) *= 1.35;
        scenarios.push_back({"bad_tick", "XOM +35% for one day on " + table.dates[200], p}); }
    {   PriceMatrix p = real; p.block(300, JNJ, 10, 1).setConstant(p(299, JNJ));
        scenarios.push_back({"stale_run", "JNJ frozen for 10 days from " + table.dates[300], p}); }
    {   PriceMatrix p = real; p.block(500, PG, 3, 1).setConstant(nan);
        scenarios.push_back({"gap", "PG missing 3 days from " + table.dates[500], p}); }
    {   PriceMatrix p = scenarios[0].prices;
        p(200, XOM) *= 1.35;
        p.block(300, JNJ, 10, 1).setConstant(p(299, JNJ));
        p.block(500, PG, 3, 1).setConstant(nan);
        scenarios.push_back({"all_combined", "all four corruptions together", p}); }

    out << "scenario,treatment,hist_var99,hist_es99,param_var99,worst_day,hist_var_err_pct,hist_es_err_pct,param_var_err_pct,"
           "split_flags,bad_tick_flags,stale_flags,missing_filled\n";
    out << "baseline,clean," << base.hist_var << "," << base.hist_es << "," << base.param_var << "," << base.worst_day
        << ",0,0,0,0,0,0,0\n";
    std::cout << "baseline: histVaR99=" << base.hist_var << " histES99=" << base.hist_es << " paramVaR99=" << base.param_var
              << " worst=" << base.worst_day << "\n";

    for(const auto& s : scenarios){
        const Metrics naive = measure(s.prices, w);
        qre::data::QualityReport rep;
        const PriceMatrix fixed = harden_and_confirm(s.prices, rep);
        const Metrics hard = measure(fixed, w);
        Eigen::Index filled = 0;
        for(std::size_t i = 0; i < rep.internal_missing.size(); ++i) filled += rep.internal_missing[i] - rep.remaining_internal_missing[i];
        for(const auto& [label, m] : {std::pair<std::string, Metrics>{"naive", naive}, {"hardened", hard}}){
            out << s.name << "," << label << "," << m.hist_var << "," << m.hist_es << "," << m.param_var << "," << m.worst_day << ","
                << pct(m.hist_var, base.hist_var) << "," << pct(m.hist_es, base.hist_es) << "," << pct(m.param_var, base.param_var) << ","
                << rep.split_candidates.size() << "," << rep.bad_ticks.size() << "," << rep.stale.size() << "," << filled << "\n";
            std::cout << s.name << " / " << label << ": histVaR " << pct(m.hist_var, base.hist_var) << "%  histES "
                      << pct(m.hist_es, base.hist_es) << "%  paramVaR " << pct(m.param_var, base.param_var) << "%  worst "
                      << m.worst_day << "\n";
        }
        std::cout << "   flags: split=" << rep.split_candidates.size() << " bad_tick=" << rep.bad_ticks.size()
                  << " stale=" << rep.stale.size() << " filled=" << filled << "   (" << s.what << ")\n";
    }

    //Survivorship: CAT delists after 2023-05-22 (row 599). Risk of the portfolio actually held on that day,
    //estimated over the previous 500 days
    const Eigen::Index t = 600, window = 500;
    PriceMatrix delisted = real;
    delisted.block(t, CAT, delisted.rows() - t, 1).setConstant(nan);
    surv << "approach,assets_used,hist_var99,param_var99\n";
    {   //Point in time: everything alive in the window, which is what was held
        const auto uni = qre::data::universe_at(delisted, t, window);
        const PriceMatrix win = qre::data::pit_window(delisted, t, window);
        PriceMatrix sub(win.rows(), static_cast<Eigen::Index>(uni.size()));
        for(std::size_t i = 0; i < uni.size(); ++i) sub.col(static_cast<Eigen::Index>(i)) = win.col(uni[i]);
        const WeightVector ws = WeightVector::Constant(sub.cols(), 1.0 / static_cast<double>(sub.cols()));
        const Metrics m = measure(sub, ws);
        surv << "point_in_time," << uni.size() << "," << m.hist_var << "," << m.param_var << "\n";
        std::cout << "survivorship / point_in_time: assets=" << uni.size() << " histVaR99=" << m.hist_var << " paramVaR99=" << m.param_var << "\n";
    }
    {   //Old approach: drop any asset missing too much of the full history -> CAT vanishes even before it delisted
        const auto cleaned = qre::data::drop_missing_assets(delisted, 0.05);
        const PriceMatrix win = qre::data::pit_window(cleaned.clean_prices, t, window);
        const WeightVector ws = WeightVector::Constant(win.cols(), 1.0 / static_cast<double>(win.cols()));
        const Metrics m = measure(win, ws);
        surv << "drop_missing_full_history," << win.cols() << "," << m.hist_var << "," << m.param_var << "\n";
        std::cout << "survivorship / drop_missing: assets=" << win.cols() << " histVaR99=" << m.hist_var << " paramVaR99=" << m.param_var << "\n";
    }
    std::cout << "Results written to " << out_dir << "\n";
    return 0;
}
