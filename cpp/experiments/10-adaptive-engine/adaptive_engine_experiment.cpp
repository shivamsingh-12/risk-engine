//Phase 13 experiment: an adaptive risk engine -- can scoring the models every day pick or blend them better than
//any one model, and can we even tell the models apart?
//Part 1: synthetic regime switch. 750 calm days (iid normal, 0.8% vol), 750 clustered days (GARCH(1,1) with
//        Student-t(5) shocks, 1.5% long-run vol), 750 calm days again. Eight models forecast 97.5% VaR/ES on a
//        rolling 250-day window; the engine scores them with FZ0 and combines them (equal weights, select the best
//        over the last 100 days, exponential weights). Which model wins in which regime, and how fast does
//        selection react to the switches?
//Part 2: real data. The same eight models and three combinations on the 6-stock portfolio, at 97.5% and 99%:
//        coverage tests, Z2, mean FZ0 and quantile loss, and Diebold-Mariano tests against the equal-weight blend
//Part 3: sensitivity of the combinations to the lookback and to eta (real data, 97.5%)
//C++ only: FZ0, quantile loss, Diebold-Mariano and the combination rules are cross-validated against the Python reference
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "qre/data/csv_loader.hpp"
#include "qre/data/quality.hpp"
#include "qre/risk/adaptive.hpp"
#include "qre/risk/backtest.hpp"
#include "qre/risk/scoring.hpp"
using namespace qre;
using namespace qre::risk;

namespace {
const std::vector<BacktestMethod> kMethods{BacktestMethod::Historical, BacktestMethod::Parametric, BacktestMethod::EwmaParametric,
                                           BacktestMethod::FilteredHistorical, BacktestMethod::GarchParametric,
                                           BacktestMethod::GarchFiltered, BacktestMethod::GarchStudentT, BacktestMethod::GarchEvt};
const std::vector<std::string> kNames{"historical", "parametric", "ewma", "fhs", "garch", "fhs_garch", "garch_t", "garch_evt"};
const char* combo_name(Combination c){
    return c == Combination::Equal ? "equal_weight" : (c == Combination::SelectBest ? "select_best" : "exp_weights");
}
double mean_of(const std::vector<double>& v, std::size_t from, std::size_t to){
    double s = 0.0;
    for(std::size_t i = from; i < to; ++i) s += v[i];
    return s / static_cast<double>(to - from);
}
std::vector<double> scores(const BacktestSeries& s, double c){ return fz0_loss(s.loss, s.var, s.es, c); }
std::vector<double> qscores(const BacktestSeries& s, double c){
    std::vector<double> q(s.loss.size());
    for(std::size_t i = 0; i < q.size(); ++i) q[i] = quantile_loss(s.loss[i], s.var[i], c);
    return q;
}
}

int main(){
    const std::string out_dir = "cpp/experiments/10-adaptive-engine/results/";
    std::ofstream c1(out_dir + "synthetic_regimes.csv"), c1s(out_dir + "synthetic_selection.csv"), c2(out_dir + "real_comparison.csv"),
                  c2s(out_dir + "real_selection.csv"), c3(out_dir + "sensitivity.csv");
    if(!c1.is_open() || !c1s.is_open() || !c2.is_open() || !c2s.is_open() || !c3.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    for(auto* f : {&c1, &c1s, &c2, &c2s, &c3}) f->precision(10);
    const Eigen::Index burn = 100, lookback = 100;
    const double eta = 1.0;
    const Combination combos[3] = {Combination::Equal, Combination::SelectBest, Combination::ExpWeights};

    //------------------------------------------------------------ Part 1
    {
        const std::size_t n_reg = 750;
        std::mt19937_64 eng(1313);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::student_t_distribution<double> td(5.0);
        const double t_scale = std::sqrt(3.0 / 5.0);            //unit-variance t(5)
        const double calm = 0.008, lr = 0.015, alpha = 0.12, beta = 0.86, omega = lr * lr * (1.0 - alpha - beta);
        std::vector<double> r;
        for(std::size_t i = 0; i < n_reg; ++i) r.push_back(calm * nd(eng));
        double h = lr * lr;
        for(std::size_t i = 0; i < n_reg; ++i){
            const double x = std::sqrt(h) * t_scale * td(eng);
            r.push_back(x);
            h = omega + alpha * x * x + beta * h;
        }
        for(std::size_t i = 0; i < n_reg; ++i) r.push_back(calm * nd(eng));
        PriceMatrix p(static_cast<Eigen::Index>(r.size()) + 1, 1);
        p(0, 0) = 100.0;
        for(std::size_t i = 0; i < r.size(); ++i) p(static_cast<Eigen::Index>(i) + 1, 0) = p(static_cast<Eigen::Index>(i), 0) * (1.0 + r[i]);
        const double conf = 0.975;
        const ModelForecasts f = run_models(p, WeightVector::Ones(1), kMethods, 250, conf);
        std::cout << "Part 1: " << f.loss.size() << " synthetic forecasts per model\n";
        //regime of each evaluated day: return on price row t is r[t-1]
        auto regime = [&](Eigen::Index row){
            const auto i = static_cast<std::size_t>(row - 1);
            return i < n_reg ? 0 : (i < 2 * n_reg ? 1 : 2);
        };
        const char* reg_name[3] = {"calm_1", "clustered", "calm_2"};
        std::vector<BacktestSeries> series;
        std::vector<std::string> names = kNames;
        for(Eigen::Index k = 0; k < f.var.cols(); ++k) series.push_back(model_series(f, k, burn));
        std::vector<CombinedForecast> comb;
        for(Combination c : combos){
            comb.push_back(combine(f.var, f.es, f.loss, conf, c, lookback, eta, burn));
            series.push_back(as_series(f, comb.back().var, comb.back().es, burn));
            names.emplace_back(combo_name(c));
        }
        c1 << "regime,days,method,mean_fz0,breach_rate,expected_rate\n";
        for(int g = 0; g < 3; ++g){
            for(std::size_t m = 0; m < series.size(); ++m){
                const std::vector<double> sc = scores(series[m], conf);
                double s = 0.0, br = 0.0, n = 0.0;
                for(std::size_t i = 0; i < sc.size(); ++i){
                    if(regime(series[m].rows[i]) != g) continue;
                    s += sc[i]; n += 1.0;
                    br += series[m].loss[i] > series[m].var[i] ? 1.0 : 0.0;
                }
                c1 << reg_name[g] << "," << n << "," << names[m] << "," << s / n << "," << br / n << "," << 1.0 - conf << "\n";
            }
        }
        //selection share by regime, and the reaction lag of select_best to each switch
        const CombinedForecast& sel = comb[1];
        c1s << "regime,model,select_share,mean_exp_weight\n";
        for(int g = 0; g < 3; ++g){
            for(Eigen::Index k = 0; k < f.var.cols(); ++k){
                double share = 0.0, ew = 0.0, n = 0.0;
                for(Eigen::Index i = 0; i < sel.weights.rows(); ++i){
                    if(regime(f.rows[static_cast<std::size_t>(i + burn)]) != g) continue;
                    share += sel.weights(i, k); ew += comb[2].weights(i, k); n += 1.0;
                }
                c1s << reg_name[g] << "," << kNames[static_cast<std::size_t>(k)] << "," << share / n << "," << ew / n << "\n";
            }
        }
        //static models (historical, parametric) vs adaptive ones, early and late in each regime
        std::ofstream c1b(out_dir + "synthetic_static_share.csv");
        c1b.precision(10);
        c1b << "segment,days,static_share_select_best,static_weight_exp_weights\n";
        const char* seg_name[5] = {"calm_1", "clustered_first_250", "clustered_last_500", "calm_2_first_250", "calm_2_last_500"};
        auto segment = [&](Eigen::Index row){
            const auto i = static_cast<std::size_t>(row - 1);
            if(i < n_reg) return 0;
            if(i < 2 * n_reg) return i < n_reg + 250 ? 1 : 2;
            return i < 2 * n_reg + 250 ? 3 : 4;
        };
        for(int g = 0; g < 5; ++g){
            double ss = 0.0, se = 0.0, n = 0.0;
            for(Eigen::Index i = 0; i < sel.weights.rows(); ++i){
                if(segment(f.rows[static_cast<std::size_t>(i + burn)]) != g) continue;
                ss += sel.weights(i, 0) + sel.weights(i, 1);
                se += comb[2].weights(i, 0) + comb[2].weights(i, 1);
                n += 1.0;
            }
            c1b << seg_name[g] << "," << n << "," << ss / n << "," << se / n << "\n";
        }
        //whole synthetic sample: mean FZ0 and Diebold-Mariano against the equal-weight blend
        std::ofstream c1o(out_dir + "synthetic_overall.csv");
        c1o.precision(10);
        c1o << "method,mean_fz0,dm_vs_equal,dm_p\n";
        const std::vector<double> eq = scores(series[kMethods.size()], conf);
        for(std::size_t m = 0; m < series.size(); ++m){
            const std::vector<double> sc = scores(series[m], conf);
            c1o << names[m] << "," << mean_of(sc, 0, sc.size());
            if(m == kMethods.size()) c1o << ",,\n";
            else{
                const DmResult dm = diebold_mariano(sc, eq);
                c1o << "," << dm.statistic << "," << dm.p_value << "\n";
            }
        }
        int switches = 0;
        Eigen::Index prev = -1;
        for(Eigen::Index i = 0; i < sel.weights.rows(); ++i){
            Eigen::Index k = 0;
            sel.weights.row(i).maxCoeff(&k);
            if(prev >= 0 && k != prev) ++switches;
            prev = k;
        }
        c1o << "#select_best_switches," << switches << "\n";
    }

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

    //------------------------------------------------------------ Part 2 and 3
    c2 << "confidence,method,days,breaches,expected,kupiec_p,christoffersen_p,z2,mean_fz0,mean_quantile_loss,dm_vs_equal,dm_p\n";
    c2s << "confidence,model,select_share,mean_exp_weight,switches_select_best\n";
    c3 << "scheme,lookback,eta,mean_fz0,breaches,kupiec_p,christoffersen_p\n";
    for(double conf : {0.975, 0.99}){
        const ModelForecasts f = run_models(prices, w, kMethods, 250, conf);
        std::cout << "Part 2: c = " << conf << ", " << f.loss.size() << " forecasts per model\n";
        std::vector<BacktestSeries> series;
        std::vector<std::string> names = kNames;
        for(Eigen::Index k = 0; k < f.var.cols(); ++k) series.push_back(model_series(f, k, burn));
        std::vector<CombinedForecast> comb;
        for(Combination c : combos){
            comb.push_back(combine(f.var, f.es, f.loss, conf, c, lookback, eta, burn));
            series.push_back(as_series(f, comb.back().var, comb.back().es, burn));
            names.emplace_back(combo_name(c));
        }
        const std::vector<double> eq_scores = scores(series[kMethods.size()], conf);
        for(std::size_t m = 0; m < series.size(); ++m){
            const BacktestSummary sm = summarize(series[m], conf);
            const std::vector<double> sc = scores(series[m], conf);
            const std::vector<double> q = qscores(series[m], conf);
            c2 << conf << "," << names[m] << "," << sm.n_obs << "," << sm.n_breaches << "," << (1.0 - conf) * static_cast<double>(sm.n_obs)
               << "," << sm.kupiec.p_value << "," << sm.christoffersen.p_value << "," << sm.z2 << "," << mean_of(sc, 0, sc.size())
               << "," << mean_of(q, 0, q.size());
            if(m == kMethods.size()) c2 << ",,\n";
            else{
                const DmResult dm = diebold_mariano(sc, eq_scores);
                c2 << "," << dm.statistic << "," << dm.p_value << "\n";
            }
        }
        const CombinedForecast& sel = comb[1];
        int switches = 0;
        Eigen::Index prev = -1;
        for(Eigen::Index i = 0; i < sel.weights.rows(); ++i){
            Eigen::Index k = 0;
            sel.weights.row(i).maxCoeff(&k);
            if(prev >= 0 && k != prev) ++switches;
            prev = k;
        }
        for(Eigen::Index k = 0; k < f.var.cols(); ++k){
            c2s << conf << "," << kNames[static_cast<std::size_t>(k)] << "," << sel.weights.col(k).mean() << ","
                << comb[2].weights.col(k).mean() << "," << switches << "\n";
        }
        if(conf == 0.975){
            for(Eigen::Index lb : {50, 100, 250}){
                for(Combination c : {Combination::SelectBest, Combination::ExpWeights}){
                    for(double e : {0.1, 1.0, 10.0}){
                        if(c == Combination::SelectBest && e != 1.0) continue;
                        const CombinedForecast cf = combine(f.var, f.es, f.loss, conf, c, lb, e, burn);
                        const BacktestSeries s = as_series(f, cf.var, cf.es, burn);
                        const BacktestSummary sm = summarize(s, conf);
                        const std::vector<double> sc = scores(s, conf);
                        c3 << combo_name(c) << "," << lb << "," << (c == Combination::SelectBest ? std::string("") : std::to_string(e).substr(0, 4))
                           << "," << mean_of(sc, 0, sc.size()) << "," << sm.n_breaches << "," << sm.kupiec.p_value << ","
                           << sm.christoffersen.p_value << "\n";
                    }
                }
            }
        }
    }
    std::cout << "Results in " << out_dir << "\n";
    return 0;
}
