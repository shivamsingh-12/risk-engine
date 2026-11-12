#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"
#include "qre/data/csv_loader.hpp"
#include "qre/data/quality.hpp"
#include "qre/engine/config.hpp"
#include "qre/engine/parallel.hpp"
#include "qre/risk/adaptive.hpp"
#include "qre/risk/backtest.hpp"
#include "qre/risk/scoring.hpp"
//Phase 14: the production engine -- one call from a config to every number in the daily risk report
//  1. read the price file (and fingerprint its bytes), harden it, refuse dirty data
//  2. forecast VaR/ES for every confidence x model x day, in parallel (every backtest day plus the next, unseen day)
//  3. per confidence: score every model daily (FZ0), combine them, backtest models and the combination over the
//     evaluation days (after `burn`), and forecast the next day's combined VaR/ES
//Each forecast depends only on its own (confidence, model, day), so the output is identical for any thread count
namespace qre::engine{
    inline constexpr const char* kVersion = "1.0.0";

    struct ConfidenceResult{
        double confidence;
        risk::ModelForecasts forecasts;             //backtest days: rows, loss, var/es (T x K)
        Eigen::MatrixXd scores;                     //FZ0, T x K
        risk::CombinedForecast combined;            //evaluation days (from burn)
        std::vector<risk::BacktestSummary> summaries;   //K models then the combination, evaluation days
        std::vector<double> mean_fz0;               //same order
        std::vector<risk::VarEs> next_day;          //per model, for the day after the last price
        Eigen::VectorXd next_weights;
        risk::VarEs next_combined;
    };

    struct StageTimes{
        double load_ms = 0.0, forecast_ms = 0.0, combine_ms = 0.0;
        std::size_t threads = 1, tasks = 0;
    };

    struct EngineResult{
        RunConfig config;
        std::vector<std::string> dates, tickers;
        PriceMatrix prices;                         //hardened
        WeightVector weights;
        data::QualityReport quality;
        uint64_t data_hash = 0;
        std::size_t data_bytes = 0;
        std::vector<ConfidenceResult> results;
    };

    //64-bit FNV-1a of a byte string: a cheap fingerprint so a report can be tied to the exact input file
    inline uint64_t fnv1a64(const std::string& bytes){
        uint64_t h = 14695981039346656037ull;
        for(char c : bytes){ h ^= static_cast<unsigned char>(c); h *= 1099511628211ull; }
        return h;
    }

    inline WeightVector resolve_weights(const RunConfig& c, const std::vector<std::string>& tickers){
        const auto n = static_cast<Eigen::Index>(tickers.size());
        if(c.equal_weights) return WeightVector::Constant(n, 1.0 / static_cast<double>(n));
        WeightVector w = WeightVector::Zero(n);
        for(const auto& [name, x] : c.weights){
            const auto it = std::find(tickers.begin(), tickers.end(), name);
            if(it == tickers.end()) throw InvalidInputError("weight for unknown ticker '" + name + "'");
            const auto j = static_cast<Eigen::Index>(it - tickers.begin());
            if(w(j) != 0.0) throw InvalidInputError("ticker '" + name + "' weighted twice");
            w(j) = x;
        }
        if(std::abs(w.sum() - 1.0) > 1e-9) throw InvalidInputError("weights must sum to 1");
        return w;
    }

    inline EngineResult run_engine(const RunConfig& cfg, std::size_t threads = 0, StageTimes* times = nullptr){
        using clock = std::chrono::steady_clock;
        auto ms = [](clock::time_point a, clock::time_point b){ return std::chrono::duration<double, std::milli>(b - a).count(); };
        const auto t0 = clock::now();
        EngineResult r;
        r.config = cfg;
        {
            std::ifstream f(cfg.data, std::ios::binary);
            if(!f.is_open()) throw InvalidInputError("cannot open data file '" + cfg.data + "'");
            const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            r.data_hash = fnv1a64(bytes);
            r.data_bytes = bytes.size();
        }
        const data::PriceTable table = data::load_price_csv(cfg.data);
        r.dates = table.dates;
        r.tickers = table.tickers;
        const data::HardenResult hardened = data::harden(table.prices);
        r.quality = hardened.report;
        if(!r.quality.clean()) throw InvalidInputError("price data failed the quality report; fix the data before running the engine");
        r.prices = hardened.prices;
        r.weights = resolve_weights(cfg, r.tickers);
        risk::check_backtest_inputs(r.prices, r.weights, cfg.window);
        const Eigen::Index T = r.prices.rows();
        const Eigen::Index first = cfg.window + 1;
        const auto n_days = static_cast<std::size_t>(T - first);          //backtest days
        if(static_cast<long>(n_days) <= cfg.burn) throw InvalidInputError("burn must leave at least one evaluation day");
        const auto t1 = clock::now();

        //every confidence x model x day, plus the next day (t = T)
        std::vector<risk::BacktestMethod> methods;
        for(const auto& m : cfg.models) methods.push_back(model_from_name(m));
        const std::size_t C = cfg.confidence.size(), K = methods.size(), D = n_days + 1;
        std::vector<risk::VarEs> out(C * K * D);
        const std::size_t used = resolve_threads(threads);
        parallel_for(out.size(), used, [&](std::size_t i){
            const std::size_t c = i / (K * D), k = (i / D) % K, d = i % D;
            const Eigen::Index t = first + static_cast<Eigen::Index>(d);
            out[i] = risk::forecast_var_es(r.prices, r.weights, t, cfg.window, cfg.confidence[c], methods[k],
                                           static_cast<std::size_t>(cfg.mc_paths), static_cast<uint64_t>(cfg.seed), 252, cfg.lambda);
        });
        const auto t2 = clock::now();

        std::vector<double> loss(n_days);
        std::vector<Eigen::Index> rows(n_days);
        for(std::size_t d = 0; d < n_days; ++d){
            rows[d] = first + static_cast<Eigen::Index>(d);
            loss[d] = risk::realized_loss(r.prices, r.weights, rows[d]);
        }
        for(std::size_t c = 0; c < C; ++c){
            ConfidenceResult cr;
            cr.confidence = cfg.confidence[c];
            cr.forecasts.methods = methods;
            cr.forecasts.loss = loss;
            cr.forecasts.rows = rows;
            cr.forecasts.var.resize(static_cast<Eigen::Index>(n_days), static_cast<Eigen::Index>(K));
            cr.forecasts.es.resize(cr.forecasts.var.rows(), cr.forecasts.var.cols());
            for(std::size_t k = 0; k < K; ++k){
                for(std::size_t d = 0; d < n_days; ++d){
                    const risk::VarEs& f = out[(c * K + k) * D + d];
                    cr.forecasts.var(static_cast<Eigen::Index>(d), static_cast<Eigen::Index>(k)) = f.var;
                    cr.forecasts.es(static_cast<Eigen::Index>(d), static_cast<Eigen::Index>(k)) = f.es;
                }
                cr.next_day.push_back(out[(c * K + k) * D + n_days]);
            }
            cr.scores = risk::model_scores(cr.forecasts.var, cr.forecasts.es, loss, cr.confidence);
            cr.combined = risk::combine(cr.forecasts.var, cr.forecasts.es, loss, cr.confidence, cfg.combination, cfg.lookback, cfg.eta, cfg.burn);
            for(std::size_t k = 0; k <= K; ++k){
                const risk::BacktestSeries s = k < K ? risk::model_series(cr.forecasts, static_cast<Eigen::Index>(k), cfg.burn)
                                                     : risk::as_series(cr.forecasts, cr.combined.var, cr.combined.es, cfg.burn);
                cr.summaries.push_back(risk::summarize(s, cr.confidence));
                const std::vector<double> sc = risk::fz0_loss(s.loss, s.var, s.es, cr.confidence);
                double m = 0.0;
                for(double x : sc) m += x;
                cr.mean_fz0.push_back(m / static_cast<double>(sc.size()));
            }
            cr.next_weights = risk::combination_weights(cr.scores, cr.scores.rows(), cfg.combination, cfg.lookback, cfg.eta);
            cr.next_combined = {0.0, 0.0};
            for(std::size_t k = 0; k < K; ++k){
                cr.next_combined.var += cr.next_weights(static_cast<Eigen::Index>(k)) * cr.next_day[k].var;
                cr.next_combined.es += cr.next_weights(static_cast<Eigen::Index>(k)) * cr.next_day[k].es;
            }
            r.results.push_back(std::move(cr));
        }
        const auto t3 = clock::now();
        if(times){
            *times = StageTimes{ms(t0, t1), ms(t1, t2), ms(t2, t3), used, out.size()};
        }
        return r;
    }
}
