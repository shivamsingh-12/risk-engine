#pragma once
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include "qre/core/exceptions.hpp"
#include "qre/engine/engine.hpp"
//Phase 14: writing the run to disk
//  forecasts_<c>.csv     every backtest day: date, realized loss, each model's VaR/ES, the combination's VaR/ES
//  backtest_summary.csv  per confidence and model (and the combination): coverage tests, Z2, mean FZ0
//  next_day.csv          the forecasts for the day after the last price, with the combination weights
//  report.md             the same, for people
//  manifest.txt          what produced the numbers: version, canonical config, data fingerprint, data range, weights
//All of these are deterministic. Timing and thread count go to run_log.txt, which is the only file that can differ
//between two runs of the same config on the same data.
namespace qre::engine{
    namespace report_detail{
        inline std::string num(double x){
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.15g", x);
            return buf;
        }
        inline std::string pct(double x){
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.3f%%", 100.0 * x);
            return buf;
        }
        inline std::string conf_tag(double c){
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%g", 100.0 * c);
            return buf;
        }
        inline std::ofstream open(const std::filesystem::path& p){
            std::ofstream f(p);
            if(!f.is_open()) throw InvalidInputError("cannot write '" + p.string() + "'");
            return f;
        }
    }

    inline std::string hex64(uint64_t h){
        char buf[24];
        std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
        return buf;
    }

    inline void write_report(const EngineResult& r, const std::string& dir){
        using namespace report_detail;
        namespace fs = std::filesystem;
        fs::create_directories(dir);
        const RunConfig& cfg = r.config;
        const std::size_t K = cfg.models.size();
        const auto burn = static_cast<std::size_t>(cfg.burn);
        const std::string next_label = "after " + r.dates.back();

        auto summary = open(fs::path(dir) / "backtest_summary.csv");
        summary << "confidence,model,n_obs,breaches,expected,breach_rate,kupiec_p,christoffersen_p,conditional_p,traffic_light,z2,mean_fz0\n";
        auto next = open(fs::path(dir) / "next_day.csv");
        next << "confidence,model,var,es,weight\n";
        for(const ConfidenceResult& cr : r.results){
            auto fc = open(fs::path(dir) / ("forecasts_" + conf_tag(cr.confidence) + ".csv"));
            fc << "date,loss";
            for(const auto& m : cfg.models) fc << "," << m << "_var," << m << "_es";
            fc << ",combined_var,combined_es\n";
            for(std::size_t d = 0; d < cr.forecasts.loss.size(); ++d){
                fc << r.dates[static_cast<std::size_t>(cr.forecasts.rows[d])] << "," << num(cr.forecasts.loss[d]);
                for(std::size_t k = 0; k < K; ++k){
                    fc << "," << num(cr.forecasts.var(static_cast<Eigen::Index>(d), static_cast<Eigen::Index>(k)))
                       << "," << num(cr.forecasts.es(static_cast<Eigen::Index>(d), static_cast<Eigen::Index>(k)));
                }
                if(d >= burn) fc << "," << num(cr.combined.var[d - burn]) << "," << num(cr.combined.es[d - burn]) << "\n";
                else fc << ",,\n";
            }
            for(std::size_t k = 0; k <= K; ++k){
                const risk::BacktestSummary& s = cr.summaries[k];
                summary << num(cr.confidence) << "," << (k < K ? cfg.models[k] : "combined") << "," << s.n_obs << "," << s.n_breaches
                        << "," << num((1.0 - cr.confidence) * static_cast<double>(s.n_obs)) << "," << num(s.breach_rate) << ","
                        << num(s.kupiec.p_value) << "," << num(s.christoffersen.p_value) << "," << num(s.conditional.p_value) << ","
                        << risk::to_string(s.traffic_light) << "," << num(s.z2) << "," << num(cr.mean_fz0[k]) << "\n";
            }
            for(std::size_t k = 0; k < K; ++k){
                next << num(cr.confidence) << "," << cfg.models[k] << "," << num(cr.next_day[k].var) << "," << num(cr.next_day[k].es)
                     << "," << num(cr.next_weights(static_cast<Eigen::Index>(k))) << "\n";
            }
            next << num(cr.confidence) << ",combined," << num(cr.next_combined.var) << "," << num(cr.next_combined.es) << ",1\n";
        }

        auto man = open(fs::path(dir) / "manifest.txt");
        man << "qre_version = " << kVersion << "\n[config]\n" << canonical(cfg) << "[data]\n"
            << "fnv1a64 = " << hex64(r.data_hash) << "\nbytes = " << r.data_bytes << "\n"
            << "dates = " << r.dates.front() << " .. " << r.dates.back() << " (" << r.dates.size() << " rows)\n"
            << "tickers = ";
        for(std::size_t j = 0; j < r.tickers.size(); ++j) man << (j ? ", " : "") << r.tickers[j];
        man << "\nweights = ";
        for(std::size_t j = 0; j < r.tickers.size(); ++j) man << (j ? ", " : "") << r.tickers[j] << "=" << num(r.weights(static_cast<Eigen::Index>(j)));
        Eigen::Index filled = 0;
        for(auto v : r.quality.internal_missing) filled += v;
        man << "\nquality = " << (r.quality.clean() ? "clean" : "dirty") << " (" << filled << " internal gaps filled)\n"
            << "[run]\nbacktest_days = " << r.results.front().forecasts.loss.size() << "\nevaluation_days = "
            << r.results.front().forecasts.loss.size() - burn << " (first " << burn << " days only build scores)\n"
            << "first_evaluated = " << r.dates[static_cast<std::size_t>(r.results.front().forecasts.rows[burn])] << "\n";

        auto md = open(fs::path(dir) / "report.md");
        md << "# Daily risk report\n\n"
           << "Portfolio of " << r.tickers.size() << " assets, prices " << r.dates.front() << " to " << r.dates.back()
           << ". Combination: " << combination_name(cfg.combination) << ". Data fingerprint " << hex64(r.data_hash) << ".\n\n"
           << "## Next-day risk (" << next_label << ")\n\n| Confidence | Model | VaR | ES | Weight |\n| :-- | :-- | --: | --: | --: |\n";
        for(const ConfidenceResult& cr : r.results){
            for(std::size_t k = 0; k < K; ++k){
                md << "| " << conf_tag(cr.confidence) << "% | " << cfg.models[k] << " | " << pct(cr.next_day[k].var) << " | "
                   << pct(cr.next_day[k].es) << " | " << pct(cr.next_weights(static_cast<Eigen::Index>(k))) << " |\n";
            }
            md << "| " << conf_tag(cr.confidence) << "% | **combined** | **" << pct(cr.next_combined.var) << "** | **"
               << pct(cr.next_combined.es) << "** | |\n";
        }
        md << "\nVaR and ES are one-day losses as a fraction of portfolio value.\n\n## Backtest over "
           << r.results.front().forecasts.loss.size() - burn << " evaluation days\n\n"
           << "| Confidence | Model | Breaches (expected) | Kupiec p | Christoffersen p | Traffic light | Z2 | Mean FZ0 |\n"
           << "| :-- | :-- | --: | --: | --: | :-- | --: | --: |\n";
        for(const ConfidenceResult& cr : r.results){
            for(std::size_t k = 0; k <= K; ++k){
                const risk::BacktestSummary& s = cr.summaries[k];
                char line[400];
                std::snprintf(line, sizeof(line), "| %s%% | %s | %zu (%.1f) | %.3f | %.3f | %s | %.3f | %.4f |\n",
                              conf_tag(cr.confidence).c_str(), k < K ? cfg.models[k].c_str() : "**combined**", s.n_breaches,
                              (1.0 - cr.confidence) * static_cast<double>(s.n_obs), s.kupiec.p_value, s.christoffersen.p_value,
                              risk::to_string(s.traffic_light).c_str(), s.z2, cr.mean_fz0[k]);
                md << line;
            }
        }
        md << "\nA p-value below 0.05 flags a model: too many or too few breaches (Kupiec) or breaches that cluster "
              "(Christoffersen). Z2 above 0 means ES was too small. Lower mean FZ0 is better.\n";
    }

    inline void write_run_log(const StageTimes& t, const std::string& dir){
        std::ofstream f(std::filesystem::path(dir) / "run_log.txt");
        f << "threads = " << t.threads << "\ntasks = " << t.tasks << "\nload_ms = " << report_detail::num(t.load_ms)
          << "\nforecast_ms = " << report_detail::num(t.forecast_ms) << "\ncombine_ms = " << report_detail::num(t.combine_ms) << "\n";
    }
}
