#include <gtest/gtest.h>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "qre/engine/config.hpp"
#include "qre/engine/engine.hpp"
#include "qre/engine/parallel.hpp"
#include "qre/engine/report.hpp"
using namespace qre;
using namespace qre::engine;
namespace fs = std::filesystem;

namespace {
//Two-asset random walk written as a price CSV, run through the engine with fast models
std::string write_prices(const fs::path& dir){
    fs::create_directories(dir);
    const fs::path p = dir / "prices.csv";
    std::ofstream f(p);
    f.precision(17);
    f << "Date,AAA,BBB\n";
    std::mt19937_64 eng(7);
    std::normal_distribution<double> nd(0.0, 0.01);
    double a = 100.0, b = 100.0;
    for(int i = 0; i < 230; ++i){
        a *= 1.0 + nd(eng);
        b *= 1.0 + nd(eng);
        char date[16];
        std::snprintf(date, sizeof(date), "2020-%02d-%02d", 1 + i / 28, 1 + i % 28);
        f << date << "," << a << "," << b << "\n";
    }
    return p.string();
}
RunConfig small_config(const fs::path& dir){
    return parse_config("data = " + write_prices(dir) + "\nconfidence = 0.99, 0.95\nwindow = 100\n"
                        "models = historical, parametric, ewma, fhs\ncombination = select\nlookback = 30\nburn = 40\n"
                        "output = " + (dir / "out").string());
}
}

TEST(EngineConfig, ParsesAndRoundTrips){
    const RunConfig c = parse_config("# comment\n data = prices.csv\nweights = AAA=0.75, BBB = 0.25  # trailing\n"
                                     "confidence = 0.99, 0.975\nmodels = historical, garch_t\ncombination = exp\neta = 0.5\n");
    EXPECT_EQ(c.data, "prices.csv");
    ASSERT_EQ(c.weights.size(), 2u);
    EXPECT_EQ(c.weights[1].first, "BBB");
    EXPECT_DOUBLE_EQ(c.weights[1].second, 0.25);
    EXPECT_EQ(c.confidence.size(), 2u);
    EXPECT_EQ(c.models[1], "garch_t");
    EXPECT_EQ(c.combination, risk::Combination::ExpWeights);
    EXPECT_EQ(c.window, 250);
    EXPECT_EQ(canonical(parse_config(canonical(c))), canonical(c));
    EXPECT_NE(canonical(c).find("confidence = 0.99, 0.975"), std::string::npos);
}

TEST(EngineConfig, ErrorsNameTheLine){
    auto msg = [](const std::string& text){
        try{ parse_config(text); } catch(const InvalidInputError& e){ return std::string(e.what()); }
        return std::string("no error");
    };
    EXPECT_EQ(msg("data = x\nwindwo = 250"), "line 2: unknown key 'windwo'");
    EXPECT_EQ(msg("data = x\nwindow = 2.5").rfind("line 2: 'window' needs an integer", 0), 0u);
    EXPECT_EQ(msg("data = x\nmodels = historical, magic"), "line 2: unknown model 'magic'");
    EXPECT_EQ(msg("data = x\ndata = y"), "line 2: 'data' given twice");
    EXPECT_EQ(msg("data = x\nweights"), "line 2: expected key = value");
    EXPECT_EQ(msg("window = 250"), "config needs 'data'");
    EXPECT_NE(msg("data = x\nmodels = ewma, ewma").find("listed twice"), std::string::npos);
}

TEST(EngineParallel, CoversEveryTaskOnceAndRethrows){
    std::vector<std::atomic<int>> hits(1000);
    parallel_for(hits.size(), 4, [&](std::size_t i){ hits[i]++; });
    for(const auto& h : hits) EXPECT_EQ(h.load(), 1);
    EXPECT_THROW(parallel_for(100, 3, [](std::size_t i){ if(i == 57) throw std::runtime_error("task 57"); }), std::runtime_error);
    EXPECT_GE(resolve_threads(0), 1u);
}

TEST(EnginePieces, FingerprintAndWeights){
    EXPECT_EQ(fnv1a64(""), 0xcbf29ce484222325ull);
    EXPECT_EQ(fnv1a64("a"), 0xaf63dc4c8601ec8cull);
    EXPECT_EQ(fnv1a64("foobar"), 0x85944171f73967e8ull);
    const WeightVector w = resolve_weights(parse_config("data = x\nweights = B=0.6, A=0.4"), {"A", "B", "C"});
    EXPECT_DOUBLE_EQ(w(0), 0.4);
    EXPECT_DOUBLE_EQ(w(1), 0.6);
    EXPECT_DOUBLE_EQ(w(2), 0.0);
    EXPECT_THROW(resolve_weights(parse_config("data = x\nweights = Z=1"), {"A"}), InvalidInputError);
    EXPECT_THROW(resolve_weights(parse_config("data = x\nweights = A=0.5"), {"A", "B"}), InvalidInputError);
}

TEST(Engine, MatchesBacktestAndIsThreadInvariant){
    const fs::path dir = fs::temp_directory_path() / "qre_phase14_engine";
    const RunConfig cfg = small_config(dir);
    const EngineResult a = run_engine(cfg, 1);
    const EngineResult b = run_engine(cfg, 4);
    ASSERT_EQ(a.results.size(), 2u);
    for(std::size_t c = 0; c < 2; ++c){
        const ConfidenceResult& x = a.results[c];
        const ConfidenceResult& y = b.results[c];
        EXPECT_EQ(x.forecasts.rows.front(), 101);
        EXPECT_EQ(x.forecasts.rows.back(), 229);
        EXPECT_TRUE(x.forecasts.var == y.forecasts.var);
        EXPECT_TRUE(x.forecasts.es == y.forecasts.es);
        EXPECT_EQ(x.combined.var, y.combined.var);
        EXPECT_TRUE(x.next_weights == y.next_weights);
        for(std::size_t k = 0; k < cfg.models.size(); ++k){
            const risk::BacktestSeries s = risk::rolling_backtest(a.prices, a.weights, 100, x.confidence, model_from_name(cfg.models[k]));
            for(std::size_t d = 0; d < s.var.size(); ++d){
                EXPECT_EQ(x.forecasts.var(static_cast<Eigen::Index>(d), static_cast<Eigen::Index>(k)), s.var[d]);
            }
            const risk::VarEs nd = risk::forecast_var_es(a.prices, a.weights, 230, 100, x.confidence, model_from_name(cfg.models[k]));
            EXPECT_EQ(x.next_day[k].var, nd.var);
        }
        EXPECT_NEAR(x.next_weights.sum(), 1.0, 1e-15);
        EXPECT_EQ(x.summaries.size(), cfg.models.size() + 1);
        EXPECT_EQ(x.summaries.front().n_obs, 89u);
    }
}

TEST(Engine, WritesIdenticalReportsForAnyThreadCount){
    const fs::path dir = fs::temp_directory_path() / "qre_phase14_report";
    RunConfig cfg = small_config(dir);
    auto read = [](const fs::path& p){ std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); };
    write_report(run_engine(cfg, 1), (dir / "t1").string());
    write_report(run_engine(cfg, 3), (dir / "t3").string());
    for(const char* name : {"forecasts_99.csv", "forecasts_95.csv", "backtest_summary.csv", "next_day.csv", "manifest.txt", "report.md"}){
        EXPECT_TRUE(fs::exists(dir / "t1" / name)) << name;
        EXPECT_EQ(read(dir / "t1" / name), read(dir / "t3" / name)) << name;
    }
    const std::string man = read(dir / "t1" / "manifest.txt");
    EXPECT_NE(man.find("evaluation_days = 89"), std::string::npos);
}

TEST(Engine, RefusesDirtyData){
    const fs::path dir = fs::temp_directory_path() / "qre_phase14_dirty";
    fs::create_directories(dir);
    {
        std::ofstream f(dir / "bad.csv");
        f << "Date,AAA\n";
        for(int d = 1; d <= 28; ++d) f << "2020-01-" << (d < 10 ? "0" : "") << d << "," << (d == 10 ? -5 : 100 + d) << "\n";
    }
    const RunConfig cfg = parse_config("data = " + (dir / "bad.csv").string() + "\nwindow = 5\nburn = 2\nmodels = historical");
    EXPECT_THROW(run_engine(cfg), InvalidInputError);
}
