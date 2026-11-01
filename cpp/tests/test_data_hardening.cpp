#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include "qre/data/corporate_actions.hpp"
#include "qre/data/csv_loader.hpp"
#include "qre/data/point_in_time.hpp"
#include "qre/data/quality.hpp"
using namespace qre;
using namespace qre::data;

namespace {
const double NaN = std::numeric_limits<double>::quiet_NaN();

//Deterministic smooth 3-asset history (same fixture as the Python cross-check)
PriceMatrix fixture(Eigen::Index T = 60, Eigen::Index N = 3){
    PriceMatrix p(T, N);
    for(Eigen::Index r = 0; r < T; ++r)
        for(Eigen::Index c = 0; c < N; ++c)
            p(r, c) = 100.0 * std::exp(0.02 * std::sin(0.9 * static_cast<double>(r) + 1.3 * static_cast<double>(c)));
    return p;
}

std::string write_tmp(const std::string& name, const std::string& body){
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << body;
    return path.string();
}

PriceMatrix col(std::initializer_list<double> v){
    PriceMatrix p(static_cast<Eigen::Index>(v.size()), 1);
    Eigen::Index i = 0;
    for(double x : v) p(i++, 0) = x;
    return p;
}
}

//------------------------------------------------------------------ loading
TEST(CsvLoader, LoadsPricesDatesAndMissing){
    const auto path = write_tmp("qre_ok.csv", "Date,AAA,BBB\n2024-01-02,10.5,20\n2024-01-03,,21\n2024-01-04,11,22.5\n");
    const PriceTable t = load_price_csv(path);
    ASSERT_EQ(t.tickers.size(), 2u);
    EXPECT_EQ(t.tickers[1], "BBB");
    ASSERT_EQ(t.dates.size(), 3u);
    EXPECT_EQ(t.dates[2], "2024-01-04");
    EXPECT_TRUE(std::isnan(t.prices(1, 0)));
    EXPECT_DOUBLE_EQ(t.prices(2, 1), 22.5);
}

TEST(CsvLoader, RejectsMalformedInput){
    const char* bodies[] = {
        "Date,AAA\n2024-01-02,1\n2024-01-02,2\n",     //duplicate date
        "Date,AAA\n2024-01-03,1\n2024-01-02,2\n",     //out of order
        "Date,AAA\n2024-01-02,1,5\n",                 //too many cells
        "Date,AAA\n2024-01-02,abc\n",                 //non-numeric
        "Date,AAA\n01/02/2024,1\n",                   //bad date format
        "Date,AAA,AAA\n2024-01-02,1,2\n",             //duplicate ticker
        "Date,AAA\n",                                 //no rows
    };
    int i = 0;
    for(const char* b : bodies){
        const auto path = write_tmp("qre_bad" + std::to_string(i++) + ".csv", b);
        EXPECT_THROW(load_price_csv(path), InvalidInputError) << b;
    }
    EXPECT_THROW(load_price_csv("/nonexistent/qre.csv"), InvalidInputError);
}

//------------------------------------------------------------------ corporate actions
TEST(CorporateActions, SplitBackAdjustmentRemovesJump){
    const PriceMatrix adj = back_adjust(col({100.0, 102.0, 51.5, 52.0}), {{0, 2, ActionType::Split, 2.0}});
    EXPECT_NEAR(adj(0, 0), 50.0, 1e-12);
    EXPECT_NEAR(adj(1, 0), 51.0, 1e-12);
    EXPECT_NEAR(adj(2, 0), 51.5, 1e-12);
    EXPECT_NEAR(adj(3, 0), 52.0, 1e-12);
}

TEST(CorporateActions, ReverseSplit){
    const PriceMatrix adj = back_adjust(col({10.0, 10.2, 20.6}), {{0, 2, ActionType::Split, 0.5}});
    EXPECT_NEAR(adj(0, 0), 20.0, 1e-12);
    EXPECT_NEAR(adj(1, 0), 20.4, 1e-12);
}

TEST(CorporateActions, DividendGivesTotalReturn){
    const PriceMatrix adj = back_adjust(col({100.0, 101.0, 99.0, 100.0}), {{0, 2, ActionType::CashDividend, 2.0}});
    const double f = 1.0 - 2.0 / 101.0;
    EXPECT_NEAR(adj(0, 0), 100.0 * f, 1e-12);
    EXPECT_NEAR(adj(1, 0), 101.0 * f, 1e-12);
    EXPECT_NEAR(adj(2, 0), 99.0, 1e-12);
    EXPECT_NEAR(adj(2, 0) / adj(1, 0) - 1.0, (99.0 + 2.0) / 101.0 - 1.0, 1e-12);
}

TEST(CorporateActions, MultipleActionsCompoundPerAsset){
    PriceMatrix raw(4, 2);
    raw << 100, 50, 100, 50, 50, 50, 25, 50;
    const PriceMatrix adj = back_adjust(raw, {{0, 2, ActionType::Split, 2.0}, {0, 3, ActionType::Split, 2.0}});
    for(Eigen::Index r = 0; r < 4; ++r){
        EXPECT_NEAR(adj(r, 0), 25.0, 1e-12);
        EXPECT_NEAR(adj(r, 1), 50.0, 1e-12);
    }
}

TEST(CorporateActions, InvalidActionsThrow){
    const PriceMatrix p = PriceMatrix::Constant(4, 1, 100.0);
    EXPECT_THROW(back_adjust(p, {{5, 2, ActionType::Split, 2.0}}), InvalidInputError);
    EXPECT_THROW(back_adjust(p, {{0, 0, ActionType::Split, 2.0}}), InvalidInputError);
    EXPECT_THROW(back_adjust(p, {{0, 2, ActionType::Split, 0.0}}), InvalidInputError);
    EXPECT_THROW(back_adjust(p, {{0, 2, ActionType::CashDividend, 500.0}}), InvalidInputError);
}

TEST(CorporateActions, DetectsSplitsNotNormalMoves){
    PriceMatrix p = fixture();
    p.block(30, 1, 30, 1) /= 4.0;
    p.block(45, 2, 15, 1) *= 2.0;
    const auto c = detect_split_candidates(p);
    ASSERT_EQ(c.size(), 2u);
    EXPECT_EQ(c[0].asset, 1); EXPECT_EQ(c[0].row, 30); EXPECT_DOUBLE_EQ(c[0].ratio, 4.0);
    EXPECT_EQ(c[1].asset, 2); EXPECT_EQ(c[1].row, 45); EXPECT_DOUBLE_EQ(c[1].ratio, 0.5);
    EXPECT_TRUE(detect_split_candidates(fixture()).empty());
}

TEST(CorporateActions, CrashThatLooksLikeSplitIsOnlyFlagged){
    PriceMatrix p = fixture();
    p.block(20, 0, 40, 1) *= 0.5;
    const auto res = harden(p);
    ASSERT_EQ(res.report.split_candidates.size(), 1u);
    EXPECT_EQ(res.report.split_candidates[0].row, 20);
    EXPECT_DOUBLE_EQ(res.prices(20, 0), p(20, 0));
}

//------------------------------------------------------------------ quality
TEST(Quality, StaleRunsIncludingAtEnd){
    PriceMatrix p = fixture();
    p.block(10, 0, 8, 1).setConstant(p(9, 0));
    p.block(40, 1, 2, 1).setConstant(p(39, 1));
    p.block(54, 2, 6, 1).setConstant(p(53, 2));
    const auto runs = stale_runs(p, 5);
    ASSERT_EQ(runs.size(), 2u);
    EXPECT_EQ(runs[0].asset, 0); EXPECT_EQ(runs[0].start, 10); EXPECT_EQ(runs[0].length, 8);
    EXPECT_EQ(runs[1].asset, 2); EXPECT_EQ(runs[1].start, 54); EXPECT_EQ(runs[1].length, 6);
}

TEST(Quality, BadTicksDetectedAndRepaired){
    PriceMatrix p = fixture();
    p(25, 0) *= 1.4;
    p(33, 2) *= 0.7;
    const auto res = harden(p);
    ASSERT_EQ(res.report.bad_ticks.size(), 2u);
    EXPECT_EQ(res.report.bad_ticks[0].asset, 0); EXPECT_EQ(res.report.bad_ticks[0].row, 25);
    EXPECT_EQ(res.report.bad_ticks[1].asset, 2); EXPECT_EQ(res.report.bad_ticks[1].row, 33);
    EXPECT_DOUBLE_EQ(res.prices(25, 0), p(24, 0));
    EXPECT_DOUBLE_EQ(res.prices(33, 2), p(32, 2));
}

TEST(Quality, SplitIsNotABadTick){
    PriceMatrix p = fixture();
    p.block(30, 1, 30, 1) /= 2.0;
    EXPECT_TRUE(bad_ticks(p).empty());
}

TEST(Quality, ForwardFillRespectsListingAndDelisting){
    PriceMatrix p = fixture();
    p.block(0, 0, 5, 1).setConstant(NaN);
    p.block(50, 1, 10, 1).setConstant(NaN);
    p.block(20, 2, 3, 1).setConstant(NaN);
    p.block(30, 2, 10, 1).setConstant(NaN);
    const PriceMatrix out = forward_fill_within_life(p, 5);
    EXPECT_TRUE(out.block(0, 0, 5, 1).array().isNaN().all());
    EXPECT_TRUE(out.block(50, 1, 10, 1).array().isNaN().all());   //no fabricated prices after delisting
    for(Eigen::Index r = 20; r < 23; ++r) EXPECT_DOUBLE_EQ(out(r, 2), p(19, 2));
    EXPECT_TRUE(out.block(30, 2, 10, 1).array().isNaN().all());
}

TEST(Quality, CleanDataPassesUntouched){
    const PriceMatrix p = fixture();
    const auto res = harden(p);
    EXPECT_TRUE(res.prices.cwiseEqual(p).all());
    EXPECT_TRUE(res.report.clean());
}

//Same corruptions as the Python cross-check -> identical report
TEST(Quality, ReportMatchesPythonReference){
    PriceMatrix q = fixture();
    q.block(0, 0, 5, 1).setConstant(NaN);
    q.block(50, 1, 10, 1).setConstant(NaN);
    q.block(20, 2, 3, 1).setConstant(NaN);
    q.block(30, 2, 10, 1).setConstant(NaN);
    q(25, 0) *= 1.4;
    q.block(44, 0, 5, 1).setConstant(q(43, 0));
    q(12, 1) = 0.0;
    q.block(52, 0, 8, 1) /= 4.0;
    const auto res = harden(q);
    const auto& rep = res.report;
    ASSERT_EQ(rep.non_positive.size(), 1u);
    EXPECT_EQ(rep.non_positive[0], (std::pair<Eigen::Index, Eigen::Index>{1, 12}));
    EXPECT_EQ(rep.leading_missing, (std::vector<Eigen::Index>{5, 0, 0}));
    EXPECT_EQ(rep.trailing_missing, (std::vector<Eigen::Index>{0, 10, 0}));
    EXPECT_EQ(rep.internal_missing, (std::vector<Eigen::Index>{1, 1, 13}));
    EXPECT_EQ(rep.remaining_internal_missing, (std::vector<Eigen::Index>{0, 0, 10}));
    ASSERT_EQ(rep.stale.size(), 1u);
    EXPECT_EQ(rep.stale[0].asset, 0); EXPECT_EQ(rep.stale[0].start, 44); EXPECT_EQ(rep.stale[0].length, 5);
    ASSERT_EQ(rep.bad_ticks.size(), 1u);
    EXPECT_EQ(rep.bad_ticks[0].asset, 0); EXPECT_EQ(rep.bad_ticks[0].row, 25);
    ASSERT_EQ(rep.split_candidates.size(), 1u);
    EXPECT_EQ(rep.split_candidates[0].asset, 0); EXPECT_EQ(rep.split_candidates[0].row, 52);
    EXPECT_DOUBLE_EQ(rep.split_candidates[0].ratio, 4.0);
    EXPECT_NEAR(res.prices(25, 0), 100.76541542489694, 1e-12);
}

//------------------------------------------------------------------ point in time
TEST(PointInTime, WindowUsesOnlyPastRows){
    const PriceMatrix p = fixture();
    const PriceMatrix w = pit_window(p, 30, 10);
    EXPECT_EQ(w.rows(), 11);
    EXPECT_TRUE(w.cwiseEqual(p.middleRows(19, 11)).all());
    EXPECT_EQ(pit_window(p, 60, 10).rows(), 11);
    EXPECT_THROW(pit_window(p, 5, 10), InvalidInputError);
    EXPECT_THROW(pit_window(p, 61, 5), InvalidInputError);
    EXPECT_THROW(pit_window(p, 30, 0), InvalidInputError);
}

TEST(PointInTime, UniverseExcludesAssetsNotAliveInWindow){
    PriceMatrix p = fixture();
    p.block(0, 0, 25, 1).setConstant(NaN);
    p.block(40, 1, 20, 1).setConstant(NaN);
    EXPECT_EQ(universe_at(p, 20, 10), (std::vector<Eigen::Index>{1, 2}));
    EXPECT_EQ(universe_at(p, 37, 10), (std::vector<Eigen::Index>{0, 1, 2}));
    EXPECT_EQ(universe_at(p, 55, 10), (std::vector<Eigen::Index>{0, 2}));
}

TEST(PointInTime, FullAdjustmentReturnsArePointInTimeButLevelsAreNot){
    PriceMatrix raw = fixture();
    raw.block(40, 0, 20, 1) /= 2.0;     //future split
    raw(20, 1) -= 1.0;                  //past dividend
    const std::vector<CorporateAction> acts = {{0, 40, ActionType::Split, 2.0}, {1, 20, ActionType::CashDividend, 1.0}};
    const PriceMatrix full = back_adjust(raw, acts);
    const Eigen::Index t = 30;
    const PriceMatrix pit = adjust_as_of(raw, acts, t);
    for(Eigen::Index c = 0; c < 3; ++c)
        for(Eigen::Index r = 1; r < t; ++r)
            EXPECT_NEAR(full(r, c) / full(r - 1, c), pit(r, c) / pit(r - 1, c), 1e-12);
    EXPECT_GT(std::abs(full(0, 0) - pit(0, 0)), 1.0);
}
