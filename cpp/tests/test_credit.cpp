#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <vector>
#include "qre/credit/merton_credit.hpp"
#include "qre/pricing/black_scholes.hpp"
using namespace qre;
using namespace qre::credit;

//Expected values from the Python reference (qre/credit/merton.py)
TEST(MertonCredit, ClosedFormsMatchPythonAndBlackScholes){
    const double V = 100, D = 70, r = 0.03, s = 0.25, T = 2;
    EXPECT_NEAR(equity_value(V, D, r, s, T), 35.74165846823506, 1e-12);
    EXPECT_NEAR(equity_value(V, D, r, s, T), pricing::price({V, D, r, s, T, pricing::OptionType::Call}), 1e-12);
    EXPECT_NEAR(debt_value(V, D, r, s, T), 64.25834153176496, 1e-12);
    EXPECT_NEAR(debt_value(V, D, r, s, T), D * std::exp(-r * T) - pricing::price({V, D, r, s, T, pricing::OptionType::Put}), 1e-11);
    EXPECT_NEAR(credit_spread(V, D, r, s, T), 0.012791848706475985, 1e-15);
    EXPECT_NEAR(risk_neutral_pd(V, D, r, s, T), 0.15823023888019555, 1e-15);
    EXPECT_NEAR(recovery_rate(V, D, r, s, T), 0.8403642191917199, 1e-14);
    EXPECT_NEAR(equity_volatility(V, D, r, s, T), 0.6381494720086952, 1e-14);
    EXPECT_NEAR(distance_to_default(V, D, 0.08, s, T), 1.2846007308163907, 1e-14);
    EXPECT_NEAR(physical_pd(V, D, 0.08, s, T), 0.09946592158811882, 1e-15);
    EXPECT_THROW(equity_value(V, D, r, 0.0, T), InvalidInputError);
}

TEST(MertonCredit, SolveAssetsRoundTripAndPython){
    const AssetSolution a = solve_assets(40.0, 0.55, 80.0, 0.02, 1.0);
    EXPECT_NEAR(a.V, 118.32457914919381, 1e-8);
    EXPECT_NEAR(a.sigma, 0.1880447155462024, 1e-10);
    for(const auto& c : std::vector<std::array<double, 4>>{{100, 70, 0.25, 2}, {100, 30, 0.15, 1}, {100, 95, 0.4, 0.5}, {50, 80, 0.3, 1}}){
        const double e = equity_value(c[0], c[1], 0.03, c[2], c[3]), se = equity_volatility(c[0], c[1], 0.03, c[2], c[3]);
        const AssetSolution s = solve_assets(e, se, c[1], 0.03, c[3]);
        EXPECT_NEAR(s.V / c[0], 1.0, 1e-10);
        EXPECT_NEAR(s.sigma / c[2], 1.0, 1e-9);
    }
}

TEST(MertonCredit, KmvMatchesPython){
    std::vector<double> E;
    for(int t = 0; t < 120; ++t){
        const double v = 100.0 * std::exp(0.0004 * t + 0.02 * std::sin(0.3 * t) + 0.01 * std::cos(1.7 * t));
        E.push_back(equity_value(v, 60.0, 0.03, 0.25, 1.0));
    }
    const KmvResult k = kmv_iterative(E, 60.0, 0.03);
    EXPECT_NEAR(k.sigma, 0.17968952152775017, 1e-9);
    EXPECT_NEAR(k.mu, 0.06372750159149652, 1e-9);
    EXPECT_EQ(k.iterations, 5);
    EXPECT_NEAR(k.distance_to_default, 3.293187051198701, 1e-8);
    EXPECT_NEAR(k.assets[50], 102.41989133844258, 1e-8);
    EXPECT_THROW(kmv_iterative(std::vector<double>(5, 1.0), 60.0, 0.03), InvalidInputError);
}

TEST(MertonCredit, BlackCoxMatchesPythonAndMonteCarlo){
    EXPECT_NEAR(black_cox_pd(100, 70, 0.08, 0.25, 2), 0.23232603114124778, 1e-14);
    EXPECT_NEAR(black_cox_pd(100, 90, -0.02, 0.3, 1), 0.7784099909098858, 1e-14);
    const double mc = first_passage_mc(100, 70, 0.08, 0.25, 2, 100000, 504, 7);
    EXPECT_NEAR(mc, 0.23232603114124778, 4.0 * std::sqrt(0.23 * 0.77 / 100000) + 1e-3);
    EXPECT_GE(black_cox_pd(100, 70, 0.08, 0.25, 2), physical_pd(100, 70, 0.08, 0.25, 2));
    EXPECT_THROW(black_cox_pd(100, 120, 0.05, 0.2, 1), InvalidInputError);
}
