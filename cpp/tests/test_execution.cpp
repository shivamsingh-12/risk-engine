#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include "qre/execution/almgren_chriss.hpp"
using namespace qre;
using namespace qre::execution;

//Expected values from the Python reference (qre/execution/almgren_chriss.py)
TEST(AlmgrenChriss, PaperExampleMatchesPython){
    const AcParams p = paper_example();
    EXPECT_NEAR(kappa(p, 1e-6), 0.6070761632470627, 1e-14);
    const auto x = optimal_holdings(p, 1e-6);
    const double expected[6] = {1e6, 541955.5543739223, 289854.2194099351, 147897.48782172316, 62141.801605766035, 0.0};
    for(std::size_t j = 0; j < 6; ++j) EXPECT_NEAR(x[j], expected[j], 1e-8);
    EXPECT_NEAR(expected_cost(p, x), 911226.9863037932, 1e-6);
    EXPECT_NEAR(cost_variance(p, x) / 364128572058.141, 1.0, 1e-13);
    const AcParams q{5e5, 10.0, 20, 0.8, 3e-6, 2e-7, 0.05, 50.0};
    const auto y = optimal_holdings(q, 2e-6);
    EXPECT_NEAR(y[7], 50361.31132064671, 1e-8);
    EXPECT_NEAR(expected_cost(q, y), 289684.97219453566, 1e-6);
    EXPECT_NEAR(cost_variance(q, y) / 86329178007.29987, 1.0, 1e-13);
}

TEST(AlmgrenChriss, ClosedFormEqualsTridiagonalSolve){
    const AcParams p = paper_example();
    for(double lam : {0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4}){
        const auto x = optimal_holdings(p, lam), xn = solve_numerically(p, lam);
        for(std::size_t j = 0; j < x.size(); ++j) EXPECT_NEAR(x[j], xn[j], 1e-6 + 1e-10 * p.X);
    }
    const AcParams q{5e5, 10.0, 20, 0.8, 3e-6, 2e-7, 0.05, 50.0};
    const auto x = optimal_holdings(q, 2e-6), xn = solve_numerically(q, 2e-6);
    for(std::size_t j = 0; j < x.size(); ++j) EXPECT_NEAR(x[j], xn[j], 1e-6);
}

TEST(AlmgrenChriss, TwapLimitsAndFrontier){
    const AcParams p = paper_example();
    const auto tw = twap_holdings(p), x0 = optimal_holdings(p, 0.0);
    for(std::size_t j = 0; j < tw.size(); ++j) EXPECT_NEAR(tw[j], x0[j], 1e-6);
    EXPECT_NEAR(expected_cost(p, tw), 0.5 * p.gamma * p.X * p.X + p.eps * p.X + p.eta_tilde() * p.X * p.X / p.T, 1e-6);
    const auto huge = optimal_holdings(p, 1e3);
    EXPECT_LT(huge[1], 1e-3 * p.X);
    std::vector<double> lams;
    for(int i = 0; i < 20; ++i) lams.push_back(std::pow(10.0, -8.0 + 4.0 * i / 19.0));
    const auto f = efficient_frontier(p, lams);
    for(std::size_t i = 1; i < f.size(); ++i){
        EXPECT_GT(f[i].expected, f[i - 1].expected);
        EXPECT_LT(f[i].sd, f[i - 1].sd);
    }
    EXPECT_THROW(kappa(p, -1.0), InvalidInputError);
    EXPECT_THROW((AcParams{1e6, 1.0, 1, 0.5, 1e-7, 1e-6}.validate()), InvalidInputError);
}

TEST(AlmgrenChriss, MonteCarloMatchesFormulas){
    const AcParams p = paper_example();
    for(double lam : {0.0, 1e-6}){
        const auto x = optimal_holdings(p, lam);
        const auto L = simulate_shortfall(p, x, 200000, 3);
        double m = 0.0, s = 0.0;
        for(double v : L) m += v;
        m /= static_cast<double>(L.size());
        for(double v : L) s += (v - m) * (v - m);
        const double sd = std::sqrt(s / static_cast<double>(L.size() - 1)), sd_true = std::sqrt(cost_variance(p, x));
        EXPECT_NEAR(m, expected_cost(p, x), 4.0 * sd_true / std::sqrt(200000.0));
        EXPECT_NEAR(sd / sd_true, 1.0, 0.01);
    }
}
