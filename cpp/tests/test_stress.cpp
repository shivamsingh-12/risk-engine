#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include "qre/risk/stress.hpp"

using namespace qre;
using namespace qre::risk;

namespace{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

Eigen::MatrixXd bivariate(double s1, double s2, double rho){
    Eigen::MatrixXd c(2, 2);
    c << s1 * s1, rho * s1 * s2, rho * s1 * s2, s2 * s2;
    return c;
}

Eigen::MatrixXd three_asset_cov(){
    Eigen::Vector3d vol(0.012, 0.018, 0.006);
    Eigen::Matrix3d corr;
    corr << 1.0, 0.7, -0.3, 0.7, 1.0, -0.2, -0.3, -0.2, 1.0;
    return corr.cwiseProduct(vol * vol.transpose());
}
}

TEST(Stress, WindowIndicesUseLastTradingDayOnOrBefore){
    const std::vector<std::string> d{"2020-03-05", "2020-03-06", "2020-03-09", "2020-03-10", "2020-03-11"};
    EXPECT_EQ(window_indices(d, "2020-03-07", "2020-03-10"), std::make_pair(Eigen::Index{1}, Eigen::Index{3}));
    EXPECT_EQ(window_indices(d, "2020-03-05", "2020-12-31"), std::make_pair(Eigen::Index{0}, Eigen::Index{4}));
    EXPECT_THROW(window_indices(d, "2020-03-01", "2020-03-10"), InvalidInputError);
    EXPECT_THROW(window_indices(d, "2020-03-09", "2020-03-09"), InvalidInputError);
    EXPECT_THROW(window_indices(d, "2020-03-07", "2020-03-08"), InvalidInputError);
}

TEST(Stress, ScenarioTableMatchesPython){
    EXPECT_EQ(std::size(kScenarios), 10u);
    EXPECT_EQ(find_scenario("covid").start, "2020-02-19");
    EXPECT_EQ(find_scenario("gfc").end, "2009-03-09");
    for(const auto& s : kScenarios) EXPECT_LT(s.start, s.end);
    EXPECT_THROW(find_scenario("nope"), InvalidInputError);
}

TEST(Stress, BetaRecoversSlopeAndSkipsMissingDays){
    std::mt19937_64 g(1);
    std::normal_distribution<double> m(0.0, 0.01), e(0.0, 0.002);
    std::vector<double> mk(2000), a(2000);
    for(std::size_t i = 0; i < mk.size(); ++i){ mk[i] = m(g); a[i] = i < 500 ? kNaN : 1.3 * mk[i] + e(g); }
    EXPECT_NEAR(beta(a, mk), 1.3, 0.01);
    EXPECT_THROW(beta(std::vector<double>(a.begin(), a.begin() + 510), std::vector<double>(mk.begin(), mk.begin() + 510)),
                 InvalidInputError);
}

TEST(Stress, ScenarioReturnsFillFromProxy){
    Eigen::Vector4d proxy(100.0, 98.0, 99.0, 90.0);
    Eigen::MatrixXd prices(4, 2);
    prices << 50.0, kNaN, 51.0, kNaN, 49.0, 20.0, 48.0, 19.0;
    const ScenarioReturns r = scenario_returns(prices, proxy, Eigen::Vector2d(kNaN, 1.5));
    EXPECT_NEAR(r.returns(0, 0), 0.02, 1e-15);
    EXPECT_NEAR(r.returns(0, 1), 1.5 * (98.0 / 100.0 - 1.0), 1e-15);
    EXPECT_NEAR(r.returns(1, 1), 1.5 * (99.0 / 98.0 - 1.0), 1e-15);
    EXPECT_NEAR(r.returns(2, 1), 19.0 / 20.0 - 1.0, 1e-15);
    EXPECT_FALSE(r.proxied[0]);
    EXPECT_TRUE(r.proxied[1]);
    EXPECT_EQ(r.proxied_days[1], 2);
    EXPECT_THROW(scenario_returns(prices, proxy, Eigen::Vector2d(kNaN, kNaN)), InvalidInputError);
    EXPECT_THROW(scenario_returns(prices, Eigen::Vector4d(100.0, kNaN, 99.0, 90.0), Eigen::Vector2d(1.0, 1.0)), InvalidInputError);
}

TEST(Stress, ReplayBuyAndHoldByHand){
    Eigen::MatrixXd r(3, 2);
    r << 0.10, -0.20, -0.50, 0.10, 0.20, 0.0;
    const Replay rp = replay(r, Eigen::Vector2d(0.6, 0.4));
    const double v[] = {1.0, 0.6 * 1.1 + 0.4 * 0.8, 0.6 * 0.55 + 0.4 * 0.88, 0.6 * 0.66 + 0.4 * 0.88};
    for(int i = 0; i < 4; ++i) EXPECT_NEAR(rp.value(i), v[i], 1e-15);
    EXPECT_NEAR(rp.total_return, v[3] - 1.0, 1e-15);
    EXPECT_NEAR(rp.contributions.sum(), rp.total_return, 1e-15);
    EXPECT_NEAR(rp.worst_day, v[2] / v[1] - 1.0, 1e-15);
    EXPECT_EQ(rp.worst_day_index, 2);
    EXPECT_NEAR(rp.worst_10d, v[3] - 1.0, 1e-15);
    EXPECT_NEAR(rp.max_drawdown, 1.0 - v[2], 1e-15);
}

TEST(Stress, ReplayCashAndShorts){
    Eigen::MatrixXd r(2, 1);
    r << -0.10, -0.10;
    EXPECT_NEAR(replay(r, Eigen::VectorXd::Constant(1, 0.5)).total_return, 0.5 * (0.81 - 1.0), 1e-15);
    const Replay s = replay(r, Eigen::VectorXd::Constant(1, -1.0));
    EXPECT_NEAR(s.total_return, 0.19, 1e-15);
    EXPECT_EQ(s.max_drawdown, 0.0);
    EXPECT_THROW(replay(Eigen::MatrixXd::Constant(1, 1, -1.0), Eigen::VectorXd::Ones(1)), InvalidInputError);
}

TEST(Stress, ConditionalShockMatchesBivariateFormula){
    const Eigen::MatrixXd cov = bivariate(0.01, 0.02, 0.6);
    const Shock sh = conditional_shock(cov, {0}, {-0.2}, 10.0);
    EXPECT_EQ(sh.mean(0), -0.2);
    EXPECT_NEAR(sh.mean(1), 0.6 * 0.02 / 0.01 * -0.2, 1e-15);
    EXPECT_NEAR(sh.cond_cov(1, 1), 0.02 * 0.02 * (1.0 - 0.36) * 10.0, 1e-17);
    EXPECT_EQ(sh.cond_cov(0, 0), 0.0);
    EXPECT_EQ(sh.cond_cov(0, 1), 0.0);
    EXPECT_NEAR(conditional_shock(cov * 25.0, {0}, {-0.2}).mean(1), sh.mean(1), 1e-15);
}

TEST(Stress, ConditionalShockCapsAndValidates){
    const Eigen::MatrixXd cov = bivariate(0.01, 0.05, 0.9);
    EXPECT_EQ(conditional_shock(cov, {0}, {-0.5}).mean(1), -1.0);
    EXPECT_THROW(conditional_shock(Eigen::MatrixXd::Constant(2, 2, 1e-4), {0, 1}, {-0.1, -0.1}), InvalidInputError);
    EXPECT_THROW(conditional_shock(cov, {0, 0}, {-0.1, -0.1}), InvalidInputError);
    EXPECT_THROW(conditional_shock(cov, {2}, {-0.1}), InvalidInputError);
    EXPECT_THROW(conditional_shock(cov, {0}, {-1.0}), InvalidInputError);
    Eigen::MatrixXd asym(2, 2);
    asym << 1.0, 0.5, 0.4, 1.0;
    EXPECT_THROW(conditional_shock(asym, {0}, {-0.1}), InvalidInputError);
}

TEST(Stress, ShockPnlAndReverseStressAgree){
    const Eigen::MatrixXd cov = three_asset_cov();
    const Eigen::Vector3d w(0.5, 0.3, 0.2);
    const auto s = reverse_stress(w, cov, 0, 0.15);
    ASSERT_TRUE(s.has_value());
    const ShockPnl p = shock_pnl(w, conditional_shock(cov, {0}, {*s}), 0.99);
    EXPECT_NEAR(p.expected_return, -0.15, 1e-14);
    EXPECT_NEAR(p.contributions.sum(), p.expected_return, 1e-15);
    EXPECT_NEAR(p.loss_at_confidence, 0.15 + 2.3263478740408408 * p.stdev, 1e-9);
    EXPECT_FALSE(reverse_stress(Eigen::Vector3d(0.0, 0.0, 1.0), cov, 0, 0.1).has_value());
}

//Same inputs as test_cross_language_fixture in python_reference/tests/test_stress.py
TEST(Stress, MatchesPythonReference){
    const Eigen::MatrixXd cov = three_asset_cov();
    const Eigen::Vector3d w(0.5, 0.3, 0.2);
    const Shock sh = conditional_shock(cov, {0, 2}, {-0.12, 0.03}, 21.0);
    const ShockPnl p = shock_pnl(w, sh, 0.975);
    Eigen::MatrixXd r(4, 3);
    r << 0.01, -0.02, 0.003, -0.04, -0.06, 0.01, 0.02, 0.03, -0.002, -0.01, 0.005, 0.0;
    const Replay rp = replay(r, Eigen::Vector3d(0.6, -0.2, 0.5));
    EXPECT_NEAR(sh.mean(1), -0.1256043956043956, 1e-15);
    EXPECT_NEAR(sh.cond_cov(1, 1), 0.003469292307692308, 1e-16);
    EXPECT_NEAR(p.loss_at_confidence, 0.12631429311790987, 1e-10);     //C++ normal quantile is a Brent root
    EXPECT_NEAR(rp.total_return, 0.0022471820000000697, 1e-15);
    EXPECT_NEAR(rp.max_drawdown, 0.009147620365793419, 1e-15);
    EXPECT_NEAR(rp.worst_day, -0.007380128521997098, 1e-15);
    EXPECT_NEAR(*reverse_stress(w, cov, 0, 0.2), -0.25477707006369427, 1e-15);
}
