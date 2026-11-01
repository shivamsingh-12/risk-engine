#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include "qre/linalg/cholesky.hpp"
#include "qre/risk/backtest.hpp"
using namespace qre;
using namespace qre::risk;

namespace {
//Expected values below were computed independently with scipy (see python_reference/tests/test_backtest.py)
CovarianceMatrix test_cov(){
    CovarianceMatrix c(3, 3);
    c << 0.04, 0.012, 0.0,
         0.012, 0.09, -0.01,
         0.0, -0.01, 0.0225;
    return c / 252.0;
}
WeightVector test_w(){ WeightVector w(3); w << 0.5, 0.3, 0.2; return w; }

PriceMatrix gaussian_prices(Eigen::Index n, uint64_t seed){
    Rng rng(seed);
    const Eigen::MatrixXd L = linalg::cholesky_decompose(test_cov());
    const auto draws = distributions::normal::sample(static_cast<std::size_t>(3 * n), 0.0, 1.0, rng);
    PriceMatrix p(n + 1, 3);
    p.row(0).setConstant(100.0);
    for(Eigen::Index t = 0; t < n; ++t){
        Eigen::Vector3d z;
        for(Eigen::Index i = 0; i < 3; ++i) z(i) = draws[static_cast<std::size_t>(3 * t + i)];
        const Eigen::Vector3d r = L * z;
        p.row(t + 1) = p.row(t).array() * (1.0 + r.transpose().array());
    }
    return p;
}

std::vector<bool> seq(std::initializer_list<int> v){ return std::vector<bool>(v.begin(), v.end()); }
}

TEST(Chi2, ClosedFormsMatchScipy){
    //chi2.sf(3.841458820694124, 1) = 0.05, chi2.sf(x, 2) = exp(-x/2)
    EXPECT_NEAR(detail::chi2_sf_df1(3.841458820694124), 0.05, 1e-12);
    EXPECT_NEAR(detail::chi2_sf_df1(0.0), 1.0, 1e-15);
    EXPECT_NEAR(detail::chi2_sf_df2(5.991464547107979), 0.05, 1e-12);
}

TEST(Kupiec, MatchesScipy){
    struct Case{ std::size_t n, x; double lr, p; };
    const Case cases[] = {
        {250, 0, 5.025167926750726, 0.02498150305344971},
        {250, 2, 0.10843521623679919, 0.7419327009526282},
        {250, 7, 5.496990447792683, 0.019049230890526528},
        {500, 5, 0.0, 1.0},
        {500, 13, 8.973292711251432, 0.0027395444626702914},
        {10, 10, 92.10340371976181, 8.226375843540734e-22},
    };
    for(const auto& c : cases){
        const TestResult r = kupiec_pof(c.n, c.x, 0.01);
        EXPECT_NEAR(r.lr, c.lr, 1e-10) << c.n << "," << c.x;
        EXPECT_NEAR(r.p_value, c.p, 1e-12 + 1e-9 * c.p) << c.n << "," << c.x;
    }
    const TestResult r975 = kupiec_pof(502, 19, 0.025);
    EXPECT_NEAR(r975.lr, 2.9446702353674823, 1e-10);
    EXPECT_NEAR(r975.p_value, 0.0861612832577268, 1e-12);
}

TEST(Kupiec, InvalidInputsThrow){
    EXPECT_THROW(kupiec_pof(0, 0, 0.01), InvalidInputError);
    EXPECT_THROW(kupiec_pof(10, 11, 0.01), InvalidInputError);
    EXPECT_THROW(kupiec_pof(10, 1, 0.0), InvalidInputError);
    EXPECT_THROW(kupiec_pof(10, 1, 1.0), InvalidInputError);
}

TEST(Christoffersen, MatchesScipyReference){
    const TestResult r = christoffersen_independence(seq({0,0,1,1,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0}));
    EXPECT_NEAR(r.lr, 1.3358104147583951, 1e-12);
    EXPECT_NEAR(r.p_value, 0.2477741635278944, 1e-12);
}

TEST(Christoffersen, ClusteredRejectedSpreadOutNot){
    std::vector<bool> clustered(500, false);
    for(int i = 100; i < 105; ++i) clustered[static_cast<std::size_t>(i)] = true;
    EXPECT_LT(christoffersen_independence(clustered).p_value, 0.001);
    std::vector<bool> spread(500, false);
    for(int i : {50, 150, 250, 350, 450}) spread[static_cast<std::size_t>(i)] = true;
    EXPECT_GT(christoffersen_independence(spread).p_value, 0.5);
    const TestResult none = christoffersen_independence(std::vector<bool>(100, false));
    EXPECT_EQ(none.lr, 0.0);
    EXPECT_EQ(none.p_value, 1.0);
}

TEST(ConditionalCoverage, MatchesScipyReference){
    std::vector<bool> b(300, false);
    for(int i : {10, 11, 80, 200, 201, 202}) b[static_cast<std::size_t>(i)] = true;
    const TestResult r = conditional_coverage(b, 0.01);
    EXPECT_NEAR(r.lr, 19.354860128758418, 1e-10);
    EXPECT_NEAR(r.p_value, 6.26823866552123e-05, 1e-14);
}

TEST(TrafficLight, ReproducesBaselTable){
    for(std::size_t x = 0; x <= 4; ++x) EXPECT_EQ(traffic_light(x, 250, 0.01), TrafficLight::Green) << x;
    for(std::size_t x = 5; x <= 9; ++x) EXPECT_EQ(traffic_light(x, 250, 0.01), TrafficLight::Yellow) << x;
    for(std::size_t x = 10; x <= 12; ++x) EXPECT_EQ(traffic_light(x, 250, 0.01), TrafficLight::Red) << x;
}

TEST(TrafficLight, MatchesScipyAt975With500Days){
    for(std::size_t x = 0; x < 30; ++x){
        const TrafficLight expect = x <= 17 ? TrafficLight::Green : (x <= 26 ? TrafficLight::Yellow : TrafficLight::Red);
        EXPECT_EQ(traffic_light(x, 500, 0.025), expect) << x;
    }
}

TEST(EsZ2, ZeroForCorrectModelPositiveWhenEsTooSmall){
    const double alpha = 0.025;
    const double z = risk::detail::normal_quantile(1.0 - alpha);
    const double es_true = distributions::normal::pdf(z, 0.0, 1.0) / alpha;
    Rng rng(1);
    const auto losses = distributions::normal::sample(400000, 0.0, 1.0, rng);
    const std::vector<double> var(losses.size(), z), es(losses.size(), es_true), es_low(losses.size(), 0.8 * es_true);
    EXPECT_LT(std::abs(es_z2(losses, var, es, alpha)), 0.03);
    EXPECT_NEAR(es_z2(losses, var, es_low, alpha), 0.25, 0.04);
    EXPECT_THROW(es_z2({1.0, 2.0}, {1.0}, {1.0, 1.0}, alpha), InvalidInputError);
    EXPECT_THROW(es_z2({1.0}, {0.5}, {0.0}, alpha), InvalidInputError);
}

TEST(Kupiec, SizeAndPower){
    std::mt19937_64 eng(3);
    const std::size_t n = 500, reps = 3000;
    auto reject_rate = [&](double p_true){
        std::binomial_distribution<std::size_t> dist(n, p_true);
        std::size_t rejected = 0;
        for(std::size_t i = 0; i < reps; ++i) if(kupiec_pof(n, dist(eng), 0.01).p_value < 0.05) ++rejected;
        return static_cast<double>(rejected) / static_cast<double>(reps);
    };
    const double size = reject_rate(0.01), power = reject_rate(0.02);
    EXPECT_GT(size, 0.02);
    EXPECT_LT(size, 0.09);
    EXPECT_GT(power, 3.0 * size);
}

TEST(RollingBacktest, ShapesAndRealizedLosses){
    const PriceMatrix p = gaussian_prices(400, 0);
    for(auto m : {BacktestMethod::Historical, BacktestMethod::Parametric, BacktestMethod::MonteCarlo}){
        const BacktestSeries s = rolling_backtest(p, test_w(), 100, 0.99, m, 4000);
        ASSERT_EQ(s.rows.size(), static_cast<std::size_t>(p.rows() - 101));
        EXPECT_EQ(s.rows.front(), 101);
        EXPECT_EQ(s.rows.back(), p.rows() - 1);
        const Eigen::Index t = s.rows[10];
        double expected = 0.0;
        for(Eigen::Index i = 0; i < 3; ++i) expected -= test_w()(i) * (p(t, i) / p(t - 1, i) - 1.0);
        EXPECT_NEAR(s.loss[10], expected, 1e-15);
        for(std::size_t i = 0; i < s.var.size(); ++i) EXPECT_GE(s.es[i], s.var[i]);
    }
}

TEST(RollingBacktest, NoLookAhead){
    const PriceMatrix p = gaussian_prices(400, 0);
    PriceMatrix future = p;
    for(Eigen::Index r = 201; r < future.rows(); ++r) future.row(r) *= 0.5 + 2.5 * static_cast<double>(r - 201) / static_cast<double>(future.rows() - 201);
    for(auto m : {BacktestMethod::Historical, BacktestMethod::Parametric, BacktestMethod::MonteCarlo}){
        const BacktestSeries a = rolling_backtest(p, test_w(), 100, 0.99, m, 4000);
        const BacktestSeries b = rolling_backtest(future, test_w(), 100, 0.99, m, 4000);
        const std::size_t i = static_cast<std::size_t>(201 - 101);
        for(std::size_t k = 0; k <= i; ++k){
            EXPECT_EQ(a.var[k], b.var[k]);
            EXPECT_EQ(a.es[k], b.es[k]);
        }
        EXPECT_NE(a.loss[i], b.loss[i]);
    }
}

TEST(RollingBacktest, ParametricOnGaussianDataPasses){
    const PriceMatrix p = gaussian_prices(1500, 4);
    const BacktestSummary s = summarize(rolling_backtest(p, test_w(), 250, 0.99, BacktestMethod::Parametric), 0.99);
    EXPECT_EQ(s.n_obs, 1250u);
    EXPECT_GT(s.kupiec.p_value, 0.01);
    EXPECT_NE(s.traffic_light, TrafficLight::Red);
}

TEST(RollingBacktest, RejectsBadInput){
    const PriceMatrix p = gaussian_prices(50, 1);
    EXPECT_THROW(rolling_backtest(p, test_w(), 60), InvalidInputError);
    WeightVector two(2); two << 0.5, 0.5;
    EXPECT_THROW(rolling_backtest(p, two, 10), InvalidInputError);
    PriceMatrix bad = p;
    bad(20, 1) = std::nan("");
    EXPECT_THROW(rolling_backtest(bad, test_w(), 10), InvalidInputError);
}
