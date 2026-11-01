#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "qre/linalg/cholesky.hpp"
#include "qre/risk/backtest.hpp"
#include "qre/risk/ewma.hpp"
using namespace qre;
using namespace qre::risk;

namespace {
//Deterministic return window shared with the Python cross-check (python_reference/qre/risk/ewma.py)
ReturnMatrix fixture_returns(){
    ReturnMatrix r(300, 3);
    for(Eigen::Index i = 0; i < 300; ++i)
        for(Eigen::Index c = 0; c < 3; ++c){
            const double di = static_cast<double>(i), dc = static_cast<double>(c);
            r(i, c) = 0.01 * std::sin(0.7 * di + 1.1 * dc) + 0.004 * std::cos(1.9 * di + 0.5 * dc)
                      + ((i % 97 == 5 && c == 0) ? 0.02 : 0.0);
        }
    return r;
}
WeightVector test_w(){ WeightVector w(3); w << 0.5, 0.3, 0.2; return w; }

CovarianceMatrix test_cov(){
    CovarianceMatrix c(3, 3);
    c << 0.04, 0.012, 0.0,
         0.012, 0.09, -0.01,
         0.0, -0.01, 0.0225;
    return c / 252.0;
}

//Gaussian prices whose volatility is multiplied by `jump` from row `from` on
PriceMatrix gaussian_prices(Eigen::Index n, uint64_t seed, Eigen::Index from = -1, double jump = 1.0){
    Rng rng(seed);
    const Eigen::MatrixXd L = linalg::cholesky_decompose(test_cov());
    const auto draws = distributions::normal::sample(static_cast<std::size_t>(3 * n), 0.0, 1.0, rng);
    PriceMatrix p(n + 1, 3);
    p.row(0).setConstant(100.0);
    for(Eigen::Index t = 0; t < n; ++t){
        Eigen::Vector3d z;
        for(Eigen::Index i = 0; i < 3; ++i) z(i) = draws[static_cast<std::size_t>(3 * t + i)];
        const double scale = (from >= 0 && t >= from) ? jump : 1.0;
        const Eigen::Vector3d r = scale * (L * z);
        p.row(t + 1) = p.row(t).array() * (1.0 + r.transpose().array());
    }
    return p;
}
}

TEST(EwmaWeights, SumToOneAndFavorRecent){
    const Eigen::VectorXd w = ewma_weights(250, 0.94);
    EXPECT_NEAR(w.sum(), 1.0, 1e-14);
    for(Eigen::Index i = 1; i < w.size(); ++i) EXPECT_GT(w(i), w(i - 1));
    EXPECT_NEAR(w(249) / w(248), 1.0 / 0.94, 1e-12);
    const Eigen::VectorXd k = ewma_weights(3, 0.5);
    EXPECT_NEAR(k(0), 0.125 / 0.875, 1e-15);
    EXPECT_NEAR(k(1), 0.25 / 0.875, 1e-15);
    EXPECT_NEAR(k(2), 0.5 / 0.875, 1e-15);
}

TEST(EwmaCovariance, MatchesPython){
    const CovarianceMatrix c = ewma_covariance(fixture_returns(), 0.94);
    const double expected[9] = {6.917705639144827e-05, 3.563460996914534e-05, -1.972314436532604e-05,
                                3.563460996914534e-05, 5.816771103566055e-05, 2.9406691917634688e-05,
                                -1.972314436532604e-05, 2.9406691917634685e-05, 6.057733948115742e-05};
    for(Eigen::Index i = 0; i < 3; ++i)
        for(Eigen::Index j = 0; j < 3; ++j)
            EXPECT_NEAR(c(i, j), expected[i * 3 + j], 1e-18);
}

TEST(EwmaVolPath, RecursionAndPythonValues){
    const std::vector<double> r = {0.01, -0.02, 0.015, 0.0, -0.005};
    const auto s = ewma_vol_path(r, 0.9);
    ASSERT_EQ(s.size(), 6u);
    double s2 = 0.0;
    for(double x : r) s2 += x * x;
    s2 /= 5.0;
    EXPECT_NEAR(s[0], std::sqrt(s2), 1e-16);
    for(std::size_t i = 0; i < r.size(); ++i){
        s2 = 0.9 * s2 + 0.1 * r[i] * r[i];
        EXPECT_NEAR(s[i + 1], std::sqrt(s2), 1e-16);
    }
    const auto pr = portfolio_returns(fixture_returns(), test_w());
    const auto p = ewma_vol_path(pr, 0.94);
    EXPECT_NEAR(p[0], 0.005571886972206866, 1e-15);
    EXPECT_NEAR(p[150], 0.005306344625879809, 1e-15);
    EXPECT_NEAR(p.back(), 0.005935234517846302, 1e-15);
}

TEST(Fhs, MatchesPython){
    const auto pr = portfolio_returns(fixture_returns(), test_w());
    const VarEs a = fhs_var_es(pr, 0.99, 0.94);
    EXPECT_NEAR(a.var, 0.011216141455003959, 1e-14);
    EXPECT_NEAR(a.es, 0.011582564300348104, 1e-14);
    const VarEs b = fhs_var_es(pr, 0.975, 0.97);
    EXPECT_NEAR(b.var, 0.009807738635540278, 1e-14);
    EXPECT_NEAR(b.es, 0.010419744371985014, 1e-14);
}

TEST(EwmaParametric, MatchesPython){
    const VarEs v = ewma_parametric_var_es(fixture_returns(), test_w(), 0.99, 0.94);
    EXPECT_NEAR(v.var, 0.013807420209635267, 1e-12);
    EXPECT_NEAR(v.es, 0.015818671446196583, 1e-12);
}

TEST(Fhs, RescalesHistoryToCurrentVolatility){
    Rng rng(3);
    const auto calm = distributions::normal::sample(200, 0.0, 0.01, rng);
    auto stressed = calm;
    const auto tail = distributions::normal::sample(50, 0.0, 0.03, rng);
    stressed.insert(stressed.end(), tail.begin(), tail.end());
    EXPECT_GT(fhs_var_es(stressed, 0.99).var, 2.0 * fhs_var_es(calm, 0.99).var);
    EXPECT_GT(fhs_var_es(stressed, 0.99).var, historical_var_es(stressed, 0.99).var);
}

TEST(Ewma, InvalidInputsThrow){
    EXPECT_THROW(ewma_weights(10, 1.0), InvalidInputError);
    EXPECT_THROW(ewma_weights(10, 0.0), InvalidInputError);
    EXPECT_THROW(ewma_weights(0, 0.9), InvalidInputError);
    EXPECT_THROW(ewma_vol_path({0.01}, 0.9), InvalidInputError);
    EXPECT_THROW(ewma_vol_path({0.0, 0.0, 0.0}, 0.9), InvalidInputError);
    EXPECT_THROW(ewma_covariance(ReturnMatrix(0, 2), 0.9), InvalidInputError);
}

TEST(AdaptiveBacktest, NoLookAhead){
    const PriceMatrix p = gaussian_prices(400, 0);
    PriceMatrix future = p;
    for(Eigen::Index r = 201; r < future.rows(); ++r) future.row(r) *= 0.5 + 2.5 * static_cast<double>(r - 201) / static_cast<double>(future.rows() - 201);
    for(auto m : {BacktestMethod::EwmaParametric, BacktestMethod::FilteredHistorical}){
        const BacktestSeries a = rolling_backtest(p, test_w(), 100, 0.99, m);
        const BacktestSeries b = rolling_backtest(future, test_w(), 100, 0.99, m);
        for(std::size_t k = 0; k <= 100; ++k){
            EXPECT_EQ(a.var[k], b.var[k]);
            EXPECT_EQ(a.es[k], b.es[k]);
        }
    }
}

TEST(AdaptiveBacktest, FewerBreachesAfterVolatilityJump){
    const PriceMatrix p = gaussian_prices(900, 6, 600, 3.0);
    auto after = [&](BacktestMethod m){
        const BacktestSeries s = rolling_backtest(p, test_w(), 250, 0.99, m);
        const auto b = s.breaches();
        std::size_t n = 0;
        for(std::size_t i = 0; i < b.size(); ++i) if(s.rows[i] >= 601 && b[i]) ++n;
        return n;
    };
    EXPECT_LT(after(BacktestMethod::EwmaParametric), after(BacktestMethod::Parametric));
    EXPECT_LT(after(BacktestMethod::FilteredHistorical), after(BacktestMethod::Historical));
}
