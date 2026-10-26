#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>
#include "qre/linalg/covariance.hpp"
#include "qre/risk/pipeline.hpp"
#include "qre/risk/var_es.hpp"
using namespace qre;
using qre::risk::VarEs;

namespace {
//Expected values in the tests below were computed independently with numpy/scipy (see python_reference)
CovarianceMatrix test_cov(){
    CovarianceMatrix c(3, 3);
    c << 0.04, 0.012, 0.0,
         0.012, 0.09, -0.01,
         0.0, -0.01, 0.0225;
    return c / 252.0;
}
WeightVector test_w(){ WeightVector w(3); w << 0.5, 0.3, 0.2; return w; }

PriceMatrix small_prices(){
    PriceMatrix p(10, 3);
    p << 100, 50, 80,
         101, 50.5, 79,
         99.5, 51.2, 80.5,
         102, 50.9, 81,
         101.2, 52, 79.8,
         103, 51.5, 82,
         102.1, 52.8, 81.1,
         104, 53, 83,
         103.3, 52.2, 82.4,
         105, 54, 84;
    return p;
}

//Gaussian price history with known covariance, for the larger pipeline test
PriceMatrix gaussian_prices(std::size_t n, uint64_t seed){
    Rng rng(seed);
    const Eigen::MatrixXd L = linalg::cholesky_decompose(test_cov());
    Eigen::MatrixXd z(3, static_cast<Eigen::Index>(n));
    const auto draws = distributions::normal::sample(3 * n, 0.0, 1.0, rng);
    for(std::size_t j = 0; j < n; ++j)
        for(Eigen::Index i = 0; i < 3; ++i) z(i, static_cast<Eigen::Index>(j)) = draws[3 * j + static_cast<std::size_t>(i)];
    const Eigen::MatrixXd rets = (L * z).transpose();
    PriceMatrix prices(static_cast<Eigen::Index>(n) + 1, 3);
    prices.row(0).setConstant(100.0);
    for(Eigen::Index t = 0; t < static_cast<Eigen::Index>(n); ++t)
        prices.row(t + 1) = prices.row(t).array() * (1.0 + rets.row(t).array());
    return prices;
}
}

TEST(NormalQuantile, MatchesScipy){
    EXPECT_NEAR(risk::detail::normal_quantile(0.9), 1.2815515655446004, 1e-9);
    EXPECT_NEAR(risk::detail::normal_quantile(0.95), 1.6448536269514722, 1e-9);
    EXPECT_NEAR(risk::detail::normal_quantile(0.975), 1.959963984540054, 1e-9);
    EXPECT_NEAR(risk::detail::normal_quantile(0.99), 2.3263478740408408, 1e-9);
    EXPECT_NEAR(risk::detail::normal_quantile(0.999), 3.090232306167813, 1e-9);
    EXPECT_NEAR(risk::detail::normal_quantile(0.5), 0.0, 1e-9);
    EXPECT_NEAR(risk::detail::normal_quantile(0.01), -2.3263478740408408, 1e-9);
}

TEST(Historical, KnownValues){
    std::vector<double> returns;
    for(int i = 1; i <= 100; ++i) returns.push_back(-i / 100.0);   //losses 0.01 .. 1.00
    const VarEs r = risk::historical_var_es(returns, 0.95);
    EXPECT_NEAR(r.var, 0.9505, 1e-12);
    EXPECT_NEAR(r.es, 0.98, 1e-12);
}

TEST(Historical, MatchesNumpyOnSmallPortfolio){
    const PriceMatrix prices = small_prices();
    const auto port = risk::portfolio_returns(data::simple_return(prices), test_w());
    const VarEs a = risk::historical_var_es(port, 0.8);
    EXPECT_NEAR(a.var, -0.00015759257264508415, 1e-12);       //a negative VaR is legitimate on a tiny sample
    EXPECT_NEAR(a.es, 0.004870350317865068, 1e-12);
    const VarEs b = risk::historical_var_es(port, 0.9, 4);
    EXPECT_NEAR(b.var, 0.004377757455519638, 1e-12);
    EXPECT_NEAR(b.es, 0.018678939269414305, 1e-12);
}

TEST(Historical, EsAtLeastVar){
    Rng rng(1);
    auto draws = distributions::normal::sample(1000, 0.0, 0.01, rng);
    const VarEs r = risk::historical_var_es(draws, 0.975);
    EXPECT_GE(r.es, r.var);
}

TEST(Parametric, KnownValues){
    const VarEs r = risk::parametric_var_es(0.0, 0.01, 0.99);
    EXPECT_NEAR(r.var, 0.01 * 2.3263478740408408, 1e-10);
    EXPECT_NEAR(r.es, 0.01 * 2.665214220345808, 1e-9);
}

TEST(Parametric, MatchesScipyWithMeanAndHorizon){
    const VarEs r = risk::parametric_var_es(0.0004, 0.009684661949647008, 0.95, 5);
    EXPECT_NEAR(r.var, 0.03362023045356477, 1e-10);
    EXPECT_NEAR(r.es, 0.04266920603695869, 1e-10);
}

TEST(Parametric, HorizonScalesLikeSqrtTime){
    const VarEs a = risk::parametric_var_es(0.0, 0.01, 0.99, 1);
    const VarEs b = risk::parametric_var_es(0.0, 0.01, 0.99, 10);
    EXPECT_NEAR(b.var, a.var * std::sqrt(10.0), 1e-12);
    EXPECT_NEAR(b.es, a.es * std::sqrt(10.0), 1e-12);
}

TEST(Parametric, HigherConfidenceMeansHigherRisk){
    const VarEs lo = risk::parametric_var_es(0.0, 0.01, 0.95);
    const VarEs hi = risk::parametric_var_es(0.0, 0.01, 0.99);
    EXPECT_GT(hi.var, lo.var);
    EXPECT_GT(hi.es, lo.es);
}

TEST(RiskContributions, SumToPortfolioVol){
    const Eigen::VectorXd rc = risk::risk_contributions(test_w(), test_cov());
    EXPECT_NEAR(rc.sum(), risk::portfolio_vol(test_w(), test_cov()), 1e-15);
}

TEST(RiskContributions, MatchesNumpyOnSampleCovariance){
    const CovarianceMatrix cov = linalg::sample_cov(data::simple_return(small_prices()));
    EXPECT_NEAR(risk::portfolio_vol(test_w(), cov), 0.009684661949647008, 1e-14);
    const Eigen::VectorXd rc = risk::risk_contributions(test_w(), cov);
    EXPECT_NEAR(rc(0), 0.0060630320532756, 1e-13);
    EXPECT_NEAR(rc(1), 0.00139828834577231, 1e-13);
    EXPECT_NEAR(rc(2), 0.0022233415505991, 1e-13);
}

TEST(RiskContributions, IndependentAssetsProportionalToVariance){
    CovarianceMatrix cov = CovarianceMatrix::Zero(2, 2);
    cov(0, 0) = 0.01; cov(1, 1) = 0.04;
    WeightVector w(2); w << 0.5, 0.5;
    const Eigen::VectorXd rc = risk::risk_contributions(w, cov);
    EXPECT_NEAR(rc(1) / rc(0), 4.0, 1e-12);
}

TEST(RiskContributions, SingleAssetIsItsOwnVol){
    CovarianceMatrix cov(1, 1); cov << 0.0004;
    WeightVector w(1); w << 1.0;
    EXPECT_NEAR(risk::risk_contributions(w, cov)(0), 0.02, 1e-15);
}

TEST(MonteCarlo, MatchesParametricOneDay){
    Rng rng(3);
    const double sigma = risk::portfolio_vol(test_w(), test_cov());
    const VarEs p = risk::parametric_var_es(0.0, sigma, 0.99);
    const VarEs m = risk::monte_carlo_var_es(test_w(), test_cov(), 0.99, 1, 400000, rng);
    EXPECT_NEAR(m.var, p.var, 0.03 * p.var);
    EXPECT_NEAR(m.es, p.es, 0.03 * p.es);
}

TEST(MonteCarlo, HorizonScalesLikeSqrtTime){
    Rng r1(4), r2(4);
    const VarEs a = risk::monte_carlo_var_es(test_w(), test_cov(), 0.99, 1, 400000, r1);
    const VarEs b = risk::monte_carlo_var_es(test_w(), test_cov(), 0.99, 10, 400000, r2);
    EXPECT_NEAR(b.var / a.var, std::sqrt(10.0), 0.05 * std::sqrt(10.0));
}

TEST(MonteCarlo, SameSeedReproducible){
    Rng a(9), b(9);
    const VarEs x = risk::monte_carlo_var_es(test_w(), test_cov(), 0.99, 1, 10000, a);
    const VarEs y = risk::monte_carlo_var_es(test_w(), test_cov(), 0.99, 1, 10000, b);
    EXPECT_EQ(x.var, y.var);
    EXPECT_EQ(x.es, y.es);
}

TEST(Pipeline, ReportIsInternallyConsistent){
    const PriceMatrix prices = gaussian_prices(750, 0);
    Rng rng(5);
    risk::RiskConfig cfg;
    cfg.n_paths = 200000;
    const auto rep = risk::run_risk_pipeline(prices, test_w(), rng, cfg);
    const CovarianceMatrix cov = linalg::sample_cov(data::simple_return(prices));
    EXPECT_EQ(rep.n_returns, 750u);
    EXPECT_NEAR(rep.portfolio_vol, risk::portfolio_vol(test_w(), cov), 1e-15);
    EXPECT_NEAR(rep.risk_contributions.sum(), rep.portfolio_vol, 1e-14);
    EXPECT_NEAR(rep.parametric.var, rep.portfolio_vol * 2.3263478740408408, 1e-9);
    //On Gaussian data all three methods should land in the same ballpark
    EXPECT_NEAR(rep.historical.var, rep.parametric.var, 0.2 * rep.parametric.var);
    EXPECT_NEAR(rep.monte_carlo.var, rep.parametric.var, 0.03 * rep.parametric.var);
    for(const VarEs& v : {rep.historical, rep.parametric, rep.monte_carlo}){
        EXPECT_GT(v.var, 0.0);
        EXPECT_GE(v.es, v.var);
    }
}

TEST(Pipeline, ForwardFillsShortGapButRejectsLongGap){
    PriceMatrix prices = gaussian_prices(300, 1);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    risk::RiskConfig cfg;
    cfg.n_paths = 20000;
    Rng rng(1);
    PriceMatrix shortgap = prices;
    shortgap.block(100, 1, 3, 1).setConstant(nan);
    EXPECT_EQ(risk::run_risk_pipeline(shortgap, test_w(), rng, cfg).n_returns, 300u);
    PriceMatrix longgap = prices;
    longgap.block(100, 1, 10, 1).setConstant(nan);
    EXPECT_THROW(risk::run_risk_pipeline(longgap, test_w(), rng, cfg), InvalidInputError);
}

TEST(Pipeline, InputValidation){
    const PriceMatrix prices = gaussian_prices(50, 2);
    Rng rng(1);
    WeightVector two(2); two << 0.5, 0.5;
    EXPECT_THROW(risk::run_risk_pipeline(prices, two, rng), InvalidInputError);
    PriceMatrix bad = prices;
    bad(5, 0) = -1.0;
    EXPECT_THROW(risk::run_risk_pipeline(bad, test_w(), rng), InvalidInputError);
    EXPECT_THROW(risk::run_risk_pipeline(prices.topRows(2), test_w(), rng), InvalidInputError);
}

TEST(VarEsInvalid, BadInputsThrow){
    for(double c : {0.0, 1.0, -0.1, 1.5}){
        EXPECT_THROW(risk::parametric_var_es(0.0, 0.01, c), InvalidInputError);
        EXPECT_THROW(risk::historical_var_es({0.01, 0.02, 0.03}, c), InvalidInputError);
    }
    EXPECT_THROW(risk::parametric_var_es(0.0, 0.0, 0.99), InvalidInputError);
    EXPECT_THROW(risk::parametric_var_es(0.0, 0.01, 0.99, 0), InvalidInputError);
    EXPECT_THROW(risk::historical_var_es({0.01}, 0.99), InvalidInputError);
    ReturnMatrix r = ReturnMatrix::Zero(5, 3);
    WeightVector w2(2); w2 << 0.5, 0.5;
    EXPECT_THROW(risk::portfolio_returns(r, w2), InvalidInputError);
}
