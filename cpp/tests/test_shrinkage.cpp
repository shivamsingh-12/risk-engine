#include <gtest/gtest.h>
#include <cmath>
#include "qre/core/rng.hpp"
#include "qre/distributions/normal.hpp"
#include "qre/linalg/cholesky.hpp"
#include "qre/linalg/covariance.hpp"
#include "qre/linalg/shrinkage.hpp"
#include "qre/stats/pca.hpp"
using namespace qre;

namespace {
//Expected values from the Python reference, whose Ledoit-Wolf matches sklearn to 1e-10
ReturnMatrix fixture(){
    ReturnMatrix x(40, 12);
    for(Eigen::Index t = 0; t < 40; ++t)
        for(Eigen::Index n = 0; n < 12; ++n){
            const double dt = static_cast<double>(t), dn = static_cast<double>(n);
            x(t, n) = 0.01 * std::sin(0.7 * dt + 1.3 * dn) + 0.006 * std::sin(0.31 * dt) + 0.003 * std::cos(2.3 * dt * (dn + 1.0));
        }
    return x;
}
CovarianceMatrix equicorr(Eigen::Index n, double rho){
    CovarianceMatrix c = CovarianceMatrix::Constant(n, n, rho);
    c.diagonal().setOnes();
    return c;
}
ReturnMatrix draw(const CovarianceMatrix& cov, Eigen::Index t, uint64_t seed){
    Rng rng(seed);
    const Eigen::MatrixXd L = linalg::cholesky_decompose(cov);
    const auto z = distributions::normal::sample(static_cast<std::size_t>(t * cov.rows()), 0.0, 1.0, rng);
    Eigen::MatrixXd zm(cov.rows(), t);
    for(Eigen::Index j = 0; j < t; ++j) for(Eigen::Index i = 0; i < cov.rows(); ++i) zm(i, j) = z[static_cast<std::size_t>(j * cov.rows() + i)];
    return (L * zm).transpose();
}
}

TEST(LedoitWolf, MatchesPythonAndSklearn){
    const auto r = linalg::ledoit_wolf(fixture());
    EXPECT_NEAR(r.intensity, 0.08196699795497114, 1e-13);
    EXPECT_NEAR(r.cov(0, 0), 7.200544842506921e-05, 1e-18);
    EXPECT_NEAR(r.cov(0, 1), 2.6617375556595353e-05, 1e-18);
    EXPECT_NEAR(r.cov(5, 11), 1.7571743115305397e-05, 1e-18);
}

TEST(LedoitWolf, PositiveDefiniteWhenNExceedsT){
    const ReturnMatrix x = draw(equicorr(80, 0.3), 50, 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> s(linalg::sample_cov(x));
    EXPECT_LT(s.eigenvalues().minCoeff(), 1e-10);                  //sample covariance is singular
    const auto r = linalg::ledoit_wolf(x);
    EXPECT_GT(r.intensity, 0.0);
    EXPECT_LE(r.intensity, 1.0);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> l(r.cov);
    EXPECT_GT(l.eigenvalues().minCoeff(), 0.0);
    EXPECT_NO_THROW(linalg::cholesky_decompose(r.cov));
}

TEST(LedoitWolf, ShrinksLessWithMoreData){
    const double a = linalg::ledoit_wolf(draw(equicorr(20, 0.3), 40, 40)).intensity;
    const double b = linalg::ledoit_wolf(draw(equicorr(20, 0.3), 400, 400)).intensity;
    const double c = linalg::ledoit_wolf(draw(equicorr(20, 0.3), 4000, 4000)).intensity;
    EXPECT_GT(a, b);
    EXPECT_GT(b, c);
}

TEST(Rmt, MatchesPython){
    const auto r = linalg::rmt_clip(fixture());
    EXPECT_NEAR(r.lambda_plus, 2.395445115010332, 1e-14);
    EXPECT_EQ(r.n_signal, 3);
    EXPECT_NEAR(r.cov(0, 0), 7.403661229597196e-05, 1e-18);
    EXPECT_NEAR(r.cov(0, 1), 2.9779702113058893e-05, 1e-17);
    EXPECT_NEAR(r.cov(5, 11), 1.976599398875995e-05, 1e-17);
}

TEST(Rmt, NoSignalInPureNoiseAndKeepsAMarketFactor){
    const auto noise = linalg::rmt_clip(draw(CovarianceMatrix::Identity(100, 100), 400, 4));
    EXPECT_EQ(noise.n_signal, 0);
    const Eigen::VectorXd sd = noise.cov.diagonal().cwiseSqrt();
    const Eigen::MatrixXd corr = noise.cov.array() / (sd * sd.transpose()).array();
    EXPECT_LT((corr - Eigen::MatrixXd::Identity(100, 100)).cwiseAbs().maxCoeff(), 1e-10);
    const auto market = linalg::rmt_clip(draw(equicorr(50, 0.3), 100, 5));
    EXPECT_GE(market.n_signal, 1);
    EXPECT_NO_THROW(linalg::cholesky_decompose(market.cov));
}

TEST(Pca, EquicorrelationAndPython){
    const auto r = stats::pca(equicorr(10, 0.3));
    EXPECT_NEAR(r.eigenvalues(0), 1.0 + 9.0 * 0.3, 1e-12);
    for(Eigen::Index k = 1; k < 10; ++k) EXPECT_NEAR(r.eigenvalues(k), 0.7, 1e-12);
    for(Eigen::Index i = 0; i < 10; ++i) EXPECT_NEAR(r.components(i, 0), 1.0 / std::sqrt(10.0), 1e-12);
    EXPECT_NEAR(r.explained.sum(), 1.0, 1e-14);
    const auto p = stats::pca(linalg::sample_cov(fixture()));
    EXPECT_NEAR(p.eigenvalues(0), 0.00032824727152548194, 1e-17);
    EXPECT_NEAR(p.eigenvalues(1), 0.0002883423996638601, 1e-17);
    EXPECT_NEAR(p.eigenvalues(2), 0.00020488710967940069, 1e-17);
    EXPECT_NEAR(p.explained(0), 0.38107392596017176, 1e-13);
    EXPECT_NEAR(p.components(0, 0), 0.33528826479398866, 1e-11);
    Eigen::MatrixXd bad(2, 2); bad << 1.0, 0.5, 0.4, 1.0;
    EXPECT_THROW(stats::pca(bad), InvalidInputError);
}

TEST(MinVariance, WeightsAndFailure){
    const Eigen::VectorXd w = linalg::min_variance_weights(Eigen::Vector3d(0.01, 0.04, 0.09).asDiagonal().toDenseMatrix());
    const double s = 100.0 + 25.0 + 100.0 / 9.0;
    EXPECT_NEAR(w(0), 100.0 / s, 1e-14);
    EXPECT_NEAR(w(2), 100.0 / 9.0 / s, 1e-14);
    EXPECT_THROW(linalg::min_variance_weights(Eigen::MatrixXd::Ones(3, 3)), InvalidInputError);
    const Eigen::VectorXd lw = linalg::min_variance_weights(linalg::ledoit_wolf(fixture()).cov);
    EXPECT_NEAR(lw(0), 0.0803113709964328, 1e-11);
    EXPECT_NEAR(lw(1), 0.07191353957525348, 1e-11);
    EXPECT_NEAR(lw(2), 0.09225800968781019, 1e-11);
}
