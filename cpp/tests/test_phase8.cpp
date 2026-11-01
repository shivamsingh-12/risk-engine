#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <vector>
#include "qre/models/merton.hpp"
#include "qre/models/ou.hpp"
#include "qre/montecarlo/sde.hpp"
#include "qre/pricing/black_scholes.hpp"
using namespace qre;
using qre::pricing::OptionType;

namespace {
//Expected values below come from the Python reference (python_reference/qre/{montecarlo/sde,models/*}.py)
constexpr double MU = 0.05, SIG = 0.4;
const montecarlo::SdeFn a_gbm = [](double x, double){ return MU * x; };
const montecarlo::SdeFn b_gbm = [](double x, double){ return SIG * x; };
const montecarlo::SdeFn bx_gbm = [](double, double){ return SIG; };

Eigen::VectorXd gbm_exact(const Eigen::MatrixXd& dW, double T){
    return ((MU - 0.5 * SIG * SIG) * T + SIG * dW.rowwise().sum().array()).exp().matrix();
}
double slope(const std::vector<double>& h, const std::vector<double>& e){
    double mx = 0, my = 0;
    for(std::size_t i = 0; i < h.size(); ++i){ mx += std::log(h[i]); my += std::log(e[i]); }
    mx /= static_cast<double>(h.size()); my /= static_cast<double>(h.size());
    double sxy = 0, sxx = 0;
    for(std::size_t i = 0; i < h.size(); ++i){ sxy += (std::log(h[i]) - mx) * (std::log(e[i]) - my); sxx += std::pow(std::log(h[i]) - mx, 2); }
    return sxy / sxx;
}
models::MertonParams base(double strike = 100.0){
    return models::MertonParams{100.0, strike, 0.05, 0.2, 0.5, 1.0, -0.1, 0.15};
}
}

//------------------------------------------------------------------ SDE schemes
TEST(Sde, SchemesMatchPythonOnAFixedPath){
    Eigen::MatrixXd dW(1, 20);
    for(Eigen::Index k = 0; k < 20; ++k) dW(0, k) = 0.05 * std::sin(1.3 * static_cast<double>(k) + 0.2);
    const montecarlo::SdeFn a = [](double x, double t){ return 0.05 * x + 0.1 * t; };
    EXPECT_NEAR(montecarlo::euler_maruyama(1.0, a, b_gbm, 1.0, dW)(0, 20), 1.0964918067374407, 1e-14);
    EXPECT_NEAR(montecarlo::milstein(1.0, a, b_gbm, bx_gbm, 1.0, dW)(0, 20), 1.0166286766881432, 1e-14);
    EXPECT_NEAR(montecarlo::ito_integral_w_dw(dW)(0), -0.012005479602463414, 1e-15);
    EXPECT_NEAR(montecarlo::stratonovich_integral_w_dw(dW)(0), 1.614606884891884e-07, 1e-16);
    EXPECT_NEAR(montecarlo::quadratic_variation(dW)(0), 0.024011282126303813, 1e-15);
}

TEST(Sde, CoarsenPreservesThePath){
    Rng rng(1);
    const Eigen::MatrixXd dW = montecarlo::brownian_increments(500, 64, 2.0, rng);
    const Eigen::MatrixXd c = montecarlo::coarsen(dW, 8);
    EXPECT_EQ(c.cols(), 8);
    EXPECT_NEAR((c.rowwise().sum() - dW.rowwise().sum()).cwiseAbs().maxCoeff(), 0.0, 1e-14);
    EXPECT_THROW(montecarlo::coarsen(dW, 5), InvalidInputError);
}

TEST(Sde, StrongOrdersHalfAndOne){
    Rng rng(3);
    const Eigen::MatrixXd fine = montecarlo::brownian_increments(20000, 256, 1.0, rng);
    const Eigen::VectorXd exact = gbm_exact(fine, 1.0);
    std::vector<double> h, em, mil;
    for(Eigen::Index steps : {8, 16, 32, 64, 128}){
        const Eigen::MatrixXd dW = montecarlo::coarsen(fine, 256 / steps);
        h.push_back(1.0 / static_cast<double>(steps));
        em.push_back((montecarlo::euler_maruyama(1.0, a_gbm, b_gbm, 1.0, dW).col(steps) - exact).cwiseAbs().mean());
        mil.push_back((montecarlo::milstein(1.0, a_gbm, b_gbm, bx_gbm, 1.0, dW).col(steps) - exact).cwiseAbs().mean());
    }
    EXPECT_NEAR(slope(h, em), 0.5, 0.12);
    EXPECT_NEAR(slope(h, mil), 1.0, 0.12);
}

TEST(Sde, WeakOrderOneOnSecondMoment){
    Rng rng(3);
    const Eigen::MatrixXd fine = montecarlo::brownian_increments(200000, 64, 1.0, rng);
    const double exact2 = gbm_exact(fine, 1.0).array().square().mean();
    std::vector<double> h, em, theory;
    for(Eigen::Index steps : {4, 8, 16, 32}){
        const double hh = 1.0 / static_cast<double>(steps);
        const Eigen::MatrixXd dW = montecarlo::coarsen(fine, 64 / steps);
        h.push_back(hh);
        em.push_back(std::abs(montecarlo::euler_maruyama(1.0, a_gbm, b_gbm, 1.0, dW).col(steps).array().square().mean() - exact2));
        theory.push_back(std::abs(std::pow(std::pow(1.0 + MU * hh, 2) + SIG * SIG * hh, static_cast<double>(steps)) - std::exp(2.0 * MU + SIG * SIG)));
    }
    EXPECT_NEAR(slope(h, em), 1.0, 0.2);
    for(std::size_t i = 0; i < em.size(); ++i) EXPECT_NEAR(em[i], theory[i], 0.25 * theory[i]);
}

TEST(ItoCalculus, StratonovichTelescopesAndGapIsHalfQuadraticVariation){
    Rng rng(4);
    const Eigen::MatrixXd dW = montecarlo::brownian_increments(100, 500, 1.0, rng);
    const Eigen::VectorXd wt = dW.rowwise().sum();
    const Eigen::VectorXd strat = montecarlo::stratonovich_integral_w_dw(dW);
    const Eigen::VectorXd ito = montecarlo::ito_integral_w_dw(dW);
    const Eigen::VectorXd qv = montecarlo::quadratic_variation(dW);
    for(Eigen::Index p = 0; p < 100; ++p){
        EXPECT_NEAR(strat(p), 0.5 * wt(p) * wt(p), 1e-12);
        EXPECT_NEAR(strat(p) - ito(p), 0.5 * qv(p), 1e-12);
    }
}

TEST(ItoCalculus, QuadraticVariationConvergesToT){
    Rng rng(6);
    for(Eigen::Index steps : {100, 10000}){
        const Eigen::VectorXd qv = montecarlo::quadratic_variation(montecarlo::brownian_increments(2000, steps, 2.0, rng));
        const double mean = qv.mean();
        const double sd = std::sqrt((qv.array() - mean).square().sum() / 1999.0);
        EXPECT_NEAR(mean, 2.0, 0.02);
        EXPECT_NEAR(sd, 2.0 * std::sqrt(2.0 / static_cast<double>(steps)), 0.1 * 2.0 * std::sqrt(2.0 / static_cast<double>(steps)));
    }
}

//------------------------------------------------------------------ Ornstein-Uhlenbeck
TEST(Ou, StationaryDistributionAndExponentialDecay){
    Rng rng(8);
    const Eigen::VectorXd x = models::simulate_ou(0.04, 3.0, 0.04, 0.05, 1.0 / 252.0, 2520, 4000, rng).col(2520);
    const double mean = x.mean();
    EXPECT_NEAR(mean, 0.04, 0.002);
    EXPECT_NEAR(std::sqrt((x.array() - mean).square().mean()), 0.05 / std::sqrt(6.0), 0.05 * 0.05 / std::sqrt(6.0));
    Rng rng2(9);
    const Eigen::VectorXd y = models::simulate_ou(0.10, 2.0, 0.04, 0.02, 0.5, 1, 200000, rng2).col(1);
    EXPECT_NEAR(y.mean(), 0.04 + 0.06 * std::exp(-1.0), 5e-4);
}

TEST(Ou, FitMatchesPythonAndRecoversParameters){
    std::vector<double> x(500);
    for(std::size_t k = 0; k < x.size(); ++k){
        const double dk = static_cast<double>(k);
        x[k] = 0.04 + 0.01 * std::sin(0.3 * dk) + 0.005 * std::cos(2.1 * dk) + 0.002 * std::sin(5.7 * dk);
    }
    const models::OuFit f = models::fit_ou(x, 1.0 / 252.0);
    EXPECT_NEAR(f.kappa, 100.66694220069796, 1e-9);
    EXPECT_NEAR(f.theta, 0.039948407802962216, 1e-14);
    EXPECT_NEAR(f.sigma, 0.11431339397610295, 1e-12);

    Rng rng(10);
    const Eigen::RowVectorXd path = models::simulate_ou(0.04, 3.0, 0.04, 0.05, 1.0 / 252.0, 252 * 50, 1, rng).row(0);
    const models::OuFit g = models::fit_ou(std::vector<double>(path.data(), path.data() + path.size()), 1.0 / 252.0);
    EXPECT_NEAR(g.kappa, 3.0, 0.9);
    EXPECT_NEAR(g.theta, 0.04, 0.005);
    EXPECT_NEAR(g.sigma, 0.05, 0.0015);
    EXPECT_NEAR(g.half_life(), std::numbers::ln2 / g.kappa, 1e-15);
}

TEST(Ou, RejectsNonMeanRevertingData){
    std::vector<double> explosive(200), alternating(100);
    for(std::size_t i = 0; i < 200; ++i) explosive[i] = std::pow(1.01, static_cast<double>(i));
    for(std::size_t i = 0; i < 100; ++i) alternating[i] = 5.0 + (i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_THROW(models::fit_ou(explosive, 1.0), InvalidInputError);
    EXPECT_THROW(models::fit_ou(alternating, 1.0), InvalidInputError);
    EXPECT_THROW(models::fit_ou(std::vector<double>(10, 1.0), 1.0), InvalidInputError);
}

//------------------------------------------------------------------ Merton
TEST(Merton, ClosedFormMatchesPython){
    EXPECT_NEAR(models::merton_price(base(80.0)), 22.96928196973409, 1e-10);
    EXPECT_NEAR(models::merton_price(base(100.0)), 8.448590383316304, 1e-10);
    EXPECT_NEAR(models::merton_price(base(120.0)), 1.8154456803976315, 1e-10);
    models::MertonParams put = base(); put.type = OptionType::Put;
    EXPECT_NEAR(models::merton_price(put), 5.97958158614958, 1e-10);
    const models::MertonParams lng{100.0, 90.0, 0.05, 0.2, 2.0, 0.3, 0.05, 0.3};
    EXPECT_NEAR(models::merton_price(lng), 24.230673734997243, 1e-10);
}

TEST(Merton, NoJumpsIsBlackScholesAndParityHolds){
    const models::MertonParams p{100.0, 95.0, 0.03, 0.25, 1.0, 0.0, -0.1, 0.15};
    EXPECT_NEAR(models::merton_price(p), pricing::price({100.0, 95.0, 0.03, 0.25, 1.0, OptionType::Call}), 1e-13);
    models::MertonParams put = base(); put.type = OptionType::Put;
    EXPECT_NEAR(models::merton_price(base()) - models::merton_price(put), 100.0 - 100.0 * std::exp(-0.025), 1e-10);
}

TEST(Merton, MonteCarloMatchesClosedFormAndMartingale){
    Rng rng(12);
    const auto s = models::simulate_merton_terminal(base(), 400000, rng);
    EXPECT_NEAR(estimators::empirical::sample_mean(s) * std::exp(-0.025), 100.0, 0.3);
    for(double k : {80.0, 100.0, 120.0}){
        const auto r = models::merton_mc_price(base(k), 400000, rng);
        EXPECT_LT(std::abs(r.price - models::merton_price(base(k))), 4.0 * r.std_error) << k;
    }
}

TEST(Merton, NegativeJumpsCreateDownsideSkew){
    std::vector<double> iv;
    for(double k : {80.0, 100.0, 120.0}){
        const auto r = pricing::implied_vol(models::merton_price(base(k)), {100.0, k, 0.05, 0.2, 0.5, OptionType::Call});
        ASSERT_TRUE(r.has_value());
        iv.push_back(r->root);
    }
    EXPECT_GT(iv[0], iv[1]);
    EXPECT_GT(iv[1], iv[2]);
    EXPECT_GT(iv[1], 0.2);
    EXPECT_THROW(models::merton_price({100.0, 100.0, 0.05, 0.2, 0.5, -1.0, 0.0, 0.1}), InvalidInputError);
}
