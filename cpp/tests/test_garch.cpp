#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <vector>
#include "qre/optimization/nelder_mead.hpp"
#include "qre/risk/backtest.hpp"
#include "qre/risk/ewma.hpp"
#include "qre/risk/garch.hpp"
using namespace qre;
using namespace qre::risk;

namespace {
//Deterministic series with volatility clustering, shared with the Python cross-check
std::vector<double> fixture(){
    std::vector<double> r(400);
    for(std::size_t i = 0; i < r.size(); ++i){
        const double di = static_cast<double>(i);
        r[i] = 0.01 * (1.0 + 0.8 * std::sin(2.0 * std::numbers::pi * di / 150.0)) * std::sin(1.7 * di + 0.3) * (1.0 + 0.5 * std::cos(0.37 * di));
    }
    return r;
}

std::vector<double> simulate_garch(std::size_t n, double omega, double alpha, double beta, uint64_t seed){
    Rng rng(seed);
    const auto z = distributions::normal::sample(n, 0.0, 1.0, rng);
    double h = omega / (1.0 - alpha - beta);
    std::vector<double> r(n);
    for(std::size_t t = 0; t < n; ++t){
        r[t] = std::sqrt(h) * z[t];
        h = omega + alpha * r[t] * r[t] + beta * h;
    }
    return r;
}
}

//------------------------------------------------------------------ Nelder-Mead
TEST(NelderMead, Rosenbrock){
    auto rosen = [](const Eigen::VectorXd& x){ return 100.0 * std::pow(x(1) - x(0) * x(0), 2) + std::pow(1.0 - x(0), 2); };
    Eigen::VectorXd x0(2); x0 << -1.2, 1.0;
    const auto r = optimization::nelder_mead(rosen, x0);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.x(0), 1.0, 1e-6);
    EXPECT_NEAR(r.x(1), 1.0, 1e-6);
    EXPECT_LT(r.fx, 1e-12);
}

TEST(NelderMead, ShiftedQuadraticIn4D){
    Eigen::VectorXd target(4); target << 0.3, -2.0, 5.0, 1.5;
    auto f = [&target](const Eigen::VectorXd& x){ return (x - target).squaredNorm() + 7.0; };
    const auto r = optimization::nelder_mead(f, Eigen::VectorXd::Zero(4), 1.0);
    EXPECT_TRUE(r.converged);
    EXPECT_LT((r.x - target).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_NEAR(r.fx, 7.0, 1e-10);
}

TEST(NelderMead, AvoidsNonFiniteRegions){
    //log barrier: undefined for x <= 0, minimum at x = 1
    auto f = [](const Eigen::VectorXd& x){ return x(0) - std::log(x(0)); };
    Eigen::VectorXd x0(1); x0 << 0.2;
    const auto r = optimization::nelder_mead(f, x0, 1.0);
    EXPECT_NEAR(r.x(0), 1.0, 1e-6);
}

TEST(NelderMead, ReportsNonConvergenceAndBadInput){
    auto f = [](const Eigen::VectorXd& x){ return x.squaredNorm(); };
    Eigen::VectorXd x0(2); x0 << 10.0, -10.0;
    EXPECT_FALSE(optimization::nelder_mead(f, x0, 0.5, 1e-12, 1e-14, 5).converged);
    EXPECT_THROW(optimization::nelder_mead(f, Eigen::VectorXd(0)), InvalidInputError);
    auto bad = [](const Eigen::VectorXd&){ return std::nan(""); };
    EXPECT_THROW(optimization::nelder_mead(bad, x0), InvalidInputError);
}

//------------------------------------------------------------------ GARCH
TEST(Garch, ReparameterizationRoundTripAndConstraints){
    for(auto [a, b] : std::vector<std::pair<double, double>>{{0.05, 0.90}, {0.2, 0.5}, {0.001, 0.998}, {0.3, 0.0001}}){
        const auto [a2, b2] = garch_params_from_theta(garch_theta_from_params(a, b));
        EXPECT_NEAR(a2, a, 1e-12);
        EXPECT_NEAR(b2, b, 1e-12);
    }
    Rng rng(0);
    const auto z = distributions::normal::sample(1000, 0.0, 20.0, rng);
    for(std::size_t i = 0; i + 1 < z.size(); i += 2){
        Eigen::VectorXd t(2); t << z[i], z[i + 1];
        const auto [a, b] = garch_params_from_theta(t);
        EXPECT_GE(a, 0.0); EXPECT_GE(b, 0.0); EXPECT_LT(a + b, 1.0);
    }
}

TEST(Garch, EwmaIsTheUnitPersistenceSpecialCase){
    const auto r = fixture();
    double v = 0.0;
    for(double x : r) v += x * x;
    v /= static_cast<double>(r.size());
    const auto h = garch_variance_path(r, 0.0, 0.06, 0.94, v);
    const auto s = ewma_vol_path(r, 0.94);
    for(std::size_t i = 0; i < h.size(); ++i) EXPECT_NEAR(std::sqrt(h[i]), s[i], 1e-15);
}

TEST(Garch, FitMatchesPythonScipyReference){
    const auto r = fixture();
    const GarchFit f = fit_garch(r);
    EXPECT_TRUE(f.converged);
    EXPECT_NEAR(f.long_run_var, 8.179088655833319e-05, 1e-18);
    EXPECT_NEAR(f.loglik, 1419.8278623751073, 1e-6);       //same optimum, independent optimizers
    EXPECT_NEAR(f.alpha, 0.3294917923604826, 1e-5);
    EXPECT_NEAR(f.beta, 0.6643193118712172, 1e-5);
    EXPECT_NEAR(garch_forecast_variance(f, r), 5.8740514356119164e-06, 1e-10);
    const VarEs p = garch_var_es(r, 0.99);
    EXPECT_NEAR(p.var, 0.0056382396412457386, 1e-7);
    EXPECT_NEAR(p.es, 0.006459531111941414, 1e-7);
    const VarEs fh = garch_var_es(r, 0.975, true);
    EXPECT_NEAR(fh.var, 0.00494447493224914, 1e-7);
    EXPECT_NEAR(fh.es, 0.005481135271783658, 1e-7);
}

TEST(Garch, RecoversTrueParametersOnLongSimulation){
    const auto r = simulate_garch(8000, 2e-6, 0.08, 0.90, 2);
    const GarchFit f = fit_garch(r);
    EXPECT_NEAR(f.alpha, 0.08, 0.025);
    EXPECT_NEAR(f.beta, 0.90, 0.03);
    EXPECT_NEAR(f.long_run_var, 1e-4, 0.15e-4);
}

TEST(Garch, ForecastReactsToShock){
    auto r = simulate_garch(1000, 2e-6, 0.08, 0.90, 5);
    const GarchFit f = fit_garch(r);
    const double calm = garch_forecast_variance(f, r);
    r.back() = -0.06;
    EXPECT_GT(garch_forecast_variance(f, r), 2.0 * calm);
}

TEST(Garch, InvalidInputsThrow){
    EXPECT_THROW(fit_garch(std::vector<double>(50, 0.0)), InvalidInputError);
    EXPECT_THROW(fit_garch(std::vector<double>(5, 0.01)), InvalidInputError);
}

TEST(GarchBacktest, NoLookAhead){
    Rng rng(7);
    const auto z = distributions::normal::sample(440, 0.0, 0.01, rng);
    PriceMatrix p(221, 2);
    p.row(0).setConstant(100.0);
    for(Eigen::Index t = 0; t < 220; ++t)
        for(Eigen::Index c = 0; c < 2; ++c) p(t + 1, c) = p(t, c) * (1.0 + z[static_cast<std::size_t>(2 * t + c)]);
    PriceMatrix future = p;
    for(Eigen::Index r = 181; r < future.rows(); ++r) future.row(r) *= 0.5 + 2.5 * static_cast<double>(r - 181) / static_cast<double>(future.rows() - 181);
    WeightVector w(2); w << 0.6, 0.4;
    for(auto m : {BacktestMethod::GarchParametric, BacktestMethod::GarchFiltered}){
        const auto a = rolling_backtest(p, w, 150, 0.99, m);
        const auto b = rolling_backtest(future, w, 150, 0.99, m);
        for(std::size_t k = 0; k <= 181 - 151; ++k) EXPECT_EQ(a.var[k], b.var[k]);
    }
}
