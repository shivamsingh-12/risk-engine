#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include "qre/risk/adaptive.hpp"
#include "qre/risk/scoring.hpp"
using namespace qre;
using namespace qre::risk;

namespace {
//Deterministic fixture shared with the Python reference
struct Fixture{
    Eigen::MatrixXd var, es;
    std::vector<double> loss;
};
Fixture fixture(){
    const Eigen::Index T = 40, K = 3;
    const double scale[3] = {1.0, 0.8, 1.2};
    Fixture f{Eigen::MatrixXd(T, K), Eigen::MatrixXd(T, K), std::vector<double>(T)};
    for(Eigen::Index i = 0; i < T; ++i){
        const double di = static_cast<double>(i);
        f.loss[static_cast<std::size_t>(i)] = 0.02 * std::sin(1.3 * di + 0.2) * (1.0 + 0.5 * std::cos(0.7 * di));
        for(Eigen::Index k = 0; k < K; ++k){
            f.var(i, k) = (0.02 + 0.002 * std::cos(0.5 * di + static_cast<double>(k))) * scale[k];
            f.es(i, k) = 1.25 * f.var(i, k);
        }
    }
    return f;
}
}

TEST(Scoring, ValuesByHand){
    EXPECT_NEAR(quantile_loss(3.0, 2.0, 0.99), 0.99, 1e-15);
    EXPECT_NEAR(quantile_loss(1.0, 2.0, 0.99), 0.01, 1e-15);
    EXPECT_NEAR(fz0_loss(3.0, 2.0, 2.5, 0.99), 1.0 / (0.01 * 2.5) + 0.8 + std::log(2.5) - 1.0, 1e-12);
    EXPECT_NEAR(fz0_loss(1.0, 2.0, 2.5, 0.99), 0.8 + std::log(2.5) - 1.0, 1e-15);
    EXPECT_THROW(fz0_loss(1.0, 1.0, -1.0, 0.99), InvalidInputError);
    const Fixture f = fixture();
    std::vector<double> v(40), e(40);
    for(std::size_t i = 0; i < 40; ++i){ v[i] = f.var(static_cast<Eigen::Index>(i), 0); e[i] = f.es(static_cast<Eigen::Index>(i), 0); }
    double s = 0.0;
    for(double x : fz0_loss(f.loss, v, e, 0.975)) s += x;
    EXPECT_NEAR(s, -134.73158274803714, 1e-10);
}

TEST(Scoring, DieboldMarianoMatchesPython){
    const std::vector<double> a{1.0, 2.0, 0.5, 1.5, 3.0, 0.2, 1.1, 0.9, 2.2, 1.4, 0.7, 1.8};
    const std::vector<double> b{1.2, 1.5, 0.9, 1.0, 2.0, 0.4, 1.0, 1.3, 1.7, 1.2, 0.5, 1.6};
    const DmResult r = diebold_mariano(a, b, 2);
    EXPECT_NEAR(r.statistic, 2.009699739794143, 1e-12);
    EXPECT_NEAR(r.p_value, 0.04446297752643425, 1e-12);
    EXPECT_NEAR(r.mean_diff, 0.16666666666666666, 1e-15);
    EXPECT_EQ(r.lag, 2u);
    EXPECT_EQ(newey_west_lag(502), 5u);
    EXPECT_THROW(diebold_mariano(std::vector<double>(5, 1.0), std::vector<double>(5, 1.0)), InvalidInputError);
}

TEST(Adaptive, CombinationsMatchPython){
    const Fixture f = fixture();
    const CombinedForecast sel = combine(f.var, f.es, f.loss, 0.975, Combination::SelectBest, 10, 2.0, 12);
    const CombinedForecast ex = combine(f.var, f.es, f.loss, 0.975, Combination::ExpWeights, 10, 2.0, 12);
    double s1 = 0.0, s2 = 0.0;
    for(double x : sel.var) s1 += x;
    for(double x : ex.var) s2 += x;
    EXPECT_NEAR(s1, 0.5718106229670405, 1e-14);
    EXPECT_NEAR(s2, 0.5779538966893062, 1e-14);
    EXPECT_NEAR(ex.es.back(), 0.027354624503597446, 1e-15);
    EXPECT_DOUBLE_EQ(sel.weights(5, 0), 1.0);
    EXPECT_DOUBLE_EQ(sel.weights(27, 2), 1.0);
    EXPECT_NEAR(ex.weights(5, 0), 9.63184620e-01, 1e-9);
    EXPECT_NEAR(ex.weights(5, 2), 3.68153798e-02, 1e-9);
    EXPECT_NEAR(ex.weights(27, 2), 9.99999141e-01, 1e-9);
    const CombinedForecast eq = combine(f.var, f.es, f.loss, 0.975, Combination::Equal, 10, 2.0, 12);
    for(std::size_t i = 0; i < eq.var.size(); ++i){
        EXPECT_NEAR(eq.var[i], f.var.row(static_cast<Eigen::Index>(i) + 12).mean(), 1e-16);
        EXPECT_GE(eq.es[i], eq.var[i]);
    }
    EXPECT_THROW(combine(f.var, f.es, f.loss, 0.975, Combination::Equal, 10, 1.0, 40), InvalidInputError);
}

TEST(Adaptive, NoLookAheadAndSelectsTheTruth){
    //model 0 = true N(0,1) VaR/ES at 97.5%, model 1 understates by 30%
    Rng rng(4);
    const std::vector<double> loss0 = distributions::normal::sample(600, 0.0, 1.0, rng);
    const double v = 1.959963984540054, e = 2.337802792201415;
    Eigen::MatrixXd var(600, 2), es(600, 2);
    var.col(0).setConstant(v); var.col(1).setConstant(0.7 * v);
    es.col(0).setConstant(e); es.col(1).setConstant(0.7 * e);
    const CombinedForecast sel = combine(var, es, loss0, 0.975, Combination::SelectBest, 250, 1.0, 250);
    EXPECT_GT(sel.weights.col(0).mean(), 0.95);
    std::vector<double> loss1 = loss0;
    for(std::size_t i = 400; i < 600; ++i) loss1[i] *= 10.0;
    const CombinedForecast a = combine(var, es, loss0, 0.975, Combination::ExpWeights, 100, 1.0, 100);
    const CombinedForecast b = combine(var, es, loss1, 0.975, Combination::ExpWeights, 100, 1.0, 100);
    for(Eigen::Index t = 0; t <= 300; ++t){
        EXPECT_EQ(a.weights(t, 0), b.weights(t, 0));
    }
}

TEST(Adaptive, RunModelsAlignsForecasts){
    Rng rng(5);
    const std::vector<double> d = distributions::normal::sample(280, 0.0, 0.01, rng);
    PriceMatrix p(140, 2);
    p.row(0) << 100.0, 100.0;
    for(Eigen::Index t = 1; t < 140; ++t){
        for(Eigen::Index j = 0; j < 2; ++j) p(t, j) = p(t - 1, j) * (1.0 + d[static_cast<std::size_t>(2 * t + j)]);
    }
    WeightVector w(2);
    w << 0.5, 0.5;
    const ModelForecasts f = run_models(p, w, {BacktestMethod::Historical, BacktestMethod::EwmaParametric}, 100, 0.99);
    EXPECT_EQ(f.var.rows(), 39);
    EXPECT_EQ(f.var.cols(), 2);
    EXPECT_EQ(f.loss.size(), 39u);
    EXPECT_TRUE((f.es.array() >= f.var.array()).all());
    const BacktestSeries s = model_series(f, 1, 10);
    EXPECT_EQ(s.var.size(), 29u);
    EXPECT_EQ(s.rows.front(), f.rows[10]);
}

//Regression found by the Phase 13 synthetic experiment: on calm normal data nu ran off to ~1e16, where the Student-t
//cdf loses all precision and VaR/ES came out negative / ~1e32. nu is now bounded at kNuMax
TEST(GarchT, NormalReturnsGiveFiniteNormalLikeRisk){
    std::mt19937_64 eng(1313);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> r;
    for(int i = 0; i < 750; ++i) r.push_back(0.008 * nd(eng));
    for(std::size_t start : {0u, 50u, 225u, 325u, 500u}){
        const std::vector<double> w(r.begin() + static_cast<std::ptrdiff_t>(start), r.begin() + static_cast<std::ptrdiff_t>(start + 250));
        const GarchFit f = fit_garch(w, GarchDist::StudentT);
        EXPECT_GT(f.nu, 2.0);
        EXPECT_LE(f.nu, kNuMax);
        const VarEs t = garch_var_es(w, 0.975, false, GarchDist::StudentT);
        const VarEs n = garch_var_es(w, 0.975, false);
        EXPECT_NEAR(t.var / n.var, 1.0, 0.05);
        EXPECT_NEAR(t.es / n.es, 1.0, 0.05);
    }
}
