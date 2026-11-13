#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "qre/hedging/deep_hedging.hpp"
using namespace qre;
using namespace qre::hedging;

namespace {
const double kT = 30.0 / 252.0;
//Same deterministic weights and paths as python_reference/tests/test_phase17.py
Policy deterministic_policy(Eigen::Index n_in, Eigen::Index hidden){
    auto w = [](Eigen::Index a, Eigen::Index b, double s){
        Matrix m(a, b);
        for(Eigen::Index i = 0; i < a; ++i) for(Eigen::Index j = 0; j < b; ++j) m(i, j) = 0.3 * std::sin(s + static_cast<double>(i + 2 * j));
        return m;
    };
    Policy p;
    p.W1 = w(n_in, hidden, 1.0);
    p.W2 = w(hidden, hidden, 2.0);
    p.W3 = w(hidden, 1, 3.0);
    p.b1 = RowVector(hidden);
    p.b2 = RowVector(hidden);
    for(Eigen::Index j = 0; j < hidden; ++j){ p.b1(j) = 0.1 * std::cos(static_cast<double>(j)); p.b2(j) = 0.05 * std::sin(static_cast<double>(j)); }
    p.b3 = 0.5;
    p.w = 1.5;
    return p;
}
void deterministic_paths(Eigen::Index n_paths, Eigen::Index n_steps, Matrix& S, Matrix& V){
    S.resize(n_paths, n_steps + 1);
    V.resize(n_paths, n_steps + 1);
    for(Eigen::Index i = 0; i < n_paths; ++i){
        for(Eigen::Index k = 0; k <= n_steps; ++k){
            const double di = static_cast<double>(i), dk = static_cast<double>(k);
            S(i, k) = 100.0 * std::exp(0.03 * std::sin(0.7 * di + 1.3 * dk) * std::sqrt(dk / static_cast<double>(n_steps))
                                       + 0.01 * dk * static_cast<double>(i % 3 - 1));
            V(i, k) = 0.04 + 0.01 * std::cos(di + dk);
        }
    }
}
}

TEST(DeepHedging, ForwardAndGradientMatchPython){
    Matrix S, V;
    deterministic_paths(6, 5, S, V);
    EXPECT_NEAR(S(3, 2), 96.17774366895308, 1e-12);
    const Policy p = deterministic_policy(4, 5);
    const ObjectiveGrad r = objective_and_grad(p, S, 100.0, kT, 0.002, 0.7, &V);
    EXPECT_NEAR(r.J, 3.8751123261167377, 1e-12);
    const double L[6] = {2.1775785313579474, 1.4238278551519294, 4.544503646891858, 1.4763822074936017, 0.40973245961751165, 2.0531200087603234};
    for(Eigen::Index i = 0; i < 6; ++i) EXPECT_NEAR(r.L(i), L[i], 1e-12);
    EXPECT_NEAR(r.grad.W1(1, 2), 0.01690417332873766, 1e-13);
    EXPECT_NEAR(r.grad.W2(3, 1), 0.029613745988649375, 1e-13);
    EXPECT_NEAR(r.grad.W3(4, 0), 0.30303032500835714, 1e-13);
    EXPECT_NEAR(r.grad.b1(0), -0.09954144697565234, 1e-13);
    EXPECT_NEAR(r.grad.b2(2), 0.9832647338767948, 1e-13);
    EXPECT_NEAR(r.grad.b3, -3.42249975250481, 1e-12);
    EXPECT_NEAR(r.grad.w, -0.6666666666666665, 1e-14);
    const Matrix d = policy_deltas(p, S, 100.0, kT, &V);
    const double row2[5] = {0.4684078011012041, 0.46604527831883624, 0.4713742155702848, 0.47649774132197065, 0.4826301437764599};
    for(Eigen::Index k = 0; k < 5; ++k) EXPECT_NEAR(d(2, k), row2[k], 1e-13);
    EXPECT_THROW(objective_and_grad(p, S, 100.0, kT, 0.0, 0.9), InvalidInputError);     //4 inputs need variance
}

TEST(DeepHedging, GradientMatchesFiniteDifferences){
    Matrix S, V;
    deterministic_paths(40, 6, S, V);
    std::mt19937_64 eng(1);
    std::normal_distribution<double> nd(0.0, 0.02);
    for(Eigen::Index i = 0; i < S.size(); ++i) S.data()[i] *= std::exp(nd(eng));
    const Policy p = deterministic_policy(3, 7);
    const ObjectiveGrad r = objective_and_grad(p, S, 100.0, kT, 0.003, 0.8);
    auto J = [&](const Policy& q){ return objective_and_grad(q, S, 100.0, kT, 0.003, 0.8).J; };
    const double h = 1e-6;
    for(auto [i, j] : {std::pair<Eigen::Index, Eigen::Index>{0, 0}, {1, 3}, {2, 6}}){
        Policy up = p, dn = p;
        up.W1(i, j) += h; dn.W1(i, j) -= h;
        EXPECT_NEAR(r.grad.W1(i, j), (J(up) - J(dn)) / (2 * h), 1e-6);
    }
    for(auto [i, j] : {std::pair<Eigen::Index, Eigen::Index>{0, 1}, {4, 5}}){
        Policy up = p, dn = p;
        up.W2(i, j) += h; dn.W2(i, j) -= h;
        EXPECT_NEAR(r.grad.W2(i, j), (J(up) - J(dn)) / (2 * h), 1e-6);
    }
    Policy up = p, dn = p;
    up.b3 += h; dn.b3 -= h;
    EXPECT_NEAR(r.grad.b3, (J(up) - J(dn)) / (2 * h), 1e-6);
}

TEST(DeepHedging, LossCvarAndBlackScholesReplication){
    Matrix S(1, 4);
    S << 100.0, 102.0, 101.0, 105.0;
    Matrix d(1, 3);
    d << 0.5, 0.6, 0.2;
    const double gains = 0.5 * 2 - 0.6 + 0.2 * 4, costs = 0.01 * (100 * 0.5 + 102 * 0.1 + 101 * 0.4 + 105 * 0.2);
    EXPECT_NEAR(hedge_loss(S, d, 100.0, 0.01)(0), 5.0 - gains + costs, 1e-12);
    Vector L(100);
    for(Eigen::Index i = 0; i < 100; ++i) L(i) = static_cast<double>(i + 1);
    EXPECT_NEAR(cvar(L, 0.9), 95.5, 1e-12);
    std::mt19937_64 eng(3);
    const Matrix P = simulate_gbm(20000, 200, 100.0, 0.0, 0.2, kT, eng);
    const Vector Lb = hedge_loss(P, bs_hedge_deltas(P, 100.0, kT, 0.2), 100.0, 0.0);
    EXPECT_NEAR(Lb.mean(), bs_call(100.0, 100.0, 0.2, kT), 0.01);
}

TEST(DeepHedging, HestonSimulationAndShortTrainingRun){
    std::mt19937_64 eng(4);
    Matrix S, V;
    simulate_heston(40000, 10, 100.0, 0.0, 0.04, 2.0, 0.04, 0.5, -0.7, 0.5, eng, S, V);
    EXPECT_EQ(S.cols(), 11);
    EXPECT_TRUE((V.array() >= 0.0).all());
    EXPECT_NEAR(S.col(10).mean(), 100.0, 0.3);
    Sampler gbm = [](std::size_t n, std::mt19937_64& e, Matrix& P, Matrix&){ P = simulate_gbm(n, 10, 100.0, 0.0, 0.2, kT, e); };
    TrainConfig cfg;
    cfg.iterations = 150; cfg.batch = 1024; cfg.lr = 1e-2; cfg.seed = 1;
    const Policy p0 = Policy::init(3, 8, 0);
    const Policy p1 = train(p0, gbm, 100.0, kT, 0.0, 0.9, cfg);
    std::mt19937_64 e2(9);
    const Matrix T1 = simulate_gbm(20000, 10, 100.0, 0.0, 0.2, kT, e2);
    const double before = cvar(hedge_loss(T1, policy_deltas(p0, T1, 100.0, kT), 100.0, 0.0), 0.9);
    const double after = cvar(hedge_loss(T1, policy_deltas(p1, T1, 100.0, kT), 100.0, 0.0), 0.9);
    const double bs = cvar(hedge_loss(T1, bs_hedge_deltas(T1, 100.0, kT, 0.2), 100.0, 0.0), 0.9);
    EXPECT_LT(after, 0.7 * before);
    EXPECT_LT(after, 1.15 * bs);
}
