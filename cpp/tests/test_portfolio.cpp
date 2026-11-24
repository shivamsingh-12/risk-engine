#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "qre/portfolio/construction.hpp"
#include "qre/risk/component.hpp"
#include "qre/risk/horizon.hpp"

using namespace qre;
using namespace qre::portfolio;
using namespace qre::risk;

//Same fixtures as python_reference/tests/test_portfolio.py
namespace{
Eigen::MatrixXd fixture_cov(){
    Eigen::VectorXd vol(5);
    vol << 0.012, 0.015, 0.005, 0.010, 0.025;
    Eigen::MatrixXd corr(5, 5);
    corr << 1.0, 0.8, -0.3, 0.1, 0.6,
            0.8, 1.0, -0.2, 0.1, 0.7,
            -0.3, -0.2, 1.0, 0.2, -0.2,
            0.1, 0.1, 0.2, 1.0, 0.1,
            0.6, 0.7, -0.2, 0.1, 1.0;
    return corr.cwiseProduct(vol * vol.transpose());
}

Eigen::VectorXd fixture_mu(){
    Eigen::VectorXd mu(5);
    mu << 0.0006, 0.0008, 0.0002, 0.0003, 0.0012;
    return mu;
}

Eigen::MatrixXd fixture_returns(int t = 300, int n = 5){
    Eigen::MatrixXd r(t, n);
    for(int i = 0; i < t; ++i)
        for(int j = 0; j < n; ++j){
            r(i, j) = 0.01 * std::sin(0.37 * i * (j + 1) + j) + 0.004 * std::cos(1.3 * i + 2.0 * j);
            if(i % 37 == 0) r(i, j) -= 0.03 * (1 + j) / n;
        }
    return r;
}

Eigen::VectorXd vec(std::initializer_list<double> x){
    Eigen::VectorXd v(static_cast<Eigen::Index>(x.size()));
    Eigen::Index i = 0;
    for(double a : x) v(i++) = a;
    return v;
}

void expect_near(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tol){
    ASSERT_EQ(a.size(), b.size());
    for(Eigen::Index i = 0; i < a.size(); ++i) EXPECT_NEAR(a(i), b(i), tol) << "index " << i;
}
}

TEST(Construction, ProjectionStaysInsideTheSet){
    const Bounds b = make_bounds(4, -0.2, 0.6);
    const Eigen::VectorXd w = project(vec({0.9, 0.5, -0.7, 0.1}), b);
    EXPECT_NEAR(w.sum(), 1.0, 1e-15);
    EXPECT_GE(w.minCoeff(), -0.2 - 1e-15);
    EXPECT_LE(w.maxCoeff(), 0.6 + 1e-15);
    expect_near(project(vec({0.1, 0.2, 0.3, 0.4}), b), vec({0.1, 0.2, 0.3, 0.4}), 1e-15);
    EXPECT_THROW(make_bounds(3, 0.0, 0.3), InvalidInputError);
    EXPECT_THROW(make_bounds(3, 0.5, 1.0), InvalidInputError);
}

TEST(Construction, UnconstrainedMinVarianceMatchesClosedForm){
    const Eigen::MatrixXd s = fixture_cov();
    const Eigen::VectorXd one = s.llt().solve(Eigen::VectorXd::Ones(5));
    const QpResult q = min_variance(s, make_bounds(5, -10.0, 10.0));
    EXPECT_TRUE(q.converged);
    expect_near(q.weights, one / one.sum(), 1e-10);
}

TEST(Construction, RiskParityEqualizesAndRespectsBudgets){
    const Eigen::MatrixXd s = fixture_cov();
    const QpResult rp = risk_parity(s);
    EXPECT_TRUE(rp.converged);
    expect_near(risk_contribution_shares(rp.weights, s), Eigen::VectorXd::Constant(5, 0.2), 1e-11);
    const Eigen::VectorXd b = vec({0.4, 0.1, 0.2, 0.2, 0.1});
    expect_near(risk_contribution_shares(risk_parity(s, b).weights, s), b, 1e-11);
    const Eigen::MatrixXd diag = s.diagonal().asDiagonal();
    expect_near(risk_parity(diag).weights, inverse_vol(diag), 1e-12);
}

TEST(Construction, MeanVarianceLimits){
    const Eigen::MatrixXd s = fixture_cov();
    const Bounds b = make_bounds(5, 0.0, 1.0);
    expect_near(mean_variance(fixture_mu(), s, 1e7, b).weights, min_variance(s, b).weights, 1e-3);
    EXPECT_NEAR(mean_variance(fixture_mu(), s, 0.1, b).weights(4), 1.0, 1e-9);
    EXPECT_THROW(mean_variance(fixture_mu(), s, 0.0, b), InvalidInputError);
}

TEST(Construction, MatchesPythonReference){
    const Eigen::MatrixXd s = fixture_cov();
    expect_near(min_variance(s, make_bounds(5, -0.3, 1.0)).weights,
                vec({0.2262963262582542, -0.029221372460156807, 0.7447291080105434, 0.05872227742542246, -0.000526339234063241}), 1e-10);
    expect_near(mean_variance(fixture_mu(), s, 4.0, make_bounds(5, 0.0, 0.4)).weights,
                vec({0.039268697038675165, 0.3579024719954555, 0.3000822992127668, 0.06121932074202628, 0.2415272110110763}), 1e-10);
    expect_near(risk_parity(s).weights,
                vec({0.13483964337452994, 0.09597423755809534, 0.5265434533894444, 0.17992690553453836, 0.06271576014339207}), 1e-12);
}

TEST(Component, ParametricIsEulerAndMatchesFiniteDifferences){
    const Eigen::MatrixXd s = fixture_cov();
    const Eigen::VectorXd w = vec({0.5, 0.4, 0.1, 0.0, 0.2});
    const ComponentRisk c = parametric_components(w, s, 0.99);
    EXPECT_NEAR(c.component_var.sum(), c.var, 1e-15);
    EXPECT_NEAR(c.component_es.sum(), c.es, 1e-15);
    const double eps = 1e-7;
    for(Eigen::Index i = 0; i < 5; ++i){
        Eigen::VectorXd up = w, dn = w;
        up(i) += eps; dn(i) -= eps;
        const double fd = (parametric_components(up, s, 0.99).var - parametric_components(dn, s, 0.99).var) / (2 * eps);
        EXPECT_NEAR(w(i) * fd, c.component_var(i), 1e-9);
    }
    EXPECT_LT(c.component_var(2), 0.0);
    EXPECT_EQ(c.component_var(3), 0.0);
}

TEST(Component, HistoricalMatchesPythonReference){
    const ComponentRisk c = historical_components(fixture_returns(), vec({0.3, 0.2, 0.2, 0.2, 0.1}), 0.975);
    EXPECT_NEAR(c.component_es.sum(), c.es, 1e-15);
    EXPECT_NEAR(c.component_var.sum(), c.var, 1e-15);
    const Eigen::VectorXd py_var = vec({0.0013628920217820958, 0.0017305886874910519, 0.00306617267018189, 0.003898189967704644, 0.002424805224873086});
    const Eigen::VectorXd py_es = vec({0.0012643583704369536, 0.002507691747119058, 0.0037314219780306065, 0.004632851697403739, 0.002856859006729886});
    expect_near(c.component_var, py_var, 1e-15);
    expect_near(c.component_es, py_es, 1e-15);
}

TEST(Horizon, GarchVarianceSumAndPathsMatchPython){
    const GarchFit fit{1e-6, 0.1, 0.85, 2e-5, 0.0, true};
    double expect = 0.0;
    for(int k = 0; k < 10; ++k) expect += 2e-5 + (4e-4 - 2e-5) * std::pow(0.95, k);
    EXPECT_NEAR(garch_variance_sum(fit, 4e-4, 10), expect, 1e-17);
    const Eigen::VectorXd pr = fixture_returns() * vec({0.3, 0.2, 0.2, 0.2, 0.1});
    const std::vector<double> r(pr.data(), pr.data() + pr.size());
    Eigen::MatrixXi draws(200, 10);
    for(int i = 0; i < 200; ++i) for(int k = 0; k < 10; ++k) draws(i, k) = (i * 7 + k * 13) % 300;
    const auto paths = fhs_garch_paths(r, 10, draws, &fit);
    EXPECT_NEAR(paths[0], -0.03385657136014031, 1e-15);
    EXPECT_NEAR(paths[57], -0.002744770095467697, 1e-15);
    EXPECT_NEAR(paths[199], 0.009462224924436935, 1e-15);
    const VarEs gn = garch_normal_var_es(r, 0.99, 10, &fit);
    EXPECT_NEAR(gn.var, 0.034724226930752604, 1e-12);       //C++ normal quantile is a Brent root
    EXPECT_NEAR(gn.es, 0.039782314777197375, 1e-12);
    const VarEs ov = overlapping_var_es(r, 0.99, 10);
    EXPECT_NEAR(ov.var, 0.031032780714151154, 1e-15);
    EXPECT_NEAR(ov.es, 0.03114560415747854, 1e-15);
    const VarEs sq = sqrt_time_var_es(r, 0.99, 10);
    EXPECT_NEAR(sq.var, historical_var_es(r, 0.99).var * std::sqrt(10.0), 1e-15);
    EXPECT_THROW(fhs_garch_paths(r, 10, Eigen::MatrixXi::Zero(5, 9), &fit), InvalidInputError);
}

TEST(Horizon, FhsMatchesGarchNormalOnNormalGarchData){
    Rng rng(3);
    std::normal_distribution<double> nd;
    std::vector<double> r(4000);
    double h = 2e-6 / (1.0 - 0.98);
    for(auto& x : r){ x = std::sqrt(h) * nd(rng.engine()); h = 2e-6 + 0.08 * x * x + 0.9 * h; }
    const GarchFit f = fit_garch(r);
    const double a = garch_normal_var_es(r, 0.99, 10, &f).var;
    const double b = fhs_garch_var_es(r, 0.99, 10, 40000, 1, &f).var;
    EXPECT_NEAR(b / a, 1.0, 0.08);
}
