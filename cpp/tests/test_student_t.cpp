#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>
#include "qre/distributions/student_t.hpp"
#include "qre/risk/backtest.hpp"
#include "qre/risk/garch.hpp"
using namespace qre;
using namespace qre::risk;
namespace st = qre::distributions::student_t;

namespace {
//Expected values from scipy (scipy.stats.t, scipy.special.betainc) and the Python reference
std::vector<double> fixture(){
    std::vector<double> r(400);
    for(std::size_t i = 0; i < r.size(); ++i){
        const double di = static_cast<double>(i);
        r[i] = 0.01 * (1.0 + 0.8 * std::sin(2.0 * std::numbers::pi * di / 150.0)) * std::sin(1.7 * di + 0.3) * (1.0 + 0.5 * std::cos(0.37 * di));
    }
    r[50] = -0.045; r[170] = 0.04; r[290] = -0.05;      //a few fat-tail shocks
    return r;
}

//GARCH(1,1) with standardized Student-t shocks (nu = 0 -> normal shocks)
std::vector<double> simulate(std::size_t n, double nu, uint64_t seed){
    std::mt19937_64 eng(seed);
    std::normal_distribution<double> z(0.0, 1.0);
    std::chi_squared_distribution<double> chi(nu > 0.0 ? nu : 1.0);
    const double omega = 2e-6, alpha = 0.08, beta = 0.90;
    double h = omega / (1.0 - alpha - beta);
    std::vector<double> r(n);
    for(std::size_t t = 0; t < n; ++t){
        const double shock = nu > 0.0 ? z(eng) / std::sqrt(chi(eng) / nu) * std::sqrt((nu - 2.0) / nu) : z(eng);
        r[t] = std::sqrt(h) * shock;
        h = omega + alpha * r[t] * r[t] + beta * h;
    }
    return r;
}
}

TEST(IncompleteBeta, MatchesScipy){
    EXPECT_NEAR(st::detail::incomplete_beta(2.5, 0.5, 0.3), 0.018927124071945658, 1e-15);
    EXPECT_NEAR(st::detail::incomplete_beta(1.0, 1.0, 0.7), 0.7, 1e-15);
    EXPECT_NEAR(st::detail::incomplete_beta(10.0, 0.5, 0.95), 0.31715157546554495, 1e-14);
    EXPECT_NEAR(st::detail::incomplete_beta(0.5, 0.5, 0.01), 0.06376856085851985, 1e-15);
}

TEST(StudentT, CdfMatchesScipy){
    struct C{ double x, nu, p; };
    for(const C& c : {C{0.0, 5, 0.5}, C{1.5, 3, 0.8847080673775886}, C{-2.0, 7.5, 0.041448497649083156},
                      C{4.0, 2.5, 0.9804935120793409}, C{-10.0, 4, 0.0002810018113579955},
                      C{0.3, 50, 0.6172895598723483}, C{25.0, 3, 0.9999298343050212}}){
        EXPECT_NEAR(st::cdf(c.x, c.nu), c.p, 1e-14) << c.x << "," << c.nu;
    }
}

TEST(StudentT, PdfAndQuantileMatchScipy){
    EXPECT_NEAR(st::pdf(0.0, 5), 0.3796066898224944, 1e-15);
    EXPECT_NEAR(st::pdf(2.0, 3), 0.06750966066389291, 1e-15);
    EXPECT_NEAR(st::pdf(-1.0, 10), 0.23036198922913867, 1e-15);
    struct Q{ double p, nu, x; };
    for(const Q& q : {Q{0.5, 5, 0.0}, Q{0.975, 3, 3.1824463052837078}, Q{0.99, 5, 3.3649299989072174},
                      Q{0.999, 4, 7.173182219782307}, Q{0.01, 8, -2.8964594477096224}, Q{0.9999, 3, 22.203742273205}}){
        EXPECT_NEAR(st::quantile(q.p, q.nu), q.x, 1e-10 * (1.0 + std::abs(q.x))) << q.p << "," << q.nu;
    }
    EXPECT_THROW(st::quantile(1.0, 5), InvalidInputError);
    EXPECT_THROW(st::cdf(1.0, 0.0), InvalidInputError);
}

TEST(StdTVarEs, MatchesPythonAndRecoversNormal){
    const VarEs a = std_t_var_es(0.01, 5.0, 0.99);
    EXPECT_NEAR(a.var, 0.02606463569384279, 1e-14);
    EXPECT_NEAR(a.es, 0.03448836760048016, 1e-14);
    const VarEs b = std_t_var_es(0.02, 3.5, 0.975);
    EXPECT_NEAR(b.var, 0.038494796381244194, 1e-13);
    EXPECT_NEAR(b.es, 0.057561771175396, 1e-13);
    const VarEs n = std_t_var_es(1.0, 1e7, 0.99);
    EXPECT_NEAR(n.var, 2.3263478740408408, 1e-5);
    EXPECT_THROW(std_t_var_es(1.0, 2.0, 0.99), InvalidInputError);
}

TEST(GarchT, FitMatchesPythonReference){
    const auto r = fixture();
    const GarchFit f = fit_garch(r, GarchDist::StudentT);
    EXPECT_TRUE(f.converged);
    EXPECT_NEAR(f.long_run_var, 9.667186187346361e-05, 1e-18);
    EXPECT_NEAR(f.loglik, 1390.470810662687, 1e-6);
    EXPECT_NEAR(f.alpha, 0.3852181764230457, 1e-4);
    EXPECT_NEAR(f.beta, 0.6067610207927583, 1e-4);
    EXPECT_NEAR(f.nu, 17.24738790970707, 1e-2);
    const VarEs v = garch_var_es(r, 0.99, false, GarchDist::StudentT);
    EXPECT_NEAR(v.var, 0.00552623036115263, 1e-7);
    EXPECT_NEAR(v.es, 0.0065393753010400445, 1e-7);
}

TEST(GarchT, RecoversFatTailsAndPrefersTOverNormal){
    const auto fat = simulate(8000, 5.0, 2);
    const GarchFit f = fit_garch(fat, GarchDist::StudentT);
    EXPECT_NEAR(f.nu, 5.0, 1.5);
    EXPECT_NEAR(f.alpha, 0.08, 0.03);
    EXPECT_NEAR(f.beta, 0.90, 0.03);
    EXPECT_GT(f.loglik, fit_garch(fat).loglik + 20.0);
    EXPECT_GT(fit_garch(simulate(8000, 0.0, 3), GarchDist::StudentT).nu, 20.0);
}

TEST(GarchT, BacktestNoLookAhead){
    Rng rng(7);
    const auto z = distributions::normal::sample(400, 0.0, 0.01, rng);
    PriceMatrix p(201, 2);
    p.row(0).setConstant(100.0);
    for(Eigen::Index t = 0; t < 200; ++t)
        for(Eigen::Index c = 0; c < 2; ++c) p(t + 1, c) = p(t, c) * (1.0 + z[static_cast<std::size_t>(2 * t + c)]);
    PriceMatrix future = p;
    for(Eigen::Index r = 181; r < future.rows(); ++r) future.row(r) *= 0.5 + 2.5 * static_cast<double>(r - 181) / static_cast<double>(future.rows() - 181);
    WeightVector w(2); w << 0.6, 0.4;
    const auto a = rolling_backtest(p, w, 150, 0.99, BacktestMethod::GarchStudentT);
    const auto b = rolling_backtest(future, w, 150, 0.99, BacktestMethod::GarchStudentT);
    for(std::size_t k = 0; k <= 181 - 151; ++k) EXPECT_EQ(a.var[k], b.var[k]);
}
