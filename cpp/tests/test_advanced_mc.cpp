#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <vector>
#include "qre/autodiff/aad.hpp"
#include "qre/montecarlo/asian.hpp"
#include "qre/montecarlo/greeks.hpp"
#include "qre/montecarlo/qmc.hpp"
#include "qre/pricing/black_scholes.hpp"
using namespace qre;
using namespace qre::montecarlo;

namespace {
//Expected values from the Python reference, whose Sobol points match scipy.stats.qmc.Sobol bit for bit
Eigen::MatrixXd corr3(){
    Eigen::MatrixXd c(3, 3);
    c << 1.0, 0.4, 0.2, 0.4, 1.0, 0.3, 0.2, 0.3, 1.0;
    return c;
}
Eigen::VectorXd vec3(double a, double b, double c){ Eigen::VectorXd v(3); v << a, b, c; return v; }
}

TEST(Sobol, MatchesPythonAndScipy){
    const std::vector<uint32_t> x = qmc::sobol_ints(6, 4);
    const std::vector<uint32_t> expected{0, 0, 0, 0,
        2147483648u, 2147483648u, 2147483648u, 2147483648u,
        3221225472u, 1073741824u, 1073741824u, 1073741824u,
        1073741824u, 3221225472u, 3221225472u, 3221225472u,
        1610612736u, 1610612736u, 2684354560u, 3758096384u,
        3758096384u, 3758096384u, 536870912u, 1610612736u};
    EXPECT_EQ(x, expected);
    const std::vector<uint32_t> big = qmc::sobol_ints(std::size_t{1} << 20, 64);
    const std::size_t last = ((std::size_t{1} << 20) - 1) * 64;
    EXPECT_EQ(big[last + 0], 4096u);
    EXPECT_EQ(big[last + 1], 4026593280u);
    EXPECT_EQ(big[last + 30], 3864023040u);
    EXPECT_EQ(big[last + 63], 2322862080u);
}

TEST(Sobol, EveryCoordinateIsStratified){
    const Eigen::MatrixXd u = qmc::sobol_points(1024, 16);
    for(Eigen::Index d = 0; d < 16; ++d){
        std::vector<int> count(1024, 0);
        for(Eigen::Index i = 0; i < 1024; ++i) ++count[static_cast<std::size_t>(std::floor(u(i, d) * 1024.0))];
        for(int c : count) EXPECT_EQ(c, 1);
    }
    EXPECT_THROW(qmc::direction_numbers(qmc::max_dim() + 1), InvalidInputError);
}

TEST(Sobol, ShiftedNormalsMatchPython){
    const Eigen::MatrixXd z = qmc::sobol_normals(512, 3, {12345u, 67890u, 13579u});
    EXPECT_NEAR(z(0, 0), -4.535422762090429, 1e-12);
    EXPECT_NEAR(z(0, 1), -4.161517492702865, 1e-12);
    EXPECT_NEAR(z(1, 1), 3.962224510576650e-05, 1e-15);
    EXPECT_NEAR(z(2, 0), 6.744987956063694e-01, 1e-14);
    EXPECT_NEAR(z(2, 2), -6.744798007114803e-01, 1e-14);
    EXPECT_NEAR(qmc::inv_normal_cdf(0.975), 1.959963984540054, 1e-14);
    EXPECT_NEAR(qmc::inv_normal_cdf(1e-12), -7.034483825301131, 1e-12);
    EXPECT_THROW(qmc::sobol_normals(8, 3, {1u, 2u}), InvalidInputError);
}

TEST(BrownianBridge, BothConstructionsHaveBrownianCovariance){
    for(Eigen::Index m : {1, 2, 5, 8, 13}){
        const Eigen::MatrixXd id = Eigen::MatrixXd::Identity(m, m);
        for(const Eigen::MatrixXd& a : {qmc::brownian_bridge(id, 1.5), qmc::incremental_path(id, 1.5)}){
            const Eigen::MatrixXd cov = a.transpose() * a;
            for(Eigen::Index i = 0; i < m; ++i){
                for(Eigen::Index j = 0; j < m; ++j){
                    const double ti = 1.5 * static_cast<double>(i + 1) / static_cast<double>(m);
                    const double tj = 1.5 * static_cast<double>(j + 1) / static_cast<double>(m);
                    EXPECT_NEAR(cov(i, j), std::min(ti, tj), 1e-14);
                }
            }
        }
    }
    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(1, 8);
    z(0, 0) = 2.0;
    const Eigen::MatrixXd w = qmc::brownian_bridge(z, 4.0);
    for(Eigen::Index k = 0; k < 8; ++k) EXPECT_NEAR(w(0, k), 4.0 * static_cast<double>(k + 1) / 8.0, 1e-14);
}

TEST(Asian, ClosedFormAndPayoffsMatchPython){
    EXPECT_NEAR(geometric_asian_call(100, 100, 0.05, 0.2, 1.0, 16), 5.841672354667301, 1e-12);
    EXPECT_NEAR(geometric_asian_call(100, 100, 0.05, 0.2, 1.0, 1), pricing::price({100, 100, 0.05, 0.2, 1.0, pricing::OptionType::Call}), 1e-12);
    std::vector<uint32_t> shift(8);
    for(std::size_t i = 0; i < 8; ++i) shift[i] = static_cast<uint32_t>((i + 1) * 123456789u);
    const Eigen::MatrixXd z = qmc::sobol_normals(1024, 8, shift);
    EXPECT_NEAR(asian_call_payoffs(100, 100, 0.05, 0.2, 1.0, qmc::incremental_path(z, 1.0)).mean(), 6.406010203300577, 1e-11);
    EXPECT_NEAR(asian_call_payoffs(100, 100, 0.05, 0.2, 1.0, qmc::brownian_bridge(z, 1.0)).mean(), 6.343029958639945, 1e-11);
}

TEST(Asian, QmcBeatsMcAndBridgeHelps){
    const double exact = geometric_asian_call(100, 100, 0.05, 0.2, 1.0, 32);
    auto rmse = [&](PathMethod m){
        Rng rng(11);
        double s = 0.0;
        for(int r = 0; r < 30; ++r){
            const double e = asian_call_estimate(100, 100, 0.05, 0.2, 1.0, 32, 4096, m, Average::Geometric, rng) - exact;
            s += e * e;
        }
        return std::sqrt(s / 30.0);
    };
    const double mc = rmse(PathMethod::Mc), sob = rmse(PathMethod::Sobol), br = rmse(PathMethod::SobolBridge);
    EXPECT_LT(br, sob);
    EXPECT_LT(sob, mc);
    EXPECT_LT(br, mc / 10.0);
}

TEST(Greeks, PathwiseAndLrUnbiasedLrNoisier){
    Rng rng(2);
    const std::vector<double> d = distributions::normal::sample(1000000, 0.0, 1.0, rng);
    const Eigen::VectorXd z = Eigen::Map<const Eigen::VectorXd>(d.data(), static_cast<Eigen::Index>(d.size()));
    const pricing::BSParams p{100, 105, 0.03, 0.25, 0.8, pricing::OptionType::Call};
    const PathwiseGreeks pw = european_call_pathwise(100, 105, 0.03, 0.25, 0.8, z);
    const LrGreeks lr = european_call_lr(100, 105, 0.03, 0.25, 0.8, z);
    EXPECT_NEAR(pw.delta.value, pricing::delta(p), 4 * pw.delta.std_error);
    EXPECT_NEAR(pw.vega.value, pricing::vega(p), 4 * pw.vega.std_error);
    EXPECT_NEAR(pw.rho.value, pricing::rho(p), 4 * pw.rho.std_error);
    EXPECT_NEAR(lr.delta.value, pricing::delta(p), 4 * lr.delta.std_error);
    EXPECT_NEAR(lr.vega.value, pricing::vega(p), 4 * lr.vega.std_error);
    EXPECT_GT(lr.delta.std_error, 2 * pw.delta.std_error);
    EXPECT_GT(lr.vega.std_error, 2 * pw.vega.std_error);
    const double exact = digital_call_delta(100, 100, 0.02, 0.2, 0.5);
    const GreekEstimate dl = digital_call_delta_lr(100, 100, 0.02, 0.2, 0.5, z);
    EXPECT_NEAR(dl.value, exact, 4 * dl.std_error);
    EXPECT_GT(digital_call_delta_bump(100, 100, 0.02, 0.2, 0.5, z, 0.01).std_error,
              5 * digital_call_delta_bump(100, 100, 0.02, 0.2, 0.5, z, 1.0).std_error);
}

TEST(Aad, BlackScholesGreeksExact){
    autodiff::Tape tape;
    const autodiff::Var s = tape.variable(100.0), k = tape.variable(95.0), r = tape.variable(0.03),
                        v = tape.variable(0.25), t = tape.variable(0.7);
    const autodiff::Var sq = autodiff::sqrt(t);
    const autodiff::Var d1 = (autodiff::log(s / k) + (r + 0.5 * v * v) * t) / (v * sq);
    const autodiff::Var d2 = d1 - v * sq;
    const autodiff::Var price = s * autodiff::ncdf(d1) - k * autodiff::exp(-r * t) * autodiff::ncdf(d2);
    tape.gradient(price);
    const pricing::BSParams p{100, 95, 0.03, 0.25, 0.7, pricing::OptionType::Call};
    EXPECT_NEAR(price.value(), pricing::price(p), 1e-12);
    EXPECT_NEAR(tape.adjoint(s), pricing::delta(p), 1e-13);
    EXPECT_NEAR(tape.adjoint(v), pricing::vega(p), 1e-11);
    EXPECT_NEAR(tape.adjoint(r), pricing::rho(p), 1e-11);
    EXPECT_NEAR(tape.adjoint(t), -pricing::theta(p), 1e-11);
    autodiff::Tape other;
    EXPECT_THROW(other.gradient(price), InvalidInputError);
}

TEST(Aad, BasketMatchesPathwiseAndPython){
    const Eigen::MatrixXd z = qmc::sobol_normals(512, 3, {12345u, 67890u, 13579u});
    const Eigen::VectorXd s = vec3(100, 90, 110), v = vec3(0.2, 0.3, 0.25), w = vec3(0.5, 0.3, 0.2);
    const BasketGreeks pw = basket_call_pathwise(s, v, corr3(), w, 100.0, 0.04, 0.9, z);
    const BasketGreeks ad = basket_call_aad(s, v, corr3(), w, 100.0, 0.04, 0.9, z);
    const BasketGreeks bp = basket_call_bump(s, v, corr3(), w, 100.0, 0.04, 0.9, z, 1e-5);
    EXPECT_NEAR(pw.price, 7.7980307857654285, 1e-11);
    const Eigen::VectorXd py_s = vec3(0.29440289001295006, 0.18291379672186775, 0.1160473005104673);
    const Eigen::VectorXd py_v = vec3(15.119152745159964, 7.186087323271586, 4.584211342778751);
    for(Eigen::Index i = 0; i < 3; ++i){
        EXPECT_NEAR(pw.d_spots(i), py_s(i), 1e-12);
        EXPECT_NEAR(pw.d_vols(i), py_v(i), 1e-10);
        EXPECT_NEAR(ad.d_spots(i), pw.d_spots(i), 1e-13);
        EXPECT_NEAR(ad.d_vols(i), pw.d_vols(i), 1e-11);
        EXPECT_NEAR(bp.d_spots(i), pw.d_spots(i), 1e-4);
        EXPECT_NEAR(bp.d_vols(i), pw.d_vols(i), 1e-3);
    }
    EXPECT_NEAR(pw.d_rate, 45.78273267898417, 1e-10);
    EXPECT_NEAR(pw.d_strike, -0.5086970297664907, 1e-13);
    EXPECT_NEAR(ad.price, pw.price, 1e-12);
    EXPECT_NEAR(ad.d_rate, pw.d_rate, 1e-10);
    EXPECT_NEAR(ad.d_strike, pw.d_strike, 1e-13);
    EXPECT_NEAR(bp.d_rate, pw.d_rate, 1e-3);
    Eigen::MatrixXd bad = corr3();
    bad(0, 1) = bad(1, 0) = 1.5;
    EXPECT_THROW(basket_call_aad(s, v, bad, w, 100.0, 0.04, 0.9, z), InvalidInputError);
}
