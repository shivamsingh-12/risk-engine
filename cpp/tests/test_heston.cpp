#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "qre/models/heston.hpp"
#include "qre/pricing/black_scholes.hpp"
using namespace qre;
using qre::models::HestonParams;
using qre::pricing::OptionType;

namespace {
//Expected values from the Python reference (scipy quad on the same Lewis integral; it also agrees with the
//original 1993 two-probability Heston formula to 1e-7 in the Python tests)
const HestonParams P{2.0, 0.04, 0.5, -0.7, 0.05};            //Feller fails
const HestonParams PF{3.0, 0.05, 0.3, -0.6, 0.04};           //Feller holds
}

TEST(Heston, CharFuncNormalizationMartingaleAndPython){
    using C = std::complex<double>;
    for(const auto& p : {P, PF}){
        EXPECT_LT(std::abs(models::heston_char_func(C(0.0, 0.0), p, 1.0) - 1.0), 1e-14);
        EXPECT_LT(std::abs(models::heston_char_func(C(0.0, -1.0), p, 1.0) - 1.0), 1e-12);
    }
    const C v = models::heston_char_func(C(1.3, -0.5), P, 1.0);
    EXPECT_NEAR(v.real(), 0.9608444999116402, 1e-14);
    EXPECT_NEAR(v.imag(), 0.0048226516774443964, 1e-14);
}

TEST(Heston, PricesMatchPython){
    struct Case{ const HestonParams* p; double K, T, price; };
    const Case cases[] = {
        {&P, 80.0, 0.25, 20.664771037903378}, {&P, 100.0, 1.0, 8.7311146801908}, {&P, 125.0, 2.0, 3.0072413875097794},
        {&P, 100.0, 0.05, 2.0159853810189787}, {&PF, 80.0, 0.25, 20.52254195407957}, {&PF, 100.0, 1.0, 9.41322689957775},
        {&PF, 125.0, 2.0, 5.052265174272748}, {&PF, 100.0, 0.05, 1.842477636944352},
    };
    for(const auto& c : cases) EXPECT_NEAR(models::heston_price(*c.p, 100.0, c.K, 0.02, c.T), c.price, 1e-8) << c.K << "," << c.T;
    EXPECT_NEAR(models::heston_price(P, 100.0, 90.0, 0.03, 0.75, OptionType::Put), 2.8470208159103976, 1e-8);
}

TEST(Heston, DeterministicVarianceLimitIsBlackScholes){
    const double k = 1.5, th = 0.04, v0 = 0.09, T = 1.0;
    const HestonParams p{k, th, 1e-3, 0.0, v0};
    const double vbar = th + (v0 - th) * (1.0 - std::exp(-k * T)) / (k * T);
    for(double K : {80.0, 100.0, 120.0})
        EXPECT_NEAR(models::heston_price(p, 100.0, K, 0.03, T), pricing::price({100.0, K, 0.03, std::sqrt(vbar), T, OptionType::Call}), 1e-5);
}

TEST(Heston, PutCallParity){
    const double c = models::heston_price(P, 100.0, 90.0, 0.03, 0.75);
    const double p = models::heston_price(P, 100.0, 90.0, 0.03, 0.75, OptionType::Put);
    EXPECT_NEAR(c - p, 100.0 - 90.0 * std::exp(-0.03 * 0.75), 1e-10);
}

TEST(Heston, MonteCarloConvergesToSemiClosedForm){
    for(const auto& p : {P, PF}){
        for(double K : {85.0, 100.0, 115.0}){
            Rng rng(static_cast<uint64_t>(K));
            const auto r = models::heston_mc_price(p, 100.0, K, 0.02, 1.0, 250, 100000, rng);
            EXPECT_LT(std::abs(r.price - models::heston_price(p, 100.0, K, 0.02, 1.0)), 4.0 * r.std_error + 0.02) << K;
        }
    }
}

TEST(Heston, RhoControlsSkew){
    auto iv = [](const HestonParams& p, double K){
        const auto r = pricing::implied_vol(models::heston_price(p, 100.0, K, 0.0, 1.0), {100.0, K, 0.0, 0.2, 1.0, OptionType::Call});
        return r->root;
    };
    const HestonParams neg{2.0, 0.04, 0.5, -0.7, 0.04}, pos{2.0, 0.04, 0.5, 0.7, 0.04};
    EXPECT_GT(iv(neg, 80.0), iv(neg, 100.0));
    EXPECT_GT(iv(neg, 100.0), iv(neg, 125.0));
    EXPECT_LT(iv(pos, 80.0), iv(pos, 100.0));
    EXPECT_LT(iv(pos, 100.0), iv(pos, 125.0));
}

TEST(Heston, CalibrationRecoversNoiselessSurface){
    const HestonParams truth{1.5, 0.05, 0.6, -0.65, 0.035};
    const std::vector<double> strikes{80.0, 90.0, 100.0, 110.0, 120.0}, mats{0.25, 1.0, 2.0};
    std::vector<std::vector<double>> ivs;
    for(double T : mats){
        std::vector<double> row;
        for(double K : strikes){
            const auto r = pricing::implied_vol(models::heston_price(truth, 100.0, K, 0.02, T), {100.0, K, 0.02, 0.2, T, OptionType::Call});
            row.push_back(r->root);
        }
        ivs.push_back(row);
    }
    const auto quotes = models::make_quotes(100.0, 0.02, strikes, mats, ivs);
    const auto fit = models::calibrate_heston(100.0, 0.02, quotes);
    EXPECT_LT(fit.rmse, 1e-4);
    EXPECT_NEAR(fit.params.rho, truth.rho, 0.05);
    EXPECT_NEAR(fit.params.v0, truth.v0, 0.05 * truth.v0);
    EXPECT_NEAR(fit.params.sigma, truth.sigma, 0.2 * truth.sigma);
}

TEST(Heston, InvalidInputsAndFeller){
    EXPECT_TRUE(PF.feller());
    EXPECT_FALSE(P.feller());
    EXPECT_THROW(models::heston_price({2.0, 0.04, 0.5, -1.0, 0.04}, 100, 100, 0, 1), InvalidInputError);
    EXPECT_THROW(models::heston_price({-2.0, 0.04, 0.5, -0.5, 0.04}, 100, 100, 0, 1), InvalidInputError);
    EXPECT_THROW(models::heston_price(P, 100, 100, 0, 0), InvalidInputError);
    const auto x = models::heston_x_from_params(P);
    const auto q = models::heston_params_from_x(x);
    EXPECT_NEAR(q.kappa, P.kappa, 1e-14); EXPECT_NEAR(q.rho, P.rho, 1e-14); EXPECT_NEAR(q.v0, P.v0, 1e-15);
}
