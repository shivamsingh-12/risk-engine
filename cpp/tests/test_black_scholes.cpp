#include <gtest/gtest.h>
#include <cmath>
#include "qre/pricing/black_scholes.hpp"
using namespace qre;
using qre::pricing::BSParams;
using qre::pricing::OptionType;

namespace {
BSParams base(OptionType t = OptionType::Call){
    return BSParams{100.0, 100.0, 0.05, 0.2, 1.0, t};
}
//Central difference of price wrt a chosen field
template <typename Setter>
double fd(const BSParams& p, Setter set, double h){
    BSParams up = p, dn = p;
    set(up, h);
    set(dn, -h);
    return (pricing::price(up) - pricing::price(dn)) / (2.0 * h);
}
}

TEST(BlackScholes, KnownValues){
    EXPECT_NEAR(pricing::price(base(OptionType::Call)), 10.450583572185565, 1e-8);
    EXPECT_NEAR(pricing::price(base(OptionType::Put)), 5.573526022256971, 1e-8);
}

TEST(BlackScholes, PutCallParity){
    const BSParams cases[] = {
        {100, 100, 0.05, 0.2, 1.0, OptionType::Call},
        {80, 100, 0.01, 0.4, 0.5, OptionType::Call},
        {120, 90, 0.03, 0.15, 2.0, OptionType::Call},
        {50, 55, 0.0, 0.3, 0.25, OptionType::Call},
    };
    for(BSParams p : cases){
        const double c = pricing::price(p);
        p.type = OptionType::Put;
        const double put = pricing::price(p);
        EXPECT_NEAR(c - put, p.spot - p.strike * std::exp(-p.rate * p.time), 1e-10);
    }
}

TEST(BlackScholes, DeepItmAndOtmCall){
    BSParams itm = base(); itm.spot = 500.0;
    EXPECT_NEAR(pricing::price(itm), 500.0 - 100.0 * std::exp(-0.05), 1e-6);
    BSParams otm = base(); otm.spot = 20.0;
    EXPECT_NEAR(pricing::price(otm), 0.0, 1e-10);
}

TEST(BlackScholes, ShortExpiryApproachesIntrinsic){
    BSParams p = base(); p.time = 1e-8; p.spot = 110.0;
    EXPECT_NEAR(pricing::price(p), 10.0, 1e-4);
    p.spot = 90.0;
    EXPECT_NEAR(pricing::price(p), 0.0, 1e-6);
}

TEST(BlackScholes, GreeksMatchFiniteDifferences){
    for(OptionType t : {OptionType::Call, OptionType::Put}){
        const BSParams p = base(t);
        EXPECT_NEAR(pricing::delta(p), fd(p, [](BSParams& q, double h){q.spot += h;}, 1e-4), 1e-6);
        EXPECT_NEAR(pricing::vega(p), fd(p, [](BSParams& q, double h){q.vol += h;}, 1e-5), 1e-5);
        EXPECT_NEAR(pricing::rho(p), fd(p, [](BSParams& q, double h){q.rate += h;}, 1e-5), 1e-4);
        //theta = dV/dt = -dV/dT
        EXPECT_NEAR(pricing::theta(p), -fd(p, [](BSParams& q, double h){q.time += h;}, 1e-5), 1e-4);
    }
}

TEST(BlackScholes, GammaMatchesSecondDifference){
    const BSParams p = base();
    const double h = 1e-3;
    BSParams up = p, dn = p;
    up.spot += h; dn.spot -= h;
    const double second = (pricing::price(up) - 2.0 * pricing::price(p) + pricing::price(dn)) / (h * h);
    EXPECT_NEAR(pricing::gamma(p), second, 1e-5);
}

TEST(BlackScholes, DeltaPutCallRelation){
    EXPECT_NEAR(pricing::delta(base(OptionType::Call)) - pricing::delta(base(OptionType::Put)), 1.0, 1e-12);
}

TEST(ImpliedVol, RoundTrip){
    for(OptionType t : {OptionType::Call, OptionType::Put}){
        for(double vol : {0.05, 0.2, 0.6, 1.5}){
            BSParams p = base(t); p.vol = vol;
            const double mkt = pricing::price(p);
            auto r = pricing::implied_vol(mkt, p);
            ASSERT_TRUE(r.has_value());
            EXPECT_TRUE(r->converged);
            EXPECT_NEAR(r->root, vol, 1e-6);
        }
    }
}

TEST(ImpliedVol, RejectsArbitragePrices){
    const BSParams p = base();
    EXPECT_THROW(pricing::implied_vol(0.0, p), InvalidInputError);
    EXPECT_THROW(pricing::implied_vol(p.spot + 1.0, p), InvalidInputError);
}

TEST(BlackScholes, InvalidInputsThrow){
    BSParams p = base(); p.spot = -1.0;
    EXPECT_THROW(pricing::price(p), InvalidInputError);
    p = base(); p.strike = 0.0;
    EXPECT_THROW(pricing::price(p), InvalidInputError);
    p = base(); p.vol = 0.0;
    EXPECT_THROW(pricing::price(p), InvalidInputError);
    p = base(); p.time = 0.0;
    EXPECT_THROW(pricing::price(p), InvalidInputError);
}
