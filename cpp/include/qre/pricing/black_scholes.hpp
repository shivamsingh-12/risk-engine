#pragma once
#include <algorithm>
#include <cmath>
#include "qre/core/config.hpp"
#include "qre/core/exceptions.hpp"
#include "qre/core/result.hpp"
#include "qre/distributions/normal.hpp"
#include "qre/optimization/calibration.hpp"
//Phase 5: Black-Scholes pricing, Greeks, implied volatility for European options (no dividends)
//Conventions: vega per 1.00 change in vol, theta per year (dV/dt), rho per 1.00 change in rate
namespace qre::pricing{
    enum class OptionType {Call, Put};

    struct BSParams{
        double spot;
        double strike;
        double rate;
        double vol;
        double time;
        OptionType type;
    };

    namespace detail{
        //Standard normal helpers -> reuse Phase 1 distribution rather than re-deriving cdf/pdf
        inline double N(double x){return distributions::normal::cdf(x, 0.0, 1.0);}
        inline double phi(double x){return distributions::normal::pdf(x, 0.0, 1.0);}

        inline void validate(const BSParams& p){
            if(p.spot <= 0.0) throw InvalidInputError("spot must be positive");
            if(p.strike <= 0.0) throw InvalidInputError("strike must be positive");
            if(p.vol <= 0.0) throw InvalidInputError("vol must be positive");
            if(p.time <= 0.0) throw InvalidInputError("time must be positive");
        }

        inline double d1(const BSParams& p){
            return (std::log(p.spot / p.strike) + (p.rate + 0.5 * p.vol * p.vol) * p.time)
                   / (p.vol * std::sqrt(p.time));
        }
        inline double d2(const BSParams& p){
            return d1(p) - p.vol * std::sqrt(p.time);
        }
    }

    //Call = S*N(d1) - K*e^(-rT)*N(d2)    Put = K*e^(-rT)*N(-d2) - S*N(-d1)
    inline double price(const BSParams& p){
        detail::validate(p);
        const double a = detail::d1(p);
        const double b = detail::d2(p);
        const double disc = std::exp(-p.rate * p.time);
        if(p.type == OptionType::Call){
            return p.spot * detail::N(a) - p.strike * disc * detail::N(b);
        }
        return p.strike * disc * detail::N(-b) - p.spot * detail::N(-a);
    }

    //Sensitivity of price to spot
    inline double delta(const BSParams& p){
        detail::validate(p);
        const double n = detail::N(detail::d1(p));
        return p.type == OptionType::Call ? n : n - 1.0;
    }

    //Second derivative wrt spot -> identical for calls and puts
    inline double gamma(const BSParams& p){
        detail::validate(p);
        return detail::phi(detail::d1(p)) / (p.spot * p.vol * std::sqrt(p.time));
    }

    //Sensitivity to vol -> identical for calls and puts
    inline double vega(const BSParams& p){
        detail::validate(p);
        return p.spot * detail::phi(detail::d1(p)) * std::sqrt(p.time);
    }

    //Time decay: dV/dt, negative for long options in normal conditions
    inline double theta(const BSParams& p){
        detail::validate(p);
        const double a = detail::d1(p);
        const double b = detail::d2(p);
        const double disc = std::exp(-p.rate * p.time);
        const double decay = -p.spot * detail::phi(a) * p.vol / (2.0 * std::sqrt(p.time));
        if(p.type == OptionType::Call){
            return decay - p.rate * p.strike * disc * detail::N(b);
        }
        return decay + p.rate * p.strike * disc * detail::N(-b);
    }

    //Sensitivity to the risk-free rate
    inline double rho(const BSParams& p){
        detail::validate(p);
        const double b = detail::d2(p);
        const double disc = std::exp(-p.rate * p.time);
        if(p.type == OptionType::Call){
            return p.strike * p.time * disc * detail::N(b);
        }
        return -p.strike * p.time * disc * detail::N(-b);
    }

    //Implied vol = calibration problem: find sigma such that BS(sigma) = market_price
    //p.vol is ignored (it's the unknown). Prices outside no-arbitrage bounds have no solution -> throw
    //Non-convergence is an expected failure -> returned via Result
    inline Result<optimization::RootResult> implied_vol(
        double market_price,
        const BSParams& p,
        double lower_vol = 1e-6,
        double upper_vol = 5.0,
        double tolerance = config::kDefTolerance,
        int max_iterations = config::kDefMaxIterations
    ) {
        BSParams probe = p;
        probe.vol = 1.0; //placeholder to pass validation of the other inputs
        detail::validate(probe);
        const double disc = std::exp(-p.rate * p.time);
        double lo_bound = 0.0;
        double hi_bound = 0.0;
        if(p.type == OptionType::Call){
            lo_bound = std::max(p.spot - p.strike * disc, 0.0);
            hi_bound = p.spot;
        } else {
            lo_bound = std::max(p.strike * disc - p.spot, 0.0);
            hi_bound = p.strike * disc;
        }
        if(!(market_price > lo_bound && market_price < hi_bound)){
            throw InvalidInputError("market price violates no-arbitrage bounds");
        }
        auto model = [probe](double sigma){
            BSParams q = probe;
            q.vol = sigma;
            return price(q);
        };
        return optimization::calibrate_brent(model, market_price, lower_vol, upper_vol, tolerance, max_iterations);
    }
}
