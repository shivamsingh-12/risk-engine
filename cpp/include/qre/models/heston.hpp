#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <random>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/rng.hpp"
#include "qre/estimators/empirical.hpp"
#include "qre/montecarlo/gbm_path.hpp"
#include "qre/optimization/nelder_mead.hpp"
#include "qre/pricing/black_scholes.hpp"
//Phase 9: Heston stochastic volatility, risk-neutral
//  dS = r S dt + sqrt(v) S dW1,  dv = kappa (theta - v) dt + sigma sqrt(v) dW2,  corr(dW1, dW2) = rho
//  kappa mean-reversion speed of variance, theta long-run variance, sigma vol of vol, rho spot/vol correlation,
//  v0 current variance. Feller condition 2 kappa theta >= sigma^2 keeps v > 0; otherwise v can touch 0
//Characteristic function of X = ln(S_T / F), F = S e^{rT}, "little Heston trap" form (no branch-cut jumps):
//  b = kappa - rho sigma i u, d = sqrt(b^2 + sigma^2 (i u + u^2)), g = (b - d) / (b + d)
//  A = kappa theta / sigma^2 [(b - d) T - 2 ln((1 - g e^{-dT}) / (1 - g))],  B = v0 / sigma^2 (b - d)(1 - e^{-dT}) / (1 - g e^{-dT})
//Call (Lewis): C = S - sqrt(S K e^{-rT}) / pi * int_0^inf Re[e^{i u k} psi(u - i/2)] / (u^2 + 1/4) du, k = ln(F/K)
//  integral by 16-point Gauss-Legendre on panels of width 2 until the panels stop contributing
//Monte Carlo: full-truncation Euler, v+ = max(v, 0) wherever v is used
//Calibration: Nelder-Mead on vega-weighted price errors (~ implied vol errors), kappa/theta/sigma/v0 = exp(.), rho = tanh(.)
namespace qre::models{
    struct HestonParams{
        double kappa;
        double theta;
        double sigma;
        double rho;
        double v0;
        bool feller() const { return 2.0 * kappa * theta >= sigma * sigma; }
    };

    namespace heston_detail{
        inline void validate(const HestonParams& p){
            if(p.kappa <= 0.0 || p.theta <= 0.0 || p.sigma <= 0.0 || p.v0 <= 0.0) throw InvalidInputError("kappa, theta, sigma and v0 must be positive");
            if(!(p.rho > -1.0 && p.rho < 1.0)) throw InvalidInputError("rho must be in (-1, 1)");
        }
        //Gauss-Legendre nodes/weights on [-1, 1] by Newton iteration on P_n
        inline const std::pair<std::vector<double>, std::vector<double>>& gauss_legendre16(){
            static const auto rule = []{
                constexpr int n = 16;
                std::vector<double> x(n), w(n);
                for(int i = 0; i < n; ++i){
                    double z = std::cos(std::numbers::pi * (i + 0.75) / (n + 0.5)), dp = 0.0;
                    for(int it = 0; it < 100; ++it){
                        double p0 = 1.0, p1 = 0.0;
                        for(int j = 1; j <= n; ++j){ const double p2 = p1; p1 = p0; p0 = ((2.0 * j - 1.0) * z * p1 - (j - 1.0) * p2) / j; }
                        dp = n * (z * p0 - p1) / (z * z - 1.0);
                        const double dz = p0 / dp;
                        z -= dz;
                        if(std::abs(dz) < 1e-16) break;
                    }
                    x[static_cast<std::size_t>(i)] = z;
                    w[static_cast<std::size_t>(i)] = 2.0 / ((1.0 - z * z) * dp * dp);
                }
                return std::make_pair(x, w);
            }();
            return rule;
        }
    }

    //E[exp(i u ln(S_T / F))] for complex u
    inline std::complex<double> heston_char_func(std::complex<double> u, const HestonParams& p, double T){
        const std::complex<double> i(0.0, 1.0), iu = i * u;
        const std::complex<double> b = p.kappa - p.rho * p.sigma * iu;
        const std::complex<double> d = std::sqrt(b * b + p.sigma * p.sigma * (iu + u * u));
        const std::complex<double> g = (b - d) / (b + d);
        const std::complex<double> edt = std::exp(-d * T);
        const double s2 = p.sigma * p.sigma;
        const std::complex<double> A = p.kappa * p.theta / s2 * ((b - d) * T - 2.0 * std::log((1.0 - g * edt) / (1.0 - g)));
        const std::complex<double> B = p.v0 / s2 * (b - d) * (1.0 - edt) / (1.0 - g * edt);
        return std::exp(A + B);
    }

    inline double heston_price(const HestonParams& p, double spot, double strike, double rate, double T,
                               pricing::OptionType type = pricing::OptionType::Call){
        heston_detail::validate(p);
        if(spot <= 0.0 || strike <= 0.0 || T <= 0.0) throw InvalidInputError("spot, strike and T must be positive");
        const double k = std::log(spot * std::exp(rate * T) / strike);
        const auto& [xs, ws] = heston_detail::gauss_legendre16();
        auto integrand = [&](double u){
            const std::complex<double> v = std::exp(std::complex<double>(0.0, u * k)) * heston_char_func(std::complex<double>(u, -0.5), p, T);
            return v.real() / (u * u + 0.25);
        };
        constexpr double width = 2.0, max_u = 4000.0;
        double total = 0.0;
        int quiet = 0;
        for(double a = 0.0; a < max_u; a += width){
            double panel = 0.0;
            for(std::size_t j = 0; j < xs.size(); ++j) panel += ws[j] * integrand(a + 0.5 * width * (xs[j] + 1.0));
            panel *= 0.5 * width;
            total += panel;
            quiet = (std::abs(panel) < 1e-15 * std::max(1.0, std::abs(total))) ? quiet + 1 : 0;
            if(quiet >= 3 && a > 10.0) break;
        }
        const double call = spot - std::sqrt(spot * strike * std::exp(-rate * T)) / std::numbers::pi * total;
        if(type == pricing::OptionType::Call) return call;
        return call - spot + strike * std::exp(-rate * T);
    }

    inline montecarlo::McResult heston_mc_price(const HestonParams& p, double spot, double strike, double rate, double T,
                                               std::size_t steps, std::size_t n_paths, Rng& rng,
                                               pricing::OptionType type = pricing::OptionType::Call){
        heston_detail::validate(p);
        if(steps < 1 || n_paths < 2) throw InvalidInputError("need steps >= 1 and n_paths >= 2");
        const double h = T / static_cast<double>(steps), sq = std::sqrt(1.0 - p.rho * p.rho), disc = std::exp(-rate * T);
        std::normal_distribution<double> z(0.0, 1.0);
        std::vector<double> y(n_paths);
        for(std::size_t i = 0; i < n_paths; ++i){
            double x = std::log(spot), v = p.v0;
            for(std::size_t s = 0; s < steps; ++s){
                const double z2 = z(rng.engine()), z1 = p.rho * z2 + sq * z(rng.engine());
                const double vp = std::max(v, 0.0), sv = std::sqrt(vp * h);
                x += (rate - 0.5 * vp) * h + sv * z1;
                v += p.kappa * (p.theta - vp) * h + p.sigma * sv * z2;
            }
            const double st = std::exp(x);
            y[i] = disc * (type == pricing::OptionType::Call ? std::max(st - strike, 0.0) : std::max(strike - st, 0.0));
        }
        return montecarlo::McResult{estimators::empirical::sample_mean(y), estimators::empirical::std_error(y), n_paths};
    }

    //------------------------------------------------------------ calibration
    struct HestonQuote{
        double strike;
        double maturity;
        double price;       //market call price
        double vega;        //Black-Scholes vega at the market implied vol (weights the error)
    };

    inline std::vector<HestonQuote> make_quotes(double spot, double rate, const std::vector<double>& strikes,
                                                const std::vector<double>& maturities, const std::vector<std::vector<double>>& ivs){
        std::vector<HestonQuote> q;
        for(std::size_t i = 0; i < maturities.size(); ++i)
            for(std::size_t j = 0; j < strikes.size(); ++j){
                const pricing::BSParams b{spot, strikes[j], rate, ivs[i][j], maturities[i], pricing::OptionType::Call};
                q.push_back(HestonQuote{strikes[j], maturities[i], pricing::price(b), pricing::vega(b)});
            }
        return q;
    }

    //RMS of vega-weighted price errors (approximately the implied vol RMSE)
    inline double calibration_error(const HestonParams& p, double spot, double rate, const std::vector<HestonQuote>& quotes){
        double s = 0.0;
        for(const auto& q : quotes){
            const double e = (heston_price(p, spot, q.strike, rate, q.maturity) - q.price) / q.vega;
            s += e * e;
        }
        return std::sqrt(s / static_cast<double>(quotes.size()));
    }

    inline HestonParams heston_params_from_x(const Eigen::VectorXd& x){
        return HestonParams{std::exp(x(0)), std::exp(x(1)), std::exp(x(2)), std::tanh(x(3)), std::exp(x(4))};
    }
    inline Eigen::VectorXd heston_x_from_params(const HestonParams& p){
        Eigen::VectorXd x(5);
        x << std::log(p.kappa), std::log(p.theta), std::log(p.sigma), std::atanh(p.rho), std::log(p.v0);
        return x;
    }

    struct HestonCalibration{
        HestonParams params;
        double rmse;
        bool converged;
    };

    inline HestonCalibration calibrate_heston(double spot, double rate, const std::vector<HestonQuote>& quotes,
                                              const std::vector<HestonParams>& starts = {{2.0, 0.04, 0.5, -0.5, 0.04},
                                                                                         {0.5, 0.09, 0.3, -0.7, 0.06},
                                                                                         {5.0, 0.02, 0.9, -0.2, 0.02}},
                                              int max_iter = 3000){
        if(quotes.empty()) throw InvalidInputError("need at least one quote");
        auto obj = [&](const Eigen::VectorXd& x){
            try{ return calibration_error(heston_params_from_x(x), spot, rate, quotes); }
            catch(const NumericalError&){ return 1e6; }
        };
        optimization::MinimizeResult best{Eigen::VectorXd(), std::numeric_limits<double>::infinity(), 0, false};
        for(const auto& s : starts){
            const auto r = optimization::nelder_mead(obj, heston_x_from_params(s), 0.5, 1e-8, 1e-12, max_iter);
            if(r.fx < best.fx) best = r;
        }
        return HestonCalibration{heston_params_from_x(best.x), best.fx, best.converged};
    }
}
