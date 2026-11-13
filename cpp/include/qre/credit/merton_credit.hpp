#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>
#include "qre/core/exceptions.hpp"
#include "qre/optimization/root_finding.hpp"
//Phase 16: the Merton (1974) structural model of credit risk
//Assets dV/V = mu dt + sigma dW, one zero-coupon debt of face D due at T
//  equity E = V N(d1) - D e^{-rT} N(d2) (a call on the assets),  debt B = V - E = D e^{-rT} - put
//  d1 = (ln(V/D) + (r + sigma^2/2) T) / (sigma sqrt T),  d2 = d1 - sigma sqrt T
//  risk-neutral PD N(-d2); distance to default DD = (ln(V/D) + (mu - sigma^2/2) T) / (sigma sqrt T), PD = N(-DD)
//  spread s = -(1/T) ln(N(d2) + V e^{rT}/D N(-d1));  recovery E_Q[V_T | V_T < D]/D = V e^{rT} N(-d1) / (D N(-d2))
//Only equity is observed: sigma_E E = N(d1) sigma V. solve_assets finds (V, sigma) from (E, sigma_E) with nested
//bracketed roots (V on [E, E + D e^{-rT}], sigma on (0, sigma_E]); kmv_iterative finds sigma from a daily equity
//series as the fixed point "sigma = volatility of the asset path implied with sigma" (Vassalou-Xing / KMV)
//Black-Cox first passage, barrier K < V, nu = mu - sigma^2/2, b = ln(K/V):
//  P(tau <= T) = N((b - nu T)/(sigma sqrt T)) + (K/V)^(2 nu/sigma^2) N((b + nu T)/(sigma sqrt T))
namespace qre::credit{
    namespace merton_detail{
        inline double N(double x){ return 0.5 * std::erfc(-x / std::numbers::sqrt2); }
        inline void check(double V, double D, double sigma, double T){
            if(!(V > 0.0 && D > 0.0 && sigma > 0.0 && T > 0.0)) throw InvalidInputError("need V, D, sigma and T positive");
        }
        inline double root(const std::function<double(double)>& f, double a, double b, double tol){
            const auto r = optimization::brent(f, a, b, tol, 400);
            if(!r || !r->converged) throw NumericalError("Merton: root search did not converge");
            return r->root;
        }
    }

    struct D1D2{ double d1, d2; };

    inline D1D2 d1_d2(double V, double D, double r, double sigma, double T){
        merton_detail::check(V, D, sigma, T);
        const double s = sigma * std::sqrt(T);
        const double d1 = (std::log(V / D) + (r + 0.5 * sigma * sigma) * T) / s;
        return {d1, d1 - s};
    }

    inline double equity_value(double V, double D, double r, double sigma, double T){
        const auto [d1, d2] = d1_d2(V, D, r, sigma, T);
        return V * merton_detail::N(d1) - D * std::exp(-r * T) * merton_detail::N(d2);
    }

    inline double debt_value(double V, double D, double r, double sigma, double T){
        const auto [d1, d2] = d1_d2(V, D, r, sigma, T);
        return V * merton_detail::N(-d1) + D * std::exp(-r * T) * merton_detail::N(d2);
    }

    inline double credit_spread(double V, double D, double r, double sigma, double T){
        const auto [d1, d2] = d1_d2(V, D, r, sigma, T);
        //B e^{rT} / D <= 1 exactly; clamp the rounding that can push it a hair above 1 for very safe firms
        return std::max(0.0, -std::log(merton_detail::N(d2) + V * std::exp(r * T) / D * merton_detail::N(-d1)) / T);
    }

    inline double risk_neutral_pd(double V, double D, double r, double sigma, double T){
        return merton_detail::N(-d1_d2(V, D, r, sigma, T).d2);
    }

    inline double distance_to_default(double V, double D, double mu, double sigma, double T){
        merton_detail::check(V, D, sigma, T);
        return (std::log(V / D) + (mu - 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    }

    inline double physical_pd(double V, double D, double mu, double sigma, double T){
        return merton_detail::N(-distance_to_default(V, D, mu, sigma, T));
    }

    inline double recovery_rate(double V, double D, double r, double sigma, double T){
        const auto [d1, d2] = d1_d2(V, D, r, sigma, T);
        return V * std::exp(r * T) * merton_detail::N(-d1) / (D * merton_detail::N(-d2));
    }

    inline double equity_volatility(double V, double D, double r, double sigma, double T){
        return merton_detail::N(d1_d2(V, D, r, sigma, T).d1) * sigma * V / equity_value(V, D, r, sigma, T);
    }

    //V such that the equity (call) value is E
    inline double asset_value(double E, double D, double r, double sigma, double T){
        if(!(E > 0.0)) throw InvalidInputError("equity value must be positive");
        const double lo = E, hi = E + D * std::exp(-r * T);
        auto f = [&](double v){ return equity_value(v, D, r, sigma, T) - E; };
        if(f(hi) <= 0.0) return hi;
        return merton_detail::root(f, lo, hi, 1e-13 * std::max(E, 1.0));
    }

    struct AssetSolution{ double V, sigma; };

    inline AssetSolution solve_assets(double E, double sigma_E, double D, double r, double T){
        if(!(E > 0.0 && sigma_E > 0.0 && D > 0.0 && T > 0.0)) throw InvalidInputError("need E, sigma_E, D and T positive");
        auto g = [&](double s){
            const double v = asset_value(E, D, r, s, T);
            return merton_detail::N(d1_d2(v, D, r, s, T).d1) * s * v - sigma_E * E;
        };
        const double lo = 1e-8;
        if(g(lo) >= 0.0) throw InvalidInputError("no asset volatility reproduces this equity volatility");
        const double s = merton_detail::root(g, lo, sigma_E, 1e-14 * std::max(E, 1.0));
        return {asset_value(E, D, r, s, T), s};
    }

    struct KmvResult{
        std::vector<double> assets;
        double sigma, mu;
        int iterations;
        double distance_to_default, pd;
    };

    inline KmvResult kmv_iterative(const std::vector<double>& equity, double D, double r, double T = 1.0, int trading_days = 252,
                                   double tol = 1e-10, int max_iter = 200){
        if(equity.size() < 20) throw InvalidInputError("need at least 20 positive equity values");
        for(double e : equity) if(!(e > 0.0)) throw InvalidInputError("need at least 20 positive equity values");
        const double ann = static_cast<double>(trading_days);
        auto log_vol = [&](const std::vector<double>& x){
            const std::size_t n = x.size() - 1;
            double m = 0.0, s = 0.0;
            for(std::size_t i = 0; i < n; ++i) m += std::log(x[i + 1] / x[i]);
            m /= static_cast<double>(n);
            for(std::size_t i = 0; i < n; ++i){ const double d = std::log(x[i + 1] / x[i]) - m; s += d * d; }
            return std::sqrt(s / static_cast<double>(n - 1) * ann);
        };
        double sigma = log_vol(equity) * equity.back() / (equity.back() + D);
        std::vector<double> V(equity.size());
        int it = 1;
        for(;; ++it){
            if(it > max_iter) throw NumericalError("KMV iteration did not converge");
            for(std::size_t i = 0; i < equity.size(); ++i) V[i] = asset_value(equity[i], D, r, sigma, T);
            const double next = log_vol(V);
            const bool done = std::abs(next - sigma) < tol;
            sigma = next;
            if(done) break;
        }
        for(std::size_t i = 0; i < equity.size(); ++i) V[i] = asset_value(equity[i], D, r, sigma, T);
        const double mu = std::log(V.back() / V.front()) / static_cast<double>(V.size() - 1) * ann + 0.5 * sigma * sigma;
        const double dd = distance_to_default(V.back(), D, mu, sigma, T);
        return {V, sigma, mu, it, dd, merton_detail::N(-dd)};
    }

    inline double black_cox_pd(double V, double K, double mu, double sigma, double T){
        if(!(K > 0.0 && K < V && sigma > 0.0 && T > 0.0)) throw InvalidInputError("need 0 < K < V, sigma > 0, T > 0");
        const double nu = mu - 0.5 * sigma * sigma, b = std::log(K / V), s = sigma * std::sqrt(T);
        return merton_detail::N((b - nu * T) / s) + std::pow(K / V, 2.0 * nu / (sigma * sigma)) * merton_detail::N((b + nu * T) / s);
    }

    //Monte Carlo first passage with daily-style monitoring and the Broadie-Glasserman-Kou continuity correction
    //(barrier moved towards the path by exp(0.5826 sigma sqrt(dt)) to make up for crossings between dates)
    inline double first_passage_mc(double V, double K, double mu, double sigma, double T, std::size_t n_paths, int n_steps, uint64_t seed){
        if(!(K > 0.0 && K < V && sigma > 0.0 && T > 0.0) || n_paths < 1 || n_steps < 1) throw InvalidInputError("bad first-passage inputs");
        std::mt19937_64 eng(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        const double dt = T / n_steps, k = std::log(K / V) + 0.5826 * sigma * std::sqrt(dt);
        const double drift = (mu - 0.5 * sigma * sigma) * dt, vol = sigma * std::sqrt(dt);
        std::size_t hits = 0;
        for(std::size_t p = 0; p < n_paths; ++p){
            double x = 0.0;
            for(int s = 0; s < n_steps; ++s){
                x += drift + vol * nd(eng);
                if(x <= k){ ++hits; break; }
            }
        }
        return static_cast<double>(hits) / static_cast<double>(n_paths);
    }
}
