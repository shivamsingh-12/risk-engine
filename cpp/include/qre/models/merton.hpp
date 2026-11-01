#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include "qre/core/exceptions.hpp"
#include "qre/core/rng.hpp"
#include "qre/estimators/empirical.hpp"
#include "qre/montecarlo/gbm_path.hpp"
#include "qre/pricing/black_scholes.hpp"
//Phase 8: Merton (1976) jump-diffusion, risk-neutral
//  dS/S = (r - lam k) dt + sigma dW + (J - 1) dN,  N Poisson with intensity lam (jumps per year),
//  ln J ~ Normal(mu_j, delta^2), k = E[J] - 1 = exp(mu_j + delta^2/2) - 1 (the -lam k term keeps e^{-rt} S a martingale)
//Terminal: ln S_T = ln S + (r - lam k - sigma^2/2) T + sigma W_T + (sum of N_T log-jumps ~ Normal(N mu_j, N delta^2))
//Closed form: sum_n e^{-lam' T} (lam' T)^n / n! * BS(S, K, r_n, sigma_n, T)
//  lam' = lam (1 + k), sigma_n^2 = sigma^2 + n delta^2 / T, r_n = r - lam k + n ln(1 + k) / T;  lam = 0 is Black-Scholes
namespace qre::models{
    struct MertonParams{
        double spot;
        double strike;
        double rate;
        double vol;         //diffusion volatility
        double time;
        double lam;         //jump intensity per year
        double mu_j;        //mean log jump
        double delta;       //sd of log jump
        pricing::OptionType type = pricing::OptionType::Call;
        double k() const { return std::exp(mu_j + 0.5 * delta * delta) - 1.0; }
    };

    namespace detail{
        inline void validate(const MertonParams& p){
            if(p.spot <= 0.0 || p.strike <= 0.0 || p.vol <= 0.0 || p.time <= 0.0) throw InvalidInputError("spot, strike, vol and time must be positive");
            if(p.lam < 0.0 || p.delta < 0.0) throw InvalidInputError("lam and delta must be non-negative");
        }
    }

    inline double merton_price(const MertonParams& p, double tol = 1e-15, int max_terms = 200){
        detail::validate(p);
        pricing::BSParams bs{p.spot, p.strike, p.rate, p.vol, p.time, p.type};
        if(p.lam == 0.0) return pricing::price(bs);
        const double k = p.k(), lam_prime = p.lam * (1.0 + k), lt = lam_prime * p.time;
        double total = 0.0;
        for(int n = 0; n < max_terms; ++n){
            const double dn = static_cast<double>(n);
            const double weight = std::exp(-lt + dn * std::log(lt) - std::lgamma(dn + 1.0));
            bs.vol = std::sqrt(p.vol * p.vol + dn * p.delta * p.delta / p.time);
            bs.rate = p.rate - p.lam * k + dn * std::log1p(k) / p.time;
            total += weight * pricing::price(bs);
            if(dn > lt && weight < tol) break;
        }
        return total;
    }

    inline std::vector<double> simulate_merton_terminal(const MertonParams& p, std::size_t n_paths, Rng& rng){
        detail::validate(p);
        std::poisson_distribution<int> poisson(p.lam * p.time);
        std::normal_distribution<double> z(0.0, 1.0);
        const double drift = (p.rate - p.lam * p.k() - 0.5 * p.vol * p.vol) * p.time;
        const double diff = p.vol * std::sqrt(p.time);
        std::vector<double> out(n_paths);
        for(std::size_t i = 0; i < n_paths; ++i){
            const int nj = p.lam > 0.0 ? poisson(rng.engine()) : 0;
            const double jumps = nj * p.mu_j + std::sqrt(static_cast<double>(nj)) * p.delta * z(rng.engine());
            out[i] = p.spot * std::exp(drift + diff * z(rng.engine()) + jumps);
        }
        return out;
    }

    inline montecarlo::McResult merton_mc_price(const MertonParams& p, std::size_t n_paths, Rng& rng){
        if(n_paths < 2) throw InvalidInputError("need at least 2 paths");
        const auto s = simulate_merton_terminal(p, n_paths, rng);
        const double disc = std::exp(-p.rate * p.time);
        std::vector<double> y(n_paths);
        for(std::size_t i = 0; i < n_paths; ++i){
            y[i] = disc * (p.type == pricing::OptionType::Call ? std::max(s[i] - p.strike, 0.0) : std::max(p.strike - s[i], 0.0));
        }
        return montecarlo::McResult{estimators::empirical::sample_mean(y), estimators::empirical::std_error(y), n_paths};
    }
}
