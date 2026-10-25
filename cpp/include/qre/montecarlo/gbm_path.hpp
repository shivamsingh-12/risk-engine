#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/rng.hpp"
#include "qre/core/types.hpp"
#include "qre/distributions/normal.hpp"
#include "qre/estimators/empirical.hpp"
#include "qre/linalg/cholesky.hpp"
#include "qre/pricing/black_scholes.hpp"
//Phase 6: Geometric Brownian Motion simulation and Monte Carlo option pricing
//Risk-neutral dynamics dS = r S dt + sigma S dW, simulated with the EXACT lognormal step
//  S(t+dt) = S(t) * exp((r - sigma^2/2) dt + sigma sqrt(dt) Z),  Z ~ N(0,1)
//so there is no time discretization error, only sampling error that shrinks like 1/sqrt(N)
//Variance reduction is optional flags on the same function, so experiments can toggle them cleanly
namespace qre::montecarlo{
    struct McOptions{
        bool antithetic = false;        //pair every draw Z with -Z and average the pair
        bool control_variate = false;   //use discounted terminal price, whose expectation is exactly spot
    };

    struct McResult{
        double price;
        double std_error;               //over independent samples (pair-averages if antithetic)
        std::size_t n_samples;          //independent samples the standard error was computed from
    };

    namespace detail{
        inline void validate_common(double spot, double vol, double time, std::size_t n_paths){
            if(spot <= 0.0) throw InvalidInputError("spot must be positive");
            if(vol <= 0.0) throw InvalidInputError("vol must be positive");
            if(time <= 0.0) throw InvalidInputError("time must be positive");
            if(n_paths < 2) throw InvalidInputError("need at least 2 paths");
        }

        //rows x n_paths matrix of standard normals. Antithetic -> second half of the columns mirrors the first
        //Reuses the Phase 1 normal sampler so all randomness flows through one seeded Rng
        inline Eigen::MatrixXd draw_normals(Eigen::Index rows, std::size_t n_paths, bool antithetic, Rng& rng){
            if(antithetic && n_paths % 2 != 0){
                throw InvalidInputError("antithetic requires an even number of paths");
            }
            const std::size_t fresh = antithetic ? n_paths / 2 : n_paths;
            const std::vector<double> draws = distributions::normal::sample(
                static_cast<std::size_t>(rows) * fresh, 0.0, 1.0, rng);
            Eigen::MatrixXd z(rows, static_cast<Eigen::Index>(n_paths));
            for(std::size_t j = 0; j < fresh; ++j){
                for(Eigen::Index i = 0; i < rows; ++i){
                    const double v = draws[j * static_cast<std::size_t>(rows) + static_cast<std::size_t>(i)];
                    z(i, static_cast<Eigen::Index>(j)) = v;
                    if(antithetic) z(i, static_cast<Eigen::Index>(fresh + j)) = -v;
                }
            }
            return z;
        }
    }

    //Single asset paths -> n_paths x (steps + 1), column 0 is spot
    inline Eigen::MatrixXd simulate_gbm_paths(double spot, double rate, double vol, double time,
                                              std::size_t steps, std::size_t n_paths, Rng& rng,
                                              bool antithetic = false){
        detail::validate_common(spot, vol, time, n_paths);
        if(steps < 1) throw InvalidInputError("need at least 1 step");
        const double dt = time / static_cast<double>(steps);
        const double drift = (rate - 0.5 * vol * vol) * dt;
        const double diffusion = vol * std::sqrt(dt);
        const Eigen::MatrixXd z = detail::draw_normals(static_cast<Eigen::Index>(steps), n_paths, antithetic, rng);
        Eigen::MatrixXd paths(static_cast<Eigen::Index>(n_paths), static_cast<Eigen::Index>(steps) + 1);
        for(Eigen::Index p = 0; p < paths.rows(); ++p){
            paths(p, 0) = spot;
            for(Eigen::Index t = 0; t < static_cast<Eigen::Index>(steps); ++t){
                paths(p, t + 1) = paths(p, t) * std::exp(drift + diffusion * z(t, p));
            }
        }
        return paths;
    }

    //Multi-asset paths with correlated shocks. Returns one n_paths x (steps + 1) matrix per asset
    //Z independent -> L*Z has covariance L L^T = corr (Phase 2 Cholesky)
    inline std::vector<Eigen::MatrixXd> simulate_correlated_gbm(
        const Eigen::VectorXd& spots, double rate, const Eigen::VectorXd& vols,
        const CorrelationMatrix& corr, double time, std::size_t steps, std::size_t n_paths,
        Rng& rng, bool antithetic = false){
        const Eigen::Index n_assets = spots.size();
        if(n_assets < 1 || vols.size() != n_assets || corr.rows() != n_assets || corr.cols() != n_assets){
            throw InvalidInputError("spots, vols and corr have incompatible dimensions");
        }
        if(time <= 0.0) throw InvalidInputError("time must be positive");
        if(steps < 1) throw InvalidInputError("need at least 1 step");
        if(n_paths < 2) throw InvalidInputError("need at least 2 paths");
        for(Eigen::Index i = 0; i < n_assets; ++i){
            if(spots(i) <= 0.0) throw InvalidInputError("spot must be positive");
            if(vols(i) <= 0.0) throw InvalidInputError("vol must be positive");
        }
        const Eigen::MatrixXd L = linalg::cholesky_decompose(corr);   //throws if not symmetric / not PD
        const double dt = time / static_cast<double>(steps);
        const double sqrt_dt = std::sqrt(dt);

        std::vector<Eigen::MatrixXd> assets(static_cast<std::size_t>(n_assets),
            Eigen::MatrixXd(static_cast<Eigen::Index>(n_paths), static_cast<Eigen::Index>(steps) + 1));
        for(Eigen::Index a = 0; a < n_assets; ++a){
            assets[static_cast<std::size_t>(a)].col(0).setConstant(spots(a));
        }
        for(std::size_t t = 0; t < steps; ++t){
            const Eigen::MatrixXd z = detail::draw_normals(n_assets, n_paths, antithetic, rng);   //assets x paths
            const Eigen::MatrixXd cz = linalg::correlated_normals(L, z);
            for(Eigen::Index a = 0; a < n_assets; ++a){
                const double drift = (rate - 0.5 * vols(a) * vols(a)) * dt;
                Eigen::MatrixXd& m = assets[static_cast<std::size_t>(a)];
                for(Eigen::Index p = 0; p < m.rows(); ++p){
                    m(p, static_cast<Eigen::Index>(t) + 1) =
                        m(p, static_cast<Eigen::Index>(t)) * std::exp(drift + vols(a) * sqrt_dt * cz(a, p));
                }
            }
        }
        return assets;
    }

    //Monte Carlo European option price. p.vol is the (true) simulation volatility
    inline McResult mc_european_price(const pricing::BSParams& p, std::size_t n_paths, Rng& rng,
                                      McOptions opts = {}){
        detail::validate_common(p.spot, p.vol, p.time, n_paths);
        if(p.strike <= 0.0) throw InvalidInputError("strike must be positive");
        const Eigen::MatrixXd z = detail::draw_normals(1, n_paths, opts.antithetic, rng);
        const double disc = std::exp(-p.rate * p.time);
        const double drift = (p.rate - 0.5 * p.vol * p.vol) * p.time;
        const double diffusion = p.vol * std::sqrt(p.time);
        const bool is_call = (p.type == pricing::OptionType::Call);

        //Per-draw discounted payoff y and control c = disc * S_T (E[c] = spot)
        std::vector<double> y(n_paths), c(n_paths);
        for(std::size_t i = 0; i < n_paths; ++i){
            const double s_t = p.spot * std::exp(drift + diffusion * z(0, static_cast<Eigen::Index>(i)));
            y[i] = disc * (is_call ? std::max(s_t - p.strike, 0.0) : std::max(p.strike - s_t, 0.0));
            c[i] = disc * s_t;
        }
        //Antithetic pair = (i, i + n/2) -> average, then treat pair-averages as the independent samples
        if(opts.antithetic){
            const std::size_t half = n_paths / 2;
            for(std::size_t i = 0; i < half; ++i){
                y[i] = 0.5 * (y[i] + y[i + half]);
                c[i] = 0.5 * (c[i] + c[i + half]);
            }
            y.resize(half);
            c.resize(half);
        }
        if(opts.control_variate){
            const double mean_y = estimators::empirical::sample_mean(y);
            const double mean_c = estimators::empirical::sample_mean(c);
            double cov_yc = 0.0;
            for(std::size_t i = 0; i < y.size(); ++i) cov_yc += (y[i] - mean_y) * (c[i] - mean_c);
            cov_yc /= static_cast<double>(y.size() - 1);
            const double beta = cov_yc / estimators::empirical::sample_var(c);   //optimal coefficient
            for(std::size_t i = 0; i < y.size(); ++i) y[i] -= beta * (c[i] - p.spot);
        }
        return McResult{estimators::empirical::sample_mean(y), estimators::empirical::std_error(y), y.size()};
    }
}
