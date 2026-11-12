#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/rng.hpp"
#include "qre/distributions/normal.hpp"
#include "qre/montecarlo/qmc.hpp"
//Phase 12: Asian options, the test bed for quasi-Monte Carlo
//Call on the average of S over m dates t_k = k T / m -> each path needs m normals, so the dimension is m
//Geometric average G is lognormal -> exact price to measure error against:
//  ln G ~ N(mu, v),  mu = ln S0 + (r - sigma^2/2) T (m+1)/(2m),  v = sigma^2 T (m+1)(2m+1)/(6 m^2)
//  price = e^{-rT} (e^{mu + v/2} N(d1) - K N(d2)),  d2 = (mu - ln K)/sqrt(v),  d1 = d2 + sqrt(v)
//The arithmetic average (the traded contract) has no closed form
namespace qre::montecarlo{
    enum class Average {Arithmetic, Geometric};
    enum class PathMethod {Mc, Sobol, SobolBridge};

    namespace asian_detail{
        inline void validate(double spot, double strike, double vol, double time, Eigen::Index steps){
            if(spot <= 0.0 || strike <= 0.0 || vol <= 0.0 || time <= 0.0){
                throw InvalidInputError("spot, strike, vol and time must be positive");
            }
            if(steps < 1) throw InvalidInputError("need at least one monitoring date");
        }
    }

    inline double geometric_asian_call(double spot, double strike, double rate, double vol, double time, Eigen::Index steps){
        asian_detail::validate(spot, strike, vol, time, steps);
        const auto m = static_cast<double>(steps);
        const double mu = std::log(spot) + (rate - 0.5 * vol * vol) * time * (m + 1.0) / (2.0 * m);
        const double v = vol * vol * time * (m + 1.0) * (2.0 * m + 1.0) / (6.0 * m * m);
        const double d2 = (mu - std::log(strike)) / std::sqrt(v);
        const double d1 = d2 + std::sqrt(v);
        return std::exp(-rate * time) * (std::exp(mu + 0.5 * v) * distributions::normal::cdf(d1, 0.0, 1.0)
                                         - strike * distributions::normal::cdf(d2, 0.0, 1.0));
    }

    //Discounted payoffs from Brownian paths w (n x m) at the monitoring dates
    inline Eigen::VectorXd asian_call_payoffs(double spot, double strike, double rate, double vol, double time,
                                              const Eigen::MatrixXd& w, Average average = Average::Arithmetic){
        const Eigen::Index m = w.cols();
        asian_detail::validate(spot, strike, vol, time, m);
        const double df = std::exp(-rate * time);
        Eigen::VectorXd out(w.rows());
        for(Eigen::Index p = 0; p < w.rows(); ++p){
            double acc = 0.0;
            for(Eigen::Index k = 0; k < m; ++k){
                const double t = time * static_cast<double>(k + 1) / static_cast<double>(m);
                const double log_s = std::log(spot) + (rate - 0.5 * vol * vol) * t + vol * w(p, k);
                acc += average == Average::Arithmetic ? std::exp(log_s) : log_s;
            }
            acc /= static_cast<double>(m);
            const double a = average == Average::Arithmetic ? acc : std::exp(acc);
            out(p) = df * std::max(a - strike, 0.0);
        }
        return out;
    }

    //Normals for one estimate: pseudo-random (Phase 1 sampler) or randomly shifted Sobol
    inline Eigen::MatrixXd path_normals(std::size_t n_paths, Eigen::Index steps, PathMethod method, Rng& rng){
        if(method == PathMethod::Mc){
            const auto m = static_cast<std::size_t>(steps);
            const std::vector<double> d = distributions::normal::sample(n_paths * m, 0.0, 1.0, rng);
            Eigen::MatrixXd z(static_cast<Eigen::Index>(n_paths), steps);
            for(std::size_t i = 0; i < n_paths; ++i){
                for(std::size_t k = 0; k < m; ++k) z(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(k)) = d[i * m + k];
            }
            return z;
        }
        const auto dim = static_cast<std::size_t>(steps);
        return qmc::sobol_normals(n_paths, dim, qmc::random_shift(dim, rng));
    }

    //One unbiased price estimate from n_paths paths; repeat with fresh randomness for a standard error
    inline double asian_call_estimate(double spot, double strike, double rate, double vol, double time, Eigen::Index steps,
                                      std::size_t n_paths, PathMethod method, Average average, Rng& rng){
        asian_detail::validate(spot, strike, vol, time, steps);
        const Eigen::MatrixXd z = path_normals(n_paths, steps, method, rng);
        const Eigen::MatrixXd w = method == PathMethod::SobolBridge ? qmc::brownian_bridge(z, time) : qmc::incremental_path(z, time);
        return asian_call_payoffs(spot, strike, rate, vol, time, w, average).mean();
    }
}
