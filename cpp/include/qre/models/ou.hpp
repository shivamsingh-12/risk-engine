#pragma once
#include <cmath>
#include <numbers>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/rng.hpp"
#include "qre/distributions/normal.hpp"
//Phase 8: Ornstein-Uhlenbeck (Vasicek) mean-reverting process
//  dX = kappa (theta - X) dt + sigma dW,  kappa > 0 speed of mean reversion, theta long-run mean
//Exact transition (no discretization error):
//  X_{t+dt} = theta + (X_t - theta) e^{-kappa dt} + sigma sqrt((1 - e^{-2 kappa dt}) / (2 kappa)) Z
//Stationary distribution Normal(theta, sigma^2 / (2 kappa)); half-life of a deviation ln 2 / kappa
//MLE: the transition is an AR(1) X_{t+1} = a + b X_t + e with b = e^{-kappa dt} -> least squares gives
//  kappa = -ln(b) / dt, theta = a / (1 - b), sigma = sd(e) sqrt(2 kappa / (1 - b^2))  (sd with divisor n)
//Caveat: on a random walk the least-squares slope is biased below 1, so the fit returns a small spurious
//kappa rather than failing -> a fitted kappa is not proof of mean reversion
namespace qre::models{
    inline Eigen::MatrixXd simulate_ou(double x0, double kappa, double theta, double sigma, double dt,
                                       Eigen::Index steps, Eigen::Index n_paths, Rng& rng){
        if(kappa <= 0.0 || sigma <= 0.0 || dt <= 0.0) throw InvalidInputError("need kappa > 0, sigma > 0 and dt > 0");
        if(steps < 1 || n_paths < 1) throw InvalidInputError("need steps >= 1 and n_paths >= 1");
        const double decay = std::exp(-kappa * dt);
        const double sd = sigma * std::sqrt((1.0 - decay * decay) / (2.0 * kappa));
        const auto z = distributions::normal::sample(static_cast<std::size_t>(n_paths * steps), 0.0, 1.0, rng);
        Eigen::MatrixXd x(n_paths, steps + 1);
        x.col(0).setConstant(x0);
        for(Eigen::Index p = 0; p < n_paths; ++p)
            for(Eigen::Index k = 0; k < steps; ++k)
                x(p, k + 1) = theta + (x(p, k) - theta) * decay + sd * z[static_cast<std::size_t>(p * steps + k)];
        return x;
    }

    struct OuFit{
        double kappa;
        double theta;
        double sigma;
        double half_life() const { return std::numbers::ln2 / kappa; }
        double stationary_sd() const { return sigma / std::sqrt(2.0 * kappa); }
    };

    inline OuFit fit_ou(const std::vector<double>& x, double dt){
        if(x.size() < 3 || dt <= 0.0) throw InvalidInputError("need at least 3 observations and dt > 0");
        const std::size_t n = x.size() - 1;
        double m0 = 0.0, m1 = 0.0;
        for(std::size_t i = 0; i < n; ++i){ m0 += x[i]; m1 += x[i + 1]; }
        m0 /= static_cast<double>(n); m1 /= static_cast<double>(n);
        double sxx = 0.0, sxy = 0.0;
        for(std::size_t i = 0; i < n; ++i){ sxx += (x[i] - m0) * (x[i] - m0); sxy += (x[i] - m0) * (x[i + 1] - m1); }
        if(sxx <= 0.0) throw InvalidInputError("series is constant");
        const double b = sxy / sxx, a = m1 - b * m0;
        if(!(b > 0.0 && b < 1.0)) throw InvalidInputError("no mean reversion detected (AR(1) slope outside (0, 1))");
        double s2 = 0.0;
        for(std::size_t i = 0; i < n; ++i){ const double e = x[i + 1] - a - b * x[i]; s2 += e * e; }
        s2 /= static_cast<double>(n);
        const double kappa = -std::log(b) / dt;
        return OuFit{kappa, a / (1.0 - b), std::sqrt(s2 * 2.0 * kappa / (1.0 - b * b))};
    }
}
