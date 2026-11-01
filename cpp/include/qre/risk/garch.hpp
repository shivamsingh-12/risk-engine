#pragma once
#include <cmath>
#include <limits>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/optimization/nelder_mead.hpp"
#include "qre/risk/var_es.hpp"
//GARCH(1,1) on zero-mean returns:
//  r_t = sqrt(h_t) z_t,  h_{t+1} = omega + alpha r_t^2 + beta h_t,  long-run variance v = omega / (1 - alpha - beta)
//EWMA is the special case omega = 0, alpha = 1 - lambda, beta = lambda (persistence 1, no mean reversion)
//Estimation (matches python_reference/qre/risk/garch.py):
//  variance targeting: v = mean(r^2), omega = v (1 - alpha - beta) -> only (alpha, beta) are estimated
//  unconstrained reparameterization: p = P_MAX logistic(a), alpha = p logistic(b), beta = p - alpha
//    -> every optimizer step is a valid model (alpha, beta >= 0, alpha + beta < 1, omega > 0)
//  h_0 = v; negative log-likelihood 0.5 sum(ln 2pi + ln h_t + r_t^2 / h_t)
//  Nelder-Mead (optimization/nelder_mead.hpp) from three starting points, best one kept
namespace qre::risk{
    struct GarchFit{
        double omega;
        double alpha;
        double beta;
        double long_run_var;
        double loglik;
        bool converged;
        double persistence() const { return alpha + beta; }
    };

    namespace garch_detail{
        inline constexpr double kPMax = 0.9999;
        inline constexpr double kLog2Pi = 1.8378770664093453;   //ln(2 pi)
        inline double logistic(double x){ return 1.0 / (1.0 + std::exp(-x)); }
        inline double logit(double p){ return std::log(p / (1.0 - p)); }
    }

    inline std::pair<double, double> garch_params_from_theta(const Eigen::VectorXd& theta){
        const double p = garch_detail::kPMax * garch_detail::logistic(theta(0));
        const double alpha = p * garch_detail::logistic(theta(1));
        return {alpha, p - alpha};
    }

    inline Eigen::VectorXd garch_theta_from_params(double alpha, double beta){
        Eigen::VectorXd t(2);
        t << garch_detail::logit((alpha + beta) / garch_detail::kPMax), garch_detail::logit(alpha / (alpha + beta));
        return t;
    }

    //h_0..h_n: h_t is the variance forecast for r_t made before seeing it, h_n is for the next day
    inline std::vector<double> garch_variance_path(const std::vector<double>& r, double omega, double alpha,
                                                   double beta, double h0){
        std::vector<double> h(r.size() + 1);
        h[0] = h0;
        for(std::size_t t = 0; t < r.size(); ++t) h[t + 1] = omega + alpha * r[t] * r[t] + beta * h[t];
        return h;
    }

    inline double garch_neg_loglik(const Eigen::VectorXd& theta, const std::vector<double>& r, double v){
        const auto [alpha, beta] = garch_params_from_theta(theta);
        const auto h = garch_variance_path(r, v * (1.0 - alpha - beta), alpha, beta, v);
        double nll = 0.0;
        for(std::size_t t = 0; t < r.size(); ++t) nll += garch_detail::kLog2Pi + std::log(h[t]) + r[t] * r[t] / h[t];
        return 0.5 * nll;
    }

    inline GarchFit fit_garch(const std::vector<double>& r){
        if(r.size() < 10) throw InvalidInputError("need at least 10 returns");
        double v = 0.0;
        for(double x : r) v += x * x;
        v /= static_cast<double>(r.size());
        if(v <= 0.0) throw InvalidInputError("returns are all zero; volatility is undefined");
        auto objective = [&r, v](const Eigen::VectorXd& theta){ return garch_neg_loglik(theta, r, v); };
        const double starts[3][2] = {{0.05, 0.90}, {0.10, 0.80}, {0.02, 0.97}};
        optimization::MinimizeResult best{Eigen::VectorXd(), std::numeric_limits<double>::infinity(), 0, false};
        for(const auto& s : starts){
            const auto res = optimization::nelder_mead(objective, garch_theta_from_params(s[0], s[1]));
            if(res.fx < best.fx) best = res;
        }
        const auto [alpha, beta] = garch_params_from_theta(best.x);
        return GarchFit{v * (1.0 - alpha - beta), alpha, beta, v, -best.fx, best.converged};
    }

    inline double garch_forecast_variance(const GarchFit& fit, const std::vector<double>& r){
        return garch_variance_path(r, fit.omega, fit.alpha, fit.beta, fit.long_run_var).back();
    }

    //filtered = false: normal with the GARCH vol forecast
    //filtered = true : filtered historical simulation with GARCH volatilities (keeps the empirical tail shape)
    inline VarEs garch_var_es(const std::vector<double>& r, double confidence, bool filtered = false){
        const GarchFit fit = fit_garch(r);
        const auto h = garch_variance_path(r, fit.omega, fit.alpha, fit.beta, fit.long_run_var);
        const double sigma_next = std::sqrt(h.back());
        if(!filtered) return parametric_var_es(0.0, sigma_next, confidence);
        std::vector<double> losses(r.size());
        for(std::size_t t = 0; t < r.size(); ++t) losses[t] = -(r[t] / std::sqrt(h[t])) * sigma_next;
        return var_es_from_losses(losses, confidence);
    }
}
