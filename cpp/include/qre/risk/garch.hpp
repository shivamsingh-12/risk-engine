#pragma once
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/distributions/student_t.hpp"
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
//Student-t shocks (GarchDist::StudentT): z_t is a STANDARDIZED Student-t (unit variance) with nu > 2,
//  estimated jointly with alpha and beta via nu = 2 + exp(c); small nu = fat tails, nu -> infinity = normal
//  log density: lgamma((nu+1)/2) - lgamma(nu/2) - 0.5 ln(pi (nu-2)) - 0.5 ln h - (nu+1)/2 ln(1 + r^2 / (h (nu-2)))
//  VaR/ES with q = t_nu^{-1}(c), s = sigma sqrt((nu-2)/nu): VaR = s q, ES = s f_nu(q) / (1-c) (nu + q^2) / (nu - 1)
namespace qre::risk{
    enum class GarchDist {Normal, StudentT};

    struct GarchFit{
        double omega;
        double alpha;
        double beta;
        double long_run_var;
        double loglik;
        bool converged;
        double nu = std::numeric_limits<double>::infinity();   //Student-t degrees of freedom; inf for normal
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

    //Upper bound on nu. The likelihood is nearly flat as nu -> infinity (the t becomes the normal), and without a
    //bound the fit can drift to nu ~ 1e16, where lgamma differences and the Student-t cdf lose all precision and the
    //VaR/ES come out as garbage (negative VaR, ES ~ 1e32). At nu = 1000 the t is already the normal for risk purposes
    //(its 97.5% quantile is 0.1% above the normal one), so a fit that reaches the bound means "normal shocks"
    inline constexpr double kNuMax = 1000.0;

    inline double garch_t_neg_loglik(const Eigen::VectorXd& theta, const std::vector<double>& r, double v){
        Eigen::VectorXd ab(2);
        ab << theta(0), theta(1);
        const auto [alpha, beta] = garch_params_from_theta(ab);
        const double nu = 2.0 + std::exp(theta(2));
        if(!(nu <= kNuMax)) return std::numeric_limits<double>::infinity();
        const auto h = garch_variance_path(r, v * (1.0 - alpha - beta), alpha, beta, v);
        const double c = std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0) - 0.5 * std::log(std::numbers::pi * (nu - 2.0));
        double ll = 0.0;
        for(std::size_t t = 0; t < r.size(); ++t){
            ll += c - 0.5 * std::log(h[t]) - 0.5 * (nu + 1.0) * std::log1p(r[t] * r[t] / (h[t] * (nu - 2.0)));
        }
        return -ll;
    }

    //VaR/ES (positive losses) of sigma * z, z a standardized Student-t with nu > 2
    inline constexpr double kNuNormalLimit = 1e6;

    inline VarEs std_t_var_es(double sigma, double nu, double confidence){
        if(!(confidence > 0.0 && confidence < 1.0)) throw InvalidInputError("confidence must be in (0, 1)");
        if(sigma <= 0.0 || !(nu > 2.0)) throw InvalidInputError("need sigma > 0 and nu > 2");
        //Normal limit for very large nu: the Student-t incomplete-beta and lgamma terms lose precision there
        //(on macOS/libc++ the continued fraction stops converging around nu = 1e7), while the t is already the
        //normal to about 1e-6 relative at nu = 1e6
        if(nu > kNuNormalLimit) return parametric_var_es(0.0, sigma, confidence);
        const double q = distributions::student_t::quantile(confidence, nu);
        const double s = sigma * std::sqrt((nu - 2.0) / nu);
        const double es = s * distributions::student_t::pdf(q, nu) / (1.0 - confidence) * (nu + q * q) / (nu - 1.0);
        return VarEs{s * q, es};
    }

    inline GarchFit fit_garch(const std::vector<double>& r, GarchDist dist = GarchDist::Normal){
        if(r.size() < 10) throw InvalidInputError("need at least 10 returns");
        double v = 0.0;
        for(double x : r) v += x * x;
        v /= static_cast<double>(r.size());
        if(v <= 0.0) throw InvalidInputError("returns are all zero; volatility is undefined");
        if(dist == GarchDist::StudentT){
            auto objective_t = [&r, v](const Eigen::VectorXd& theta){ return garch_t_neg_loglik(theta, r, v); };
            const double starts_t[3][3] = {{0.05, 0.90, 8.0}, {0.10, 0.80, 5.0}, {0.02, 0.97, 20.0}};
            optimization::MinimizeResult best{Eigen::VectorXd(), std::numeric_limits<double>::infinity(), 0, false};
            for(const auto& s0 : starts_t){
                Eigen::VectorXd x0(3);
                const Eigen::VectorXd ab = garch_theta_from_params(s0[0], s0[1]);
                x0 << ab(0), ab(1), std::log(s0[2] - 2.0);
                const auto res = optimization::nelder_mead(objective_t, x0, 0.5, 1e-10, 1e-12, 6000);
                if(res.fx < best.fx) best = res;
            }
            Eigen::VectorXd ab(2);
            ab << best.x(0), best.x(1);
            const auto [alpha, beta] = garch_params_from_theta(ab);
            return GarchFit{v * (1.0 - alpha - beta), alpha, beta, v, -best.fx, best.converged, 2.0 + std::exp(best.x(2))};
        }
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

    //filtered = false: the model's own distribution (normal, or standardized t) at the GARCH vol forecast
    //filtered = true : filtered historical simulation with GARCH volatilities (keeps the empirical tail shape)
    inline VarEs garch_var_es(const std::vector<double>& r, double confidence, bool filtered = false,
                              GarchDist dist = GarchDist::Normal){
        const GarchFit fit = fit_garch(r, dist);
        const auto h = garch_variance_path(r, fit.omega, fit.alpha, fit.beta, fit.long_run_var);
        const double sigma_next = std::sqrt(h.back());
        if(!filtered){
            if(dist == GarchDist::StudentT) return std_t_var_es(sigma_next, fit.nu, confidence);
            return parametric_var_es(0.0, sigma_next, confidence);
        }
        std::vector<double> losses(r.size());
        for(std::size_t t = 0; t < r.size(); ++t) losses[t] = -(r[t] / std::sqrt(h[t])) * sigma_next;
        return var_es_from_losses(losses, confidence);
    }
}
