#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/rng.hpp"
#include "qre/optimization/nelder_mead.hpp"
#include "qre/risk/garch.hpp"
#include "qre/risk/var_es.hpp"
//Phase 11: extreme value theory for tail risk, and the uncertainty around risk estimates
//Peaks over threshold: above a high threshold u the excess Y = L - u is approximately generalized Pareto
//  P(Y <= y) = 1 - (1 + xi y / beta)^(-1/xi)  (xi != 0),  1 - exp(-y / beta)  (xi = 0)
//  xi > 0 heavy power-law tail (Student-t with nu dof has xi = 1/nu), xi = 0 thin tail, xi < 0 finite end point
//With n losses, N_u above u:  VaR_p = u + beta/xi ((n/N_u (1-p))^(-xi) - 1),  ES_p = VaR_p/(1-xi) + (beta - xi u)/(1-xi)
//  -> extrapolates beyond the largest observed losses using the fitted tail shape
//GPD fit: MLE over (xi, ln beta) by Nelder-Mead from the method-of-moments start, xi restricted to (-0.5, 0.95):
//  below -0.5 MLE is irregular, at xi >= 1 ES is infinite. With few exceedances the unrestricted MLE of xi is very
//  unstable; GpdFit::at_bound reports when the restriction binds
//Hill: xi_H = (1/k) sum_{i<=k} ln X_(i) - ln X_(k+1) from the k largest (positive) losses; only for heavy tails
//Conditional EVT (McNeil-Frey 2000): GARCH filter, POT on standardized residual losses, rescale by next-day vol
//Bootstrap: moving-block resampling of losses (block 1 = iid), percentile intervals for historical VaR/ES
namespace qre::risk{
    inline constexpr double kXiMin = -0.5, kXiMax = 0.95;

    struct GpdFit{
        double xi;
        double beta;
        bool at_bound;      //xi restriction binding
    };

    struct PotResult{
        VarEs risk;
        double xi;
        double beta;
        double threshold;
        std::size_t n_exceed;
        bool at_bound;
    };

    struct BootstrapCi{
        double var_lo, var_hi, es_lo, es_hi;
    };

    namespace evt_detail{
        //Linear-interpolation empirical quantile (numpy's default), as in var_es_from_losses
        inline double quantile(std::vector<double> x, double q){
            std::sort(x.begin(), x.end());
            const double h = static_cast<double>(x.size() - 1) * q;
            const auto lo = static_cast<std::size_t>(std::floor(h));
            const std::size_t hi = std::min(lo + 1, x.size() - 1);
            return x[lo] + (h - static_cast<double>(lo)) * (x[hi] - x[lo]);
        }
    }

    inline double gpd_neg_loglik(const Eigen::VectorXd& theta, const std::vector<double>& y){
        const double xi = theta(0), beta = std::exp(theta(1));
        if(!(xi > kXiMin && xi < kXiMax)) return std::numeric_limits<double>::infinity();
        const double n = static_cast<double>(y.size());
        if(std::abs(xi) < 1e-9){
            double s = 0.0;
            for(double v : y) s += v;
            return n * std::log(beta) + s / beta;
        }
        double s = 0.0;
        for(double v : y){
            const double z = 1.0 + xi * v / beta;
            if(z <= 0.0) return std::numeric_limits<double>::infinity();
            s += std::log(z);
        }
        return n * std::log(beta) + (1.0 + 1.0 / xi) * s;
    }

    inline GpdFit fit_gpd(const std::vector<double>& y){
        if(y.size() < 5) throw InvalidInputError("need at least 5 positive exceedances");
        for(double v : y) if(!(v > 0.0)) throw InvalidInputError("exceedances must be positive");
        const double m = estimators::empirical::sample_mean(y), var = estimators::empirical::sample_var(y);
        const double xi0 = std::clamp(0.5 * (1.0 - m * m / var), -0.4, 0.8);
        const double beta0 = std::max(0.5 * m * (m * m / var + 1.0), 1e-12);
        auto obj = [&y](const Eigen::VectorXd& t){ return gpd_neg_loglik(t, y); };
        optimization::MinimizeResult best{Eigen::VectorXd(), std::numeric_limits<double>::infinity(), 0, false};
        for(const auto& [a, b] : {std::pair<double, double>{xi0, std::log(beta0)}, {0.1, std::log(m)}}){
            Eigen::VectorXd x0(2);
            x0 << a, b;
            //the moment start can lie outside the support (xi < 0 with an exceedance beyond -beta/xi): skip it,
            //the second start (xi = 0.1 > 0) is always feasible. Found on a min-variance portfolio in Phase 15
            if(!std::isfinite(obj(x0))) continue;
            const auto r = optimization::nelder_mead(obj, x0, 0.2, 1e-10, 1e-12, 4000);
            if(r.fx < best.fx) best = r;
        }
        const double xi = best.x(0);
        return GpdFit{xi, std::exp(best.x(1)), xi <= kXiMin + 1e-4 || xi >= kXiMax - 1e-4};
    }

    inline VarEs gpd_var_es(double u, double xi, double beta, std::size_t n, std::size_t n_u, double confidence){
        if(!(confidence > 0.0 && confidence < 1.0)) throw InvalidInputError("confidence must be in (0, 1)");
        if(xi >= 1.0) throw InvalidInputError("ES is infinite for xi >= 1");
        const double tail = static_cast<double>(n) / static_cast<double>(n_u) * (1.0 - confidence);
        if(std::abs(xi) < 1e-9){
            const double v = u - beta * std::log(tail);
            return VarEs{v, v + beta};
        }
        const double v = u + beta / xi * (std::pow(tail, -xi) - 1.0);
        return VarEs{v, v / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)};
    }

    inline PotResult pot_var_es(const std::vector<double>& losses, double confidence, double threshold_quantile = 0.9){
        if(losses.size() < 20) throw InvalidInputError("need at least 20 losses");
        if(!(threshold_quantile > 0.0 && threshold_quantile < confidence && confidence < 1.0)){
            throw InvalidInputError("need 0 < threshold_quantile < confidence < 1");
        }
        const double u = evt_detail::quantile(losses, threshold_quantile);
        std::vector<double> exc;
        for(double l : losses) if(l > u) exc.push_back(l - u);
        const GpdFit f = fit_gpd(exc);
        return PotResult{gpd_var_es(u, f.xi, f.beta, losses.size(), exc.size(), confidence), f.xi, f.beta, u, exc.size(), f.at_bound};
    }

    inline double hill_estimator(std::vector<double> losses, std::size_t k){
        std::sort(losses.begin(), losses.end(), std::greater<>());
        if(k < 1 || k >= losses.size() || !(losses[k] > 0.0)) throw InvalidInputError("need 1 <= k < n and the (k+1)-th largest loss positive");
        double s = 0.0;
        for(std::size_t i = 0; i < k; ++i) s += std::log(losses[i]);
        return s / static_cast<double>(k) - std::log(losses[k]);
    }

    inline VarEs garch_evt_var_es(const std::vector<double>& r, double confidence, double threshold_quantile = 0.9){
        const GarchFit fit = fit_garch(r);
        const auto h = garch_variance_path(r, fit.omega, fit.alpha, fit.beta, fit.long_run_var);
        std::vector<double> z_loss(r.size());
        for(std::size_t t = 0; t < r.size(); ++t) z_loss[t] = -r[t] / std::sqrt(h[t]);
        const PotResult p = pot_var_es(z_loss, confidence, threshold_quantile);
        const double s = std::sqrt(h.back());
        return VarEs{s * p.risk.var, s * p.risk.es};
    }

    inline BootstrapCi bootstrap_var_es(const std::vector<double>& losses, double confidence, std::size_t n_boot, std::size_t block,
                                        Rng& rng, double level = 0.90){
        const std::size_t n = losses.size();
        if(n < 20 || block < 1 || block > n || n_boot < 10) throw InvalidInputError("need n >= 20, 1 <= block <= n and n_boot >= 10");
        std::uniform_int_distribution<std::size_t> start(0, n - block);
        std::vector<double> vs(n_boot), es(n_boot), sample(n);
        for(std::size_t b = 0; b < n_boot; ++b){
            std::size_t filled = 0;
            while(filled < n){
                const std::size_t s = start(rng.engine());
                for(std::size_t j = 0; j < block && filled < n; ++j) sample[filled++] = losses[s + j];
            }
            const VarEs v = var_es_from_losses(sample, confidence);
            vs[b] = v.var; es[b] = v.es;
        }
        const double a = (1.0 - level) / 2.0;
        return BootstrapCi{evt_detail::quantile(vs, a), evt_detail::quantile(vs, 1.0 - a), evt_detail::quantile(es, a), evt_detail::quantile(es, 1.0 - a)};
    }
}
