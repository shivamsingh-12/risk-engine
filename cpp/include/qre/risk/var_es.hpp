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
#include "qre/montecarlo/gbm_path.hpp"
#include "qre/optimization/root_finding.hpp"
//Phase 7: Value-at-Risk and Expected Shortfall three ways + risk contributions
//Conventions: VaR and ES are POSITIVE LOSSES as a fraction of portfolio value
//  confidence c in (0,1): VaR is the loss exceeded with probability 1 - c
//  ES = average loss given loss >= VaR, so ES >= VaR
//  weights are fractions of portfolio value (buy and hold over the horizon)
//  horizon in trading days: historical and parametric scale the 1-day figure by sqrt(h)
//  (assumes i.i.d. returns, an approximation) while Monte Carlo simulates the horizon directly
namespace qre::risk{
    struct VarEs{
        double var;
        double es;
    };

    namespace detail{
        inline void check_confidence(double confidence){
            if(!(confidence > 0.0 && confidence < 1.0)) throw InvalidInputError("confidence must be in (0, 1)");
        }
        inline void check_horizon(int horizon_days){
            if(horizon_days < 1) throw InvalidInputError("horizon_days must be at least 1");
        }
        //Standard normal quantile = root of cdf(x) - c -> reuse Phase 1 cdf with Phase 4 Brent
        inline double normal_quantile(double c){
            check_confidence(c);
            auto f = [c](double x){ return distributions::normal::cdf(x, 0.0, 1.0) - c; };
            auto res = optimization::brent(f, -12.0, 12.0, 1e-14, 200);
            if(!res.has_value() || !res->converged){
                throw ConvergenceError("normal quantile did not converge");
            }
            return res->root;
        }
    }

    //Empirical VaR (linear interpolated quantile, same as numpy's default) and ES (mean of losses >= VaR)
    inline VarEs var_es_from_losses(const std::vector<double>& losses, double confidence){
        detail::check_confidence(confidence);
        if(losses.size() < 2) throw InvalidInputError("need at least 2 observations");
        std::vector<double> sorted = losses;
        std::sort(sorted.begin(), sorted.end());
        const double h = static_cast<double>(sorted.size() - 1) * confidence;
        const std::size_t lo = static_cast<std::size_t>(std::floor(h));
        const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
        const double var = sorted[lo] + (h - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
        double sum = 0.0;
        std::size_t count = 0;
        for(double l : losses){
            if(l >= var){ sum += l; ++count; }
        }
        return VarEs{var, sum / static_cast<double>(count)};
    }

    //Historical simulation on 1-day portfolio returns, sqrt-time scaled to the horizon
    inline VarEs historical_var_es(const std::vector<double>& returns, double confidence, int horizon_days = 1){
        detail::check_horizon(horizon_days);
        std::vector<double> losses;
        losses.reserve(returns.size());
        for(double r : returns) losses.push_back(-r);
        const VarEs base = var_es_from_losses(losses, confidence);
        const double scale = std::sqrt(static_cast<double>(horizon_days));
        return VarEs{base.var * scale, base.es * scale};
    }

    //Normal (variance-covariance) VaR/ES from 1-day portfolio mean and volatility
    //VaR = -mu + sigma z_c    ES = -mu + sigma phi(z_c) / (1 - c)
    inline VarEs parametric_var_es(double mu, double sigma, double confidence, int horizon_days = 1){
        detail::check_confidence(confidence);
        detail::check_horizon(horizon_days);
        if(sigma <= 0.0) throw InvalidInputError("sigma must be positive");
        const double h = static_cast<double>(horizon_days);
        const double mu_h = mu * h;
        const double sigma_h = sigma * std::sqrt(h);
        const double z = detail::normal_quantile(confidence);
        return VarEs{-mu_h + sigma_h * z,
                     -mu_h + sigma_h * distributions::normal::pdf(z, 0.0, 1.0) / (1.0 - confidence)};
    }

    //Asset returns (rows = time, cols = assets) * weights -> portfolio return per period
    inline std::vector<double> portfolio_returns(const ReturnMatrix& returns, const WeightVector& weights){
        if(returns.cols() != weights.size()) throw InvalidInputError("returns columns must match number of weights");
        const Eigen::VectorXd p = returns * weights;
        return std::vector<double>(p.data(), p.data() + p.size());
    }

    inline double portfolio_vol(const WeightVector& weights, const CovarianceMatrix& cov){
        if(cov.rows() != weights.size() || cov.cols() != weights.size()){
            throw InvalidInputError("cov must be square and match the number of weights");
        }
        const double var = weights.dot(cov * weights);
        if(var <= 0.0) throw InvalidInputError("portfolio variance must be positive");
        return std::sqrt(var);
    }

    //Euler decomposition: RC_i = w_i (cov w)_i / sigma_p, so the contributions sum to sigma_p
    inline Eigen::VectorXd risk_contributions(const WeightVector& weights, const CovarianceMatrix& cov){
        const double sigma_p = portfolio_vol(weights, cov);
        const Eigen::VectorXd marginal = cov * weights;
        return weights.cwiseProduct(marginal) / sigma_p;
    }

    //Monte Carlo VaR/ES from the Phase 6 correlated GBM, zero expected simple return over the horizon
    inline VarEs monte_carlo_var_es(const WeightVector& weights, const CovarianceMatrix& cov, double confidence,
                                    int horizon_days, std::size_t n_paths, Rng& rng,
                                    int trading_days = 252, bool antithetic = true){
        detail::check_confidence(confidence);
        detail::check_horizon(horizon_days);
        const Eigen::Index n = weights.size();
        if(cov.rows() != n || cov.cols() != n) throw InvalidInputError("cov must be square and match the number of weights");
        Eigen::VectorXd daily_vol(n);
        for(Eigen::Index i = 0; i < n; ++i){
            if(cov(i, i) <= 0.0) throw InvalidInputError("variances must be positive");
            daily_vol(i) = std::sqrt(cov(i, i));
        }
        const CorrelationMatrix corr = cov.array() / (daily_vol * daily_vol.transpose()).array();
        const Eigen::VectorXd ann_vol = daily_vol * std::sqrt(static_cast<double>(trading_days));
        const auto assets = montecarlo::simulate_correlated_gbm(
            Eigen::VectorXd::Ones(n), 0.0, ann_vol, corr,
            static_cast<double>(horizon_days) / static_cast<double>(trading_days), 1, n_paths, rng, antithetic);
        std::vector<double> losses(n_paths, 0.0);
        for(Eigen::Index a = 0; a < n; ++a){
            const Eigen::MatrixXd& m = assets[static_cast<std::size_t>(a)];
            for(Eigen::Index p = 0; p < m.rows(); ++p){
                losses[static_cast<std::size_t>(p)] -= weights(a) * (m(p, 1) - 1.0);
            }
        }
        return var_es_from_losses(losses, confidence);
    }
}
