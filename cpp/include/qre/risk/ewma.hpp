#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"
#include "qre/risk/var_es.hpp"
//Adaptive volatility: EWMA (RiskMetrics) and filtered historical simulation
//Zero-mean returns, as in RiskMetrics: at daily horizons the mean is tiny next to the volatility
//EWMA weights on a window of n returns (oldest first): w_i = (1 - lam) lam^(n-1-i) / (1 - lam^n)
//  -> newest return weighs most, weights sum to exactly 1, lam = 0.94 is the RiskMetrics daily standard
//Vol path for a return series r_0..r_{n-1}:
//  s2_0 = mean(r^2) over the window (seed),  s2_{i+1} = lam s2_i + (1 - lam) r_i^2
//  s_i is the forecast for r_i made BEFORE r_i was seen; s_n is the forecast for the next day
//Filtered historical simulation (Hull-White): z_i = r_i / s_i, scenarios = z_i s_n, empirical VaR/ES of those
//  -> keeps the shape (fat tails) of the historical distribution but rescales it to today's volatility
namespace qre::risk{
    namespace detail{
        inline void check_lambda(double lam){
            if(!(lam > 0.0 && lam < 1.0)) throw InvalidInputError("lambda must be in (0, 1)");
        }
    }

    inline Eigen::VectorXd ewma_weights(Eigen::Index n, double lam = 0.94){
        detail::check_lambda(lam);
        if(n < 1) throw InvalidInputError("need at least 1 observation");
        Eigen::VectorXd w(n);
        const double norm = (1.0 - lam) / (1.0 - std::pow(lam, static_cast<double>(n)));
        for(Eigen::Index i = 0; i < n; ++i) w(i) = norm * std::pow(lam, static_cast<double>(n - 1 - i));
        return w;
    }

    //Zero-mean EWMA covariance of a (T x N) return window: sum_i w_i r_i r_i^T
    inline CovarianceMatrix ewma_covariance(const ReturnMatrix& returns, double lam = 0.94){
        if(returns.rows() < 1 || returns.cols() < 1) throw InvalidInputError("returns must be non-empty");
        const Eigen::VectorXd w = ewma_weights(returns.rows(), lam);
        return returns.transpose() * w.asDiagonal() * returns;
    }

    inline std::vector<double> ewma_vol_path(const std::vector<double>& r, double lam = 0.94){
        detail::check_lambda(lam);
        if(r.size() < 2) throw InvalidInputError("need at least 2 returns");
        double seed = 0.0;
        for(double x : r) seed += x * x;
        seed /= static_cast<double>(r.size());
        if(seed <= 0.0) throw InvalidInputError("returns are all zero; volatility is undefined");
        std::vector<double> s(r.size() + 1);
        double s2 = seed;
        s[0] = std::sqrt(s2);
        for(std::size_t i = 0; i < r.size(); ++i){
            s2 = lam * s2 + (1.0 - lam) * r[i] * r[i];
            s[i + 1] = std::sqrt(s2);
        }
        return s;
    }

    inline VarEs fhs_var_es(const std::vector<double>& returns, double confidence, double lam = 0.94){
        const std::vector<double> s = ewma_vol_path(returns, lam);
        std::vector<double> losses(returns.size());
        for(std::size_t i = 0; i < returns.size(); ++i) losses[i] = -(returns[i] / s[i]) * s.back();
        return var_es_from_losses(losses, confidence);
    }

    inline VarEs ewma_parametric_var_es(const ReturnMatrix& returns, const WeightVector& weights,
                                        double confidence, double lam = 0.94){
        return parametric_var_es(0.0, portfolio_vol(weights, ewma_covariance(returns, lam)), confidence);
    }
}
