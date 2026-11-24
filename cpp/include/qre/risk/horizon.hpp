#pragma once
#include <cmath>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/risk/garch.hpp"
#include "qre/risk/var_es.hpp"

//Phase 21: multi-day (h-day) VaR and ES. Python reference: python_reference/qre/risk/horizon.py
//  sqrt_time     1-day historical x sqrt(h)
//  overlapping   historical on overlapping compounded h-day returns
//  garch_normal  normal with the GARCH h-day variance h v + (s1 - v)(1 - p^h)/(1 - p)
//  fhs_garch     filtered historical simulation: bootstrapped standardized residuals through the GARCH recursion
namespace qre::risk{
    namespace horizon_detail{
        inline void check(const std::vector<double>& r, int h){
            if(r.size() < 20) throw InvalidInputError("need a finite 1-D series of at least 20 daily returns");
            for(double x : r) if(!std::isfinite(x)) throw InvalidInputError("need a finite 1-D series of at least 20 daily returns");
            if(h < 1) throw InvalidInputError("horizon must be a whole number of days, at least 1");
        }
    }

    inline VarEs sqrt_time_var_es(const std::vector<double>& r, double confidence, int h){
        horizon_detail::check(r, h);
        return historical_var_es(r, confidence, h);
    }

    inline VarEs overlapping_var_es(const std::vector<double>& r, double confidence, int h){
        horizon_detail::check(r, h);
        const auto n = static_cast<int>(r.size());
        if(n - h + 1 < 20) throw InvalidInputError("window too short for overlapping multi-day returns");
        std::vector<double> growth(static_cast<std::size_t>(n) + 1, 1.0);
        for(int i = 0; i < n; ++i) growth[static_cast<std::size_t>(i) + 1] = growth[static_cast<std::size_t>(i)] * (1.0 + r[static_cast<std::size_t>(i)]);
        std::vector<double> losses;
        losses.reserve(static_cast<std::size_t>(n - h + 1));
        for(int i = h; i <= n; ++i) losses.push_back(-(growth[static_cast<std::size_t>(i)] / growth[static_cast<std::size_t>(i - h)] - 1.0));
        return var_es_from_losses(losses, confidence);
    }

    inline double garch_variance_sum(const GarchFit& fit, double sigma2_next, int h){
        const double p = fit.alpha + fit.beta, v = fit.long_run_var;
        if(p >= 1.0) return h * sigma2_next;
        return h * v + (sigma2_next - v) * (1.0 - std::pow(p, h)) / (1.0 - p);
    }

    inline VarEs garch_normal_var_es(const std::vector<double>& r, double confidence, int h, const GarchFit* fit = nullptr){
        horizon_detail::check(r, h);
        const GarchFit f = fit ? *fit : fit_garch(r);
        const double s2 = garch_forecast_variance(f, r);
        return parametric_var_es(0.0, std::sqrt(garch_variance_sum(f, s2, h)), confidence);
    }

    //draws: n_paths x h indices into r. Returns the simulated h-day compounded returns
    inline std::vector<double> fhs_garch_paths(const std::vector<double>& r, int h, const Eigen::MatrixXi& draws,
                                               const GarchFit* fit = nullptr){
        horizon_detail::check(r, h);
        const auto n = static_cast<int>(r.size());
        if(draws.cols() != h || draws.rows() < 1 || draws.minCoeff() < 0 || draws.maxCoeff() >= n)
            throw InvalidInputError("draws must be n_paths x h indices into the returns");
        const GarchFit f = fit ? *fit : fit_garch(r);
        const auto hv = garch_variance_path(r, f.omega, f.alpha, f.beta, f.long_run_var);
        std::vector<double> z(r.size());
        for(std::size_t i = 0; i < r.size(); ++i) z[i] = r[i] / std::sqrt(hv[i]);
        std::vector<double> out(static_cast<std::size_t>(draws.rows()));
        for(Eigen::Index p = 0; p < draws.rows(); ++p){
            double s2 = hv.back(), g = 1.0;
            for(int k = 0; k < h; ++k){
                const double x = std::sqrt(s2) * z[static_cast<std::size_t>(draws(p, k))];
                g *= 1.0 + x;
                s2 = f.omega + f.alpha * x * x + f.beta * s2;
            }
            out[static_cast<std::size_t>(p)] = g - 1.0;
        }
        return out;
    }

    inline VarEs fhs_garch_var_es(const std::vector<double>& r, double confidence, int h, int n_paths = 10000,
                                  uint64_t seed = 0, const GarchFit* fit = nullptr){
        horizon_detail::check(r, h);
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> pick(0, static_cast<int>(r.size()) - 1);
        Eigen::MatrixXi draws(n_paths, h);
        for(int p = 0; p < n_paths; ++p) for(int k = 0; k < h; ++k) draws(p, k) = pick(rng);
        const auto paths = fhs_garch_paths(r, h, draws, fit);
        std::vector<double> losses(paths.size());
        for(std::size_t i = 0; i < paths.size(); ++i) losses[i] = -paths[i];
        return var_es_from_losses(losses, confidence);
    }
}
