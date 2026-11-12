#pragma once
#include <cmath>
#include <cstddef>
#include <vector>
#include "qre/core/exceptions.hpp"
//Phase 13: scoring risk forecasts, and testing whether one model is really better than another
//A scoring function is strictly consistent if its expected value is minimized exactly at the true forecast
//  VaR (quantile) loss:  QL(v, L) = (1{L <= v} - c)(v - L)
//  ES alone has no such score, but (VaR, ES) jointly does (Fissler-Ziegel 2016). FZ0 (Patton-Ziegel-Chen 2019),
//  for losses with v > 0, e > 0, alpha = 1 - c:   FZ0 = 1{L > v}(L - v)/(alpha e) + v/e + ln(e) - 1
//  Score differences depend only on forecast ratios, so high-volatility days do not dominate the average
//Diebold-Mariano: d_t = S_A - S_B, DM = mean(d) / sqrt(LRV / T), Newey-West LRV with Bartlett weights
//  LRV = gamma_0 + 2 sum_{k<=L} (1 - k/(L+1)) gamma_k, default L = floor(4 (T/100)^(2/9)); DM < 0 -> A better
namespace qre::risk{
    struct DmResult{
        double statistic;
        double p_value;         //two-sided, asymptotic normal
        double mean_diff;       //mean of A - B
        std::size_t lag;
    };

    namespace scoring_detail{
        inline void check_confidence(double c){
            if(!(c > 0.0 && c < 1.0)) throw InvalidInputError("confidence must be in (0, 1)");
        }
    }

    inline double quantile_loss(double loss, double var, double confidence){
        scoring_detail::check_confidence(confidence);
        return ((loss <= var ? 1.0 : 0.0) - confidence) * (var - loss);
    }

    inline double fz0_loss(double loss, double var, double es, double confidence){
        scoring_detail::check_confidence(confidence);
        if(es <= 0.0) throw InvalidInputError("ES forecasts must be positive");
        const double hit = loss > var ? (loss - var) / ((1.0 - confidence) * es) : 0.0;
        return hit + var / es + std::log(es) - 1.0;
    }

    inline std::vector<double> fz0_loss(const std::vector<double>& loss, const std::vector<double>& var,
                                        const std::vector<double>& es, double confidence){
        if(var.size() != loss.size() || es.size() != loss.size()) throw InvalidInputError("series must have equal length");
        std::vector<double> out(loss.size());
        for(std::size_t i = 0; i < loss.size(); ++i) out[i] = fz0_loss(loss[i], var[i], es[i], confidence);
        return out;
    }

    inline std::size_t newey_west_lag(std::size_t n){
        return static_cast<std::size_t>(std::floor(4.0 * std::pow(static_cast<double>(n) / 100.0, 2.0 / 9.0)));
    }

    //lag < 0 -> rule of thumb
    inline DmResult diebold_mariano(const std::vector<double>& a, const std::vector<double>& b, long lag = -1){
        if(a.size() != b.size() || a.size() < 10) throw InvalidInputError("need two equal-length series of at least 10 scores");
        const std::size_t n = a.size();
        const std::size_t L = lag < 0 ? newey_west_lag(n) : static_cast<std::size_t>(lag);
        if(L >= n) throw InvalidInputError("lag must be below the sample size");
        std::vector<double> d(n);
        double mean = 0.0;
        for(std::size_t i = 0; i < n; ++i){ d[i] = a[i] - b[i]; mean += d[i]; }
        mean /= static_cast<double>(n);
        for(double& x : d) x -= mean;
        auto gamma = [&](std::size_t k){
            double s = 0.0;
            for(std::size_t i = k; i < n; ++i) s += d[i] * d[i - k];
            return s / static_cast<double>(n);
        };
        double lrv = gamma(0);
        for(std::size_t k = 1; k <= L; ++k) lrv += 2.0 * (1.0 - static_cast<double>(k) / static_cast<double>(L + 1)) * gamma(k);
        if(lrv <= 0.0) return {0.0, 1.0, mean, L};
        const double stat = mean / std::sqrt(lrv / static_cast<double>(n));
        return {stat, std::erfc(std::abs(stat) / std::sqrt(2.0)), mean, L};
    }
}
