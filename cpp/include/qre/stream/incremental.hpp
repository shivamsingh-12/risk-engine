#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/distributions/normal.hpp"
#include "qre/risk/var_es.hpp"

//Phase 22: incremental portfolio risk, updated on every price tick. Python reference:
//python_reference/qre/stream/incremental.py (same updates, same numbers)
//
//Exposures x_i = q_i p_i (shares times price). Kept up to date on each tick in asset i, with d the change in x_i:
//  parametric  sigma^2 = x'Sx:   sigma^2 += 2 d m_i + d^2 S_ii,  m += d S(:, i)    with m = Sx          O(N)
//  historical  L = -R x:         L -= d R(:, i)                                                         O(T)
//against O(N^2 + N T) to recompute. VaR from L by selection (std::nth_element), O(T). resync() recomputes all.
//R is stored column-major (Eigen's default), so the column R(:, i) a tick touches is contiguous in memory.
namespace qre::stream{
    struct RiskSnapshot{
        long long ticks = 0;
        double value = 0.0, pnl = 0.0, sigma = 0.0;
        double var_parametric = 0.0, es_parametric = 0.0, var_historical = 0.0, es_historical = 0.0;
        Eigen::VectorXd component_var;
    };

    //numpy.quantile's default (linear interpolation) by selection; reorders scratch
    inline double quantile_linear(std::vector<double>& scratch, double q){
        const auto n = scratch.size();
        const double h = static_cast<double>(n - 1) * q;
        const auto lo = static_cast<std::size_t>(std::floor(h));
        const auto hi = std::min(lo + 1, n - 1);
        std::nth_element(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(lo), scratch.end());
        const double a = scratch[lo];
        double b = a;
        if(hi != lo) b = *std::min_element(scratch.begin() + static_cast<std::ptrdiff_t>(hi), scratch.end());   //next order statistic
        return a + (h - static_cast<double>(lo)) * (b - a);
    }

    class IncrementalRisk{
    public:
        IncrementalRisk(Eigen::VectorXd shares, Eigen::VectorXd prices, Eigen::MatrixXd cov, Eigen::MatrixXd scenarios,
                        double confidence = 0.99)
            : q_(std::move(shares)), p_(std::move(prices)), cov_(std::move(cov)), r_(std::move(scenarios)), c_(confidence){
            const Eigen::Index n = q_.size();
            if(p_.size() != n || cov_.rows() != n || cov_.cols() != n || r_.cols() != n)
                throw InvalidInputError("shares, prices, covariance (N x N) and scenarios (T x N) must agree on N");
            if(r_.rows() < 20) throw InvalidInputError("need at least 20 return scenarios");
            if(!p_.allFinite() || (p_.array() <= 0.0).any()) throw InvalidInputError("prices must be finite and positive");
            if(!(c_ >= 0.5 && c_ < 1.0)) throw InvalidInputError("confidence must be in [0.5, 1)");
            z_ = risk::detail::normal_quantile(c_);
            k_es_ = distributions::normal::pdf(z_, 0.0, 1.0) / (1.0 - c_);
            resync();
            value0_ = value_;
            scratch_.resize(static_cast<std::size_t>(r_.rows()));
        }

        void resync(){
            x_ = q_.cwiseProduct(p_);
            value_ = x_.sum();
            m_ = cov_ * x_;
            var2_ = x_.dot(m_);
            loss_ = -(r_ * x_);
        }

        void on_tick(Eigen::Index i, double price){
            if(!(price > 0.0) || i < 0 || i >= q_.size()) throw InvalidInputError("tick needs a valid asset and a positive price");
            const double d = q_(i) * (price - p_(i));
            p_(i) = price;
            x_(i) += d;
            value_ += d;
            var2_ += 2.0 * d * m_(i) + d * d * cov_(i, i);
            m_.noalias() += d * cov_.col(i);
            loss_.noalias() -= d * r_.col(i);
            ++ticks_;
        }

        RiskSnapshot snapshot(){
            RiskSnapshot s;
            s.ticks = ticks_;
            s.value = value_;
            s.pnl = value_ - value0_;
            s.sigma = std::sqrt(std::max(var2_, 0.0));
            s.var_parametric = z_ * s.sigma;
            s.es_parametric = k_es_ * s.sigma;
            std::copy(loss_.data(), loss_.data() + loss_.size(), scratch_.begin());
            s.var_historical = quantile_linear(scratch_, c_);
            double sum = 0.0;
            int k = 0;
            for(Eigen::Index t = 0; t < loss_.size(); ++t) if(loss_(t) >= s.var_historical){ sum += loss_(t); ++k; }
            s.es_historical = sum / k;
            s.component_var = s.sigma > 0.0 ? (z_ * x_.cwiseProduct(m_) / s.sigma).eval() : Eigen::VectorXd::Zero(x_.size()).eval();
            return s;
        }

        //largest relative differences between the running values and a full recomputation
        struct Drift{ double variance, exposure, scenario_loss; };
        Drift drift() const{
            const Eigen::VectorXd x = q_.cwiseProduct(p_);
            const Eigen::VectorXd m = cov_ * x;
            const double v2 = x.dot(m);
            const Eigen::VectorXd loss = -(r_ * x);
            return {std::abs(var2_ - v2) / v2, (x_ - x).cwiseAbs().maxCoeff() / x.cwiseAbs().maxCoeff(),
                    (loss_ - loss).cwiseAbs().maxCoeff() / std::max(loss.cwiseAbs().maxCoeff(), 1e-300)};
        }

        Eigen::Index assets() const{ return q_.size(); }
        Eigen::Index scenarios() const{ return r_.rows(); }
        double price(Eigen::Index i) const{ return p_(i); }
        long long ticks() const{ return ticks_; }

    private:
        Eigen::VectorXd q_, p_;
        Eigen::MatrixXd cov_, r_;
        double c_, z_ = 0.0, k_es_ = 0.0;
        Eigen::VectorXd x_, m_, loss_;
        double value_ = 0.0, value0_ = 0.0, var2_ = 0.0;
        long long ticks_ = 0;
        std::vector<double> scratch_;
    };

    //The same numbers recomputed from scratch on every tick: the baseline incremental updating is measured against
    inline RiskSnapshot full_recompute(const Eigen::VectorXd& shares, const Eigen::VectorXd& prices, const Eigen::MatrixXd& cov,
                                       const Eigen::MatrixXd& scenarios, double confidence, double value0, std::vector<double>& scratch){
        RiskSnapshot s;
        const Eigen::VectorXd x = shares.cwiseProduct(prices);
        const Eigen::VectorXd m = cov * x;
        s.value = x.sum();
        s.pnl = s.value - value0;
        s.sigma = std::sqrt(std::max(x.dot(m), 0.0));
        const double z = risk::detail::normal_quantile(confidence);
        s.var_parametric = z * s.sigma;
        s.es_parametric = distributions::normal::pdf(z, 0.0, 1.0) / (1.0 - confidence) * s.sigma;
        const Eigen::VectorXd loss = -(scenarios * x);
        scratch.assign(loss.data(), loss.data() + loss.size());
        s.var_historical = quantile_linear(scratch, confidence);
        double sum = 0.0;
        int k = 0;
        for(Eigen::Index t = 0; t < loss.size(); ++t) if(loss(t) >= s.var_historical){ sum += loss(t); ++k; }
        s.es_historical = sum / k;
        s.component_var = z * x.cwiseProduct(m) / s.sigma;
        return s;
    }
}
