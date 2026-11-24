#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"

//Phase 22: price ticks and a synthetic tick source
//A Tick is one asset's new price, stamped with the time it entered the pipeline (for latency measurement).
//SyntheticFeed simulates a market with a given daily covariance of returns: each step draws one correlated vector
//of log-returns for a time slice dt (in days) through the Cholesky factor, then emits one tick per asset in turn,
//so correlations hold across assets from step to step.
namespace qre::stream{
    struct Tick{
        std::uint32_t asset = 0;
        double price = 0.0;
        std::int64_t t_ns = 0;          //time the tick was created (steady clock)
        std::uint64_t seq = 0;
    };

    class SyntheticFeed{
    public:
        SyntheticFeed(Eigen::VectorXd prices, const Eigen::MatrixXd& daily_cov, double step_days, std::uint64_t seed)
            : p_(std::move(prices)), dt_(step_days), rng_(seed){
            const Eigen::Index n = p_.size();
            if(n == 0 || daily_cov.rows() != n || daily_cov.cols() != n) throw InvalidInputError("covariance must be N x N for N prices");
            if(!(step_days > 0.0)) throw InvalidInputError("step must be positive");
            Eigen::LLT<Eigen::MatrixXd> llt(daily_cov);
            if(llt.info() != Eigen::Success){
                //nearly singular sample matrices: add a tiny ridge
                llt.compute(daily_cov + 1e-12 * Eigen::MatrixXd::Identity(n, n) * daily_cov.diagonal().mean());
                if(llt.info() != Eigen::Success) throw InvalidInputError("covariance is not positive definite");
            }
            chol_ = llt.matrixL();
            drift_ = -0.5 * daily_cov.diagonal() * dt_;            //zero expected simple return
            z_.resize(n);
            step_.resize(n);
            next_step();
        }

        Tick next(){
            if(k_ == p_.size()) next_step();
            Tick t;
            t.asset = static_cast<std::uint32_t>(k_);
            p_(k_) *= std::exp(step_(k_));
            t.price = p_(k_);
            ++k_;
            return t;
        }

        Eigen::Index assets() const{ return p_.size(); }

    private:
        void next_step(){
            for(Eigen::Index i = 0; i < z_.size(); ++i) z_(i) = nd_(rng_);
            step_ = drift_ + std::sqrt(dt_) * (chol_ * z_);
            k_ = 0;
        }
        Eigen::VectorXd p_;
        double dt_;
        Eigen::MatrixXd chol_;
        Eigen::VectorXd drift_, z_, step_;
        Eigen::Index k_ = 0;
        std::mt19937_64 rng_;
        std::normal_distribution<double> nd_;
    };
}
