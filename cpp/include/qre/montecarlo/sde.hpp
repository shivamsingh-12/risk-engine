#pragma once
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/rng.hpp"
#include "qre/distributions/normal.hpp"
//Phase 8: discretizing stochastic differential equations, and the Ito integral
//SDE dX = a(X, t) dt + b(X, t) dW on [0, T], `steps` equal steps of size h = T / steps
//  Euler-Maruyama: X_{k+1} = X_k + a h + b dW_k
//  Milstein:       X_{k+1} = X_k + a h + b dW_k + 0.5 b b' (dW_k^2 - h),  b' = db/dx
//  Milstein's extra term comes from Ito's lemma applied to b(X); it lifts strong (pathwise) order from 1/2 to 1.
//  Both schemes have weak (distributional) order 1.
//Brownian increments are passed in explicitly (rows = paths, cols = steps) so schemes and step sizes can share
//one path: coarsen() sums fine increments into coarser ones, which is how strong error is measured
//Ito vs Stratonovich for the integral of W dW:
//  Ito (left point) sum W_k dW_k -> (W_T^2 - T) / 2;  Stratonovich (average) = W_T^2 / 2 exactly (telescopes)
//  the gap is half the quadratic variation sum dW_k^2, which converges to T: the dt term in Ito's lemma
namespace qre::montecarlo{
    using SdeFn = std::function<double(double x, double t)>;

    inline Eigen::MatrixXd brownian_increments(Eigen::Index n_paths, Eigen::Index steps, double T, Rng& rng){
        if(n_paths < 1 || steps < 1 || T <= 0.0) throw InvalidInputError("need n_paths >= 1, steps >= 1 and T > 0");
        const auto z = distributions::normal::sample(static_cast<std::size_t>(n_paths * steps), 0.0,
                                                     std::sqrt(T / static_cast<double>(steps)), rng);
        Eigen::MatrixXd dW(n_paths, steps);
        for(Eigen::Index p = 0; p < n_paths; ++p)
            for(Eigen::Index k = 0; k < steps; ++k) dW(p, k) = z[static_cast<std::size_t>(p * steps + k)];
        return dW;
    }

    //Same Brownian path on a grid `factor` times coarser
    inline Eigen::MatrixXd coarsen(const Eigen::MatrixXd& dW, Eigen::Index factor){
        if(factor < 1 || dW.cols() % factor != 0) throw InvalidInputError("factor must divide the number of steps");
        Eigen::MatrixXd out(dW.rows(), dW.cols() / factor);
        for(Eigen::Index k = 0; k < out.cols(); ++k) out.col(k) = dW.middleCols(k * factor, factor).rowwise().sum();
        return out;
    }

    inline Eigen::MatrixXd euler_maruyama(double x0, const SdeFn& a, const SdeFn& b, double T, const Eigen::MatrixXd& dW){
        if(T <= 0.0 || dW.cols() < 1) throw InvalidInputError("need T > 0 and at least one step");
        const double h = T / static_cast<double>(dW.cols());
        Eigen::MatrixXd x(dW.rows(), dW.cols() + 1);
        x.col(0).setConstant(x0);
        for(Eigen::Index k = 0; k < dW.cols(); ++k){
            const double t = static_cast<double>(k) * h;
            for(Eigen::Index p = 0; p < dW.rows(); ++p){
                const double xk = x(p, k);
                x(p, k + 1) = xk + a(xk, t) * h + b(xk, t) * dW(p, k);
            }
        }
        return x;
    }

    inline Eigen::MatrixXd milstein(double x0, const SdeFn& a, const SdeFn& b, const SdeFn& b_x, double T,
                                    const Eigen::MatrixXd& dW){
        if(T <= 0.0 || dW.cols() < 1) throw InvalidInputError("need T > 0 and at least one step");
        const double h = T / static_cast<double>(dW.cols());
        Eigen::MatrixXd x(dW.rows(), dW.cols() + 1);
        x.col(0).setConstant(x0);
        for(Eigen::Index k = 0; k < dW.cols(); ++k){
            const double t = static_cast<double>(k) * h;
            for(Eigen::Index p = 0; p < dW.rows(); ++p){
                const double xk = x(p, k), bk = b(xk, t), dw = dW(p, k);
                x(p, k + 1) = xk + a(xk, t) * h + bk * dw + 0.5 * bk * b_x(xk, t) * (dw * dw - h);
            }
        }
        return x;
    }

    //Left-point (Ito) sum of W dW per path
    inline Eigen::VectorXd ito_integral_w_dw(const Eigen::MatrixXd& dW){
        Eigen::VectorXd out(dW.rows());
        for(Eigen::Index p = 0; p < dW.rows(); ++p){
            double w = 0.0, s = 0.0;
            for(Eigen::Index k = 0; k < dW.cols(); ++k){ s += w * dW(p, k); w += dW(p, k); }
            out(p) = s;
        }
        return out;
    }

    //Average-point (Stratonovich) sum of W dW per path
    inline Eigen::VectorXd stratonovich_integral_w_dw(const Eigen::MatrixXd& dW){
        Eigen::VectorXd out(dW.rows());
        for(Eigen::Index p = 0; p < dW.rows(); ++p){
            double w = 0.0, s = 0.0;
            for(Eigen::Index k = 0; k < dW.cols(); ++k){ const double w_next = w + dW(p, k); s += 0.5 * (w + w_next) * dW(p, k); w = w_next; }
            out(p) = s;
        }
        return out;
    }

    inline Eigen::VectorXd quadratic_variation(const Eigen::MatrixXd& dW){
        return dW.array().square().rowwise().sum().matrix();
    }
}
