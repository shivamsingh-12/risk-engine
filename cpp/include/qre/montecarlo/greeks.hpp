#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <Eigen/Dense>
#include "qre/autodiff/aad.hpp"
#include "qre/core/exceptions.hpp"
#include "qre/distributions/normal.hpp"
//Phase 12: Monte Carlo Greeks -- bumping, pathwise, likelihood ratio, and AAD
//Bump and revalue: (V(theta+h) - V(theta-h)) / 2h on the same random numbers. O(h^2) bias, 2 pricings per input,
//  variance explodes as h -> 0 for discontinuous payoffs
//Pathwise: dV/dtheta = E[f'(S_T) dS_T/dtheta]; unbiased for continuous payoffs (kinks fine), low variance;
//  for a digital f' = 0 a.e. so the estimate is identically 0.  GBM: dS_T/dS0 = S_T/S0,
//  dS_T/dsigma = S_T (sqrt(T) Z - sigma T),  dS_T/dr = T S_T
//Likelihood ratio: dV/dtheta = E[f(S_T) dln p/dtheta]; no payoff derivative needed (works for digitals), noisier.
//  delta weight Z / (S0 sigma sqrt(T)),  vega weight (Z^2 - 1)/sigma - Z sqrt(T)
//AAD computes the pathwise estimator for every input in one backward sweep per path
//Basket call: B = sum w_i S_i(T), x = L z (L Cholesky of the correlation), I = 1{B > K}
//  dV/dS_i0 = e^{-rT} I w_i S_i/S_i0,  dV/dsigma_i = e^{-rT} I w_i S_i (sqrt(T) x_i - sigma_i T)
//  dV/dr = -T V_path + e^{-rT} I T B,  dV/dK = -e^{-rT} I
namespace qre::montecarlo{
    struct GreekEstimate{
        double value;
        double std_error;
    };

    struct PathwiseGreeks{
        GreekEstimate price, delta, vega, rho;
    };

    struct LrGreeks{
        GreekEstimate price, delta, vega;
    };

    struct BasketGreeks{
        double price;
        Eigen::VectorXd d_spots;
        Eigen::VectorXd d_vols;
        double d_rate;
        double d_strike;
    };

    namespace greeks_detail{
        inline GreekEstimate mean_se(const Eigen::ArrayXd& x){
            const double n = static_cast<double>(x.size());
            const double m = x.mean();
            return {m, std::sqrt((x - m).square().sum() / (n - 1.0) / n)};
        }
        inline Eigen::ArrayXd terminal(double spot, double rate, double vol, double time, const Eigen::VectorXd& z){
            return spot * ((rate - 0.5 * vol * vol) * time + vol * std::sqrt(time) * z.array()).exp();
        }
        inline void validate(double spot, double strike, double vol, double time, Eigen::Index n){
            if(spot <= 0.0 || strike <= 0.0 || vol <= 0.0 || time <= 0.0) throw InvalidInputError("spot, strike, vol and time must be positive");
            if(n < 2) throw InvalidInputError("need at least 2 draws");
        }
        inline double phi(double x){ return distributions::normal::pdf(x, 0.0, 1.0); }
        inline double ncdf(double x){ return distributions::normal::cdf(x, 0.0, 1.0); }
        inline double d2(double s, double k, double r, double v, double t){
            return (std::log(s / k) + (r - 0.5 * v * v) * t) / (v * std::sqrt(t));
        }
        //x = z L^T row by row, plus input checks shared by the basket functions
        inline Eigen::MatrixXd correlate(const Eigen::VectorXd& spots, const Eigen::VectorXd& vols, const Eigen::MatrixXd& corr,
                                         const Eigen::VectorXd& weights, double strike, double time, const Eigen::MatrixXd& z){
            const Eigen::Index n = spots.size();
            if(vols.size() != n || weights.size() != n || z.cols() != n || corr.rows() != n || corr.cols() != n){
                throw InvalidInputError("spots, vols, weights, corr and the columns of z must all have size N");
            }
            if((spots.array() <= 0.0).any() || (vols.array() <= 0.0).any() || strike <= 0.0 || time <= 0.0){
                throw InvalidInputError("spots, vols, strike and time must be positive");
            }
            if(z.rows() < 1) throw InvalidInputError("need at least one path");
            Eigen::LLT<Eigen::MatrixXd> llt(corr);
            if(llt.info() != Eigen::Success) throw InvalidInputError("correlation matrix must be positive definite");
            return z * llt.matrixL().transpose();
        }
    }

    inline PathwiseGreeks european_call_pathwise(double spot, double strike, double rate, double vol, double time, const Eigen::VectorXd& z){
        greeks_detail::validate(spot, strike, vol, time, z.size());
        const Eigen::ArrayXd st = greeks_detail::terminal(spot, rate, vol, time, z);
        const double df = std::exp(-rate * time);
        const Eigen::ArrayXd itm = (st > strike).cast<double>();
        const Eigen::ArrayXd payoff = df * (st - strike).max(0.0);
        return {greeks_detail::mean_se(payoff), greeks_detail::mean_se(df * itm * st / spot),
                greeks_detail::mean_se(df * itm * st * (std::sqrt(time) * z.array() - vol * time)),
                greeks_detail::mean_se(-time * payoff + df * itm * st * time)};
    }

    inline LrGreeks european_call_lr(double spot, double strike, double rate, double vol, double time, const Eigen::VectorXd& z){
        greeks_detail::validate(spot, strike, vol, time, z.size());
        const Eigen::ArrayXd st = greeks_detail::terminal(spot, rate, vol, time, z);
        const Eigen::ArrayXd payoff = std::exp(-rate * time) * (st - strike).max(0.0);
        const Eigen::ArrayXd za = z.array();
        return {greeks_detail::mean_se(payoff), greeks_detail::mean_se(payoff * za / (spot * vol * std::sqrt(time))),
                greeks_detail::mean_se(payoff * ((za.square() - 1.0) / vol - za * std::sqrt(time)))};
    }

    inline double digital_call_price(double spot, double strike, double rate, double vol, double time){
        return std::exp(-rate * time) * greeks_detail::ncdf(greeks_detail::d2(spot, strike, rate, vol, time));
    }

    inline double digital_call_delta(double spot, double strike, double rate, double vol, double time){
        return std::exp(-rate * time) * greeks_detail::phi(greeks_detail::d2(spot, strike, rate, vol, time)) / (spot * vol * std::sqrt(time));
    }

    inline GreekEstimate digital_call_delta_lr(double spot, double strike, double rate, double vol, double time, const Eigen::VectorXd& z){
        greeks_detail::validate(spot, strike, vol, time, z.size());
        const Eigen::ArrayXd pay = std::exp(-rate * time) * (greeks_detail::terminal(spot, rate, vol, time, z) > strike).cast<double>();
        return greeks_detail::mean_se(pay * z.array() / (spot * vol * std::sqrt(time)));
    }

    //central difference with common random numbers; h is the absolute spot bump
    inline GreekEstimate digital_call_delta_bump(double spot, double strike, double rate, double vol, double time,
                                                 const Eigen::VectorXd& z, double h){
        greeks_detail::validate(spot, strike, vol, time, z.size());
        if(h <= 0.0 || h >= spot) throw InvalidInputError("bump must be in (0, spot)");
        const double df = std::exp(-rate * time);
        const Eigen::ArrayXd up = (greeks_detail::terminal(spot + h, rate, vol, time, z) > strike).cast<double>();
        const Eigen::ArrayXd dn = (greeks_detail::terminal(spot - h, rate, vol, time, z) > strike).cast<double>();
        return greeks_detail::mean_se(df * (up - dn) / (2.0 * h));
    }

    //Basket call price only (for bumping)
    inline double basket_call_price(const Eigen::VectorXd& spots, const Eigen::VectorXd& vols, const Eigen::MatrixXd& corr,
                                    const Eigen::VectorXd& weights, double strike, double rate, double time, const Eigen::MatrixXd& z){
        const Eigen::MatrixXd x = greeks_detail::correlate(spots, vols, corr, weights, strike, time, z);
        const Eigen::ArrayXd drift = (rate - 0.5 * vols.array().square()) * time;
        const Eigen::ArrayXd sv = vols.array() * std::sqrt(time);
        double sum = 0.0;
        for(Eigen::Index p = 0; p < x.rows(); ++p){
            const double b = (weights.array() * spots.array() * (drift + sv * x.row(p).transpose().array()).exp()).sum();
            sum += std::max(b - strike, 0.0);
        }
        return std::exp(-rate * time) * sum / static_cast<double>(x.rows());
    }

    //Hand-derived pathwise gradient
    inline BasketGreeks basket_call_pathwise(const Eigen::VectorXd& spots, const Eigen::VectorXd& vols, const Eigen::MatrixXd& corr,
                                             const Eigen::VectorXd& weights, double strike, double rate, double time, const Eigen::MatrixXd& z){
        const Eigen::MatrixXd x = greeks_detail::correlate(spots, vols, corr, weights, strike, time, z);
        const Eigen::Index n_assets = spots.size();
        const double df = std::exp(-rate * time), sq = std::sqrt(time);
        const Eigen::ArrayXd drift = (rate - 0.5 * vols.array().square()) * time;
        BasketGreeks g{0.0, Eigen::VectorXd::Zero(n_assets), Eigen::VectorXd::Zero(n_assets), 0.0, 0.0};
        for(Eigen::Index p = 0; p < x.rows(); ++p){
            const Eigen::ArrayXd xp = x.row(p).transpose().array();
            const Eigen::ArrayXd ws = weights.array() * spots.array() * (drift + vols.array() * sq * xp).exp();
            const double b = ws.sum();
            const double pay = df * std::max(b - strike, 0.0);
            g.price += pay;
            g.d_rate += -time * pay;
            if(b > strike){
                g.d_spots.array() += df * ws / spots.array();
                g.d_vols.array() += df * ws * (sq * xp - vols.array() * time);
                g.d_rate += df * time * b;
                g.d_strike -= df;
            }
        }
        const double n = static_cast<double>(x.rows());
        g.price /= n; g.d_spots /= n; g.d_vols /= n; g.d_rate /= n; g.d_strike /= n;
        return g;
    }

    //Same gradient by AAD: inputs and shared terms recorded once, then record / sweep / rewind per path
    inline BasketGreeks basket_call_aad(const Eigen::VectorXd& spots, const Eigen::VectorXd& vols, const Eigen::MatrixXd& corr,
                                        const Eigen::VectorXd& weights, double strike, double rate, double time, const Eigen::MatrixXd& z){
        using autodiff::Var;
        const Eigen::MatrixXd x = greeks_detail::correlate(spots, vols, corr, weights, strike, time, z);
        const auto n_assets = static_cast<std::size_t>(spots.size());
        autodiff::Tape tape;
        std::vector<Var> s(n_assets), v(n_assets), drift(n_assets), sv(n_assets);
        for(std::size_t i = 0; i < n_assets; ++i) s[i] = tape.variable(spots(static_cast<Eigen::Index>(i)));
        for(std::size_t i = 0; i < n_assets; ++i) v[i] = tape.variable(vols(static_cast<Eigen::Index>(i)));
        const Var r = tape.variable(rate), k = tape.variable(strike);
        const double sq = std::sqrt(time);
        for(std::size_t i = 0; i < n_assets; ++i){
            drift[i] = (r - 0.5 * v[i] * v[i]) * time;
            sv[i] = v[i] * sq;
        }
        const Var df = autodiff::exp(-r * time);
        const std::size_t mark = tape.size();
        const double inv_n = 1.0 / static_cast<double>(x.rows());
        double price = 0.0;
        for(Eigen::Index p = 0; p < x.rows(); ++p){
            Var b = tape.variable(0.0);
            for(std::size_t i = 0; i < n_assets; ++i){
                const double w = weights(static_cast<Eigen::Index>(i));
                b = b + w * s[i] * autodiff::exp(drift[i] + sv[i] * x(p, static_cast<Eigen::Index>(i)));
            }
            const Var pay = df * autodiff::max0(b - k);
            price += pay.value();
            tape.accumulate(pay, mark, inv_n);
            tape.rewind(mark);
        }
        tape.propagate(mark - 1, 0);
        BasketGreeks g{price * inv_n, Eigen::VectorXd(spots.size()), Eigen::VectorXd(spots.size()), tape.adjoint(r), tape.adjoint(k)};
        for(std::size_t i = 0; i < n_assets; ++i){
            g.d_spots(static_cast<Eigen::Index>(i)) = tape.adjoint(s[i]);
            g.d_vols(static_cast<Eigen::Index>(i)) = tape.adjoint(v[i]);
        }
        return g;
    }

    //Central differences on the price with common random numbers: 2 (2N + 2) extra pricings
    inline BasketGreeks basket_call_bump(const Eigen::VectorXd& spots, const Eigen::VectorXd& vols, const Eigen::MatrixXd& corr,
                                         const Eigen::VectorXd& weights, double strike, double rate, double time, const Eigen::MatrixXd& z,
                                         double rel_bump = 1e-4){
        auto price = [&](const Eigen::VectorXd& s, const Eigen::VectorXd& v, double k, double r){
            return basket_call_price(s, v, corr, weights, k, r, time, z);
        };
        const Eigen::Index n = spots.size();
        BasketGreeks g{price(spots, vols, strike, rate), Eigen::VectorXd(n), Eigen::VectorXd(n), 0.0, 0.0};
        for(Eigen::Index i = 0; i < n; ++i){
            Eigen::VectorXd up = spots, dn = spots;
            const double h = rel_bump * spots(i);
            up(i) += h; dn(i) -= h;
            g.d_spots(i) = (price(up, vols, strike, rate) - price(dn, vols, strike, rate)) / (2.0 * h);
            Eigen::VectorXd vu = vols, vd = vols;
            const double hv = rel_bump * vols(i);
            vu(i) += hv; vd(i) -= hv;
            g.d_vols(i) = (price(spots, vu, strike, rate) - price(spots, vd, strike, rate)) / (2.0 * hv);
        }
        const double hr = 1e-6, hk = rel_bump * strike;
        g.d_rate = (price(spots, vols, strike, rate + hr) - price(spots, vols, strike, rate - hr)) / (2.0 * hr);
        g.d_strike = (price(spots, vols, strike + hk, rate) - price(spots, vols, strike - hk, rate)) / (2.0 * hk);
        return g;
    }
}
