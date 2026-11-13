#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numbers>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
//Phase 17: deep hedging (Buehler, Gonon, Teichmann, Wood 2019)
//Short one call (strike K, maturity T, zero rates), trade the underlying at n dates, position delta_k over (t_k, t_k+1],
//proportional cost c per unit traded, unwind at T:
//  L = max(S_T - K, 0) - sum_k delta_k (S_k+1 - S_k) + c sum_{k=0}^{n} S_k |delta_k - delta_k-1|,  delta_-1 = delta_n = 0
//Policy: one network for all dates, delta_k = f(k/n, ln(S_k/K)/sqrt(T), delta_k-1 [, sqrt(v_k)]),
//two tanh hidden layers and a linear output. Objective CVaR_alpha(L) in Rockafellar-Uryasev form
//  J = w + E[(L - w)^+] / (1 - alpha), w = alpha-quantile of the batch's losses (the exact minimizer)
//Gradient by backpropagation through time: delta_k enters the loss directly (gains and the trading costs at k and
//k+1) and through the network input at k+1, so the backward pass runs from the last date to the first
//Mirrors python_reference/qre/hedging/deep_hedging.py; forward values and gradients agree to rounding
namespace qre::hedging{
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using RowVector = Eigen::RowVectorXd;

    struct Policy{
        Matrix W1, W2, W3;          //n_in x H, H x H, H x 1
        RowVector b1, b2;           //1 x H
        double b3 = 0.5;
        double w = 0.0;             //CVaR threshold

        Eigen::Index n_in() const { return W1.rows(); }

        static Policy init(Eigen::Index n_in, Eigen::Index hidden, uint64_t seed){
            std::mt19937_64 eng(seed);
            std::uniform_real_distribution<double> u(-1.0, 1.0);
            auto g = [&](Eigen::Index a, Eigen::Index b){
                Matrix m(a, b);
                const double s = std::sqrt(6.0 / static_cast<double>(a + b));
                for(Eigen::Index i = 0; i < a; ++i) for(Eigen::Index j = 0; j < b; ++j) m(i, j) = s * u(eng);
                return m;
            };
            Policy p;
            p.W1 = g(n_in, hidden);
            p.W2 = g(hidden, hidden);
            p.W3 = g(hidden, 1);
            p.b1 = RowVector::Zero(hidden);
            p.b2 = RowVector::Zero(hidden);
            return p;
        }
    };

    //---------------------------------------------------------------- markets
    inline Matrix simulate_gbm(std::size_t n_paths, int n_steps, double s0, double mu, double sigma, double T, std::mt19937_64& eng){
        if(n_paths < 1 || n_steps < 1 || !(s0 > 0.0 && sigma > 0.0 && T > 0.0)) throw InvalidInputError("bad GBM inputs");
        std::normal_distribution<double> nd(0.0, 1.0);
        const double dt = T / n_steps, drift = (mu - 0.5 * sigma * sigma) * dt, vol = sigma * std::sqrt(dt);
        Matrix S(static_cast<Eigen::Index>(n_paths), n_steps + 1);
        for(Eigen::Index p = 0; p < S.rows(); ++p){
            double x = std::log(s0);
            S(p, 0) = s0;
            for(int k = 1; k <= n_steps; ++k){ x += drift + vol * nd(eng); S(p, k) = std::exp(x); }
        }
        return S;
    }

    //Full-truncation Euler with `substeps` per hedging interval; S and v at the hedging dates
    inline void simulate_heston(std::size_t n_paths, int n_steps, double s0, double mu, double v0, double kappa, double theta,
                                double xi, double rho, double T, std::mt19937_64& eng, Matrix& S, Matrix& V, int substeps = 4){
        if(n_paths < 1 || n_steps < 1 || substeps < 1 || !(s0 > 0.0 && v0 >= 0.0 && T > 0.0) || std::abs(rho) > 1.0){
            throw InvalidInputError("bad Heston inputs");
        }
        std::normal_distribution<double> nd(0.0, 1.0);
        const int m = n_steps * substeps;
        const double dt = T / m, rc = std::sqrt(1.0 - rho * rho);
        S.resize(static_cast<Eigen::Index>(n_paths), n_steps + 1);
        V.resize(S.rows(), n_steps + 1);
        for(Eigen::Index p = 0; p < S.rows(); ++p){
            double s = std::log(s0), v = v0;
            S(p, 0) = s0;
            V(p, 0) = v0;
            for(int i = 0; i < m; ++i){
                const double z1 = nd(eng), z2 = rho * z1 + rc * nd(eng), vp = std::max(v, 0.0);
                s += (mu - 0.5 * vp) * dt + std::sqrt(vp * dt) * z1;
                v += kappa * (theta - vp) * dt + xi * std::sqrt(vp * dt) * z2;
                if((i + 1) % substeps == 0){
                    S(p, (i + 1) / substeps) = std::exp(s);
                    V(p, (i + 1) / substeps) = std::max(v, 0.0);
                }
            }
        }
    }

    inline double bs_call(double s, double k, double sigma, double tau){
        if(tau <= 0.0) return std::max(s - k, 0.0);
        const double d1 = (std::log(s / k) + 0.5 * sigma * sigma * tau) / (sigma * std::sqrt(tau));
        const double d2 = d1 - sigma * std::sqrt(tau);
        return s * 0.5 * std::erfc(-d1 / std::numbers::sqrt2) - k * 0.5 * std::erfc(-d2 / std::numbers::sqrt2);
    }

    //---------------------------------------------------------------- losses
    inline Vector hedge_loss(const Matrix& S, const Matrix& deltas, double K, double cost){
        const Eigen::Index n = deltas.cols();
        if(S.cols() != n + 1 || S.rows() != deltas.rows()) throw InvalidInputError("S must have one more column than deltas");
        Vector L(S.rows());
        for(Eigen::Index p = 0; p < S.rows(); ++p){
            double gains = 0.0, costs = 0.0, prev = 0.0;
            for(Eigen::Index k = 0; k < n; ++k){
                gains += deltas(p, k) * (S(p, k + 1) - S(p, k));
                costs += S(p, k) * std::abs(deltas(p, k) - prev);
                prev = deltas(p, k);
            }
            costs += S(p, n) * std::abs(prev);
            L(p) = std::max(S(p, n) - K, 0.0) - gains + cost * costs;
        }
        return L;
    }

    //Black-Scholes deltas at the hedging dates: constant sigma, or sqrt(v_k) per path when V is given
    inline Matrix bs_hedge_deltas(const Matrix& S, double K, double T, double sigma, const Matrix* V = nullptr){
        const Eigen::Index n = S.cols() - 1;
        Matrix out(S.rows(), n);
        for(Eigen::Index k = 0; k < n; ++k){
            const double tau = T * static_cast<double>(n - k) / static_cast<double>(n);
            for(Eigen::Index p = 0; p < S.rows(); ++p){
                const double sig = V ? std::sqrt(std::max((*V)(p, k), 1e-8)) : sigma;
                const double d1 = (std::log(S(p, k) / K) + 0.5 * sig * sig * tau) / (sig * std::sqrt(tau));
                out(p, k) = 0.5 * std::erfc(-d1 / std::numbers::sqrt2);
            }
        }
        return out;
    }

    //lower ("inverted cdf") alpha-quantile: the smallest loss with at least alpha of the sample at or below it
    inline double quantile(const Vector& L, double alpha){
        std::vector<double> v(L.data(), L.data() + L.size());
        const auto idx = static_cast<std::size_t>(std::ceil(alpha * static_cast<double>(v.size()))) - 1;
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(idx), v.end());
        return v[idx];
    }

    inline double cvar(const Vector& L, double alpha){
        if(!(alpha > 0.0 && alpha < 1.0) || L.size() < 1) throw InvalidInputError("need 0 < alpha < 1 and losses");
        const double w = quantile(L, alpha);
        return w + (L.array() - w).max(0.0).mean() / (1.0 - alpha);
    }

    //---------------------------------------------------------------- network
    namespace dh_detail{
        inline Matrix features(const Matrix& S, const Matrix* V, double K, double T, Eigen::Index k, const Vector& prev, Eigen::Index n_in){
            const Eigen::Index n = S.cols() - 1;
            Matrix x(S.rows(), n_in);
            x.col(0).setConstant(static_cast<double>(k) / static_cast<double>(n));
            x.col(1) = (S.col(k).array() / K).log() / std::sqrt(T);
            x.col(2) = prev;
            if(n_in == 4) x.col(3) = V->col(k).array().sqrt();
            return x;
        }
        inline void check(const Policy& p, const Matrix& S, const Matrix* V){
            if(S.cols() < 2) throw InvalidInputError("need at least one hedging date");
            if(p.n_in() != (V ? 4 : 3)) throw InvalidInputError("network inputs must be 3 (no variance) or 4 (with variance)");
            if(V && (V->rows() != S.rows() || V->cols() != S.cols())) throw InvalidInputError("V must have the shape of S");
        }
    }

    inline Matrix policy_deltas(const Policy& p, const Matrix& S, double K, double T, const Matrix* V = nullptr){
        dh_detail::check(p, S, V);
        const Eigen::Index n = S.cols() - 1;
        Matrix out(S.rows(), n);
        Vector prev = Vector::Zero(S.rows());
        for(Eigen::Index k = 0; k < n; ++k){
            const Matrix x = dh_detail::features(S, V, K, T, k, prev, p.n_in());
            const Matrix h1 = ((x * p.W1).rowwise() + p.b1).array().tanh();
            const Matrix h2 = ((h1 * p.W2).rowwise() + p.b2).array().tanh();
            prev = (h2 * p.W3).col(0).array() + p.b3;
            out.col(k) = prev;
        }
        return out;
    }

    struct ObjectiveGrad{
        double J;
        Policy grad;            //same shapes as the policy; grad.b3 and grad.w hold dJ/db3 and dJ/dw
        Vector L;
    };

    inline ObjectiveGrad objective_and_grad(const Policy& p, const Matrix& S, double K, double T, double cost, double alpha,
                                            const Matrix* V = nullptr){
        dh_detail::check(p, S, V);
        if(!(alpha > 0.0 && alpha < 1.0) || cost < 0.0) throw InvalidInputError("need 0 < alpha < 1 and cost >= 0");
        const Eigen::Index P = S.rows(), n = S.cols() - 1;
        std::vector<Matrix> xs, h1s, h2s;
        Matrix deltas(P, n);
        Vector prev = Vector::Zero(P);
        for(Eigen::Index k = 0; k < n; ++k){
            xs.push_back(dh_detail::features(S, V, K, T, k, prev, p.n_in()));
            h1s.push_back(((xs.back() * p.W1).rowwise() + p.b1).array().tanh());
            h2s.push_back(((h1s.back() * p.W2).rowwise() + p.b2).array().tanh());
            prev = (h2s.back() * p.W3).col(0).array() + p.b3;
            deltas.col(k) = prev;
        }
        ObjectiveGrad r{0.0, Policy{}, hedge_loss(S, deltas, K, cost)};
        const Vector excess = r.L.array() - p.w;
        const double share = (excess.array() > 0.0).cast<double>().mean();
        r.J = p.w + excess.array().max(0.0).mean() / (1.0 - alpha);
        const Vector gL = (excess.array() > 0.0).cast<double>() / ((1.0 - alpha) * static_cast<double>(P));
        Policy& g = r.grad;
        g.W1 = Matrix::Zero(p.W1.rows(), p.W1.cols());
        g.W2 = Matrix::Zero(p.W2.rows(), p.W2.cols());
        g.W3 = Matrix::Zero(p.W3.rows(), 1);
        g.b1 = RowVector::Zero(p.b1.size());
        g.b2 = RowVector::Zero(p.b2.size());
        g.b3 = 0.0;
        g.w = 1.0 - share / (1.0 - alpha);
        auto sgn = [](double x){ return static_cast<double>((x > 0.0) - (x < 0.0)); };
        Vector carry = Vector::Zero(P);
        for(Eigen::Index k = n - 1; k >= 0; --k){
            Vector g_out(P);
            for(Eigen::Index i = 0; i < P; ++i){
                const double before = k > 0 ? deltas(i, k - 1) : 0.0, after = k + 1 < n ? deltas(i, k + 1) : 0.0;
                const double direct = -(S(i, k + 1) - S(i, k))
                                      + cost * (S(i, k) * sgn(deltas(i, k) - before) - S(i, k + 1) * sgn(after - deltas(i, k)));
                g_out(i) = gL(i) * direct + carry(i);
            }
            const Matrix& h1 = h1s[static_cast<std::size_t>(k)];
            const Matrix& h2 = h2s[static_cast<std::size_t>(k)];
            const Matrix& x = xs[static_cast<std::size_t>(k)];
            g.W3 += h2.transpose() * g_out;
            g.b3 += g_out.sum();
            const Matrix g2 = ((g_out * p.W3.transpose()).array() * (1.0 - h2.array().square())).matrix();
            g.W2 += h1.transpose() * g2;
            g.b2 += g2.colwise().sum();
            const Matrix g1 = ((g2 * p.W2.transpose()).array() * (1.0 - h1.array().square())).matrix();
            g.W1 += x.transpose() * g1;
            g.b1 += g1.colwise().sum();
            carry = (g1 * p.W1.transpose()).col(2);
        }
        return r;
    }

    inline double var_threshold(const Policy& p, const Matrix& S, double K, double T, double cost, double alpha, const Matrix* V = nullptr){
        return quantile(hedge_loss(S, policy_deltas(p, S, K, T, V), K, cost), alpha);
    }

    //sampler(n, eng, S, V): fill S (and V when the model has stochastic variance; leave it empty otherwise)
    using Sampler = std::function<void(std::size_t, std::mt19937_64&, Matrix&, Matrix&)>;

    struct TrainConfig{
        int iterations = 2000;
        std::size_t batch = 4096;
        double lr = 1e-3;
        double lr_final_fraction = 1.0;     //linear decay of the learning rate to lr * this at the last iteration
        uint64_t seed = 0;
        int log_every = 100;
    };

    struct TrainLog{ std::vector<std::pair<int, double>> objective; };

    //Adam on fresh simulated batches; w is set to each batch's alpha-quantile before the gradient step
    inline Policy train(Policy p, const Sampler& sampler, double K, double T, double cost, double alpha, const TrainConfig& cfg,
                        TrainLog* log = nullptr){
        std::mt19937_64 eng(cfg.seed);
        const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
        Policy m = p, s = p;
        for(Matrix* a : {&m.W1, &m.W2, &m.W3, &s.W1, &s.W2, &s.W3}) a->setZero();
        for(RowVector* a : {&m.b1, &m.b2, &s.b1, &s.b2}) a->setZero();
        m.b3 = s.b3 = 0.0;
        Matrix S, V;
        for(int it = 1; it <= cfg.iterations; ++it){
            V.resize(0, 0);
            sampler(cfg.batch, eng, S, V);
            const Matrix* vp = V.size() ? &V : nullptr;
            p.w = var_threshold(p, S, K, T, cost, alpha, vp);
            const ObjectiveGrad og = objective_and_grad(p, S, K, T, cost, alpha, vp);
            const double frac = cfg.iterations > 1 ? static_cast<double>(it - 1) / static_cast<double>(cfg.iterations - 1) : 0.0;
            const double lr = cfg.lr * (1.0 - frac * (1.0 - cfg.lr_final_fraction));
            const double c1 = 1.0 - std::pow(b1, it), c2 = 1.0 - std::pow(b2, it);
            auto step = [&](auto& param, auto& mm, auto& ss, const auto& gg){
                mm = b1 * mm + (1.0 - b1) * gg;
                ss = (b2 * ss.array() + (1.0 - b2) * gg.array().square()).matrix();
                param -= (lr * (mm.array() / c1) / ((ss.array() / c2).sqrt() + eps)).matrix();
            };
            step(p.W1, m.W1, s.W1, og.grad.W1);
            step(p.W2, m.W2, s.W2, og.grad.W2);
            step(p.W3, m.W3, s.W3, og.grad.W3);
            step(p.b1, m.b1, s.b1, og.grad.b1);
            step(p.b2, m.b2, s.b2, og.grad.b2);
            m.b3 = b1 * m.b3 + (1.0 - b1) * og.grad.b3;
            s.b3 = b2 * s.b3 + (1.0 - b2) * og.grad.b3 * og.grad.b3;
            p.b3 -= lr * (m.b3 / c1) / (std::sqrt(s.b3 / c2) + eps);
            if(log && (it % cfg.log_every == 0 || it == 1)) log->objective.emplace_back(it, og.J);
        }
        return p;
    }
}
