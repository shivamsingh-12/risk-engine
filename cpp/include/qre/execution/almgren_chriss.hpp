#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include "qre/core/exceptions.hpp"
//Phase 18: optimal execution, Almgren and Chriss (2000)
//Sell X shares over [0, T] in N periods, tau = T/N; holdings x_0 = X .. x_N = 0, trades n_k = x_k-1 - x_k, rate v = n/tau
//  mid    S_k = S_k-1 + sigma sqrt(tau) xi_k - tau gamma v_k                (permanent impact, linear)
//  fill   S~_k = S_k-1 - (eps sgn(v_k) + eta v_k)                            (temporary impact)
//  shortfall IS = X S_0 - sum n_k S~_k
//  E[IS] = gamma X^2/2 + eps sum|n_k| + (eta~/tau) sum n_k^2,  eta~ = eta - gamma tau/2;  Var[IS] = sigma^2 tau sum_{k>=1} x_k^2
//Minimizing E + lambda Var: x_j = X sinh(kappa (T - t_j)) / sinh(kappa T),  (2/tau^2)(cosh(kappa tau) - 1) = lambda sigma^2 / eta~
//lambda = 0 is TWAP (a straight line); the same optimum solves a tridiagonal system (solve_numerically, Thomas algorithm)
namespace qre::execution{
    struct AcParams{
        double X, T;
        int N;
        double sigma, eta, gamma, eps = 0.0, S0 = 50.0;

        double tau() const { return T / N; }
        double eta_tilde() const { return eta - 0.5 * gamma * tau(); }
        void validate() const{
            if(!(X > 0.0 && T > 0.0 && N >= 1 && sigma >= 0.0 && eta > 0.0 && gamma >= 0.0 && eps >= 0.0 && S0 > 0.0)){
                throw InvalidInputError("need X, T, eta, S0 > 0, N >= 1 and sigma, gamma, eps >= 0");
            }
            if(!(eta_tilde() > 0.0)) throw InvalidInputError("need eta > gamma tau / 2 (otherwise splitting trades has no cost)");
        }
    };

    //Almgren and Chriss (2000) section 3 example: 1e6 shares at $50 over 5 days, 30% annual vol, eps = 1/16, eta = 2.5e-6, gamma = 2.5e-7
    inline AcParams paper_example(){ return AcParams{1e6, 5.0, 5, 0.95, 2.5e-6, 2.5e-7, 0.0625, 50.0}; }

    inline double kappa(const AcParams& p, double lambda){
        p.validate();
        if(lambda < 0.0) throw InvalidInputError("risk aversion must be >= 0");
        const double k2 = lambda * p.sigma * p.sigma / p.eta_tilde();
        return std::acosh(1.0 + 0.5 * k2 * p.tau() * p.tau()) / p.tau();
    }

    inline std::vector<double> optimal_holdings(const AcParams& p, double lambda){
        const double k = kappa(p, lambda);
        std::vector<double> x(static_cast<std::size_t>(p.N) + 1);
        for(int j = 0; j <= p.N; ++j){
            const double t = p.tau() * j;
            double v;
            if(k * p.T < 1e-10) v = 1.0 - t / p.T;
            else if(k * p.T > 700.0) v = std::exp(-k * t) * (1.0 - std::exp(-2.0 * k * (p.T - t))) / (1.0 - std::exp(-2.0 * k * p.T));
            else v = std::sinh(k * (p.T - t)) / std::sinh(k * p.T);
            x[static_cast<std::size_t>(j)] = p.X * v;
        }
        return x;
    }

    inline std::vector<double> twap_holdings(const AcParams& p){
        std::vector<double> x(static_cast<std::size_t>(p.N) + 1);
        for(int j = 0; j <= p.N; ++j) x[static_cast<std::size_t>(j)] = p.X * (1.0 - static_cast<double>(j) / p.N);
        return x;
    }

    inline double expected_cost(const AcParams& p, const std::vector<double>& x){
        double abs_sum = 0.0, sq = 0.0;
        for(std::size_t k = 1; k < x.size(); ++k){ const double n = x[k - 1] - x[k]; abs_sum += std::abs(n); sq += n * n; }
        return 0.5 * p.gamma * p.X * p.X + p.eps * abs_sum + p.eta_tilde() / p.tau() * sq;
    }

    inline double cost_variance(const AcParams& p, const std::vector<double>& x){
        double s = 0.0;
        for(std::size_t k = 1; k < x.size(); ++k) s += x[k] * x[k];
        return p.sigma * p.sigma * p.tau() * s;
    }

    //First-order conditions a (2 x_j - x_j-1 - x_j+1) + 2 lambda sigma^2 tau x_j = 0, a = 2 eta~/tau, solved by Thomas' algorithm
    inline std::vector<double> solve_numerically(const AcParams& p, double lambda){
        p.validate();
        if(lambda < 0.0) throw InvalidInputError("risk aversion must be >= 0");
        std::vector<double> x(static_cast<std::size_t>(p.N) + 1, 0.0);
        x[0] = p.X;
        const int m = p.N - 1;
        if(m <= 0) return x;
        const double a = 2.0 * p.eta_tilde() / p.tau(), diag = 2.0 * a + 2.0 * lambda * p.sigma * p.sigma * p.tau();
        std::vector<double> c(static_cast<std::size_t>(m)), d(static_cast<std::size_t>(m));
        //forward sweep for A x = b, A tridiagonal (-a, diag, -a), b = (a X, 0, ..., 0)
        c[0] = -a / diag;
        d[0] = a * p.X / diag;
        for(int i = 1; i < m; ++i){
            const double den = diag + a * c[static_cast<std::size_t>(i - 1)];
            c[static_cast<std::size_t>(i)] = -a / den;
            d[static_cast<std::size_t>(i)] = a * d[static_cast<std::size_t>(i - 1)] / den;
        }
        x[static_cast<std::size_t>(m)] = d[static_cast<std::size_t>(m - 1)];
        for(int i = m - 2; i >= 0; --i){
            x[static_cast<std::size_t>(i + 1)] = d[static_cast<std::size_t>(i)] - c[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i + 2)];
        }
        return x;
    }

    struct FrontierPoint{ double lambda, expected, sd; };

    inline std::vector<FrontierPoint> efficient_frontier(const AcParams& p, const std::vector<double>& lambdas){
        std::vector<FrontierPoint> out;
        for(double l : lambdas){
            const auto x = optimal_holdings(p, l);
            out.push_back({l, expected_cost(p, x), std::sqrt(cost_variance(p, x))});
        }
        return out;
    }

    //Monte Carlo implementation shortfall of a schedule under the model (arithmetic Brownian mid price)
    inline std::vector<double> simulate_shortfall(const AcParams& p, const std::vector<double>& x, std::size_t n_paths, uint64_t seed){
        p.validate();
        if(x.size() != static_cast<std::size_t>(p.N) + 1) throw InvalidInputError("schedule must have N + 1 holdings");
        std::mt19937_64 eng(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::vector<double> out(n_paths);
        const double sq = std::sqrt(p.tau());
        for(std::size_t i = 0; i < n_paths; ++i){
            double S = p.S0, proceeds = 0.0;
            for(int k = 0; k < p.N; ++k){
                const double n = x[static_cast<std::size_t>(k)] - x[static_cast<std::size_t>(k + 1)], v = n / p.tau();
                const double sgn = static_cast<double>((v > 0.0) - (v < 0.0));
                proceeds += n * (S - (p.eps * sgn + p.eta * v));
                S += p.sigma * sq * nd(eng) - p.tau() * p.gamma * v;
            }
            out[i] = p.X * p.S0 - proceeds;
        }
        return out;
    }
}
